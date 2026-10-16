"""The strip's mode switch: character replace <-> background replace, one per job.

Flipping clears nothing: each mode keeps its own subject and prompts (flipping back brings them back), the
takes stay listed and only the job's mode's count, so an assembly never mixes the two, and a pin never reaches
a take of the other mode. A plan from before the switch (invert) reads as background."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
from test_swap import job  # noqa: E402,F401

CHAR = "<Subject 1> is the clown whose motion comes from <Video 1> and whose appearance comes from <Picture 1>: pink hair."
PLACE = "<Subject 1> is the diner environment in <Picture 1>, featuring striped walls."


def _take(tid, invert):
    opts = {"mark": True, "invert": True} if invert else {"mark": True}
    return {"id": tid, "state": "ok", "render": [0, 10], "settings": {"options": opts}}


def test_an_old_plan_reads_its_invert_as_the_mode():
    assert sp.job_mode({}) == "character" and sp.job_mode({"invert": True}) == "background"
    assert sp.job_mode({"mode": "character", "invert": True}) == "character"
    assert sp.take_mode(_take("c1-t001", True)) == "background" and sp.take_mode(_take("c1-t002", False)) == "character"
    assert sp.take_mode({"settings": {"options": {"invert": "true"}}}) == "background"
    assert sp.take_mode({"id": "x"}) == "character"                    # a take from before the switch: character


def test_only_the_jobs_mode_takes_count():
    c = {"takes": [_take("c1-t001", False), _take("c1-t002", True), _take("c1-t003", True)]}
    assert sp.effective_take(c, "character")[0]["id"] == "c1-t001"
    assert sp.effective_take(c, "background")[0]["id"] == "c1-t002"
    c["chosen"] = "c1-t003"
    assert sp.effective_take(c, "background") == (c["takes"][2], "chosen")
    assert sp.effective_take(c, "character")[0]["id"] == "c1-t001"     # a choice of the other mode doesn't carry
    assert sp.effective_take({"takes": [_take("c1-t001", True)]}, "character") == (None, "pending")
    assert sp.effective_take(c)[0]["id"] == "c1-t003"                  # no mode: as before


def test_flipping_keeps_each_modes_subject_and_prompts(job):
    jobs = str(job["dir"].parent.parent)
    name = job["dir"].name
    ids = [c["id"] for c in sp.load_plan(job["plan"])["chunks"]]

    def setup(p):
        p["subject"] = CHAR
        for i, c in enumerate(p["chunks"]):
            c.update(prompt=f"character prompt {i}", prompt_state="edited" if i == 0 else "draft")
            c.setdefault("options", {})["invert"] = True                 # an old chunk-level invert
    sp.update_plan(job["plan"], setup)
    r = spl.do_op({"job": name, "op": "set_mode", "mode": "background"}, jobs)
    p = sp.load_plan(job["plan"])
    assert sp.job_mode(p) == "background" and "invert" not in p and (r.get("result") or r).get("switched", True)
    assert p["subject"] == "" and all(c["prompt"] == "" and c["prompt_state"] == "empty" for c in p["chunks"])
    assert all("invert" not in (c.get("options") or {}) for c in p["chunks"])
    # the background side is written ...
    sp.update_plan(job["plan"], lambda q: [q.update(subject=PLACE)] + [c.update(prompt=f"background prompt {i}",
                                                                                prompt_state="draft")
                                                                       for i, c in enumerate(q["chunks"])])
    # ... flip back: the character side comes back exactly, the background side waits
    spl.do_op({"job": name, "op": "set_mode", "mode": "character"}, jobs)
    p = sp.load_plan(job["plan"])
    assert p["subject"] == CHAR and [c["prompt"] for c in p["chunks"]] == [f"character prompt {i}" for i in range(len(ids))]
    assert p["chunks"][0]["prompt_state"] == "edited" and p["chunks"][1]["prompt_state"] == "draft"
    assert p["by_mode"]["background"]["plan"]["subject"] == PLACE
    spl.do_op({"job": name, "op": "set_mode", "mode": "background"}, jobs)
    p = sp.load_plan(job["plan"])
    assert p["subject"] == PLACE and p["chunks"][1]["prompt"] == "background prompt 1"
    # the same mode again: nothing moves
    spl.do_op({"job": name, "op": "set_mode", "mode": "background"}, jobs)
    assert sp.load_plan(job["plan"])["chunks"][1]["prompt"] == "background prompt 1"
    with pytest.raises(Exception):
        spl.do_op({"job": name, "op": "set_mode", "mode": "both"}, jobs)


def test_the_strip_has_one_switch_and_no_invert_ticks():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "js", "swap_planner.js"),
               encoding="utf-8").read()
    assert '"Character replace"' in src and '"Background replace"' in src and 'op: "set_mode"' in src
    assert "invert (mark everything else)" not in src and "invSel" not in src and "invertIn" not in src


def test_each_mode_chooses_its_own_take_and_refuses_the_others(job):
    jobs = str(job["dir"].parent.parent)
    name = job["dir"].name
    cid = sp.load_plan(job["plan"])["chunks"][0]["id"]
    sp.update_plan(job["plan"], lambda p: p["chunks"][0].update(takes=[
        dict(_take(f"{cid}-t001", False), render=p["chunks"][0]["render"]),
        dict(_take(f"{cid}-t002", True), render=p["chunks"][0]["render"]),
        dict(_take(f"{cid}-t003", True), render=p["chunks"][0]["render"])]))
    # character: a background take can't be chosen (it would do nothing here and bite on the switch)
    with pytest.raises(Exception) as e:
        spl.do_op({"job": name, "op": "choose_take", "chunk": cid, "take": f"{cid}-t003"}, jobs)
    assert "background replace take" in str(e.value)
    spl.do_op({"job": name, "op": "choose_take", "chunk": cid, "take": f"{cid}-t001"}, jobs)
    spl.do_op({"job": name, "op": "set_mode", "mode": "background"}, jobs)
    assert sp.load_plan(job["plan"])["chunks"][0]["chosen"] is None             # background hasn't chosen yet
    spl.do_op({"job": name, "op": "choose_take", "chunk": cid, "take": f"{cid}-t003"}, jobs)
    spl.do_op({"job": name, "op": "set_mode", "mode": "character"}, jobs)
    c = sp.load_plan(job["plan"])["chunks"][0]
    assert c["chosen"] == f"{cid}-t001" and sp.effective_take(c, "character")[1] == "chosen"
    spl.do_op({"job": name, "op": "set_mode", "mode": "background"}, jobs)
    assert sp.load_plan(job["plan"])["chunks"][0]["chosen"] == f"{cid}-t003"     # neither choice lost


def test_the_strip_lists_only_the_jobs_mode_takes():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "js", "swap_planner.js"),
               encoding="utf-8").read()
    assert "(c.takes || []).filter(t => takeMode(t) === jobMode())" in src and "kept (switch to see" in src


def test_the_view_menu_shows_the_character_and_the_background_mask():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "js", "swap_planner.js"),
               encoding="utf-8").read()
    assert 'setMode("mask")' in src and 'setMode("mask_bg")' in src
    assert 'S.mode === "mask" || S.mode === "mask_bg"' in src                      # the same cached previews
    assert 'video.style.filter = S.mode === "mask_bg" ? "invert(1)" : ""' in src    # inverted in the player
    assert "setMode(maskView())" in src                                             # the row opens the job's kind


def test_a_character_replace_can_mark_the_backdrop_and_its_takes_stay_character(job):
    # a plain green screen: the guide marks the backdrop (inverted), the prompt still replaces the person
    jobs = str(job["dir"].parent.parent)
    name = job["dir"].name
    p = sp.load_plan(job["plan"])
    assert sp.job_guide(p) == "person" and not sp.chunk_invert(p, p["chunks"][0])
    spl.do_op({"job": name, "op": "set_guide", "guide": "backdrop"}, jobs)
    p = sp.load_plan(job["plan"])
    assert sp.job_mode(p) == "character" and sp.chunk_invert(p, p["chunks"][0])
    # the take records its mode, so an inverted character take still counts in character mode
    t = {"id": "x", "state": "ok", "settings": {"options": {"mark": True, "invert": True, "mode": "character"}}}
    assert sp.take_mode(t) == "character"
    assert sp.take_mode({"settings": {"options": {"invert": True}}}) == "background"   # before the mode was recorded
    # background always inverts; the guide waits for the next character replace
    spl.do_op({"job": name, "op": "set_mode", "mode": "background"}, jobs)
    spl.do_op({"job": name, "op": "set_guide", "guide": "person"}, jobs)
    p = sp.load_plan(job["plan"])
    assert sp.chunk_invert(p, p["chunks"][0]) and sp.job_guide(p) == "person"
    with pytest.raises(Exception):
        spl.do_op({"job": name, "op": "set_guide", "guide": "both"}, jobs)


def test_the_strip_offers_the_guide_in_character_mode_only():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "js", "swap_planner.js"),
               encoding="utf-8").read()
    assert 'op: "set_guide"' in src and 'guideBox.style.display = has && jm === "character"' in src
    assert '(jobMode() === "background" || jobGuide() === "backdrop") ? "mask_bg" : "mask"' in src
