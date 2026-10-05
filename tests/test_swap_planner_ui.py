"""SeamStitch Swap Planner, the strip's backend (B2): auto splits, cut detection and confirmation,
the take trash, "use this prompt and seed", failure states, chunk states and join verdicts for the
strip, and the mark run with its mask cache (SeamStitch Swap Mask) feeding later render runs."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
from test_swap import CUTS_978, FR, H_, W_, _add_take, _clip, _frames, _write, job  # noqa: E402,F401
from test_swap_nodes import _plan978, _run, nodes  # noqa: E402,F401


# ---------------------------------------------------------------- splits and cuts

def test_auto_splits_op_gives_the_signed_off_plan_and_guards_edits():
    p = _plan978()
    res = spl.apply_op(p, {"op": "auto_splits"})
    assert res["splits"] == [209, 406, 603, 785] and res["straight"] == [785]      # B5b: the 209 ceiling
    assert [c["length"] for c in p["chunks"]] == [209, 209, 209, 209, 209]
    assert [s["mode"] for s in p["splits"]] == ["anchored", "anchored", "anchored", "cut"]
    spl.apply_op(p, {"op": "set_prompt", "chunk": p["chunks"][1]["id"], "prompt": "x"})
    with pytest.raises(sp.PlanError, match="force"):
        spl.apply_op(p, {"op": "auto_splits"})
    spl.apply_op(p, {"op": "auto_splits", "force": True})
    assert [s["frame"] for s in p["splits"]] == [209, 406, 603, 785]
    # suggested (unconfirmed) cuts don't drive placement: without the cut at 785, no nudge
    q = _plan978(cuts=())
    for f in CUTS_978:
        spl.apply_op(q, {"op": "add_cut", "frame": f, "from": "detected", "confirmed": False})
    assert spl.apply_op(q, {"op": "auto_splits"})["splits"] == [209, 406, 603, 800]
    spl.apply_op(q, {"op": "confirm_cuts"})
    assert all(c["confirmed"] for c in q["cuts"])
    assert spl.apply_op(q, {"op": "auto_splits"})["splits"] == [209, 406, 603, 785]


def test_detected_cuts_merge_as_suggestions():
    p = _plan978(cuts=())
    spl.apply_op(p, {"op": "add_cut", "frame": 96, "from": "detected", "confirmed": True})
    spl.apply_op(p, {"op": "add_cut", "frame": 500, "from": "manual"})
    spl.apply_op(p, {"op": "add_cut", "frame": 700, "from": "detected", "confirmed": False})
    added = spl.merge_detected(p, [96, 155, 250])
    assert added == [155, 250]
    assert [(c["frame"], c["confirmed"]) for c in p["cuts"]] == [(96, True), (155, False), (250, False), (500, True)]
    spl.apply_op(p, {"op": "confirm_cuts", "frames": [250]})
    assert [c["frame"] for c in p["cuts"] if c["confirmed"]] == [96, 250, 500]


def test_detect_cuts_finds_a_hard_cut(tmp_path):
    frames = _clip(20, 40, seed=1) + _clip(25, 210, seed=2) + _clip(20, 90, seed=3)
    path = _write(str(tmp_path / "cuts.mkv"), frames)
    assert spl.detect_cuts(path, FR) == [20, 45]


def test_detect_and_plan_is_one_step_and_keeps_worked_splits():
    p = _plan978(cuts=())
    spl.apply_op(p, {"op": "add_cut", "frame": 785, "from": "detected", "confirmed": False})
    res = spl.detect_and_plan(p, CUTS_978)
    assert all(c["confirmed"] for c in p["cuts"]) and [c["frame"] for c in p["cuts"]] == CUTS_978
    assert res["splits"] == [209, 406, 603, 785] and not res["kept_splits"]
    spl.apply_op(p, {"op": "set_prompt", "chunk": p["chunks"][1]["id"], "prompt": "x"})
    spl.apply_op(p, {"op": "move_split", "split": p["splits"][0]["id"], "to": 200})
    res = spl.detect_and_plan(p, CUTS_978 + [500])
    assert res["kept_splits"] and res["splits"] is None and p["splits"][0]["frame"] == 200
    assert 500 in [c["frame"] for c in p["cuts"] if c["confirmed"]]


def test_jobs_list_and_free_names(tmp_path):
    out = str(tmp_path)
    assert spl.list_jobs(out) == [] and spl.free_job_name("100d clip", out) == "100d_clip"
    for name in ("100d_clip", "100d_clip_2"):
        jd, pp = spl.job_paths(name, out)
        p = sp.new_plan(name, {"path": "C:/v/100d clip.mp4", "frames": 978, "fps": 25, "width": 8, "height": 6})
        sp.rebuild_chunks(p)
        sp.save_plan(pp, p)
    assert {j["job"] for j in spl.list_jobs(out)} == {"100d_clip", "100d_clip_2"}
    assert spl.free_job_name("100d clip", out) == "100d_clip_3"


def test_short_mask_holes_fill_from_the_nearer_side():
    import swap_mask as sm
    m = torch.zeros((12, 2, 2))
    for i, v in enumerate([1, 1, 0, 1, 2, 0, 0, 0, 3, 0, 0, 4]):
        m[i, 0, 0] = 1.0 if v else 0.0
        m[i, 1, 1] = v / 4.0
    out, filled, empty = sm.fill_holes(m, 3)
    assert filled == [2, 5, 6, 7, 9, 10] and empty == []
    assert torch.equal(out[2], m[1]) and torch.equal(out[5], m[4]) and torch.equal(out[6], m[4])   # tie -> earlier
    assert torch.equal(out[7], m[8]) and torch.equal(out[9], m[8]) and torch.equal(out[10], m[11])
    _o, filled, empty = sm.fill_holes(m, 2)                    # the 3-frame hole is too long now
    assert filled == [2, 9, 10] and empty == [5, 6, 7]
    edge = torch.zeros((6, 2, 2)); edge[2:4, 0, 0] = 1; edge[3, 1, 1] = 1   # holes at both ends: one side each
    out, filled, empty = sm.fill_holes(edge, 6)
    assert filled == [0, 1, 4, 5] and empty == []
    assert torch.equal(out[0], edge[2]) and torch.equal(out[5], edge[3])
    assert sm.fill_holes(edge, 1)[1:] == ([], [0, 1, 4, 5])            # too long for max_len 1
    assert sm.fill_holes(torch.zeros((4, 2, 2)), 6)[1:] == ([], [0, 1, 2, 3])   # nobody at all: left empty


# ---------------------------------------------------------------- takes, prompts, states

def _take_files(jd, c, tid, seed=5, prompt="old prompt"):
    rel = f"chunks/{c['id']}/{tid.split('-')[1]}"
    os.makedirs(os.path.join(jd, rel), exist_ok=True)
    with open(os.path.join(jd, rel, "take.mkv"), "wb") as f:
        f.write(b"x")
    with open(os.path.join(jd, rel, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    t = {"id": tid, "state": "ok", "seed": seed, "render": list(c["render"]), "file": f"{rel}/take.mkv",
         "splits": {}, "pins": {"start": None, "end": None}}
    c.setdefault("takes", []).append(t)
    return t


def test_take_trash_restore_and_reuse(tmp_path):
    out = str(tmp_path)
    jd, pp = spl.job_paths("j", out)
    p = _plan978([(203, "cut"), (412, "anchored")])
    p["job"] = "j"
    c = p["chunks"][1]
    t1 = _take_files(jd, c, f"{c['id']}-t001", seed=11, prompt="prompt one")
    _take_files(jd, c, f"{c['id']}-t002", seed=22)
    c["chosen"] = t1["id"]
    sp.save_plan(pp, p)
    r = spl.do_op({"job": "j", "op": "delete_take", "take": t1["id"]}, out)
    cc = sp.find_chunk(r["plan"], c["id"])
    assert [t["id"] for t in cc["takes"]] == [f"{c['id']}-t002"] and cc["chosen"] is None
    assert not os.path.exists(os.path.join(jd, "chunks", c["id"], "t001"))
    trash = r["plan"]["trash"][0]
    assert trash["take"]["id"] == t1["id"] and os.path.isfile(os.path.join(jd, trash["trash"], "take.mkv"))
    with pytest.raises(sp.PlanError):
        spl.do_op({"job": "j", "op": "delete_take", "take": t1["id"]}, out)
    r = spl.do_op({"job": "j", "op": "restore_take", "take": t1["id"]}, out)
    cc = sp.find_chunk(r["plan"], c["id"])
    assert [t["id"] for t in cc["takes"]] == [t1["id"], f"{c['id']}-t002"] and not r["plan"]["trash"]
    assert os.path.isfile(os.path.join(jd, "chunks", c["id"], "t001", "take.mkv"))
    r = spl.do_op({"job": "j", "op": "use_take", "take": t1["id"]}, out)
    cc = sp.find_chunk(r["plan"], c["id"])
    assert cc["prompt"] == "prompt one" and cc["seed_mode"] == {"fixed": 11} and cc["prompt_state"] == "edited"


def test_drafts_failures_and_chunk_states(tmp_path):
    jd = str(tmp_path)
    p = _plan978([(203, "cut"), (412, "anchored"), (609, "anchored")])
    c = p["chunks"]
    c[1]["draft"] = "a drafted prompt"
    spl.apply_op(p, {"op": "adopt_draft", "chunk": c[1]["id"]})
    assert c[1]["prompt"] == "a drafted prompt" and c[1]["prompt_state"] == "draft" and c[1]["draft"] is None
    with pytest.raises(sp.PlanError, match="no draft"):
        spl.apply_op(p, {"op": "adopt_draft", "chunk": c[1]["id"]})
    t = _take_files(jd, c[1], f"{c[1]['id']}-t001")
    _take_files(jd, c[2], f"{c[2]['id']}-t001")
    c[2]["chosen"] = f"{c[2]['id']}-t001"
    c[3]["state"], c[3]["rendering"] = "rendering", {"nonce": "n1", "take": f"{c[3]['id']}-t001"}
    st = {s["chunk"]: s for s in spl.chunk_status(p, jd)}
    assert st[c[0]["id"]]["state"] == "pending" and st[c[0]["id"]]["prompt"] == "empty"
    assert st[c[1]["id"]]["state"] == "unreviewed" and st[c[1]["id"]]["prompt"] == "draft"
    assert st[c[2]["id"]]["state"] == "chosen" and st[c[3]["id"]]["rendering"]["nonce"] == "n1"
    # a newer render's failure is ignored; this one's is recorded, then cleared
    assert spl.apply_op(p, {"op": "render_failed", "chunk": c[3]["id"], "nonce": "old", "error": "x"}).get("ignored")
    spl.apply_op(p, {"op": "render_failed", "chunk": c[3]["id"], "nonce": "n1", "error": "CUDA out of memory"})
    assert {s["chunk"]: s for s in spl.chunk_status(p, jd)}[c[3]["id"]]["failed"] == "CUDA out of memory"
    spl.apply_op(p, {"op": "clear_state", "chunk": c[3]["id"]})
    assert "failed" not in {s["chunk"]: s for s in spl.chunk_status(p, jd)}[c[3]["id"]]
    # the effective take's file gone; then a split moved off its range
    os.remove(os.path.join(jd, t["file"]))
    assert {s["chunk"]: s for s in spl.chunk_status(p, jd)}[c[1]["id"]]["state"] == "missing file"
    spl.apply_op(p, {"op": "move_split", "split": p["splits"][0]["id"], "to": 190})
    assert {s["chunk"]: s for s in spl.chunk_status(p, jd)}[c[1]["id"]]["state"] == "range changed"


def test_join_verdicts_for_the_pills():
    j = {"split": "s1", "type": sp.FORWARD, "stale": False, "left_take": "a", "right_take": "b", "left_chunk": "c1",
         "right_chunk": "c2", "repair": "lock", "override": None, "hand_back": 12}
    assert sp.join_verdict(j, None) is None
    assert sp.join_verdict(j, {"frame_luma": {"at_splice": 0.4}, "char_luma": {"at_splice": 0.3},
                               "join_ratio": 0.9}) == "green"
    assert sp.join_verdict(j, {"frame_luma": {"at_splice": 2.06}, "char_luma": {"at_splice": 0.1}}) == "red"
    assert sp.join_flags(j, {"frame_luma": {"at_splice": 0.4}, "join_ratio": 99})["motion"] == "red"
    assert sp.join_verdict(dict(j, type=sp.STALE, stale=True), None) == "amber"
    assert sp.join_verdict(dict(j, type=sp.STRAIGHT), {"frame_luma": {"at_splice": 9}}) is None
    plan = {"join_cache": {}, "chunks": [{"id": "c2", "takes": [{"id": "b", "joins": [
        {"split": "s1", "left_take": "a", "right_take": "b", "frame_luma": {"at_splice": 0.5}}]}]}]}
    assert sp.join_measure(plan, j)["source"] == "review"
    plan["join_cache"]["s1|a|b|lock|12"] = {"frame_luma": {"at_splice": 0.2}}
    assert sp.join_measure(plan, j) == {"frame_luma": {"at_splice": 0.2}, "source": "assembly"}


# ---------------------------------------------------------------- the mask cache

def test_mask_cover_newest_wins_and_gaps_refuse():
    p = {"masks": [{"id": "m1", "range": [0, 99], "size": [8, 6], "created": "2026-10-03T10:00:00"},
                   {"id": "m2", "range": [50, 149], "size": [8, 6], "created": "2026-10-03T11:00:00"}]}
    cov = sp.mask_cover(p, 20, 120)
    assert [(s["id"], a, b) for s, a, b in cov] == [("m1", 20, 49), ("m2", 50, 120)]
    assert sp.mask_cover(p, 140, 160) is None
    assert sp.mask_coverage(p, 140, 160) == 10
    p["masks"].append({"id": "m3", "range": [100, 200], "size": [16, 12], "created": "2026-10-03T12:00:00"})
    assert sp.mask_cover(p, 90, 110) is None                     # sizes disagree


def test_mark_run_caches_a_mask_that_the_render_run_reuses(nodes):
    import swap_mask as sm
    p = sp.load_plan(nodes["plan"])
    c = p["chunks"]
    out = _run(nodes, "mark", chunk=c[1]["id"])["result"]
    assert all(isinstance(o, spl.ExecutionBlocker) for i, o in enumerate(out) if i not in (12, 15, 16))
    assert out[12] == "{}"                                    # no target named: the render group's "person"
    mark, imgs = out[15], out[16]
    r0, r1 = c[1]["render"]
    assert mark["range"] == [r0, r1] and imgs.shape[0] == r1 - r0 + 1
    got = (imgs.numpy() * 255 + 0.5).astype(np.uint8)
    assert np.array_equal(got, np.stack(nodes["src"][r0:r1 + 1]))
    m = (torch.rand((r1 - r0 + 1, 12, 16)) > 0.5).float()
    m[2] = 0                                                  # SAM3 lost the person on one frame
    with pytest.raises(sm.MaskError, match="1:1"):
        sm.save_mask(mark, m[:-1])
    e = sm.SeamStitchSwapMask().save(mark, m, save_preview=True, fill_holes=0)["result"][0]
    q = sp.load_plan(nodes["plan"])
    seg = q["masks"][0]
    assert seg["id"] == e == "m001" and seg["range"] == [r0, r1] and seg["size"] == [16, 12]
    assert seg["empty"] == [r0 + 2]
    # a segment cached before "empty" was recorded is scanned from its file, with the same answer
    del seg["empty"]
    assert spl.mask_empty_frames(str(nodes["dir"]), seg, FR) == [r0 + 2]
    seg["empty"] = [r0 + 2]
    for f in ("mask.mkv", "preview.mp4"):
        assert os.path.isfile(os.path.join(nodes["dir"], os.path.dirname(seg["file"]), f))
    st = {s["chunk"]: s for s in spl.chunk_status(q, str(nodes["dir"]))}
    assert st[c[1]["id"]]["mask"] == 1.0 and st[c[0]["id"]]["mask"] < 1.0
    assert st[c[1]["id"]]["mask_empty"] == [r0 + 2]
    # the render run of that chunk carries the cached mask, lossless, held frames repeated
    out = _run(nodes, "render", chunk=c[1]["id"], seed=3)["result"]
    assert out[18] is True and out[17].shape[0] == out[5]
    n = r1 - r0 + 1
    assert torch.equal(out[17][:n], m) and torch.equal(out[17][n:], m[-1:].expand(out[5] - n, -1, -1))
    # a chunk the cache doesn't cover tracks its own
    out = _run(nodes, "render", chunk=c[2]["id"], seed=3)["result"]
    assert out[18] is False
    # deleting the mask moves it to the trash
    r = spl.do_op({"job": "t", "op": "delete_mask", "mask": "m001"}, str(nodes["dir"].parent.parent))
    assert not r["plan"]["masks"] and r["plan"]["trash"][-1]["kind"] == "mask"
    assert os.path.isfile(os.path.join(nodes["dir"], r["plan"]["trash"][-1]["trash"], "mask.mkv"))


def test_weak_scene_hits_come_in_as_faint_suggestions_only():
    # 100d's scores over 0.10 (ffmpeg scene, B5b): the six same-room jump cuts B5a found by eye read 0.10-0.14;
    # 903 is the 902 cut read a second time
    scored = [(96, 0.264), (131, 0.120), (155, 0.221), (186, 0.216), (250, 0.202), (349, 0.575), (385, 0.279),
              (413, 0.181), (429, 0.224), (448, 0.118), (669, 0.119), (713, 0.104), (764, 0.134), (785, 0.187),
              (850, 0.119), (902, 0.471), (903, 0.112), (939, 0.192)]
    strong, weak = spl.split_scenes(scored)
    assert strong == CUTS_978
    assert [f for f, _ in weak] == [131, 448, 669, 713, 764, 850]
    p = _plan978(cuts=())
    res = spl.detect_and_plan(p, strong, weak)
    assert res["splits"] == [209, 406, 603, 785]                     # weak hits never move the auto splits
    sug = [c for c in p["cuts"] if not c["confirmed"]]
    assert [c["frame"] for c in sug] == [131, 448, 669, 713, 764, 850] and all(c["weak"] for c in sug)
    assert sug[3]["score"] == 0.104
    # a re-detect replaces the old suggestions and leaves the confirmed cuts alone
    spl.merge_detected(p, strong, weak=weak[:2])
    assert [c["frame"] for c in p["cuts"] if not c["confirmed"]] == [131, 448]
    assert [c["frame"] for c in p["cuts"] if c["confirmed"]] == CUTS_978


def test_a_split_near_an_unconfirmed_cut_says_so():
    p = _plan978()
    spl.apply_op(p, {"op": "add_cut", "frame": 713, "from": "detected", "confirmed": False})
    spl.apply_op(p, {"op": "add_cut", "frame": 448, "from": "detected", "confirmed": False})
    spl.apply_op(p, {"op": "add_split", "frame": 715, "mode": "cut"})       # 2 frames off the suggestion
    spl.apply_op(p, {"op": "add_split", "frame": 455, "mode": "anchored"})  # overlap 443-454 holds 448
    w = [x for x in sp.warnings(978, p["splits"], p["cuts"], p["settings"]) if x["code"] == "suggested_cut"]
    assert [(x["frame"], x["cuts"]) for x in w] == [(455, [448]), (715, [713])]
    spl.apply_op(p, {"op": "confirm_cut", "frame": 448, "confirmed": True})
    codes = {(x.get("frame"), x["code"]) for x in sp.warnings(978, p["splits"], p["cuts"], p["settings"])}
    assert (455, "guard") in codes and (455, "suggested_cut") not in codes


def test_the_strip_frees_memory_before_every_render_and_draft():
    # B6: a render queued from the strip had no free in front of it (0.7 GB of RAM at its lowest). ComfyUI applies
    # a free only between queue items, so the strip frees before queueing and when a render starts with another behind it.
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "js", "swap_planner.js"), encoding="utf-8").read()
    queue_run = src[src.index("async function queueRun("):src.index("function emptyPrompts(")]
    assert 'if (kind === "render" || kind === "draft") await freeMemory(true);' in queue_run.splitlines()[1]
    on_event = src[src.index("function onQueueEvent("):src.index("// ------------------------------------------------------------ prompts and takes")]
    start = on_event[on_event.index('type === "execution_start"'):on_event.index('type === "progress"')]
    assert 'q.kind === "render"' in start and 'x.kind === "render"' in start and "freeMemory()" in start
    assert src.count('fetchApi("/free"') == 1 and "freeForDraft" not in src


def test_a_named_target_reaches_sam3_and_keys_the_mask_cache(nodes):
    # B7: a scene with two people; the job names who to replace, a chunk can name its own, invert marks everyone else
    import swap_mask as sm
    jobs = str(nodes["dir"].parent.parent)
    c = sp.load_plan(nodes["plan"])["chunks"]
    spl.do_op({"job": "t", "op": "set_target", "target": "woman in a purple top"}, jobs)
    spl.do_op({"job": "t", "op": "set_options", "chunk": c[1]["id"], "options": {"target": "woman in a red dress"}}, jobs)
    out = _run(nodes, "mark", chunk=c[1]["id"])["result"]
    assert json.loads(out[12]) == {"target": "woman in a red dress"} and out[15]["target"] == "woman in a red dress"
    r0, r1 = c[1]["render"]
    m = torch.ones((r1 - r0 + 1, 12, 16))
    sm.SeamStitchSwapMask().save(out[15], m, save_preview=False, fill_holes=0)
    out = _run(nodes, "render", chunk=c[1]["id"], seed=3)["result"]
    assert json.loads(out[12]) == {"mark": True, "target": "woman in a red dress"}
    assert out[18] is True                                    # cached for this target
    out = _run(nodes, "render", chunk=c[2]["id"], seed=3)["result"]
    assert json.loads(out[12])["target"] == "woman in a purple top"
    # another target: the cached mask isn't theirs, so the render group tracks them itself
    spl.do_op({"job": "t", "op": "set_options", "chunk": c[1]["id"], "options": {"target": ""}}, jobs)
    q = sp.load_plan(nodes["plan"])
    assert sp.chunk_target(q, q["chunks"][1]) == "woman in a purple top"
    assert {s["chunk"]: s for s in spl.chunk_status(q, str(nodes["dir"]))}[c[1]["id"]]["mask"] == 0.0
    out = _run(nodes, "render", chunk=c[1]["id"], seed=3)["result"]
    assert out[18] is False
    # invert: the job's, unless the chunk says
    spl.do_op({"job": "t", "op": "set_target", "invert": True}, jobs)
    assert json.loads(_run(nodes, "render", chunk=c[1]["id"], seed=3)["result"][12])["invert"] is True
    spl.do_op({"job": "t", "op": "set_options", "chunk": c[1]["id"], "options": {"invert": False}}, jobs)
    assert "invert" not in json.loads(_run(nodes, "render", chunk=c[1]["id"], seed=3)["result"][12])
    assert sp.load_plan(nodes["plan"])["target"] == "woman in a purple top"      # an invert edit leaves the target


def test_the_strip_refuses_a_target_the_workflow_cant_pass_on():
    # B7: a workflow from before the target option marked "person" and cached the mask under the target's name
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "js", "swap_planner.js"), encoding="utf-8").read()
    queue_run = src[src.index("async function queueRun("):src.index("function emptyPrompts(")]
    guard = queue_run[queue_run.index("const named = "):queue_run.index("// The frontend's own queuePrompt")]
    assert '(kind === "render" || kind === "mark") && named' in guard and '"SeamStitchSwapOption"' in guard
    assert '["target", "invert"]' in guard and "throw new Error" in guard
    assert queue_run.index("const named = ") < queue_run.index("queue.call(api")
