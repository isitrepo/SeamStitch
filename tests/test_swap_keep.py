"""SeamStitch Swap: keep-original chunks (design §4.10, B3b).

A kept chunk is never rendered, marked or drafted; the assembly delivers the source there, bit for
bit. Its joins: a straight cut, or "anchored onto the original" (source pins, a fade from the source
into the render on the left; the mirror, a fade out of the render into the untouched source, on the
right). Every plan here has a kept chunk next to a rendered one, the likeliest place for code that
assumes every chunk renders."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
import swap_join as sj  # noqa: E402
from test_swap import CUTS_978, _add_take, _clip, _frames, job  # noqa: E402,F401
from test_swap_nodes import _plan978, _run, nodes  # noqa: E402,F401


def _keep(p, **body):
    return spl.apply_op(p, dict(op="keep", **body))


# ---------------------------------------------------------------- geometry and warnings

def test_a_kept_chunk_has_no_render_and_no_length_warning():
    p = _plan978([(96, "cut"), (300, "anchored")])
    _keep(p, chunk=p["chunks"][0]["id"])
    c = p["chunks"]
    assert c[0]["keep"] and c[0]["render"] is None and c[0]["length"] == 0 and c[0]["fill"] is None
    assert c[1]["render"][0] == 96                         # a straight cut on the left: renders from J
    assert not [w for w in p["warnings"] if w.get("chunk") == 0]     # 96 frames, but no floor warning
    assert not [w for w in p["warnings"] if w.get("split") == c[1]["left"]]   # on the cut at 96: clean


def test_anchored_onto_the_original_on_either_side():
    p = _plan978([(300, "anchored"), (600, "anchored")])
    ov = p["settings"]["overlap"]
    before = [dict(x) for x in p["chunks"]]
    _keep(p, chunk=p["chunks"][0]["id"])
    _keep(p, chunk=p["chunks"][2]["id"])
    m = p["chunks"][1]
    # left kept: the render starts `overlap` early as usual; right kept: it runs `overlap` past 600
    assert m["render"][0] == 300 - ov
    assert m["render"][1] >= 599 + ov and m["deliver"] == before[1]["deliver"]
    assert (m["render"][1] - m["render"][0] + 1 + m["held"]) == m["length"] and m["length"] % 17 == 5


def test_straight_split_mid_shot_next_to_a_kept_chunk_warns_with_its_remedies():
    p = _plan978([(300, "cut")])
    _keep(p, chunk=p["chunks"][0]["id"])
    w = [x for x in p["warnings"] if x["code"] == "no_cut"]
    assert len(w) == 1 and w[0]["kept"] == "left" and "original" in w[0]["text"]
    assert {r["action"] for r in w[0]["remedies"]} == {"anchor_onto_original", "move_to_cut"}
    # two kept chunks side by side: the source either side, no warning at all
    _keep(p, chunk=p["chunks"][1]["id"])
    assert not [x for x in p["warnings"] if x.get("split")]


def test_the_overlap_guard_looks_into_the_kept_stretch_on_the_right():
    p = _plan978([(450, "anchored"), (780, "anchored")])     # the right overlap 780..791 holds the cut at 785
    _keep(p, chunk=p["chunks"][2]["id"])
    g = [x for x in p["warnings"] if x["code"] == "guard"]
    assert [x["frame"] for x in g] == [780] and g[0]["cuts"] == [785]


def test_old_plans_load_unchanged():
    p = _plan978([(209, "anchored"), (406, "anchored")])
    q = json.loads(json.dumps(p))
    sp.rebuild_chunks(q)
    assert [c["render"] for c in q["chunks"]] == [c["render"] for c in p["chunks"]]
    assert q["warnings"] == p["warnings"] and not any(c.get("keep") for c in q["chunks"])


# ---------------------------------------------------------------- the trim handles and the switch

def test_trim_handles_place_the_split_and_keep_the_outer_piece_in_one_write():
    p = _plan978([(406, "anchored"), (603, "anchored")])
    first = p["chunks"][0]
    first["prompt"] = "the old first chunk"
    r = _keep(p, edge="start", frame=95)                    # snaps onto the confirmed cut at 96
    c = p["chunks"]
    assert r["trim"] == 96 and c[0]["keep"] and c[0]["deliver"] == [0, 95]
    s = next(d for d in p["splits"] if d["frame"] == 96)
    assert s["mode"] == "cut" and c[1]["left"] == s["id"]
    assert c[1]["prompt"] == "the old first chunk"          # the rendered remainder keeps its data
    r = _keep(p, edge="end", frame=939)
    assert p["chunks"][-1]["keep"] and p["chunks"][-1]["deliver"] == [939, 977]
    # drag the start handle: its split moves; drag it home: the frames go back to the next chunk
    _keep(p, edge="start", frame=155)
    assert p["chunks"][0]["deliver"] == [0, 154] and p["chunks"][1]["prompt"] == "the old first chunk"
    _keep(p, edge="start", frame=0)
    assert not p["chunks"][0].get("keep") and p["chunks"][0]["deliver"][0] == 0
    assert p["chunks"][0]["prompt"] == "the old first chunk" and 155 not in [d["frame"] for d in p["splits"]]
    _keep(p, edge="end", frame=978)
    assert not any(c.get("keep") for c in p["chunks"])
    with pytest.raises(sp.PlanError, match="pass"):
        _keep(p, edge="start", frame=700)                   # can't jump over the split at 406


def test_the_switch_keeps_and_unkeeps_any_chunk():
    p = _plan978([(300, "anchored"), (600, "anchored")])
    mid = p["chunks"][1]["id"]
    _keep(p, chunk=mid)
    assert sp.find_chunk(p, mid)["render"] is None
    _keep(p, chunk=mid, keep=False)
    assert sp.find_chunk(p, mid)["render"][0] == 300 - p["settings"]["overlap"]


def test_auto_splits_and_detect_and_plan_keep_the_kept_chunks():
    p = _plan978()
    _keep(p, edge="start", frame=96)
    _keep(p, edge="end", frame=939)
    res = spl.detect_and_plan(p, CUTS_978)
    c = p["chunks"]
    assert c[0]["keep"] and c[0]["deliver"] == [0, 95] and c[-1]["keep"] and c[-1]["deliver"] == [939, 977]
    assert res["splits"] and all(96 < f < 939 for f in res["splits"])
    assert {96, 939} <= {d["frame"] for d in p["splits"]}
    assert [x["code"] for x in p["warnings"] if x.get("chunk") in (0, len(c) - 1)] == []
    # a re-plan keeps them again
    spl.apply_op(p, {"op": "auto_splits"})
    assert p["chunks"][0]["keep"] and p["chunks"][-1]["keep"]


# ---------------------------------------------------------------- runs: pins, refusals, drafts

def test_source_pins_next_to_a_kept_chunk_and_the_refusals():
    p = _plan978([(300, "anchored"), (600, "anchored")])
    _keep(p, chunk=p["chunks"][0]["id"])
    _keep(p, chunk=p["chunks"][2]["id"])
    m = p["chunks"][1]
    r0, r1 = m["render"]
    pins, _notes = spl.resolve_pins(p, 1)
    assert pins == {"start": {"source": True, "frames": [r0, r0 + 4]}, "end": {"source": True, "frames": [r1 - 4, r1]}}
    assert spl.resolve_pins(p, 0) == ({"start": None, "end": None}, ["kept as the original: never rendered"])
    with pytest.raises(spl.PlannerError, match="kept as the original"):
        spl.render_descriptor(p, "plan.json", {"action": "render", "chunk": p["chunks"][0]["id"], "prompt": "x"})
    with pytest.raises(spl.PlannerError, match="no mask"):
        spl.mark_descriptor(p, "plan.json", {"action": "mark", "chunk": p["chunks"][2]["id"]})
    assert [c["id"] for c in spl.draft_chunks(p)] == [m["id"]]
    st = {s["chunk"]: s for s in spl.chunk_status(p, ".")}
    assert st[p["chunks"][0]["id"]]["state"] == "original" and st[p["chunks"][0]["id"]]["mask"] is None
    assert "original (kept" in spl.plan_text(p)


def test_join_types_next_to_the_original():
    src = sp.source_take(978)
    left = {"id": "s1", "frame": 300, "mode": "anchored"}
    b = {"id": "b", "render": [288, 520], "splits": {"left": {"frame": 300, "mode": "anchored"}},
         "pins": {"start": {"source": True, "frames": [288, 292]}, "end": None}}
    j = sp.join_info(left, src, b, {"overlap": 12})
    assert (j["type"], j["repair"], j["splice"], j["fade"], j["original"]) == ("entry", "fade", 300, [288, 299], "left")
    j = sp.join_info(left, src, dict(b, pins={}), {"overlap": 12})     # not pinned to the source: stale, still a fade
    assert (j["type"], j["repair"], j["stale"]) == ("stale", "fade", True)
    right = {"id": "s2", "frame": 600, "mode": "anchored"}
    a = {"id": "a", "render": [288, 613], "splits": {"right": {"frame": 600, "mode": "anchored"}},
         "pins": {"start": None, "end": {"source": True, "frames": [609, 613]}}}
    j = sp.join_info(right, a, src, {"overlap": 12})
    assert (j["type"], j["repair"], j["fade"], j["splice"], j["fade_dir"]) == ("exit", "fade", [600, 611], 612, "out")
    assert sp.join_info({"id": "s3", "frame": 600, "mode": "cut"}, a, src)["repair"] == "none"
    assert sp.join_info(right, src, src)["type"] == "original"
    # an override can't lock the original
    assert sp.join_info(dict(left, repair={"mode": "lock"}), src, b, {"overlap": 12})["repair"] == "fade"


# ---------------------------------------------------------------- the assembly, exact

def _keep_job(job, mode):
    def upd(p):
        for d in p["splits"]:
            d["mode"] = mode
        p["chunks"][0]["keep"] = True
        p["chunks"][2]["keep"] = True
        sp.rebuild_chunks(p)
    sp.update_plan(job["plan"], upd)
    return sp.load_plan(job["plan"])


def test_assemble_kept_ends_straight_are_the_source_bit_for_bit(job):
    p = _keep_job(job, "cut")
    r = p["chunks"][1]["render"]
    B = _clip(r[1] - r[0] + 1, 90, seed=6)
    _add_take(job, 1, B, r)
    rep = job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit")
    assert [(j["type"], j["repair"], j["original"]) for j in rep["joins"]] == [("straight", "none", "left"),
                                                                              ("straight", "none", "right")]
    assert rep["takes"][p["chunks"][0]["id"]]["state"] == "original"
    assert "pending" not in {f["code"] for f in rep["flags"]}
    out, src = _frames(rep["file"]), job["src"]
    assert all(np.array_equal(out[f], src[f]) for f in list(range(30)) + list(range(55, 80)))
    assert all(np.array_equal(out[f], B[f - r[0]]) for f in range(30, 55))


def test_assemble_anchored_onto_the_original_fades_in_and_out(job):
    p = _keep_job(job, "anchored")
    m = p["chunks"][1]
    r = m["render"]
    hb = 8
    assert r[0] == 24 and r[1] >= 60
    B = _clip(r[1] - r[0] + 1, 92, -0.3, seed=3)
    _add_take(job, 1, B, r, start={"source": True, "frames": [r[0], r[0] + 4]},
              end={"source": True, "frames": [r[1] - 4, r[1]]})
    rep = job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit", hand_back=hb)
    j1, j2 = rep["joins"]
    assert (j1["type"], j1["repair"], j1["splice"], j1["fade"]) == ("entry", "fade", 30, [24, 29])
    assert (j2["type"], j2["repair"], j2["splice"], j2["fade"], j2["fade_dir"]) == ("exit", "fade", 61, [55, 60], "out")
    src = job["src"]
    at = lambda f: B[f - r[0]]  # noqa: E731
    rin = sj.fade_ratios(sj.means([src[f] for f in range(24, 30)]), sj.means([at(f) for f in range(24, 30)]))
    win = sj.fade_weights(6)
    rout = sj.fade_ratios(sj.means([src[f] for f in range(55, 61)]), sj.means([at(f) for f in range(55, 61)]))
    wout = sj.fade_weights(6)
    exp = [src[f] for f in range(24)]
    exp += [sj.fade_frame(src[f], at(f), rin[f - 24], win[f - 24]) for f in range(24, 30)]
    for f in range(30, 55):
        x = at(f)
        g = sj.decay_gain(rin[-1], f - 30, hb)
        if g is not None:
            x = sj.apply_gain(x, g)
        g = sj.pre_gain(rout[0], 55 - f, hb)
        if g is not None:
            x = sj.apply_gain(x, g)
        exp.append(x)
    exp += [sj.fade_frame_out(at(f), src[f], rout[f - 55], wout[f - 55]) for f in range(55, 61)]
    exp += [src[f] for f in range(61, 80)]
    out = _frames(rep["file"])
    bad = [f for f in range(80) if not np.array_equal(out[f], exp[f])]
    assert bad == [], f"first mismatching output frame: {bad[:1]}"
    # the original either side is untouched
    assert all(np.array_equal(out[f], src[f]) for f in list(range(24)) + list(range(61, 80)))


# ---------------------------------------------------------------- the node: source pins at execution, the Take refuses

def test_render_run_pins_the_source_and_the_take_refuses_a_kept_chunk(nodes):
    import torch
    import swap_take as stk

    def upd(p):
        p["chunks"][0]["keep"] = True
        p["chunks"][2]["keep"] = True
        sp.rebuild_chunks(p)
    sp.update_plan(nodes["plan"], upd)
    p = sp.load_plan(nodes["plan"])
    c = p["chunks"]
    r0, r1 = c[1]["render"]
    out = _run(nodes, "render", chunk=c[1]["id"], seed=3)["result"]
    desc, sp_, ep_ = out[0], out[6], out[7]
    assert desc["pins"] == {"start": {"source": True, "frames": [r0, r0 + 4]}, "end": {"source": True, "frames": [r1 - 4, r1]}}
    u8 = lambda t: (t.numpy() * 255 + 0.5).astype(np.uint8)  # noqa: E731
    assert np.array_equal(u8(sp_), np.stack(nodes["src"][r0:r0 + 5]))
    assert np.array_equal(u8(ep_), np.stack(nodes["src"][r1 - 4:r1 + 1]))
    with pytest.raises(spl.PlannerError, match="kept as the original"):
        _run(nodes, "render", chunk=c[0]["id"])
    with pytest.raises(spl.PlannerError, match="no mask"):
        _run(nodes, "mark", chunk=c[2]["id"])
    # a chunk kept after its render was queued: the take is refused
    sp.update_plan(nodes["plan"], lambda q: q["chunks"][1].__setitem__("keep", True))
    imgs = torch.zeros((desc["length"], 8, 8, 3))
    with pytest.raises(stk.TakeError, match="kept as the original"):
        stk.save_take(desc, imgs)
