"""Object keep: the source's own pixels of a named object pasted over the takes at assembly (swap_keep)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_keep as sk  # noqa: E402
import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
from test_swap import FR, H_, W_, _add_take, _clip, _frames, _write, job  # noqa: F401,E402  (job is a fixture)


def _plan(masks=(), keeps=(), target="man"):
    return {"source": {"frames": 100, "fps": 25}, "target": target, "masks": list(masks), "object_keeps": list(keeps)}


def _mask(mid, a, b, target="white shoe", created="2026-10-20T10:00:00"):
    return {"id": mid, "range": [a, b], "frames": b - a + 1, "file": f"masks/{mid}/mask.mkv", "target": target,
            "created": created, "size": [W_, H_]}


def test_keep_segments_skip_uncovered_frames_and_take_the_newest():
    keep = {"target": "white shoe", "range": [5, 60]}
    p = _plan([_mask("m1", 0, 20), _mask("m2", 30, 50), _mask("m3", 40, 70, created="2026-10-20T11:00:00"),
               _mask("m4", 0, 99, target="man")])
    segs = [(s["id"], a, b) for s, a, b in sp.keep_segments(p, keep, 0, 99)]
    assert segs == [("m1", 5, 20), ("m2", 30, 39), ("m3", 40, 60)]          # 21-29 has no mask: nothing pasted there
    assert sp.keep_segments(p, keep, 21, 29) == []
    assert [(s["id"], a, b) for s, a, b in sp.keep_segments(p, keep, 45, 47)] == [("m3", 45, 47)]


def test_object_keeps_fill_defaults_and_skip_off():
    p = _plan(keeps=[{"id": "k001", "target": "white shoe", "range": [0, 9]},
                     {"id": "k002", "target": "hat", "range": [0, 9], "on": False, "grow": 2}])
    ks = sp.object_keeps(p)
    assert [k["id"] for k in ks] == ["k001"]
    assert (ks[0]["grow"], ks[0]["feather"], ks[0]["level"]) == (sp.KEEP_GROW, sp.KEEP_FEATHER, sp.KEEP_LEVEL_NONE)
    assert [k["grow"] for k in sp.object_keeps(p, on_only=False)] == [sp.KEEP_GROW, 2]


def test_alpha_resizes_grows_and_feathers():
    m = np.zeros((24, 32), np.uint8)
    m[8:16, 8:16] = 255                                    # half size: 16..32 x 16..32 at 64x48
    a, g = sk.alpha(m, 64, 48, grow=0, feather=0)
    assert a.shape == (48, 64) and g[20, 20] and not g[10, 10] and set(np.unique(a)) == {0.0, 1.0}
    a2, g2 = sk.alpha(m, 64, 48, grow=3, feather=0)
    assert g2.sum() > g.sum() and g2[14, 20] and not g[14, 20]
    a3, _ = sk.alpha(m, 64, 48, grow=0, feather=2)
    assert 0 < a3[16, 24] < 1 and a3[24, 24] > 0.99 and a3[2, 2] < 0.01
    e, ge = sk.alpha(np.zeros((24, 32), np.uint8), 64, 48, 6, 4)
    assert not ge.any() and float(e.max()) == 0.0


def test_ring_gain_matches_the_take_and_is_clamped():
    src = np.full((48, 64, 3), 100, np.uint8)
    grown = np.zeros((48, 64), bool)
    grown[16:32, 20:44] = True
    assert np.allclose(sk.ring_gain(src, src, grown), 1.0)
    take = np.full((48, 64, 3), 110, np.uint8)
    assert np.allclose(sk.ring_gain(take, src, grown), 1.1, atol=1e-3)
    take[:] = 40                                           # far darker: clamped
    assert np.allclose(sk.ring_gain(take, src, grown), sk.GAIN_CLAMP[0])
    assert np.allclose(sk.ring_gain(take, src, np.zeros_like(grown)), 1.0)


def test_paste_blends_by_alpha():
    t = np.full((2, 2, 3), 10, np.uint8)
    s = np.full((2, 2, 3), 200, np.uint8)
    a = np.array([[0.0, 1.0], [0.5, 0.25]], np.float32)
    out = sk.paste(t, s, a)
    assert out[0, 0, 0] == 10 and out[0, 1, 0] == 200 and out[1, 0, 0] == 105 and out[1, 1, 0] == 58
    assert sk.paste(t, s, a, np.array([0.5, 0.5, 0.5], np.float32))[0, 1, 0] == 100


def test_ops_add_set_remove_object_keep():
    p = {"source": {"frames": 100}, "target": "man in a maroon suit",
         "chunks": [{"id": "c2", "deliver": [0, 49]}, {"id": "c3", "deliver": [50, 99]}], "splits": []}
    r = spl.apply_op(p, {"op": "add_object_keep", "target": " white shoe ", "chunk": "c3"})
    assert r["keep"] == {"id": "k001", "target": "white shoe", "range": [50, 99]}
    assert spl.apply_op(p, {"op": "add_object_keep", "target": "hat"})["keep"]["range"] == [0, 99]
    spl.apply_op(p, {"op": "set_object_keep", "id": "k001", "grow": 500, "feather": 3, "level": "none", "on": False})
    k = p["object_keeps"][0]
    assert (k["grow"], k["feather"], k["level"], k["on"], k["target"]) == (200, 3, "none", False, "white shoe")
    with pytest.raises(sp.PlanError):
        spl.apply_op(p, {"op": "add_object_keep", "target": ""})
    with pytest.raises(sp.PlanError):                     # the job's own target is the thing being replaced
        spl.apply_op(p, {"op": "add_object_keep", "target": "man in a maroon suit"})
    with pytest.raises(sp.PlanError):
        spl.apply_op(p, {"op": "set_object_keep", "id": "k001", "range": [90, 120]})
    with pytest.raises(sp.PlanError):
        spl.apply_op(p, {"op": "set_object_keep", "id": "k001", "level": "loud"})
    spl.apply_op(p, {"op": "remove_object_keep", "id": "k001"})
    assert [k["id"] for k in p["object_keeps"]] == ["k002"]
    assert spl.apply_op(p, {"op": "add_object_keep", "target": "cane"})["keep"]["id"] == "k003"


def test_a_keep_mark_run_tracks_the_keeps_target():
    p = {"job": "j", "rev": 3, "target": "man", "source": {"path": "s.mp4", "frames": 100, "fps": 25, "width": 64,
                                                           "height": 48, "audio": True},
         "chunks": [{"id": "c2", "deliver": [0, 99], "render": [0, 99]}], "splits": [],
         "object_keeps": [{"id": "k001", "target": "white shoe", "range": [10, 60]}]}
    d = spl.mark_descriptor(p, "plan.json", {"action": "mark", "keep": "k001"})
    assert (d["target"], d["range"], d["kind"], d["keep"], d["chunk"]) == ("white shoe", [10, 60], "keep", "k001", None)
    d = spl.mark_descriptor(p, "plan.json", {"action": "mark", "keep": "k001", "range": [20, 30]})
    assert d["range"] == [20, 30] and d["target"] == "white shoe"
    d = spl.mark_descriptor(p, "plan.json", {"action": "mark", "chunk": "c2"})
    assert d["target"] == "man" and "kind" not in d
    with pytest.raises(spl.PlannerError):
        spl.mark_descriptor(p, "plan.json", {"action": "mark", "keep": "k009"})


def test_assembly_pastes_the_source_inside_the_keep_mask_only(job):  # noqa: F811
    p = sp.load_plan(job["plan"])
    r = [c["render"] for c in p["chunks"]]
    for i, base in enumerate((60, 70, 80)):                # takes far from the source (120)
        _add_take(job, i, _clip(r[i][1] - r[i][0] + 1, base, seed=5 + i), r[i])
    plain = _frames(job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit")["file"])
    box = np.zeros((H_, W_, 3), np.uint8)
    box[10:26, 20:36] = 255
    a, b = 20, 40                                          # the mask covers 20..40, across the split at 30
    (job["dir"] / "masks" / "m001").mkdir(parents=True)
    _write(str(job["dir"] / "masks" / "m001" / "mask.mkv"), [box] * (b - a + 1))

    def upd(q):
        q["masks"] = [{"id": "m001", "range": [a, b], "frames": b - a + 1, "file": "masks/m001/mask.mkv",
                       "target": "white shoe", "kind": "keep", "created": sp.now(), "size": [W_, H_]}]
        q["object_keeps"] = [{"id": "k001", "target": "white shoe", "range": [0, 79], "grow": 0, "feather": 0,
                              "level": "none"}]
    sp.update_plan(job["plan"], upd)
    rep = job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit")
    out = _frames(rep["file"])
    src = np.stack(job["src"])
    inside = box[..., 0] > 0
    for f in range(job["N"]):
        if a <= f <= b:
            assert np.array_equal(out[f][inside], src[f][inside]), f
            assert np.array_equal(out[f][~inside], plain[f][~inside]), f
        else:
            assert np.array_equal(out[f], plain[f]), f
    assert rep["object_keeps"]["k001"]["frames"] == b - a + 1
    assert rep["checks"]["frames_ok"] and rep["checks"]["audio_ok"]
    # off: the plain assembly
    sp.update_plan(job["plan"], lambda q: q["object_keeps"][0].__setitem__("on", False))
    assert np.array_equal(_frames(job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit")["file"]), plain)


def test_take_side_reads_which_shot_a_take_frame_is_in():
    rng = np.random.default_rng(0)
    wide = rng.integers(0, 255, (54, 96, 3)).astype(np.uint8)
    close = rng.integers(0, 255, (54, 96, 3)).astype(np.uint8)
    b, a = sk.thumb(wide), sk.thumb(close)
    graded = np.clip(wide.astype(np.int16) * 0.8 + 10, 0, 255).astype(np.uint8)    # another grade, same shot
    assert sk.take_side(sk.thumb(graded), b, a) == "before"
    assert sk.take_side(sk.thumb(close), b, a) == "after"
    mix = ((wide.astype(np.uint16) + close) // 2).astype(np.uint8)                 # a dissolve: neither
    assert sk.take_side(sk.thumb(mix), b, a) is None
