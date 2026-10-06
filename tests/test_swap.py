"""SeamStitch Swap: the plan file and chunk geometry (swap_plan), the join repairs (swap_join),
and the streamed assembly (swap_assemble).

Geometry is checked against the numbers the design was signed off with (a 978-frame, 25 fps
source with 11 cuts). Assembly runs on small synthetic clips saved lossless (FFV1 RGB), so
every output pixel is compared with what the join maths says it must be: the likeliest bug
is an off-by-one between source frames, take frames and splice points, and an exact
comparison names it."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_plan as sp  # noqa: E402
import swap_join as sj  # noqa: E402
from test_timeline import _ffmpeg  # noqa: E402

CUTS_978 = [96, 155, 186, 250, 349, 385, 413, 429, 785, 902, 939]


# ---------------------------------------------------------------- geometry

def test_snap_up_to_h3_grid():
    assert [sp.snap_up(n) for n in (1, 5, 6, 22, 23, 175, 176, 209, 210, 250)] == [5, 5, 22, 22, 39, 175, 192, 209, 226, 260]


def test_auto_plan_978_frames():
    # B5b: the nudge to 815 makes a 226-frame render (over the 209 ceiling; it failed, then crashed ComfyUI), so
    # the split becomes a straight cut on the cut at 785 instead: five 209-frame renders
    js = sp.auto_splits(978, CUTS_978, with_modes=True)
    assert js == [(209, "anchored"), (406, "anchored"), (603, "anchored"), (785, "cut")]
    splits = [{"frame": f, "mode": m} for f, m in js]
    g = sp.geometry(978, splits, CUTS_978)
    assert [c["length"] for c in g] == [209, 209, 209, 209, 209]
    assert g[3]["fill"] == {"kind": "head", "frames": 15} and g[3]["render"] == [576, 784]
    assert g[4]["fill"] == {"kind": "hold", "frames": 16} and g[4]["render"] == [785, 977]
    assert sp.warnings(978, splits, CUTS_978) == []
    assert sp.auto_splits(978, CUTS_978) == [209, 406, 603, 785]
    # with the old 226 ceiling the nudge stands (B1a's signed-off plan)
    js = sp.auto_splits(978, CUTS_978, {"ceiling": 226})
    assert js == [209, 406, 603, 815]               # 815: nudged +15 off the cut at 785, after the head fill
    g = sp.geometry(978, js, CUTS_978)
    assert [c["length"] for c in g] == [209, 209, 209, 226, 175]
    assert sum(c["length"] for c in g) == 1028
    assert g[3]["fill"] == {"kind": "tail", "frames": 2} and g[3]["render"] == [591, 816]
    assert sp.warnings(978, js, CUTS_978, {"ceiling": 226}) == []


def test_auto_plan_nudge_is_needed_without_it():
    """At 800 the last chunk's head fill pulls its render start to 786, so the guard holds 785."""
    g = sp.geometry(978, [209, 406, 603, 800], CUTS_978)
    assert g[4]["fill"] == {"kind": "head", "frames": 2} and g[4]["render"][0] == 786
    w = sp.warnings(978, [209, 406, 603, 800], CUTS_978)
    assert [(x["code"], x["frame"], x["cuts"]) for x in w] == [("guard", 800, [785])]


def test_auto_plan_merges_a_short_last_chunk():
    # 209 + 197 = 406; a 400-frame source leaves 400 - 394 = 6 render frames after it: merged (209 + 191 -> 209).
    assert sp.auto_splits(400) == [209]
    assert sp.auto_splits(200) == []
    # 420 leaves 26: merged it would render 226, over the ceiling, so the last split moves back to the floor
    assert sp.auto_splits(420) == [209, 324]
    assert [c["length"] for c in sp.geometry(420, [209, 324])] == [209, 141, 124]


def test_auto_plan_never_passes_the_ceiling_on_short_shots():
    # B7, a sitcom scene (1437 frames, shots down to 18): the split due at 993 sits in the guard of the cut at 999;
    # forward clears at 1018 (a 243 render), the straight cut at 999 renders 226, so it nudges back to 992.
    cuts = [56, 99, 153, 193, 238, 256, 287, 315, 370, 533, 725, 844, 916, 999, 1037, 1126, 1150, 1329, 1392]
    js = sp.auto_splits(1437, cuts, with_modes=True)
    assert js == [(193, "cut"), (402, "anchored"), (599, "anchored"), (796, "anchored"), (992, "anchored"),
                  (1189, "anchored"), (1322, "anchored")]
    splits = [{"frame": f, "mode": m} for f, m in js]
    assert max(c["length"] for c in sp.geometry(1437, splits, cuts)) == 209
    assert sp.warnings(1437, splits, cuts) == []


def test_hand_edited_plan_fills():
    splits = [{"frame": 250, "mode": "cut"}, {"frame": 429, "mode": "cut"}, {"frame": 609, "mode": "anchored"},
              {"frame": 785, "mode": "cut"}]
    g = sp.geometry(978, splits, CUTS_978)
    assert [(c["fill"]["kind"], c["fill"]["frames"]) for c in g] == [
        ("hold", 10), ("hold", 13), ("tail", 12), ("head", 4), ("hold", 16)]
    assert [c["render"] for c in g] == [[0, 249], [250, 428], [429, 620], [593, 784], [785, 977]]
    assert [c["held"] for c in g] == [10, 13, 0, 0, 16]
    assert all((c["render"][1] - c["render"][0] + 1 + c["held"]) == c["length"] for c in g)
    # its first render (0-249 + 10 held) is 260 frames: over the ceiling since B5a (226; 243 / 260 failed two-pass)
    assert [(w["code"], w.get("chunk")) for w in sp.warnings(978, splits, CUTS_978)] == [("ceiling", 0)]


def test_tail_fill_refused_when_a_cut_lies_in_the_extension():
    # chunk 429..608 needs +12; with a cut at 615 the tail would cross it, so the frame is held
    g = sp.geometry(978, [{"frame": 429, "mode": "cut"}, {"frame": 609, "mode": "anchored"}], [429, 615])
    assert g[1]["fill"] == {"kind": "hold", "frames": 12}


def test_anchored_split_near_a_cut_warns_with_remedies():
    w = sp.warnings(978, [{"id": "s1", "frame": 256, "mode": "anchored"}], CUTS_978)
    guard = [x for x in w if x["code"] == "guard"]
    assert len(guard) == 1 and guard[0]["cuts"] == [250] and guard[0]["frame"] == 256
    acts = [r["action"] for r in guard[0]["remedies"]]
    assert acts == ["straight_cut", "move"] and guard[0]["remedies"][0]["frame"] == 250
    clear = guard[0]["remedies"][1]["clear"]
    for J in clear.values():          # both suggested positions really clear the guard
        assert not [x for x in sp.warnings(978, [{"id": "s1", "frame": J, "mode": "anchored"}], CUTS_978)
                    if x["code"] == "guard"]


def test_length_and_straight_split_warnings():
    # 0..99 renders 107 (< 124); 100..499 renders 413 and 500..977 renders 481 (> 362);
    # no cut within 1 frame of 100 or 500
    w = sp.warnings(978, [{"frame": 100, "mode": "cut"}, {"frame": 500, "mode": "cut"}], CUTS_978)
    assert sorted((x["code"], x.get("chunk", x.get("frame"))) for x in w) == [
        ("floor", 0), ("no_cut", 100), ("no_cut", 500), ("trained", 1), ("trained", 2)]
    # a cut one frame away is close enough
    assert not [x for x in sp.warnings(978, [{"frame": 97, "mode": "cut"}], CUTS_978) if x["code"] == "no_cut"]
    assert sp.warnings(400, [{"frame": 200, "mode": "anchored"}], [], {"ceiling": 226}) == []      # 209 and 226 frames
    assert [x["code"] for x in sp.warnings(400, [{"frame": 200, "mode": "anchored"}], [])] == ["ceiling"]   # 226 > 209
    w = sp.warnings(600, [{"frame": 300, "mode": "cut"}], [299])
    assert [x["code"] for x in w] == ["ceiling", "ceiling"]                       # 311 frames each
    # B5a T-CEIL: 226 passed, 243 crashed and 260 failed; B5b: 226 failed, then crashed ComfyUI (two-pass, at the refine)
    assert sp.DEFAULT_SETTINGS["ceiling"] == 209
    assert [x["code"] for x in sp.warnings(418, [{"frame": 209, "mode": "cut"}], [209])] == []          # 209 / 209
    assert [x["code"] for x in sp.warnings(470, [{"frame": 226, "mode": "cut"}], [226])] == ["ceiling", "ceiling"]   # 226 / 244


def test_geometry_rejects_bad_splits():
    with pytest.raises(sp.PlanError):
        sp.geometry(100, [0], [])
    with pytest.raises(sp.PlanError):
        sp.geometry(100, [{"frame": 40}, {"frame": 40}], [])
    with pytest.raises(sp.PlanError):
        sp.geometry(100, [{"frame": 40, "mode": "straight"}], [])


# ---------------------------------------------------------------- lineage

def _take(tid, render, left=None, right=None, start=None, end=None, state="ok"):
    return {"id": tid, "state": state, "render": list(render),
            "splits": {"left": left, "right": right},
            "pins": {"start": start, "end": end}}


ANCH = lambda J: {"frame": J, "mode": "anchored"}  # noqa: E731


def test_join_types_from_lineage():
    split = {"id": "s2", "frame": 412, "mode": "anchored"}
    W = _take("c2-t001", [203, 411], right=ANCH(412))
    X = _take("c3-t001", [400, 608], left=ANCH(412), right=ANCH(609))                     # free
    Xf = _take("c3-t003", [400, 608], left=ANCH(412), right=ANCH(609),
               start={"take": "c2-t001", "frames": [400, 404]})                           # forward-pinned on W
    X2 = _take("c3-t002", [400, 608], left=ANCH(412), right=ANCH(609),
               start={"take": "c2-t001", "frames": [400, 404]}, end={"take": "c4-t001", "frames": [604, 608]})
    Y = _take("c4-t001", [597, 805], left=ANCH(609), start={"take": "c3-t001", "frames": [597, 601]})
    Yfree = _take("c4-t002", [597, 805], left=ANCH(609))

    j = sp.join_info(split, W, Xf)
    assert (j["type"], j["splice"], j["repair"], j["fade"], j["linked"]) == ("forward", 412, "lock", None, True)
    j = sp.join_info(split, W, X2)
    assert (j["type"], j["splice"], j["repair"], j["fade"]) == ("entry", 412, "fade", [400, 411])
    j = sp.join_info(split, W, X)
    assert (j["type"], j["repair"], j["stale"]) == ("stale", "lock", True)
    s609 = {"id": "s3", "frame": 609, "mode": "anchored"}
    j = sp.join_info(s609, X2, Y)
    assert (j["type"], j["splice"], j["repair"]) == ("exit", 604, "lock")
    j = sp.join_info(s609, X, Y)
    assert (j["type"], j["splice"], j["repair"]) == ("forward", 609, "lock")
    j = sp.join_info(s609, X, Yfree)
    assert (j["type"], j["splice"]) == ("stale", 609)
    j = sp.join_info(dict(s609, repair={"mode": "cut"}), X, Yfree)
    assert (j["type"], j["repair"], j["override"]) == ("stale", "none", "cut")
    j = sp.join_info(dict(s609, repair={"mode": "fade", "hand_back": 25}), X, Y)
    assert (j["repair"], j["fade"], j["hand_back"]) == ("fade", [597, 608], 25)
    j = sp.join_info({"id": "s1", "frame": 203, "mode": "cut"}, None, W)
    assert (j["type"], j["repair"]) == ("straight", "none")
    j = sp.join_info(s609, X, None)
    assert j["type"] == "pending"
    # a moved split: Y was rendered for 609, the split is now at 615 -> stale
    j = sp.join_info({"id": "s3", "frame": 615, "mode": "anchored"}, X, Y)
    assert j["type"] == "stale"


def test_effective_take():
    c = {"takes": [_take("c1-t001", [0, 9]), _take("c1-t002", [0, 9])], "chosen": None}
    t, st = sp.effective_take(c)
    assert (t["id"], st) == ("c1-t001", "unreviewed")
    c["chosen"] = "c1-t002"
    assert sp.effective_take(c)[0]["id"] == "c1-t002" and sp.effective_take(c)[1] == "chosen"
    c["takes"][0]["state"] = "failed"
    c["chosen"] = None
    assert sp.effective_take(c)[0]["id"] == "c1-t002"
    assert sp.effective_take({"takes": []}) == (None, "pending")


# ---------------------------------------------------------------- plan file

def _source(n=978):
    return {"path": "src.mp4", "frames": n, "fps": 25, "width": 64, "height": 48, "audio": True, "size": 0, "mtime_ns": 0}


def test_plan_store_revisions_history_and_conflicts(tmp_path):
    path = sp.plan_path(str(tmp_path / "job"))
    p = sp.new_plan("job", _source())
    for f in (209, 406):
        sp.add_split(p, f)
    sp.rebuild_chunks(p)
    saved = sp.save_plan(path, p)
    assert saved["rev"] == 1 and os.path.isfile(path)
    assert os.path.isfile(os.path.join(str(tmp_path / "job"), "history", "plan_1.json"))
    for i in range(60):
        sp.update_plan(path, lambda q: q.__setitem__("subject", f"edit {i}"))
    plan = sp.load_plan(path)
    assert plan["rev"] == 61 and plan["subject"] == "edit 59"
    hist = sorted(os.listdir(os.path.join(str(tmp_path / "job"), "history")))
    assert len(hist) == 50 and "plan_61.json" in hist and "plan_11.json" not in hist
    with pytest.raises(sp.PlanConflict):
        sp.update_plan(path, lambda q: None, expect_rev=3)
    with pytest.raises(sp.PlanConflict):
        sp.save_plan(path, plan, expect_rev=60)
    assert not [f for f in os.listdir(str(tmp_path / "job")) if f.endswith(".tmp")]


def test_plan_refuses_a_newer_format(tmp_path):
    path = str(tmp_path / "plan.json")
    with open(path, "w") as f:
        json.dump({"format": "seamstitch_swap_plan_v2", "rev": 3}, f)
    with pytest.raises(sp.PlanError, match="v2"):
        sp.load_plan(path)


def test_concurrent_updates_are_never_lost(tmp_path):
    import threading
    path = sp.plan_path(str(tmp_path / "job"))
    sp.save_plan(path, sp.new_plan("job", _source()))

    def bump():
        for _ in range(20):
            sp.update_plan(path, lambda q: q.__setitem__("n", q.get("n", 0) + 1))

    th = [threading.Thread(target=bump) for _ in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert sp.load_plan(path)["n"] == 80


def test_rebuild_keeps_chunk_data_when_splits_move():
    p = sp.new_plan("job", _source())
    s1 = sp.add_split(p, 209)
    sp.add_split(p, 406)
    sp.rebuild_chunks(p)
    c2 = p["chunks"][1]
    c2["prompt"] = "kept"
    c2["takes"].append(_take("c2-t001", [197, 405]))
    s1["frame"] = 215
    sp.rebuild_chunks(p)
    assert p["chunks"][1]["id"] == c2["id"] and p["chunks"][1]["prompt"] == "kept"
    assert p["chunks"][1]["deliver"] == [215, 405]
    assert sp.take_covers(p["chunks"][1]["takes"][0], 215, 405)
    assert sp.next_take_id(p["chunks"][1]) == f"{c2['id']}-t002"
    sp.add_split(p, 600)
    sp.rebuild_chunks(p)
    assert len(p["chunks"]) == 4 and p["chunks"][1]["id"] == c2["id"]


def test_plan_joins_treats_a_take_that_no_longer_covers_as_pending():
    p = sp.new_plan("job", _source())
    sp.add_split(p, 209)
    sp.rebuild_chunks(p)
    a = _take("c1-t001", [0, 208], right=ANCH(209))
    b = _take("c2-t001", [197, 977], left=ANCH(209), start={"take": "c1-t001", "frames": [197, 201]})
    p["chunks"][0]["takes"].append(a)
    p["chunks"][1]["takes"].append(b)
    assert sp.plan_joins(p)[0]["type"] == "forward"
    p["splits"][0]["frame"] = 220
    sp.rebuild_chunks(p)
    j = sp.plan_joins(p)[0]
    assert j["type"] == "pending" and j["left_state"] == "range changed"


# ---------------------------------------------------------------- join maths

def test_lock_gains_known_answer():
    left = np.full((12, 3), 100.0)
    right = np.full((12, 3), 95.0)
    g = sj.lock_gains(left, right, 12)
    assert g.shape == (12, 3)
    # prediction 100, right line 95: delta 5 ramped to 0 by frame 12
    expect = (95 + 5 * (1 - np.arange(12) / 12)) / 95
    assert np.allclose(g[:, 0], expect)
    assert np.allclose(sj.lock_gains(left, right, 12, clamp=True).max(), min(100 / 95, sj.MAX_GAIN))
    # the gate: a 0.3-level step on quiet footage is left alone
    assert np.allclose(sj.lock_gains(left, np.full((12, 3), 99.7), 12, gate=True), 1.0)
    assert not np.allclose(sj.lock_gains(left, right, 12, gate=True), 1.0)


def test_lock_follows_the_left_trend():
    """A left take walking into shade is extrapolated, not pulled back to its last level."""
    left = np.repeat((100 - np.arange(12.0))[:, None], 3, 1)     # 100 .. 89, heading for 88
    right = np.full((12, 3), 88.0)
    assert np.allclose(sj.lock_gains(left, right, 12), 1.0)


def test_swing_straightens_an_oscillating_opening():
    left = np.full((12, 3), 100.0)
    right = np.repeat((100 + 3 * (-1) ** np.arange(12.0))[:, None], 3, 1)
    g = sj.lock_gains(left, right, 12, swing=True)
    straightened = right * g
    assert np.ptp(straightened[:, 0]) < 0.5 * np.ptp(right[:, 0])


def test_fade_and_decay_known_answer():
    w = sj.fade_weights(12)
    assert w[0] == 0 and abs(w[-1] - 1) < 1e-12 and np.all(np.diff(w) > 0)
    r = sj.fade_ratios(np.full((3, 3), 100.0), np.full((3, 3), 80.0))
    assert np.allclose(r, 1.25)
    a = np.full((2, 2, 3), 100, np.uint8)
    b = np.full((2, 2, 3), 80, np.uint8)
    assert (sj.fade_frame(a, b, r[0], 0.5) == 100).all()   # level-matched: no step mid-fade
    assert float(sj.decay_gain(1.25, 0, 12)) == pytest.approx(1 + 0.25 * 11 / 12)
    assert sj.decay_gain(1.25, 11, 12) is None


def test_apply_gain_truncates_like_the_cli():
    f = np.full((1, 1, 3), 99, np.uint8)
    assert sj.apply_gain(f, np.array([1.004, 1.0, 0.999]))[0, 0].tolist() == [99, 99, 98]


def test_jumps():
    t = [10, 10, 10, 13, 13, 13]
    j = sj.jumps(t, 100, 103, 100, 106)
    assert (j["at_splice"], j["max"], j["max_at"], j["median"]) == (3, 3, 103, 0)
    # B7: no character on the frame before the splice (an empty mask): nothing to measure, and no NaN for the
    # plan's JSON (the browser refused the whole plan)
    j = sj.jumps([10, 10, float("nan"), 13, 13, 13], 100, 103, 100, 106)
    assert j["at_splice"] is None and (j["max"], j["max_at"]) == (0, 101)
    assert json.loads(json.dumps(j)) == j


def test_tone_compensate_frame_shift_matches_the_overlap():
    left = [np.full((2, 2, 3), 100, np.uint8)] * 4
    right = [np.full((2, 2, 3), 90, np.uint8)] * 8
    out = sj.tone_compensate(left, right, "frame_shift", 4)
    assert np.allclose(out * 255, 100, atol=1e-3)            # the whole segment shifted +10
    out = sj.tone_compensate(left, right, "gain_bias", 4)
    assert np.allclose(out * 255, 100, atol=1e-3)


def test_seam_score_on_a_steady_pan():
    """Five frames either side of a 'join' in a steadily panning texture: seam ~0."""
    rng = np.random.default_rng(0)
    tex = (rng.random((300, 900)) * 255).astype(np.uint8)
    import cv2
    tex = cv2.GaussianBlur(tex, (0, 0), 2)
    frames = [np.ascontiguousarray(tex[20:290, 30 + 4 * i:510 + 4 * i]) for i in range(10)]
    r = sj.seam_score(frames[:5], frames[5:])
    assert r["seam"] < 0.3 and 0.8 < r["diff_ratio"] < 1.2


# ---------------------------------------------------------------- assembly

W_, H_, FR = 64, 48, 25
H264_SPEC = {"main_pass": ["-n", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-vf",
                           "scale=out_color_matrix=bt709", "-color_range", "tv", "-colorspace", "bt709",
                           "-color_primaries", "bt709", "-color_trc", "bt709"],
             "fake_trc": "bt709", "audio_pass": ["-c:a", "aac"], "extension": "mp4"}
# 8-bit RGB FFV1, so the decoded output can be compared bit for bit (the Timeline's lossless cut)
FFV1_SPEC = {"main_pass": ["-n", "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "gbrp"],
             "audio_pass": ["-c:a", "flac"], "extension": "mkv"}
# VHS's own ffv1-mkv: 16-bit RGB in and out
FFV1_16_SPEC = {"main_pass": ["-n", "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "rgba64le"],
                "audio_pass": ["-c:a", "flac"], "input_color_depth": "16bit", "extension": "mkv"}


def _write(path, frames, audio_s=None):
    cmd = [_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_}x{H_}",
           "-r", str(FR), "-i", "-"]
    if audio_s:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={audio_s}", "-c:a", "aac"]
    cmd += ["-c:v", "ffv1", "-pix_fmt", "gbrp", "-g", "1", path]
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)
    return path


def _clip(n, base, slope=0.0, seed=0):
    """n frames: a fixed noise texture around `base`, drifting by `slope` levels per frame."""
    rng = np.random.default_rng(seed)
    tex = rng.integers(-20, 21, (H_, W_, 3))
    return [np.clip(base + slope * i + tex, 0, 255).astype(np.uint8) for i in range(n)]


@pytest.fixture
def job(tmp_path, monkeypatch):
    import swap_assemble as sa
    monkeypatch.setattr(sa, "_format_spec", lambda fmt, crf, pix: dict(FFV1_SPEC if fmt == "test/ffv1-8bit" else
                                                      FFV1_16_SPEC if "ffv1" in fmt else H264_SPEC))
    N = 80
    src = _clip(N, 120, seed=1)
    src_path = _write(str(tmp_path / "source.mkv"), src, audio_s=N / FR)
    jdir = tmp_path / "out" / "seamstitch_swap" / "t"
    p = sp.new_plan("t", {"path": src_path, "frames": N, "fps": FR, "width": W_, "height": H_, "audio": True},
                    {"overlap": 6, "floor": 10})
    sp.add_split(p, 30, "anchored")
    sp.add_split(p, 55, "anchored")
    sp.rebuild_chunks(p)
    path = sp.save_plan(sp.plan_path(str(jdir)), p) and sp.plan_path(str(jdir))
    return {"dir": jdir, "plan": path, "src": src, "N": N, "sa": sa}


def _add_take(ctx, chunk_i, frames, render, **lineage):
    def upd(p):
        c = p["chunks"][chunk_i]
        tid = sp.next_take_id(c)
        rel = f"chunks/{c['id']}/{tid.split('-')[1]}/take.mkv"
        os.makedirs(os.path.dirname(str(ctx["dir"] / rel)), exist_ok=True)
        _write(str(ctx["dir"] / rel), frames)
        left = c["left"] and next(s for s in p["splits"] if s["id"] == c["left"])
        right = next((s for s in p["splits"] if chunk_i + 1 < len(p["chunks"]) and s["id"] == p["chunks"][chunk_i + 1]["left"]), None)
        t = {"id": tid, "state": "ok", "render": list(render), "file": rel,
             "splits": {"left": left and {"frame": left["frame"], "mode": left["mode"]},
                        "right": right and {"frame": right["frame"], "mode": right["mode"]}},
             "pins": {"start": lineage.get("start"), "end": lineage.get("end")}}
        c["takes"].append(t)
        ctx["last"] = tid
    sp.update_plan(ctx["plan"], upd)
    return ctx["last"]


def _frames(path):
    import timeline as tl
    return np.stack(list(tl._iter_frames(path, FR, 0, None)))


def test_assemble_forward_entry_exit_exact(job):
    p = sp.load_plan(job["plan"])
    r = [c["render"] for c in p["chunks"]]
    assert [c["deliver"] for c in p["chunks"]] == [[0, 29], [30, 54], [55, 79]]
    A = _clip(r[0][1] - r[0][0] + 1, 100, 0.5, seed=2)
    B = _clip(r[1][1] - r[1][0] + 1, 92, -0.3, seed=3)
    C = _clip(r[2][1] - r[2][0] + 1, 130, 0.0, seed=4)
    ta = _add_take(job, 0, A, r[0])
    tb = _add_take(job, 1, B, r[1], start={"take": ta, "frames": [r[1][0], r[1][0] + 4]})
    tc = _add_take(job, 2, C, r[2], start={"take": tb, "frames": [r[2][0], r[2][0] + 4]})
    at = lambda T, rr, f: T[f - rr[0]]  # noqa: E731
    src = job["src"]

    def heading(lm, J, mode):       # the default (B5b): the left take's last 3 frames carried by the source's change
        return sj.source_heading(lm, sj.source_ratios(sj.means(src[J - sj.HEADING_K:J + 1]))) if mode == "source" else None

    for mode in ("source", "line"):  # "line": the port, per split (repair {"heading": "line"})
        if mode == "line":
            sp.update_plan(job["plan"], lambda q: [d.__setitem__("repair", {"heading": "line"}) for d in q["splits"]])
        rep = job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit")
        assert [j["type"] for j in rep["joins"]] == ["forward", "forward"]
        assert [j.get("lock_heading") for j in rep["joins"]] == [mode, mode]
        out = _frames(rep["file"])
        assert len(out) == job["N"] and rep["checks"]["frames_ok"] and rep["checks"]["audio_ok"]
        assert rep["checks"]["audio"] == "copy"
        # expected, from the join maths on the takes' own frames
        exp = []
        for f in range(0, 30):
            exp.append(at(A, r[0], f))
        lm = sj.means([at(A, r[0], f) for f in range(18, 30)])
        g1 = sj.lock_gains(lm, sj.means([at(B, r[1], f) for f in range(30, 42)]), 12, heading=heading(lm, 30, mode))
        for f in range(30, 55):
            x = at(B, r[1], f)
            exp.append(sj.apply_gain(x, g1[f - 30]) if f - 30 < 12 else x)
        lm = sj.means([at(B, r[1], f) for f in range(43, 55)])
        g2 = sj.lock_gains(lm, sj.means([at(C, r[2], f) for f in range(55, 67)]), 12, heading=heading(lm, 55, mode))
        for f in range(55, 80):
            x = at(C, r[2], f)
            exp.append(sj.apply_gain(x, g2[f - 55]) if f - 55 < 12 else x)
        bad = [f for f in range(job["N"]) if not np.array_equal(out[f], exp[f])]
        assert bad == [], f"{mode}: first mismatching output frame: {bad[:1]}"
    sp.update_plan(job["plan"], lambda q: [d.pop("repair", None) for d in q["splits"]])

    # re-roll the middle chunk two-sided: an entry fade at 30, an exit lock at C's end-pin start
    B2 = _clip(r[1][1] - r[1][0] + 1, 85, 0.2, seed=5)
    e0 = r[2][0]
    tb2 = _add_take(job, 1, B2, r[1], start={"take": ta, "frames": [r[1][0], r[1][0] + 4]},
                    end={"take": tc, "frames": [e0, e0 + 4]})
    sp.update_plan(job["plan"], lambda q: q["chunks"][1].__setitem__("chosen", tb2))
    rep = job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit", hand_back=8)
    assert [(j["type"], j["splice"], j["repair"]) for j in rep["joins"]] == [("entry", 30, "fade"), ("exit", e0, "lock")]
    out = _frames(rep["file"])
    f0 = r[1][0]
    lm = sj.means([at(A, r[0], f) for f in range(f0, 30)])
    rm = sj.means([at(B2, r[1], f) for f in range(f0, 30)])
    ratio, w = sj.fade_ratios(lm, rm), sj.fade_weights(30 - f0)
    exp = [at(A, r[0], f) for f in range(0, f0)]
    exp += [sj.fade_frame(at(A, r[0], f), at(B2, r[1], f), ratio[f - f0], w[f - f0]) for f in range(f0, 30)]
    for f in range(30, e0):
        g = sj.decay_gain(ratio[-1], f - 30, 8)
        exp.append(sj.apply_gain(at(B2, r[1], f), g) if g is not None else at(B2, r[1], f))
    lm = sj.means([at(B2, r[1], f) for f in range(e0 - 12, e0)])
    g3 = sj.lock_gains(lm, sj.means([at(C, r[2], f) for f in range(e0, e0 + 8)]), 8, heading=heading(lm, e0, "source"))
    for f in range(e0, 80):
        x = at(C, r[2], f)
        exp.append(sj.apply_gain(x, g3[f - e0]) if f - e0 < 8 else x)
    bad = [f for f in range(job["N"]) if not np.array_equal(out[f], exp[f])]
    assert bad == [], f"first mismatching output frame: {bad[:1]}"
    plan = sp.load_plan(job["plan"])
    assert len(plan["assembled"]) == 3 and plan["join_cache"]
    assert os.path.isfile(os.path.splitext(rep["file"])[0] + ".report.json")


def test_assemble_pending_stale_override_and_refusals(job):
    sa = job["sa"]
    p = sp.load_plan(job["plan"])
    r = [c["render"] for c in p["chunks"]]
    B = _clip(r[1][1] - r[1][0] + 1, 90, seed=6)
    C = _clip(r[2][1] - r[2][0] + 1, 140, seed=7)
    _add_take(job, 1, B, r[1])                    # free: stale against nothing (chunk 0 is pending)
    _add_take(job, 2, C, r[2])                    # free: stale against B

    def cut_s2(q):
        q["splits"][1]["repair"] = {"mode": "cut"}
    sp.update_plan(job["plan"], cut_s2)
    rep = sa.assemble(job["plan"], fmt="test/ffv1-8bit")
    assert [(j["type"], j["repair"], j["override"]) for j in rep["joins"]] == [("pending", "none", None),
                                                                              ("stale", "none", "cut")]
    assert {f["code"] for f in rep["flags"]} >= {"pending", "stale", "unreviewed"}
    out = _frames(rep["file"])
    src = _frames(p["source"]["path"])
    assert all(np.array_equal(out[f], src[f]) for f in range(30))            # source frames under the pending chunk
    assert all(np.array_equal(out[f], B[f - r[1][0]]) for f in range(30, 55))
    assert all(np.array_equal(out[f], C[f - r[2][0]]) for f in range(55, 80))  # plain cut
    with pytest.raises(sa.AssembleError, match="refuse"):
        sa.assemble(job["plan"], fmt="test/ffv1-8bit", pending_chunks=sa.PENDING_REFUSE)
    with pytest.raises(sa.AssembleError, match="unreviewed"):
        sa.assemble(job["plan"], fmt="test/ffv1-8bit", require_reviewed=True)
    os.remove(str(job["dir"] / sp.load_plan(job["plan"])["chunks"][2]["takes"][0]["file"]))
    with pytest.raises(sa.AssembleError, match="missing"):
        sa.assemble(job["plan"], fmt="test/ffv1-8bit")


def test_assemble_h264_output_is_1to1_with_the_source_audio(job):
    """The default format: VHS's h264-mp4 arguments, the source's AAC copied in whole."""
    sa = job["sa"]
    rep = sa.assemble(job["plan"])            # every chunk pending: the source re-encoded
    assert rep["file"].endswith(".mp4") and rep["checks"]["frames_ok"] and rep["checks"]["audio_ok"]
    out = _frames(rep["file"])
    src = _frames(sp.load_plan(job["plan"])["source"]["path"])
    assert len(out) == len(src) == job["N"]
    # 4:2:0 costs this per-pixel colour-noise texture its chroma and ~1 level of mean (one h264
    # generation: -0.85 measured on real takes, -1.2 in build_cut's notes); luma survives
    y = lambda f: f.astype(np.float64) @ sj.LUMA_BT709  # noqa: E731
    psnr_y = 10 * np.log10(255 ** 2 / np.mean((y(out) - y(src)) ** 2))
    assert psnr_y > 35 and abs(out.mean() - src.mean()) < 1.5
    # stream copy: every packet kept; MKV's millisecond timestamps vs MP4's leave < 1 ms
    assert abs(rep["checks"]["audio_seconds"] - rep["checks"]["source_audio_seconds"]) < 1e-3


def test_assemble_formats_drop_json_entries(monkeypatch):
    import swap_assemble as sa
    import result_preview as rp
    monkeypatch.setattr(rp, "_formats", lambda: ["video/h264-mp4", "video/h264-mp4.json", "video/ffv1-mkv"])
    assert sa.formats() == ["video/h264-mp4", "video/ffv1-mkv"]


def test_assemble_node_widget_order():
    """ComfyUI restores widget values by position: this order is frozen (append only)."""
    import swap_assemble as sa
    req = sa.SeamStitchSwapAssemble.INPUT_TYPES()["required"]
    assert list(req) == ["assemble_plan", "hand_back_frames", "format", "crf", "pix_fmt", "filename_prefix",
                         "pending_chunks", "require_reviewed"]
    assert sa.SeamStitchSwapAssemble.RETURN_NAMES == ("video_path", "Filenames", "report")


def test_assemble_16bit_master_holds_the_8bit_frames_exactly(job):
    """VHS's ffv1-mkv takes 16-bit input: each 8-bit level v is written as v * 257, exactly what
    VHS's tensor_to_shorts makes of v / 255, so the master holds the frames losslessly."""
    import av
    rep = job["sa"].assemble(job["plan"], fmt="video/ffv1-mkv")
    with av.open(rep["file"]) as c:
        out = np.stack([f.to_ndarray(format="rgb48le") for f in c.decode(video=0)])
    src = _frames(sp.load_plan(job["plan"])["source"]["path"])
    assert out.shape[0] == job["N"] and np.array_equal(out, src.astype(np.uint16) * 257)
    assert np.array_equal(_frames(rep["file"]), src)                                  # and reads back as itself


def test_16bit_ffv1_reads_back_to_the_exact_8bit_levels(tmp_path):
    """Every 8-bit level, written as v * 257 through the 16-bit ffv1 path (FFV1_16_SPEC's main pass),
    comes back through timeline._iter_frames as v. swscale's own 16->8 bit conversion lifted levels
    from about 110 up by one (B3b); the decoder now rounds x / 257 itself."""
    import av
    import timeline as tl
    lv = np.arange(256, dtype=np.uint8).reshape(16, 16)
    src = np.stack([np.stack([np.roll(lv, i, 0), lv[::-1], lv.T], -1) for i in range(8)])
    path = str(tmp_path / "master16.mkv")
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", "16x16",
                    "-r", str(FR), "-i", "-", *FFV1_16_SPEC["main_pass"][1:], path],
                   input=(src.astype(np.uint16) * 257).tobytes(), check=True)
    with av.open(path) as c:
        assert c.streams.video[0].codec_context.format.name == "gbrap16le"
        swscale = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    assert not np.array_equal(swscale, src)                                           # the old read was off
    assert tl.probe(path, FR)["deep_rgb"]
    assert np.array_equal(np.stack(list(tl._iter_frames(path, FR, 0, None))), src)
    assert np.array_equal(np.stack(list(tl._iter_frames(path, FR, 3, 6))), src[3:6])


# ---------------------------------------------------------------- B1b: the cut-aware, regional lock

def test_lock_fits_stop_at_a_cut():
    m = np.array([[60.0] * 3, [60.2] * 3, [45.0] * 3, [44.8] * 3, [44.5] * 3])       # the take cuts at frame 2
    assert sj.opening_frames(m, 412, 12) == 2                                          # its own step
    assert sj.opening_frames(m, 412, 12, cuts=[413]) == 1                             # the confirmed cut, earlier
    assert sj.opening_frames(m, 412, 12, cuts=[412]) == 0                             # the splice is a cut
    left = np.array([[90.0] * 3] * 4 + [[60.0] * 3] * 8)                               # the left take cut 8 frames ago
    assert sj.heading_frames(left, 609) == 8
    assert sj.heading_frames(left, 609, cuts=[605]) == 4
    assert sj.line_fit(np.array([[5.0, 6.0, 7.0]]))[1].tolist() == [0.0, 0.0, 0.0]


def test_solved_field_gains_land_each_region_on_its_target():
    rng = np.random.default_rng(0)
    f = rng.integers(30, 200, (40, 60, 3)).astype(np.uint8)
    mask = np.zeros((40, 60), bool)
    mask[10:30, 20:40] = True
    wgt = sj.soft_mask(mask, (60, 40), feather=0.05)
    cm, bm = sj.region_means([f], [mask])
    tc, tb = cm[0] * 1.02, bm[0] * 1.06
    gc, gb = sj.solve_field_gains(f, mask, wgt, tc, tb)
    g = wgt[..., None] * gc + (1 - wgt[..., None]) * gb
    out = f.astype(np.float64) * g
    assert np.allclose(out[mask].mean(0), tc, atol=1e-6) and np.allclose(out[~mask].mean(0), tb, atol=1e-6)


def test_the_lock_heading_follows_the_source_not_a_bent_line():
    """B5b, the test clip at 406: the left take dips then brightens over its last frames (render flicker); a line through
    its last 12 frames lands under its last frame and the lock steps the colour down. The source-carried
    heading continues from the last frame as the source moves."""
    left = np.array([[80.4 - 0.5 * min(k, 6) + 0.6 * max(0, k - 6)] * 3 for k in range(12)])   # dip, then rise
    right = np.full((6, 3), 77.6)
    line = sj.lock_gains(left, right, 6)
    src = sj.lock_gains(left, right, 6, heading=sj.source_heading(left, [1.0, 1.0, 1.0]))
    out_line, out_src = right[0] * line[0], right[0] * src[0]
    assert abs(out_src[0] - left[-1][0]) < 1e-9                  # no step against a still source
    assert out_line[0] < left[-1][0] - 0.5                         # the line's step down
    assert np.allclose(src[-1], 1 + (src[0] - 1) / 6, atol=1e-9)   # the same ramp back to 1
    # the source's own change is carried: a source brightening 2% at the splice brightens the target 2%
    g = sj.lock_gains(left, right, 6, heading=sj.source_heading(left, [1.02, 1.02, 1.02]))
    assert np.allclose(right[0] * g[0], left[-1] * 1.02)
    # three frames, each carried by the source's change since: a still source averages the last three
    k3 = sj.source_heading(left, sj.source_ratios(np.full((4, 3), 50.0)))
    assert np.allclose(k3, left[-3:].mean(0)) and sj.HEADING_K == 3
