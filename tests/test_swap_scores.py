"""SeamStitch Swap: the scores and flags (swap_scores, design §4.5).

The measures are ports of the CR test scripts (scorer.py, mouth.py, r10's pose IoU, r16's join
jumps). B3 reproduced r10-r16's numbers with them from the saved renders (plan doc "#### B3");
here they run on constructed inputs whose answers are known, and the bands are checked at their
edges through THRESHOLDS, so a refit moves the test with it."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import swap_scores as ss  # noqa: E402

T = ss.THRESHOLDS


def _box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


# ---------------------------------------------------------------- following

def test_pose_iou_mean_and_p10():
    a = _box(40, 40, 0, 20, 0, 20)
    same = [a] * 9
    half = _box(40, 40, 0, 20, 10, 30)                # overlaps a on 10x20 of a 30x20 union: 1/3
    r = ss.pose_iou([a] * 10, same + [half])
    assert r["frames"] == 10
    assert r["pose_iou"] == pytest.approx((9 + 1 / 3) / 10, abs=1e-4)
    assert r["pose_iou_p10"] == pytest.approx(np.percentile([1] * 9 + [1 / 3], 10), abs=1e-4)


def test_pose_iou_resizes_the_output_mask_and_skips_empty_frames():
    a = _box(40, 80, 10, 30, 20, 60)
    b = _box(20, 40, 5, 15, 10, 30)                   # the same box at half size
    e = np.zeros((40, 80), bool)
    r = ss.pose_iou([a, e], [b, e])
    assert r["frames"] == 1 and r["pose_iou"] == 1.0
    assert ss.pose_iou([e], [e]) is None


# ---------------------------------------------------------------- lost cuts

def _scorer_cut_stats(d, c):
    """tools/cr_tests/scorer.py cut_stats, verbatim: the port must equal it."""
    lo, hi = max(1, c - 6), min(len(d), c + 7)
    neigh = [d[t] for t in range(lo, hi) if t != c]
    w = d[max(1, c - 5): c + 6]
    near = d[max(1, c - 3): c + 4]
    off = int(np.argmax(near)) + max(1, c - 3) - c
    p = c + off
    neigh_p = [d[t] for t in range(max(1, p - 6), min(len(d), p + 7)) if t != p]
    return {"peak_ratio": round(float(d[c] / (np.median(neigh) + 1e-6)), 2),
            "spread": int((w > 0.5 * w.max()).sum()), "offset": off,
            "peak_ratio_at_offset": round(float(d[p] / (np.median(neigh_p) + 1e-6)), 2)}


def test_cut_stats_is_scorers():
    rng = np.random.default_rng(1)
    d = rng.uniform(0.5, 1.5, 60)
    d[0] = 0
    d[30] = 9.0
    for c in (5, 28, 30, 31, 54):
        assert ss.cut_stats(d, c) == _scorer_cut_stats(d, c)


def test_lost_cuts_copied_unsure_lost():
    base = np.full(80, 1.0)
    base[0] = 0
    copied = base.copy()
    copied[21] = 8.0                                  # rendered a frame late: found at the offset
    smeared = base.copy()
    smeared[17:26] = 1.5                              # a morph spread over 9 frames
    unsure = base.copy()
    unsure[40] = 2.5
    r = ss.lost_cuts(copied, [20], 0)
    assert r["20"]["state"] == "copied" and r["20"]["offset"] == 1
    assert ss.lost_cuts(smeared, [20], 0)["20"]["state"] == "lost"
    assert ss.lost_cuts(unsure, [40], 0)["40"]["state"] == "unsure"
    # source frame numbers, and only cuts strictly inside with a frame either side
    assert list(ss.lost_cuts(copied, [420, 400, 479, 500], 400)) == ["420"]


def test_frame_diffs_of_a_hard_step():
    f = [np.full((54, 96, 3), v, np.uint8) for v in (10, 10, 60, 60)]
    d = ss.frame_diffs(f)
    assert d.tolist() == [0.0, 0.0, 50.0, 0.0]


# ---------------------------------------------------------------- mouth sync

def test_mouth_score_finds_the_lag_within_two_and_reports_face_coverage():
    rng = np.random.default_rng(3)
    src = list(rng.uniform(0, 0.5, 120))
    out = [None] * 120
    for t in range(120):
        if 0 <= t + 2 < 120:
            out[t] = src[t + 2]                       # the output leads the source by 2 frames
    for t in range(0, 120, 4):
        out[t] = None                                 # a hand over the mouth: no face found
    r = ss.mouth_score(src, out)
    assert r["lag"] == 2 and r["mouth"] == pytest.approx(1.0, abs=1e-6)
    assert r["face"] == pytest.approx(sum(o is not None for o in out) / 120, abs=1e-3)
    # a lag of 3 is outside the search: not found
    out3 = [src[t + 3] if t + 3 < 120 else None for t in range(120)]
    r3 = ss.mouth_score(src, out3)
    assert r3["lag"] != 3 and r3["mouth"] < 0.5


def test_mouth_score_leaves_out_frames_by_a_cut_and_needs_enough_pairs():
    src = [0.1, 0.4] * 30
    out = list(src)
    r = ss.mouth_score(src, out, cuts=[30])
    assert r["pairs"] == 60 - 5
    few = [None] * 60
    few[:5] = src[:5]
    assert ss.mouth_score(src, few)["mouth"] is None


def test_mouth_halves():
    src = [0.1, 0.4] * 50
    out = src[:50] + [0.25] * 50
    r = ss.mouth_halves(src, out)
    assert r["halves"][0]["mouth"] == pytest.approx(1.0) and r["halves"][1]["mouth"] is None


def test_mouth_is_na_without_the_landmarker(monkeypatch):
    monkeypatch.setattr(ss, "landmarker_path", lambda: None)
    ok, why = ss.mouth_available()
    if ok:
        pytest.fail("mouth_available() must say no without the model")
    f = [np.zeros((54, 96, 3), np.uint8)] * 3
    s, _ = ss.score_frames(f, f, mouth=True)
    assert s["mouth"] is None and s["mouth_info"]["why"]


# ---------------------------------------------------------------- the scene alarm

def test_scene_psnr_outside_the_dilated_person():
    src = np.full((540, 960, 3), 100, np.uint8)
    out = src.copy()
    out[200:340, 400:560] = 0                         # the character: inside the mask, ignored
    out[0:10, :] = 110                                # the room: a 10-level change on 10 rows
    m = _box(540, 960, 200, 340, 400, 560)
    r = ss.scene([src], [out], [m])
    bg = ~(ss.cv2.dilate(m.astype(np.uint8), ss._KERN) > 0)
    mse = (10 * 960 * 3 * 100) / (bg.sum() * 3)
    assert r["bg_psnr"] == pytest.approx(10 * np.log10(255 ** 2 / mse), abs=0.01)
    assert ss.scene([src], [src], [m])["bg_psnr"] > 100


# ---------------------------------------------------------------- flags

def test_chunk_flags_bands():
    g, a = T["following"]["green"], T["following"]["amber"]
    p_g, p_a = T["following_p10"]["green"], T["following_p10"]["amber"]
    assert ss.chunk_flags({"pose_iou": g, "pose_iou_p10": p_g})["following"] == "green"
    assert ss.chunk_flags({"pose_iou": g - 0.001, "pose_iou_p10": 0.9})["following"] == "amber"
    assert ss.chunk_flags({"pose_iou": 0.9, "pose_iou_p10": p_a - 0.001})["following"] == "red"
    assert ss.chunk_flags({"pose_iou": a - 0.001, "pose_iou_p10": 0.9})["following"] == "red"
    assert ss.chunk_flags({})["following"] is None
    assert ss.chunk_flags({"cuts": {"96": "copied", "155": "unsure"}})["cuts"] == "amber"
    assert ss.chunk_flags({"cuts": {"96": "copied", "155": "lost"}})["cuts"] == "amber"     # never red (B3)
    assert ss.chunk_flags({"cuts": {"96": "copied"}})["cuts"] == "green"
    assert ss.chunk_flags({"cuts": {}})["cuts"] is None


def test_mouth_is_never_red():
    m = T["mouth"]
    assert ss.mouth_flag(m["green"]) == "green"
    assert ss.mouth_flag(m["amber"]) == "amber"
    assert ss.mouth_flag(-0.9) == "grey"
    assert ss.mouth_flag(None) is None


def test_scene_alarm_is_amber_only():
    assert ss.scene_flag({"bg_psnr": T["scene_db"] - 0.01}) == "amber"
    assert ss.scene_flag({"bg_psnr": T["scene_db"]}) == "green"


def test_join_verdict_is_the_worst_of_four():
    j = {"type": "forward", "stale": False}
    c, mo = T["join_colour"], T["join_motion"]
    good = {"frame_luma": {"at_splice": 0.2}, "char_luma": {"at_splice": 0.3}, "join_ratio": mo["green"] - 0.01,
            "follow": {"pose_iou": 0.8, "pose_iou_p10": 0.7}}
    assert ss.join_flags(j, good)["verdict"] == "green"
    assert ss.join_flags(j, dict(good, frame_luma={"at_splice": c["green"] + 0.01}))["verdict"] == "amber"
    assert ss.join_flags(j, dict(good, char_luma={"at_splice": 9.0}))["verdict"] == "green"   # shown, not judged
    assert ss.join_flags(j, dict(good, frame_luma={"at_splice": c["amber"] + 0.01}))["verdict"] == "red"
    assert ss.join_flags(j, dict(good, join_ratio=mo["amber"]))["motion"] == "red"
    assert ss.join_flags(j, dict(good, follow={"pose_iou": 0.4, "pose_iou_p10": 0.4}))["verdict"] == "red"
    st = ss.join_flags({"type": "stale", "stale": True}, good)
    assert st["lineage"] == "amber" and st["verdict"] == "amber"
    assert ss.join_flags({"type": "straight"}, good)["verdict"] is None
    assert ss.join_flags({"type": "pending"}, None)["verdict"] is None
    assert ss.join_flags(j, None)["verdict"] is None              # linked, nothing measured yet: no colour


def test_rank_is_following_then_lost_cuts_never_mouth():
    takes = [{"id": "a", "scores": {"pose_iou": 0.6, "cuts": {}, "mouth": 0.9}},
             {"id": "b", "scores": {"pose_iou": 0.7, "cuts": {"96": "lost"}, "mouth": 0.0}},
             {"id": "c", "scores": {"pose_iou": 0.7, "cuts": {"96": "copied"}, "mouth": -0.5}},
             {"id": "d", "scores": {}}]
    assert [t["id"] for t in sorted(takes, key=ss.rank_key)] == ["c", "b", "a", "d"]


def test_splice_jump_frame_and_character():
    a = np.full((540, 960, 3), 100, np.uint8)        # at the analysis size: no resampling at the box's edge
    b = a.copy()
    b[100:300, 250:500] = 120
    m = _box(540, 960, 100, 300, 250, 500)
    j = ss.splice_jump(a, b, m, m)
    assert j["char"] == pytest.approx(20.0, abs=0.01)
    assert j["frame"] == pytest.approx(20.0 * m.mean(), abs=0.05)


def test_score_frames_puts_it_together():
    n = 30
    rng = np.random.default_rng(5)
    src = [rng.integers(0, 255, (54, 96, 3), dtype=np.uint8) for _ in range(n)]
    out = [f.copy() for f in src]
    m = [_box(54, 96, 10, 40, 30, 60)] * n
    s, _ = ss.score_frames(out, src, m, m, cuts=[115], first=100, mouth=False)
    assert s["pose_iou"] == 1.0 and s["frames"] == n
    assert set(s["cuts"]) == {"115"} and s["scene"]["bg_psnr"] > 100
    assert "mouth" not in s
