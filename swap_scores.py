"""SeamStitch Swap: the quality scores and flags of takes and joins (design §4.5).

numpy + OpenCV (+ mediapipe for mouth sync, optional); no torch and no comfy imports. The
maths is a port, not a rewrite, of the measured CR test scripts, so a number here compares
with r1-r16's:
  * following: pose IoU, the SAM3 "person" mask of the source against the output's, per
    frame; mean and p10 (r10). Masks at their own size (the output's resized to the source's).
  * lost cuts: scorer.cut_stats on the output's frame differences (960x540, area, grey), per
    confirmed cut inside the take's render: copied / unsure / lost (T2, r5-r7).
  * mouth sync: mouth.py. MediaPipe FaceLandmarker, inner-lip gap (13-14) over mouth width
    (78-308); the correlation of source and output over the frames where both have a face,
    at the best lag within +-2. A secondary signal, never a gate (Kay). The face coverage is
    reported beside every number: a hand or a prop over the mouth makes it noisy (r15).
  * scene alarm: scorer's background PSNR, outside the 21 px dilated union of the two person
    masks, at 960x540 (T7). Only an alarm: never rank or pick on it (r10).
  * joins: colour (the whole-frame luma jump at the splice; the character's is shown beside it,
    not judged: see THRESHOLDS), motion (Result Preview's join_ratio), following (pose IoU
    over +-25 frames) and lineage (linked or stale). A join's verdict is the worst of the four.

Nothing here picks a take. `rank_key` orders a chunk's takes for display only: following
first, then lost cuts; mouth sync is shown, never ranked.

THRESHOLDS is the one table every caller reads (the Take node, the Planner's view, the UI
through the plan view). B3 fitted it against Kay's verdicts on REVIEW_1-8 and B1b (T-FLAGS,
plan doc "#### B3").
"""

import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# thresholds (§4.5, as fitted in B3)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    # pose IoU, chunk mean and p10; a join's +-25 frames use the mean bands only. B3 moved the mean's
    # green from 0.65 to 0.60: the 400-608 renders sit at 0.649-0.654 (X2, X3, X4, X5, X10), and 0.65 split
    # Kay's best render X10 (0.6496, amber) from its equals; the plain renders he ranked below marked 124
    # (mean 0.57-0.62) stay amber through their p10 (0.37-0.39).
    "following": {"green": 0.60, "amber": 0.50},
    "following_p10": {"green": 0.45, "amber": 0.30},
    # scorer.cut_stats: peak_ratio_at_offset and spread, per confirmed cut. Kept. A lost or unsure cut is
    # amber, not red (B3): Kay's favourites lose cuts too (marked 124 at 96; X10 at 413 / 429), so a red
    # dot on nearly every 100d chunk would not say what to re-roll.
    "cut_copied": 3.0, "cut_spread": 2, "cut_lost": 2.0,
    # mouth sync (information only: below amber it is shown grey, never red). Green moved 0.50 -> 0.60
    # (B3): CR-T1's plain 243, whose mouth Kay called "terrible", scores 0.51; marked 243 0.63; r5-r7's
    # good seeds 0.60-0.71.
    "mouth": {"green": 0.60, "amber": 0.25}, "mouth_lag": 2, "mouth_min_pairs": 8,
    # the scene alarm: mean background PSNR under this is amber. Kept at 15 (no REVIEW item came near it:
    # 15.7-22.8 dB).
    "scene_db": 15.0,
    # a join: the whole-frame luma jump at the splice. Bands kept. The character's jump is shown, not
    # judged (B3): read with the takes' own SAM3 masks it doesn't measure colour. r12 / r16's lock 12
    # reads 7.49 on the right take's track and 1.71 on two tracks, against 0.86 on one SAM3 track over the
    # joined file (r16's method), and the unrepaired join 1.59 against 4.98: the right take drifts off the
    # left by the splice, and two SAM3 runs segment the character differently (wings in or out). The
    # whole-frame jump alone agrees with Kay's verdicts (T-FLAGS).
    "join_colour": {"green": 1.0, "amber": 2.0},
    # a join: Result Preview's join_ratio. Moved (B3) from < 1.8 / < 3.0 to < 3.0 / < 5.0: every anchored
    # lock Kay chose reads 2.5-2.9 (r12 / r16 609: 2.88; B1b 609: 2.51), a "soft bump" on Result Preview's
    # scale, from the small content step on near-static frames that a level lock can't remove; nothing he
    # rejected was rejected for motion. A real source cut reads 5.4 (B1b, 250).
    "join_motion": {"green": 3.0, "amber": 5.0},
}

GREEN, AMBER, RED, GREY = "green", "amber", "red", "grey"
_RANK = {None: -1, GREY: -1, GREEN: 0, AMBER: 1, RED: 2}

AN_W, AN_H = 960, 540        # the CLI scripts' analysis size
DILATE = 21                  # scorer.DILATE: px at 960x540
FOLLOW_SIDE = 25             # a join's following: frames either side of the splice


def worst(*colours):
    """The worst of some colours (None and grey never win over a real colour)."""
    out = None
    for c in colours:
        if c is not None and _RANK[c] > _RANK[out]:
            out = c
    return out


def _band(v, t, higher_is_better=True):
    if v is None:
        return None
    v = float(v)
    if higher_is_better:
        return GREEN if v >= t["green"] else AMBER if v >= t["amber"] else RED
    return GREEN if v <= t["green"] else AMBER if v <= t["amber"] else RED


# ---------------------------------------------------------------------------
# frames at the analysis size
# ---------------------------------------------------------------------------

def thumb(rgb, w=AN_W, h=AN_H, interp=cv2.INTER_AREA):
    """A frame at the analysis size (ffmpeg scale flags=area in the CLI scripts)."""
    if rgb.shape[1] == w and rgb.shape[0] == h:
        return rgb
    return cv2.resize(rgb, (w, h), interpolation=interp)


def mask_thumb(mask, w=AN_W, h=AN_H):
    """A bool mask at the analysis size: an area resize of 0/255 and > 127, as the CLI read them."""
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = (m > 0.5).astype(np.uint8) * 255 if m.dtype != bool else m.astype(np.uint8) * 255
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    return m > 127


def to_bool(mask):
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype == bool:
        return m
    return m > (127 if m.dtype == np.uint8 else 0.5)


# ---------------------------------------------------------------------------
# following: pose IoU (r10)
# ---------------------------------------------------------------------------

def iou(a, b):
    """Intersection over union of two bool masks; b resized to a's size. None when both are empty."""
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    union = np.logical_or(a, b).sum()
    if not union:
        return None
    return float(np.logical_and(a, b).sum() / union)


def pose_iou(src_masks, out_masks):
    """Per-frame IoU of source and output person masks: {pose_iou (mean), pose_iou_p10, frames}, or None."""
    vals = []
    for a, b in zip(src_masks, out_masks):
        v = iou(to_bool(a), to_bool(b))
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return {"pose_iou": round(float(np.mean(vals)), 4), "pose_iou_p10": round(float(np.percentile(vals, 10)), 4),
            "frames": len(vals)}


# ---------------------------------------------------------------------------
# lost cuts (scorer.cut_stats)
# ---------------------------------------------------------------------------

def frame_diffs(thumbs):
    """d[t] = mean |grey[t] - grey[t-1]| at the analysis size (d[0] = 0), scorer's d_out."""
    d, prev = [], None
    for f in thumbs:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
        d.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    return np.array(d)


def cut_stats(d, c):
    """scorer.cut_stats, unchanged: peak_ratio, spread, offset, peak_ratio_at_offset of the cut at index c."""
    lo, hi = max(1, c - 6), min(len(d), c + 7)
    neigh = [d[t] for t in range(lo, hi) if t != c]
    w = d[max(1, c - 5): c + 6]
    near = d[max(1, c - 3): c + 4]
    off = int(np.argmax(near)) + max(1, c - 3) - c
    p = c + off
    neigh_p = [d[t] for t in range(max(1, p - 6), min(len(d), p + 7)) if t != p]
    return {"peak_ratio": round(float(d[c] / (np.median(neigh) + 1e-6)), 2),
            "spread": int((w > 0.5 * w.max()).sum()),
            "offset": off,
            "peak_ratio_at_offset": round(float(d[p] / (np.median(neigh_p) + 1e-6)), 2)}


def cut_state(st):
    """copied (>= 3.0 with spread <= 2), lost (< 2.0), else unsure."""
    t = THRESHOLDS
    r = st["peak_ratio_at_offset"]
    if r >= t["cut_copied"] and st["spread"] <= t["cut_spread"]:
        return "copied"
    if r < t["cut_lost"]:
        return "lost"
    return "unsure"


def lost_cuts(d, cuts, first=0):
    """Every cut (source frame numbers) strictly inside the frames d covers (d[i] = source frame first + i),
    with at least one frame either side: {str(cut): {state, peak_ratio_at_offset, spread, offset}}."""
    out = {}
    for c in sorted(set(int(x) for x in cuts)):
        i = c - first
        if 1 <= i < len(d) - 1:
            st = cut_stats(d, i)
            out[str(c)] = dict(state=cut_state(st), **st)
    return out


# ---------------------------------------------------------------------------
# mouth sync (mouth.py)
# ---------------------------------------------------------------------------

LANDMARKER = "face_landmarker.task"


def landmarker_path():
    """models/mediapipe/face_landmarker.task (placed there with Kay's OK, 2026-10-03), else None."""
    try:
        import folder_paths
        p = os.path.join(folder_paths.models_dir, "mediapipe", LANDMARKER)
    except Exception:
        p = None
    return p if p and os.path.isfile(p) else None


def mouth_available():
    """(True, path) or (False, why)."""
    try:
        import mediapipe  # noqa: F401
    except Exception as e:
        return False, f"mediapipe isn't installed in ComfyUI's Python ({e})"
    p = landmarker_path()
    if not p:
        return False, f"no {LANDMARKER} under models/mediapipe/"
    return True, p


def mouth_series(frames, model_path, fps=25.0):
    """Mouth opening per frame (inner-lip gap / mouth width), None where no face is found.
    frames: RGB uint8 at any size; analysed at 960x540 (mouth.py's ffmpeg scale, bicubic)."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    out = []
    step = 1000.0 / float(fps or 25.0)
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for t, f in enumerate(frames):
            f = thumb(f, interp=cv2.INTER_CUBIC)
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)),
                                    int(round(t * step)))
            if not r.face_landmarks:
                out.append(None)
                continue
            p = r.face_landmarks[0]
            gap = np.hypot((p[13].x - p[14].x) * AN_W, (p[13].y - p[14].y) * AN_H)
            wid = np.hypot((p[78].x - p[308].x) * AN_W, (p[78].y - p[308].y) * AN_H)
            out.append(float(gap / wid) if wid > 1 else None)
    return out


def _corr(a, b):
    if len(a) < THRESHOLDS["mouth_min_pairs"] or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def mouth_score(src, out, cuts=(), lags=None):
    """mouth.py's score, the lag search narrowed to +-mouth_lag: frames within 2 of a cut (indices
    into the series) are left out. {mouth (best corr), lag, corr_lag0, face (the share of frames
    with a face in both), pairs, detect_src, detect_out}; mouth None when there are too few pairs."""
    lags = THRESHOLDS["mouth_lag"] if lags is None else int(lags)
    n = min(len(src), len(out))
    if n == 0:
        return {"mouth": None, "lag": None, "corr_lag0": None, "face": 0.0, "pairs": 0, "detect_src": 0.0,
                "detect_out": 0.0}
    bad = {c + d for c in cuts for d in range(-2, 3)}
    by_lag = {}
    pairs0 = 0
    for lag in range(-lags, lags + 1):
        ts = [t for t in range(n) if 0 <= t + lag < n and t not in bad and out[t] is not None
              and src[t + lag] is not None]
        if lag == 0:
            pairs0 = len(ts)
        by_lag[lag] = _corr(np.array([out[t] for t in ts]), np.array([src[t + lag] for t in ts]))
    best = max((k for k in by_lag if by_lag[k] is not None), key=lambda k: by_lag[k], default=None)
    r = lambda v: None if v is None else round(float(v), 3)  # noqa: E731
    return {"mouth": r(by_lag.get(best)) if best is not None else None, "lag": best, "corr_lag0": r(by_lag.get(0)),
            "face": round(pairs0 / float(n), 3), "pairs": pairs0,
            "detect_src": round(sum(v is not None for v in src[:n]) / float(n), 3),
            "detect_out": round(sum(v is not None for v in out[:n]) / float(n), 3)}


def mouth_halves(src, out, cuts=()):
    """The whole series, plus its first and second halves on their own (X10: 0.90 in the first half,
    weak in the second, r15)."""
    n = min(len(src), len(out))
    h = n // 2
    res = mouth_score(src, out, cuts)
    res["halves"] = [mouth_score(src[:h], out[:h], cuts),
                     mouth_score(src[h:n], out[h:n], [c - h for c in cuts])]
    return res


# ---------------------------------------------------------------------------
# the scene alarm (scorer's background PSNR)
# ---------------------------------------------------------------------------

_KERN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1))


def bg_psnr_frame(src_t, out_t, src_m, out_m=None):
    """One frame's background PSNR, scorer's way: analysis-size RGB, outside the dilated union."""
    m = src_m.copy()
    if out_m is not None:
        m |= out_m
    m = cv2.dilate(m.astype(np.uint8), _KERN) > 0
    bg = ~m
    if not bg.any():
        return None
    diff = np.abs(src_t.astype(np.float32) - out_t.astype(np.float32))[bg]
    mse = float((diff ** 2).mean())
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-6)))


def scene(src_thumbs, out_thumbs, src_masks, out_masks=None):
    """{bg_psnr (mean), p5, worst, worst_frame (index)}, scorer's bg_psnr; masks at the analysis size."""
    vals, idx = [], []
    oms = out_masks if out_masks is not None else [None] * len(src_masks)
    for i, (s, o, sm, om) in enumerate(zip(src_thumbs, out_thumbs, src_masks, oms)):
        v = bg_psnr_frame(s, o, sm, om)
        if v is not None:
            vals.append(v)
            idx.append(i)
    if not vals:
        return None
    k = int(np.argmin(vals))
    return {"bg_psnr": round(float(np.mean(vals)), 2), "p5": round(float(np.percentile(vals, 5)), 2),
            "worst": round(float(vals[k]), 2), "worst_frame": idx[k]}


# ---------------------------------------------------------------------------
# a whole take
# ---------------------------------------------------------------------------

def score_frames(out_frames, src_frames=None, src_masks=None, out_masks=None, cuts=(), first=0, fps=25.0,
                 mouth=True, mouth_src=None, model_path=None):
    """Every chunk score of one take, from its delivered frames.

    out_frames / src_frames: RGB uint8 iterables, frame i = source frame first + i (the source's
    are needed for mouth sync and the scene alarm); masks: bool (or 0..1 / 0..255) arrays, same
    indexing, any size. cuts: confirmed source cuts. mouth_src: a cached source mouth series
    (it is computed from src_frames otherwise). Returns (scores, extras): extras carries the
    source mouth series, for a cache."""
    outs = [thumb(np.asarray(f)) for f in out_frames]
    n = len(outs)
    srcs = None if src_frames is None else [thumb(np.asarray(f)) for f in src_frames]
    scores, extras = {}, {}
    if src_masks is not None and out_masks is not None:
        pi = pose_iou(src_masks, out_masks)
        if pi:
            scores.update(pi)
    d = frame_diffs(outs)
    cs = lost_cuts(d, cuts, first)
    scores["cuts"] = {k: v["state"] for k, v in cs.items()}
    scores["cut_stats"] = cs
    if srcs is not None and src_masks is not None:
        sm = [mask_thumb(m) for m in src_masks]
        om = [mask_thumb(m) for m in out_masks] if out_masks is not None else None
        sc = scene(srcs, outs, sm, om)
        if sc:
            scores["scene"] = sc
    if mouth:
        ok, why = (True, model_path) if model_path else mouth_available()
        if not ok:
            scores["mouth"] = None
            scores["mouth_info"] = {"why": why}
        else:
            ms = mouth_src
            if ms is None and srcs is not None:
                ms = mouth_series(srcs, why, fps)
            if ms is None:
                scores["mouth"] = None
                scores["mouth_info"] = {"why": "no source frames"}
            else:
                mo = mouth_series(outs, why, fps)
                rel = [c - first for c in cuts if 0 <= c - first < n]
                m = mouth_halves(ms[:n], mo, rel)
                scores["mouth"] = m.pop("mouth")
                scores["mouth_info"] = m
                extras["mouth_src"] = ms
    return scores, extras


# ---------------------------------------------------------------------------
# flags and verdicts
# ---------------------------------------------------------------------------

def following_flag(pi):
    """The worst of the mean and p10 bands; None without masks."""
    if not pi or pi.get("pose_iou") is None:
        return None
    return worst(_band(pi["pose_iou"], THRESHOLDS["following"]),
                 _band(pi.get("pose_iou_p10"), THRESHOLDS["following_p10"]))


def cuts_flag(cuts):
    """green when every confirmed cut inside was copied; amber when one was lost or unsure (B3: never red,
    see THRESHOLDS); None with no cut inside."""
    if not cuts:
        return None
    v = list(cuts.values())
    return GREEN if all(x == "copied" for x in v) else AMBER


def mouth_flag(v):
    """green / amber / grey: never red, never a gate."""
    if v is None:
        return None
    t = THRESHOLDS["mouth"]
    return GREEN if v >= t["green"] else AMBER if v >= t["amber"] else GREY


def scene_flag(sc):
    if not sc or sc.get("bg_psnr") is None:
        return None
    return AMBER if sc["bg_psnr"] < THRESHOLDS["scene_db"] else GREEN


def chunk_flags(scores):
    """{following, cuts, mouth, scene}: colour per flag (None = not measured). Information for
    Kay; nothing reads them to choose."""
    s = scores or {}
    return {"following": following_flag(s), "cuts": cuts_flag(s.get("cuts")),
            "mouth": mouth_flag(s.get("mouth")), "scene": scene_flag(s.get("scene"))}


def join_flags(j, m):
    """A join's four flags and its verdict (the worst of them). j: swap_plan.join_info (+ type /
    stale); m: its measurement (frame_luma, char_luma, join_ratio, follow), or None. Straight and
    pending joins have no verdict."""
    if j.get("type") in ("straight", "pending"):
        return {"colour": None, "motion": None, "following": None, "lineage": None, "verdict": None}
    lineage = AMBER if j.get("stale") or j.get("type") == "stale" else GREEN
    colour = motion = follow = None
    if m:
        v = (m.get("frame_luma") or {}).get("at_splice")      # the character's: shown, not judged (THRESHOLDS)
        colour = None if v is None else _band(abs(float(v)), THRESHOLDS["join_colour"], higher_is_better=False)
        r = m.get("join_ratio")
        if r is not None:      # Result Preview's bands: < green, < amber, else red
            t = THRESHOLDS["join_motion"]
            motion = GREEN if r < t["green"] else AMBER if r < t["amber"] else RED
        fo = m.get("follow") or {}
        follow = _band(fo.get("pose_iou"), THRESHOLDS["following"])     # the mean only: p10 of 50 frames is 5
    # nothing measured yet: only a stale lineage says anything (a linked join isn't green unmeasured)
    verdict = worst(colour, motion, follow, lineage) if any(x is not None for x in (colour, motion, follow))         else (AMBER if lineage == AMBER else None)
    return {"colour": colour, "motion": motion, "following": follow, "lineage": lineage, "verdict": verdict}


def join_verdict(j, m):
    return join_flags(j, m)["verdict"]


def rank_key(take):
    """Display-only order of a chunk's takes: following (pose IoU, higher first), then lost cuts
    (fewer first). Mouth sync is shown, never ranked. Nothing picks a take from this."""
    s = take.get("scores") or {}
    pi = s.get("pose_iou")
    lost = sum(1 for v in (s.get("cuts") or {}).values() if v == "lost")
    return (-(pi if pi is not None else -1.0), lost)


def splice_jump(prev, cur, mask_prev=None, mask_cur=None):
    """The luma jump INTO a splice frame, r16's way: both frames at 960x540 (area), OpenCV grey,
    whole frame and, with masks, the character (each frame under its own mask)."""
    gp = cv2.cvtColor(thumb(prev), cv2.COLOR_RGB2GRAY).astype(np.float64)
    gc = cv2.cvtColor(thumb(cur), cv2.COLOR_RGB2GRAY).astype(np.float64)
    out = {"frame": round(float(abs(gc.mean() - gp.mean())), 3)}
    if mask_prev is not None and mask_cur is not None:
        mp_, mc = mask_thumb(mask_prev), mask_thumb(mask_cur)
        if mp_.any() and mc.any():
            out["char"] = round(float(abs(gc[mc].mean() - gp[mp_].mean())), 3)
    return out


def legend():
    """The thresholds and colour rules, for the UI (through the plan view)."""
    return {"thresholds": THRESHOLDS,
            "rules": {"following": "pose IoU mean >= 0.60 green, 0.50-0.60 amber, < 0.50 red; p10 >= 0.45 / 0.30; "
                                   "the worse of the two",
                      "cuts": "per confirmed cut: copied >= 3.0 with spread <= 2; lost < 2.0; else unsure. "
                              "Amber when any is lost or unsure",
                      "mouth": "best lag within +-2, frames with a face: >= 0.60 green, 0.25-0.60 amber, below grey "
                               "(information only)",
                      "scene": "background PSNR outside the person union: amber under 15 dB (an alarm, never ranked)",
                      "join": "the worst of colour (the whole-frame jump at the splice: <= 1.0 / 2.0; the character's "
                              "is shown, not judged), motion "
                              "(join ratio < 3.0 / 5.0), following (pose IoU mean +-25 frames) and lineage (stale = amber)"}}
