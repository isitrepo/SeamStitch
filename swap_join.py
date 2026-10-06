"""SeamStitch Swap: the join repairs and the join measurements, on decoded frames.

numpy + OpenCV, no torch and no comfy imports. The maths is a port, not a rewrite, of the
measured CR test scripts, so a number here compares with r11-r16's:
  * the level lock (`lock_gains`) and the level-matched fade: the measured seam_repair script,
    itself a port of obvpm-timeline's levellock.py and crossfade.py (GPL-3.0, credit chanon);
  * the luma jumps: seam_repair.py / bridge_join.py;
  * the seam score: handover.py / bridge_join.py `join`, on seamweave.flow's motion measures
    (ported below, the same functions);
  * the motion verdict: result_preview.join_ratio, imported by the Assemble node.

Frames are uint8 HxWx3 RGB. Every repair does its arithmetic in float32 and truncates back
to uint8 exactly as seam_repair.py does, so the node's frames equal the CLI's bit for bit
on the same decoded input.

LOCK (default for forward, re-roll exit and stale joins). Per RGB channel:
  1. predict where the left take was heading: a line through its last FIT frames' means,
     extrapolated one frame past its end;
  2. fit the right take's first `frames` (the hand-back) frames' means to a line;
  3. delta = prediction - that line's start, ramped to 0 by frame `frames`;
  4. gain = (line + delta) / line, multiplicative, identity from frame `frames` on.

FADE (default for a re-roll entry). Over the right take's overlap, each right frame is
level-matched to the left one (gain = mean left / mean right, per frame and channel) and
crossfaded in with equal-power weights (sin^2). After the overlap the last match gain decays
to 1 over `frames` frames.

Two refinements from B1b's renders (2026-10-03), on by default:
  * CUT-AWARE: the lock's fits never cross a cut. The right take's opening (fit and hand-back)
    ends before the first cut after the splice: a confirmed source cut, or the take's own luma
    step >= CUT_STEP (H3 often renders a cut a frame or more late). The left take's heading is
    fitted only on its frames after its last cut. (412 in B1b: a cut one frame after the split,
    rendered at 414, dragged the fit and the lock landed 5.9 off.)
  * REGIONAL: with both takes' SAM3 person masks, the character and the background each get
    their own lock, blended through a feathered mask. A single gain trades one region's step
    for the other's when only one of them differs (609 in B1b: the characters matched, the
    background sat 2.7 under, and the frame-level gain stepped the character +3 to +4). On a
    uniform offset the two gains agree and it equals the global lock. This is obvpm's "local"
    option in its simplest, two-region form.

The HEADING (B5b, 2026-10-05): where the left take is going, the level the right take's opening is
locked to at the splice. As ported, a line through the left take's last 12 frames, one frame on; a
render's own flicker bends that line (the test clip at 406: c3 dips then brightens 2.5 levels over its last 7
frames, the line lands ~1 level under its last frame, and the lock stepped the colour down: visible by eye).
Now, by default, the left take's last 3 frames, each carried by the source's own change to the splice,
averaged: mean over i = 1..3 of L[J-i] * S[J] / S[J-i]. On the 7 lock joins on disk (frame means) the step
error (max over R, G, B, against the source's step) fell from mean 0.70 / worst 1.63 to 0.36 / 0.66 (the
last frame alone: 0.23 / 0.65); through the regional lock on the test clip's 209 / 406 / 603 the frame step against
the source's is 0.56 / 0.80 / 0.35 (as ported 0.54 / 1.40 / 0.45; the last frame alone put 603's room
step at 1.31). heading="line" keeps the port.

obvpm's lock extras, all OFF by default (T-JOIN measures them):
  gate   leave the join alone unless its step beats max(0.6, 3 x the left tail's local noise)
  clamp  clip gains to [1/1.06, 1.06] (MAX_GAIN)
  swing  straighten an oscillating opening onto its fitted line (divide by the actual
         per-frame means instead of the line) when its swing beats max(1.5, 3 x local noise)
"""

import threading

import cv2
import numpy as np

FIT = 12            # levellock.FIT: left frames used to predict the heading
HAND_BACK = 12      # levellock.FRAMES: the default hand-back

# obvpm-timeline levellock constants (the T-JOIN options)
STEP_OVER_NOISE, STEP_FLOOR = 3.0, 0.6
SWING_OVER_NOISE, SWING_FLOOR = 3.0, 1.5
MAX_GAIN = 1.06

LUMA_BT709 = np.array([0.2126, 0.7152, 0.0722])
CUT_STEP = 6.0      # a frame-to-frame mean-luma step this big inside a take's opening is a cut
HEADING_SOURCE, HEADING_LINE = "source", "line"
DEFAULT_HEADING = HEADING_SOURCE
HEADING_K = 3       # the left take's last k frames, each carried by the source's change since, averaged
FEATHER = 0.01      # the regional lock's mask feather: Gaussian sigma as a fraction of the width


# ---------------------------------------------------------------------------
# the repairs (seam_repair.py)
# ---------------------------------------------------------------------------

def frame_means(frame):
    """Per-channel mean of one frame, float64 (seam_repair.means, one row)."""
    return frame.reshape(-1, 3).mean(0)


def means(frames):
    return np.array([frame_means(f) for f in frames], np.float64).reshape(-1, 3)


def line_fit(y):
    """Least-squares line through y[0..n-1] per channel; (intercept at 0, slope). One point: flat."""
    y = np.asarray(y, np.float64)
    if len(y) == 1:
        return y[0], np.zeros_like(y[0])
    t = np.arange(len(y))
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[0], coef[1]


def lock_report(left_means, right_means, frames=HAND_BACK):
    """obvpm's measure() on channel means: step, local noise, swing (all in luma levels)."""
    lt = np.asarray(left_means, np.float64)[-FIT:] @ LUMA_BT709
    rh = np.asarray(right_means, np.float64)[:frames] @ LUMA_BT709
    s1, i1 = np.polyfit(np.arange(float(len(lt))), lt, 1)
    pred = float(s1 * len(lt) + i1)
    local = float(np.abs(np.diff(lt)).mean()) if len(lt) > 1 else 0.0
    s2, i2 = np.polyfit(np.arange(float(len(rh))), rh, 1)
    resid = rh - (s2 * np.arange(float(len(rh))) + i2)
    return {"step": float(rh[:3].mean() - pred), "local": local, "swing": float(resid.max() - resid.min()),
            "step_bar": max(STEP_FLOOR, STEP_OVER_NOISE * local), "swing_bar": max(SWING_FLOOR, SWING_OVER_NOISE * local)}


def source_heading(left_means, source_ratio):
    """The left take's last frames carried by the source's own change to the splice, averaged: the default
    heading (B5b). source_ratio: (k, 3) of S[J] / S[J-i] for i = k..1 (source_ratios), or one (3,) ratio
    S[J] / S[J-1] for the last frame alone."""
    lm = np.asarray(left_means, np.float64).reshape(-1, 3)
    r = np.asarray(source_ratio, np.float64).reshape(-1, 3)
    k = min(len(r), len(lm))
    return (lm[-k:] * r[-k:]).mean(0)


def source_ratio(src_before, src_at):
    """S[J] / S[J-1] per channel from two source frames' channel means."""
    return np.asarray(src_at, np.float64) / np.maximum(np.asarray(src_before, np.float64), 1e-3)


def source_ratios(src_means):
    """S[J] / S[J-i] for i = k..1, from the source's channel means over [J-k, J] (k + 1 frames)."""
    sm = np.asarray(src_means, np.float64).reshape(-1, 3)
    return sm[-1] / np.maximum(sm[:-1], 1e-3)


def lock_gains(left_means, right_means, frames=HAND_BACK, gate=False, clamp=False, swing=False, heading=None):
    """Per-frame, per-channel gains (n, 3) for the right take's opening `frames` frames.

    left_means: channel means of the left take's frames up to the splice (its last FIT are
    used); right_means: of the right take's frames from the splice (its first `frames`).
    heading: the level (3,) to lock to at the splice (source_heading); None = the port's line
    through the left take's last FIT frames. With heading None and every option off this is
    seam_repair.lock_gains exactly. Returns all-ones when the gate says the join has nothing to fix."""
    mx = np.asarray(left_means, np.float64)[-FIT:]
    b, s = line_fit(mx)
    pred = b + s * len(mx) if heading is None else np.asarray(heading, np.float64)   # the left take's heading
    my = np.asarray(right_means, np.float64)[:frames]
    yb, ys = line_fit(my)
    t = np.arange(len(my))[:, None]
    yline = yb + ys * t
    delta = (pred - yb) * (1 - t / frames)       # ramp to 0 by frame `frames`
    rep = lock_report(left_means, right_means, frames) if (gate or swing) else None
    do_step = not gate or abs(rep["step"]) > rep["step_bar"]
    do_swing = swing and rep["swing"] > rep["swing_bar"]
    if not do_step and not do_swing:
        return np.ones_like(yline)
    desired = yline + (delta if do_step else 0.0)
    actual = my if do_swing else yline          # swing: straighten the opening onto its line
    g = desired / np.maximum(actual, 1e-3)
    if clamp:
        g = np.clip(g, 1.0 / MAX_GAIN, MAX_GAIN)
    return g


def _luma_steps(m):
    lum = np.asarray(m, np.float64).reshape(-1, 3) @ LUMA_BT709
    return np.abs(np.diff(lum))


def opening_frames(right_means, splice, frames, cuts=()):
    """How many of the right take's opening frames (from the splice) the lock may fit and hand
    back over: up to `frames`, ending before the first confirmed cut after the splice or the
    take's own first luma step >= CUT_STEP. 0 when the splice itself is a cut."""
    n = min(int(frames), len(right_means))
    for c in cuts:
        if splice <= c < splice + n:
            n = c - splice
    steps = _luma_steps(right_means[:n])
    hit = np.nonzero(steps >= CUT_STEP)[0]
    if len(hit):
        n = int(hit[0]) + 1
    return n


def heading_frames(left_means, splice, cuts=()):
    """How many of the left take's last frames (up to FIT, ending at the splice) the heading is
    fitted on: only those after its last cut (confirmed, or its own luma step >= CUT_STEP)."""
    n = min(FIT, len(left_means))
    for c in cuts:
        if splice - n < c <= splice - 1:
            n = splice - c
    steps = _luma_steps(np.asarray(left_means)[-n:])
    hit = np.nonzero(steps >= CUT_STEP)[0]
    if len(hit):
        n = n - (int(hit[-1]) + 1)
    return max(1, n)


def region_means(frames, masks):
    """(character means, background means), each (n, 3), under bool masks (resized to the
    frames as needed). NaN rows where a region is empty."""
    ch, bg = [], []
    for f, m in zip(frames, masks):
        if m.shape != f.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        px = f.reshape(-1, 3)
        mm = m.reshape(-1)
        ch.append(px[mm].mean(0) if mm.any() else np.full(3, np.nan))
        bg.append(px[~mm].mean(0) if (~mm).any() else np.full(3, np.nan))
    return np.array(ch, np.float64).reshape(-1, 3), np.array(bg, np.float64).reshape(-1, 3)


def soft_mask(mask, size, feather=FEATHER):
    """A bool mask as float32 weights at size (w, h), feathered (Gaussian, sigma feather x w)."""
    w, h = size
    m = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    sig = max(1.0, feather * w)
    return np.clip(cv2.GaussianBlur(m, (0, 0), sig), 0.0, 1.0)


def solve_field_gains(frame, mask, weight, target_char, target_bg):
    """The two gains (character, background; per channel) that put each region's mean exactly on
    its target once blended through the feathered weight: per channel, for region R,
    sum_R f (w gc + (1 - w) gb) = target_R |R|, a 2x2 system. Falls back to the plain ratios
    when it is singular."""
    m = mask if mask.shape == frame.shape[:2] else cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                                                              interpolation=cv2.INTER_NEAREST) > 0
    f = frame.reshape(-1, 3).astype(np.float64)
    wv = weight.reshape(-1, 1).astype(np.float64)
    mm = m.reshape(-1)
    gc, gb = np.ones(3), np.ones(3)
    for ch in range(3):
        fc = f[:, ch]
        a1, b1 = (fc[mm] * wv[mm, 0]).sum(), (fc[mm] * (1 - wv[mm, 0])).sum()
        a2, b2 = (fc[~mm] * wv[~mm, 0]).sum(), (fc[~mm] * (1 - wv[~mm, 0])).sum()
        t1, t2 = target_char[ch] * mm.sum(), target_bg[ch] * (~mm).sum()
        det = a1 * b2 - a2 * b1
        if abs(det) > 1e-9 * max(1.0, abs(a1 * b2)):
            gc[ch], gb[ch] = (t1 * b2 - t2 * b1) / det, (a1 * t2 - a2 * t1) / det
        else:
            gc[ch] = t1 / max(fc[mm].sum(), 1e-6)
            gb[ch] = t2 / max(fc[~mm].sum(), 1e-6)
    return gc, gb


def apply_gain_field(frame, g_char, g_bg, weight):
    """apply_gain with a per-pixel gain: weight x the character's + (1 - weight) x the background's."""
    wgt = weight[..., None]
    g = wgt * np.asarray(g_char, np.float32) + (1.0 - wgt) * np.asarray(g_bg, np.float32)
    return np.clip(frame.astype(np.float32) * g, 0, 255).astype(np.uint8)


def apply_gain(frame, g):
    """seam_repair.apply_gain: float32 multiply, clip, truncate to uint8."""
    return np.clip(frame.astype(np.float32) * np.asarray(g).astype(np.float32), 0, 255).astype(np.uint8)


def fade_weights(n):
    """Equal-power weights for the incoming side, 0 -> 1 over n frames (sin^2)."""
    return np.sin(np.linspace(0, 1, n) * np.pi / 2) ** 2


def fade_ratios(left_means, right_means):
    """Per-frame, per-channel level match over the overlap: mean left / mean right."""
    gx, gy = np.asarray(left_means, np.float64), np.asarray(right_means, np.float64)
    return gx / np.maximum(gy, 1e-3)


def fade_frame(left, right, ratio, w):
    """One overlap frame: the right take level-matched by `ratio`, crossfaded in at weight w."""
    ym = right.astype(np.float32) * np.asarray(ratio).astype(np.float32)
    return np.clip((1 - w) * left.astype(np.float32) + w * ym, 0, 255).astype(np.uint8)


def fade_frame_out(left, right, ratio, w):
    """One frame of a fade OUT into the original (a kept chunk on the right, §4.10): the left (the
    render) level-matched to the right by `ratio`, the right (the source) faded in untouched."""
    lm = left.astype(np.float32) * np.asarray(ratio).astype(np.float32)
    return np.clip((1 - w) * lm + w * right.astype(np.float32), 0, 255).astype(np.uint8)


def pre_gain(first_ratio, k, frames=HAND_BACK):
    """Gain for the render's frame k frames BEFORE a fade out into the original (k = 1 is the frame
    just before it): ramping from 1 toward the fade's first match ratio, the mirror of decay_gain.
    None outside the ramp."""
    if k < 1 or k >= frames:
        return None
    f = 1 - k / frames
    return 1 + (np.asarray(first_ratio) - 1) * f


def decay_gain(last_ratio, k, frames=HAND_BACK):
    """Gain for the k-th right frame after the overlap (k = 0 is the splice frame): the last
    match ratio decaying to 1 by frame `frames`. None once it is identity."""
    f = max(0.0, 1 - (k + 1) / frames)
    return 1 + (np.asarray(last_ratio) - 1) * f if f > 0 else None


def tone_compensate(left_tail, right, mode, overlap):
    """MiniMax H3 Tone Compensate (ComfyUI-MiniMaxH3-ToneCompensate) on uint8 frames, for
    T-TONE only: `left_tail` = the source (its last `overlap` frames are used), `right` = the
    whole right take from its overlap start. frame_shift: per-frame per-channel mean shift on
    the overlap, the last one held over the rest; gain_bias: one per-channel affine fit.
    Returns float32 0..1 frames, as the node does (no uint8 truncation)."""
    src = np.stack(left_tail).astype(np.float32) / 255.0
    tgt = np.stack(right).astype(np.float32) / 255.0
    n = min(int(overlap), src.shape[0], tgt.shape[0])
    fit_src, fit_tgt = src[-n:], tgt[:n]
    if mode == "frame_shift":
        drift = fit_tgt.mean(axis=(1, 2), keepdims=True) - fit_src.mean(axis=(1, 2), keepdims=True)
        out = tgt.copy()
        out[:n] -= drift
        out[n:] -= drift[-1]
    elif mode == "gain_bias":
        out = np.empty_like(tgt)
        for c in range(3):
            s = fit_src[..., c].reshape(-1).astype(np.float64)
            g = fit_tgt[..., c].reshape(-1).astype(np.float64)
            dg = g - g.mean()
            den = (dg * dg).sum()
            A = 1.0 if den < 1e-12 else float((dg * (s - s.mean())).sum() / den)
            if abs(A) < 1e-6:
                A = 1.0
            C = float(s.mean() - A * g.mean())
            out[..., c] = A * tgt[..., c] + C
    else:
        raise ValueError(f"tone_compensate mode {mode!r}")
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# measurement (seam_repair.py / bridge_join.py luma tracks)
# ---------------------------------------------------------------------------

def luma(frame, mask=None):
    """Mean luma (OpenCV's RGB->GRAY, as the CLI scripts) of a frame, or of the masked part."""
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if mask is not None:
        if mask.shape != g.shape:
            mask = cv2.resize(mask.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if not mask.any():
            return float("nan")
        return float(g[mask].mean())
    return float(g.mean())


def iou(a, b):
    """Intersection over union of two bool masks (r10's pose IoU, one frame); b is resized to a's
    size when they differ. None when both are empty."""
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    union = np.logical_or(a, b).sum()
    if not union:
        return None
    return float(np.logical_and(a, b).sum() / union)


def measure_window(right_render_start, splice, hand_back=HAND_BACK):
    """seam_repair.py's measuring window around a join, in source frames [lo, hi): from 15
    before the right take's first frame to max(30, hand_back + 20) after the splice."""
    return right_render_start - 15, splice + max(30, hand_back + 20)


def jumps(track, first, splice, lo, hi):
    """Luma-jump stats of a track (track[i] = frame first + i) around a splice:
    at_splice = |t[splice] - t[splice - 1]|; max over [lo, hi) and where (the frame jumped
    INTO); median of |diff| over the whole track (the natural median). A frame without the
    character (an empty mask: NaN) on either side of the splice leaves at_splice None."""
    t = np.asarray(track, np.float64)
    lo_i, hi_i = max(0, lo - first), min(len(t), hi - first)
    d = np.abs(np.diff(t[lo_i:hi_i]))
    s = splice - first
    step = abs(t[s] - t[s - 1]) if 0 < s < len(t) else np.nan
    out = {"at_splice": round(float(step), 3) if np.isfinite(step) else None,
           "max": round(float(np.nanmax(d)), 3) if d.size else None,
           "max_at": int(first + lo_i + 1 + int(np.nanargmax(d))) if d.size else None,
           "median": round(float(np.nanmedian(np.abs(np.diff(t)))), 3) if len(t) > 1 else None}
    return out


# ---------------------------------------------------------------------------
# the seam score (handover.py / bridge_join.py `join`, on seamweave.flow)
# ---------------------------------------------------------------------------

SEAM_THUMB_W, SEAM_THUMB_H = 960, 540   # the CLI scripts' analysis size (ffmpeg scale, flags=area)
FLOW_W = 480                            # seamweave flow width


def gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def resize_to_width(img, width):
    if img.shape[1] == width:
        return img
    h = max(2, int(round(img.shape[0] * width / img.shape[1])))
    interp = cv2.INTER_AREA if width < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (width, h), interpolation=interp)


def thumb(rgb):
    """A frame at the CLI's analysis size (960x540, area)."""
    if rgb.shape[1] == SEAM_THUMB_W and rgb.shape[0] == SEAM_THUMB_H:
        return rgb
    return cv2.resize(rgb, (SEAM_THUMB_W, SEAM_THUMB_H), interpolation=cv2.INTER_AREA)


_LOCAL = threading.local()


def _dis():
    dis = getattr(_LOCAL, "dis", None)
    if dis is None:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setUseSpatialPropagation(True)
        _LOCAL.dis = dis
    return dis


def prep(g):
    """High-pass a grey frame before flow (two generations differ in level, not in edges)."""
    f = g.astype(np.float32)
    hp = f - cv2.GaussianBlur(f, (0, 0), 8.0)
    return np.clip(hp * 1.6 + 128.0, 0, 255).astype(np.uint8)


def _moving_pixels(pa, pb):
    diff = cv2.GaussianBlur(np.abs(pa.astype(np.float32) - pb.astype(np.float32)), (0, 0), 1.5)
    energy = cv2.GaussianBlur(np.abs(cv2.Laplacian(pa, cv2.CV_32F)), (0, 0), 1.5)
    score = diff * np.sqrt(energy + 1e-3)
    thr = float(np.percentile(score, 98.0))
    if thr <= 1e-3:
        return None
    return score >= thr


def _speed_and_vector(f, pa, pb):
    sel = _moving_pixels(pa, pb)
    if sel is None:
        return 0.0, np.array([0.0, 0.0])
    sf = cv2.GaussianBlur(f, (0, 0), 2.0)
    mag = np.sqrt(sf[..., 0] ** 2 + sf[..., 1] ** 2)
    return float(np.median(mag[sel])), np.array([float(np.median(sf[..., 0][sel])), float(np.median(sf[..., 1][sel]))])


class Motion:
    """Cached motion between two grey frames (handover.Motion)."""

    def __init__(self):
        self.prep = {}

    def _p(self, key, g):
        if key not in self.prep:
            self.prep[key] = prep(resize_to_width(g, FLOW_W))
        return self.prep[key]

    def pair(self, ka, a, kb, b):
        pa, pb = self._p(ka, a), self._p(kb, b)
        f = _dis().calc(pa, pb, None)
        sp, v = _speed_and_vector(f, pa, pb)
        return sp, v, float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def vec_err(v, ref):
    return float(np.linalg.norm(v - ref) / max(np.linalg.norm(ref), 0.3))


def seam_score(grays_before, grays_after):
    """bridge_join.join: the join between grays_before[-1] and grays_after[0], against the
    natural pairs either side. grays_*: grey frames at the analysis size, 5 each side.
    seam = vec_err + |1 - diff_ratio| + |1 - speed_ratio|; ~0.8-1.1 for r16's joins."""
    m = Motion()
    before, after = list(grays_before), list(grays_after)
    sp, v, d = m.pair(("s", -1), before[-1], ("s", 0), after[0])
    nb = [m.pair(("pb", i), before[i], ("pb", i + 1), before[i + 1]) for i in range(len(before) - 1)]
    na = [m.pair(("pa", i), after[i], ("pa", i + 1), after[i + 1]) for i in range(len(after) - 1)]
    neigh = nb + na
    d_nat = float(np.median([n[2] for n in neigh]))
    s_nat = max(float(np.median([n[0] for n in neigh])), 0.15)
    v_nat = (nb[-1][1] + na[0][1]) / 2
    r = {"diff_ratio": round(d / max(d_nat, 1e-6), 3), "speed_ratio": round(sp / s_nat, 3),
         "vec_err": round(vec_err(v, v_nat), 3)}
    r["seam"] = round(r["vec_err"] + abs(1 - r["diff_ratio"]) + abs(1 - r["speed_ratio"]), 3)
    return r
