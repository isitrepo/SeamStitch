"""Match a regenerated bridge to the source either side of it: detail and colour, as ramps.

A generator rarely reproduces the source's own sharpness (MiniMax H3 at a seam came out at
about a third of the detail of the footage either side), and the two sides of a seam often
differ too: a Swap join between two takes went from detail ~850 to ~2000 at one frame. Pasting
the bridge in as it is keeps that jump, just moved to the bridge's far edge.

For bridge frame k of n, the target is the left side's value plus (k + 1) / (n + 1) of the way
to the right side's, so the bridge walks from one side to the other:
  - detail (variance of the Laplacian of luma): an unsharp mask whose amount is solved per frame
    to hit the target (a negative amount blends toward the blur, for a bridge sharper than its
    sides);
  - contrast: a per-channel gain about the frame's mean that moves the bridge's own spread
    trend (standard deviation, first to last frame, after the sharpening) onto the ramp between
    the two sides' spreads. MiniMax H3 bridges came out ~5% more contrasty than the footage
    (std 36.1 -> 37.8 in one frame on a Swap join), which reads as the bridge going darker;
  - colour: a per-channel offset that moves the bridge's own trend (its first to its last
    frame's mean, after the sharpening) onto the ramp between the two sides' means, keeping its
    frame-to-frame changes;
  - flicker cap: the frame-to-frame changes kept (each frame's brightness, colour and spread
    around the bridge's trend) are capped at what the source either side does: 1.5x its own
    wobble, never less than FLICKER_FLOOR. A bridge that dips 5.7 levels mid-way (join 422,
    2026-10-14) is flattened; a bridge that breathes like the footage keeps it.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))   # bridge frames matched at once (each holds a few full-size float frames)
SIDE = 6         # source frames each side whose detail / colour set the ramp's ends
SIGMA = 1.0       # unsharp radius at the source's own size
AMOUNT = (-1.0, 8.0)
GAIN = (0.5, 2.0)  # contrast gain limits: a near-flat frame must not be blown up
FLICKER_MULT = 1.5        # the bridge may wobble this much more than the source either side does
FLICKER_FLOOR = (0.5, 0.01)   # ... and always this much: levels of mean, fraction of spread


def detail(rgb):
    """Variance of the Laplacian of luma (uint8 HxWx3 RGB)."""
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F, ksize=1).var())


def _unsharp(f32, blur, amount):
    return np.clip(f32 + amount * (f32 - blur), 0, 255)


def _q(f32):
    """8 bits as the output is stored (rounded, not truncated: truncating a near-identity
    unsharp drops half the pixels a level, and the solve then aimed at the wrong detail)."""
    return np.rint(f32).astype(np.uint8)


def _solve(rgb, target, iters=22):
    """The unsharp amount whose result has `target` detail (bisection; detail rises with amount)."""
    f = rgb.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), SIGMA)
    diff = f - blur
    buf = np.empty_like(f)

    def trial(amount):
        # detail of the sharpened frame as it is stored: the same operations, in the same order, as
        # _q(_unsharp(f, blur, amount)) but into one reused buffer (no full-size array allocated per trial,
        # which was most of the time); the same bytes
        np.multiply(diff, amount, out=buf)
        np.add(buf, f, out=buf)
        np.clip(buf, 0, 255, out=buf)
        np.rint(buf, out=buf)
        return detail(buf.astype(np.uint8))

    lo, hi = AMOUNT
    if trial(hi) <= target:
        return hi, _unsharp(f, blur, hi)
    if trial(lo) >= target:
        return lo, _unsharp(f, blur, lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if trial(mid) < target:
            lo = mid
        else:
            hi = mid
    return lo, _unsharp(f, blur, lo)


def _wobble(values):
    """(n, c) per-frame values -> (c,) the largest distance of any frame from the straight line through
    them: how much the footage flickers around its own trend."""
    v = np.asarray(values, np.float64)
    if v.shape[0] < 3:
        return np.zeros(v.shape[1])
    x = np.arange(v.shape[0], dtype=np.float64)
    fit = np.polyfit(x, v, 1)
    return np.abs(v - (np.outer(x, fit[0]) + fit[1])).max(0)


def _spread(frames):
    """(n, 3): each frame's per-channel standard deviation."""
    a = np.asarray(frames).astype(np.float64)
    return a.reshape(a.shape[0], -1, 3).std(1)


def match_bridge(bridge, left, right, workers=None):
    """bridge: (n, H, W, 3) uint8 array; left / right: up to SIDE source frames before / after it.
    Returns (matched uint8 array, rows) - rows say what was done to each frame. workers = frames matched at
    once (default WORKERS; 1 = one at a time); the result is the same bytes either way."""
    bridge = np.asarray(bridge)
    n = bridge.shape[0]
    if n == 0 or len(left) == 0 or len(right) == 0:
        return bridge, []
    left, right = np.asarray(left)[-SIDE:], np.asarray(right)[:SIDE]
    d_l = float(np.mean([detail(f) for f in left]))
    d_r = float(np.mean([detail(f) for f in right]))
    m_l = left.reshape(-1, 3).astype(np.float64).mean(0)
    m_r = right.reshape(-1, 3).astype(np.float64).mean(0)
    b_first = bridge[0].reshape(-1, 3).astype(np.float64).mean(0)
    b_last = bridge[-1].reshape(-1, 3).astype(np.float64).mean(0)
    # spread: each frame's own, averaged (pooling the frames would count their brightness
    # changes as contrast)
    s_l, s_r = _spread(left).mean(0), _spread(right).mean(0)
    sb_first, sb_last = _spread(bridge[:1])[0], _spread(bridge[-1:])[0]
    # how much the source wobbles around its own trend, either side: the most the bridge may keep
    means_l, means_r = left.reshape(len(left), -1, 3).mean(1), right.reshape(len(right), -1, 3).mean(1)
    cap_m = np.maximum(FLICKER_MULT * np.maximum(_wobble(means_l), _wobble(means_r)), FLICKER_FLOOR[0])
    sp_l, sp_r = _spread(left), _spread(right)
    rel = np.maximum(_wobble(sp_l) / np.maximum(sp_l.mean(0), 1e-6), _wobble(sp_r) / np.maximum(sp_r.mean(0), 1e-6))
    cap_s = np.maximum(FLICKER_MULT * rel, FLICKER_FLOOR[1])
    out = np.empty_like(bridge)

    def one(k):
        # one bridge frame; it reads only the shared ramps above and writes only out[k], so frames are
        # independent and can run on separate threads (numpy and OpenCV release the GIL) with the same bytes
        t = (k + 1) / (n + 1)
        target = d_l + (d_r - d_l) * t
        # Contrast and colour first, then the sharpening solved on the frame as it will be (an
        # offset applied after the solve clips dark or bright pixels and moves the detail).
        #  - contrast: a gain about the frame's own mean, moving the bridge's spread trend onto
        #    the ramp; sharpening moves the spread a little too, so a second pass divides the
        #    gain by what the first pass's sharpening did to it;
        #  - colour: the bridge's mean trend moved onto the ramp (each frame keeps its own
        #    flicker around the trend).
        want_s = s_l + (s_r - s_l) * t
        trend_s = sb_first + (sb_last - sb_first) * (k / max(1, n - 1))
        raw = bridge[k].astype(np.float32)
        raw_s = raw.reshape(-1, 3).std(0)
        # the frame's own spread flicker (raw / trend), capped; the gain takes the spread to want x flicker
        ratio = np.where(trend_s > 1e-6, raw_s / np.maximum(trend_s, 1e-6), 1.0)
        ratio_kept = np.clip(ratio, 1.0 - cap_s, 1.0 + cap_s)
        gain = np.where(raw_s > 1e-6, want_s * ratio_kept / np.maximum(raw_s, 1e-6), 1.0)
        want = m_l + (m_r - m_l) * t
        trend = b_first + (b_last - b_first) * (k / max(1, n - 1))
        mu = raw.reshape(-1, 3).mean(0)
        flicker = mu - trend
        kept = np.clip(flicker, -cap_m, cap_m)
        aim = want + kept                             # where this frame's mean should end up
        for rnd in range(2):
            used = np.clip(gain, *GAIN)
            g8 = np.rint(np.clip((raw - mu) * used.astype(np.float32) + aim.astype(np.float32), 0, 255)).astype(np.uint8)
            amount, f = _solve(g8, target)
            if rnd == 0:
                s_g, s_f = g8.reshape(-1, 3).std(0), f.reshape(-1, 3).std(0)
                gain = used * np.where(s_f > 1e-6, s_g / np.maximum(s_f, 1e-6), 1.0)
        gain = used
        # what clipping and the sharpening moved the mean by, put back in whole levels: a
        # fractional shift re-rounded to 8 bits moves half the pixels by one level, which reads
        # as added fine detail on a soft frame (44 -> 46)
        fix = np.rint(aim - f.reshape(-1, 3).mean(0))
        f = np.clip(f + fix[None, None, :], 0, 255)
        out[k] = np.rint(f).astype(np.uint8)
        return {"frame": k, "detail_before": round(detail(bridge[k]), 1), "detail_target": round(target, 1),
                "detail_after": round(detail(out[k]), 1), "amount": round(amount, 3),
                "gain": [round(float(x), 4) for x in gain],
                "flicker_cut": [round(float(x), 2) for x in (flicker - kept)],
                "offset": [round(float(x), 2) for x in (aim - mu + fix)]}

    workers = max(1, min(int(workers if workers is not None else WORKERS), n))
    if workers == 1:
        rows = [one(k) for k in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(one, range(n)))      # in frame order
    return out, rows


# ---------------------------------------------------------------------------
# geometry: the set redrawn a little wider or narrower on one side of a join
# ---------------------------------------------------------------------------
# A background replace renders the place afresh in every chunk, and a 60 s background-replace test job had the
# couch 0.6-2.7% wider or narrower from one chunk to the next (height unchanged, 10-14 px sideways) at 5 of
# its 6 joins (2026-10-14). The bridge between two such chunks is pinned to both, so H3 has to switch
# geometry somewhere inside it, and it snapped (a 7.1 step mid-bridge at join 422). match_geometry warps
# each bridge frame so the set moves evenly from the left side's geometry to the right side's, the same
# ramp as detail / contrast / colour. A camera can't stretch the picture one way only, but spread over the
# bridge (about 0.1% a frame) the change reads as nothing.
GEO_MIN = (0.002, 1.0)    # a side-to-side change under 0.2% and 1 px: nothing to spread
GEO_MAX = (0.06, 64.0)    # over 6% or 64 px: another framing, not a redrawn set - left alone
_GEO_W = 960              # features are found at up to this width (speed); the warp is at full size


def _grey(rgb, s):
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1.0 else g


def _m3(m23):
    return np.vstack([np.asarray(m23, np.float64), [0.0, 0.0, 1.0]])


def _affine(a, b, orb=None):
    """3x3 affine taking frame a's picture coordinates to frame b's (full size), or None. ORB features
    and RANSAC, so a person moving in front of the set is outvoted by the set itself."""
    h, w = a.shape[:2]
    s = min(1.0, _GEO_W / float(w))
    orb = orb or cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(_grey(a, s), None)
    kb, db = orb.detectAndCompute(_grey(b, s), None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db), key=lambda x: x.distance)[:800]
    if len(m) < 30:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffine2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if M is None or inl is None or int(inl.sum()) < max(20, len(m) // 4):
        return None
    S = np.diag([s, s, 1.0])
    return np.linalg.inv(S) @ _m3(M) @ S


def _size(M, w, h):
    """(largest scale change, largest corner shift in px) of a 3x3 affine on a w x h frame."""
    sx, sy = np.hypot(M[0, 0], M[1, 0]), np.hypot(M[0, 1], M[1, 1])
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64).T
    shift = np.abs((M @ corners)[:2] - corners[:2]).max()
    return max(abs(sx - 1.0), abs(sy - 1.0)), float(shift)


def match_geometry(bridge, left_last, right_first):
    """bridge: (n, H, W, 3) uint8; left_last / right_first: the source frames either side of it.
    Returns (warped uint8 array, info). info["applied"] is False when the sides' geometry is the same,
    too different to be a redrawn set, or can't be measured; then the bridge is returned untouched."""
    bridge = np.asarray(bridge)
    n = bridge.shape[0]
    info = {"applied": False, "frames": []}
    if n == 0:
        return bridge, info
    h, w = bridge.shape[1:3]
    orb = cv2.ORB_create(4000)
    A = _affine(left_last, right_first, orb)
    if A is None:
        info["reason"] = "the two sides could not be matched"
        return bridge, info
    ds, dp = _size(A, w, h)
    info.update(scale_x=round(float(np.hypot(A[0, 0], A[1, 0])), 4),
                scale_y=round(float(np.hypot(A[0, 1], A[1, 1])), 4), shift=round(dp, 1))
    if ds < GEO_MIN[0] and dp < GEO_MIN[1]:
        info["reason"] = "the sides already match"
        return bridge, info
    if ds > GEO_MAX[0] or dp > GEO_MAX[1]:
        info["reason"] = "the sides differ too much to be the same set redrawn"
        return bridge, info
    # Where H3 put the set in each frame, three ways: against the left side, through the right side
    # (A after the frame-to-right match, inverted), and chained from the frame before. The person moving
    # throws any one of them off at times (on join 422 matching to the left side alone made the correction
    # wobble +-0.3% a frame; the chain alone was 0.6% off at the frame where H3 snapped), so each frame
    # takes the element-wise median of the ones that matched.
    G_all, prev = [], None
    for k in range(n):
        ests = []
        g_left = _affine(left_last, bridge[k], orb)
        if g_left is not None:
            ests.append(g_left)
        to_right = _affine(bridge[k], right_first, orb)
        if to_right is not None:
            ests.append(np.linalg.inv(to_right) @ A)
        if k > 0 and prev is not None:
            step = _affine(bridge[k - 1], bridge[k], orb)
            if step is not None:
                ests.append(step @ prev)
        G = None
        if len(ests) >= 2:
            G = np.median(np.stack(ests), axis=0)
        elif ests:
            G = ests[0]
        G_all.append(G)
        prev = G
    out = bridge.copy()
    eye = np.eye(3)
    for k in range(n):
        t = (k + 1) / (n + 1)
        D = eye + t * (A - eye)                      # where the set should be at this frame
        G = G_all[k]
        row = {"frame": k, "t": round(t, 3)}
        if G is None:
            row["skipped"] = "unmatched"
        else:
            C = D @ np.linalg.inv(G)
            cs, cp = _size(C, w, h)
            if cs > GEO_MAX[0] or cp > GEO_MAX[1]:
                row["skipped"] = "correction too large"
            else:
                out[k] = cv2.warpAffine(bridge[k], C[:2], (w, h), flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REFLECT101)
                row.update(scale=round(cs, 4), shift=round(cp, 1))
        info["frames"].append(row)
    info["applied"] = any("skipped" not in r for r in info["frames"])
    return out, info
