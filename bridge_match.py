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
  - colour: a per-channel offset that moves the bridge's own trend (its first to its last
    frame's mean, after the sharpening) onto the ramp between the two sides' means, keeping its
    frame-to-frame changes.
"""
import cv2
import numpy as np

SIDE = 6          # source frames each side whose detail / colour set the ramp's ends
SIGMA = 1.0       # unsharp radius at the source's own size
AMOUNT = (-1.0, 8.0)


def detail(rgb):
    """Variance of the Laplacian of luma (uint8 HxWx3 RGB)."""
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F, ksize=1).var())


def _unsharp(f32, blur, amount):
    return np.clip(f32 + amount * (f32 - blur), 0, 255)


def _solve(rgb, target, iters=22):
    """The unsharp amount whose result has `target` detail (bisection; detail rises with amount)."""
    f = rgb.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), SIGMA)
    lo, hi = AMOUNT
    if detail(_unsharp(f, blur, hi).astype(np.uint8)) <= target:
        return hi, _unsharp(f, blur, hi)
    if detail(_unsharp(f, blur, lo).astype(np.uint8)) >= target:
        return lo, _unsharp(f, blur, lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if detail(_unsharp(f, blur, mid).astype(np.uint8)) < target:
            lo = mid
        else:
            hi = mid
    return lo, _unsharp(f, blur, lo)


def match_bridge(bridge, left, right):
    """bridge: (n, H, W, 3) uint8 array; left / right: up to SIDE source frames before / after it.
    Returns (matched uint8 array, rows) - rows say what was done to each frame."""
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
    out, rows = np.empty_like(bridge), []
    for k in range(n):
        t = (k + 1) / (n + 1)
        target = d_l + (d_r - d_l) * t
        amount, f = _solve(bridge[k], target)
        want = m_l + (m_r - m_l) * t
        # the bridge's own trend at this frame, plus whatever the sharpening moved its mean by,
        # so only the frame's own flicker around its trend survives
        trend = b_first + (b_last - b_first) * (k / max(1, n - 1))
        moved = f.reshape(-1, 3).mean(0) - bridge[k].reshape(-1, 3).astype(np.float64).mean(0)
        have = trend + moved
        f = np.clip(f + (want - have)[None, None, :], 0, 255)
        out[k] = np.rint(f).astype(np.uint8)
        rows.append({"frame": k, "detail_before": round(detail(bridge[k]), 1), "detail_target": round(target, 1),
                     "detail_after": round(detail(out[k]), 1), "amount": round(amount, 3),
                     "offset": [round(float(x), 2) for x in (want - have)]})
    return out, rows
