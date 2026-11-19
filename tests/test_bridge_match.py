"""Recombine's match_to_source: a bridge's detail and colour ramp from one side to the other."""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bridge_match as bm  # noqa: E402

H, W = 72, 128


def _texture(seed, blur, level=(120, 100, 90)):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (H, W, 3)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.5)          # some structure, not white noise
    base = (base - base.mean()) * 2.0 + np.array(level, np.float32)
    if blur:
        base = cv2.GaussianBlur(base, (0, 0), blur)
    return np.clip(base, 0, 255).astype(np.uint8)


def test_detail_and_colour_walk_from_the_left_side_to_the_right():
    left = np.stack([_texture(i, 0.7, (120, 100, 90)) for i in range(6)])       # softer side
    right = np.stack([_texture(10 + i, 0.0, (110, 96, 92)) for i in range(6)])  # sharper side
    bridge = np.stack([_texture(20 + i, 1.3, (114, 92, 88)) for i in range(10)])  # softest, a bit dark
    out, rows = bm.match_bridge(bridge, left, right)
    d_l = np.mean([bm.detail(f) for f in left])
    d_r = np.mean([bm.detail(f) for f in right])
    after = [bm.detail(f) for f in out]
    assert all(bm.detail(b) < d_l for b in bridge)                      # the bridge started softer than both
    assert all(after[i] < after[i + 1] for i in range(len(after) - 1))   # a ramp, not a jump
    assert d_l < after[0] < after[-1] < d_r
    for r in rows:
        assert abs(r["detail_after"] - r["detail_target"]) <= 0.03 * r["detail_target"]
    # colour: each frame lands on the ramp between the sides' means, give or take its own
    # flicker around the bridge's first-to-last trend (kept on purpose)
    m_l, m_r = left.reshape(-1, 3).mean(0), right.reshape(-1, 3).mean(0)
    means, orig = out.reshape(10, -1, 3).mean(1), bridge.reshape(10, -1, 3).astype(float).mean(1)
    for k in range(10):
        t = (k + 1) / 11
        flicker = orig[k] - (orig[0] + (orig[-1] - orig[0]) * k / 9)
        assert np.allclose(means[k] - flicker, m_l + (m_r - m_l) * t, atol=1.0)


def test_a_bridge_sharper_than_its_sides_is_softened():
    left = np.stack([_texture(i, 2.0) for i in range(6)])
    right = np.stack([_texture(10 + i, 2.0) for i in range(6)])
    bridge = np.stack([_texture(20 + i, 0.0) for i in range(4)])
    out, rows = bm.match_bridge(bridge, left, right)
    assert all(r["amount"] < 0 for r in rows)
    assert all(bm.detail(o) < bm.detail(b) for o, b in zip(out, bridge))


def test_nothing_either_side_leaves_the_bridge_alone():
    bridge = np.stack([_texture(1, 1.0)])
    out, rows = bm.match_bridge(bridge, np.zeros((0, H, W, 3), np.uint8), np.stack([_texture(2, 0)]))
    assert rows == [] and np.array_equal(out, bridge)


# ---------------------------------------------------------------- through Recombine

@pytest.fixture(scope="module")
def recombine(tmp_path_factory):
    import importlib
    import shutil
    root = tmp_path_factory.mktemp("custom_nodes")
    os.makedirs(str(root / "comfyui-videohelpersuite" / "videohelpersuite"))
    pkg = root / "ss_match"
    pkg.mkdir()
    open(str(pkg / "__init__.py"), "w").close()
    for name in ("recombine.py", "audio_splice.py", "video_colour.py", "bridge_match.py"):
        shutil.copyfile(os.path.join(ROOT, name), str(pkg / name))
    sys.path.insert(0, str(root))
    mod = importlib.import_module("ss_match.recombine")
    mod._encode_video = lambda images, *a, **k: {"ui": {"gifs": []}, "result": ((False, []),)}
    yield mod
    sys.path.remove(str(root))


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg isn't installed")
    p = str(tmp_path_factory.mktemp("m") / "src.mp4")
    subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate=24", "-frames:v", "48",
                    "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv420p", p], check=True)
    return p


def test_recombine_matches_only_when_asked(recombine, clip):
    import torch
    src = recombine._decode_range(clip, 24, 0, None)
    soft = torch.from_numpy(np.stack([cv2.GaussianBlur(f, (0, 0), 0.9) for f in src[20:30].numpy()]))
    images = soft.to(torch.float32) / 255.0
    node = recombine.SeamStitchRecombine()
    args = (images, clip, 20, 29, 24, 0.0, 6, "t", "video/h264-mp4")
    off = node.recombine(*args, save_output=False, skip_encode=True)["result"][1]
    on = node.recombine(*args, save_output=False, skip_encode=True, match_to_source=True)["result"][1]
    to_u8 = lambda t: (t.numpy() * 255).round().astype(np.uint8)
    off, on = to_u8(off), to_u8(on)
    assert np.array_equal(off[:20], on[:20]) and np.array_equal(off[30:], on[30:])   # the source untouched
    assert np.array_equal(off[20:30], soft.numpy())                                  # off: pasted as given
    d_side = np.mean([bm.detail(f) for f in np.concatenate([off[14:20], off[30:36]])])
    assert np.mean([bm.detail(f) for f in on[20:30]]) > 0.8 * d_side > np.mean([bm.detail(f) for f in off[20:30]])


def test_match_to_source_is_appended_last():
    import ast
    src = open(os.path.join(ROOT, "recombine.py"), encoding="utf-8").read()
    i = src.index('"optional": {', src.index("def INPUT_TYPES"))
    names = [n for n in ("original_audio_override", "bridge_audio", "audio_mode", "audio_crossfade_ms",
                         "audio_bridge_weight", "insert", "context_frames", "skip_encode", "match_to_source")]
    pos = [src.index(f'"{n}":', i) for n in names]
    assert pos == sorted(pos), "a widget moved: saved workflows restore widgets by position"
    assert ast.parse(src)


def _spread(frames):
    return np.asarray(frames).astype(np.float64).reshape(len(frames), -1, 3).std(1)


def test_contrast_walks_from_the_left_side_to_the_right():
    # sides with a plain spread; a bridge ~5% more contrasty (what H3 did at a Swap join)
    left = np.stack([_texture(i, 1.0, (60, 55, 50)) for i in range(6)])
    right = np.stack([_texture(10 + i, 1.0, (64, 58, 52)) for i in range(6)])
    mid = np.stack([_texture(20 + i, 1.0, (60, 55, 50)) for i in range(8)]).astype(np.float32)
    mu = mid.reshape(8, -1, 3).mean(1)[:, None, None, :]
    bridge = np.clip((mid - mu) * 1.15 + mu, 0, 255).astype(np.uint8)
    out, rows = bm.match_bridge(bridge, left, right)
    s_l, s_r = _spread(left).mean(0), _spread(right).mean(0)
    before, after = _spread(bridge), _spread(out)
    assert (before.mean(1) > 1.05 * s_l.mean()).all()
    for k in range(8):
        t = (k + 1) / 9
        want = s_l + (s_r - s_l) * t
        # each frame keeps its own flicker around the bridge's first-to-last spread trend
        flicker = before[k] / (before[0] + (before[-1] - before[0]) * k / 7)
        assert np.allclose(after[k] / flicker, want, rtol=0.02), (k, after[k] / flicker, want)
        assert rows[k]["gain"][0] < 1.0
    # detail still lands on its own ramp after the gain
    for r in rows:
        assert abs(r["detail_after"] - r["detail_target"]) <= 0.03 * r["detail_target"]


# ---------------------------------------------------------------- the flicker cap

def _frames(n, level, seed0, wobble=None):
    """n frames of ONE texture (so clipping is the same in each and a frame's mean moves only by the
    level given), soft enough that nothing clips: level plus wobble[i] per frame."""
    base = _texture(seed0, 1.0, (128, 128, 128)).astype(np.float32) - 128.0
    base *= 0.35
    out = []
    for i in range(n):
        lv = np.array(level, np.float32) + (0.0 if wobble is None else wobble[i])
        out.append(np.clip(base + lv, 0, 255).round().astype(np.uint8))
    return np.stack(out)


def test_a_dip_in_the_middle_of_the_bridge_is_flattened():
    # steady sides; a bridge that darkens by ~6 levels in its middle (join 422's 418, 2026-10-14)
    left = _frames(6, (100, 90, 80), 0)
    right = _frames(6, (100, 90, 80), 10)
    dip = [0, 0, -2, -6, -6, -4, -1, 0]
    bridge = _frames(8, (100, 90, 80), 20, dip)
    out, rows = bm.match_bridge(bridge, left, right)
    means = out.reshape(8, -1, 3).astype(float).mean(1)
    ramp = (left.reshape(-1, 3).mean(0) + right.reshape(-1, 3).mean(0)) / 2
    assert np.abs(bridge.reshape(8, -1, 3).astype(float).mean(1) - ramp).max() > 5
    assert np.abs(means - ramp).max() <= bm.FLICKER_FLOOR[0] + 1.0
    assert any(abs(c) > 3 for r in rows for c in r["flicker_cut"])


def test_flicker_like_the_sources_own_is_kept():
    # the sides breathe +-2 levels; the bridge breathes the same - nothing is cut
    w = [0, 2, 0, -2, 0, 2]
    left = _frames(6, (100, 90, 80), 0, w)
    right = _frames(6, (100, 90, 80), 10, w)
    bridge = _frames(6, (100, 90, 80), 20, [0, 1.5, 0, -1.5, 0, 1.5])
    out, rows = bm.match_bridge(bridge, left, right)
    assert all(abs(c) < 0.05 for r in rows for c in r["flicker_cut"])


def _solve_reference(rgb, target, iters=22):
    """_solve as it was before the buffer-reusing trial: a full-size array allocated for every trial."""
    f = rgb.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), bm.SIGMA)
    lo, hi = bm.AMOUNT
    if bm.detail(bm._q(bm._unsharp(f, blur, hi))) <= target:
        return hi, bm._unsharp(f, blur, hi)
    if bm.detail(bm._q(bm._unsharp(f, blur, lo))) >= target:
        return lo, bm._unsharp(f, blur, lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if bm.detail(bm._q(bm._unsharp(f, blur, mid))) < target:
            lo = mid
        else:
            hi = mid
    return lo, bm._unsharp(f, blur, lo)


@pytest.mark.parametrize("target_scale", [0.3, 1.0, 3.0, 1000.0, 0.0001])   # in range, and past both ends
def test_solve_matches_the_original_expression_byte_for_byte(target_scale):
    frame = _texture(5, 0.9)
    target = bm.detail(frame) * target_scale
    a_new, f_new = bm._solve(frame, target)
    a_old, f_old = _solve_reference(frame, target)
    assert a_new == a_old and np.array_equal(f_new, f_old)


def test_matching_frames_on_several_threads_gives_the_same_bytes_and_rows():
    left = np.stack([_texture(i, 0.7) for i in range(6)])
    right = np.stack([_texture(10 + i, 0.0, (110, 96, 92)) for i in range(6)])
    bridge = np.stack([_texture(20 + i, 1.3, (114, 92, 88)) for i in range(9)])
    one, rows_one = bm.match_bridge(bridge, left, right, workers=1)
    many, rows_many = bm.match_bridge(bridge, left, right, workers=4)
    assert np.array_equal(one, many) and rows_one == rows_many
    assert [r["frame"] for r in rows_many] == list(range(9))               # in frame order
