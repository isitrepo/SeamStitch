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
