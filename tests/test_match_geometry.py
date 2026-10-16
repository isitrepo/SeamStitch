"""Recombine's match_geometry: a set redrawn a little wider on one side of a bridge is spread over the bridge.

A background replace renders the place afresh in every chunk; at 5 of 6 joins on a real Swap job the couch
was 0.6-2.7% wider or narrower on one side (height unchanged), and H3 snapped between the two inside the
bridge. The warp puts each bridge frame on a ramp from the left side's geometry to the right side's."""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bridge_match as bm  # noqa: E402

H, W = 360, 640


def _set(seed=3):
    """A textured 'set': blocks and lines with plenty of corners, as wallpaper and upholstery give ORB."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 90, np.uint8)
    for _ in range(260):
        x, y = int(rng.integers(0, W - 20)), int(rng.integers(0, H - 20))
        w, h = int(rng.integers(6, 40)), int(rng.integers(6, 40))
        cv2.rectangle(img, (x, y), (x + w, y + h), tuple(int(c) for c in rng.integers(20, 235, 3)), -1)
    return cv2.GaussianBlur(img, (0, 0), 0.6)


def _stretch(img, sx, tx=0.0):
    M = np.float32([[sx, 0, tx], [0, 1, 0]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)


def _sx(a, b):
    M = bm._affine(a, b)
    return float(np.hypot(M[0, 0], M[1, 0]))


def test_a_snap_inside_the_bridge_becomes_a_ramp():
    base = _set()
    SX, TX = 1.015, -9.0                       # join 422: 1.5% wider, 9 px left
    left, right = base, _stretch(base, SX, TX)
    n = 10
    bridge = np.stack([left if k < 6 else right for k in range(n)])   # H3 held, then snapped at frame 6
    out, info = bm.match_geometry(bridge, left, right)
    assert info["applied"] and abs(info["scale_x"] - SX) < 0.002 and abs(info["scale_y"] - 1) < 0.002
    got = [_sx(left, f) for f in out]
    want = [1 + (k + 1) / (n + 1) * (SX - 1) for k in range(n)]
    assert max(abs(g - w) for g, w in zip(got, want)) < 0.0015, (got, want)
    steps = np.diff([1.0] + got + [SX])
    assert steps.max() < 0.004                 # the 1.5% snap is gone: no step over 0.4%


def test_matching_sides_are_left_alone():
    base = _set()
    bridge = np.stack([base] * 5)
    out, info = bm.match_geometry(bridge, base, base)
    assert not info["applied"] and info["reason"] == "the sides already match"
    assert np.array_equal(out, bridge)


def test_a_real_reframe_is_left_alone():
    base = _set()
    right = cv2.warpAffine(base, np.float32([[1.10, 0, -32], [0, 1.10, -18]]), (W, H), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REFLECT101)   # 10% closer: another framing
    bridge = np.stack([base] * 4)
    out, info = bm.match_geometry(bridge, base, right)
    assert not info["applied"] and "too much" in info["reason"]
    assert np.array_equal(out, bridge)


def test_featureless_sides_are_left_alone():
    flat = np.full((H, W, 3), 128, np.uint8)
    bridge = np.stack([flat] * 3)
    out, info = bm.match_geometry(bridge, flat, flat)
    assert not info["applied"] and np.array_equal(out, bridge)


def test_match_geometry_is_appended_after_match_to_source():
    src = open(os.path.join(ROOT, "recombine.py"), encoding="utf-8").read()
    i = src.index('"optional": {', src.index("def INPUT_TYPES"))
    names = ("skip_encode", "match_to_source", "match_geometry")
    pos = [src.index(f'"{n}":', i) for n in names]
    assert pos == sorted(pos), "a widget moved: saved workflows restore widgets by position"
    assert "match_geometry=False" in src


@pytest.fixture(scope="module")
def recombine(tmp_path_factory):
    import importlib
    import shutil
    root = tmp_path_factory.mktemp("custom_nodes_geo")
    os.makedirs(str(root / "comfyui-videohelpersuite" / "videohelpersuite"))
    pkg = root / "ss_geo"
    pkg.mkdir()
    open(str(pkg / "__init__.py"), "w").close()
    for name in ("recombine.py", "audio_splice.py", "video_colour.py", "bridge_match.py"):
        shutil.copyfile(os.path.join(ROOT, name), str(pkg / name))
    sys.path.insert(0, str(root))
    mod = importlib.import_module("ss_geo.recombine")
    mod._encode_video = lambda images, *a, **k: {"ui": {"gifs": []}, "result": ((False, []),)}
    yield mod
    sys.path.remove(str(root))


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """40 frames of the set, the last 20 drawn 1.5% wider (two chunks), lossless so ORB sees the same corners."""
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg isn't installed")
    base = _set()
    frames = [base] * 20 + [_stretch(base, 1.015, -9.0)] * 20
    p = str(tmp_path_factory.mktemp("g") / "src.mkv")
    proc = subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24",
                             "-i", "-", "-c:v", "ffv1", "-pix_fmt", "gbrp", p], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    return p, frames


def test_recombine_warps_only_when_asked(recombine, clip):
    import torch
    p, frames = clip
    # the bridge replaces 14..25 (12 frames) and snaps at its middle, as H3 did
    snap = np.stack([frames[0]] * 6 + [frames[39]] * 6)
    images = torch.from_numpy(snap).to(torch.float32) / 255.0
    node = recombine.SeamStitchRecombine()
    args = (images, p, 14, 25, 24, 0.0, 0, "t", "video/h264-mp4")
    to_u8 = lambda t: (t.numpy() * 255).round().astype(np.uint8)
    off = to_u8(node.recombine(*args, save_output=False, skip_encode=True)["result"][1])
    on = to_u8(node.recombine(*args, save_output=False, skip_encode=True, match_geometry=True)["result"][1])
    assert np.array_equal(off[:14], on[:14]) and np.array_equal(off[26:], on[26:])   # the source untouched
    assert np.array_equal(off[14:26], snap)                                          # off: pasted as given
    widths = [_sx(frames[0], f) for f in on[13:27]]
    assert max(np.diff(widths)) < 0.004 and widths[-1] > 1.012                       # a ramp to the right side
