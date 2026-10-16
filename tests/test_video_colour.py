"""The matrix every decoder uses: the stream's own tags, whatever type PyAV returns them as.

A tagged BT.709 clip under 720 wide used to decode with the BT.601 guess (PyAV 17 returns the
tag as a plain int), about 1-3 levels darker in the midtones, carried into every result."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import video_colour as vc  # noqa: E402
from av.video.reformatter import ColorRange, Colorspace  # noqa: E402


@pytest.mark.parametrize("tag, w, want", [
    (1, 624, Colorspace.ITU709),                   # PyAV 17: an int, under 720 wide
    (Colorspace.ITU709, 624, Colorspace.ITU709),   # an enum
    ("itu709", 624, "itu709"),                     # a string
    (5, 1920, Colorspace.ITU601),                  # tagged 601 on an HD frame stays 601
    (6, 1920, Colorspace.ITU601),                  # SMPTE 170M is the 601 matrix
    (2, 624, Colorspace.ITU601),                   # untagged: the size guess
    (2, 1280, Colorspace.ITU709),
    (None, 624, Colorspace.ITU601),
    ("unspecified", 1280, Colorspace.ITU709),
    (9, 624, Colorspace.ITU709),                   # BT.2020 has no PyAV member: nearest
])
def test_colorspace_tag_wins_over_the_size_guess(tag, w, want):
    assert vc.stream_colorspace(tag, w, 352) == want


@pytest.mark.parametrize("tag, want", [
    (1, ColorRange.MPEG), (2, ColorRange.JPEG), (0, ColorRange.MPEG), (None, ColorRange.MPEG),
    (ColorRange.JPEG, ColorRange.JPEG), ("unspecified", ColorRange.MPEG),
])
def test_color_range_tag(tag, want):
    assert vc.stream_color_range(tag) == want


def _ffmpeg():
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg isn't installed")
    return ff


def _bt709_clip(path, w=320, h=176, n=6):
    """Saturated colour bars, encoded with the BT.709 matrix and tagged so (TV range)."""
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=24",
                    "-frames:v", str(n), "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                    "-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", str(path)], check=True)


def _reference(path, n):
    """ffmpeg's own decode with the BT.709 matrix, rounded accurately, the ground truth (its
    default fast path rounds down, the bias to_rgb24 removes)."""
    raw = subprocess.run([_ffmpeg(), "-v", "error", "-i", str(path), "-vf",
                          "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int,format=rgb24",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(n, 176, 320, 3).astype(np.float32)


@pytest.fixture()
def clip(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path), raising=False)
    p = tmp_path / "bars.mp4"
    _bt709_clip(p)
    return p, _reference(p, 6)


def test_the_timeline_decodes_a_small_bt709_clip_with_its_own_matrix(clip):
    import timeline as tl
    p, ref = clip
    got = np.stack([f for f in tl._iter_frames(str(p), 24, 0, 6)]).astype(np.float32)
    # mean abs: 1.9 with the right matrix (PyAV and ffmpeg place chroma differently at the
    # bars' hard colour edges), 7.6 with the old BT.601 guess
    assert abs(float((got - ref).mean())) < 0.3 and float(np.abs(got - ref).mean()) < 3.0


def test_recombine_decodes_a_small_bt709_clip_with_its_own_matrix(clip, tmp_path_factory):
    import importlib
    root = tmp_path_factory.mktemp("custom_nodes")
    os.makedirs(str(root / "comfyui-videohelpersuite" / "videohelpersuite"))
    pkg = root / "ss_colour"
    pkg.mkdir()
    open(str(pkg / "__init__.py"), "w").close()
    for name in ("recombine.py", "audio_splice.py", "video_colour.py", "bridge_match.py"):
        shutil.copyfile(os.path.join(ROOT, name), str(pkg / name))
    sys.path.insert(0, str(root))
    try:
        rc = importlib.import_module("ss_colour.recombine")
        p, ref = clip
        got = rc._decode_range(str(p), 24, 0, 6).numpy().astype(np.float32)
    finally:
        sys.path.remove(str(root))
    # the old size guess (BT.601 under 720 wide) was ~1-3 levels off on these bars
    # mean abs: 1.9 with the right matrix (PyAV and ffmpeg place chroma differently at the
    # bars' hard colour edges), 7.6 with the old BT.601 guess
    assert abs(float((got - ref).mean())) < 0.3 and float(np.abs(got - ref).mean()) < 3.0


def test_the_loader_decodes_a_small_bt709_clip_with_its_own_matrix(clip):
    import loader
    p, ref = clip
    node = loader.SeamStitchLoader()
    src = open(os.path.join(ROOT, "loader.py"), encoding="utf-8").read()
    assert "color_args(" in src and "fallback_cs" not in src
    # the Loader's decode is to_rgb24 with color_args' result
    import av
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        cs, cr, dst = loader.color_args(s.codec_context, s.codec_context.width, s.codec_context.height)
        f0 = loader.to_rgb24(next(c.decode(s)), cs, cr, dst).astype(np.float32)
    assert node is not None and float(np.abs(f0 - ref[0]).mean()) < 3.0


# ---------------------------------------------------------------- rounding (to_rgb24)
# swscale's fast YUV -> RGB path rounds down: ~-0.9 / -1.3 / -0.9 levels on a real BT.709 H.264
# clip, which every Result Preview save carried into the result (~1 luma level darker per
# round). to_rgb24 asks for accurate rounding; the per-channel bias against ffmpeg's own
# accurate_rnd decode must be ~0, not just the mean abs error small.

def _accurate_reference(path, n, w=320, h=176):
    raw = subprocess.run([_ffmpeg(), "-v", "error", "-i", str(path), "-vf",
                          "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int,format=rgb24",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(n, h, w, 3).astype(np.float64)


@pytest.fixture()
def dark_clip(tmp_path, monkeypatch):
    """Dim, soft, low-saturation footage-like frames (where the bias showed), BT.709 TV range.
    Saturated hard-edged bars are left out here: at their colour edges PyAV and ffmpeg place
    the chroma samples differently (a few tenths on red), which is not a rounding bias."""
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path), raising=False)
    p = tmp_path / "dark.mp4"
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x176:rate=24",
                    "-frames:v", "6", "-vf", "hue=s=0.4,gblur=sigma=2,eq=contrast=0.7,scale=out_color_matrix=bt709:out_range=tv",
                    "-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", str(p)], check=True)
    return p, _accurate_reference(p, 6)


def _bias(got, ref):
    """(|per-channel bias|, |luma bias|) against the reference, in levels."""
    d = (np.asarray(got, np.float64) - ref).mean(axis=(0, 1, 2))
    return np.abs(d), abs(float(d @ np.array([0.2126, 0.7152, 0.0722])))


def _unbiased(got, ref):
    ch, luma = _bias(got, ref)
    return luma < 0.1 and ch.max() < 0.3


def test_to_rgb24_has_no_rounding_bias(dark_clip):
    import av
    p, ref = dark_clip
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        cs, cr, dst = vc.color_args(s.codec_context, s.codec_context.width, s.codec_context.height)
        frames = list(c.decode(s))
        fast = np.stack([f.reformat(format="rgb24", src_colorspace=cs, src_color_range=cr,
                                    dst_color_range=dst).to_ndarray() for f in frames])
        got = np.stack([vc.to_rgb24(f, cs, cr, dst) for f in frames])
    assert _bias(fast, ref)[1] > 0.8             # the fast path really is ~1 level dark on this clip
    assert _unbiased(got, ref)


def test_the_timeline_cut_has_no_rounding_bias(dark_clip):
    import timeline as tl
    p, ref = dark_clip
    got = np.stack([f for f in tl._iter_frames(str(p), 24, 0, 6)])
    assert _unbiased(got, ref)


def test_recombine_has_no_rounding_bias(dark_clip, tmp_path_factory):
    import importlib
    root = tmp_path_factory.mktemp("custom_nodes_rnd")
    os.makedirs(str(root / "comfyui-videohelpersuite" / "videohelpersuite"))
    pkg = root / "ss_round"
    pkg.mkdir()
    open(str(pkg / "__init__.py"), "w").close()
    for name in ("recombine.py", "audio_splice.py", "video_colour.py", "bridge_match.py"):
        shutil.copyfile(os.path.join(ROOT, name), str(pkg / name))
    sys.path.insert(0, str(root))
    try:
        rc = importlib.import_module("ss_round.recombine")
        p, ref = dark_clip
        got = rc._decode_range(str(p), 24, 0, 6).numpy()
    finally:
        sys.path.remove(str(root))
    assert _unbiased(got, ref)


def test_the_loader_converts_through_to_rgb24():
    src = open(os.path.join(ROOT, "loader.py"), encoding="utf-8").read()
    assert "to_rgb24(frame, src_colorspace, src_color_range, dst_range)" in src
