"""seam_frame on SeamStitchCombine: measured from the written file, raises on mismatch.

Run: python_embeded/python.exe -m pytest tests/test_combine_seam_frame.py -q
"""
import os
import subprocess
import sys

import numpy as np
import pytest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDS = os.environ.get("SEAMSTITCH_TEST_VIDS") or os.path.join(os.path.dirname(os.path.dirname(PACK)), "Test vids")


@pytest.fixture(scope="module")
def combine_mod(tmp_path_factory):
    input_dir = str(tmp_path_factory.mktemp("input"))
    sys.modules["folder_paths"].get_input_directory = lambda: input_dir
    sys.path.insert(0, PACK)
    import combine
    combine.INPUT_DIR = input_dir
    return combine


def _vid(name):
    p = os.path.join(VIDS, name)
    if not os.path.exists(p):
        pytest.skip(f"missing {p}")
    return p


def _ffmpeg(combine, *args):
    r = subprocess.run([combine._ffmpeg_exe(), "-y", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]


def test_stream_copy_join(combine_mod):
    a, b = _vid("4.mp4"), _vid("2.mp4")  # both 832x1280, 48 fps, 248 frames
    out = combine_mod.SeamStitchCombine().combine(a, b, "off", "crop", "t_copy", free_vram_first=False)
    images, _audio, path, seam = out["result"]
    assert seam == 248 and images.shape[0] == 496
    print("stream-copy seam", seam)


def test_mixed_fps_transcode_join(combine_mod, tmp_path):
    a = _vid("4.mp4")
    b24 = str(tmp_path / "b24.mp4")  # 2.mp4 retimed to 24 fps: forces the transcode path
    _ffmpeg(combine_mod, "-i", _vid("2.mp4"), "-r", "24", "-c:v", "libx264", "-crf", "12",
            "-c:a", "aac", b24)
    out = combine_mod.SeamStitchCombine().combine(a, b24, "off", "crop", "t_mix", free_vram_first=False)
    _images, _audio, path, seam = out["result"]
    assert abs(seam - 248) <= 2, seam  # A's 248 frames at 48 fps re-timed to A's own rate
    print("transcode seam", seam)


def test_wrong_candidate_raises(combine_mod, monkeypatch):
    a, b = _vid("4.mp4"), _vid("2.mp4")
    real = combine_mod._boundary_frames

    def lying(path):
        first, last, count = real(path)
        if path == a:
            # claim A is 100 frames longer: no output frame pair can match
            return first, last, count + 100
        return first, last, count

    monkeypatch.setattr(combine_mod, "_boundary_frames", lying)
    with pytest.raises(ValueError, match="Refusing to emit an unverified seam_frame"):
        combine_mod.SeamStitchCombine().combine(a, b, "off", "crop", "t_bad", free_vram_first=False)
