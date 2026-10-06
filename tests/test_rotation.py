"""A phone clip's rotation tag: every decoder turns the picture the way players show it.

A phone stores the sensor's picture and tags the file (a display matrix: "rotate 180" when held
upside down) instead of turning the pixels. PyAV hands back the stored picture, so the Swap
Planner, Loader, Recombine and Combine read a Pixel clip upside down while every player showed
it upright. Ground truth here is ffmpeg's own decode, which follows the tag."""
import importlib
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import video_colour as vc  # noqa: E402

W, H, N = 160, 96, 6


def _ffmpeg():
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg isn't installed")
    return ff


def _tagged_clip(path, rotation):
    """An asymmetric test card, encoded once and stream-copied with a display rotation tag."""
    plain = str(path) + ".plain.mp4"
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate=24",
                    "-frames:v", str(N), "-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv444p", plain], check=True)
    cmd = [_ffmpeg(), "-v", "error", "-y"]
    if rotation:
        cmd += ["-display_rotation:v", str(rotation)]
    subprocess.run(cmd + ["-i", plain, "-c", "copy", str(path)], check=True)
    os.remove(plain)


def _shown(path):
    """What a player shows: ffmpeg's decode, autorotated (its default)."""
    raw = subprocess.run([_ffmpeg(), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    # a quarter turn swaps the shape (same byte count): ask ffprobe for the shown size
    size = subprocess.run([shutil.which("ffprobe") or "ffprobe", "-v", "error", "-select_streams", "v:0",
                           "-show_entries", "stream_side_data=rotation", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True).stdout.strip()
    turns = int(round(float(size or 0) / 90)) % 4
    h, w = (W, H) if turns % 2 else (H, W)
    return np.frombuffer(raw, np.uint8).reshape(N, h, w, 3).astype(np.float32)


@pytest.fixture(params=[0, 90, -90, 180], ids=["untagged", "rot90", "rot-90", "rot180"])
def clip(request, tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path), raising=False)
    p = tmp_path / f"phone_{request.param}.mp4"
    _tagged_clip(p, request.param)
    return p, request.param, _shown(p)


def _close(got, ref):
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert float(np.abs(got.astype(np.float32) - ref).mean()) < 2.0


@pytest.mark.parametrize("rotation, turns", [(0, 0), (90, 1), (-90, 3), (180, 2), (-180, 2), (270, 3)])
def test_frame_turns_reads_the_tag_in_quarter_turns(rotation, turns):
    class F:
        pass
    f = F()
    f.rotation = rotation
    assert vc.frame_turns(f) == turns
    assert vc.frame_turns(object()) == 0


def test_display_size_swaps_on_a_quarter_turn():
    assert vc.display_size(1920, 1080, 0) == (1920, 1080)
    assert vc.display_size(1920, 1080, 2) == (1920, 1080)
    assert vc.display_size(1920, 1080, 1) == (1080, 1920)
    assert vc.display_size(1920, 1080, 3) == (1080, 1920)


def test_the_file_reads_its_tag(clip):
    p, rotation, _ = clip
    assert vc.file_turns(str(p)) == int(round(rotation / 90)) % 4


def test_the_timeline_and_swap_planner_decode_upright(clip):
    import timeline as tl
    p, _, ref = clip
    info = tl.probe(str(p), 24)
    assert (info["height"], info["width"]) == ref.shape[1:3]
    _close(np.stack(list(tl._iter_frames(str(p), 24, 0, N))), ref)


def test_recombine_decodes_upright(clip, tmp_path_factory):
    root = tmp_path_factory.mktemp("custom_nodes")
    os.makedirs(str(root / "comfyui-videohelpersuite" / "videohelpersuite"))
    pkg = root / "ss_rotation"
    pkg.mkdir()
    open(str(pkg / "__init__.py"), "w").close()
    for name in ("recombine.py", "audio_splice.py", "video_colour.py", "bridge_match.py"):
        shutil.copyfile(os.path.join(ROOT, name), str(pkg / name))
    sys.path.insert(0, str(root))
    try:
        rc = importlib.import_module("ss_rotation.recombine")
        p, _, ref = clip
        assert rc._source_size(str(p)) == (ref.shape[2], ref.shape[1])
        _close(rc._decode_range(str(p), 24, 0, N).numpy(), ref)
    finally:
        sys.path.remove(str(root))


def test_the_loader_decodes_upright(clip):
    import loader
    p, _, ref = clip
    res = loader.SeamStitchLoader().load_video(str(p), 24, "frames", 0.0, 0.0, 0.0, 0, N, 0, snap_to_multiple=0)
    _close((res[0].numpy() * 255.0).round(), ref)


def test_combine_reads_upright(clip):
    import combine
    p, _, ref = clip
    g = combine._probe_geometry(str(p))
    assert (g["height"], g["width"]) == ref.shape[1:3]
    images, _ = combine._decode_for_outputs(str(p))
    _close((images.numpy() * 255.0).round(), ref)
    first, last, count = combine._boundary_frames(str(p))
    assert count == N
    _close(first[None], ref[:1])
