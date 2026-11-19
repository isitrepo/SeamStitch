"""Frame Grab: a first and a last frame out of a clip, with the Planner's frame numbering."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import frame_grab as fg  # noqa: E402


def test_numbers_clamp():
    assert fg.pick_numbers(0, -1, 133) == (0, 132)
    assert fg.pick_numbers(5, 999, 133) == (5, 132)
    assert fg.pick_numbers(80, 80, 133) == (80, 80)
    with pytest.raises(ValueError):
        fg.clamp_frame(0, 0)


def test_batch_input_indexes_the_batch():
    imgs = torch.rand(10, 4, 6, 3)
    out = fg.SeamStitchFrameGrab().grab("none", 2, -1, save_png=False, images=imgs)["result"]
    assert torch.equal(out[0], imgs[2:3]) and torch.equal(out[1], imgs[9:10])
    assert out[2:4] == (2, 9)


def test_png_is_lossless_and_never_overwrites(tmp_path):
    img = torch.rand(1, 8, 8, 3)
    p1 = fg.write_png(img, str(tmp_path), "a.png")
    p2 = fg.write_png(img, str(tmp_path), "a.png")
    assert p1 != p2 and os.path.isfile(p1) and os.path.isfile(p2)
    from PIL import Image
    back = np.asarray(Image.open(p1)).astype(np.float32) / 255.0
    assert np.abs(back - img[0].numpy()).max() <= 0.5 / 255 + 1e-6


SRC = os.environ.get("SEAMSTITCH_TEST_CLIP", "")   # a 133-frame, 25 fps video (the checks below are for that clip)


@pytest.mark.skipif(not os.path.isfile(SRC), reason="set SEAMSTITCH_TEST_CLIP to a 133-frame 25 fps clip to run this")
def test_file_frames_match_ffmpeg():
    import subprocess
    import tempfile
    import timeline as tl
    info = tl.probe(SRC, 0)
    assert info["frames"] == 133 and info["frame_rate"] == 25
    for n in (0, 80, 132):
        mine = (fg.decode_one(SRC, 25, n)[0].numpy() * 255 + 0.5).astype(np.int16)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "f.png")
            subprocess.run(["ffmpeg", "-v", "error", "-i", SRC, "-vf", f"select=eq(n{chr(92)},{n})", "-frames:v", "1",
                            "-fps_mode", "passthrough", out], check=True)
            from PIL import Image
            ref = np.asarray(Image.open(out).convert("RGB")).astype(np.int16)
        assert mine.shape == ref.shape
        assert np.abs(mine - ref).mean() < 2.0, f"frame {n} differs from ffmpeg"
