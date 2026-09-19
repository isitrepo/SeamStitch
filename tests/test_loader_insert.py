"""Loader insert mode: the pure frame arithmetic, plus anchors decoded through the node.

Run: python_embeded/python.exe -m pytest tests/test_loader_insert.py -q
"""
import os
import sys

import pytest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)
import insert_math as im  # noqa: E402

VIDS = os.environ.get("SEAMSTITCH_TEST_VIDS") or os.path.join(os.path.dirname(os.path.dirname(PACK)), "Test vids")


def test_plan_n0():
    p = im.plan_insert(join=10, trim_each_side=0, clip_frames=100)
    assert p == {"first_anchor": 9, "last_anchor": 10, "start_frame": 10, "end_frame": 9}


def test_plan_n_positive():
    p = im.plan_insert(join=10, trim_each_side=3, clip_frames=100)
    assert p == {"first_anchor": 6, "last_anchor": 13, "start_frame": 7, "end_frame": 12}
    # removed range is 2N frames
    assert p["end_frame"] - p["start_frame"] + 1 == 6


def test_boundaries_ok():
    im.plan_insert(join=1, trim_each_side=0, clip_frames=100)    # first anchor = 0
    im.plan_insert(join=99, trim_each_side=0, clip_frames=100)   # last anchor = 99 = last frame
    im.plan_insert(join=5, trim_each_side=4, clip_frames=100)    # first anchor = 0


@pytest.mark.parametrize("join,n", [(0, 0), (4, 4), (1, 1), (100, 0), (99, 1), (98, 2)])
def test_boundaries_fail_and_name_clip_length(join, n):
    with pytest.raises(ValueError, match="100 frames"):
        im.plan_insert(join, n, 100)


@pytest.mark.parametrize("fps", [24, 25, 30])
def test_snap_8n1(fps):
    for secs in (0.1, 1.0, 2.0, 2.5, 3.3, 5.0):
        n, dur = im.bridge_length(secs, 0, "seconds", fps)
        assert n >= 9 and (n - 1) % 8 == 0
        assert dur == pytest.approx(n / fps)
        assert abs(n - secs * fps) <= 4 or n == 9  # nearest grid point
    assert im.bridge_length(2.0, 0, "seconds", 24)[0] == 49
    assert im.bridge_length(2.0, 0, "seconds", 25)[0] == 49   # 50 -> 49
    assert im.bridge_length(2.0, 0, "seconds", 30)[0] == 57   # 60 -> 57
    assert im.bridge_length(0, 33, "frames", 24)[0] == 33
    assert im.bridge_length(0, 36, "frames", 24)[0] == 33     # 36 nearer 33 than 41
    assert im.bridge_length(0, 37, "frames", 24)[0] == 41     # tie goes up
    assert im.bridge_length(0.0, 0, "seconds", 24)[0] == 9    # floor


def test_precedence():
    assert im.resolve_join(50, None) == (50, None)
    assert im.resolve_join(50, 50) == (50, None)
    j, note = im.resolve_join(50, 72)
    assert j == 72 and "72" in note and "50" in note
    assert im.resolve_join(0, 72)[0] == 72   # wired wins even over the default 0


@pytest.fixture(scope="module")
def loader_mod(tmp_path_factory):
    import conftest  # noqa: F401  (stubs)
    d = os.path.abspath(VIDS)
    sys.modules["folder_paths"].get_input_directory = lambda: d
    import loader
    return loader


def _decode_independently(path, indices, fps):
    """Output frame i at fps = the first source frame at or after time i/fps (the
    loader's own resampling rule, re-implemented here from a plain decode)."""
    import av
    import numpy as np
    out = {}
    with av.open(path) as c:
        vs = c.streams.video[0]
        for f in c.decode(vs):
            for i in indices:
                if i not in out and float(f.time) >= i / fps - 1e-6:
                    out[i] = f.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            if len(out) == len(indices):
                break
    return out


def test_insert_anchors_match_independent_decode(loader_mod):
    import numpy as np
    path = os.path.join(VIDS, "4.mp4")
    if not os.path.exists(path):
        pytest.skip(f"missing {path}")
    node = loader_mod.SeamStitchLoader()
    join = 40
    res = node.load_video(path, 24, "seconds", 0.0, 0.0, 2.0, 0, 0, 0, snap_to_multiple=0,
                          mode="insert at join", join_frame=join, trim_each_side=0)
    images, audio, dur, count, _, first, last, _, sf, ef, fr, w, h, full, insert = res
    assert insert is True and images.shape[0] == 2
    assert (sf, ef) == (join, join - 1)
    assert count == 49 and dur == pytest.approx(49 / 24)
    ref = _decode_independently(path, {join - 1, join}, 24)
    for got, idx in ((first, join - 1), (last, join)):
        diff = np.abs(got[0].numpy() - ref[idx]).max()
        assert diff < 3 / 255.0, (idx, diff)   # only YUV->RGB matrix rounding may differ
    assert audio["waveform"].shape[-1] < 4096
    assert full["waveform"].shape[-1] > 4096


def test_insert_out_of_range_names_clip_length(loader_mod):
    path = os.path.join(VIDS, "4.mp4")
    if not os.path.exists(path):
        pytest.skip(f"missing {path}")
    node = loader_mod.SeamStitchLoader()
    with pytest.raises(ValueError, match="frames"):
        node.load_video(path, 24, "seconds", 0.0, 0.0, 2.0, 0, 0, 0, mode="insert at join",
                        join_frame=0, trim_each_side=0)
