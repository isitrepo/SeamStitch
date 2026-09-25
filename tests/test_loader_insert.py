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


def test_insert_bridge_length_follows_bridge_frame_grid():
    # MiniMax H3: nearest 17n+5, floored at 5, ties go up.
    for frames in range(0, 120):
        n, _ = im.bridge_length(0, frames, "frames", 24, im.GRID_MINIMAX)
        assert n >= 5 and (n - 5) % 17 == 0
    assert im.bridge_length(0, 48, "frames", 24, im.GRID_MINIMAX)[0] == 56   # nearer 56 than 39
    assert im.bridge_length(0, 30, "frames", 24, im.GRID_MINIMAX)[0] == 22   # nearer 22 than 39
    assert im.bridge_length(0, 31, "frames", 24, im.GRID_MINIMAX)[0] == 39   # nearer 39 than 22
    assert im.bridge_length(0, 0, "frames", 24, im.GRID_MINIMAX)[0] == 5
    # none: exact, floored at 3 so Recombine's anchor drop leaves a frame.
    assert im.bridge_length(0, 36, "frames", 24, im.GRID_NONE)[0] == 36
    assert im.bridge_length(0, 1, "frames", 24, im.GRID_NONE)[0] == 3
    # The default and an explicit LTX grid agree.
    assert im.bridge_length(0, 36, "frames", 24, im.GRID_LTX)[0] == im.bridge_length(0, 36, "frames", 24)[0] == 33


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
    images, audio, dur, count, _, first, last, _, sf, ef, fr, w, h, full, insert = res[:15]
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


# --- Regression: a stream that does not start at t=0 (found 2026-09-19, S5) ---

OFFSET_TICKS = 381   # what SeamStitchCombine's concat leaves on 48 fps output:
                     # 381/12288 s = 31.006 ms = 1.49 frames at 48 fps


@pytest.fixture(scope="module")
def offset_clip(tmp_path_factory):
    """A short remux of 4.mp4 with every video PTS shifted forward, so the stream's
    first frame sits at t = OFFSET_TICKS/time_base instead of t = 0 - the shape
    SeamStitchCombine's own output has (AAC priming delay on the concat demuxer).
    Frame CONTENT is untouched, so decode-order index i here is decode-order
    index i of 4.mp4 and the two can be compared frame for frame."""
    import av
    src = os.path.join(VIDS, "4.mp4")
    if not os.path.exists(src):
        pytest.skip(f"missing {src}")
    out = str(tmp_path_factory.mktemp("offset") / "offset_4.mp4")
    with av.open(src) as ic, av.open(out, "w") as oc:
        istream = ic.streams.video[0]
        ostream = oc.add_stream_from_template(istream)
        ostream.time_base = istream.time_base
        shift = OFFSET_TICKS
        n = 0
        for packet in ic.demux(istream):
            if packet.dts is None:
                continue
            packet.stream = ostream
            packet.pts += shift
            packet.dts += shift
            oc.mux(packet)
            n += 1
            if n >= 120:
                break
    with av.open(out) as c:
        vs = c.streams.video[0]
        base = float(vs.start_time * vs.time_base)
    assert base > 0.02, f"fixture did not get a start offset (got {base})"
    return out, base


def test_offset_fixture_really_is_offset(offset_clip):
    """Guard the guard: if this file ever gets written at t=0 the test below stops
    testing anything."""
    _, base = offset_clip
    assert base == pytest.approx(OFFSET_TICKS / 12288.0, abs=1e-6)


def test_insert_anchors_are_decode_order_on_an_offset_stream(offset_clip, loader_mod):
    """The anchors must be decode-order frames join-1 and join - the convention
    SeamStitchCombine measures seam_frame in and SeamStitchRecombine cuts on.

    Before the fix the loader asked for them as a bare index/fps absolute time, so
    a +31 ms start offset (1.49 frames at 48 fps) returned decode frames join-2 and
    join-1: BOTH anchors were clip A's last frame, the generator was handed a frame
    to morph into itself, and it rendered a freeze with the cut still in it.
    """
    import av
    import numpy as np
    path, _ = offset_clip
    node = loader_mod.SeamStitchLoader()
    join = 40
    res = node.load_video(path, 48, "frames", 0.0, 0.0, 0.0, 0, 0, 33, snap_to_multiple=0,
                          mode="insert at join", join_frame=join, trim_each_side=0)
    images, _, _, count, _, first, last, _, sf, ef, _, _, _, _, insert = res[:15]
    assert insert is True and (sf, ef) == (join, join - 1) and count == 33

    # Decode order, straight off the container - no timestamps involved at all.
    want = {}
    with av.open(path) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i in (join - 1, join):
                want[i] = f.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            if i > join:
                break

    for got, idx in ((first, join - 1), (last, join)):
        diff = np.abs(got[0].numpy() - want[idx]).max()
        assert diff < 3 / 255.0, f"anchor for decode frame {idx} is off (max diff {diff})"

    # And the two anchors must actually be different frames, which is the symptom
    # that reached the render: a freeze is what you get when they are the same.
    assert np.abs(first[0].numpy() - last[0].numpy()).mean() > 0.0


# --- Motion guides: context_frames (replace mode) ---

def _decode_order(path, indices):
    import av
    import numpy as np
    want = {}
    with av.open(path) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i in indices:
                want[i] = f.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            if i > max(indices):
                break
    return want


@pytest.mark.parametrize("which", ["plain", "offset"])
def test_context_frames_are_decode_order_either_side_of_the_range(which, offset_clip, loader_mod):
    """start_context = [sf-K, sf-1], end_context = [ef+1, ef+K], in decode order -
    the frames SeamStitchRecombine keeps either side of the cut - including on a
    stream that starts late, as SeamStitchCombine's output does."""
    import numpy as np
    path = offset_clip[0] if which == "offset" else os.path.join(VIDS, "4.mp4")
    node = loader_mod.SeamStitchLoader()
    k, sf, ef = 4, 30, 54                      # a 25-frame gap at 48 fps
    res = node.load_video(path, 48, "frames", 0.0, 0.0, 0.0, sf, ef + 1, 0, snap_to_multiple=0,
                          context_frames=k)
    assert len(res) == 18
    images, count, got_sf, got_ef, start_ctx, end_ctx, got_k = (res[0], res[3], res[8], res[9],
                                                                res[15], res[16], res[17])
    assert got_k == k and images.shape[0] == got_ef - got_sf + 1
    assert start_ctx.shape[0] == k and end_ctx.shape[0] == k
    gap = images.shape[0]
    assert count >= gap + 2 * k and (count - 1) % 8 == 0 and count - (gap + 2 * k) < 8
    want = _decode_order(path, set(range(got_sf - k, got_sf)) | set(range(got_ef + 1, got_ef + 1 + k)))
    for i in range(k):
        for got, idx in ((start_ctx[i], got_sf - k + i), (end_ctx[i], got_ef + 1 + i)):
            diff = np.abs(got.numpy() - want[idx]).max()
            assert diff < 3 / 255.0, f"context frame for decode index {idx} is off (max diff {diff})"


def test_context_zero_is_first_and_last_frame(loader_mod):
    import torch
    path = os.path.join(VIDS, "4.mp4")
    if not os.path.exists(path):
        pytest.skip(f"missing {path}")
    res = loader_mod.SeamStitchLoader().load_video(path, 24, "frames", 0.0, 0.0, 0.0, 10, 20, 0,
                                                   snap_to_multiple=0)
    assert res[17] == 0
    assert torch.equal(res[15], res[5]) and torch.equal(res[16], res[6])


def test_context_from_input_video_slices_by_index(loader_mod):
    import torch
    frames = torch.stack([torch.full((8, 8, 3), i / 100.0) for i in range(60)])
    res = loader_mod.SeamStitchLoader().load_video(
        "none", 24, "frames", 0.0, 0.0, 0.0, 20, 30, 0, snap_to_multiple=0,
        input_video=frames, context_frames=3)
    sf, ef = res[8], res[9]
    assert (sf, ef) == (20, 29)
    assert [round(float(f[0, 0, 0]) * 100) for f in res[15]] == [17, 18, 19]
    assert [round(float(f[0, 0, 0]) * 100) for f in res[16]] == [30, 31, 32]
    assert res[3] == 17          # 10 + 2*3 = 16 -> 17 on the LTX grid


def test_context_extend_and_grid(loader_mod):
    import torch
    frames = torch.zeros((200, 8, 8, 3))
    res = loader_mod.SeamStitchLoader().load_video(
        "none", 24, "frames", 0.0, 0.0, 0.0, 50, 75, 0, snap_to_multiple=0, input_video=frames,
        context_frames=4, extend_bridge=True, extend_amount=1.0, extend_unit="seconds")
    assert res[3] == 57          # 25 gap + 24 extension + 8 context = 57, already 8k+1
    assert res[2] == pytest.approx(57 / 24)


@pytest.mark.parametrize("sf,ef,k", [(2, 20, 3), (50, 58, 2)])
def test_context_out_of_range_names_clip_length(loader_mod, sf, ef, k):
    import torch
    frames = torch.zeros((60, 8, 8, 3))
    with pytest.raises(ValueError, match="60 frames"):
        loader_mod.SeamStitchLoader().load_video(
            "none", 24, "frames", 0.0, 0.0, 0.0, sf, ef + 1, 0, snap_to_multiple=0,
            input_video=frames, context_frames=k)


def test_context_refused_in_insert_mode(loader_mod):
    import torch
    with pytest.raises(ValueError, match="replace mode only"):
        loader_mod.SeamStitchLoader().load_video(
            "none", 24, "frames", 0.0, 0.0, 0.0, 0, 0, 33, input_video=torch.zeros((60, 8, 8, 3)),
            mode="insert at join", join_frame=30, context_frames=2)
