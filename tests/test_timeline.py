"""SeamStitchTimeline: sequence math, the assembled cut, and the Loader hand-off.

Clips are synthetic and frame-coded: frame i of a clip is a flat grey whose level
encodes i, written losslessly, so every check below is about WHICH frames came
out, not roughly-similar pictures."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import timeline_math as tm  # noqa: E402

FFMPEG = None


def _ffmpeg():
    global FFMPEG
    if FFMPEG is None:
        try:
            import imageio_ffmpeg
            FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            FFMPEG = "ffmpeg"
    return FFMPEG


LEVEL_STEP = 4


def make_clip(path, n, fps, w=64, h=48, offset=0, audio=True):
    """n frames at fps; frame i is grey level 16 + (offset + i) * LEVEL_STEP."""
    frames = np.stack([np.full((h, w, 3), 16 + (offset + i) * LEVEL_STEP, np.uint8) for i in range(n)])
    cmd = [_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=32000:duration={n / fps}",
                "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", path]
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    return path


def codes(frames):
    return [int(round((float(f.mean()) - 16) / LEVEL_STEP)) for f in frames]


# ---------------------------------------------------------------- pure math

def test_parse_and_format_roundtrip():
    text = "a.mp4\nb clip.mp4 @ 12..119\n# note\n~ 40\nc.mp4 @ ..50\nd.mp4 @ 7.."
    e = tm.parse_sequence(text)
    assert [x["kind"] for x in e] == ["clip", "clip", "gap", "clip", "clip"]
    assert e[1] == {"kind": "clip", "path": "b clip.mp4", "enter": 12, "exit": 119}
    assert e[3]["enter"] == 0 and e[3]["exit"] == 50
    assert e[4]["enter"] == 7 and e[4]["exit"] is None
    assert tm.parse_sequence(tm.format_sequence(e)) == e


@pytest.mark.parametrize("bad", ["~ x", "~ 0", "a.mp4 @ 5", "a.mp4 @ 9..3", "a.mp4 @ a..b"])
def test_parse_rejects(bad):
    with pytest.raises(tm.SequenceError):
        tm.parse_sequence(bad)


def test_resolve_cut_positions_and_gap():
    e = tm.parse_sequence("a @ 10..30\n~ 12\nb\nc @ ..5")
    cut = tm.resolve_cut(e, {"a": 100, "b": 40, "c": 9}.__getitem__)
    assert [(p["cut_start"], p["enter"], p["exit"]) for p in cut["pieces"]] == [(0, 10, 30), (20, 0, 40), (60, 0, 5)]
    assert cut["frames"] == 65
    assert cut["gaps"] == [{"frames": 12, "cut_pos": 20, "index": 1}]
    assert tm.seam_positions(cut) == [20, 60]


def test_resolve_cut_clamps_exit_and_rejects_empty():
    cut = tm.resolve_cut(tm.parse_sequence("a @ 3..500"), {"a": 50}.get)
    assert cut["pieces"][0]["exit"] == 50 and cut["frames"] == 47
    with pytest.raises(tm.SequenceError):
        tm.resolve_cut(tm.parse_sequence("a @ 60.."), {"a": 50}.get)


def test_targets():
    cut = tm.resolve_cut(tm.parse_sequence("a\nb"), {"a": 30, "b": 30}.get)
    p = tm.resolve_target({"mode": "replace", "start": 25, "end": 34}, cut)
    assert (p["mode"], p["start"], p["end"]) == ("replace range", 25, 34)
    for bad in ({"mode": "replace", "start": 0, "end": 4}, {"mode": "replace", "start": 50, "end": 59},
                {"mode": "gap"}, {}):
        with pytest.raises(tm.SequenceError):
            tm.resolve_target(bad, cut)
    gcut = tm.resolve_cut(tm.parse_sequence("a\n~ 20\nb"), {"a": 30, "b": 30}.get)
    # older saves: trim N = markers N frames either side -> replace across the join + the gap
    g = tm.resolve_target({"mode": "gap", "trim": 2}, gcut)
    assert g == {"mode": "replace range", "start": 28, "end": 31, "join": 30, "trim": 0, "length": 4, "extra": 20}
    # markers around the gap, asymmetric
    g = tm.resolve_target({"mode": "gap", "start": 22, "end": 33}, gcut)
    assert (g["mode"], g["start"], g["end"], g["extra"]) == ("replace range", 22, 33, 20)
    assert tm.generator_frames(g, tm.GRID_LTX) == 33            # 12 + 20 = 32 -> 8k+1
    # nothing marked: pure insert, the gap is the number of new frames
    g = tm.resolve_target({"mode": "gap", "start": 30, "end": 29}, gcut)
    assert (g["mode"], g["length"]) == ("insert at join", 22)
    assert tm.resolve_target({"mode": "gap"}, gcut)["mode"] == "insert at join"
    for bad in ({"mode": "gap", "start": 31, "end": 35}, {"mode": "gap", "start": 25, "end": 28},
                {"mode": "gap", "start": 0, "end": 33}):
        with pytest.raises(tm.SequenceError):
            tm.resolve_target(bad, gcut)
    with pytest.raises(tm.SequenceError):
        tm.resolve_target({"mode": "replace", "start": 5, "end": 9}, gcut)
    two = tm.resolve_cut(tm.parse_sequence("a\n~ 3\nb\n~ 4\na"), {"a": 30, "b": 30}.get)
    with pytest.raises(tm.SequenceError):
        tm.resolve_target({"mode": "gap"}, two)


def test_generator_frames_matches_loader_rules():
    from loader import SeamStitchLoader
    import insert_math
    for grid in (tm.GRID_LTX, tm.GRID_MINIMAX, tm.GRID_NONE):
        for gap in (1, 5, 17, 23, 40):
            for k in (0, 3):
                for ext in (0, 7):
                    plan = {"mode": "replace range", "start": 10, "end": 10 + gap - 1}
                    want = SeamStitchLoader._extended_frame_count(gap + 2 * k, 24, True, ext, "frames", grid)
                    assert tm.generator_frames(plan, grid, k, ext) == want
            plan = {"mode": "insert at join", "length": gap}
            assert tm.generator_frames(plan, grid) == insert_math.snap_to_grid(gap, grid)


# ---------------------------------------------------------------- real files

@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    import folder_paths
    inp, outp = tmp_path / "input", tmp_path / "output"
    inp.mkdir()
    outp.mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(inp), raising=False)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(outp), raising=False)
    return inp, outp


def test_probe_counts_frames_at_rate(dirs):
    import timeline as tl
    inp, _ = dirs
    a = make_clip(str(inp / "a.mp4"), 48, 48)
    assert tl.probe(a, 48)["frames"] == 48
    assert tl.probe(a, 24)["frames"] == 24
    assert tl.probe(a, 0)["frames"] == 48 and tl.probe(a, 0)["frame_rate"] == 48


@pytest.fixture(scope="module")
def recombine(tmp_path_factory):
    """recombine.py laid out beside an empty VHS folder, as tests/test_recombine_insert.py
    does (conftest stubs VHS itself); encoding is stubbed so the splice is compared as tensors."""
    import importlib
    import shutil
    root = tmp_path_factory.mktemp("custom_nodes")
    os.makedirs(str(root / "comfyui-videohelpersuite" / "videohelpersuite"))
    pkg = root / "ss_tl"
    pkg.mkdir()
    open(str(pkg / "__init__.py"), "w").close()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("recombine.py", "audio_splice.py", "video_colour.py", "bridge_match.py"):
        shutil.copyfile(os.path.join(here, name), str(pkg / name))
    sys.path.insert(0, str(root))
    mod = importlib.import_module("ss_tl.recombine")
    mod._encode_video = lambda images, *a, **k: {"ui": {"gifs": []}, "result": ((False, []),)}
    yield mod
    sys.path.remove(str(root))


def test_iter_frames_matches_recombine_decode(dirs, recombine):
    """The cut is only exact if it decodes on Recombine's own timeline."""
    import timeline as tl
    inp, _ = dirs
    a = make_clip(str(inp / "a.mp4"), 48, 48)
    for fr, s, e in ((48, 0, 48), (48, 7, 30), (24, 3, 20), (30, 0, None)):
        mine = np.stack(list(tl._iter_frames(a, fr, s, e)))
        theirs = recombine._decode_range(a, fr, s, e).numpy()
        assert mine.shape == theirs.shape and (mine == theirs).all(), (fr, s, e)


def test_passthrough_single_clip(dirs):
    import timeline as tl
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 24)
    cut, fr = tl.plan_cut("a.mp4", 0)
    assert fr == 24
    assert tl.build_cut(cut, fr) == str(inp / "a.mp4")
    cut, fr = tl.plan_cut("a.mp4 @ 2..", 0)
    assert tl.passthrough_path(cut, fr) is None


def test_build_cut_is_frame_exact(dirs):
    import timeline as tl
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 24, offset=0)
    make_clip(str(inp / "b.mp4"), 20, 24, offset=30)
    cut, fr = tl.plan_cut("a.mp4 @ 5..25\nb.mp4 @ ..12", 0)
    out = tl.build_cut(cut, fr, crf=0)
    assert out.startswith(str(inp)) and os.path.basename(os.path.dirname(out)) == tl.CUT_SUBDIR
    got = codes(tl._iter_frames(out, fr, 0, None))
    assert got == list(range(5, 25)) + list(range(30, 42))
    info = tl.probe(out, fr)
    assert info["frames"] == 32 and abs(info["base_time"]) < 1e-6
    # audio covers exactly the picture
    import av
    with av.open(out) as c:
        a = c.streams.audio[0]
        n = sum(f.samples for f in c.decode(a))
    assert abs(n / a.rate - 32 / 24) < 0.05
    # cached: same inputs, same file, not rebuilt
    mtime = os.stat(out).st_mtime_ns
    assert tl.build_cut(cut, fr, crf=0) == out and os.stat(out).st_mtime_ns == mtime


def test_build_cut_resamples_rate_and_fits_size(dirs):
    import timeline as tl
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 24, 24, offset=0)
    make_clip(str(inp / "b.mp4"), 48, 48, w=96, h=48, offset=0, audio=False)
    cut, fr = tl.plan_cut("a.mp4\nb.mp4", 24)
    out = tl.build_cut(cut, fr, crf=0)
    info = tl.probe(out, 24)
    assert (info["width"], info["height"]) == (64, 48)
    got = codes(tl._iter_frames(out, 24, 0, None))
    assert got[:24] == list(range(24))
    assert got[24:] == list(range(0, 48, 2))       # 48 -> 24 keeps every other frame


def test_timeline_node_outputs_match_loader(dirs):
    import timeline as tl
    from loader import SeamStitchLoader
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 24, offset=0)
    make_clip(str(inp / "b.mp4"), 30, 24, offset=30)
    node = tl.SeamStitchTimeline()
    seq = "a.mp4\nb.mp4"
    args = dict(frame_rate=0, bridge_frame_grid=tm.GRID_LTX, context_frames=0, extend_frames=0,
                snap_to_multiple=0, mismatch_fit="crop", assemble_crf=0)
    res = node.run(seq, json.dumps({"mode": "replace", "start": 26, "end": 33}), **args)
    # the Loader's 18, then end_seconds / picture_timing, then the conform's two - appended,
    # so every saved wire keeps its output slot
    n = len(SeamStitchLoader.RETURN_NAMES)
    assert len(res) == n + 4
    assert tl.SeamStitchTimeline.RETURN_NAMES[n:n + 2] == ("end_seconds", "picture_timing")
    assert res[n] == round((res[3] - 1) / 24, 2)                     # 9 frames at 24 fps -> 0.33 s
    assert "Picture 2 (from Shot 1) aligns with the 0.33-second mark" in res[n + 1]
    assert res[n + 2] == 24                                          # source_frame_rate: no conform
    names = dict(zip(SeamStitchLoader.RETURN_NAMES, res))
    assert (names["start_frame"], names["end_frame"]) == (26, 33)
    assert names["frame_count"] == 9                       # 8 frames -> 8k+1 rounds up
    assert codes(names["images"].numpy() * 255) == list(range(26, 34))
    assert codes(names["first_frame"].numpy() * 255) == [26]
    assert codes(names["last_frame"].numpy() * 255) == [33]
    assert os.path.isfile(names["source_video_path"])

    res = node.run(seq, json.dumps({"mode": "replace", "start": 26, "end": 33}),
                   **dict(args, context_frames=3))
    names = dict(zip(SeamStitchLoader.RETURN_NAMES, res))
    assert codes(names["start_context"].numpy() * 255) == [23, 24, 25]
    assert codes(names["end_context"].numpy() * 255) == [34, 35, 36]
    assert names["context_frames"] == 3 and names["frame_count"] == 17   # 8 + 6 -> 17

    res = node.run("a.mp4\n~ 20\nb.mp4", json.dumps({"mode": "gap", "trim": 0}), **args)
    names = dict(zip(SeamStitchLoader.RETURN_NAMES, res))
    assert names["insert"] is True
    assert (names["start_frame"], names["end_frame"]) == (30, 29)
    assert codes(names["first_frame"].numpy() * 255) == [29]
    assert codes(names["last_frame"].numpy() * 255) == [30]
    assert names["frame_count"] == 25                      # 20 + 2 anchors -> nearest 8k+1


def test_identity_roundtrip_through_recombine(dirs, recombine):
    """Timeline -> (a 'generator' that returns the real frames) -> Recombine must give
    back the cut frame for frame: the whole chain agrees on every index."""
    import timeline as tl
    from loader import SeamStitchLoader
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 24, offset=0)
    make_clip(str(inp / "b.mp4"), 30, 24, offset=30)
    node = tl.SeamStitchTimeline()
    args = dict(frame_rate=0, bridge_frame_grid=tm.GRID_NONE, context_frames=2, extend_frames=0,
                snap_to_multiple=0, mismatch_fit="crop", assemble_crf=0)
    res = dict(zip(SeamStitchLoader.RETURN_NAMES,
                   node.run("a.mp4 @ 4..\nb.mp4 @ ..25", json.dumps({"mode": "replace", "start": 20, "end": 31}), **args)))
    import torch
    regenerated = torch.cat([res["start_context"], res["images"], res["end_context"]])
    out = recombine.SeamStitchRecombine().recombine(
        regenerated, res["source_video_path"], res["start_frame"], res["end_frame"], res["frame_rate"],
        0.0, 0, "t", "video/h264-mp4", save_output=False, context_frames=res["context_frames"])
    combined = out["result"][1]
    assert codes(combined.numpy() * 255) == list(range(4, 30)) + list(range(30, 55))


@pytest.mark.parametrize("fr,s,e", [(48, 10, 20), (48, 20, 30), (48, 0, 48), (24, 5, 17)])
def test_loader_replace_decode_is_index_exact(dirs, fr, s, e):
    """Regression: the Loader's replace-mode sampler accumulated frame_interval and
    drifted - 10..19 came back 10, 12, 12, ... and 20..29 lost its last frame."""
    from loader import SeamStitchLoader
    inp, _ = dirs
    a = make_clip(str(inp / "a.mp4"), 48, 48)
    r = SeamStitchLoader().load_video(video=a, frame_rate=fr, display_mode="frames", start_time=0,
                                      end_time=0, duration=0, start_frame=s, end_frame=e,
                                      duration_frames=0, snap_to_multiple=0)
    step = 48 // fr
    assert codes(r[0].numpy() * 255) == list(range(s * step, e * step, step))
    assert (r[8], r[9]) == (s, e - 1)


def test_recombine_passes_format_settings_and_can_skip_encode(dirs, recombine):
    """crf/pix_fmt reached the encoder as an empty dict (format defaults, crf 19, always);
    skip_encode returns the splice without writing anything."""
    import torch
    import timeline as tl
    from loader import SeamStitchLoader
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 24, offset=0)
    res = dict(zip(SeamStitchLoader.RETURN_NAMES, tl.SeamStitchTimeline().run(
        "a.mp4", json.dumps({"mode": "replace", "start": 10, "end": 14}), 0, tm.GRID_NONE, 0, 0, 0, "crop", 0)))
    seen = {}
    real = recombine._encode_video

    def spy(images, *a, **k):
        seen.update(k)
        return {"ui": {"gifs": []}, "result": ((False, []),)}
    recombine._encode_video = spy
    try:
        node = recombine.SeamStitchRecombine()
        args = (res["images"], res["source_video_path"], 10, 14, 24, 0.0, 0, "t", "video/h264-mp4")
        node.recombine(*args, save_output=False, crf=12, pix_fmt="yuv420p10le")
        assert seen["format_kwargs"] == {"crf": 12, "pix_fmt": "yuv420p10le"}
        seen.clear()
        out = node.recombine(*args, save_output=False, skip_encode=True)
        assert not seen and out["result"][0] == (False, []) and out["result"][1].shape[0] == 30
    finally:
        recombine._encode_video = real


def test_rgb_mkv_decodes_exactly(dirs, recombine):
    """FFV1 in MKV: RGB planes (no YUV matrix) and millisecond timestamps (a 48 fps frame
    due at 83.333 ms is stored at 83 ms). Both used to shift or scramble frames."""
    from loader import SeamStitchLoader
    import timeline as tl
    inp, _ = dirs
    n, w, h = 40, 64, 48
    frames = np.stack([np.full((h, w, 3), 16 + i * LEVEL_STEP, np.uint8) for i in range(n)])
    frames[:, :, :, 0] = 200                       # a colour, so a YUV mix-up shows
    path = str(inp / "m.mkv")
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-r", "48", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "gbrp", path],
                   input=frames.tobytes(), check=True)
    assert tl.probe(path, 48)["frames"] == n
    a = np.stack(list(tl._iter_frames(path, 48, 5, 30)))
    b = recombine._decode_range(path, 48, 5, 30).numpy()
    c = (SeamStitchLoader().load_video(video=path, frame_rate=48, display_mode="frames", start_time=0,
                                       end_time=0, duration=0, start_frame=5, end_frame=30,
                                       duration_frames=0, snap_to_multiple=0)[0].numpy() * 255).round()
    for got in (a, b, c):
        assert np.abs(got.astype(int) - frames[5:30].astype(int)).max() <= 1



def test_gap_with_markers_is_an_extended_replace(dirs, recombine):
    """I/O around a gap: the marked frames are replaced and the generator gets the gap's
    frames on top (Loader extend_bridge); Recombine makes the video that much longer."""
    import timeline as tl
    from loader import SeamStitchLoader
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 24, offset=0)
    make_clip(str(inp / "b.mp4"), 30, 24, offset=30)
    res = dict(zip(SeamStitchLoader.RETURN_NAMES, tl.SeamStitchTimeline().run(
        "a.mp4\n~ 12\nb.mp4", json.dumps({"mode": "gap", "start": 26, "end": 33}),
        0, tm.GRID_NONE, 2, 0, 0, "crop", 0)))
    assert res["insert"] is False
    assert (res["start_frame"], res["end_frame"]) == (26, 33)
    assert res["frame_count"] == 8 + 12 + 2 * 2                  # range + gap + context
    assert codes(res["start_context"].numpy() * 255) == [24, 25]
    import torch
    gen = torch.cat([res["start_context"]] + [res["images"][:1]] * (res["frame_count"] - 4) + [res["end_context"]])
    out = recombine.SeamStitchRecombine().recombine(
        gen, res["source_video_path"], 26, 33, 24, 0.0, 0, "t", "video/h264-mp4", save_output=False,
        context_frames=2, skip_encode=True)
    assert out["result"][1].shape[0] == 60 + 12                   # 12 frames longer


@pytest.mark.parametrize("fr", [24, 48, 16])
def test_loader_matches_recombine_on_a_late_starting_clip(dirs, recombine, fr):
    """Regression (found 2026-09-29 on seamstitch_combined_00005 / None_00005, 48 fps, video
    starting +31 ms, read at 24): the Loader timed frame i as i/fr, Recombine and the
    Timeline as start + i/fr, so the Loader's range and anchors were up to a frame early
    against the frames Recombine cut out. Same file, same rate: same frames, same indices."""
    from loader import SeamStitchLoader
    inp, _ = dirs
    n = 60
    frames = np.stack([np.full((48, 64, 3), 16 + i * LEVEL_STEP, np.uint8) for i in range(n)])
    path = str(inp / "late.mp4")
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x48",
                    "-r", "48", "-i", "-", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
                    "-output_ts_offset", "0.031", path], input=frames.tobytes(), check=True)
    import av
    with av.open(path) as c:
        v = c.streams.video[0]
        assert float(v.start_time * v.time_base) == pytest.approx(0.031, abs=1e-3)   # the real files' offset
    k = 2
    for s, e in ((5, 12), (10, 20), (3, 9)):
        if e + k > 60 * fr // 48:
            continue
        r = SeamStitchLoader().load_video(video=path, frame_rate=fr, display_mode="frames", start_time=0,
                                          end_time=0, duration=0, start_frame=s, end_frame=e,
                                          duration_frames=0, snap_to_multiple=0, context_frames=k)
        want = codes(recombine._decode_range(path, fr, s, e).numpy())
        assert codes(r[0].numpy() * 255) == want, (fr, s, e)
        assert codes(tl_frames(path, fr, s, e)) == want
        assert (r[8], r[9]) == (s, e - 1)
        assert codes(r[5].numpy() * 255) == want[:1] and codes(r[6].numpy() * 255) == want[-1:]
        assert codes(r[15].numpy() * 255) == codes(recombine._decode_range(path, fr, s - k, s).numpy())
        assert codes(r[16].numpy() * 255) == codes(recombine._decode_range(path, fr, e, e + k).numpy())


def tl_frames(path, fr, s, e):
    import timeline as tl
    return list(tl._iter_frames(path, fr, s, e))
