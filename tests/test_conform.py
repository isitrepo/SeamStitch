"""The 24 fps conform (SeamStitch Timeline conform_to_24fps) and its restore (SeamStitch
Result Preview restore_source_frame_rate).

MiniMax H3 takes frames 1:1 but times reference audio and Picture timings on a fixed
24 fps clock. The conform keeps every frame and re-labels the cut at 24 with the audio
slowed to match; the restore puts the source rate and original audio back. Every check
here is on frame-coded synthetic clips (see test_timeline.make_clip): which frames,
how many, at what rate, and how many audio samples."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import timeline_math as tm  # noqa: E402
from test_timeline import _ffmpeg, codes, make_clip, dirs, recombine  # noqa: E402,F401  (fixtures)

SR = 32000          # make_clip's audio rate


def _frames(path, fr):
    import timeline as tl
    return np.stack(list(tl._iter_frames(path, fr, 0, None)))


def _audio(path):
    """(samples per channel, sample rate, float32 [C, n]) of the file's first audio track."""
    import av
    with av.open(path) as c:
        a = c.streams.audio[0]
        rate = a.rate
        rs = av.AudioResampler(format="fltp", layout="stereo", rate=rate)   # planar float, own rate
        chunks = [r.to_ndarray() for f in c.decode(a) for r in rs.resample(f)]
        chunks += [r.to_ndarray() for r in rs.resample(None)]
    wav = np.concatenate(chunks, axis=1).astype(np.float32)
    return wav.shape[1], rate, wav


def _video_packets(path):
    import av
    with av.open(path) as c:
        return [bytes(p) for p in c.demux(c.streams.video[0]) if p.size]


def _native_fps(path):
    import av
    with av.open(path) as c:
        return float(c.streams.video[0].average_rate)


def _widget_names(input_types):
    """Widget order as the ComfyUI frontend builds it: required then optional, every
    input of a widget type that is not forceInput. widgets_values are restored by this
    position."""
    names = []
    for section in ("required", "optional"):
        for name, spec in input_types.get(section, {}).items():
            kind = spec[0]
            opts = spec[1] if len(spec) > 1 else {}
            if opts.get("forceInput"):
                continue
            if isinstance(kind, list) or kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                names.append(name)
    return names


def _args(**kw):
    a = dict(frame_rate=0, bridge_frame_grid=tm.GRID_NONE, context_frames=0, extend_frames=0,
             snap_to_multiple=0, mismatch_fit="crop", assemble_crf=0)
    a.update(kw)
    return a


# ---------------------------------------------------------------- maths


def test_conform_maths():
    import timeline as tl
    assert tl.conform_rate(25, True) == 24 and tl.conform_rate(30, True) == 24
    assert tl.conform_rate(24, True) == 24 and tl.conform_rate(25, False) == 25
    assert tl._atempo_chain(0.96) == "atempo=0.9600000000"
    assert tl._atempo_chain(0.4).split(",") == ["atempo=0.5000000000", "atempo=0.8000000000"]
    # 978 frames of 25 fps footage (the test clip): 39.12 s -> 40.75 s on H3's clock, frames kept
    assert 978 / 25 == pytest.approx(39.12) and 978 / 24 == pytest.approx(40.75)
    assert (978 / 24) / (978 / 25) == pytest.approx(25 / 24)


def test_time_stretch_is_exact_length_and_keeps_pitch():
    import timeline as tl
    t = np.arange(48000) / 48000.0
    tone = np.stack([np.sin(2 * np.pi * 440 * t)] * 2).astype(np.float32) * 0.5
    want = int(round(48000 / 0.96))                       # 1 s slowed to 24/25 -> 1.041667 s
    out = tl.time_stretch(tone, 48000, 0.96, want)
    assert out.shape == (2, want)
    spec = np.abs(np.fft.rfft(out[0]))
    peak = np.argmax(spec) * 48000 / want
    assert abs(peak - 440) < 3                             # pitch kept (a resample would be 422 Hz)
    # atempo leaves a few ms short at the tail at most; the rest is real sound, padded to length
    nz = np.nonzero(np.abs(out[0]) > 1e-4)[0]
    assert nz[-1] > want - 0.02 * 48000
    back = tl.time_stretch(out, 48000, 1 / 0.96, 48000)   # and the inverse lands on 1 s exactly
    assert back.shape == (2, 48000)


# ---------------------------------------------------------------- the conformed cut


def test_build_cut_conform_single_clip_is_a_lossless_relabel(dirs):
    """One untouched 25 fps clip: the video stream is copied (same packets, so the
    same pixels), labelled 24 fps, with the audio slowed to exactly frames/24."""
    import timeline as tl
    inp, _ = dirs
    src = make_clip(str(inp / "a.mp4"), 50, 25)
    cut, fr = tl.plan_cut("a.mp4", 0)
    assert fr == 25 and tl.build_cut(cut, fr) == src            # off: the source itself, as before
    out = tl.build_cut(cut, fr, conform=True)
    assert out != src and out.endswith(".mkv") and os.path.basename(os.path.dirname(out)) == tl.CUT_SUBDIR
    assert _video_packets(out) == _video_packets(src)           # bitstream copied, not re-encoded
    assert _native_fps(out) == pytest.approx(24, abs=0.01)
    info = tl.probe(out, 24)
    assert info["frames"] == 50 and abs(info["base_time"]) < 1e-3
    a, b = _frames(out, 24), _frames(src, 25)
    assert a.shape == b.shape and (a == b).all()                 # frame n is frame n
    assert codes(a) == list(range(50))
    n, rate, _ = _audio(out)
    assert rate == SR and n == int(round(50 / 24 * SR))         # exactly frames/24 s (FLAC)
    # cached by its own key: rebuilt neither on a second call nor confused with the unconformed cut
    mtime = os.stat(out).st_mtime_ns
    assert tl.build_cut(cut, fr, conform=True) == out and os.stat(out).st_mtime_ns == mtime


def test_build_cut_conform_strip_is_frame_exact(dirs):
    import timeline as tl
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 25, offset=0)
    make_clip(str(inp / "b.mp4"), 20, 25, offset=30)
    cut, fr = tl.plan_cut("a.mp4 @ 5..25\nb.mp4 @ ..12", 0)
    plain = tl.build_cut(cut, fr, crf=0)
    conf = tl.build_cut(cut, fr, crf=0, conform=True)
    assert conf != plain
    assert _native_fps(plain) == pytest.approx(25, abs=0.01) and _native_fps(conf) == pytest.approx(24, abs=0.01)
    a, b = _frames(conf, 24), _frames(plain, 25)
    assert a.shape == b.shape and (a == b).all()                 # lossless FFV1 either way, same frames
    assert codes(a) == list(range(5, 25)) + list(range(30, 42))
    assert tl.probe(conf, 24)["frames"] == 32
    assert _audio(conf)[0] == int(round(32 / 24 * SR))
    assert _audio(plain)[0] == int(round(32 / 25 * SR))
    # h264 cut codec conforms the same way
    h = tl.build_cut(cut, fr, crf=0, codec=tl.CODEC_H264, conform=True)
    assert tl.probe(h, 24)["frames"] == 32 and codes(_frames(h, 24)) == list(range(5, 25)) + list(range(30, 42))


def test_conform_off_and_24fps_sources_are_unchanged(dirs):
    import hashlib
    import timeline as tl
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 25, offset=0)
    make_clip(str(inp / "b.mp4"), 20, 25, offset=30)
    cut, fr = tl.plan_cut("a.mp4\nb.mp4", 0)
    # the unconformed cache key is the pre-conform key, byte for byte (cached cuts stay valid)
    h = hashlib.sha256()
    h.update(json.dumps([tl._ASSEMBLY_VERSION, fr, 12, "crop", tl.CODEC_LOSSLESS]).encode())
    for p in cut["pieces"]:
        st = os.stat(p["path"])
        h.update(json.dumps([os.path.abspath(p["path"]), st.st_mtime_ns, st.st_size, p["enter"], p["exit"]]).encode())
    old_key = h.hexdigest()[:16]
    assert tl._cut_key(cut, fr, 12, "crop") == old_key == tl._cut_key(cut, fr, 12, "crop", conform=False)
    assert tl._cut_key(cut, fr, 12, "crop", conform=True) != old_key
    # 24 fps sources: the conform is a no-op - same passthrough, same key, same file
    make_clip(str(inp / "c.mp4"), 30, 24)
    make_clip(str(inp / "d.mp4"), 30, 24, offset=30)
    c1, f24 = tl.plan_cut("c.mp4", 0)
    assert tl.build_cut(c1, f24, conform=True) == str(inp / "c.mp4")
    c2, f24 = tl.plan_cut("c.mp4\nd.mp4", 0)
    assert tl._cut_key(c2, f24, 0, "crop", conform=True) == tl._cut_key(c2, f24, 0, "crop")
    assert tl.build_cut(c2, f24, crf=0, conform=True) == tl.build_cut(c2, f24, crf=0)


@pytest.mark.parametrize("first,second", [(".mkv", ".mkv"), (".mkv", ".mp4"), (".mp4", ".mkv"),
                                          (".mp4", ".mp4"), (".mov", ".mov")])
def test_relabel_video_is_frame_exact_and_labelled(dirs, first, second):
    """25 -> 24 -> 25 by stream copy through any pair of containers: every frame lands on
    its own index and the header says the new rate. (Scaling timestamps instead put MKV's
    millisecond-rounded frames ~1 ms early at 25 fps and read frame 26 as 27.)"""
    import timeline as tl
    inp, _ = dirs
    src = make_clip(str(inp / "a.mp4"), 50, 25, audio=False)
    ref = _frames(src, 25)
    a = tl.relabel_video(src, str(inp / f"a24{first}"), 25, 24)
    b = tl.relabel_video(a, str(inp / f"a24_25{second}"), 24, 25)
    for path, fr in ((a, 24), (b, 25)):
        assert _native_fps(path) == pytest.approx(fr, abs=1e-6), path
        got = _frames(path, fr)
        assert got.shape == ref.shape and (got == ref).all(), path
        assert _video_packets(path) == _video_packets(src)


# ---------------------------------------------------------------- the node


def test_timeline_node_conform_outputs(dirs):
    """Frame numbers are unchanged by the conform (markers, range, context and grid rounding
    are frame counts); only the clock is H3's: frame_rate, duration, end_seconds and
    picture_timing come out at 24, source_frame_rate and original_audio at the source's."""
    import timeline as tl
    from loader import SeamStitchLoader
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 25, offset=0)        # frame codes stay < 60 (8-bit grey)
    make_clip(str(inp / "b.mp4"), 28, 25, offset=30)
    names = list(tl.SeamStitchTimeline.RETURN_NAMES)
    assert names[:len(SeamStitchLoader.RETURN_NAMES)] == list(SeamStitchLoader.RETURN_NAMES)
    assert names[-4:] == ["end_seconds", "picture_timing", "source_frame_rate", "original_audio"]
    node = tl.SeamStitchTimeline()
    seq, tgt = "a.mp4\nb.mp4", json.dumps({"mode": "replace", "start": 20, "end": 39})
    for grid, ctx in ((tm.GRID_NONE, 0), (tm.GRID_MINIMAX, 5), (tm.GRID_LTX, 3)):
        off = dict(zip(names, node.run(seq, tgt, **_args(bridge_frame_grid=grid, context_frames=ctx))))
        on = dict(zip(names, node.run(seq, tgt, **_args(bridge_frame_grid=grid, context_frames=ctx),
                                      conform_to_24fps=True)))
        for k in ("start_frame", "end_frame", "frame_count", "context_frames", "insert", "width", "height"):
            assert on[k] == off[k], k
        for k in ("images", "first_frame", "last_frame", "start_context", "end_context"):
            assert torch.equal(on[k], off[k]), k
        assert (off["frame_rate"], off["source_frame_rate"]) == (25, 25)
        assert (on["frame_rate"], on["source_frame_rate"]) == (24, 25)
        assert on["source_video_path"] != off["source_video_path"]
        fc = on["frame_count"]
        assert on["end_seconds"] == round((fc - 1) / 24, 2) and off["end_seconds"] == round((fc - 1) / 25, 2)
        assert f"aligns with the {on['end_seconds']:.2f}-second mark" in on["picture_timing"]
        assert on["duration"] == pytest.approx(fc / 24)
    assert codes(on["images"].numpy() * 255) == list(range(20, 40))
    # sound: original_audio is the cut's own, unstretched, exactly 58/25 s; everything the
    # Loader reads off the conformed cut is the slowed copy, 58/24 s
    o = on["original_audio"]
    assert o["sample_rate"] == SR and o["waveform"].shape[-1] == int(round(58 / 25 * SR))
    assert on["full_clip_audio"]["waveform"].shape[-1] == pytest.approx(58 / 24 * SR, abs=SR * 0.03)
    assert off["original_audio"] is off["full_clip_audio"]
    # the conform flag reaches the change detection
    assert tl.SeamStitchTimeline.IS_CHANGED(seq, tgt, 0, conform_to_24fps=True) != \
        tl.SeamStitchTimeline.IS_CHANGED(seq, tgt, 0, conform_to_24fps=False)


def test_timeline_node_conform_gap_insert(dirs):
    import timeline as tl
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 25, offset=0)
    make_clip(str(inp / "b.mp4"), 30, 25, offset=30)
    names = list(tl.SeamStitchTimeline.RETURN_NAMES)
    on = dict(zip(names, tl.SeamStitchTimeline().run("a.mp4\n~ 20\nb.mp4", json.dumps({"mode": "gap", "trim": 0}),
                                                     **_args(bridge_frame_grid=tm.GRID_LTX), conform_to_24fps=True)))
    assert on["insert"] is True and (on["start_frame"], on["end_frame"]) == (30, 29)
    assert codes(on["first_frame"].numpy() * 255) == [29] and codes(on["last_frame"].numpy() * 255) == [30]
    assert on["frame_count"] == 25 and on["frame_rate"] == 24


def test_recombine_splices_the_conformed_cut_unchanged(dirs, recombine):
    """Recombine needs no change: fed the Timeline's frame_rate (24) and the conformed cut,
    an identity 'generator' gives the cut back frame for frame, audio frames/24 long."""
    import timeline as tl
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 25, offset=0)
    make_clip(str(inp / "b.mp4"), 30, 25, offset=30)
    names = list(tl.SeamStitchTimeline.RETURN_NAMES)
    res = dict(zip(names, tl.SeamStitchTimeline().run(
        "a.mp4 @ 4..\nb.mp4 @ ..25", json.dumps({"mode": "replace", "start": 20, "end": 31}),
        **_args(context_frames=2), conform_to_24fps=True)))
    regenerated = torch.cat([res["start_context"], res["images"], res["end_context"]])
    out = recombine.SeamStitchRecombine().recombine(
        regenerated, res["source_video_path"], res["start_frame"], res["end_frame"], res["frame_rate"],
        0.0, 0, "t", "video/h264-mp4", save_output=False, context_frames=res["context_frames"], skip_encode=True)
    combined = out["result"][1]
    assert codes(combined.numpy() * 255) == list(range(4, 30)) + list(range(30, 55))
    # (its audio splice reads the cut's track through VHS, which is stubbed under pytest -
    # the real ComfyUI check covers that part)


# ---------------------------------------------------------------- the restore


def _tone(sr, n, hz, amp=0.3):
    t = np.arange(n) / sr
    return torch.from_numpy(np.stack([np.sin(2 * np.pi * hz * t)] * 2).astype(np.float32) * amp)


def test_restore_audio_keeps_the_original_outside_and_the_bridge_inside():
    import result_preview as rp
    sr, src_fr, clip_fr = 48000, 25, 24
    s_frames, start, end = 100, 40, 59                     # 20 frames replaced by 20 new ones
    original = {"waveform": _tone(sr, int(round(s_frames / src_fr * sr)), 440).unsqueeze(0), "sample_rate": sr}
    # Recombine's audio on the 24 fps clock: silence, with a 1 kHz bridge under the new frames
    spliced_w = torch.zeros((2, int(round(s_frames / clip_fr * sr))))
    a, b = round(start * sr / clip_fr), round((start + 20) * sr / clip_fr)
    spliced_w[:, a:b] = _tone(sr, b - a, 1000)
    spliced = {"waveform": spliced_w.unsqueeze(0), "sample_rate": sr}
    out, notes = rp.restore_audio(spliced, original, clip_fr, src_fr, start, end, s_frames, s_frames)
    w = out["waveform"][0]
    assert out["sample_rate"] == sr and w.shape[-1] == int(round(s_frames / src_fr * sr))   # source length
    fade = int(0.02 * sr)
    j0, j1 = round(start * sr / src_fr), round((end + 1) * sr / src_fr)
    o = original["waveform"][0]
    assert torch.equal(w[:, :j0], o[:, :j0])                          # original, sample for sample
    assert torch.equal(w[:, j1 + fade:], o[:, j1 + fade:])
    mid = w[0, j0 + fade:j1 - fade].numpy()
    peak = np.argmax(np.abs(np.fft.rfft(mid))) * sr / mid.size
    assert abs(peak - 1000) < 30                                      # the bridge, pitch kept
    # no original wired: the whole spliced track sped back up, still the source length
    out2, notes2 = rp.restore_audio(spliced, None, clip_fr, src_fr, start, end, s_frames, s_frames)
    assert out2["waveform"].shape[-1] == w.shape[-1] and notes2


def _fake_encode(captured):
    """rp._encode stand-in that writes a real file (FFV1 + FLAC) at the rate it is given."""
    def enc(images, audio, fr, prefix, fmt, save_output, settings, prompt, extra):
        import folder_paths
        out = os.path.join(folder_paths.get_output_directory(), f"{prefix}.mkv")
        u8 = (images.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        n, h, w = u8.shape[:3]
        cmd = [_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(fr), "-i", "-"]
        apath = out + ".f32"
        if audio is not None:
            audio["waveform"][0].numpy().T.astype("<f4").tofile(apath)
            cmd += ["-f", "f32le", "-ar", str(audio["sample_rate"]), "-ac", "2", "-i", apath, "-c:a", "flac"]
        subprocess.run(cmd + ["-c:v", "ffv1", "-pix_fmt", "gbrp", out], input=u8.tobytes(), check=True)
        captured.update(fr=fr, audio=audio, path=out)
        return (save_output, [out])
    return enc


def test_restore_round_trip_images(dirs, monkeypatch):
    """Timeline (conform on) -> identity splice -> Result Preview (restore on): the saved
    video is the source's frames at the source's 25 fps, exactly as long, original audio."""
    import timeline as tl
    import result_preview as rp
    import folder_paths
    inp, outp = dirs
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(inp), raising=False)
    src = make_clip(str(inp / "a.mp4"), 50, 25)
    names = list(tl.SeamStitchTimeline.RETURN_NAMES)
    res = dict(zip(names, tl.SeamStitchTimeline().run(
        "a.mp4", json.dumps({"mode": "replace", "start": 20, "end": 29}), **_args(), conform_to_24fps=True)))
    assert res["frame_rate"] == 24 and res["source_frame_rate"] == 25
    cut = res["source_video_path"]
    images = torch.from_numpy(_frames(cut, 24)).float() / 255.0           # what Recombine hands over
    w = tl._clip_audio(cut, 0.0, 50 / 24, SR)                              # Recombine's audio: the slowed track
    cap = {}
    monkeypatch.setattr(rp, "_encode", _fake_encode(cap))
    out = rp.SeamStitchResultPreview().preview(
        cut, 20, 29, 24, filename_prefix="restored", images=images,
        audio={"waveform": torch.from_numpy(w).unsqueeze(0), "sample_rate": SR},
        source_frame_rate=25, original_audio=res["original_audio"], restore_source_frame_rate=True)
    path = out["result"][0]
    assert cap["fr"] == 25 and path == cap["path"]
    assert _native_fps(path) == pytest.approx(25, abs=0.01) and tl.probe(path, 25)["frames"] == 50
    a, b = _frames(path, 25), _frames(src, 25)
    assert a.shape == b.shape and (a == b).all()                           # identical pixels
    n, rate, saved = _audio(path)
    assert rate == SR and n == int(round(50 / 25 * SR))                    # the source cut's length
    o = res["original_audio"]["waveform"][0].numpy()
    j0 = round(20 * SR / 25)
    assert np.abs(saved[:, :j0] - o[:, :j0]).max() < 1e-3                  # original audio (FLAC-quantised)
    ui = out["ui"]["seamstitch_result"][0]
    assert ui["frame_rate"] == 25 and ui["frames"] == 50 and ui["source_frames"] == 50
    assert (ui["inserted"], ui["removed"]) == (10, 10)
    # restore off: today's behaviour - saved at the cut's 24 fps with the slowed audio
    cap.clear()
    out = rp.SeamStitchResultPreview().preview(
        cut, 20, 29, 24, filename_prefix="as_is", images=images,
        audio={"waveform": torch.from_numpy(w).unsqueeze(0), "sample_rate": SR},
        source_frame_rate=25, original_audio=res["original_audio"], restore_source_frame_rate=False)
    assert cap["fr"] == 24 and cap["audio"]["waveform"].shape[-1] == w.shape[-1]
    assert out["ui"]["seamstitch_result"][0]["frame_rate"] == 24


def test_restore_round_trip_filenames_is_a_stream_copy(dirs, monkeypatch):
    """Given a file Recombine already wrote at 24 (here: the conformed cut itself), the
    restore re-labels it at 25 without re-encoding the picture."""
    import timeline as tl
    import result_preview as rp
    import folder_paths
    inp, _ = dirs
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(inp), raising=False)
    src = make_clip(str(inp / "a.mp4"), 50, 25)
    names = list(tl.SeamStitchTimeline.RETURN_NAMES)
    res = dict(zip(names, tl.SeamStitchTimeline().run(
        "a.mp4", json.dumps({"mode": "replace", "start": 20, "end": 29}), **_args(), conform_to_24fps=True)))
    cut = res["source_video_path"]
    out = rp.SeamStitchResultPreview().preview(
        cut, 20, 29, 24, filenames=(True, [cut]), source_frame_rate=25,
        original_audio=res["original_audio"])                              # restore defaults on
    path = out["result"][0]
    assert path.endswith("_25fps.mkv") and out["result"][1][1][-1] == path
    assert _video_packets(path) == _video_packets(src)                     # still the source's own bitstream
    assert _native_fps(path) == pytest.approx(25, abs=0.01)
    assert (_frames(path, 25) == _frames(src, 25)).all()
    n, rate, saved = _audio(path)
    assert n == int(round(50 / 25 * SR))
    o = res["original_audio"]["waveform"][0].numpy()
    assert np.abs(saved[:, :round(20 * SR / 25)] - o[:, :round(20 * SR / 25)]).max() < 1e-3
    assert out["ui"]["seamstitch_result"][0]["frame_rate"] == 25


def test_no_restore_when_rates_match(dirs, monkeypatch):
    """24 fps source (or no source_frame_rate wired): Result Preview behaves as before."""
    import timeline as tl
    import result_preview as rp
    import folder_paths
    inp, _ = dirs
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(inp), raising=False)
    make_clip(str(inp / "a.mp4"), 30, 24, offset=0)
    make_clip(str(inp / "b.mp4"), 30, 24, offset=20)
    cut, fr = tl.plan_cut("a.mp4\nb.mp4", 0)
    path = tl.build_cut(cut, fr, crf=0, conform=True)
    assert path == tl.build_cut(cut, fr, crf=0)
    for kw in ({}, {"source_frame_rate": 24, "restore_source_frame_rate": True}):
        out = rp.SeamStitchResultPreview().preview(path, 24, 35, 24, filenames=(True, [path]), **kw)
        assert out["result"][0] == path and out["ui"]["seamstitch_result"][0]["frame_rate"] == 24
    assert not [f for f in os.listdir(os.path.dirname(path)) if "fps" in f]


# ---------------------------------------------------------------- saved workflows


# widgets_values as saved in "Seamstitch - Timeline 3.json" (before the conform existed)
OLD_TIMELINE = ["clip.mp4\n~ 24\na.mp4 @ 62..", "{\"mode\":\"gap\",\"start\":220,\"end\":276}", 24,
                "minimax (17k+5)", 5, 48, 64, "crop", 12, "lossless (ffv1)"]
OLD_TIMELINE_NAMES = ["sequence", "target", "frame_rate", "bridge_frame_grid", "context_frames", "extend_frames",
                      "snap_to_multiple", "mismatch_fit", "assemble_crf", "cut_codec"]
OLD_PREVIEW = ["seamstitch_%date:yyyyMMdd_hhmmss%", "video/h264-mp4", 12, "yuv420p", True, True]
OLD_PREVIEW_NAMES = ["filename_prefix", "format", "crf", "pix_fmt", "save_metadata", "save_output"]


def test_old_workflows_widget_values_land_on_the_same_widgets():
    import timeline as tl
    import result_preview as rp
    t = _widget_names(tl.SeamStitchTimeline.INPUT_TYPES())
    assert t[:len(OLD_TIMELINE)] == OLD_TIMELINE_NAMES and t[len(OLD_TIMELINE):] == ["conform_to_24fps"]
    assert dict(zip(t, OLD_TIMELINE))["cut_codec"] == "lossless (ffv1)"
    assert tl.SeamStitchTimeline.INPUT_TYPES()["required"]["conform_to_24fps"][1]["default"] is False
    p = _widget_names(rp.SeamStitchResultPreview.INPUT_TYPES())
    assert p[:len(OLD_PREVIEW)] == OLD_PREVIEW_NAMES and p[len(OLD_PREVIEW):] == ["restore_source_frame_rate"]
    opt = rp.SeamStitchResultPreview.INPUT_TYPES()["optional"]
    assert list(opt)[:3] == ["images", "audio", "filenames"]                 # existing sockets unmoved
    assert opt["source_frame_rate"][1]["forceInput"] is True


def test_kays_saved_timeline_workflow_still_maps(request):
    """The real saved graph, when this machine has it: every SeamStitch Timeline / Result
    Preview node's widgets_values maps onto the same names as before."""
    import timeline as tl
    import result_preview as rp
    path = os.path.join("C:/AI/ComfyUI/Five/ComfyUI_windows_portable/ComfyUI/user/default/workflows",
                        "Seamstitch - Timeline 3.json")
    if not os.path.isfile(path):
        pytest.skip("Timeline 3 workflow not on this machine")
    w = json.load(open(path, encoding="utf-8"))
    t = _widget_names(tl.SeamStitchTimeline.INPUT_TYPES())
    p = _widget_names(rp.SeamStitchResultPreview.INPUT_TYPES())
    seen = 0
    for n in w["nodes"]:
        if n["type"] == "SeamStitchTimeline":
            vals = n["widgets_values"]
            assert len(vals) == len(OLD_TIMELINE_NAMES) and t[:len(vals)] == OLD_TIMELINE_NAMES
            assert vals[t.index("cut_codec")] in (tl.CODEC_LOSSLESS, tl.CODEC_H264)
            seen += 1
        elif n["type"] == "SeamStitchResultPreview":
            vals = n["widgets_values"]
            assert len(vals) == len(OLD_PREVIEW_NAMES) and p[:len(vals)] == OLD_PREVIEW_NAMES
            seen += 1
    assert seen >= 2
