"""Batch splice: several seams of one cut, each bridged alone, laid back into the cut."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import batch_splice as bs  # noqa: E402
from audio_splice import frame_sample  # noqa: E402

FPS, SR, N = 10, 1000, 100


def seam(start, end, inserted, k=0):
    return {"result": f"r{k}", "cut": "cut.mkv", "cut_fps": FPS, "fps": FPS, "start": start, "end": end,
            "removed": end - start + 1, "inserted": inserted, "frames": N + inserted - (end - start + 1),
            "cut_frames": N}


def test_plan_orders_segments_and_counts_shift():
    seams = [seam(20, 29, 14), seam(60, 69, 6)]
    segs, starts, total = bs.plan_frames(seams)
    assert segs == [("cut", 0, 20), ("res", 0, 20, 34), ("cut", 30, 60), ("res", 1, 60, 66), ("cut", 70, 100)]
    assert starts == [20, 60 + 4] and total == 100 + 4 - 4


def test_overlap_is_refused():
    with pytest.raises(bs.BatchError, match="overlaps"):
        bs.check_seams([seam(20, 40, 21), seam(40, 50, 11, 1)])


def test_mixed_cuts_refused():
    a, b = seam(10, 19, 10), seam(50, 59, 10, 1)
    b["cut"] = "other.mkv"
    with pytest.raises(bs.BatchError, match="different cuts"):
        bs.check_seams([a, b])


def _decoders(seams):
    """Frames are the frame's own value: the cut's frame f is f, seam k's result's frame f is 1000*(k+1)+f."""
    def cut(a, b):
        return torch.arange(a, b, dtype=torch.uint8).view(-1, 1, 1, 1).expand(-1, 2, 2, 3).contiguous()

    def res(k, a, b):
        return torch.full((b - a, 2, 2, 3), 200 + k, dtype=torch.uint8)
    return cut, res


def test_frames_are_the_cuts_between_and_the_bridges_at_the_seams():
    seams = bs.check_seams([seam(60, 69, 6, 1), seam(20, 29, 14, 0)])
    cut, res = _decoders(seams)
    out = bs.splice_frames(seams, cut, res)
    vals = out[:, 0, 0, 0].tolist()
    assert len(vals) == 100
    assert vals[:20] == list(range(20)) and vals[20:34] == [200] * 14
    assert vals[34:64] == list(range(30, 60)) and vals[64:70] == [201] * 6
    assert vals[70:] == list(range(70, 100))


def _result_track(seams, k):
    """What seam k's own result carries: the cut's sound (a ramp = the sample's original index)
    with its own bridge (value 5000+k) over its span. Rate 1000, 10 fps, so 100 samples a frame."""
    r = seams[k]
    orig = torch.arange(N * SR // FPS, dtype=torch.float32)
    a, b = frame_sample(r["start"], FPS, SR), frame_sample(r["end"] + 1, FPS, SR)
    bridge = torch.full((frame_sample(r["inserted"], FPS, SR),), 5000.0 + k)
    return torch.cat([orig[:a], bridge, orig[b:]]).unsqueeze(0)


def test_audio_takes_cut_sound_between_and_each_bridge_from_its_own_result():
    seams = bs.check_seams([seam(20, 29, 14, 0), seam(60, 69, 6, 1)])
    waves = [_result_track(seams, k) for k in range(2)]
    out = bs.splice_batch_audio(seams, waves, SR, FPS)[0]
    assert out.shape[0] == frame_sample(100, FPS, SR)
    orig = torch.arange(N * SR // FPS, dtype=torch.float32)
    # untouched stretches are the cut's own samples, in order
    assert torch.equal(out[:1900], orig[:1900])
    assert torch.equal(out[3500:6300], orig[3100:5900])
    assert torch.equal(out[7000:], orig[7000:])
    # the bridges are their own result's
    assert (out[2100:3300] == 5000.0).all()
    assert (out[6400:7000] == 5001.0).all()


def test_nothing_is_left_silent():
    seams = bs.check_seams([seam(20, 29, 14, 0), seam(60, 69, 6, 1)])
    waves = [_result_track(seams, k) + 1.0 for k in range(2)]
    out = bs.splice_batch_audio(seams, waves, SR, FPS)[0]
    assert (out != 0).all()


def test_close_seams_share_a_short_stretch_without_overwriting_each_other():
    seams = bs.check_seams([seam(20, 29, 10, 0), seam(32, 41, 10, 1)])   # 2 frames (200 samples) between
    waves = [_result_track(seams, k) + 1.0 for k in range(2)]
    out = bs.splice_batch_audio(seams, waves, SR, FPS)[0]
    assert (out != 0).all() and out.shape[0] == frame_sample(100, FPS, SR)


# ---------------------------------------------------------------- real files
from test_timeline import make_clip, codes, dirs, recombine, LEVEL_STEP  # noqa: E402,F401


def _write(path, code_list, fps, w=64, h=48):
    import subprocess
    from test_timeline import _ffmpeg
    frames = np.stack([np.full((h, w, 3), 16 + c * LEVEL_STEP, np.uint8) for c in code_list])
    subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-r", str(fps), "-i", "-", "-f", "lavfi", "-i",
                    f"sine=frequency=440:sample_rate=32000:duration={len(code_list) / fps}", "-c:a", "aac", "-shortest",
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", path], input=frames.tobytes(), check=True)
    return path


def test_two_independent_seams_lay_into_the_cut(dirs, recombine):
    inp, outp = dirs
    import timeline as tl
    fps = 24
    cut = make_clip(str(inp / "cut.mp4"), 40, fps)             # frame i has code i
    n = tl.probe(cut, fps)["frames"]                           # the AAC track can cost the clip its last frame
    folder = outp / "batch"
    folder.mkdir()
    # seam A replaces 8..11 with 6 frames (codes 45..50); seam B replaces 25..27 with 3 (codes 52..54)
    specs = [(8, 11, list(range(45, 51))), (25, 27, list(range(52, 55)))]
    for k, (s, e, new) in enumerate(specs):
        code_list = list(range(s)) + new + list(range(e + 1, n))
        res = _write(str(folder / f"seam_{k}.mp4"), code_list, fps)
        bs.write_seam_record(res, cut, fps, fps, s, e,
                             {"removed": e - s + 1, "inserted": len(new), "frames": len(code_list),
                              "source_frames": n, "joins": []})
    frames, audio, out_fps, info = bs.run_batch(cut, str(folder), recombine._decode_range)
    want = list(range(8)) + list(range(45, 51)) + list(range(12, 25)) + list(range(52, 55)) + list(range(28, n))
    assert len(want) == n + (6 - 4) + (3 - 3)
    got = torch.stack(list(frames)).mul(255).round().numpy()   # lazy frames, floats 0-1
    assert tuple(frames.shape[:1]) == (len(want),) and frames[0].shape == tuple(got[0].shape)
    assert codes(got) == want
    assert out_fps == fps
    assert audio is not None and audio["waveform"].shape[-1] == int(round(len(want) / fps * audio["sample_rate"]))


def test_a_rerolled_seam_uses_its_newest_result(dirs, recombine):
    inp, outp = dirs
    import timeline as tl
    fps = 24
    cut = make_clip(str(inp / "cut.mp4"), 40, fps)
    n = tl.probe(cut, fps)["frames"]
    folder = outp / "batch"
    folder.mkdir()
    for name, code in (("old.mp4", 50), ("new.mp4", 55)):
        code_list = list(range(10)) + [code] * 4 + list(range(14, n))
        res = _write(str(folder / name), code_list, fps)
        bs.write_seam_record(res, cut, fps, fps, 10, 13,
                             {"removed": 4, "inserted": 4, "frames": n, "source_frames": n, "joins": []})
        os.utime(res, (os.path.getatime(res), os.path.getmtime(res) + (5 if name == "new.mp4" else 0)))
    recs, _ = bs.read_seam_records(str(folder))
    assert len(recs) == 1 and recs[0]["result"].endswith("new.mp4")


def test_a_changed_result_is_skipped_not_trusted(dirs, recombine):
    inp, outp = dirs
    cut = make_clip(str(inp / "cut.mp4"), 30, 24)
    folder = outp / "batch"
    folder.mkdir()
    res = _write(str(folder / "r.mp4"), list(range(30)), 24)
    bs.write_seam_record(res, cut, 24, 24, 5, 8, {"removed": 4, "inserted": 4, "frames": 30, "source_frames": 30, "joins": []})
    with open(res, "ab") as f:
        f.write(b"x")
    recs, skipped = bs.read_seam_records(str(folder))
    assert recs == [] and len(skipped) == 1


def test_result_preview_leaves_the_record_beside_a_saved_result(dirs, recombine, monkeypatch, tmp_path):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path), raising=False)
    import result_preview as rp
    import timeline as tl
    inp, outp = dirs
    fps = 24
    cut = make_clip(str(inp / "cut.mp4"), 40, fps)
    n = tl.probe(cut, fps)["frames"]
    res = _write(str(outp / "seam_a.mp4"), list(range(8)) + list(range(45, 51)) + list(range(12, n)), fps)
    out = rp.SeamStitchResultPreview().preview(cut, 8, 11, fps, filenames=(True, [res]))
    assert out["result"][0] == res
    recs, _ = bs.read_seam_records(str(outp))
    assert len(recs) == 1 and (recs[0]["start"], recs[0]["end"], recs[0]["inserted"]) == (8, 11, 6)
