"""SeamStitchRecombine insert mode: frames, anchors, audio drift, and a v0.1.0 regression.

Insert mode puts a generated bridge *between* two kept frames instead of replacing a
range that was scrubbed out. Nothing is removed (at trim_each_side = 0), the bridge's
first and last frame are always dropped because they are the kept frames either side of
the join, and the audio has to be cut at exactly the same frames or the picture drifts
against its own sound for the rest of the file (Defect 4, REAL_FOOTAGE_FINDINGS.md).

The fixture is synthetic and exact: every source frame carries a numeric fingerprint, so
"which source frame is this" is answered by equality rather than by eyeball, and the
audio is a linear chirp, whose autocorrelation has one sharp peak, so a lag is measured
in whole samples by normalised cross-correlation - the Defect 4 method, on a signal that
cannot alias onto itself the way a pure tone can.

`_encode_video` is replaced with a no-op: the splice result (the combined IMAGE tensor
and the AUDIO dict) is what this session changed, and comparing it directly keeps the
v0.1.0 regression free of encoder nondeterminism.

Run: python_embeded/python.exe -m pytest tests/test_recombine_insert.py -q
"""
import contextlib
import hashlib
import importlib
import io
import os
import shutil
import subprocess
import sys
import wave

import numpy as np
import pytest
import torch

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FPS = 24
SR = 48000
SPF = SR // FPS          # 2000 samples per frame, exactly - no rounding anywhere
W = H = 64
NSRC = 48                # source frames; 2.000 s of picture and 2.000 s of sound
FADE_MS = 20.0
FADE = int(round(FADE_MS * SR / 1000.0))   # 960 samples


def q(frame):
    """Same frame -> sample map the splice uses."""
    return int(round(frame * SR / float(FPS)))


# ---------------------------------------------------------------------------
# Fixture media
# ---------------------------------------------------------------------------
def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _source_frames():
    """NSRC frames, each with a unique colour and a unique two-block fingerprint."""
    a = np.zeros((NSRC, H, W, 3), np.uint8)
    for i in range(NSRC):
        a[i, :, :, 0] = (i * 5 + 3) % 256
        a[i, :, :, 1] = (i * 11 + 37) % 256
        a[i, :, :, 2] = (211 - i * 3) % 256
        a[i, :8, :8, :] = np.uint8((i * 5) % 256)
        a[i, 8:16, :8, :] = np.uint8((251 - i * 5) % 256)
    return a


def _chirp(n, f0, f1, sr=SR):
    """Linear chirp: non-periodic, so cross-correlation has one unambiguous peak."""
    t = np.arange(n, dtype=np.float64) / sr
    total = n / float(sr)
    return np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * total)))


def _source_audio():
    n = NSRC * SPF
    left = 0.5 * _chirp(n, 200.0, 8000.0)
    right = 0.4 * np.sin(2 * np.pi * 1000.0 * np.arange(n) / SR)   # known phase
    return np.stack([left, right]).astype(np.float32)


def _write_wav(path, wav):
    pcm = np.clip(wav.T, -1.0, 1.0)
    with wave.open(path, "wb") as f:
        f.setnchannels(wav.shape[0])
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((pcm * 32767.0).astype("<i2").tobytes())


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    """A lossless 64x64 / 24 fps / 48 kHz clip whose every frame is identifiable."""
    d = tmp_path_factory.mktemp("media")
    wav_path = str(d / "tone.wav")
    out = str(d / "src.mp4")
    src_audio = _source_audio()
    _write_wav(wav_path, src_audio)
    # mp4, not mkv: Matroska stores frame times at millisecond resolution, which is
    # coarser than a frame interval and makes an index -> time decode land on the
    # wrong frame. Real footage is mp4 (1/12288 here); so is the fixture.
    args = [_ffmpeg_exe(), "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
            "-i", wav_path,
            "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
            "-c:a", "alac", "-shortest", out]
    r = subprocess.run(args, input=_source_frames().tobytes(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-2000:]
    return {"path": out, "audio": torch.from_numpy(src_audio)}


# ---------------------------------------------------------------------------
# Importing the node outside ComfyUI
#
# recombine.py locates ComfyUI-VideoHelperSuite as a sibling of its own package
# folder at import time, and reaches its own audio_splice.py by relative import.
# So each version under test is laid out as a package in a throwaway custom_nodes
# tree next to an (empty) VHS folder; conftest has already put stub modules in
# sys.modules, so nothing of VHS is actually executed.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def custom_nodes(tmp_path_factory):
    root = tmp_path_factory.mktemp("custom_nodes")
    os.makedirs(str(root / "comfyui-videohelpersuite" / "videohelpersuite"))
    sys.path.insert(0, str(root))
    yield root
    sys.path.remove(str(root))


def _install(custom_nodes, label, src_dir):
    pkg = custom_nodes / label
    os.makedirs(str(pkg), exist_ok=True)
    open(str(pkg / "__init__.py"), "w").close()
    for name in ("recombine.py", "audio_splice.py"):
        shutil.copyfile(os.path.join(src_dir, name), str(pkg / name))
    mod = importlib.import_module(f"{label}.recombine")
    mod._encode_video = lambda images, *a, **k: {"ui": {"gifs": []}, "result": ((False, []),)}
    return mod


@pytest.fixture(scope="module")
def node(custom_nodes):
    """The working copy under test."""
    return _install(custom_nodes, "ss_head", PACK)


@pytest.fixture(scope="module")
def node_v010(custom_nodes, tmp_path_factory):
    """v0.1.0 of the same two files, from a detached worktree at the tag."""
    wt = str(tmp_path_factory.mktemp("wt") / "v010")
    r = subprocess.run(["git", "-C", PACK, "worktree", "add", "--detach", wt, "v0.1.0"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    try:
        return _install(custom_nodes, "ss_v010", wt)
    finally:
        subprocess.run(["git", "-C", PACK, "worktree", "remove", "--force", wt],
                       capture_output=True, text=True)


@pytest.fixture(scope="module")
def ref(node, media):
    """Every source frame as the node itself decodes it - the ground truth the
    output frames are compared against, so codec loss cancels out."""
    frames = node._decode_range(media["path"], FPS, 0, None)
    assert frames.shape[0] == NSRC, frames.shape
    flat = frames.reshape(NSRC, -1)
    assert len({bytes(f.numpy().tobytes()) for f in flat}) == NSRC, "fingerprints collide"
    return frames


@pytest.fixture(scope="module")
def av_offset(node, media):
    """The file's own video-minus-audio start offset, as the node reads it."""
    import av
    with av.open(media["path"]) as c:
        v, a = c.streams.video[0], c.streams.audio[0]
        v_start = float(v.start_time * v.time_base) if v.start_time is not None else 0.0
        a_start = float(a.start_time * a.time_base) if a.start_time is not None else 0.0
    return int(round((v_start - a_start) * SR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _f32(u8_frames):
    return u8_frames.to(torch.float32).div_(255.0)


def _bridge_images(ref, lead_anchor, tail_anchor, middle):
    """A bridge as a real generator hands one over: pinned on the two anchor frames,
    `middle` invented frames between them."""
    inner = torch.zeros((middle, H, W, 3), dtype=torch.float32)
    for i in range(middle):
        inner[i, :, :, 0] = (i * 9 + 17) / 255.0
        inner[i, :, :, 1] = (i * 3 + 200) % 256 / 255.0
        inner[i, :, :, 2] = (i * 29 + 5) % 256 / 255.0
    return torch.cat([_f32(ref[lead_anchor]).unsqueeze(0), inner,
                      _f32(ref[tail_anchor]).unsqueeze(0)], dim=0)


def _audio_dict(wave_2n):
    return {"waveform": wave_2n.unsqueeze(0), "sample_rate": SR}


def _bridge_audio(n_frames, f0=8000.0, f1=200.0):
    n = n_frames * SPF
    return torch.from_numpy(np.stack([0.45 * _chirp(n, f0, f1),
                                      0.45 * _chirp(n, f1, f0)]).astype(np.float32))


def _lag(out_seg, reference, at, max_lag=64):
    """Normalised cross-correlation of `out_seg` against `reference` around sample
    `at`; returns (best lag in samples, correlation). The Defect 4 measurement."""
    a = out_seg.numpy().astype(np.float64)
    a = a - a.mean()
    na = np.linalg.norm(a)
    best, best_lag = -2.0, None
    for lag in range(-max_lag, max_lag + 1):
        s = at + lag
        b = reference[s:s + a.shape[0]].numpy().astype(np.float64)
        if b.shape[0] != a.shape[0]:
            continue
        b = b - b.mean()
        den = na * np.linalg.norm(b)
        c = float(np.dot(a, b) / den) if den > 0 else -2.0
        if c > best:
            best, best_lag = c, lag
    return best_lag, best


def _call(node, media, *, images, start, end, insert, dedup=0.0, max_dedup=6,
          audio_mode="original", bridge_audio=None, override=None, v010=False):
    kw = dict(save_output=False, original_audio_override=override,
              bridge_audio=bridge_audio, audio_mode=audio_mode, audio_crossfade_ms=FADE_MS)
    if not v010:
        kw["insert"] = insert
    out = node.SeamStitchRecombine().recombine(
        images, media["path"], start, end, FPS, dedup, max_dedup,
        "ss_test", "video/h264-mp4", **kw)
    return out["result"][1], out["result"][2]


# ---------------------------------------------------------------------------
# Gate 1 - frame count: A + (bridge - 2, then the held-duplicate strip) + B
# ---------------------------------------------------------------------------
def test_insert_n0_keeps_every_source_frame(node, media, ref):
    join, middle = 20, 33
    images = _bridge_images(ref, join - 1, join, middle)
    combined, _ = _call(node, media, images=images, start=join, end=join - 1, insert=True)

    assert combined.shape[0] == NSRC + middle == 81
    assert torch.equal(combined[:join], _f32(ref[:join]))
    assert torch.equal(combined[join + middle:], _f32(ref[join:]))
    print(f"insert N=0: {NSRC} source + {middle} inserted = {combined.shape[0]} frames, "
          f"0 source frames removed")


def test_insert_runs_the_held_strip_after_the_anchor_drop(node, media, ref):
    join = 20
    held = torch.full((1, H, W, 3), 0.25, dtype=torch.float32)
    inner = _bridge_images(ref, join - 1, join, 10)[1:-1]
    images = torch.cat([_f32(ref[join - 1]).unsqueeze(0), held, held, inner,
                        _f32(ref[join]).unsqueeze(0)], dim=0)
    assert images.shape[0] == 14

    combined, audio = _call(node, media, images=images, start=join, end=join - 1,
                            insert=True, dedup=0.008, max_dedup=6,
                            override=_audio_dict(media["audio"]))
    # 14 - 2 anchors = 12, then one leading held duplicate stripped = 11.
    assert combined.shape[0] == NSRC + 11
    assert audio["waveform"].shape[-1] == q(NSRC + 11)
    print(f"anchor drop then held strip: 14 -> 12 -> 11 inserted frames")


# ---------------------------------------------------------------------------
# Gate 2 - the frames either side of the bridge are the source's own
# ---------------------------------------------------------------------------
def test_insert_anchor_frames_are_bit_identical(node, media, ref):
    join, middle = 20, 33
    images = _bridge_images(ref, join - 1, join, middle)
    combined, _ = _call(node, media, images=images, start=join, end=join - 1, insert=True)

    assert torch.equal(combined[join - 1], _f32(ref[join - 1]))      # A's last frame
    assert torch.equal(combined[join + middle], _f32(ref[join]))     # B's first frame
    # and the bridge's own pinned copies of them are gone, not doubled up
    assert not torch.equal(combined[join], _f32(ref[join - 1]))
    assert not torch.equal(combined[join + middle - 1], _f32(ref[join]))
    print("anchors bit-identical to source frames 19 and 20; no duplicate at either join")


def test_insert_with_trim_each_side_removes_only_that_range(node, media, ref):
    join, n, middle = 20, 2, 33
    start, end = join - n, join + n - 1          # removes source frames 18-21
    images = _bridge_images(ref, start - 1, end + 1, middle)
    combined, audio = _call(node, media, images=images, start=start, end=end, insert=True,
                            override=_audio_dict(media["audio"]))

    assert combined.shape[0] == NSRC - 2 * n + middle == 77
    assert torch.equal(combined[start - 1], _f32(ref[start - 1]))
    assert torch.equal(combined[start + middle], _f32(ref[end + 1]))
    gone = {_f32(ref[i]).numpy().tobytes() for i in range(start, end + 1)}
    assert not any(f.numpy().tobytes() in gone for f in combined)
    assert audio["waveform"].shape[-1] == q(combined.shape[0])

    # The trimmed frames' own audio does not get stretched under the bridge either:
    # insert mode puts silence under inserted frames whatever was trimmed to make
    # room. (At trim_each_side = 0 the pre-insert-mode gap piece happened to land
    # past its own limit and render as silence anyway - this is the case where the
    # old "keep the source's gap audio" rule actually reaches live samples.)
    out = audio["waveform"][0]
    j1 = q(start)
    gap_len = q(start + middle) - q(start)
    assert float(torch.abs(out[:, j1 + FADE:j1 + gap_len - FADE]).max()) == 0.0
    assert float(torch.abs(media["audio"][:, j1:j1 + q(end + 1 - start)]).max()) > 0.1
    print(f"trim_each_side={n}: removed source frames {start}-{end}, output {combined.shape[0]} frames, "
          f"{gap_len} samples of silence under the bridge")


# ---------------------------------------------------------------------------
# Gate 3 - audio: same length as the picture, 0-sample drift at both joins,
# and nothing of the source's own audio replayed under the inserted frames
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["silence", "bridge"])
def test_insert_audio_is_locked_to_the_picture(node, media, ref, av_offset, mode):
    join, middle = 20, 33
    nbridge = middle + 2
    images = _bridge_images(ref, join - 1, join, middle)
    bridge = _bridge_audio(nbridge) if mode == "bridge" else None
    combined, audio = _call(node, media, images=images, start=join, end=join - 1, insert=True,
                            audio_mode="bridge" if mode == "bridge" else "original",
                            bridge_audio=_audio_dict(bridge) if bridge is not None else None,
                            override=_audio_dict(media["audio"]))

    src = media["audio"]
    out = audio["waveform"][0]
    before_len = q(join)
    gap_len = q(join + middle) - q(join)
    j1, j2 = before_len, before_len + gap_len
    after_start = av_offset + q(join)

    # length: audio duration == picture duration, exactly
    assert combined.shape[0] == NSRC + middle
    assert out.shape[-1] == q(combined.shape[0]) == src.shape[-1] + gap_len

    # nothing either side of the insertion moved: sample-identical outside the fades
    assert torch.equal(out[:, :j1 - FADE], src[:, av_offset:av_offset + before_len - FADE])
    assert torch.equal(out[:, j2 + FADE:], src[:, after_start + FADE:])

    # drift, measured the Defect 4 way on the chirp channel
    win = 12000        # 250 ms
    lag_in, corr_in = _lag(out[0, j1 - FADE - win:j1 - FADE],
                           src[0], av_offset + before_len - FADE - win)
    lag_out, corr_out = _lag(out[0, j2 + FADE:j2 + FADE + win], src[0], after_start + FADE)
    assert (lag_in, lag_out) == (0, 0)
    assert corr_in > 0.99 and corr_out > 0.99

    # what is actually under the inserted frames
    interior = out[:, j1 + FADE:j2 - FADE]
    if mode == "silence":
        assert torch.count_nonzero(interior) == 0
    else:
        assert torch.equal(interior, bridge[:, q(1) + FADE:q(1) + gap_len - FADE])
        # ...and it is the bridge, not the source, that is under them
        assert not torch.equal(interior, src[:, after_start + FADE:after_start + gap_len - FADE])
    print(f"insert audio ({mode}): {out.shape[-1]} samples = {combined.shape[0]} frames; "
          f"drift {lag_in}/{lag_out} samples at the two joins "
          f"(corr {corr_in:.4f}/{corr_out:.4f})")


def test_insert_never_replays_the_source_audio_at_the_join(node, media, ref, av_offset):
    """The defect this session fixes: with no bridge audio the gap used to be filled
    from `after_start`, i.e. the sound of frames that are still in the output."""
    join, middle = 20, 33
    images = _bridge_images(ref, join - 1, join, middle)
    _, audio = _call(node, media, images=images, start=join, end=join - 1, insert=True,
                     override=_audio_dict(media["audio"]))
    out = audio["waveform"][0]
    gap_len = q(join + middle) - q(join)
    j1 = q(join)
    # The fades at the two joins are the documented, length-preserving crossfade;
    # everything between them must be untouched silence, where the pre-insert-mode
    # splice would have laid the source's own audio from `after_start`.
    src_at_join = media["audio"][:, av_offset + j1 + FADE:av_offset + j1 + gap_len - FADE]
    assert float(torch.abs(out[:, j1 + FADE:j1 + gap_len - FADE]).max()) == 0.0
    assert float(torch.abs(src_at_join).max()) > 0.1       # the audio that must NOT appear


def test_insert_bridge_audio_shortfall_is_padded_with_silence(node, media, ref, av_offset):
    """A bridge whose audio decodes short of its frames: replace mode borrows the
    source's audio behind those frames, insert mode cannot - there is none."""
    join, middle = 20, 33
    nbridge = middle + 2
    short = 3000                                   # samples the bridge is short by
    bridge = _bridge_audio(nbridge)[:, :nbridge * SPF - short]
    images = _bridge_images(ref, join - 1, join, middle)
    _, audio = _call(node, media, images=images, start=join, end=join - 1, insert=True,
                     audio_mode="bridge", bridge_audio=_audio_dict(bridge),
                     override=_audio_dict(media["audio"]))
    out = audio["waveform"][0]
    j1 = q(join)
    gap_len = q(join + middle) - q(join)
    # gap = bridge[q(1):] (real part) then `short - q(1)` samples of pad before `after`
    real = bridge.shape[-1] - q(1)
    missing = gap_len - real
    assert missing == short - q(1) > 0
    pad = out[:, j1 + real:j1 + gap_len - FADE]
    assert torch.count_nonzero(pad) == 0
    print(f"bridge audio {missing} samples short: padded with silence, not source audio")


# ---------------------------------------------------------------------------
# The duration warning: right in replace mode, silent about an empty gap
# ---------------------------------------------------------------------------
def test_gap_warning_does_not_fire_on_an_empty_gap(node, media, ref):
    """`expected_gap` is 0 in insert mode by design, so the replace-mode warning
    ("the combined video's duration will differ") must not fire - but it must still
    fire when a replace really does change the duration."""
    join, middle = 20, 33
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _call(node, media, images=_bridge_images(ref, join - 1, join, middle),
              start=join, end=join - 1, insert=True)
    said = buf.getvalue()
    assert "but the original gap was" not in said, said
    assert f"{middle} frame(s) inserted at the join, 0 removed" in said, said

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _call(node, media, images=_bridge_images(ref, 10, 20, 28),
              start=10, end=20, insert=False)
    assert "but the original gap was 11 frames" in buf.getvalue(), buf.getvalue()


# ---------------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------------
def test_empty_range_allowed_only_in_insert_mode(node, media, ref):
    images = _bridge_images(ref, 19, 20, 5)
    with pytest.raises(ValueError, match="must be >= start_frame"):
        _call(node, media, images=images, start=20, end=19, insert=False)
    with pytest.raises(ValueError, match="must be >= start_frame - 1"):
        _call(node, media, images=images, start=20, end=18, insert=True)


def test_insert_needs_more_than_its_two_anchors(node, media, ref):
    images = _bridge_images(ref, 19, 20, 0)          # anchors only
    with pytest.raises(ValueError, match="at least 3 regenerated frames"):
        _call(node, media, images=images, start=20, end=19, insert=True)


# ---------------------------------------------------------------------------
# Gate 4 - replace mode is byte-identical to v0.1.0 on a fixed fixture
# ---------------------------------------------------------------------------
def _replace_fixture(ref):
    """13 bridge frames for the range [10, 20]: two held duplicates at each end, so
    the dedup, the audio shortfall fade and the crossfade all get exercised."""
    inner = _bridge_images(ref, 10, 20, 9)[1:-1]
    a = torch.full((1, H, W, 3), 0.75, dtype=torch.float32)
    b = torch.full((1, H, W, 3), 0.10, dtype=torch.float32)
    return torch.cat([a, a, inner, b, b], dim=0)


@pytest.mark.parametrize("audio_mode", ["original", "bridge"])
def test_replace_mode_identical_to_v0_1_0(node, node_v010, media, ref, audio_mode):
    images = _replace_fixture(ref)
    assert images.shape[0] == 13
    bridge = _audio_dict(_bridge_audio(13)) if audio_mode == "bridge" else None
    kw = dict(images=images, start=10, end=20, insert=False, dedup=0.008, max_dedup=6,
              audio_mode=audio_mode, bridge_audio=bridge,
              override=_audio_dict(media["audio"]))

    new_img, new_aud = _call(node, media, **kw)
    old_img, old_aud = _call(node_v010, media, v010=True, **kw)

    h_new = hashlib.sha256(new_img.numpy().tobytes()).hexdigest()
    h_old = hashlib.sha256(old_img.numpy().tobytes()).hexdigest()
    assert h_new == h_old, f"picture changed: {h_new} vs {h_old}"
    assert new_aud["sample_rate"] == old_aud["sample_rate"]
    assert torch.equal(new_aud["waveform"], old_aud["waveform"]), "audio samples changed"
    print(f"replace/{audio_mode}: {new_img.shape[0]} frames, picture sha256 {h_new[:16]}…, "
          f"{new_aud['waveform'].shape[-1]} audio samples - identical to v0.1.0")


# ---------------------------------------------------------------------------
# The source is the resolution ground truth
# ---------------------------------------------------------------------------
def test_output_is_always_the_source_resolution(node, media, ref, capsys):
    """A regenerated segment at another size is resized onto the source; the
    untouched footage either side comes through pixel-for-pixel."""
    small = torch.nn.functional.interpolate(
        _bridge_images(ref, 9, 20, 9).permute(0, 3, 1, 2), size=(32, 32), mode="area"
    ).permute(0, 2, 3, 1)
    img, _ = _call(node, media, images=small, start=10, end=20, insert=False)
    assert tuple(img.shape[1:3]) == (H, W)
    assert img.shape[0] == NSRC
    assert torch.equal(img[:10], _f32(ref[:10]))
    assert torch.equal(img[21:], _f32(ref[21:]))
    out = capsys.readouterr().out
    assert "resizing the 11 regenerated frame(s)" in out and "Warning" not in out


def test_aspect_mismatch_is_called_out(node, media, ref, capsys):
    wide = torch.zeros((11, 32, 64, 3), dtype=torch.float32)
    img, _ = _call(node, media, images=wide, start=10, end=20, insert=False)
    assert tuple(img.shape[1:3]) == (H, W)
    assert "aspect ratio differs by" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Motion guides: context_frames
# ---------------------------------------------------------------------------
def _call_ctx(node, media, images, start, end, k, **kw):
    out = node.SeamStitchRecombine().recombine(
        images, media["path"], start, end, FPS, 0.0, 6, "ss_test", "video/h264-mp4",
        save_output=False, audio_crossfade_ms=FADE_MS, context_frames=k, **kw)
    return out["result"][1], out["result"][2]


@pytest.mark.parametrize("k", [1, 3])
def test_context_frames_are_dropped_and_the_bridge_replaces_the_range(node, media, ref, k):
    """Generated as K context + bridge + K context: only the bridge lands, in place of
    [start, end], and everything either side is the source's own frames."""
    start, end = 20, 29
    inner = _bridge_images(ref, 0, 0, 12)[1:-1]
    images = torch.cat([_f32(ref[start - k:start]), inner, _f32(ref[end + 1:end + 1 + k])])
    img, aud = _call_ctx(node, media, images, start, end, k)
    assert img.shape[0] == start + inner.shape[0] + (NSRC - end - 1)
    assert torch.equal(img[:start], _f32(ref[:start]))
    assert torch.equal(img[start + inner.shape[0]:], _f32(ref[end + 1:]))
    got = img[start:start + inner.shape[0]].mul(255).round()
    assert torch.equal(got, inner.mul(255).round())


def test_context_frames_bridge_audio_starts_after_the_context(node, media, ref):
    """Bridge audio runs on the generated clip's own timeline, so its first K frames
    of sound belong to the dropped context: the gap must start Q(K) samples in."""
    k, start, end = 3, 20, 29
    inner = _bridge_images(ref, 0, 0, 12)[1:-1]
    n = k + inner.shape[0] + k
    images = torch.cat([_f32(ref[start - k:start]), inner, _f32(ref[end + 1:end + 1 + k])])
    bridge = _bridge_audio(n)
    _, aud = _call_ctx(node, media, images, start, end, k, audio_mode="bridge",
                       bridge_audio=_audio_dict(bridge),
                       original_audio_override=_audio_dict(media["audio"]))
    out = aud["waveform"][0]
    mid = q(start) + q(inner.shape[0] // 2) - 400
    seg = out[0, mid:mid + 800]
    lag, corr = _lag(seg, bridge[0], q(k) + q(inner.shape[0] // 2) - 400)
    assert corr > 0.99 and lag == 0, (lag, corr)


def test_context_frames_refused_with_insert_and_when_too_short(node, media, ref):
    images = _bridge_images(ref, 19, 20, 5)
    with pytest.raises(ValueError, match="replace mode only"):
        _call_ctx(node, media, images, 20, 19, 2, insert=True)
    with pytest.raises(ValueError, match="more than 6 frames"):
        _call_ctx(node, media, images[:6], 20, 25, 3)
