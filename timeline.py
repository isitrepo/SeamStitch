"""SeamStitchTimeline: a one-track mini editor that feeds the SeamStitch splice.

Arrange clips on a strip (drag in, reorder, trim, cut at the playhead, open a gap),
mark the ONE thing to regenerate - a range, a cut between two clips, or a gap - and
the node hands the generator and SeamStitchRecombine exactly what SeamStitchLoader
would. It replaces the Combine -> Loader pair: the strip IS the combine, the mark IS
the trim.

How it stays exact:
  * the strip's played ranges are assembled into ONE file, the "cut", by decoding
    every piece on the same index-exact timeline SeamStitchRecombine cuts on
    (_iter_frames mirrors recombine._decode_range) and encoding it once at the
    timeline's frame rate. A single untouched clip already at that rate is passed
    through as-is - no re-encode at all.
  * the node's outputs come from SeamStitchLoader itself, run on that cut, so
    every downstream node (guides, generators, Recombine) sees nothing new.
  * the full preview button builds the very same cut (same function, same cache
    key), so what you scrub before queueing is what gets spliced.

UI state lives in two hidden widgets: `sequence` (text, see timeline_math) and
`target` (JSON). js/timeline.js draws and edits both.

Conform to 24 fps (MiniMax H3): H3 has no frame-rate input - it places reference audio
and the prompt's Picture timings on a fixed 24 fps clock while taking frames 1:1, so
on 25 fps footage the lips ran ~4% ahead of the speech (r15). With conform_to_24fps
on, the cut keeps every frame (frame n is still frame n) but is LABELLED 24 fps, and
its audio is slowed by 24/fps with the pitch kept, exactly frames/24 s long. The
Loader then runs on that cut at 24, so every time it reports is on H3's clock.
SeamStitch Result Preview's restore puts the source frame rate and original audio back.
"""

import hashlib
import json
import os
import subprocess
import tempfile
import threading

import av
import numpy as np
import torch
import folder_paths
from aiohttp import web
from server import PromptServer

try:
    from . import timeline_math as tm
    from .loader import SeamStitchLoader, _list_input_videos
    from .video_colour import color_args
except ImportError:  # imported as a top-level module (tests)
    import timeline_math as tm
    from loader import SeamStitchLoader, _list_input_videos
    from video_colour import color_args

_VIDEO_EXTENSIONS = ('.mp4', '.webm', '.mkv', '.avi', '.mov', '.m4v', '.flv', '.wmv')
_RGB_PIX = ('gbr', 'rgb', 'bgr', 'argb', 'abgr', 'rgba', 'bgra')
CUT_SUBDIR = "seamstitch_timeline"
CODEC_LOSSLESS = "lossless (ffv1)"
CODEC_H264 = "h264"
# Cached cuts kept in input/seamstitch_timeline (a lossless cut is large).
KEEP_CUTS = 8
# Bumped whenever the assembly would produce different bytes for the same inputs,
# so a cached cut from an older rule is not reused.
_ASSEMBLY_VERSION = "1"
_BUILD_LOCK = threading.Lock()
# MiniMax H3's fixed clock (comfy_extras/nodes_minimax_h3.py hard-codes FPS = 24).
H3_FPS = 24


# ---------------------------------------------------------------------------
# paths and probing
# ---------------------------------------------------------------------------

def resolve_path(p):
    """Sequence path -> existing absolute path. Absolute paths are used as-is,
    anything else is looked up in the input directory, then the output directory
    (where Recombine and the Result Preview's 'use as timeline' leave files)."""
    if not p:
        raise tm.SequenceError("empty clip path")
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    for base in (folder_paths.get_input_directory(), folder_paths.get_output_directory()):
        cand = os.path.join(base, p)
        if os.path.isfile(cand):
            return cand
    raise tm.SequenceError(f"clip not found: {p} (looked in the input and output folders)")


_PROBE_CACHE = {}


def probe(path, frame_rate):
    """Stream facts without decoding a frame. `frames` is how many frames the clip
    yields at `frame_rate` on _iter_frames' timeline: every target time
    base + i/fr up to the last frame's presentation time (a thousandth-of-a-frame
    tolerance, same as the decoder)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size, float(frame_rate))
    if key in _PROBE_CACHE:
        return dict(_PROBE_CACHE[key])
    with av.open(path) as c:
        if not c.streams.video:
            raise tm.SequenceError(f"{os.path.basename(path)} has no video stream")
        vs = c.streams.video[0]
        tb = float(vs.time_base)
        pts = [p.pts for p in c.demux(vs) if p.pts is not None]
        if not pts:
            raise tm.SequenceError(f"{os.path.basename(path)} has no video frames")
        base = float(vs.start_time * vs.time_base) if vs.start_time is not None else min(pts) * tb
        last = max(pts) * tb
        native = float(vs.average_rate) if vs.average_rate else 0.0
        a = c.streams.audio[0] if c.streams.audio else None
        info = {
            "width": vs.codec_context.width,
            "height": vs.codec_context.height,
            "native_fps": native,
            "native_frames": len(pts),
            "base_time": base,
            "has_audio": a is not None,
            "sample_rate": int(a.rate) if a is not None else 0,
            "deep_rgb": _deep_rgb(vs.codec_context.format),
        }
    fr = float(frame_rate) if frame_rate and frame_rate > 0 else (round(native) or 24)
    info["frame_rate"] = fr
    info["frames"] = int(np.floor((last - base) * fr + max(1e-3, tb * fr))) + 1
    info["duration"] = info["frames"] / fr
    if len(_PROBE_CACHE) > 512:
        _PROBE_CACHE.clear()
    _PROBE_CACHE[key] = dict(info)
    return info


def auto_frame_rate(frame_rate, first_clip_path):
    """0 = the first clip's own rate, rounded (the pack's frame rates are ints)."""
    if frame_rate and int(frame_rate) > 0:
        return int(frame_rate)
    with av.open(first_clip_path) as c:
        r = c.streams.video[0].average_rate
    return int(round(float(r))) if r else 24


# ---------------------------------------------------------------------------
# decoding: identical timeline to recombine._decode_range, but streamed
# ---------------------------------------------------------------------------

def _color_args(cc, w, h):
    return color_args(cc, w, h)


def _deep_rgb(fmt):
    """An RGB-coded pixel format with more than 8 bits a channel (FFV1 gbrp16le / gbrap16le,
    as VHS's ffv1-mkv and SeamStitch Swap Assemble's 16-bit master write)."""
    return fmt is not None and fmt.name.startswith(_RGB_PIX) and max(c.bits for c in fmt.components) > 8


def _rgb8_from_16(frame):
    """16-bit RGB to uint8, nearest level: round(x / 257), the exact inverse of v * 257 (how VHS
    and Swap Assemble store 8-bit level v). swscale's own 16->8 bit conversion (to_ndarray("rgb24"))
    is not: it lifts some pixels one level, a share that depends on the level, channel and
    position (none below ~110, all from ~200 up), so a 16-bit master read back mean +0.13."""
    x = frame.to_ndarray(format="rgb48le").astype(np.uint32)
    return ((x + 128) // 257).astype(np.uint8)


def _iter_frames(path, frame_rate, start_idx, end_idx):
    """Yield uint8 HxWx3 frames start_idx..end_idx-1 at frame_rate, decode order from
    the stream's own first frame. Same selection rule as recombine._decode_range:
    frame i is the last source frame whose time is <= base + i/fr (+ 1e-3 frame)."""
    if end_idx is not None and end_idx <= start_idx:
        return
    fr = float(frame_rate)
    with av.open(path) as c:
        vs = c.streams.video[0]
        cc = vs.codec_context
        cs, cr, dst = _color_args(cc, cc.width, cc.height)
        base = float(vs.start_time * vs.time_base) if vs.start_time is not None and vs.time_base else 0.0
        t_start = base + start_idx / fr
        t_end = base + end_idx / fr if end_idx is not None else None
        interval = 1.0 / fr
        # A frame's timestamp can sit up to one tick of the stream's clock early: MKV counts
        # whole milliseconds, so a 48 fps frame due at 83.333 ms is stored at 83 ms. With only
        # a thousandth-of-a-frame tolerance that frame was skipped and every later index was
        # off by one (FFV1 masters read back one frame late). Tolerate one tick.
        tol = max(interval * 1e-3, float(vs.time_base or 0))
        vs.thread_type = "AUTO"
        c.seek(int(t_start / float(vs.time_base)), stream=vs, backward=True)
        idx, target = start_idx, t_start
        for frame in c.decode(vs):
            ft = frame.time
            if ft is None:
                ft = float(frame.pts * float(vs.time_base)) if frame.pts else 0.0
            if ft < t_start - interval:
                continue
            if t_end is not None and ft > t_end + interval:
                break
            rgb = None
            while target <= ft + tol:
                if end_idx is not None and idx >= end_idx:
                    break
                if rgb is None and frame.format.name.startswith(_RGB_PIX):
                    # RGB-coded (FFV1 rgb / gbrp): no YUV matrix to apply - see loader._RGB_PIX.
                    rgb = _rgb8_from_16(frame) if _deep_rgb(frame.format) else frame.to_ndarray(format="rgb24")
                if rgb is None:
                    try:
                        rgb = frame.reformat(format="rgb24", src_colorspace=cs, src_color_range=cr,
                                             dst_color_range=dst).to_ndarray(format="rgb24")
                    except Exception:
                        rgb = frame.to_ndarray(format="rgb24")
                yield rgb
                idx += 1
                target = base + idx / fr
            if end_idx is not None and idx >= end_idx:
                break


def _fit(frame, w, h, fit):
    """Place a frame onto the cut's w x h canvas, aspect kept (crop = fill and trim
    the overhang, pad = fit inside with black bars)."""
    import cv2
    fh, fw = frame.shape[:2]
    if (fw, fh) == (w, h):
        return frame
    scale = max(w / fw, h / fh) if fit == "crop" else min(w / fw, h / fh)
    sw, sh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    r = cv2.resize(frame, (sw, sh), interpolation=interp)
    if fit == "crop":
        x0, y0 = (sw - w) // 2, (sh - h) // 2
        return np.ascontiguousarray(r[y0:y0 + h, x0:x0 + w])
    out = np.zeros((h, w, 3), dtype=np.uint8)
    x0, y0 = (w - sw) // 2, (h - sh) // 2
    out[y0:y0 + sh, x0:x0 + sw] = r
    return out


def _clip_audio(path, t0, t1, sr_out):
    """Stereo float32 [2, n] of the file's audio between presentation times t0..t1
    (absolute, same clock as the video's frame times), resampled to sr_out.
    Silence when the file has no audio."""
    n = max(0, int(round((t1 - t0) * sr_out)))
    out = np.zeros((2, n), dtype=np.float32)
    try:
        with av.open(path) as c:
            if not c.streams.audio:
                return out
            a = c.streams.audio[0]
            rs = av.AudioResampler(format="fltp", layout="stereo", rate=sr_out)
            chunks, first = [], None
            c.seek(int(max(0.0, t0 - 1.0) / float(a.time_base)), stream=a, backward=True)
            for fr_ in c.decode(a):
                ft = fr_.time if fr_.time is not None else 0.0
                if ft > t1 + 1.0:
                    break
                for r in rs.resample(fr_):
                    if first is None:
                        first = r.time if r.time is not None else ft
                    chunks.append(r.to_ndarray())
            for r in rs.resample(None):
                chunks.append(r.to_ndarray())
            if not chunks:
                return out
            wav = np.concatenate(chunks, axis=1).astype(np.float32)
            off = int(round((t0 - first) * sr_out))
            src_lo, dst_lo = max(0, off), max(0, -off)
            take = min(n - dst_lo, wav.shape[1] - src_lo)
            if take > 0:
                out[:, dst_lo:dst_lo + take] = wav[:, src_lo:src_lo + take]
    except Exception as e:  # a broken audio track must not stop the picture
        print(f"[SeamStitch] Timeline: audio of {os.path.basename(path)} skipped: {e}")
    return out


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


# ---------------------------------------------------------------------------
# the 24 fps conform: frames re-labelled, audio time-stretched with pitch kept
# ---------------------------------------------------------------------------

def conform_rate(frame_rate, conform):
    """The rate the cut is labelled at: H3's 24 when conforming a non-24 source, else its own."""
    return H3_FPS if conform and int(frame_rate) != H3_FPS else int(frame_rate)


def _atempo_chain(tempo):
    """ffmpeg's atempo takes 0.5-2.0 per stage; chain stages for anything wider
    (25 -> 24 is one stage, 0.96; 60 -> 24 would be 0.5 x 0.8)."""
    stages = []
    while tempo < 0.5:
        stages.append(0.5)
        tempo /= 0.5
    while tempo > 2.0:
        stages.append(2.0)
        tempo /= 2.0
    stages.append(tempo)
    return ",".join(f"atempo={s:.10f}" for s in stages)


_TICKS = 512   # timebase ticks per frame for the re-label: 1/(fps*512) s, exact for any int rate


def _write_f32(wav, out_dir):
    fd, path = tempfile.mkstemp(suffix=".f32", dir=out_dir)
    os.close(fd)
    np.ascontiguousarray(np.asarray(wav, dtype=np.float32).T).astype("<f4").tofile(path)
    return path


def relabel_video(src, dst, from_fr, to_fr, wav=None, sample_rate=0, audio_codec=("-c:a", "flac")):
    """Copy src's video stream into dst (no re-encode) so that frame n plays at exactly n/to_fr,
    plus `wav` ([C, n] float32) as its audio when given.

    Each timestamp is snapped to its frame on the new grid (ffmpeg's setts bitstream filter:
    pts = round(t * from_fr) frames), with an explicit one-frame duration so the rate label
    comes out exact. Scaling instead (-itsscale) compounded MKV's millisecond rounding: a
    24 fps MKV re-labelled at 25 put frames ~1 ms early, past the decoders' one-tick
    tolerance, and frame 26 read back as 27. -r cannot be used with it (it moved frames).
    MP4/MOV take the label from the filter; MKV/WebM copy it from their input's header, so
    those go through a MOV intermediate (MOV holds h264/hevc/prores/ffv1...)."""
    ff = _ffmpeg_exe()
    bsf = (f"setts=time_base=1/{int(to_fr) * _TICKS}:pts={_TICKS}*round(PTS*TB*{int(from_fr)})"
           f":dts={_TICKS}*round(DTS*TB*{int(from_fr)}):duration={_TICKS}")
    out_dir = os.path.dirname(os.path.abspath(dst))
    ext = os.path.splitext(dst)[1].lower()
    apath = _write_f32(wav, out_dir) if wav is not None else None
    a_in = ["-f", "f32le", "-ar", str(int(sample_rate)), "-ac", str(np.asarray(wav).shape[0]), "-i", apath] \
        if apath else []
    a_out = ["-map", "1:a"] + list(audio_codec) if apath else []
    mov_flags = ["-movflags", "+faststart"] if ext in (".mp4", ".m4v", ".mov") else []

    def run(cmd, what):
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed {what}: {p.stderr.decode(errors='replace').strip()[-800:]}")

    mid = None
    try:
        if ext in (".mp4", ".m4v", ".mov"):
            run([ff, "-v", "error", "-y", "-i", src] + a_in + ["-map", "0:v:0", "-c:v", "copy", "-bsf:v", bsf]
                + a_out + mov_flags + [dst], f"re-labelling {os.path.basename(src)} at {to_fr} fps")
            return dst
        mid = os.path.splitext(dst)[0] + ".relabel.mov"
        try:
            run([ff, "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-c:v", "copy", "-bsf:v", bsf, mid], "")
            v_in = ["-i", mid]
            v_out = ["-map", "0:v:0", "-c:v", "copy"]
        except RuntimeError:
            # a codec MOV cannot hold: the frames are still exact, only the header's rate is the input's
            print(f"[SeamStitch] re-label: {os.path.basename(src)} cannot pass through MOV; "
                  f"{os.path.basename(dst)}'s header keeps the input's frame rate (timestamps are {to_fr} fps).")
            v_in = ["-i", src]
            v_out = ["-map", "0:v:0", "-c:v", "copy", "-bsf:v", bsf]
        run([ff, "-v", "error", "-y"] + v_in + a_in + v_out + a_out + [dst],
            f"re-labelling {os.path.basename(src)} at {to_fr} fps")
        return dst
    finally:
        for f in (apath, mid):
            if f:
                try:
                    os.remove(f)
                except OSError:
                    pass


def time_stretch(wav, sample_rate, tempo, out_samples):
    """[C, N] float32 played `tempo` times as fast, pitch kept (ffmpeg atempo, the filter
    the r15 test used), then padded/trimmed to exactly `out_samples`: atempo rounds a few
    ms off the tail (8.696 s for an 8.708 s target in r15)."""
    wav = np.ascontiguousarray(np.asarray(wav, dtype=np.float32))
    ch = wav.shape[0]
    out = np.zeros((ch, int(out_samples)), dtype=np.float32)
    if wav.shape[1] == 0 or out_samples <= 0:
        return out
    if abs(tempo - 1.0) < 1e-9:
        got = wav
    else:
        proc = subprocess.run(
            [_ffmpeg_exe(), "-v", "error", "-f", "f32le", "-ar", str(int(sample_rate)), "-ac", str(ch),
             "-i", "-", "-af", _atempo_chain(float(tempo)), "-f", "f32le", "-ar", str(int(sample_rate)),
             "-ac", str(ch), "-"],
            input=wav.T.tobytes(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg atempo failed: {proc.stderr.decode(errors='replace').strip()[-400:]}")
        got = np.frombuffer(proc.stdout, dtype="<f4").reshape(-1, ch).T
    take = min(out.shape[1], got.shape[1])
    out[:, :take] = got[:, :take]
    return out


# ---------------------------------------------------------------------------
# the cut
# ---------------------------------------------------------------------------

def plan_cut(sequence, frame_rate):
    """Sequence text -> (cut layout with absolute paths, frame_rate used)."""
    entries = tm.parse_sequence(sequence)
    clips = [e for e in entries if e["kind"] == "clip"]
    if not clips:
        raise tm.SequenceError("the timeline has no clips - drag a video onto it or use + add")
    for e in clips:
        e["path"] = resolve_path(e["path"])
    fr = auto_frame_rate(frame_rate, clips[0]["path"])
    cut = tm.resolve_cut(entries, lambda p: probe(p, fr)["frames"])
    return cut, fr


def _cut_key(cut, fr, crf, fit, codec=CODEC_LOSSLESS, conform=False):
    h = hashlib.sha256()
    key = [_ASSEMBLY_VERSION, fr, int(crf), fit, codec]
    if conform_rate(fr, conform) != fr:
        key.append(f"conform{H3_FPS}")   # only when it changes the file: unconformed keys stay as they were
    if any(probe(p["path"], fr).get("deep_rgb") for p in cut["pieces"]):
        key.append("rgb16exact")         # same: an all-8-bit cut keeps its key (and its bytes)
    h.update(json.dumps(key).encode())
    for p in cut["pieces"]:
        st = os.stat(p["path"])
        h.update(json.dumps([os.path.abspath(p["path"]), st.st_mtime_ns, st.st_size,
                             p["enter"], p["exit"]]).encode())
    return h.hexdigest()[:16]


def passthrough_path(cut, fr):
    """The one source file itself, when the cut is exactly that file at its own rate:
    nothing to assemble, and no generation loss before the splice."""
    if len(cut["pieces"]) != 1:
        return None
    p = cut["pieces"][0]
    info = probe(p["path"], fr)
    if p["enter"] != 0 or p["exit"] != info["frames"]:
        return None
    if abs(info["native_fps"] - fr) > 1e-3 or abs(info["base_time"]) > 0.25 / fr:
        return None
    if info["frames"] != info["native_frames"]:
        return None
    return p["path"]


def _prune_cuts(out_dir, keep_path):
    """Keep the newest KEEP_CUTS cuts (and their previews); the rest are rebuilt on demand."""
    try:
        cuts = sorted((os.path.join(out_dir, f) for f in os.listdir(out_dir)
                       if f.startswith("cut_") and "_preview" not in f and not f.endswith(".part.mp4")
                       and not f.endswith(".part.mkv")), key=os.path.getmtime, reverse=True)
        for old in cuts[KEEP_CUTS:]:
            if os.path.abspath(old) == os.path.abspath(keep_path):
                continue
            for f in (old, os.path.splitext(old)[0] + "_preview.mp4"):
                try:
                    os.remove(f)
                except OSError:
                    pass
    except OSError:
        pass


def cut_audio(cut, fr):
    """The cut's own sound at its own rate: (stereo float32 [2, n], sample_rate), exactly
    frames/fr long, sample 0 under frame 0 - what an unconformed cut carries, and what
    the conform's Result Preview restore puts back."""
    rates = [probe(p["path"], fr)["sample_rate"] for p in cut["pieces"]]
    sr = next((r for r in rates if r), 48000)
    audio = []
    for p in cut["pieces"]:
        base = probe(p["path"], fr)["base_time"]
        audio.append(_clip_audio(p["path"], base + p["enter"] / fr, base + p["exit"] / fr, sr))
    wav = np.concatenate(audio, axis=1) if audio else np.zeros((2, 0), np.float32)
    # exactly frames/fr of sound: per-piece rounding must not drift the tail
    want = int(round(cut["frames"] / fr * sr))
    if wav.shape[1] < want:
        wav = np.pad(wav, ((0, 0), (0, want - wav.shape[1])))
    return wav[:, :want], sr


def conform_audio(wav, sr, frames, fr):
    """The cut's sound slowed onto H3's clock: tempo 24/fr, pitch kept, exactly frames/24 s."""
    return time_stretch(wav, sr, H3_FPS / float(fr), int(round(frames / float(H3_FPS) * sr)))


def _cut_dir():
    out_dir = os.path.join(folder_paths.get_input_directory(), CUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _relabel_cut(cut, src, fr, crf, fit, codec):
    """The conform of one untouched clip: its video stream COPIED with frame n re-timed to
    n/24 (relabel_video), so the picture is the source's own bitstream - lossless, frame
    for frame - now playing at 24 fps, with the stretched audio (FLAC) beside it. MKV, as
    the lossless cut; it holds any source codec."""
    out_dir = _cut_dir()
    out = os.path.join(out_dir, f"cut_{_cut_key(cut, fr, crf, fit, codec, conform=True)}.mkv")
    with _BUILD_LOCK:
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return out
        wav, sr = cut_audio(cut, fr)
        wav = conform_audio(wav, sr, cut["frames"], fr)
        tmp = os.path.splitext(out)[0] + ".part.mkv"
        try:
            relabel_video(src, tmp, fr, H3_FPS, wav, sr)
            os.replace(tmp, out)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        print(f"[SeamStitch] Timeline: conformed {os.path.basename(src)} to {H3_FPS} fps by stream copy "
              f"({cut['frames']} frames kept, {fr} -> {H3_FPS} fps, audio slowed x{H3_FPS / float(fr):.4f}, "
              f"pitch kept) -> {out}")
        _prune_cuts(out_dir, out)
    return out


def build_cut(cut, fr, crf=12, fit="crop", codec=CODEC_LOSSLESS, conform=False):
    """Assemble the cut (or pass the single source through). Returns the path.

    Lossless (default): FFV1, RGB planes, FLAC - the frames Recombine reads back are
    exactly the decoded sources (measured: an h264 4:2:0 cut at crf 12 shifted the
    picture -1.2 levels on test clip 4.mp4, and that loss stacked with the final encode).
    h264: much smaller, one lossy generation.

    conform (and fr != 24): the same frames, labelled 24 fps, audio slowed to match -
    see the module docstring. The single untouched clip is re-labelled by stream copy."""
    out_fr = conform_rate(fr, conform)
    same = passthrough_path(cut, fr)
    if same:
        return same if out_fr == fr else _relabel_cut(cut, same, fr, crf, fit, codec)
    lossless = codec != CODEC_H264
    out_dir = _cut_dir()
    out = os.path.join(out_dir, f"cut_{_cut_key(cut, fr, crf, fit, codec, conform)}{'.mkv' if lossless else '.mp4'}")
    with _BUILD_LOCK:
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return out
        first = probe(cut["pieces"][0]["path"], fr)
        w, h = first["width"] - first["width"] % 2, first["height"] - first["height"] % 2
        wav, sr = cut_audio(cut, fr)
        if out_fr != fr:
            wav = conform_audio(wav, sr, cut["frames"], fr)
        wav = np.ascontiguousarray(wav.T)

        fd, apath = tempfile.mkstemp(suffix=".f32", dir=out_dir)
        os.close(fd)
        tmp = os.path.splitext(out)[0] + (".part.mkv" if lossless else ".part.mp4")
        try:
            wav.astype("<f4").tofile(apath)
            # Frames are decoded at the source rate fr (which frames) and written at out_fr
            # (how fast they play): equal unless conforming.
            cmd = [_ffmpeg_exe(), "-v", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(out_fr), "-i", "-",
                   "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", apath,
                   "-map", "0:v", "-map", "1:a"]
            if lossless:
                cmd += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", "-g", "1", "-slices", "16",
                        "-slicecrc", "1", "-color_primaries", "bt709", "-color_trc", "bt709",
                        "-c:a", "flac", tmp]
            else:
                cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                        "-c:v", "libx264", "-crf", str(int(crf)), "-preset", "medium",
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                        "-color_range", "tv", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", tmp]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            written = 0
            try:
                for p in cut["pieces"]:
                    n = 0
                    for f in _iter_frames(p["path"], fr, p["enter"], p["exit"]):
                        f = _fit(f, w, h, fit)
                        proc.stdin.write(f.tobytes())
                        n += 1
                    if n != p["exit"] - p["enter"]:
                        raise RuntimeError(
                            f"{os.path.basename(p['path'])}: decoded {n} frames, expected "
                            f"{p['exit'] - p['enter']} ({p['enter']}..{p['exit'] - 1} at {fr} fps)")
                    written += n
            finally:
                proc.stdin.close()
                err = proc.stderr.read().decode(errors="replace")
                proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed assembling the timeline: {err.strip()[-800:]}")
            os.replace(tmp, out)
        finally:
            for f in (apath, tmp):
                try:
                    os.remove(f)
                except OSError:
                    pass
        print(f"[SeamStitch] Timeline: assembled {len(cut['pieces'])} piece(s), {written} frames at "
              f"{fr} fps, {w}x{h}, {codec}"
              + (f", conformed to {out_fr} fps (audio slowed x{out_fr / float(fr):.4f}, pitch kept)"
                 if out_fr != fr else "") + f" -> {out}")
        _prune_cuts(out_dir, out)
    return out


def preview_copy(path):
    """What the browser plays for the full preview: the cut itself when it is H.264, else a
    small H.264 copy beside it (Chrome cannot play FFV1). Colour-tagged BT.709 like the rest."""
    if not path.lower().endswith(".mkv"):
        return path
    prev = os.path.splitext(path)[0] + "_preview.mp4"
    if os.path.isfile(prev) and os.path.getmtime(prev) >= os.path.getmtime(path):
        return prev
    tmp = prev + ".part.mp4"
    subprocess.run([_ffmpeg_exe(), "-v", "error", "-y", "-i", path, "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-color_range", "tv", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", tmp],
                   check=True, capture_output=True)
    os.replace(tmp, prev)
    return prev


# ---------------------------------------------------------------------------
# Swap jobs: an assembly's joins as marks on the strip (read-only)
# ---------------------------------------------------------------------------

# swap_plan.JOBS_SUBDIR, kept here so the Timeline does not import the Swap nodes.
SWAP_JOBS_SUBDIR = "seamstitch_swap"
_SWAP_ASSEMBLY_EXT = (".mp4", ".mkv")


def _swap_base(out_dir=None):
    return os.path.join(out_dir or folder_paths.get_output_directory(), SWAP_JOBS_SUBDIR)


def swap_assemblies(out_dir=None):
    """Every Swap job holding an assembly, newest job first, each job's assemblies newest
    first. An assembly is <job>/assembled/<stem>.mp4|.mkv with its <stem>.report.json; its
    `path` is output-relative with forward slashes, the sequence line the Timeline writes
    (resolve_path finds it in the output folder). Only real files inside the jobs folder:
    a link that resolves outside it is skipped."""
    base = _swap_base(out_dir)
    if not os.path.isdir(base):
        return []
    real_base = os.path.realpath(base)
    jobs = []
    for job in os.listdir(base):
        adir = os.path.join(base, job, "assembled")
        if not os.path.isdir(adir):
            continue
        items = []
        for f in os.listdir(adir):
            stem, ext = os.path.splitext(f)
            full = os.path.join(adir, f)
            rep = os.path.join(adir, stem + ".report.json")
            if ext.lower() not in _SWAP_ASSEMBLY_EXT or not os.path.isfile(full) or not os.path.isfile(rep):
                continue
            if os.path.commonpath([real_base, os.path.realpath(full)]) != real_base:
                continue
            items.append({"job": job, "file": f, "path": f"{SWAP_JOBS_SUBDIR}/{job}/assembled/{f}",
                          "report": rep, "abs": full, "mtime": os.path.getmtime(full)})
        if items:
            items.sort(key=lambda a: a["mtime"], reverse=True)
            jobs.append({"job": job, "assemblies": items, "mtime": items[0]["mtime"]})
    jobs.sort(key=lambda j: j["mtime"], reverse=True)
    return jobs


def _public_assembly(a):
    return {k: a[k] for k in ("job", "file", "path", "mtime")}


def find_swap_assembly(path, out_dir=None):
    """The assembly a sequence path names, or None. The path is only ever compared with the
    assemblies swap_assemblies() listed (as the output-relative line, or as that file's
    absolute path): nothing the client sends is opened, so '..', another folder or a file
    outside a job's assembled/ folder simply matches nothing."""
    if not isinstance(path, str) or not path.strip():
        return None
    want = path.strip().replace("\\", "/")
    want_abs = os.path.normcase(os.path.normpath(path.strip())) if os.path.isabs(path.strip()) else None
    for j in swap_assemblies(out_dir):
        for a in j["assemblies"]:
            if want == a["path"] or (want_abs and want_abs == os.path.normcase(os.path.normpath(a["abs"]))):
                return a
    return None


def swap_joins(path, out_dir=None):
    """An assembly's joins in ITS OWN frame numbers (a windowed assembly starts at the
    window's first source frame), from its report: frame, type, verdict and the facts
    behind them. Raises SequenceError for a path that is not a listed Swap assembly."""
    a = find_swap_assembly(path, out_dir)
    if a is None:
        raise tm.SequenceError("not a Swap job's assembly")
    with open(a["report"], encoding="utf-8") as f:
        rep = json.load(f)
    lo = (rep.get("window") or [0])[0] or 0
    n = rep.get("frames")
    joins = []
    for j in rep.get("joins") or []:
        fr = j.get("frame", j.get("splice"))
        if not isinstance(fr, (int, float)):
            continue
        fr = int(fr) - int(lo)
        if fr < 0 or (isinstance(n, int) and fr >= n):
            continue
        joins.append({"frame": fr, "type": j.get("type"), "verdict": j.get("verdict"),
                      "join_verdict": j.get("join_verdict"), "split": j.get("split"),
                      "left_chunk": j.get("left_chunk"), "right_chunk": j.get("right_chunk"),
                      "repair": j.get("repair")})
    joins.sort(key=lambda j: j["frame"])
    return dict(_public_assembly(a), frames=n, fps=rep.get("fps"), window=rep.get("window"), joins=joins)


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

def _json_error(e, status=400):
    return web.json_response({"error": str(e)}, status=status)


@PromptServer.instance.routes.get("/seamstitch/timeline/probe")
async def _probe_route(request):
    q = request.query
    try:
        path = resolve_path(q.get("path", ""))
        fr = int(q.get("frame_rate", "0") or 0)
        fr = auto_frame_rate(fr, path)
        info = probe(path, fr)
        info["path"] = path
        return web.json_response(info)
    except Exception as e:
        return _json_error(e)


@PromptServer.instance.routes.get("/seamstitch/timeline/list")
async def _list_route(request):
    files = [f for f in _list_input_videos() if f != "none" and not f.startswith(CUT_SUBDIR)]
    return web.json_response({"files": files})


@PromptServer.instance.routes.post("/seamstitch/timeline/build")
async def _build_route(request):
    try:
        body = await request.json()
        cut, fr = plan_cut(body.get("sequence", ""), int(body.get("frame_rate", 0) or 0))
        crf = int(body.get("crf", 12))
        fit = body.get("fit", "crop")
        codec = body.get("codec", CODEC_LOSSLESS)
        conform = bool(body.get("conform", False))
        import asyncio
        loop = asyncio.get_event_loop()
        path = await loop.run_in_executor(None, build_cut, cut, fr, crf, fit, codec, conform)
        play = await loop.run_in_executor(None, preview_copy, path)
        # frame_rate: how fast the cut plays (24 when conformed); frame numbers are the strip's either way.
        return web.json_response({"path": path, "play_path": play, "frames": cut["frames"],
                                  "frame_rate": conform_rate(fr, conform), "source_frame_rate": fr,
                                  "passthrough": path == passthrough_path(cut, fr)})
    except Exception as e:
        return _json_error(e)


@PromptServer.instance.routes.get("/seamstitch/timeline/swap_jobs")
async def _swap_jobs_route(request):
    try:
        return web.json_response({"jobs": [{"job": j["job"], "assemblies": [_public_assembly(a) for a in j["assemblies"]]}
                                           for j in swap_assemblies()]})
    except Exception as e:
        return _json_error(e)


@PromptServer.instance.routes.get("/seamstitch/timeline/swap_joins")
async def _swap_joins_route(request):
    try:
        return web.json_response(swap_joins(request.query.get("path", "")))
    except Exception as e:
        return _json_error(e, 404)


# ---------------------------------------------------------------------------
# the node
# ---------------------------------------------------------------------------

def picture_end_seconds(frame_count, frame_rate):
    """Time of the generator's last frame - where the pinned end frame (Picture 2) sits."""
    return round(max(0, int(frame_count) - 1) / float(frame_rate or 24), 2)


def picture_timing(end_seconds):
    """MiniMax H3's reference-alignment line, in the form its prompt builders write."""
    return ("How the reference pictures align with the target video \u2014 Picture 1 (from Shot 1) "
            "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns "
            f"with the {end_seconds:.2f}-second mark of the target video.")


class SeamStitchTimeline:
    """One-track timeline: arrange clips, mark one splice, feed the generator.

    Outputs are SeamStitchLoader's, in the same order, computed by the Loader
    itself on the assembled cut - so this node drops into any SeamStitch graph in
    place of Combine + Loader. source_video_path is the cut."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": ("STRING", {"default": "", "multiline": True, "tooltip":
                    "The strip, one entry per line (managed by the timeline UI; the pencil "
                    "button edits it as text). 'path @ enter..exit' plays frames enter..exit-1 of "
                    "a clip, '~ N' is an N-frame gap."}),
                "target": ("STRING", {"default": "", "multiline": False, "tooltip":
                    "What to regenerate (managed by the timeline UI): a marked range, a bridged "
                    "cut, or the gap."}),
                "frame_rate": ("INT", {"default": 0, "min": 0, "max": 120, "step": 1, "tooltip":
                    "Frame rate of the cut, and of every frame number on the strip. 0 = the first "
                    "clip's own rate. A single untouched clip at its own rate is spliced with no "
                    "re-encode before Recombine."}),
                "bridge_frame_grid": ([tm.GRID_LTX, tm.GRID_MINIMAX, tm.GRID_NONE], {"default": tm.GRID_LTX,
                    "tooltip": "The generator's frame grid. frame_count is rounded UP onto it for a "
                    "range/cut, to the NEAREST length for a gap (the Loader's rules)."}),
                "context_frames": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip":
                    "Motion guides, as on the Loader: K real frames either side of the marked range "
                    "go to start_context/end_context and frame_count grows by 2K. Wire the "
                    "context_frames output into Recombine."}),
                "extend_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip":
                    "Give the generator this many frames more than the marked range (the Loader's "
                    "extend_bridge), before grid rounding. The spliced video gets longer by the "
                    "difference."}),
                "snap_to_multiple": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8, "tooltip":
                    "As on the Loader: width/height outputs are rounded to this multiple."}),
                "mismatch_fit": (["crop", "pad"], {"default": "crop", "tooltip":
                    "When clips differ in size, each is fitted onto the FIRST clip's size: crop "
                    "fills and trims the overhang, pad fits inside with black bars."}),
                "assemble_crf": ("INT", {"default": 12, "min": 0, "max": 51, "step": 1, "tooltip":
                    "x264 quality of the assembled cut when cut_codec is h264 (lower = better, "
                    "bigger). Only used when the strip is more than one untouched clip."}),
                # Appended last: ComfyUI restores widget values by position.
                "cut_codec": ([CODEC_LOSSLESS, CODEC_H264], {"default": CODEC_LOSSLESS, "tooltip":
                    "How the strip is assembled when it is more than one untouched clip. lossless "
                    "(ffv1): the cut is exactly the decoded clips - no colour shift before the final "
                    "save - but large (~1.6 GB a minute at 832x1280). h264: small, one lossy "
                    "generation (about -1 luma level on 4:2:0). The browser always plays a small "
                    "H.264 copy."}),
                # Appended last, for the same positional reason.
                "conform_to_24fps": ("BOOLEAN", {"default": False, "tooltip":
                    "MiniMax H3 runs on a fixed 24 fps clock. On: frames kept 1:1 and labelled 24 fps, "
                    "audio slowed to match with pitch kept (25 fps → 4%). Turn Result Preview's "
                    "restore on to get the source frame rate and original audio back. Tested on 25 fps; "
                    "30 fps is 20% and untested."}),
            },
        }

    # The Loader's 18 outputs in the Loader's order (so the node drops in where the Loader
    # was), then two of its own, appended so no existing wire moves:
    #   end_seconds     time of the generator's LAST frame, (frame_count - 1) / frame_rate:
    #                   where the pinned end frame sits, i.e. when "Picture 2" (the last
    #                   frame, fed to MiniMaxH3ReferenceToVideo) must appear.
    #   picture_timing  the MiniMax prompt's alignment line with those numbers filled in -
    #                   concatenate it in front of the scene description. A hand-typed time
    #                   goes stale whenever the markers, gap, extension or grid change the
    #                   length (a 3.04 s left over from another seam put Picture 2 mid-clip
    #                   while the pins held it at 5.83 s, and the model cut at the end).
    # and two for the 24 fps conform, appended after those (wire both into Result Preview):
    #   source_frame_rate  the strip's own rate before any conform (frame_rate is the cut's:
    #                      24 when conformed). Equal to frame_rate when the conform is off.
    #   original_audio     the cut's sound at the source rate, unstretched, sample 0 under
    #                      frame 0. Result Preview needs it to give the restored video its
    #                      real audio: everything the cut carries (and so full_clip_audio,
    #                      Recombine's audio) is the slowed copy, and stretching that back
    #                      would be a second lossy generation of the whole track.
    RETURN_TYPES = SeamStitchLoader.RETURN_TYPES + ("FLOAT", "STRING", "INT", "AUDIO")
    RETURN_NAMES = SeamStitchLoader.RETURN_NAMES + ("end_seconds", "picture_timing", "source_frame_rate",
                                                    "original_audio")
    FUNCTION = "run"
    CATEGORY = "SeamStitch"
    DESCRIPTION = ("A one-track mini video editor for SeamStitch: arrange and trim clips, mark one "
                   "range, cut or gap to regenerate. Outputs are SeamStitch Loader's, computed on "
                   "the assembled cut.")

    @classmethod
    def IS_CHANGED(cls, sequence="", target="", frame_rate=0, **kw):
        h = hashlib.sha256(json.dumps([sequence, target, frame_rate, sorted(kw.items())],
                                      default=str).encode())
        try:
            for e in tm.parse_sequence(sequence):
                if e["kind"] == "clip":
                    st = os.stat(resolve_path(e["path"]))
                    h.update(f"{st.st_mtime_ns}:{st.st_size}".encode())
        except Exception:
            pass
        return h.hexdigest()

    def run(self, sequence, target, frame_rate, bridge_frame_grid, context_frames, extend_frames,
            snap_to_multiple, mismatch_fit, assemble_crf, cut_codec=CODEC_LOSSLESS, conform_to_24fps=False):
        cut, fr = plan_cut(sequence, frame_rate)
        plan = tm.resolve_target(tm.parse_target(target), cut)
        tm.validate_context(plan, cut, context_frames)
        path = build_cut(cut, fr, assemble_crf, mismatch_fit, cut_codec, conform_to_24fps)
        # The conformed cut has the same frames at the same indices, so the plan (markers,
        # range, context, grid rounding - all frame counts) is unchanged; only the clock is
        # H3's. The Loader reads the cut at the rate it is labelled.
        src_fr, fr = fr, conform_rate(fr, conform_to_24fps)

        common = dict(video=path, frame_rate=fr, display_mode="frames", start_time=0.0,
                      end_time=0.0, duration=0.0, snap_to_multiple=snap_to_multiple,
                      bridge_frame_grid=bridge_frame_grid, context_frames=int(context_frames))
        loader = SeamStitchLoader()
        if plan["mode"] == "insert at join":
            res = loader.load_video(start_frame=0, end_frame=0, duration_frames=plan["length"],
                                    mode=plan["mode"], join_frame=plan["join"],
                                    trim_each_side=plan["trim"], **common)
        else:
            extra = int(extend_frames) + int(plan.get("extra", 0))
            res = loader.load_video(start_frame=plan["start"], end_frame=plan["end"] + 1,
                                    duration_frames=0, extend_bridge=bridge_frame_grid != tm.GRID_NONE
                                    or extra > 0, extend_amount=float(extra),
                                    extend_unit="frames", mode=plan["mode"], **common)
        expect = tm.generator_frames(plan, bridge_frame_grid, context_frames, extend_frames)
        if int(res[3]) != expect:
            print(f"[SeamStitch] Timeline: warning - Loader reports frame_count {res[3]}, the "
                  f"timeline predicted {expect}.")
        if plan["mode"] == "replace range" and (int(res[8]), int(res[9])) != (plan["start"], plan["end"]):
            raise RuntimeError(f"Timeline/Loader disagree on the range: timeline {plan['start']}..{plan['end']}, "
                               f"Loader {res[8]}..{res[9]} on {path}")
        print(f"[SeamStitch] Timeline: {plan['mode']} on the cut ({cut['frames']} frames at {fr} fps"
              + (f", conformed from {src_fr} fps" if fr != src_fr else "")
              + f"), frames {plan['start']}..{plan['end']}, generator gets {res[3]} frames.")
        end_s = picture_end_seconds(int(res[3]), fr)
        print(f"[SeamStitch] Timeline: Picture 2 (the last frame) is at {end_s:.2f}s"
              + (f" on H3's {fr} fps clock." if fr != src_fr else "."))
        if fr != src_fr:
            wav, sr = cut_audio(cut, src_fr)
            original = {"waveform": torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0), "sample_rate": sr}
        else:
            original = res[13]   # full_clip_audio: the cut's own track, already at the source rate
        return tuple(res) + (end_s, picture_timing(end_s), src_fr, original)
