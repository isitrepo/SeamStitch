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
"""

import hashlib
import json
import os
import subprocess
import tempfile
import threading

import av
import numpy as np
import folder_paths
from aiohttp import web
from server import PromptServer

try:
    from . import timeline_math as tm
    from .loader import SeamStitchLoader, _list_input_videos
except ImportError:  # imported as a top-level module (tests)
    import timeline_math as tm
    from loader import SeamStitchLoader, _list_input_videos

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
    try:
        from av.video.reformatter import Colorspace, ColorRange
        cs, cr, dst = (Colorspace.ITU709 if max(w, h) >= 720 else Colorspace.ITU601,
                       ColorRange.MPEG, ColorRange.JPEG)
    except ImportError:
        cs, cr, dst = ("itu709" if max(w, h) >= 720 else "itu601", "mpeg", "jpeg")
    c_space = getattr(cc, 'colorspace', getattr(cc, 'color_space', None))
    if c_space is not None and getattr(c_space, "name", str(c_space)).upper() != "UNSPECIFIED" \
            and "unspecified" not in str(c_space).lower():
        cs = c_space
    c_range = getattr(cc, 'color_range', None)
    if c_range is not None and getattr(c_range, "name", str(c_range)).upper() != "UNSPECIFIED" \
            and "unspecified" not in str(c_range).lower():
        cr = c_range
    return cs, cr, dst


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
                    rgb = frame.to_ndarray(format="rgb24")
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


def _cut_key(cut, fr, crf, fit, codec=CODEC_LOSSLESS):
    h = hashlib.sha256()
    h.update(json.dumps([_ASSEMBLY_VERSION, fr, int(crf), fit, codec]).encode())
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


def build_cut(cut, fr, crf=12, fit="crop", codec=CODEC_LOSSLESS):
    """Assemble the cut (or pass the single source through). Returns the path.

    Lossless (default): FFV1, RGB planes, FLAC - the frames Recombine reads back are
    exactly the decoded sources (measured: an h264 4:2:0 cut at crf 12 shifted the
    picture -1.2 levels on Test vids/4.mp4, and that loss stacked with the final encode).
    h264: much smaller, one lossy generation."""
    same = passthrough_path(cut, fr)
    if same:
        return same
    lossless = codec != CODEC_H264
    out_dir = os.path.join(folder_paths.get_input_directory(), CUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"cut_{_cut_key(cut, fr, crf, fit, codec)}{'.mkv' if lossless else '.mp4'}")
    with _BUILD_LOCK:
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return out
        first = probe(cut["pieces"][0]["path"], fr)
        w, h = first["width"] - first["width"] % 2, first["height"] - first["height"] % 2
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
        wav = np.ascontiguousarray(wav[:, :want].T)

        fd, apath = tempfile.mkstemp(suffix=".f32", dir=out_dir)
        os.close(fd)
        tmp = os.path.splitext(out)[0] + (".part.mkv" if lossless else ".part.mp4")
        try:
            wav.astype("<f4").tofile(apath)
            cmd = [_ffmpeg_exe(), "-v", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fr), "-i", "-",
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
              f"{fr} fps, {w}x{h}, {codec} -> {out}")
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
        import asyncio
        loop = asyncio.get_event_loop()
        path = await loop.run_in_executor(None, build_cut, cut, fr, crf, fit, codec)
        play = await loop.run_in_executor(None, preview_copy, path)
        return web.json_response({"path": path, "play_path": play, "frames": cut["frames"],
                                  "frame_rate": fr, "passthrough": path == passthrough_path(cut, fr)})
    except Exception as e:
        return _json_error(e)


# ---------------------------------------------------------------------------
# the node
# ---------------------------------------------------------------------------

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
            },
        }

    RETURN_TYPES = SeamStitchLoader.RETURN_TYPES
    RETURN_NAMES = SeamStitchLoader.RETURN_NAMES
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
            snap_to_multiple, mismatch_fit, assemble_crf, cut_codec=CODEC_LOSSLESS):
        cut, fr = plan_cut(sequence, frame_rate)
        plan = tm.resolve_target(tm.parse_target(target), cut)
        tm.validate_context(plan, cut, context_frames)
        path = build_cut(cut, fr, assemble_crf, mismatch_fit, cut_codec)

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
        print(f"[SeamStitch] Timeline: {plan['mode']} on the cut ({cut['frames']} frames at {fr} fps), "
              f"frames {plan['start']}..{plan['end']}, generator gets {res[3]} frames.")
        return res
