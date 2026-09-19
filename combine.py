"""Copyright 2026 SeamStitch contributors (https://github.com/isitrepo/SeamStitch), licensed GPL-3.0-only.

SeamStitchCombine - concatenate two clips, no seam logic.

Deliberately dumb, and now actually fast: clip A followed by clip B, done via
ffmpeg's concat demuxer with ``-c copy`` - a container-level splice of the
existing encoded streams, not a decode/re-encode. No frame ever gets
touched, so there is no re-compression and no colour-space round-trip to
introduce a shift; the two source files must already share the same codec
parameters for this to work, which is the honest trade a stream copy makes
for being close to instant instead of a multi-second transcode.

The result is written to ComfyUI's input directory so it can be re-opened in
Load Video UI to hand-pick bridge start/end points - see simple_combine.js's
auto-select handler, which does that automatically once this node finishes
running. Clips are picked with file dropdowns (like Load Video UI's own
"video" widget) so this node is usable on its own, without wiring in two
separate Load Video UI nodes first.

The ``images``/``audio`` outputs still involve a real decode (ComfyUI's
IMAGE/AUDIO types are always tensors, so there is no way around that if
something downstream wants them directly) - but it is a plain decode of the
already-concatenated file, once, with no colour-space override and no
encode, which is a fraction of what the old decode-both-sides-then-re-encode
approach cost.
"""

import gc
import os
import subprocess

import numpy as np
import torch
import av
from aiohttp import web
from server import PromptServer

import comfy.model_management as model_management
import folder_paths

VIDEO_EXTENSIONS = ('.mp4', '.webm', '.mkv', '.avi', '.mov', '.m4v', '.flv', '.wmv')


def _safe_upload_filename(filename):
    """Reduce a client-supplied upload filename to a bare basename and reject
    anything that could escape the input directory (empty name, ".." segments,
    or an embedded path separator from either OS)."""
    if not filename:
        return None
    base = os.path.basename(filename)
    if not base or base in (".", ".."):
        return None
    if "/" in filename or "\\" in filename or ".." in filename:
        return None
    return base


def _next_free_path(directory, prefix, ext):
    os.makedirs(directory, exist_ok=True)
    counter = 1
    while True:
        candidate = os.path.join(directory, f"{prefix}_{counter:05}.{ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _list_input_videos():
    input_dir = folder_paths.get_input_directory()
    files = []
    if os.path.exists(input_dir):
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        wdc_dir = os.path.join(input_dir, "whatdreamscost")
        if os.path.exists(wdc_dir):
            all_files.extend(
                f"whatdreamscost/{f}" for f in os.listdir(wdc_dir)
                if os.path.isfile(os.path.join(wdc_dir, f))
            )
        try:
            files = folder_paths.filter_files_content_types(all_files, ["video"])
        except Exception:
            files = [f for f in all_files if f.lower().endswith(VIDEO_EXTENSIONS)]

        # Newest first, not alphabetical - makes a just-uploaded or just-written
        # file easy to spot in an input directory with hundreds of entries.
        def _mtime(f):
            try:
                return os.path.getmtime(os.path.join(input_dir, f))
            except OSError:
                return 0
        files = sorted(files, key=_mtime, reverse=True)
    return files or ["none"]


def _resolve_video_path(filename):
    if not filename or filename == "none":
        return None
    if os.path.exists(filename):
        return filename
    annotated = folder_paths.get_annotated_filepath(filename)
    if os.path.exists(annotated):
        return annotated
    in_input = os.path.join(folder_paths.get_input_directory(), filename)
    if os.path.exists(in_input):
        return in_input
    return None


def _probe_geometry(path):
    """Metadata only (codec, resolution, whether there's audio) - opening a
    container and reading its stream headers doesn't decode a single frame,
    so this is effectively instant regardless of clip length."""
    container = av.open(path)
    try:
        vstream = container.streams.video[0] if container.streams.video else None
        if vstream is None:
            raise RuntimeError(f"SeamStitchCombine: no video stream in {path}")
        return {
            "width": vstream.codec_context.width,
            "height": vstream.codec_context.height,
            "has_audio": len(container.streams.audio) > 0,
            "fps": round(float(vstream.average_rate), 3) if vstream.average_rate else None,
        }
    finally:
        container.close()


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _concat_stream_copy(path_a, path_b, out_path):
    """Splice two files together at the container level - ``-c copy`` means
    ffmpeg remuxes the existing encoded packets into one file rather than
    decoding and re-encoding them, so this is near-instant and there is
    nothing for it to introduce a quality or colour change in. Requires both
    clips to already share compatible codec parameters; refuses with
    ffmpeg's own reason rather than silently falling back to a transcode."""
    list_path = out_path + ".concat.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in (path_a, path_b):
            escaped = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    try:
        result = subprocess.run(
            [_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(
                "SeamStitchCombine: lossless concat failed - clip A and clip B "
                "are probably not stream-copy compatible (different codec, profile, "
                "or audio format). No fancy logic here - re-export them to matching "
                f"settings. ffmpeg said:\n{result.stderr[-2000:]}"
            )
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)


class _AudioDecodeError(RuntimeError):
    """Raised when the concatenated file's audio track fails to decode -
    signals combine() to retry with the audio re-encoded instead of
    stream-copied, rather than surfacing a raw PyAV traceback."""


def _concat_transcode(path_a, path_b, out_path, fps, sizes=None, target_size=None, fit="pad"):
    """Fallback for _concat_stream_copy: used whenever clip A and clip B
    aren't stream-copy compatible - either a differing audio sample rate
    (an AAC splice with mismatched rates decodes as invalid past the seam),
    a differing video frame rate, or (with ``target_size`` set) a differing
    resolution that the caller has chosen to resize past rather than refuse.

    The frame-rate case is the nastier one: ffmpeg's concat *demuxer*, in
    ``-c copy`` mode, doesn't raise an error for it at all - it just reuses
    clip A's per-frame duration for clip B's frames too, silently producing
    a file that plays clip B at the wrong speed and then holds on the last
    frame once it runs out. So this fallback doesn't touch the demuxer or
    try to salvage anything from ``_concat_stream_copy``'s output - it uses
    ffmpeg's concat *filter* instead, which operates on decoded frames and
    therefore retimes and resamples both video and audio correctly no
    matter which kind of mismatch caused the fast path to fail. The cost is
    a real transcode instead of a container-level splice.

    ``fps`` forces a single constant frame rate on the output (via
    ``-r``/``-fps_mode cfr``) rather than leaving each clip at its own
    native rate. Concat filter output would otherwise be VFR - fine for a
    smart player, but the rest of this project's pipeline (ffio.py) detects
    VFR and resamples to a blended average frame rate that matches neither
    source clip, which is what made clip A's segment look choppy once it
    got that far. Emitting true CFR here instead of leaving it to be
    guessed downstream avoids that.

    ``target_size``, when given along with ``sizes`` (the ``(width, height)``
    of clip A and clip B, in that order, from the caller's own
    ``_probe_geometry`` calls), is the ``(width, height)`` both inputs must
    end up at before the concat filter (which otherwise refuses to join
    streams of different size). Each input already at ``target_size`` is
    passed straight into the filtergraph unscaled; the other gets scaled
    (aspect ratio kept, never stretched) to reach it, by one of two
    strategies picked by ``fit``:

    - ``"pad"``: scale down to fit *within* the target box, then letterbox
      the leftover with black. Keeps every source pixel, at the cost of
      visible bars whenever the two aspect ratios don't already match -
      including a near-miss, where a barely-there mismatch (e.g. 16:9 vs.
      a generator's slightly-off native ratio) still produces a thin but
      visible bar popping in right at the seam.
    - ``"crop"``: scale up to *cover* the target box, then crop the
      overhang. No bars ever, at the cost of losing a sliver off two edges
      - for a small mismatch that sliver is imperceptible, which is the
      common case when the two clips are close to the same aspect ratio
      already (e.g. two different generators' "16:9-ish" outputs) rather
      than genuinely different framings."""
    filter_parts = []
    video_labels = []
    for i in (0, 1):
        size = sizes[i] if sizes else None
        if target_size and size != target_size:
            w, h = target_size
            if fit == "crop":
                filter_parts.append(
                    f"[{i}:v:0]scale={w}:{h}:force_original_aspect_ratio=increase,"
                    f"crop={w}:{h},setsar=1[v{i}]"
                )
            else:
                filter_parts.append(
                    f"[{i}:v:0]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[v{i}]"
                )
            video_labels.append(f"[v{i}]")
        else:
            video_labels.append(f"[{i}:v:0]")
    concat_inputs = "".join(f"{video_labels[i]}[{i}:a:0]" for i in range(2))
    filter_complex = ";".join(filter_parts + [f"{concat_inputs}concat=n=2:v=1:a=1[outv][outa]"])

    result = subprocess.run(
        [_ffmpeg_exe(), "-y", "-i", path_a, "-i", path_b,
         "-filter_complex", filter_complex,
         "-map", "[outv]", "-map", "[outa]",
         "-fps_mode", "cfr", "-r", f"{fps:.5f}",
         "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
         "-c:a", "aac", "-b:a", "192k", out_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            "SeamStitchCombine: transcoding clip A and clip B together "
            f"failed. ffmpeg said:\n{result.stderr[-2000:]}"
        )


def _decode_for_outputs(path):
    """Plain decode of the already-concatenated file, for the images/audio
    outputs - no colour-space override (trusting ffmpeg/PyAV's own default
    YUV->RGB conversion rather than a heuristic guess that can be wrong) and
    no re-encode, since the file itself is already final."""
    container = av.open(path)
    try:
        vstream = container.streams.video[0] if container.streams.video else None
        if vstream is None:
            raise RuntimeError(f"SeamStitchCombine: no video stream in {path}")
        vstream.thread_type = "AUTO"

        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(vstream)]
        if not frames:
            raise RuntimeError(f"SeamStitchCombine: no frames decoded from {path}")
        images = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
    finally:
        container.close()

    audio_dict = None
    if len(av.open(path).streams.audio) > 0:
        acontainer = av.open(path)
        try:
            astream = acontainer.streams.audio[0]
            astream.thread_type = "AUTO"
            sample_rate = getattr(astream, "rate", 44100) or 44100
            resampler = av.AudioResampler(format="fltp")
            chunks = []
            try:
                for frame in acontainer.decode(astream):
                    for r_frame in resampler.resample(frame):
                        chunks.append(r_frame.to_ndarray())
            except av.error.FFmpegError as exc:
                raise _AudioDecodeError(str(exc)) from exc
            if chunks:
                waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).float().unsqueeze(0)
                audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
        finally:
            acontainer.close()

    return images, audio_dict


# Mean absolute per-pixel difference (0-255 scale) allowed between a source
# clip's boundary frame and the matching frame of the joined output, compared
# as 64x64 thumbnails. Measured 2026-09-19 on the true pair: stream-copy
# 4.mp4+2.mp4 = 0.00/0.00 (A-last/B-first); transcode (2.mp4 retimed to 24 fps)
# = 0.63/0.21; transcode with resize_to=match_a, 4.mp4+1.mp4 = 0.65/1.19. A
# pairing one frame off across the cut measured 62.5-64.0 on that footage, so
# 6.0 sits well clear of both. (A smooth, cut-free join cannot be told apart
# from its neighbours by any picture test - inherent, not a tolerance issue.)
SEAM_TOLERANCE = 6.0
SEAM_SEARCH = 2  # frames either side of the computed candidate to also try


def _thumb(frame_rgb, size=64):
    import cv2
    return cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def _fit_frame(frame_rgb, target_size, fit):
    """Mirror _concat_transcode's scale/crop|pad so a resized clip's boundary
    frame can be compared with what actually landed in the output."""
    import cv2
    h, w = frame_rgb.shape[:2]
    tw, th = target_size
    if (w, h) == (tw, th):
        return frame_rgb
    if fit == "crop":
        k = max(tw / w, th / h)
        nw, nh = max(tw, round(w * k)), max(th, round(h * k))
        r = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        x0, y0 = (nw - tw) // 2, (nh - th) // 2
        return r[y0:y0 + th, x0:x0 + tw]
    k = min(tw / w, th / h)
    nw, nh = max(1, round(w * k)), max(1, round(h * k))
    r = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=frame_rgb.dtype)
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def _boundary_frames(path):
    """(first_frame, last_frame, frame_count) of a clip, decoded as RGB."""
    container = av.open(path)
    try:
        first = last = None
        count = 0
        for frame in container.decode(container.streams.video[0]):
            last = frame.to_ndarray(format="rgb24")
            if first is None:
                first = last
            count += 1
    finally:
        container.close()
    if first is None:
        raise RuntimeError(f"SeamStitchCombine: no frames decoded from {path}")
    return first, last, count


def _output_fps(path):
    container = av.open(path)
    try:
        v = container.streams.video[0]
        return float(v.average_rate)
    finally:
        container.close()


def _measure_seam_frame(path_a, path_b, out_images, out_fps, target_size=None, fit="crop",
                        tolerance=SEAM_TOLERANCE):
    """Index of clip B's first frame in the combined output, measured, not
    trusted: candidate = A's duration x the output's real frame rate, then
    output frames seam-1 / seam must reproduce A's true last frame and B's
    true first frame. A few neighbouring candidates are tried too, since the
    transcode path can shift A's frame count by one. Raises if none fits."""
    a_first, a_last, a_count = _boundary_frames(path_a)
    b_first, _, _ = _boundary_frames(path_b)
    ca = av.open(path_a)
    try:
        v = ca.streams.video[0]
        a_fps = float(v.average_rate)
    finally:
        ca.close()
    if target_size:
        a_last = _fit_frame(a_last, target_size, fit)
        b_first = _fit_frame(b_first, target_size, fit)
    ta, tb = _thumb(a_last), _thumb(b_first)
    n = out_images.shape[0]
    candidate = int(round(a_count / a_fps * out_fps))

    def err(idx, ref):
        if not 0 <= idx < n:
            return float("inf")
        out = (out_images[idx].numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        return float(np.abs(_thumb(out) - ref).mean())

    tried = []
    order = [candidate] + [candidate + d for k in range(1, SEAM_SEARCH + 1) for d in (-k, k)]
    for seam in order:
        if seam < 1:
            continue
        ea, eb = err(seam - 1, ta), err(seam, tb)
        tried.append((seam, ea, eb))
        if ea <= tolerance and eb <= tolerance:
            return seam
    detail = "; ".join(f"seam {s}: A-last diff {ea:.2f}, B-first diff {eb:.2f}" for s, ea, eb in tried)
    raise ValueError(
        f"SeamStitchCombine: could not verify where clip B starts in the combined "
        f"file (A='{os.path.basename(path_a)}', B='{os.path.basename(path_b)}', "
        f"{n} output frames, candidate {candidate}, tolerance {tolerance}). "
        f"Measured differences - {detail}. Refusing to emit an unverified seam_frame."
    )


# --- API routes: file upload/check (mirrors Load Video UI's own pattern, kept
# separate so this package has no hard dependency on that one) and a fast
# header-only probe so the node's "Load Video" button can confirm a pick is
# valid without paying for a full decode. ---

@PromptServer.instance.routes.get("/seamstitch/combine/check_file")
async def seamstitch_check_file(request):
    filename = request.query.get("filename", "")
    file_size = request.query.get("size", "")
    if not filename:
        return web.json_response({"exists": False})
    upload_dir = folder_paths.get_input_directory()
    candidate = os.path.join(upload_dir, filename)
    if os.path.exists(candidate) and os.path.isfile(candidate):
        if file_size:
            try:
                if os.path.getsize(candidate) == int(file_size):
                    return web.json_response({"exists": True, "name": filename})
            except ValueError:
                return web.json_response({"exists": True, "name": filename})
        else:
            return web.json_response({"exists": True, "name": filename})
    return web.json_response({"exists": False})


@PromptServer.instance.routes.post("/seamstitch/combine/upload_chunk")
async def seamstitch_upload_chunk(request):
    post = await request.post()
    file = post.get("file")
    filename = _safe_upload_filename(post.get("filename"))
    if filename is None:
        return web.Response(status=400, text="Invalid filename")
    chunk_index = int(post.get("chunk_index"))
    total_chunks = int(post.get("total_chunks"))

    upload_dir = folder_paths.get_input_directory()
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    mode = "ab" if chunk_index > 0 else "wb"
    with open(file_path, mode) as f:
        f.write(file.file.read())

    if chunk_index == total_chunks - 1:
        return web.json_response({"name": filename})
    return web.json_response({"status": "ok"})


@PromptServer.instance.routes.get("/seamstitch/combine/probe")
async def seamstitch_probe(request):
    filename = request.query.get("filename", "")
    path = _resolve_video_path(filename)
    if not path:
        return web.json_response({"ok": False, "error": "File not found"}, status=404)
    try:
        container = av.open(path)
        try:
            vstream = container.streams.video[0] if container.streams.video else None
            has_audio = len(container.streams.audio) > 0
            width = vstream.codec_context.width if vstream else 0
            height = vstream.codec_context.height if vstream else 0
            duration = 0.0
            if vstream is not None and vstream.duration and vstream.time_base:
                duration = float(vstream.duration * vstream.time_base)
            frame_count = vstream.frames if vstream else 0
            fps = float(vstream.average_rate) if vstream and vstream.average_rate else 0.0
        finally:
            container.close()
        return web.json_response({
            "ok": True, "width": width, "height": height, "duration": duration,
            "frame_count": frame_count, "fps": fps, "has_audio": has_audio,
        })
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


class SeamStitchCombine:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_input_videos()
        return {
            "required": {
                "video_a": (files,),
                "video_b": (files,),
                "resize_to": (["off", "match_a", "match_b"], {"default": "off", "tooltip":
                    "If clip A and clip B aren't the same resolution: 'off' "
                    "refuses (the honest default - re-export upstream instead). "
                    "'match_a'/'match_b' resizes the other clip onto whichever "
                    "one you picked (never stretched - see resize_fit). Forces "
                    "a real transcode either way, since a resized clip can no "
                    "longer be stream-copied."}),
                "resize_fit": (["crop", "pad"], {"default": "crop", "tooltip":
                    "Only matters when resize_to isn't 'off'. 'crop' scales the "
                    "other clip to fill the target frame and crops the overhang "
                    "- no bars, at the cost of a sliver off two edges; the right "
                    "choice when the two aspect ratios are already close (e.g. "
                    "two generators' slightly different 16:9-ish outputs), which "
                    "is the common case. 'pad' scales it to fit inside the frame "
                    "and letterboxes the rest with black - keeps every pixel, at "
                    "the cost of a visible bar popping in at the seam, even for "
                    "a near-miss aspect ratio; use it when the two clips are "
                    "genuinely differently framed and cropping would cut off "
                    "something that matters."}),
                "filename_prefix": ("STRING", {"default": "seamstitch_combined"}),
                "free_vram_first": ("BOOLEAN", {"default": True, "tooltip":
                    "Unload all models and clear the VRAM/CUDA cache before "
                    "combining - this node has no upstream dependencies, so "
                    "it's usually the first thing to run in a workflow and a "
                    "reasonable place to free whatever the previous run left "
                    "loaded before the heavier generative-bridge backends "
                    "need it."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("images", "audio", "seam_frame")
    FUNCTION = "combine"
    CATEGORY = "SeamStitch"
    # Writes a real file as a side effect (like a save node), so it's a valid
    # terminal action on its own - lets "bypass everything downstream, run just
    # this and Load Video UI" work at all. Without this, ComfyUI's own prompt
    # validator refuses to queue a subgraph that contains no OUTPUT_NODE at all
    # ("Prompt has no outputs"), since nothing in it would be considered worth
    # actually running.
    OUTPUT_NODE = True

    def combine(self, video_a, video_b, resize_to, resize_fit, filename_prefix, free_vram_first=True):
        if free_vram_first:
            # Same sequence ComfyUI's own queue-level "free memory" flag uses
            # between prompt runs (main.py) - unload everything the model
            # manager is tracking, then collect and empty the CUDA cache.
            model_management.unload_all_models()
            gc.collect()
            model_management.soft_empty_cache()

        path_a = self._resolve(video_a, "A")
        path_b = self._resolve(video_b, "B")

        geom_a, geom_b = _probe_geometry(path_a), _probe_geometry(path_b)
        size_a = (geom_a["width"], geom_a["height"])
        size_b = (geom_b["width"], geom_b["height"])
        size_mismatch = size_a != size_b
        if size_mismatch and resize_to == "off":
            raise RuntimeError(
                "SeamStitchCombine: clip A and clip B don't match "
                f"({geom_a['width']}x{geom_a['height']} vs {geom_b['width']}x{geom_b['height']}). "
                "Either re-export upstream so both sides agree, or set "
                "resize_to to 'match_a'/'match_b' to resize the other "
                "clip onto that size."
            )
        target_size = {"match_a": size_a, "match_b": size_b}.get(resize_to) if size_mismatch else None
        if geom_a["has_audio"] != geom_b["has_audio"]:
            raise RuntimeError(
                "SeamStitchCombine: clip A and clip B don't both have audio "
                f"(A: {'yes' if geom_a['has_audio'] else 'no'}, B: {'yes' if geom_b['has_audio'] else 'no'}). "
                "A lossless concat needs both sides to match - add a silent audio "
                "track to the one missing it, or strip audio from the one that has it."
            )

        input_dir = folder_paths.get_input_directory()
        out_path = _next_free_path(input_dir, filename_prefix, "mp4")

        # A frame-rate mismatch never raises on its own - ffmpeg's concat
        # demuxer just silently mis-times the second clip's frames in
        # -c copy mode (plays at the wrong speed, then freezes once it runs
        # out) - so this has to be caught upfront rather than by reacting to
        # an exception like the audio case below.
        # Clip A sets the target frame rate for the transcode fallback - it's
        # the leading clip, and matches this node's own "A followed by B"
        # framing.
        fps_target = geom_a["fps"] or geom_b["fps"] or 24.0
        fps_mismatch = geom_a["fps"] and geom_b["fps"] and geom_a["fps"] != geom_b["fps"]
        # A resize always goes straight to the transcode path rather than
        # trying _concat_stream_copy first - a scaled/cropped-or-padded clip
        # is a fresh encode by definition, so there was never a chance of a
        # lossless stream copy succeeding for it.
        if fps_mismatch or target_size:
            _concat_transcode(path_a, path_b, out_path, fps_target,
                               sizes=(size_a, size_b), target_size=target_size, fit=resize_fit)
            images, audio = _decode_for_outputs(out_path)
        else:
            _concat_stream_copy(path_a, path_b, out_path)
            try:
                images, audio = _decode_for_outputs(out_path)
            except _AudioDecodeError:
                _concat_transcode(path_a, path_b, out_path, fps_target)
                images, audio = _decode_for_outputs(out_path)

        # Where clip B's first frame landed, measured from the written file and
        # verified against both clips' true boundary frames (raises otherwise).
        seam_frame = _measure_seam_frame(
            path_a, path_b, images, _output_fps(out_path),
            target_size=target_size, fit=resize_fit)
        print(f"[SeamStitchCombine] seam_frame = {seam_frame} "
              f"(clip B starts at output frame {seam_frame} of {images.shape[0]})")

        # The written file's full resolved path goes to the frontend via "ui" only
        # (the same channel VHS_VideoCombine uses to refresh its own preview after a
        # render, with no button click needed) - js/combine.js listens for this and
        # auto-selects the file in any downstream Load Video UI node, so picking
        # bridge points can be a bypass-and-run rather than a
        # run-then-hunt-through-the-dropdown step.
        #
        # It is deliberately NOT an output socket. It used to be one, for wiring into
        # SeamStitchRecombine.original_video_path, but that auto-select means the
        # downstream Loader is always pointing at this very file - so its own
        # source_video_path already carries the identical path, and every shipped
        # template wires that one. Two sockets for one string was the only thing the
        # extra output bought.
        return {"ui": {"video_path": [out_path]}, "result": (images, audio, seam_frame)}

    def _resolve(self, video, label):
        path = _resolve_video_path(video)
        if path is None:
            raise RuntimeError(
                f"SeamStitchCombine: clip {label}'s file ('{video}') was not found. "
                "Choose a file (or upload one) in the node."
            )
        return path
