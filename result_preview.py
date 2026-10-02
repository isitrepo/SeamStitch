"""SeamStitchResultPreview: the spliced result, shown around its splice, with a verdict.

Wire Recombine's Filenames in, plus the Timeline's (or Loader's) source_video_path /
start_frame / end_frame / frame_rate. After the run the node plays the result with
the regenerated span marked, and rates each of its two joins:

    ratio = the picture change across the join / the typical change around it

on small grey thumbnails (codec noise averages out). ~1 means the join moves like
the footage around it; a hard cut reads many times higher. The same measurement
is taken on the ORIGINAL cut over the replaced range, so the node can say "before:
hard cut 9.8x -> after: seamless 1.1x".

"use as timeline" in the node's UI puts the result back onto the Timeline node as
its only clip, so the next splice starts from this one - one splice at a time,
as many times as it takes.

SAVING. Wire Recombine's combined_images / audio into `images` / `audio` and this
node writes the final video itself, through the very encode path VHS Video Combine
uses (recombine._encode_video is a fork of it): RGB in, tagged as BT.709, converted
with scale=out_color_matrix=bt709 and written with bt709 primaries/trc/matrix and
tv range, plus the first-frame PNG carrying the workflow. So it replaces a VHS
Video Combine "Final Video" node with no change in colour. Set Recombine's
skip_encode on so the splice is not encoded twice. Formats a browser cannot play
(ProRes, FFV1, H.265) get a small H.264 proxy in temp for the player; the analysis
and the saved file are the real one. "save frame" in the UI writes the frame under
the playhead as a PNG, colour-converted from the saved file - for first/last-frame
references, instead of a screen grab through the browser's player.

RESTORE (the Timeline's conform_to_24fps). Wire the Timeline's source_frame_rate and
original_audio in. When source_frame_rate differs from frame_rate (the cut was conformed
to H3's 24 fps clock) and restore_source_frame_rate is on, the saved video is the same
frames played at the source rate, so it is exactly as long as the source cut for a
length-keeping splice. Its sound: the ORIGINAL audio, sample for sample, everywhere
outside the regenerated span; under the new frames, whatever Recombine put there (the
bridge's own audio, or the source's), sped back up by source_fps/24 with the pitch kept;
the two joins crossfaded as Recombine does (audio_splice.splice_audio). Given filenames
instead of images, the video stream is re-labelled by stream copy (no re-encode of the
picture) into <name>_<fps>fps beside it. The rating and the player use the restored file.

"Typical" is the 75th percentile of the frame-to-frame changes in the 24 frames
either side, not the median: footage whose frames repeat in pairs (24 fps content
in a 48 fps file - test clip 1.mp4 is) has every other change near zero, so a
median baseline called ordinary motion a 40x cut. With the 75th percentile the
natural worst step of all four real test clips is 1.27-1.66x, and the 4.mp4 ->
2.mp4 hard cut reads 6.5x. Thresholds (pixel domain): < 1.8 seamless, < 3.0 soft
bump, else hard cut.
"""

import datetime
import os
import re
import subprocess

import av
import numpy as np
import folder_paths
from aiohttp import web
from server import PromptServer

import torch

try:
    from . import timeline as tl
    from .audio_splice import frame_sample, match_format, splice_audio
except ImportError:
    import timeline as tl
    from audio_splice import frame_sample, match_format, splice_audio

SEAMLESS, SOFT = 1.8, 3.0
BASELINE_PCT = 75
WINDOW = 24            # frames either side of a join used for the baseline
THUMB_W = 96


def verdict(ratio):
    if ratio is None:
        return "n/a"
    return "seamless" if ratio < SEAMLESS else ("soft bump" if ratio < SOFT else "hard cut")


def _thumbs(path, fr, start, end):
    """Grey float thumbnails of frames start..end-1 (clamped to the clip)."""
    start = max(0, start)
    out = []
    for f in tl._iter_frames(path, fr, start, end):
        h, w = f.shape[:2]
        step = max(1, w // THUMB_W)
        g = f[::step, ::step].astype(np.float32)
        out.append(g @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    return start, out


def join_ratio(path, fr, join, window=WINDOW):
    """Ratio of the change INTO frame `join` to the typical (75th percentile) change of the frames
    around it. None when there is not enough footage either side."""
    s, th = _thumbs(path, fr, join - window, join + window)
    k = join - s
    if k < 1 or k >= len(th):
        return None, None
    d = np.array([np.abs(th[i] - th[i - 1]).mean() for i in range(1, len(th))])
    step = float(d[k - 1])
    rest = np.delete(d, k - 1)
    base = float(np.percentile(rest, BASELINE_PCT)) if rest.size else 0.0
    return step / max(base, 0.5), step


def worst_in_range(path, fr, lo, hi, window=WINDOW):
    """Worst join ratio over frames lo..hi (the original cut's replaced span): where
    the thing being fixed was."""
    s, th = _thumbs(path, fr, lo - window, hi + window + 1)
    if len(th) < 3:
        return None, None
    d = np.array([np.abs(th[i] - th[i - 1]).mean() for i in range(1, len(th))])
    # d[i-1] is the change into absolute frame s + i
    best = (None, None)
    for j in range(max(lo, s + 1), min(hi + 1, s + len(th))):
        k = j - s
        rest = np.delete(d, k - 1)
        r = float(d[k - 1]) / max(float(np.percentile(rest, BASELINE_PCT)), 0.5)
        if best[0] is None or r > best[0]:
            best = (r, j)
    return best


def _result_path(filenames):
    files = filenames[1] if isinstance(filenames, (list, tuple)) and len(filenames) > 1 else filenames
    if isinstance(files, str):
        files = [files]
    vids = [f for f in files or [] if str(f).lower().endswith(tl._VIDEO_EXTENSIONS)]
    if not vids:
        raise ValueError("Filenames holds no video - wire SeamStitch Recombine's Filenames output")
    with_audio = [f for f in vids if os.path.splitext(f)[0].endswith("-audio")]
    return (with_audio or vids)[-1]


def analyse(result, source, fr, start, end, source_fr=None):
    """Everything the UI shows. start/end: the replaced range on the source (end =
    start - 1 when nothing was removed). source_fr: the rate the source is read at when
    it differs from the result's (a restored result at 25 fps against its 24 fps
    conformed cut - same frame numbers, different clocks)."""
    sfr = source_fr or fr
    r_frames = tl.probe(result, fr)["frames"]
    s_frames = tl.probe(source, sfr)["frames"]
    removed = max(0, end - start + 1)
    inserted = r_frames - (s_frames - removed)
    joins = [start, start + inserted]
    rows = []
    for name, j in (("into the new frames", joins[0]), ("back to the footage", joins[1])):
        ratio, step = join_ratio(result, fr, j) if 0 < j < r_frames else (None, None)
        rows.append({"name": name, "frame": j, "ratio": ratio, "verdict": verdict(ratio)})
    # The two joins can both be clean while the new frames themselves still cut
    # (a generator that ignored its prompt, or a bridge that is the old frames):
    # rate the worst step INSIDE the regenerated span as well.
    inside, at = (worst_in_range(result, fr, joins[0] + 1, joins[1] - 1)
                  if joins[1] - joins[0] >= 3 else (None, None))
    rows.append({"name": "inside the new frames", "frame": at if at is not None else joins[0],
                 "ratio": inside, "verdict": verdict(inside)})
    before, where = (worst_in_range(source, sfr, max(1, start), min(s_frames - 1, max(start, end)))
                     if s_frames > 2 else (None, None))
    return {"frames": r_frames, "source_frames": s_frames, "inserted": inserted,
            "removed": removed, "joins": rows,
            "before": {"ratio": before, "frame": where, "verdict": verdict(before)}}


def _view_params(path):
    """/view query for a file under output, input or temp, else None (served through
    the Loader's own file route instead)."""
    for kind, base in (("output", folder_paths.get_output_directory()),
                       ("input", folder_paths.get_input_directory()),
                       ("temp", folder_paths.get_temp_directory())):
        base = os.path.abspath(base)
        ap = os.path.abspath(path)
        if ap.lower().startswith(base.lower() + os.sep):
            rel = os.path.relpath(ap, base)
            sub, name = os.path.split(rel)
            return {"filename": name, "subfolder": sub.replace("\\", "/"), "type": kind}
    return None


# Container/codec pairs Chrome plays natively; anything else gets an H.264 proxy.
_PLAYABLE = {"h264-mp4", "nvenc_h264-mp4", "webm", "av1-webm", "nvenc_av1-mp4"}
_DATE = re.compile(r"%date:([^%]+)%")


def _expand_date(prefix):
    """VHS-style %date:yyyyMMdd_hhmmss% in filename_prefix, expanded server side so it
    works the same queued from the UI or the API."""
    def sub(m):
        f = m.group(1)
        for a, b in (("yyyy", "%Y"), ("yy", "%y"), ("MM", "%m"), ("dd", "%d"),
                     ("hh", "%H"), ("mm", "%M"), ("ss", "%S")):
            f = f.replace(a, b)
        return datetime.datetime.now().strftime(f)
    return _DATE.sub(sub, prefix)


def _formats():
    """VHS's own video formats (recombine.pack_video_formats: no other pack's `.json` copies,
    which can save without BT.709 tags), image sequences left out."""
    try:
        try:
            from .recombine import pack_video_formats
        except ImportError:
            from recombine import pack_video_formats
        names = [f for f in pack_video_formats()[0] if "png" not in f]
    except Exception:
        names = []
    return names or ["video/h264-mp4"]


def format_settings(fmt, crf, pix_fmt, save_metadata, widgets=None):
    """The node's settings, handed only to formats that have them. pix_fmt must not
    reach FFV1 (its default rgba64le is 16-bit RGB - no YUV conversion, truly lossless)
    and ProRes is written as 4444 (4:4:4 10-bit) rather than VHS's default hq (4:2:2):
    4:2:0/4:2:2 chroma is what costs ~1-2 levels of colour on re-encode (measured
    -1.2 mean on test clip 4.mp4 even at crf 0, -0.1 at 4:4:4)."""
    if widgets is None:
        try:
            try:
                from .recombine import get_video_formats
            except ImportError:
                from recombine import get_video_formats
            widgets = get_video_formats()[1].get(fmt, [])
        except Exception:
            widgets = []
    names = {w[0]: w for w in widgets}
    # has_alpha: ProRes picks its pix_fmt from it (VHS Video Combine always supplies it);
    # the frames reaching here are RGB, alpha is stripped before encoding.
    out = {"save_metadata": bool(save_metadata), "trim_to_audio": False, "has_alpha": False}
    if "crf" in names:
        out["crf"] = int(crf)
    if "pix_fmt" in names and isinstance(names["pix_fmt"][1], list) and pix_fmt in names["pix_fmt"][1] \
            and "ffv1" not in fmt:
        out["pix_fmt"] = pix_fmt
    if "profile" in names and "ProRes" in fmt:
        out["profile"] = "4444"
    return out


def _encode(images, audio, fr, prefix, fmt, save_output, format_kwargs, prompt, extra_pnginfo):
    """Write the video through VHS's own encode path (recombine._encode_video)."""
    try:
        from . import recombine as rc
    except ImportError:
        import recombine as rc
    if images.shape[-1] == 4:
        images = images[..., :3]
    res = rc._encode_video(images, fr, _expand_date(prefix), fmt, save_output, audio,
                           prompt, extra_pnginfo, format_kwargs=format_kwargs)
    return res["result"][0]


# ---------------------------------------------------------------------------
# restore: a result made on the Timeline's 24 fps conform, back at the source rate
# ---------------------------------------------------------------------------

def restore_audio(spliced, original, clip_fr, src_fr, start, end, r_frames, s_frames, crossfade_ms=20.0):
    """The restored video's sound, exactly r_frames/src_fr long. Outside the regenerated
    span it IS `original` (the Timeline's original_audio: the cut's unstretched track),
    sample for sample. Under the new frames it is `spliced` (Recombine's audio, on the
    conformed cut's clip_fr clock) over the same frames, sped back up by src_fr/clip_fr
    with the pitch kept - so a bridge's own audio survives. The joins get Recombine's
    equal-power crossfade. Returns (AUDIO dict or None, notes)."""
    removed = max(0, end - start + 1)
    inserted = max(0, r_frames - (s_frames - removed))
    tempo = src_fr / float(clip_fr)
    if original is None:
        if spliced is None:
            return None, []
        sr = int(spliced["sample_rate"])
        w = spliced["waveform"][0].detach().to("cpu", torch.float32).numpy()
        out = tl.time_stretch(w, sr, tempo, frame_sample(r_frames, src_fr, sr))
        return ({"waveform": torch.from_numpy(out).unsqueeze(0), "sample_rate": sr},
                [f"original_audio not wired: the whole track was sped back up x{tempo:.4f} "
                 f"(wire the Timeline's original_audio for the real sound)"])
    sr = int(original["sample_rate"])
    orig = original["waveform"][0].detach().to("cpu", torch.float32)
    bridge = None
    if spliced is not None and inserted > 0:
        sp = match_format(spliced["waveform"][0], spliced["sample_rate"], orig.shape[0], sr)
        a, b = frame_sample(start, clip_fr, sr), frame_sample(start + inserted, clip_fr, sr)
        seg = torch.zeros((orig.shape[0], b - a))
        got = sp[:, a:b]
        seg[:, :got.shape[1]] = got
        n = frame_sample(start + inserted, src_fr, sr) - frame_sample(start, src_fr, sr)
        bridge = torch.from_numpy(tl.time_stretch(seg.numpy(), sr, tempo, n))
    out, notes = splice_audio(orig, sr, src_fr, start, end, 0, inserted, bridge=bridge,
                              crossfade_ms=crossfade_ms, insert=removed == 0)
    want = frame_sample(r_frames, src_fr, sr)
    if out.shape[1] < want:
        out = torch.nn.functional.pad(out, (0, want - out.shape[1]))
    return {"waveform": out[:, :want].contiguous().unsqueeze(0), "sample_rate": sr}, notes


_AUDIO_CODEC = {".mp4": ["-c:a", "aac"], ".m4v": ["-c:a", "aac"], ".mov": ["-c:a", "aac"],
                ".webm": ["-c:a", "libopus"], ".mkv": ["-c:a", "flac"]}


def restore_file(path, source, clip_fr, src_fr, start, end, original):
    """A video already written on the conformed clock, re-labelled at src_fr: the video
    stream is COPIED with frame n re-timed to n/src_fr (timeline.relabel_video, no
    re-encode), the sound rebuilt by restore_audio from the file's own track. Written
    beside it as <name>_<src_fr>fps<ext>. Returns the new path."""
    info = tl.probe(path, clip_fr)
    r_frames = info["frames"]
    s_frames = tl.probe(source, clip_fr)["frames"]
    sr = int(original["sample_rate"]) if original is not None else (info["sample_rate"] or 48000)
    spliced = None
    if info["has_audio"]:
        w = tl._clip_audio(path, info["base_time"], info["base_time"] + r_frames / float(clip_fr), sr)
        spliced = {"waveform": torch.from_numpy(w).unsqueeze(0), "sample_rate": sr}
    audio, notes = restore_audio(spliced, original, clip_fr, src_fr, start, end, r_frames, s_frames)
    for n in notes:
        print(f"[SeamStitch] Result: restore audio: {n}")
    stem, ext = os.path.splitext(path)
    out = f"{stem}_{int(src_fr)}fps{ext}"
    wav = audio["waveform"][0].numpy() if audio is not None else None
    return tl.relabel_video(path, out, clip_fr, src_fr, wav, audio["sample_rate"] if audio is not None else 0,
                            _AUDIO_CODEC.get(ext.lower(), ["-c:a", "aac"]))


def _proxy(path):
    """Small H.264 copy for the browser player, in temp. Colour tags carried over."""
    out_dir = os.path.join(folder_paths.get_temp_directory(), "seamstitch_preview")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_proxy.mp4")
    cmd = [tl._ffmpeg_exe(), "-v", "error", "-y", "-i", path, "-map", "0:v:0", "-map", "0:a:0?",
           "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-color_range", "tv", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _allowed(path):
    ap = os.path.abspath(path).lower()
    return any(ap.startswith(os.path.abspath(b).lower() + os.sep) for b in
               (folder_paths.get_output_directory(), folder_paths.get_input_directory(),
                folder_paths.get_temp_directory()))


def grab_frame(path, frame, fr):
    """Frame `frame` of the video as a PNG in output/seamstitch_frames, converted with the
    file's own colour metadata (the Loader/Recombine decode). Returns the PNG path."""
    from PIL import Image
    f = next(tl._iter_frames(path, fr, int(frame), int(frame) + 1), None)
    if f is None:
        raise ValueError(f"frame {frame} is past the end of {os.path.basename(path)}")
    out_dir = os.path.join(folder_paths.get_output_directory(), "seamstitch_frames")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(out_dir, f"{stem}_f{int(frame):05d}.png")
    Image.fromarray(f).save(out, compress_level=4)
    return out


@PromptServer.instance.routes.post("/seamstitch/result/grab")
async def _grab_route(request):
    try:
        body = await request.json()
        path = body.get("path", "")
        if not (os.path.isfile(path) and _allowed(path)):
            return web.json_response({"error": "not a video in the output/input/temp folders"}, status=400)
        import asyncio
        out = await asyncio.get_event_loop().run_in_executor(
            None, grab_frame, path, int(body.get("frame", 0)), int(body.get("frame_rate", 24)) or 24)
        return web.json_response({"path": out, "view": _view_params(out)})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


class SeamStitchResultPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video_path": ("STRING", {"forceInput": True, "tooltip":
                    "The video the splice was cut from - the Timeline's or Loader's source_video_path."}),
                "start_frame": ("INT", {"forceInput": True}),
                "end_frame": ("INT", {"forceInput": True}),
                "frame_rate": ("INT", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "seamstitch_%date:yyyyMMdd_hhmmss%", "tooltip":
                    "Only used when images is wired. %date:yyyyMMdd_hhmmss% is replaced, as on VHS Video Combine."}),
                "format": (_formats(), {"default": "video/h264-mp4", "tooltip":
                    "VHS's own formats and encode path - h264-mp4 matches VHS Video Combine pixel for pixel. "
                    "Any 4:2:0 format (h264/h265 yuv420p) costs ~1-2 levels of colour on re-encode, VHS "
                    "included. For a master with none: video/ffv1-mkv (16-bit RGB, lossless) or "
                    "video/ProRes (written as 4444). Those play through an H.264 proxy here."}),
                "crf": ("INT", {"default": 12, "min": 0, "max": 51, "step": 1, "tooltip":
                    "Quality for h264/h265/webm formats: lower = better, 0 = lossless. 12 matches the "
                    "Final Video setting this node replaces."}),
                "pix_fmt": (["yuv420p", "yuv420p10le"], {"default": "yuv420p", "tooltip":
                    "yuv420p plays everywhere; yuv420p10le keeps 10-bit gradients (h264/h265 only)."}),
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip":
                    "Embed the workflow in the video, as VHS does (the first-frame PNG always carries it)."}),
                "save_output": ("BOOLEAN", {"default": True, "tooltip":
                    "On: save to the output folder. Off: temp only (preview)."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Recombine's combined_images: this node saves the final video "
                                     "(set Recombine's skip_encode on)."}),
                "audio": ("AUDIO", {"tooltip": "Recombine's audio, muxed in when images is wired."}),
                "filenames": ("VHS_FILENAMES", {"tooltip": "Instead of images: a video Recombine (or VHS) "
                                                "already wrote - shown and rated, not re-encoded."}),
                # Appended last: ComfyUI restores widget values by position.
                "source_frame_rate": ("INT", {"forceInput": True, "tooltip":
                    "The Timeline's source_frame_rate. When it differs from frame_rate (the cut was "
                    "conformed to MiniMax H3's 24 fps clock), the restore below applies."}),
                "original_audio": ("AUDIO", {"tooltip":
                    "The Timeline's original_audio: the cut's unstretched sound, which the restored video "
                    "keeps outside the regenerated frames."}),
                "restore_source_frame_rate": ("BOOLEAN", {"default": True, "tooltip":
                    "Only used when source_frame_rate differs from frame_rate (Timeline conform_to_24fps on). "
                    "On: the same frames are saved at the source frame rate - the source cut's exact length - "
                    "with the original audio, and the regenerated span's own audio sped back up under it. "
                    "Given filenames, the picture is re-labelled without re-encoding. Off: saved as it "
                    "comes, at 24 fps with the slowed audio."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "VHS_FILENAMES")
    RETURN_NAMES = ("result_path", "Filenames")
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "SeamStitch"
    DESCRIPTION = ("Saves the spliced video (VHS's encode path and colours), plays it with the "
                   "regenerated span marked, rates both joins against the original, saves frames as "
                   "PNGs, and can put the result back on the Timeline for the next splice.")

    def preview(self, source_video_path, start_frame, end_frame, frame_rate,
                filename_prefix="seamstitch_%date:yyyyMMdd_hhmmss%", format="video/h264-mp4", crf=12,
                pix_fmt="yuv420p", save_metadata=True, save_output=True, images=None, audio=None,
                filenames=None, source_frame_rate=0, original_audio=None, restore_source_frame_rate=True,
                prompt=None, extra_pnginfo=None):
        fr = int(frame_rate) or 24
        src_fr = int(source_frame_rate or 0) or fr
        restore = bool(restore_source_frame_rate) and src_fr != fr
        out_fr = src_fr if restore else fr   # the rate the saved/shown result plays at
        if images is not None:
            if restore:
                s_frames = tl.probe(source_video_path, fr)["frames"]
                audio, notes = restore_audio(audio, original_audio, fr, src_fr, int(start_frame), int(end_frame),
                                             int(images.shape[0]), s_frames)
                for n in notes:
                    print(f"[SeamStitch] Result: restore audio: {n}")
                print(f"[SeamStitch] Result: restore - {images.shape[0]} frames saved at {src_fr} fps "
                      f"(made on the {fr} fps conform), {images.shape[0] / float(src_fr):.3f}s, original audio.")
            settings = format_settings(format, crf, pix_fmt, save_metadata)
            filenames = _encode(images, audio, out_fr, filename_prefix, format, save_output, settings,
                                prompt, extra_pnginfo)
            print(f"[SeamStitch] Result: saved {format} {settings} -> {filenames[1][-1]}")
        elif filenames is None:
            raise ValueError("Wire Recombine's combined_images (+ audio) into images to save the video "
                             "here, or its Filenames into filenames.")
        result = _result_path(filenames)
        if restore and images is None:
            result = restore_file(result, source_video_path, fr, src_fr, int(start_frame), int(end_frame),
                                  original_audio)
            print(f"[SeamStitch] Result: restore - re-labelled at {src_fr} fps (picture stream-copied) -> {result}")
            filenames = ((filenames[0], list(filenames[1]) + [result])
                         if isinstance(filenames, (list, tuple)) and len(filenames) > 1
                         and isinstance(filenames[1], (list, tuple)) else (True, [result]))
        a = analyse(result, source_video_path, out_fr, int(start_frame), int(end_frame), source_fr=fr)
        for row in a["joins"]:
            r = "n/a" if row["ratio"] is None else f"{row['ratio']:.2f}x"
            print(f"[SeamStitch] Result: join {row['name']} at frame {row['frame']}: {r} {row['verdict']}")
        b = a["before"]
        if b["ratio"] is not None:
            print(f"[SeamStitch] Result: original worst join in the replaced range: {b['ratio']:.2f}x "
                  f"{b['verdict']} at frame {b['frame']}")
        fmt_name = str(format).split("/")[-1]
        play = result
        if images is not None and fmt_name not in _PLAYABLE:
            try:
                play = _proxy(result)
            except Exception as e:
                print(f"[SeamStitch] Result: no browser proxy for {os.path.basename(result)}: {e}")
        master = _view_params(result)
        timeline_path = (f"{master['subfolder']}/{master['filename']}" if master['subfolder'] else master['filename']) \
            if master and master["type"] == "output" else result
        ui = dict(a, path=result, frame_rate=out_fr, start=int(start_frame), view=_view_params(play),
                  play_path=play, saved=bool(images is not None and save_output), timeline_path=timeline_path)
        return {"ui": {"seamstitch_result": [ui]}, "result": (result, filenames)}
