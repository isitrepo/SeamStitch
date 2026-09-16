# Based on ComfyUI-VideoHelperSuite's VHS_VideoCombine node
# (https://github.com/kosinkadink/ComfyUI-VideoHelperSuite), licensed GPL-3.0.
# Modified 2026 for SeamStitch (https://github.com/isitrepo/SeamStitch).

import os
import sys
import json
import re
import datetime
import subprocess

import numpy as np
import torch
import cv2
import av
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.utils import ProgressBar

# ---------------------------------------------------------------------------
# Locate the installed ComfyUI-VideoHelperSuite package and import its
# internals directly. VHS's own __init__.py loads its `videohelpersuite`
# subpackage via a *relative* import, so it isn't globally importable as
# `import videohelpersuite` from an unrelated custom_nodes package by default.
# We add VHS's actual folder to sys.path ourselves so the same subpackage
# resolves as a plain top-level import here too.
# ---------------------------------------------------------------------------
_CUSTOM_NODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _locate_vhs_package_dir():
    for name in ("comfyui-videohelpersuite", "ComfyUI-VideoHelperSuite"):
        cand = os.path.join(_CUSTOM_NODES_DIR, name)
        if os.path.isdir(os.path.join(cand, "videohelpersuite")):
            return cand
    # Fallback: scan every custom_nodes sibling for a videohelpersuite/nodes.py,
    # in case the package folder was renamed.
    try:
        for entry in os.listdir(_CUSTOM_NODES_DIR):
            cand = os.path.join(_CUSTOM_NODES_DIR, entry, "videohelpersuite")
            if os.path.isfile(os.path.join(cand, "nodes.py")):
                return os.path.join(_CUSTOM_NODES_DIR, entry)
    except OSError:
        pass
    return None


_vhs_dir = _locate_vhs_package_dir()
if _vhs_dir is None:
    raise ImportError(
        "SeamStitchRecombine requires ComfyUI-VideoHelperSuite to be installed "
        "alongside it in custom_nodes (looked for a 'videohelpersuite' package inside every "
        "custom_nodes subfolder and found none)."
    )
if _vhs_dir not in sys.path:
    sys.path.insert(0, _vhs_dir)

from videohelpersuite.utils import (
    ffmpeg_path, ENCODE_ARGS, merge_filter_args, get_audio, BIGMAX,
)
from videohelpersuite.nodes import (
    get_video_formats, apply_format_widgets, tensor_to_bytes, tensor_to_shorts, ffmpeg_process,
)

from .audio_splice import splice_audio, match_format, frame_sample


# ---------------------------------------------------------------------------
# Dedup: strip held/duplicate frames from the regenerated segment's own
# leading/trailing edge (e.g. a keyframe-anchored generation holding its first
# or last frame for an extra tick or two before/after the real motion starts).
# ---------------------------------------------------------------------------
def _frame_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return 1.0
    return torch.mean(torch.abs(a.float() - b.float())).item()


def _held_duplicate_bounds(images: torch.Tensor, threshold: float, max_strip: int):
    """Inclusive (first, last) indices that survive stripping leading/trailing frames
    that are near-identical holds of their neighbor, up to max_strip frames from each
    end. Always keeps at least one frame at each boundary and never collapses the
    clip to zero length. The bounds, not just the frames, are needed so the audio
    can be cut at the same place."""
    n = images.shape[0]
    if n <= 1 or max_strip <= 0 or threshold <= 0:
        return 0, n - 1

    start = 0
    while start < min(max_strip, n - 1) and _frame_diff(images[start], images[start + 1]) < threshold:
        start += 1

    end = n - 1
    while (n - 1 - end) < max_strip and end > start + 1 and _frame_diff(images[end], images[end - 1]) < threshold:
        end -= 1

    if start >= end:
        return 0, n - 1
    return start, end


def _strip_held_duplicates(images: torch.Tensor, threshold: float, max_strip: int) -> torch.Tensor:
    start, end = _held_duplicate_bounds(images, threshold, max_strip)
    return images[start:end + 1]


# ---------------------------------------------------------------------------
# Decode a [start_frame_idx, end_frame_idx) range from the original video at a
# forced frame_rate, applying the same crop_x/y/w/h convention as LoadVideoUI,
# then resizing (stretch) to (target_w, target_h) so it lines up with the
# regenerated segment's resolution. end_frame_idx=None decodes to EOF.
#
# Returns a uint8 tensor (not float32) — for a long "before"/"after" span this
# is 4x smaller, which is the difference between fitting in RAM and an
# `Unable to allocate` MemoryError once frame count x resolution gets large.
# Callers are responsible for the final /255.0 float32 cast, and should do it
# once on the fully-concatenated result rather than per-chunk, so the 4x-larger
# array only ever exists once, right before the encoder needs it.
# ---------------------------------------------------------------------------
def _decode_range(video_path, frame_rate, start_frame_idx, end_frame_idx,
                   target_w, target_h, crop_x=0.0, crop_y=0.0, crop_w=1.0, crop_h=1.0):
    if end_frame_idx is not None and end_frame_idx <= start_frame_idx:
        return torch.zeros((0, target_h, target_w, 3), dtype=torch.uint8)

    container = av.open(video_path)
    video_stream = container.streams.video[0] if len(container.streams.video) > 0 else None
    if video_stream is None:
        container.close()
        return torch.zeros((0, target_h, target_w, 3), dtype=torch.uint8)

    orig_w = video_stream.codec_context.width
    orig_h = video_stream.codec_context.height

    try:
        from av.video.reformatter import Colorspace, ColorRange
        fallback_cs = Colorspace.ITU709 if max(orig_w, orig_h) >= 720 else Colorspace.ITU601
        fallback_cr = ColorRange.MPEG
        dst_range = ColorRange.JPEG
    except ImportError:
        fallback_cs = "itu709" if max(orig_w, orig_h) >= 720 else "itu601"
        fallback_cr = "mpeg"
        dst_range = "jpeg"

    src_colorspace = fallback_cs
    src_color_range = fallback_cr
    if video_stream.codec_context:
        cc = video_stream.codec_context
        c_space = getattr(cc, 'colorspace', getattr(cc, 'color_space', None))
        if c_space and hasattr(c_space, 'name') and c_space.name != "UNSPECIFIED":
            src_colorspace = c_space
        elif c_space and isinstance(c_space, str) and "unspecified" not in c_space.lower():
            src_colorspace = c_space
        c_range = getattr(cc, 'color_range', None)
        if c_range and hasattr(c_range, 'name') and c_range.name != "UNSPECIFIED":
            src_color_range = c_range
        elif c_range and isinstance(c_range, str) and "unspecified" not in c_range.lower():
            src_color_range = c_range

    manual_crop_left = max(0, min(int(orig_w * crop_x), orig_w - 1))
    manual_crop_top = max(0, min(int(orig_h * crop_y), orig_h - 1))
    manual_crop_right = max(0, min(orig_w - int(orig_w * (crop_x + crop_w)), orig_w - manual_crop_left - 1))
    manual_crop_bottom = max(0, min(orig_h - int(orig_h * (crop_y + crop_h)), orig_h - manual_crop_top - 1))
    has_crop = manual_crop_left > 0 or manual_crop_top > 0 or manual_crop_right > 0 or manual_crop_bottom > 0

    fr = float(frame_rate) if frame_rate > 0 else 24.0
    frame_interval = 1.0 / fr

    # Frame indices count from the stream's own first frame, which is not
    # necessarily at t=0. SeamStitchCombine's concat demuxer leaves a small
    # positive start offset on its output (+31 ms measured on real 48 fps
    # footage), and mapping index -> time as a bare idx/fr silently ignored it:
    # once the offset exceeded one frame interval, the "after" range began one
    # frame early, so the last frame of the span being replaced was re-emitted
    # verbatim immediately after the regenerated segment - one extra frame in
    # the output and a held frame at the trailing join. Anchor on the real
    # start time instead. The "before" range was unaffected (it starts at
    # index 0 and its end is capped by a frame count, not a time).
    if video_stream.start_time is not None and video_stream.time_base:
        base_time = float(video_stream.start_time * video_stream.time_base)
    else:
        base_time = 0.0
    start_time = base_time + start_frame_idx / fr
    end_time = base_time + end_frame_idx / fr if end_frame_idx is not None else None

    video_stream.thread_type = "AUTO"
    if video_stream.time_base:
        seek_pts = int(start_time / float(video_stream.time_base))
    else:
        seek_pts = int(start_time * av.time_base)
    container.seek(seek_pts, stream=video_stream, backward=True)

    frames_out = []
    frame_idx = start_frame_idx
    expected_target_time = start_time

    for frame in container.decode(video_stream):
        frame_time = frame.time
        if frame_time is None:
            frame_time = float(frame.pts * float(video_stream.time_base)) if frame.pts and video_stream.time_base else 0.0

        if frame_time < start_time - frame_interval:
            continue
        if end_time is not None and frame_time > end_time + frame_interval:
            break

        try:
            frame = frame.reformat(format="rgb24", src_colorspace=src_colorspace,
                                    src_color_range=src_color_range, dst_color_range=dst_range)
            frame_rgb = frame.to_ndarray(format='rgb24')
        except Exception:
            frame_rgb = frame.to_ndarray(format='rgb24')

        if has_crop:
            frame_rgb = frame_rgb[manual_crop_top:orig_h - manual_crop_bottom,
                                   manual_crop_left:orig_w - manual_crop_right, :]

        # Tolerance of a thousandth of a frame (~21 us at 48 fps): the target is
        # derived from frame_idx rather than accumulated, but pts -> float still
        # lands a hair either side of an exactly-equal target, and losing that
        # comparison drops the range's final frame.
        while expected_target_time <= frame_time + frame_interval * 1e-3:
            if end_frame_idx is not None and frame_idx >= end_frame_idx:
                break
            if (frame_rgb.shape[1], frame_rgb.shape[0]) != (target_w, target_h):
                resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                resized = frame_rgb
            frames_out.append(resized)
            frame_idx += 1
            # Derived from the index, never accumulated: adding frame_interval
            # 239 times drifted far enough to lose the final frame of a range.
            expected_target_time = base_time + frame_idx / fr

        if end_frame_idx is not None and frame_idx >= end_frame_idx:
            break

    container.close()
    if not frames_out:
        return torch.zeros((0, target_h, target_w, 3), dtype=torch.uint8)
    arr = np.stack(frames_out)  # uint8, HxWx3 per frame — no float32 cast here
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Encode: adapted from VHS_VideoCombine's own ffmpeg pipeline (VideoCombine.
# combine_video in videohelpersuite/nodes.py), stripped of VAE/meta-batch/
# pingpong/animated-image-format handling — this fork only writes standard
# ffmpeg video containers (mp4 etc.), not gif/webp. Pipe the result through
# the stock VHS_VideoCombine node afterward if an animated-image output or
# per-format encode widgets (crf/preset/etc.) are needed.
# ---------------------------------------------------------------------------
def _encode_video(images, frame_rate, filename_prefix, format, save_output,
                   audio, prompt, extra_pnginfo):
    if images.shape[0] == 0:
        return {"ui": {"gifs": []}, "result": ((save_output, []),)}

    num_frames = images.shape[0]
    pbar = ProgressBar(num_frames)
    first_image = images[0]

    output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
    (full_output_folder, filename, _, subfolder, _) = folder_paths.get_save_image_path(filename_prefix, output_dir)
    output_files = []

    metadata = PngInfo()
    video_metadata = {}
    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt))
        video_metadata["prompt"] = json.dumps(prompt)
    if extra_pnginfo is not None:
        for x in extra_pnginfo:
            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            video_metadata[x] = extra_pnginfo[x]
    metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

    max_counter = 0
    matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
    for existing_file in os.listdir(full_output_folder):
        match = matcher.fullmatch(existing_file)
        if match:
            max_counter = max(max_counter, int(match.group(1)))
    counter = max_counter + 1

    first_image_file = f"{filename}_{counter:05}.png"
    file_path = os.path.join(full_output_folder, first_image_file)
    Image.fromarray(tensor_to_bytes(first_image)).save(file_path, pnginfo=metadata, compress_level=4)
    output_files.append(file_path)

    if ffmpeg_path is None:
        raise ProcessLookupError(
            "ffmpeg is required for video outputs and could not be found.\n"
            "Install imageio-ffmpeg with pip, place an ffmpeg executable next to ComfyUI, "
            "or install ffmpeg and add it to the system path."
        )

    format_type, format_ext = format.split("/")
    video_format = apply_format_widgets(format_ext, {})
    has_alpha = first_image.shape[-1] == 4
    video_format["has_alpha"] = has_alpha

    dim_alignment = video_format.get("dim_alignment", 2)
    images_iter = iter(images)
    if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
        to_pad = (-first_image.shape[1] % dim_alignment, -first_image.shape[0] % dim_alignment)
        padding = (to_pad[0] // 2, to_pad[0] - to_pad[0] // 2,
                   to_pad[1] // 2, to_pad[1] - to_pad[1] // 2)
        padfunc = torch.nn.ReplicationPad2d(padding)

        def pad(image):
            image = image.permute((2, 0, 1))
            padded = padfunc(image.to(dtype=torch.float32))
            return padded.permute((1, 2, 0))

        images_iter = map(pad, images_iter)
        dimensions = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                      -first_image.shape[0] % dim_alignment + first_image.shape[0])
    else:
        dimensions = (first_image.shape[1], first_image.shape[0])

    if video_format.get('input_color_depth', '8bit') == '16bit':
        images_iter = map(tensor_to_shorts, images_iter)
        i_pix_fmt = 'rgba64' if has_alpha else 'rgb48'
    else:
        images_iter = map(tensor_to_bytes, images_iter)
        i_pix_fmt = 'rgba' if has_alpha else 'rgb24'

    file = f"{filename}_{counter:05}.{video_format['extension']}"
    file_path = os.path.join(full_output_folder, file)
    bitrate_arg = []
    bitrate = video_format.get('bitrate')
    if bitrate is not None:
        bitrate_arg = ["-b:v", str(bitrate) + ("M" if video_format.get('megabit') == 'True' else "K")]

    args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
            "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
            "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),
            "-s", f"{dimensions[0]}x{dimensions[1]}", "-r", str(frame_rate), "-i", "-"]

    images_bytes = map(lambda x: x.tobytes(), images_iter)
    env = os.environ.copy()
    if "environment" in video_format:
        env.update(video_format["environment"])

    if "inputs_main_pass" in video_format:
        in_args_len = args.index("-i") + 2
        args = args[:in_args_len] + video_format['inputs_main_pass'] + args[in_args_len:]

    args += video_format['main_pass'] + bitrate_arg
    merge_filter_args(args)
    output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
    output_process.send(None)
    total_frames_output = 0
    for image in images_bytes:
        pbar.update(1)
        output_process.send(image)
    try:
        total_frames_output = output_process.send(None)
        output_process.send(None)
    except StopIteration:
        pass
    output_files.append(file_path)

    a_waveform = None
    if audio is not None:
        try:
            a_waveform = audio['waveform']
        except Exception:
            pass
    if a_waveform is not None:
        output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
        output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
        if "audio_pass" not in video_format:
            video_format["audio_pass"] = ["-c:a", "libopus"]
        channels = audio['waveform'].size(1)
        min_audio_dur = (total_frames_output or num_frames) / frame_rate + 1
        apad = ["-af", "apad=whole_dur=" + str(min_audio_dur)]
        mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                    "-ar", str(audio['sample_rate']), "-ac", str(channels),
                    "-f", "f32le", "-i", "-", "-c:v", "copy"] \
            + video_format["audio_pass"] + apad + ["-shortest", output_file_with_audio_path]
        audio_data = audio['waveform'].squeeze(0).transpose(0, 1).numpy().tobytes()
        merge_filter_args(mux_args, '-af')
        try:
            res = subprocess.run(mux_args, input=audio_data, env=env, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception("An error occurred in the ffmpeg audio-mux subprocess:\n"
                             + e.stderr.decode(*ENCODE_ARGS))
        if res.stderr:
            print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
        output_files.append(output_file_with_audio_path)
        file = output_file_with_audio

    preview = {
        "filename": file,
        "subfolder": subfolder,
        "type": "output" if save_output else "temp",
        "format": format,
        "frame_rate": frame_rate,
        "workflow": first_image_file,
        "fullpath": output_files[-1],
    }
    return {"ui": {"gifs": [preview]}, "result": ((save_output, output_files),)}


class SeamStitchRecombine:
    """Fork of VHS_VideoCombine that splices a regenerated replacement clip back into
    the original video at the exact location it was cut from, then encodes the result —
    instead of encoding `images` as a standalone clip."""

    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats, format_widgets = get_video_formats()
        return {
            "required": {
                "regenerated_images": ("IMAGE",),
                "original_video_path": ("STRING", {"default": "", "multiline": False}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "step": 1}),
                "dedup_threshold": ("FLOAT", {"default": 0.008, "min": 0.0, "max": 1.0, "step": 0.001,
                                               "tooltip": "Mean abs pixel difference (0-1) below which two adjacent frames at the regenerated segment's boundary are treated as a held duplicate and dropped. 0 disables detection."}),
                "max_dedup_frames": ("INT", {"default": 6, "min": 0, "max": 60, "step": 1,
                                              "tooltip": "Cap on how many leading/trailing frames can be stripped from the regenerated segment as held duplicates."}),
                "filename_prefix": ("STRING", {"default": "seamstitch_recombined"}),
                "format": (ffmpeg_formats, {'default': 'video/h264-mp4', 'formats': format_widgets}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "original_audio_override": ("AUDIO", {"tooltip": "Replaces the source file's audio track. Must be on the source file's own timeline (SeamStitchLoader's full_clip_audio is); it is cut to match the picture exactly as the file's own track would be."}),
                "bridge_audio": ("AUDIO", {"tooltip": "Audio generated alongside the regenerated frames (e.g. LTXVAudioVAEDecode on the bridge's audio latent), starting at its first frame. Used when audio_mode is 'bridge'."}),
                "crop_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                      "tooltip": "Only needed if the same crop was applied in LoadVideoUI when the regenerated segment was produced."}),
                "crop_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "crop_w": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "crop_h": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "audio_mode": (["original", "bridge"], {"default": "original",
                               "tooltip": "What plays under the regenerated frames. 'original': the source's own audio for exactly the frames that survived dedup, so dropped frames take their sound with them. 'bridge': the bridge_audio input. Either way the audio is cut at the same frames as the picture, so both stay in sync after the splice."}),
                "audio_crossfade_ms": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 500.0, "step": 1.0,
                                       "tooltip": "Equal-power crossfade at each audio join that is not already continuous, so a cut mid-waveform cannot click. Length-preserving: it never moves either side of the join. 0 = hard cut."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES", "IMAGE", "AUDIO")
    RETURN_NAMES = ("Filenames", "combined_images", "audio")
    OUTPUT_NODE = True
    CATEGORY = "SeamStitch"
    FUNCTION = "recombine"

    def recombine(self, regenerated_images, original_video_path, start_frame, end_frame,
                  frame_rate, dedup_threshold, max_dedup_frames, filename_prefix, format,
                  save_output=True, original_audio_override=None, bridge_audio=None,
                  crop_x=0.0, crop_y=0.0, crop_w=1.0, crop_h=1.0,
                  audio_mode="original", audio_crossfade_ms=20.0,
                  prompt=None, extra_pnginfo=None, **kwargs):

        if not original_video_path or not os.path.exists(original_video_path):
            raise FileNotFoundError(f"original_video_path not found: {original_video_path}")
        if end_frame < start_frame:
            raise ValueError(f"end_frame ({end_frame}) must be >= start_frame ({start_frame})")
        if audio_mode == "bridge" and bridge_audio is None:
            raise ValueError("audio_mode is 'bridge' but nothing is connected to bridge_audio.")

        # Regenerated segment is the resolution ground truth for the splice.
        regenerated = regenerated_images
        if regenerated.shape[-1] == 4:
            regenerated = regenerated[..., :3]
        target_h = regenerated.shape[1]
        target_w = regenerated.shape[2]

        # 1. Drop held/duplicate frames at the regenerated segment's own boundaries.
        first_kept, last_kept = _held_duplicate_bounds(regenerated, dedup_threshold, max_dedup_frames)
        deduped = regenerated[first_kept:last_kept + 1]
        dropped = regenerated.shape[0] - deduped.shape[0]
        if dropped > 0:
            print(f"[SeamStitch] Dropped {dropped} held/duplicate frame(s) from the regenerated "
                  f"segment's boundaries ({first_kept} leading, "
                  f"{regenerated.shape[0] - 1 - last_kept} trailing).")

        # 2. Decode the original video's "before" and "after" chunks on the same
        #    timeline the segment was cut from, resized to match its resolution.
        before = _decode_range(original_video_path, frame_rate, 0, start_frame,
                                target_w, target_h, crop_x, crop_y, crop_w, crop_h)
        after = _decode_range(original_video_path, frame_rate, end_frame + 1, None,
                               target_w, target_h, crop_x, crop_y, crop_w, crop_h)

        # Concatenate in uint8 (before/after already are; deduped is the model's
        # float32 [0,1] output, so downcast it to match). This keeps the
        # concatenated buffer 4x smaller than the final float32 IMAGE tensor —
        # for a long splice at high resolution that's the difference between a
        # few GB and tens of GB of transient peak memory.
        deduped_u8 = deduped.clamp(0, 1).mul(255).round().to(torch.uint8)
        chunks = [t for t in (before, deduped_u8, after) if t is not None and t.shape[0] > 0]
        if not chunks:
            raise RuntimeError("Nothing to combine — before/regenerated/after all produced zero frames.")
        combined_u8 = torch.cat(chunks, dim=0)
        del before, deduped_u8, after, chunks
        # Single float32 cast of the whole spliced video, done once and as late
        # as possible — this is the only point a full-size float32 copy exists.
        combined = combined_u8.to(torch.float32).div_(255.0)
        del combined_u8

        expected_gap = end_frame - start_frame + 1
        if abs(deduped.shape[0] - expected_gap) > max_dedup_frames:
            print(f"[SeamStitch] Warning: regenerated segment has {deduped.shape[0]} "
                  f"frames after dedup, but the original gap was {expected_gap} frames at "
                  f"frame_rate={frame_rate}. The combined video's total duration will differ "
                  f"from the original by that much at this splice point.")

        # 3. Audio, cut at the same frames as the picture. Laying the original track
        #    down untouched from t=0 shifted everything after the splice by
        #    (expected_gap - kept) / frame_rate - 20.8 ms per dropped frame at 48 fps.
        audio = self._splice_audio(original_video_path, original_audio_override, bridge_audio,
                                   audio_mode, audio_crossfade_ms, frame_rate,
                                   start_frame, end_frame, first_kept, deduped.shape[0])

        result = _encode_video(combined, frame_rate, filename_prefix, format, save_output,
                                audio, prompt, extra_pnginfo)
        result["result"] = result["result"] + (combined, audio)
        return result

    @staticmethod
    def _splice_audio(video_path, override, bridge_audio, audio_mode, crossfade_ms, frame_rate,
                      start_frame, end_frame, first_kept, kept):
        with av.open(video_path) as container:
            v = container.streams.video[0] if container.streams.video else None
            a = container.streams.audio[0] if container.streams.audio else None
            v_start = float(v.start_time * v.time_base) if v is not None and v.start_time is not None else 0.0
            a_start = float(a.start_time * a.time_base) if a is not None and a.start_time is not None else 0.0
            has_audio = a is not None
            duration_s = float(container.duration) / av.time_base if container.duration else 0.0

        if override is not None:
            source = override
        elif has_audio:
            source = get_audio(video_path, start_time=0, duration=0)
        else:
            source = None
        use_bridge = audio_mode == "bridge"
        if source is None and not use_bridge:
            return None

        if source is not None:
            sample_rate = int(source["sample_rate"])
            wave = source["waveform"][0].detach().to("cpu", torch.float32)
            # Source frame f plays at v_start + f/fps, against the audio sample at
            # (v_start - a_start + f/fps). SeamStitchCombine's output delays its
            # video 31 ms with an empty edit to cover AAC priming it does not skip,
            # so this offset is real, not rounding.
            av_offset = v_start - a_start
        else:
            sample_rate = int(bridge_audio["sample_rate"])
            channels = bridge_audio["waveform"].shape[1]
            # No source track to cut: silence either side of the bridge's own audio.
            wave = torch.zeros((channels, int(round(duration_s * sample_rate))))
            av_offset = 0.0

        bridge = None
        if use_bridge:
            bridge = match_format(bridge_audio["waveform"][0], bridge_audio["sample_rate"],
                                  wave.shape[0], sample_rate)

        out, notes = splice_audio(wave, sample_rate, frame_rate, start_frame, end_frame,
                                  first_kept, kept, av_offset_s=av_offset, bridge=bridge,
                                  crossfade_ms=crossfade_ms)
        for note in notes:
            print(f"[SeamStitch] audio: {note}")
        return {"waveform": out.unsqueeze(0), "sample_rate": sample_rate}
