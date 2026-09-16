# Based on WhatDreamsCost-ComfyUI's Load Video UI node
# (https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI), licensed GPL-3.0.
# Modified 2026 for SeamStitch (https://github.com/isitrepo/SeamStitch).

import os
import asyncio
import torch
import numpy as np
import folder_paths
import av
from server import PromptServer
from aiohttp import web
import comfy.utils
from PIL import Image

_VIDEO_EXTENSIONS = ('.mp4', '.webm', '.mkv', '.avi', '.mov', '.m4v', '.flv', '.wmv')


# Custom API route to serve video files from anywhere on the user's system for the frontend preview.
# Namespaced under /seamstitch/ so this doesn't collide with the upstream loader project's own
# view route if both are installed side by side.
#
# Deliberately not restricted to the ComfyUI input directory - the "choose file to upload"
# flow has a fast path (js/loader.js) that points this straight at a desktop absolute path,
# skipping the upload entirely. Restricted instead to real, existing files with a known video
# extension, with any ".." path segment rejected outright, to shrink an arbitrary-file-read
# down to "read video files that already exist on disk".
@PromptServer.instance.routes.get("/seamstitch/loader/view")
async def custom_view(request):
    file_path = request.query.get("filename", "")
    if not file_path:
        return web.Response(status=404, text="File not found")
    normalized = file_path.replace("\\", "/")
    if any(part == ".." for part in normalized.split("/")):
        return web.Response(status=404, text="File not found")
    if not file_path.lower().endswith(_VIDEO_EXTENSIONS):
        return web.Response(status=404, text="File not found")
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return web.FileResponse(file_path)
    return web.Response(status=404, text="File not found")


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


def _read_and_write_file_chunk(file, file_path, mode):
    chunk_bytes = file.file.read()
    with open(file_path, mode) as f:
        f.write(chunk_bytes)


def _save_frame_png(image_tensor, filename_prefix):
    """Save a single-frame IMAGE tensor ([1,H,W,C] or [H,W,C]) as a PNG in the main
    ComfyUI output directory, using the same auto-incrementing counter convention
    as the built-in Save Image node."""
    frame = image_tensor[0] if image_tensor.dim() == 4 else image_tensor
    height, width = frame.shape[0], frame.shape[1]
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir, width, height
    )
    file = f"{filename}_{counter:05}.png"
    file_path = os.path.join(full_output_folder, file)
    arr = (frame.cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(file_path)
    return file_path


# Custom API route for Chunked Uploads to bypass the 413 Payload Too Large error
@PromptServer.instance.routes.post("/seamstitch/loader/upload_chunk")
async def upload_chunk(request):
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

    # Append to file if it's not the first chunk, otherwise write new
    mode = "ab" if chunk_index > 0 else "wb"

    # Offload the blocking read/write disk I/O to a thread executor
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _read_and_write_file_chunk, file, file_path, mode)

    if chunk_index == total_chunks - 1:
        return web.json_response({"name": filename})
    return web.json_response({"status": "ok"})


# File-exists dedup check, used by the "choose file to upload" flow to avoid re-uploading
# a file that's already sitting in the input directory.
@PromptServer.instance.routes.get("/seamstitch/loader/check_file")
async def check_file(request):
    filename = request.query.get("filename", "")
    file_size = request.query.get("size", "")
    if not filename:
        return web.json_response({"exists": False})

    upload_dir = folder_paths.get_input_directory()
    temp_dir = os.path.join(upload_dir, "whatdreamscost")

    possible_paths = [
        os.path.join(temp_dir, filename),
        os.path.join(upload_dir, filename),
    ]

    found_path = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            if file_size:
                try:
                    if os.path.getsize(p) == int(file_size):
                        found_path = p
                        break
                except ValueError:
                    found_path = p
                    break
            else:
                found_path = p
                break

    if found_path:
        rel_name = os.path.relpath(found_path, upload_dir).replace('\\', '/')
        return web.json_response({"exists": True, "name": rel_name})

    base_name = os.path.basename(filename)
    suffix = f"_{base_name}"
    try:
        for search_dir in [temp_dir, upload_dir]:
            if os.path.exists(search_dir):
                for f_name in os.listdir(search_dir):
                    if f_name.endswith(suffix) or f_name == base_name:
                        pot_path = os.path.join(search_dir, f_name)
                        if os.path.isfile(pot_path):
                            if file_size:
                                try:
                                    if os.path.getsize(pot_path) == int(file_size):
                                        rel_name = os.path.relpath(pot_path, upload_dir).replace('\\', '/')
                                        return web.json_response({"exists": True, "name": rel_name})
                                except ValueError:
                                    pass
    except Exception:
        pass

    return web.json_response({"exists": False})


def _resolve_video_path(video):
    """Same lookup as load_video's own (exact path, then annotated, then the
    input directory), but returns None on a miss instead of raising - used by
    the tensor-input path, where a video file is a nice-to-have (it's what
    lets source_video_path still work for a downstream splice node) rather
    than a hard requirement."""
    if not video or video == "none":
        return None
    if os.path.exists(video):
        return video
    annotated = folder_paths.get_annotated_filepath(video)
    if os.path.exists(annotated):
        return annotated
    in_input = os.path.join(folder_paths.get_input_directory(), video)
    if os.path.exists(in_input):
        return in_input
    return None


def _list_input_videos():
    input_dir = folder_paths.get_input_directory()
    files = []
    if os.path.exists(input_dir):
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        wdc_dir = os.path.join(input_dir, "whatdreamscost")
        if os.path.exists(wdc_dir):
            wdc_files = [f"whatdreamscost/{f}" for f in os.listdir(wdc_dir) if os.path.isfile(os.path.join(wdc_dir, f))]
            all_files.extend(wdc_files)
        try:
            files = folder_paths.filter_files_content_types(all_files, ["video"])
        except:
            video_extensions = ('.mp4', '.webm', '.mkv', '.avi', '.mov', '.m4v', '.flv', '.wmv')
            files = [f for f in all_files if f.lower().endswith(video_extensions)]

        # Newest first, not alphabetical - an input directory that's years of
        # accumulated files (as this one is) makes a freshly-written file (e.g.
        # a combine node's own output) nearly impossible to spot by name alone
        # in an alphabetically sorted list of hundreds of entries.
        def _mtime(f):
            try:
                return os.path.getmtime(os.path.join(input_dir, f))
            except OSError:
                return 0
        files = sorted(files, key=_mtime, reverse=True)
    return files or ["none"]


# Lets the "Load Video" button re-scan the input directory on demand - the dropdown's
# own option list is only ever set once, at node-creation time, so a file written to
# disk afterwards (e.g. by a combine node upstream) would otherwise need the node
# deleted and re-added before it could be picked.
@PromptServer.instance.routes.get("/seamstitch/loader/list_files")
async def list_files(request):
    return web.json_response({"files": _list_input_videos()})


class SeamStitchLoader:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_input_videos()
        return {
            "required": {
                "video": (files,),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
                "end_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1}),
                "duration_frames": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1}),
                "resize_method": (["maintain aspect ratio", "stretch to fit", "pad", "crop"], {"default": "maintain aspect ratio"}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 8, "tooltip": "Custom width. 0 means original width."}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 8, "tooltip": "Custom height. 0 means original height."}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1, "tooltip": "Force the video to a specific frame rate for extraction."}),
                "display_mode": (["seconds", "frames"], {"default": "seconds"}),
                "crop_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "crop_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "crop_w": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "crop_h": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "save_first_frame": ("BOOLEAN", {"default": False, "tooltip": "Save the extracted first frame as a PNG to the main ComfyUI output directory."}),
                "save_last_frame": ("BOOLEAN", {"default": False, "tooltip": "Save the extracted last frame as a PNG to the main ComfyUI output directory."}),
            },
            "optional": {
                "input_video": ("IMAGE", {"tooltip": "Feed frames in directly from an upstream node instead of picking a file below - the video dropdown is ignored while this is connected."}),
                "input_audio": ("AUDIO", {"tooltip": "Audio to go with input_video. Optional even when input_video is connected - a silent track is used if omitted."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING", "IMAGE", "IMAGE", "STRING", "INT", "INT", "INT", "INT", "INT", "AUDIO")
    RETURN_NAMES = ("images", "audio", "duration", "frame_count", "filename", "first_frame", "last_frame", "source_video_path", "start_frame", "end_frame", "frame_rate", "width", "height", "full_clip_audio")
    FUNCTION = "load_video"
    CATEGORY = "SeamStitch"

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        return True

    def load_video(self, video, frame_rate, display_mode, start_time, end_time, duration, start_frame, end_frame, duration_frames, custom_width=0, custom_height=0, resize_method="maintain aspect ratio", crop_x=0.0, crop_y=0.0, crop_w=1.0, crop_h=1.0, save_first_frame=False, save_last_frame=False, input_video=None, input_audio=None, **kwargs):
        if input_video is not None:
            return self._load_from_tensor(
                input_video, input_audio, video, frame_rate, display_mode,
                start_time, end_time, duration, start_frame, end_frame, duration_frames,
                custom_width, custom_height, resize_method, crop_x, crop_y, crop_w, crop_h,
                save_first_frame, save_last_frame,
            )
        if not video:
            # Return blank defaults if no video is loaded
            empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            empty_audio = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100}
            return (empty_image, empty_audio, 0.0, 0, "", empty_image.clone(), empty_image.clone(), "", 0, 0, int(frame_rate), 512, 512, empty_audio.copy())

        # 1. Resolve path using ComfyUI standard paths or Absolute Path
        video_path = video  # Try exact/absolute path first
        if not os.path.exists(video_path):
            video_path_annotated = folder_paths.get_annotated_filepath(video)
            if os.path.exists(video_path_annotated):
                video_path = video_path_annotated
            else:
                video_path_input = os.path.join(folder_paths.get_input_directory(), video)
                if os.path.exists(video_path_input):
                    video_path = video_path_input
                else:
                    raise FileNotFoundError(f"Video file not found: {video}")

        # Open container to read streams and metadata
        container = av.open(video_path)

        # Determine video stream and duration
        video_stream = container.streams.video[0] if len(container.streams.video) > 0 else None
        video_duration = 0
        if video_stream and video_stream.duration and video_stream.time_base:
            video_duration = float(video_stream.duration * video_stream.time_base)

        orig_w = video_stream.codec_context.width if video_stream else 512
        orig_h = video_stream.codec_context.height if video_stream else 512

        # Determine correct colorspace and color range for PyAV conversion to prevent color shift
        try:
            from av.video.reformatter import Colorspace, ColorRange
            # Improve fallback heuristic to check both dimensions (e.g. 720x1280 vertical video is HD)
            fallback_cs = Colorspace.ITU709 if max(orig_w, orig_h) >= 720 else Colorspace.ITU601
            fallback_cr = ColorRange.MPEG
            dst_range = ColorRange.JPEG # RGB should always be full range
        except ImportError:
            fallback_cs = "itu709" if max(orig_w, orig_h) >= 720 else "itu601"
            fallback_cr = "mpeg"
            dst_range = "jpeg"

        src_colorspace = fallback_cs
        src_color_range = fallback_cr

        if video_stream and video_stream.codec_context:
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

        target_w = custom_width if custom_width > 0 else orig_w
        target_h = custom_height if custom_height > 0 else orig_h

        target_w = target_w - (target_w % 2)
        target_h = target_h - (target_h % 2)

        # Calculate manual crop from interactive UI first
        manual_crop_left = int(orig_w * crop_x)
        manual_crop_top = int(orig_h * crop_y)
        manual_crop_right = orig_w - int(orig_w * (crop_x + crop_w))
        manual_crop_bottom = orig_h - int(orig_h * (crop_y + crop_h))

        # Ensure we don't crop more than the image
        manual_crop_left = max(0, min(manual_crop_left, orig_w - 1))
        manual_crop_top = max(0, min(manual_crop_top, orig_h - 1))
        manual_crop_right = max(0, min(manual_crop_right, orig_w - manual_crop_left - 1))
        manual_crop_bottom = max(0, min(manual_crop_bottom, orig_h - manual_crop_top - 1))

        # After manual crop, the new original dimensions are:
        cropped_orig_w = orig_w - manual_crop_left - manual_crop_right
        cropped_orig_h = orig_h - manual_crop_top - manual_crop_bottom

        # If no custom width/height is provided, use the cropped original dimensions
        if custom_width == 0:
            target_w = cropped_orig_w
            target_w = target_w - (target_w % 2)
        if custom_height == 0:
            target_h = cropped_orig_h
            target_h = target_h - (target_h % 2)

        scale_w, scale_h = target_w, target_h
        pad_left = pad_right = pad_top = pad_bottom = 0
        crop_left = crop_right = crop_top = crop_bottom = 0

        if custom_width > 0 or custom_height > 0:
            if resize_method == "maintain aspect ratio" or resize_method == "pad":
                ratio = min(target_w / cropped_orig_w, target_h / cropped_orig_h)
                scale_w = int(cropped_orig_w * ratio)
                scale_h = int(cropped_orig_h * ratio)
                scale_w = scale_w - (scale_w % 2)
                scale_h = scale_h - (scale_h % 2)

                if resize_method == "pad":
                    pad_x = target_w - scale_w
                    pad_y = target_h - scale_h
                    pad_left = pad_x // 2
                    pad_right = pad_x - pad_left
                    pad_top = pad_y // 2
                    pad_bottom = pad_y - pad_top
                else:
                    target_w, target_h = scale_w, scale_h

            elif resize_method == "crop":
                ratio = max(target_w / cropped_orig_w, target_h / cropped_orig_h)
                scale_w = int(cropped_orig_w * ratio)
                scale_h = int(cropped_orig_h * ratio)
                scale_w = scale_w - (scale_w % 2)
                scale_h = scale_h - (scale_h % 2)

                crop_x = scale_w - target_w
                crop_y = scale_h - target_h
                crop_left = crop_x // 2
                crop_right = crop_x - crop_left
                crop_top = crop_y // 2
                crop_bottom = crop_y - crop_top

            elif resize_method == "stretch to fit":
                scale_w, scale_h = target_w, target_h

        # Determine exact bounds based on frontend mode
        if display_mode == "frames":
            fr = float(frame_rate) if frame_rate > 0 else 24.0
            actual_start_time = float(start_frame) / fr
            actual_end_time = float(end_frame) / fr if (end_frame > 0 and end_frame > start_frame) else video_duration
        else:
            actual_start_time = start_time
            actual_end_time = end_time if (end_time > 0 and end_time > start_time) else video_duration

        if actual_end_time <= 0:
            actual_end_time = float('inf') # Fallback if duration is unknown

        # 2. Extract Video Frames (PyAV)
        frames = []
        image_tensor = None
        frames_loaded = 0

        if video_stream:
            video_stream.thread_type = "AUTO" # Enable multithreaded decoding

            # Efficiently seek backwards to the nearest keyframe
            if video_stream.time_base:
                seek_pts = int(actual_start_time / float(video_stream.time_base))
            else:
                seek_pts = int(actual_start_time * av.time_base)

            container.seek(seek_pts, stream=video_stream, backward=True)

            # Custom sampling to force specific framerate
            frame_interval = 1.0 / float(frame_rate) if frame_rate > 0 else 1.0/24.0
            expected_target_time = actual_start_time

            # Pre-calculate expected frames
            alloc_end_time = actual_end_time if actual_end_time != float('inf') else video_duration
            expected_frames = 0
            if alloc_end_time > 0:
                duration_to_extract = alloc_end_time - actual_start_time
                if duration_to_extract > 0:
                    expected_frames = int(np.ceil(duration_to_extract / frame_interval)) + 2

            pbar = comfy.utils.ProgressBar(expected_frames) if expected_frames > 0 else None

            for frame in container.decode(video_stream):
                frame_time = frame.time
                if frame_time is None:
                    frame_time = float(frame.pts * float(video_stream.time_base)) if frame.pts and video_stream.time_base else 0.0

                if frame_time < actual_start_time:
                    continue

                # Add a slight buffer (interval) to ensure we evaluate the boundary correctly
                if frame_time > actual_end_time + frame_interval:
                    break

                # Fix PyAV color shift by forcing proper colorspace and range conversion.
                # Omit dst_colorspace so swscale defaults naturally for RGB output
                # (passing it can cause the YUV matrix to be applied incorrectly).
                try:
                    frame = frame.reformat(
                        format="rgb24",
                        src_colorspace=src_colorspace,
                        src_color_range=src_color_range,
                        dst_color_range=dst_range
                    )
                    frame_rgb = frame.to_ndarray(format='rgb24')
                except Exception as e:
                    # Fallback: if explicit color reformat fails, use PyAV's default conversion
                    print(f"[SeamStitch] Color reformat failed, using default: {e}")
                    frame_rgb = frame.to_ndarray(format='rgb24')

                # Apply interactive crop first
                if manual_crop_left > 0 or manual_crop_top > 0 or manual_crop_right > 0 or manual_crop_bottom > 0:
                    frame_rgb = frame_rgb[manual_crop_top:orig_h-manual_crop_bottom, manual_crop_left:orig_w-manual_crop_right, :]

                # Now resize to the scaled dimensions
                if scale_w != cropped_orig_w or scale_h != cropped_orig_h:
                    import cv2
                    frame_rgb = cv2.resize(frame_rgb, (scale_w, scale_h), interpolation=cv2.INTER_AREA)

                if crop_left > 0 or crop_top > 0 or crop_right > 0 or crop_bottom > 0:
                    frame_rgb = frame_rgb[crop_top:scale_h-crop_bottom, crop_left:scale_w-crop_right, :]
                if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                    frame_rgb = np.pad(frame_rgb, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

                # Duplicate or skip frames perfectly based on timestamps to meet forced framerate.
                # FIX: Use strictly less than (<) for actual_end_time to prevent the loop from fetching an extra +1 frame
                # at the exact boundary of the duration slice!
                while expected_target_time <= frame_time and expected_target_time < actual_end_time - 1e-5:
                    if image_tensor is None and expected_frames > 0:
                        # First frame: allocate the tensor
                        height, width = frame_rgb.shape[:2]
                        alloc_frames = expected_frames + 50 # Add generous buffer to prevent reallocation
                        try:
                            image_tensor = torch.zeros((alloc_frames, height, width, 3), dtype=torch.float32)
                        except Exception as e:
                            print(f"[SeamStitch] Pre-allocation failed, falling back to list: {e}")
                            expected_frames = 0 # Disable pre-allocation

                    if image_tensor is not None:
                        # Check bounds (just in case)
                        if frames_loaded >= image_tensor.shape[0]:
                            # Extend tensor if we underestimated
                            extension = torch.zeros((50, image_tensor.shape[1], image_tensor.shape[2], 3), dtype=torch.float32)
                            image_tensor = torch.cat((image_tensor, extension), dim=0)

                        # Insert frame with minimal memory copy directly to tensor
                        image_tensor[frames_loaded] = torch.from_numpy(frame_rgb).float().div_(255.0)
                        frames_loaded += 1
                    else:
                        # Fallback list append if pre-allocation failed
                        frames.append(frame_rgb)

                    if pbar:
                        pbar.update(1)

                    expected_target_time += frame_interval

        # Convert frames to ComfyUI Image standard format [N, H, W, C], float32, range 0.0-1.0
        if image_tensor is not None:
            if frames_loaded > 0:
                image_tensor = image_tensor[:frames_loaded]
            else:
                image_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        elif len(frames) > 0:
            frames_np = np.array(frames, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(frames_np)
        else:
            # Fallback for an empty slice
            image_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

        # 3. Extract Audio (PyAV)
        audio_dict = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100} # Default empty audio

        if len(container.streams.audio) > 0:
            try:
                audio_stream = container.streams.audio[0]
                audio_stream.thread_type = "AUTO"
                sample_rate = getattr(audio_stream, 'rate', 44100) or 44100

                # We must seek again on the container specifically for the audio stream
                if audio_stream.time_base:
                    seek_pts = int(actual_start_time / float(audio_stream.time_base))
                else:
                    seek_pts = int(actual_start_time * av.time_base)

                container.seek(seek_pts, stream=audio_stream, backward=True)

                # Resample to standard float planar format (fltp)
                resampler = av.AudioResampler(format='fltp')

                audio_data = []
                first_frame_time = None

                for frame in container.decode(audio_stream):
                    frame_time = frame.time
                    if frame_time is None:
                        frame_time = float(frame.pts * float(audio_stream.time_base)) if frame.pts and audio_stream.time_base else 0.0

                    # Give a small 1-second buffer to ensure we catch end frames
                    if frame_time > actual_end_time + 1.0:
                        break

                    if first_frame_time is None:
                        first_frame_time = frame_time

                    resampled_frames = resampler.resample(frame)
                    for r_frame in resampled_frames:
                        audio_data.append(r_frame.to_ndarray())

                if audio_data:
                    # Concatenate all frames horizontally along the sample axis
                    waveform_np = np.concatenate(audio_data, axis=1)
                    waveform = torch.from_numpy(waveform_np).float()

                    if first_frame_time is None:
                        first_frame_time = 0.0

                    # Calculate exact slice points to trim precisely
                    offset_sec = max(0.0, actual_start_time - first_frame_time)
                    start_sample = int(offset_sec * sample_rate)

                    duration_sec_audio = actual_end_time - actual_start_time
                    end_sample = start_sample + int(duration_sec_audio * sample_rate)

                    # Trim array bounds properly
                    if end_sample > start_sample:
                        waveform = waveform[:, start_sample:end_sample]
                    else:
                        waveform = waveform[:, start_sample:]

                    # Expand to ComfyUI Audio standard [batch_size, channels, samples]
                    waveform = waveform.unsqueeze(0)
                    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

            except Exception as e:
                # Catch gracefully without breaking the pipeline execution
                print(f"[SeamStitch] Audio track extraction skipped or failed: {e}")

        # 3b. Full clip's audio (entire source file, ignoring the trim range) — for feeding
        # into a final combine step that reuses the original audio end to end, independent
        # of whatever segment was selected above.
        #
        # Decoded from a FRESH container, not the one already used above for the video pass
        # and the trimmed-audio pass. Reusing one PyAV container across multiple seeks on
        # different streams is a known footgun: a later decode can silently come back short
        # instead of erroring, which previously made this output only cover a second or two
        # instead of the whole file.
        full_clip_audio_dict = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100}

        if len(container.streams.audio) > 0:
            full_container = None
            try:
                full_container = av.open(video_path)
                audio_stream = full_container.streams.audio[0]
                audio_stream.thread_type = "AUTO"
                sample_rate = getattr(audio_stream, 'rate', 44100) or 44100

                resampler = av.AudioResampler(format='fltp')
                audio_data = []

                for frame in full_container.decode(audio_stream):
                    resampled_frames = resampler.resample(frame)
                    for r_frame in resampled_frames:
                        audio_data.append(r_frame.to_ndarray())

                if audio_data:
                    waveform_np = np.concatenate(audio_data, axis=1)
                    waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
                    full_clip_audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

            except Exception as e:
                # Catch gracefully without breaking the pipeline execution
                print(f"[SeamStitch] Full-clip audio extraction skipped or failed: {e}")
            finally:
                if full_container is not None:
                    full_container.close()

        # Always close container to free up system memory lock
        container.close()

        # Output accurate final duration in seconds
        final_duration_sec = float(max(0.0, actual_end_time - actual_start_time))

        # Accurately output the true number of extracted frames
        # (Using the shape of the array provides exact 1:1 parity with the timeline's math)
        frame_count = image_tensor.shape[0] if (frames_loaded > 0 or len(frames) > 0) else 0
        if frame_count == 0 and final_duration_sec > 0:
             # Fallback estimation only if PyAV completely failed to decode a valid chunk
             calc_fr = float(frame_rate) if frame_rate > 0 else 24.0
             frame_count = int(np.floor(final_duration_sec * calc_fr))

        # 4. First/last frame of the selected (trimmed) range, as single-image batches.
        # Cloned so downstream in-place ops on one output can't affect the others.
        first_frame = image_tensor[0:1].clone()
        last_frame = image_tensor[-1:].clone()

        video_stem = os.path.splitext(os.path.basename(video))[0]
        if save_first_frame:
            saved_path = _save_frame_png(first_frame, f"{video_stem}_first_frame")
            print(f"[SeamStitch] Saved first frame to {saved_path}")
        if save_last_frame:
            saved_path = _save_frame_png(last_frame, f"{video_stem}_last_frame")
            print(f"[SeamStitch] Saved last frame to {saved_path}")

        # 5. Frame-index bounds of the selection, in the ORIGINAL video's own timeline at
        # this same forced frame_rate — lets a downstream node re-decode source_video_path
        # at the identical rate and know exactly which frames this selection replaces.
        effective_fr = float(frame_rate) if frame_rate > 0 else 24.0
        start_frame_idx = int(round(actual_start_time * effective_fr))
        end_frame_idx = start_frame_idx + frame_count - 1 if frame_count > 0 else start_frame_idx

        # Read width/height directly off the actual output tensor (post-crop/resize/pad)
        # rather than re-deriving through the resize math, so they can never drift from `images`.
        out_height = image_tensor.shape[1]
        out_width = image_tensor.shape[2]

        return (image_tensor, audio_dict, final_duration_sec, frame_count, os.path.basename(video), first_frame, last_frame, video_path, start_frame_idx, end_frame_idx, int(frame_rate), out_width, out_height, full_clip_audio_dict)

    @staticmethod
    def _resize_batch(images, w, h):
        import cv2
        arr = images.cpu().numpy()
        out = np.stack([cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) for frame in arr])
        return torch.from_numpy(out)

    def _load_from_tensor(self, input_video, input_audio, video, frame_rate, display_mode,
                           start_time, end_time, duration, start_frame, end_frame, duration_frames,
                           custom_width, custom_height, resize_method, crop_x, crop_y, crop_w, crop_h,
                           save_first_frame, save_last_frame):
        """Same trim/crop/resize contract as load_video's file path, but driven off
        frames already in the graph instead of decoding a file - no ffmpeg/PyAV
        decode, no colour-space handling (the tensor is already RGB), and this
        deliberately does not share code with the file-decode loop above so that
        path stays exactly as tested."""
        num_source_frames = input_video.shape[0]
        orig_h, orig_w = input_video.shape[1], input_video.shape[2]
        fr = float(frame_rate) if frame_rate > 0 else 24.0

        if display_mode == "frames":
            s_idx = max(0, int(start_frame))
            e_idx = int(end_frame) if (end_frame > 0 and end_frame > start_frame) else num_source_frames
        else:
            s_idx = max(0, int(round(start_time * fr)))
            e_idx = int(round(end_time * fr)) if (end_time > 0 and end_time > start_time) else num_source_frames
        e_idx = min(e_idx, num_source_frames)
        s_idx = min(s_idx, e_idx)

        sliced = input_video[s_idx:e_idx]
        if sliced.shape[0] == 0:
            sliced = input_video[0:1]
            s_idx, e_idx = 0, 1

        # Manual crop (interactive UI), same fractional math as the file path
        manual_crop_left = int(orig_w * crop_x)
        manual_crop_top = int(orig_h * crop_y)
        manual_crop_right = orig_w - int(orig_w * (crop_x + crop_w))
        manual_crop_bottom = orig_h - int(orig_h * (crop_y + crop_h))
        manual_crop_left = max(0, min(manual_crop_left, orig_w - 1))
        manual_crop_top = max(0, min(manual_crop_top, orig_h - 1))
        manual_crop_right = max(0, min(manual_crop_right, orig_w - manual_crop_left - 1))
        manual_crop_bottom = max(0, min(manual_crop_bottom, orig_h - manual_crop_top - 1))
        if manual_crop_left or manual_crop_top or manual_crop_right or manual_crop_bottom:
            sliced = sliced[:, manual_crop_top:orig_h - manual_crop_bottom, manual_crop_left:orig_w - manual_crop_right, :]

        cropped_h, cropped_w = sliced.shape[1], sliced.shape[2]
        target_w = custom_width if custom_width > 0 else cropped_w
        target_h = custom_height if custom_height > 0 else cropped_h
        target_w = target_w - (target_w % 2)
        target_h = target_h - (target_h % 2)

        if custom_width > 0 or custom_height > 0:
            if resize_method in ("maintain aspect ratio", "pad"):
                ratio = min(target_w / cropped_w, target_h / cropped_h)
                scale_w = int(cropped_w * ratio); scale_w -= scale_w % 2
                scale_h = int(cropped_h * ratio); scale_h -= scale_h % 2
                if scale_w != cropped_w or scale_h != cropped_h:
                    sliced = self._resize_batch(sliced, scale_w, scale_h)
                if resize_method == "pad":
                    pad_x, pad_y = target_w - scale_w, target_h - scale_h
                    pad_left, pad_top = pad_x // 2, pad_y // 2
                    pad_right, pad_bottom = pad_x - pad_left, pad_y - pad_top
                    sliced = torch.nn.functional.pad(
                        sliced.permute(0, 3, 1, 2), (pad_left, pad_right, pad_top, pad_bottom)
                    ).permute(0, 2, 3, 1)
                else:
                    target_w, target_h = scale_w, scale_h
            elif resize_method == "crop":
                ratio = max(target_w / cropped_w, target_h / cropped_h)
                scale_w = int(cropped_w * ratio); scale_w -= scale_w % 2
                scale_h = int(cropped_h * ratio); scale_h -= scale_h % 2
                sliced = self._resize_batch(sliced, scale_w, scale_h)
                cx, cy = (scale_w - target_w) // 2, (scale_h - target_h) // 2
                sliced = sliced[:, cy:cy + target_h, cx:cx + target_w, :]
            elif resize_method == "stretch to fit":
                sliced = self._resize_batch(sliced, target_w, target_h)

        image_tensor = sliced.clone()
        frame_count = image_tensor.shape[0]

        if input_audio is not None:
            sample_rate = input_audio["sample_rate"]
            waveform = input_audio["waveform"]
            start_sample = int((s_idx / fr) * sample_rate)
            end_sample = min(int((e_idx / fr) * sample_rate), waveform.shape[-1])
            start_sample = min(start_sample, end_sample)
            audio_dict = {"waveform": waveform[:, :, start_sample:end_sample], "sample_rate": sample_rate}
            full_clip_audio_dict = input_audio
        else:
            audio_dict = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100}
            full_clip_audio_dict = audio_dict

        first_frame = image_tensor[0:1].clone()
        last_frame = image_tensor[-1:].clone()

        label = video if video and video != "none" else "input_video"
        if save_first_frame:
            _save_frame_png(first_frame, f"{os.path.splitext(os.path.basename(label))[0]}_first_frame")
        if save_last_frame:
            _save_frame_png(last_frame, f"{os.path.splitext(os.path.basename(label))[0]}_last_frame")

        final_duration_sec = frame_count / fr
        out_height, out_width = image_tensor.shape[1], image_tensor.shape[2]

        # If the video dropdown also names a real file (e.g. the user picked the
        # combine node's own output there, via the Load Video button, purely for
        # reference/scrubbing while input_video supplies the actual frames),
        # resolve it for source_video_path so a downstream splice node still gets
        # something real to reopen. Otherwise there is genuinely no file to point
        # at, and this comes back empty.
        source_video_path = _resolve_video_path(video) or ""

        return (image_tensor, audio_dict, final_duration_sec, frame_count, os.path.basename(label),
                first_frame, last_frame, source_video_path, s_idx, s_idx + frame_count - 1, int(frame_rate),
                out_width, out_height, full_clip_audio_dict)
