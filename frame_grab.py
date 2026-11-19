"""Copyright 2026 SeamStitch contributors (https://github.com/isitrepo/SeamStitch), licensed GPL-3.0-only.

SeamStitchFrameGrab - pull one or two exact frames out of a video as lossless images, for an edit
(Qwen) before they go back in as start / end pins. Pick the video, type a first-frame and a
last-frame number, and both frames come out as IMAGE plus a lossless PNG saved in the output folder.

Frame numbers count the way the Swap Planner and the Timeline count them: from the file's own first
frame, at the file's frame rate rounded to a whole number (25 fps stays 25). A last frame of -1 means
the final frame of the clip; any number past the end is clamped to the final frame.

Optional `images` input: connect a frame batch (the Timeline's output, a Loader, any IMAGE batch) and the
numbers index that batch instead of a file. Nothing is re-encoded and no frame is resized.
"""

import os

import numpy as np
import torch

try:
    import folder_paths
except ImportError:  # imported as a plain module (tests)
    folder_paths = None

try:
    from . import timeline as tl
except ImportError:  # imported as a top-level module (tests)
    import timeline as tl

LAST = -1


def clamp_frame(n, total):
    """A frame number made usable: -1 (or any negative) is the last frame, anything past the end is the last."""
    total = int(total)
    if total < 1:
        raise ValueError("the video has no frames")
    n = int(n)
    if n < 0:
        return total - 1
    return min(n, total - 1)


def pick_numbers(first, last, total):
    """(first, last) as real frame numbers, in order. The two may be equal (one frame twice)."""
    a = clamp_frame(first, total)
    b = clamp_frame(last, total)
    return a, b


def decode_one(path, frame_rate, n):
    """Frame `n` of a file as a [1, H, W, 3] float IMAGE (the Planner's decoder, so the numbers agree)."""
    for f in tl._iter_frames(path, frame_rate, n, n + 1):
        return torch.from_numpy(f).to(torch.float32).div_(255.0).unsqueeze(0)
    raise ValueError(f"{os.path.basename(path)}: could not decode frame {n}")


def write_png(image, directory, name, overwrite=False):
    """Save a [1, H, W, 3] IMAGE as a lossless 8-bit PNG; the path. Never overwrites a kept frame: a taken name gets _2, _3 ..."""
    from PIL import Image

    os.makedirs(directory, exist_ok=True)
    arr = (image[0].cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    base, ext = os.path.splitext(name)
    path = os.path.join(directory, base + ext)
    k = 2
    while os.path.exists(path) and not overwrite:
        path = os.path.join(directory, f"{base}_{k}{ext}")
        k += 1
    Image.fromarray(arr).save(path)
    return path


def _input_videos():
    try:
        from .loader import _list_input_videos
    except ImportError:
        from loader import _list_input_videos
    return _list_input_videos()


def _resolve(video):
    if os.path.isfile(video):
        return video
    p = os.path.join(folder_paths.get_input_directory(), video)
    if os.path.isfile(p):
        return p
    raise ValueError(f"video not found: {video}")


class SeamStitchFrameGrab:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_input_videos(), {"tooltip": "The video to take the frames from (ignored when `images` is connected)."}),
                "first_frame": ("INT", {"default": 0, "min": 0, "max": 10000000, "step": 1,
                                        "tooltip": "Frame number of the first frame. 0 is the clip's first frame."}),
                "last_frame": ("INT", {"default": LAST, "min": -1, "max": 10000000, "step": 1,
                                       "tooltip": "Frame number of the last frame. -1 is the final frame of the clip."}),
                "save_png": ("BOOLEAN", {"default": True, "tooltip": "Also save both frames as lossless PNGs in output/seamstitch_frames."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Optional: take the frames from this batch (e.g. the Timeline's output) instead of the video file."}),
                "frame_rate": ("INT", {"default": 0, "min": 0, "max": 240, "step": 1,
                                       "tooltip": "0 = the file's own rate, rounded (what the Planner uses). Only change it to match another tool's numbering."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("first_image", "last_image", "first_number", "last_number", "info")
    OUTPUT_NODE = True
    FUNCTION = "grab"
    CATEGORY = "SeamStitch"
    DESCRIPTION = ("Pull a first and a last frame out of a video (or a connected frame batch) as lossless images, "
                   "to edit and use as start / end pins.")

    def grab(self, video, first_frame, last_frame, save_png=True, images=None, frame_rate=0):
        if images is not None:
            total = int(images.shape[0])
            a, b = pick_numbers(first_frame, last_frame, total)
            img_a, img_b = images[a:a + 1].clone(), images[b:b + 1].clone()
            stem, source = "batch", f"the connected batch ({total} frames)"
        else:
            path = _resolve(video)
            info = tl.probe(path, frame_rate if frame_rate and frame_rate > 0 else 0)
            fr = info["frame_rate"]
            total = int(info["frames"])
            a, b = pick_numbers(first_frame, last_frame, total)
            img_a = decode_one(path, fr, a)
            img_b = img_a.clone() if b == a else decode_one(path, fr, b)
            stem = os.path.splitext(os.path.basename(path))[0]
            source = f"{os.path.basename(path)} ({total} frames at {fr:g} fps)"

        lines = [f"{source}: first = frame {a}, last = frame {b}; {img_a.shape[2]}x{img_a.shape[1]} px"]
        if save_png and folder_paths is not None:
            out = os.path.join(folder_paths.get_output_directory(), "seamstitch_frames")
            pa = write_png(img_a, out, f"{stem}_f{a:05d}.png")
            pb = pa if b == a else write_png(img_b, out, f"{stem}_f{b:05d}.png")
            lines.append(f"saved: {pa}")
            if pb != pa:
                lines.append(f"saved: {pb}")

        # preview in the node: the two frames, via the temp folder (previews are not kept)
        ui = []
        if folder_paths is not None and hasattr(folder_paths, "get_temp_directory"):
            tmp = folder_paths.get_temp_directory()
            for tag, im in (("first", img_a), ("last", img_b)):
                p = write_png(im, tmp, f"seamstitch_grab_{tag}.png", overwrite=True)
                ui.append({"filename": os.path.basename(p), "subfolder": "", "type": "temp"})
        return {"ui": {"images": ui}, "result": (img_a, img_b, a, b, "\n".join(lines))}



# The node's timeline widget (js/frame_grab.js) asks for the clip's facts and plays it from
# /seamstitch/loader/view (the Loader's route) by absolute path.
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/seamstitch/framegrab/info")
    async def _info(request):
        try:
            path = _resolve(request.query.get("video", ""))
            fr = int(request.query.get("frame_rate", "0") or 0)
            info = tl.probe(path, fr if fr > 0 else 0)
            return web.json_response({"path": path.replace("\\", "/"), "frames": int(info["frames"]),
                                      "frame_rate": float(info["frame_rate"]), "width": int(info["width"]),
                                      "height": int(info["height"])})
        except Exception as e:  # a missing file or an unreadable one: the widget shows the message
            return web.json_response({"error": str(e)}, status=400)
except ImportError:  # plain module (tests without a server)
    pass
