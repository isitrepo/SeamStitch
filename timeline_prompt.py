"""SeamStitch Timeline Prompt: the 27B writes the Timeline's prompt from Picture 1, Picture 2 and the footage either side.

An alternative to the 8B QwenVL captioner, which only sees the first and last frame. It uses Swap Draft's own
GGUF runner (GgufQwenRunner on QwenVL's GGUF side), so the model, its chat handler and its sampling are the ones
Swap Draft already runs. Besides the two pictures it can show a few frames from the seconds just before Picture 1
and just after Picture 2 (read from the Timeline's source video), so the prompt can carry on the movement and the
camera and lead into what follows. The model is unloaded when the call ends: the generator needs the VRAM.
"""

import cv2
import numpy as np
import torch

try:
    from . import swap_draft as sd
    from . import timeline as tl
except ImportError:  # imported as a top-level module (tests, tools)
    import swap_draft as sd
    import timeline as tl

CONTEXT_MAX_SIDE = 448      # the extra frames are only for movement, so small (the two pictures go in at full size)
NO_MODEL = "(no GGUF vision model on disk)"


def side_frames(path, fr, a, b, k):
    """k frames spread evenly over source frames a..b (cut short by the start or end of the clip), each a
    [H, W, 3] float tensor shrunk to CONTEXT_MAX_SIDE; an empty list when there are none."""
    a = max(0, a)
    if k <= 0 or b < a:
        return []
    small = []
    for f in tl._iter_frames(path, fr, a, b + 1):
        s = CONTEXT_MAX_SIDE / float(max(f.shape[:2]))
        if s < 1:
            f = cv2.resize(f, (int(round(f.shape[1] * s)), int(round(f.shape[0] * s))), interpolation=cv2.INTER_AREA)
        small.append(f)
    return [torch.from_numpy(np.ascontiguousarray(small[i])).to(torch.float32).div_(255.0)
            for i in sd.sample_indices(len(small), k)]


def context_note(before, after, secs):
    """The line that tells the model what the extra pictures are (they follow Picture 2, before then after)."""
    parts = []
    if before:
        parts.append(f"The next {len(before)} pictures are frames in time order from the {secs:.1f} seconds of the "
                     f"video just before PICTURE_1.")
    if after:
        which = f"The {len(after)} pictures after those" if before else f"The next {len(after)} pictures"
        parts.append(f"{which} are frames in time order from the {secs:.1f} seconds just after PICTURE_2.")
    return ("Extra footage for reference. " + " ".join(parts) + " They are not part of the shot you are writing: use "
            "them only to read the direction and speed of movement, the camera movement and anything already in "
            "progress, so the shot carries on from the first and leads into the second. The shot still opens exactly "
            "on PICTURE_1 and ends exactly on PICTURE_2, and your answer keeps the format above exactly.")


class SeamStitchTimelinePrompt:
    CATEGORY = "SeamStitch"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        models = sd.gguf_models() or [NO_MODEL]
        default = next((m for m in models if "27B" in m or "RVN" in m), models[0])
        return {
            "required": {
                "first_frame": ("IMAGE", {"tooltip": "Picture 1: the Timeline's first_frame."}),
                "last_frame": ("IMAGE", {"tooltip": "Picture 2: the Timeline's last_frame."}),
                "prompt": ("STRING", {"forceInput": True, "tooltip":
                    "The instruction the model answers, the same text the 8B captioner gets (the MMH3 template)."}),
                "qwen_model": (models, {"default": default, "tooltip":
                    "A QwenVL GGUF vision model that is on disk with its mmproj (Swap Draft's GGUF list)."}),
                "context_pictures": ("INT", {"default": 4, "min": 0, "max": 8, "step": 1, "tooltip":
                    "Frames shown from before Picture 1 and from after Picture 2, each side. 0 = just the two "
                    "pictures, as the 8B captioner sees them."}),
                "context_seconds": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip":
                    "How many seconds of footage each side those frames are spread over."}),
                "think": ("BOOLEAN", {"default": False, "tooltip":
                    "Let the model reason before it answers: slower (minutes), sometimes more careful."}),
                "max_tokens": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 64}),
            },
            "optional": {
                "source_video_path": ("STRING", {"forceInput": True, "tooltip":
                    "The Timeline's source_video_path: where the footage either side is read from."}),
                "start_frame": ("INT", {"forceInput": True, "tooltip": "The Timeline's start_frame."}),
                "end_frame": ("INT", {"forceInput": True, "tooltip": "The Timeline's end_frame."}),
                "frame_rate": ("INT", {"forceInput": True, "tooltip": "The Timeline's frame_rate."}),
            },
        }

    def run(self, first_frame, last_frame, prompt, qwen_model, context_pictures, context_seconds, think, max_tokens,
            source_video_path=None, start_frame=0, end_frame=0, frame_rate=0):
        if qwen_model == NO_MODEL:
            raise sd.DraftError("no QwenVL GGUF vision model with its mmproj is on disk (models/LLM/GGUF)")
        gmod = sd.locate_qwen_gguf()
        if gmod is None:
            raise sd.DraftError("SeamStitch Timeline Prompt needs ComfyUI-QwenVL's GGUF side (llama-cpp-python), "
                                "which didn't load")
        before = after = []
        if context_pictures > 0 and source_video_path and frame_rate:
            span = max(1, int(round(context_seconds * frame_rate)))
            before = side_frames(source_video_path, frame_rate, start_frame - span, start_frame - 1, context_pictures)
            after = side_frames(source_video_path, frame_rate, end_frame + 1, end_frame + span, context_pictures)
        parts = [prompt, first_frame[0], last_frame[0]]
        if before or after:
            parts += [context_note(before, after, context_seconds)] + before + after
        sd._say(f"[SeamStitch] Timeline Prompt: {qwen_model}, pictures 1+2 and {len(before)} before / {len(after)} "
                f"after, thinking {'on' if think else 'off'}")
        sd.free_comfy_models()
        qwen = sd.GgufQwenRunner(gmod, qwen_model)
        try:
            qwen.load()
            text = qwen.ask_parts(parts, max_tokens=max_tokens, think=think)
        finally:
            qwen.close()
        # H3's prompt wants one blank line between the alignment line (the Timeline's picture_timing, which
        # String Concatenate puts in front with no gap) and the fields, so the answer opens with one
        return ("\n\n" + (text or "").strip(),)
