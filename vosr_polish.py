"""Copyright 2026 SeamStitch contributors (https://github.com/isitrepo/SeamStitch), licensed GPL-3.0-only.

SeamStitchVosrPolish - an optional last step that sharpens a finished clip with VOSR 2.0
(one-step per-frame super-resolution, the ComfyUI-VOSR2 pack). Put it after Recombine / Swap
Assemble and before the save node; switch `enable` off and the frames pass through untouched
(the model is not even loaded).

mode:
- "sharpen (same size)" (default): the tested recipe. Each frame is shrunk to half size, then
  VOSR enlarges it x2 back, so the output has the input's size and gains the fine detail
  a render at ~1 MP loses (hair, skin, wall texture).
- "enlarge x2": VOSR runs straight on the frames and the output is twice the size.

VOSR is generative per frame: sharper than a plain upscale but about 10% more frame-to-frame
flicker, and it deviates from the original at pixel level (tested on a bird-on-shoulder shot).
"""

import torch
import torch.nn.functional as F

MODE_SHARPEN = "sharpen (same size)"
MODE_ENLARGE = "enlarge x2"
VOSR_MODEL = "VOSR2"   # the bundle folder under models/vosr2 (the ComfyUI-VOSR2 pack's own default)


def _registered(name):
    try:
        import nodes
        return nodes.NODE_CLASS_MAPPINGS.get(name)
    except Exception:
        return None


def _resize(images, h, w, mode):
    """images [B,H,W,C] float -> [B,h,w,C]; a no-op when the size already matches."""
    if images.shape[1] == h and images.shape[2] == w:
        return images
    x = images.permute(0, 3, 1, 2)
    kw = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(x, size=(h, w), mode=mode, **kw).clamp(0, 1).permute(0, 2, 3, 1).contiguous()


def plan_sizes(h, w, mode):
    """(pre_size, vosr_upscale, out_size): the frame size fed to VOSR, its multiplier, the final size."""
    if mode == MODE_ENLARGE:
        return (h, w), 2, (h * 2, w * 2)
    return (max(1, round(h / 2)), max(1, round(w / 2))), 2, (h, w)


class SeamStitchVosrPolish:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enable": ("BOOLEAN", {"default": True, "tooltip": "Off = the frames pass through unchanged and VOSR is not loaded."}),
                "mode": ([MODE_SHARPEN, MODE_ENLARGE], {"default": MODE_SHARPEN, "tooltip":
                    "'sharpen (same size)': shrink to half, VOSR x2 back to the original size. "
                    "'enlarge x2': VOSR on the frames as they are, output twice the size."}),
                "vosr_seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "color_alignment": (["wavelet", "adain", "none"], {"default": "wavelet"}),
                "tile_size": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 64, "tooltip":
                    "VOSR was trained at up to 512 px; keep tiling on (512) for anything larger."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "polish"
    CATEGORY = "SeamStitch"

    def polish(self, images, enable=True, mode=MODE_SHARPEN, vosr_seed=1, color_alignment="wavelet", tile_size=512):
        if not enable:
            return (images,)
        loader, upscaler = _registered("VOSR2ModelLoader"), _registered("VOSR2Upscale")
        if loader is None or upscaler is None:
            raise RuntimeError("SeamStitchVosrPolish needs the ComfyUI-VOSR2 pack (VOSR 2.0 Model Loader / "
                               "Upscale nodes). Install it, or switch 'enable' off.")
        b, h, w, _ = images.shape
        (ph, pw), up, (oh, ow) = plan_sizes(h, w, mode)
        small = _resize(images[..., :3].float(), ph, pw, "area") if (ph, pw) != (h, w) else images[..., :3].float()
        model = loader.execute(VOSR_MODEL, "default").args[0]
        out = upscaler.execute(model, small, up, vosr_seed, color_alignment, tile_size, 32, 1024 if tile_size else 0, 32).args[0]
        out = _resize(out, oh, ow, "bicubic")
        print(f"[SeamStitch] VOSR polish ({mode}): {b} frame(s) {w}x{h} -> {ow}x{oh}.")
        return (out,)

