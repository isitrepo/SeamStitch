"""SeamStitch Swap Zone Latent: the latent H3 samples for an object add, a zone regenerated and the rest held.

An object job used to paste a cut-out on every frame and re-draw the composite img2img (denoise 0.5), which
reads as a sticker. This builds the other thing: the frames the job renders (the Planner's crop around the
target), encoded, with a mask over the ZONE where the object goes. H3 reads the mask on its audio-video latent
natively: masked cells are generated in the context of everything around them (light, shadow, contact), the
rest is held to the source, and the source's audio is held so a person added to the scene is lip-synced to it.

The mask maths (any-pixel pooling to 16 px cells and to latent frames, the feathered ramp, the held audio) is
adapted from ComfyUI-Fantastic-MiniMaxH3-PromptBuilder's video_edit.py (MIT, Copyright (c) 2026 Adudeguyman), so
the pack does not depend on it.

    mask   white = regenerate. One frame for the whole clip, or one per frame (a zone that follows its target).
    detail enlarges the frames before they are sampled (a small object gets more pixels; the render is brought
           back to the crop's size by the caller), never past max_megapixels.
"""

import math

import torch
import torch.nn.functional as F

CHUNK = 16      # frames per GPU pass


def usable_frames(n):
    """The longest H3 clip length (17k + 5 frames) that fits in n frames."""
    if n < 5:
        raise ValueError(f"The clip has {n} frames; H3 needs at least 5.")
    return (n - 5) // 17 * 17 + 5


def sample_size(w, h, detail, max_megapixels):
    """(tw, th): the frames' size times `detail`, at no more than max_megapixels (0 = no cap), multiples of 32."""
    scale = max(1.0, float(detail))
    if max_megapixels and w * h * scale * scale > max_megapixels * 1e6:
        scale = max(1.0, math.sqrt(max_megapixels * 1e6 / float(w * h)))
    return max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)


def resize_frames(frames, tw, th, device):
    """[N, H, W, 3] float 0..1 -> [N, th, tw, 3], a chunk at a time ('area' when shrinking, bicubic when enlarging)."""
    n, h, w = frames.shape[:3]
    out = torch.empty((n, th, tw, 3), dtype=torch.float32)
    for i in range(0, n, CHUNK):
        x = frames[i:i + CHUNK, ..., :3].to(device).float().movedim(-1, 1)
        if (h, w) != (th, tw):
            if th * tw < h * w:
                x = F.interpolate(x, size=(th, tw), mode="area")
            else:
                x = F.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False).clamp(0, 1)
        out[i:i + CHUNK] = x.movedim(1, -1).cpu()
    return out


def grow_mask(mask, px, device):
    """Widen [N, H, W] by px pixels (a square dilation as two 1-D passes)."""
    if px <= 0:
        return mask
    k = 2 * px + 1
    out = torch.empty_like(mask)
    for i in range(0, mask.shape[0], CHUNK):
        m = mask[i:i + CHUNK].to(device, torch.float32)[:, None]
        m = F.max_pool2d(m, (1, k), stride=1, padding=(0, px))
        m = F.max_pool2d(m, (k, 1), stride=1, padding=(px, 0))
        out[i:i + CHUNK] = m[:, 0].cpu()
    return out


def latent_mask(mask, frame_counts, lat_h, lat_w):
    """Pixel mask [F, H, W] -> [1, 1, T, lat_h, lat_w]. Each latent frame and each 16x16 cell is masked when any
    pixel it covers is, so a thin edge never rounds away."""
    cells = F.adaptive_max_pool2d(mask[:, None], (lat_h, lat_w))[:, 0]
    rows, at = [], 0
    for n in frame_counts:
        rows.append(cells[at:at + n].amax(dim=0))
        at += n
    return torch.stack(rows)[None, None]


def feather_latent(mask, cells):
    """[1, 1, T, h, w] latent mask -> the same with a ramp `cells` wide outside it (a cell k away gets
    1 - k / (cells + 1)), so H3 blends the edge instead of cutting it. The inside stays 1."""
    if cells <= 0:
        return mask
    x = near = mask[0].movedim(0, 1)
    out = x
    for k in range(1, cells + 1):
        near = F.max_pool2d(near, 3, stride=1, padding=1)
        out = torch.maximum(out, near * (1 - k / (cells + 1)))
    return out.movedim(1, 0)[None]


def build(vae, frames, mask, grow, feather, detail, max_megapixels, audio_vae=None, audio=None):
    """Encode the zone edit. frames [N, H, W, 3] float, mask [N or 1, h, w] (white = regenerate), grow and feather
    in the frames' pixels. Returns ({"samples", "noise_mask"} nested latent, tw, th, n frames)."""
    import comfy.model_management as mm
    import comfy.nested_tensor
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN
    from comfy_extras.nodes_minimax_h3 import temporal_shape, _encode_ref_audio

    if audio is not None and audio_vae is None:
        raise ValueError("Holding the source audio needs the H3 audio VAE.")
    n = usable_frames(frames.shape[0])
    h, w = frames.shape[1], frames.shape[2]
    m = mask.reshape(-1, mask.shape[-2], mask.shape[-1]).float()
    if m.shape[0] == 1:
        m = m.expand(n, -1, -1)
    elif m.shape[0] < n:
        raise ValueError(f"The zone has {m.shape[0]} frames and the clip {n}: give one zone per frame, or a single frame.")
    m = (m[:n] > 0.5).float()
    if m.shape[-2:] != (h, w):
        m = F.interpolate(m[:, None], size=(h, w), mode="nearest")[:, 0]
    dev = mm.get_torch_device()
    obj = grow_mask(m, int(grow), dev)
    if not obj.any():
        raise ValueError("The zone is empty on every frame, so nothing would be drawn.")
    tw, th = sample_size(w, h, detail, max_megapixels)

    clip = resize_frames(frames[:n], tw, th, dev)
    video = vae.encode(clip).to(mm.intermediate_device())
    del clip
    _count, latent_t, audio_t = temporal_shape(n)
    if video.shape[2] != latent_t:
        raise ValueError(f"The VAE gave {video.shape[2]} latent frames for {n} pixel frames; H3 expects {latent_t}.")
    rows = []
    for i in range(0, n, CHUNK):
        r = F.interpolate(obj[i:i + CHUNK, None].to(dev), size=(th, tw), mode="area" if th * tw < h * w else "nearest")
        rows.append(F.adaptive_max_pool2d((r > 0).float(), (th // 16, tw // 16))[:, 0].cpu())
    counts = [FRAME_PER_TOKEN[k % 5] for k in range(latent_t)]
    cells = latent_mask(torch.cat(rows), counts, th // 16, tw // 16)
    video_mask = feather_latent(cells, math.ceil(feather * tw / w / 16))[0].to(video.device)

    audio_latent = torch.zeros([1, 32, 2, audio_t], device=video.device)
    audio_mask = torch.ones_like(audio_latent)
    if audio is not None:
        z, _t = _encode_ref_audio(audio_vae, audio)
        z = z.to(video.device)[..., :audio_t]
        audio_latent[..., :z.shape[-1]] = z
        audio_mask[..., :z.shape[-1]] = 0.0
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio_latent)),
              "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask))}
    return latent, tw, th, n


class SeamStitchSwapZoneLatent:
    CATEGORY = "SeamStitch/Swap"
    DESCRIPTION = (
        "The latent MiniMax H3 samples for an object add: your frames encoded, the white zone of the mask "
        "regenerated, everything else held to the source (and the source's audio, if given). Use it in place of "
        "the empty latent and wire width, height and length into the H3 conditioning.")
    RETURN_TYPES = ("LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height", "length")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "The H3 video VAE."}),
                "images": ("IMAGE", {"tooltip": "The frames the job renders (the Planner's crop), at 24 fps."}),
                "zone": ("MASK", {"tooltip": "White = drawn by H3. One frame for the whole clip, or one per frame."}),
                "grow": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1,
                                 "tooltip": "Widen the zone by this many pixels so edges and shadows are drawn too."}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1,
                                    "tooltip": "A ramp this many pixels wide outside the zone, blended rather than cut."}),
                "detail": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1,
                                     "tooltip": "Enlarge the frames before sampling: a small object gets more pixels."}),
                "max_megapixels": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 16.0, "step": 0.05,
                                             "tooltip": "Never sample bigger than this (0 = no cap). About 1.3 keeps a "
                                                        "175-frame clip under the token limit."}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "The H3 audio VAE, needed to hold the source audio."}),
                "audio": ("AUDIO", {"tooltip": "The source's soundtrack, held as it is (a person added to the scene "
                                               "is lip-synced to it). Without it the sound is generated."}),
            },
        }

    def build(self, vae, images, zone, grow=16, feather=0, detail=1.0, max_megapixels=1.3, audio_vae=None, audio=None):
        latent, tw, th, n = build(vae, images, zone, grow, feather, detail, max_megapixels, audio_vae, audio)
        return (latent, tw, th, n)


class SeamStitchSwapZoneFirst:
    """The picture H3 starts from: the frames' first frame, or the anchor when the thing is there from the start."""
    CATEGORY = "SeamStitch/Swap"
    DESCRIPTION = ("The first frame for the H3 conditioning of a zone add: frame 0 of the render's frames, or the "
                   "anchor picture itself when anchor_frame is 0 (the thing is in the scene from the first frame).")
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("first_frame",)
    FUNCTION = "pick"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE", {"tooltip": "The Planner's images (the render's crop)."}),
            "anchor": ("IMAGE", {"tooltip": "The anchor: the thing drawn into one frame, at the frames' size."}),
            "anchor_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "The Planner's anchor_frame."}),
        }}

    def pick(self, frames, anchor, anchor_frame):
        return ((anchor if int(anchor_frame) == 0 else frames)[:1],)


class SeamStitchAnchorHold:
    """A stop sign between the anchor picture and the H3 render: with `hold` on, the run ends here (as if cancelled: the Planner
    puts the chunk back, nothing is queued behind it) once every preview and comparer fed by the anchor has run, so a bad anchor
    costs a minute of Qwen and not a whole H3 render. Turn `hold` off and render again to go on."""
    CATEGORY = "SeamStitch/Swap"
    DESCRIPTION = ("Stops the run after the anchor picture is drawn (hold on), so it can be looked at in a comparer before H3 "
                   "starts; with hold off the picture passes through unchanged.")
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("anchor",)
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "The anchor picture (after the Qwen / cut-out / own-picture switch)."}),
            "hold": ("BOOLEAN", {"default": True, "label_on": "stop here: review the anchor", "label_off": "go on to H3", "tooltip":
                                 "On: the run stops here, after the anchor is drawn and shown. Off: it carries on into the H3 render."}),
        }}

    @classmethod
    def IS_CHANGED(cls, image=None, hold=True):
        return float("nan") if hold else ""            # a held run is never taken from the cache: it must stop every time

    def go(self, image, hold):
        if hold:
            import comfy.model_management as mm
            print("[SeamStitch] Anchor Hold: stopped after the anchor, before H3 (turn the hold off to render)", flush=True)
            raise mm.InterruptProcessingException()
        return (image,)


def guide_frames(enter_frame, anchor_frame, step):
    """[(frame, "empty" | "anchor")] the zone add pins after its first frame: the empty scene every `step` frames
    before the thing enters (so it isn't there yet), then the anchor at anchor_frame (frame 0 is the first frame)."""
    out = []
    if enter_frame > 0 and step > 0:
        out += [(f, "empty") for f in range(step, int(enter_frame), int(step)) if f != anchor_frame]
    if anchor_frame > 0:
        out.append((int(anchor_frame), "anchor"))
    return out


class SeamStitchSwapZoneGuides:
    """Pins a zone add's anchor and its empty-scene frames on the H3 conditioning (core MiniMaxH3AddGuide)."""
    CATEGORY = "SeamStitch/Swap"
    DESCRIPTION = ("Adds the guides of a zone add to the H3 conditioning: the empty scene (the render's own frames) every "
                   "`empty_step` frames until enter_frame, so the thing is not there before it enters, and the anchor "
                   "picture at anchor_frame. One anchor only: more of them freeze the thing's pose.")
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "latent": ("LATENT", {"tooltip": "The Swap Zone Latent's latent (sets the guides' size)."}),
            "vae": ("VAE", {"tooltip": "The H3 video VAE."}),
            "frames": ("IMAGE", {"tooltip": "The Planner's images (the render's crop): the empty scene."}),
            "anchor": ("IMAGE", {"tooltip": "The anchor: the thing drawn into one frame, at the frames' size."}),
            "enter_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "The Planner's enter_frame: the zone stays empty before it."}),
            "anchor_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "The Planner's anchor_frame."}),
            "empty_step": ("INT", {"default": 4, "min": 1, "max": 64, "tooltip": "An empty-scene guide every this many frames."}),
        }}

    def apply(self, positive, latent, vae, frames, anchor, enter_frame, anchor_frame, empty_step=4):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
        n = frames.shape[0]
        done = []
        for f, kind in guide_frames(int(enter_frame), int(anchor_frame), int(empty_step)):
            if f >= n:
                continue
            img = anchor[:1] if kind == "anchor" else frames[f:f + 1]
            positive = MiniMaxH3AddGuide.execute(positive, latent, f, vae=vae, image=img).args[0]
            done.append(f"{f}{'a' if kind == 'anchor' else ''}")
        print(f"[SeamStitch] Swap Zone Guides: {len(done)} guide(s) at frames {', '.join(done) or 'none'} "
              f"(enter {int(enter_frame)}, anchor {int(anchor_frame)})")
        return (positive,)
