"""Copyright 2026 SeamStitch contributors (https://github.com/isitrepo/SeamStitch), licensed GPL-3.0-only.

SeamStitchLTXGuides - pin SeamStitchLoader's start_context / end_context onto an
LTX-2.x latent, one frame per guide.

A single first/last-frame guide tells the model where each end of the bridge must
be, but not how fast anything is moving there, so it eases into its endpoint and
parks - then the real footage takes off at full speed. Pinning K consecutive real
frames at each end fixes the velocity too.

Each context frame goes in as its own single-frame LTXVAddGuide (core ComfyUI). A
multi-frame guide cannot do this: LTX places a 9+ frame guide only at frame 8n+1,
and on an 8n+1-long video no such guide can end on the last frame. Single-frame
guides accept any index. start_context[i] is pinned at frame i; end_context[j] of K
at frame -(K - j), so its last frame is the video's last frame.

With context_frames = 0 on the Loader, start_context / end_context are just
first_frame / last_frame, and this node does exactly what the usual pair of
LTXVAddGuide nodes (frame_idx 0 and -1) does.
"""

from comfy_extras.nodes_lt import LTXVAddGuide, preprocess


class SeamStitchLTXGuides:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "start_context": ("IMAGE", {"tooltip": "SeamStitchLoader.start_context. Frame i is pinned at bridge frame i."}),
                "end_context": ("IMAGE", {"tooltip": "SeamStitchLoader.end_context. Its last frame is pinned at the bridge's last frame, the rest just before it."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "LTXVAddGuide strength for every pinned frame."}),
                "img_compression": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                                            "tooltip": "LTXVPreprocess compression applied to each frame before it is pinned. 0 = none - leave it at 0 if the frames already went through an LTXVPreprocess node."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "apply"
    CATEGORY = "SeamStitch"

    def apply(self, positive, negative, vae, latent, start_context, end_context, strength,
              img_compression=0):
        def prep(frame):
            if img_compression > 0:
                frame = preprocess(frame, img_compression)
            return frame.unsqueeze(0)

        k_end = end_context.shape[0]
        placements = [(start_context[i], i) for i in range(start_context.shape[0])]
        placements += [(end_context[j], -(k_end - j)) for j in range(k_end)]
        for frame, idx in placements:
            positive, negative, latent = LTXVAddGuide.execute(
                positive, negative, vae, latent, prep(frame), idx, strength).args
        print(f"[SeamStitch] LTX guides: pinned {start_context.shape[0]} frame(s) at the start "
              f"and {k_end} at the end, strength {strength:g}.")
        return positive, negative, latent
