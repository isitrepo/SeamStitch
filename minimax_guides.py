"""Copyright 2026 SeamStitch contributors (https://github.com/isitrepo/SeamStitch), licensed GPL-3.0-only.

SeamStitchMiniMaxGuides - pin SeamStitchLoader's start_context / end_context onto a
MiniMax H3 AV latent, the MiniMax counterpart of SeamStitchLTXGuides.

A first/last-frame anchor fixes where each end of the bridge must be, not how fast
anything is moving there. Pinning K consecutive real frames at each end fixes the
velocity too.

Each anchor goes through core ComfyUI's own MiniMaxH3AddGuide. H3 places every
keyframe on the target timeline by pixel frame (FRAME_RESCALE per frame), so a
single-frame anchor is valid at any index.

anchor_mode:
- "per frame" (default): one single-frame anchor per context frame. Any K works.
  start_context[i] is pinned at frame i; end_context[j] of K at frame -(K - j).
- "clip": each side goes in as ONE multi-frame clip anchor (start at 0, end at -K),
  which H3 encodes temporally, the way it anchors motion natively. Costs fewer
  tokens, but H3 only takes clips of 5, 22, 39... (17k + 5) frames, so K must be
  one of those.

For MiniMaxH3ReferenceToVideo's ref_image_0 / ref_image_1, take start_context[0] and
end_context[-1] with core ImageFromBatch (batch_index 0 and -1, length 1), so "Picture 1"
and "Picture 2" in the prompt still mean the bridge's first and last frame. They cannot
come out of this node: its conditioning comes from MiniMaxH3ReferenceToVideo, so feeding
anything back into that node's references would be a dependency cycle.

With context_frames = 0 on the Loader, start_context / end_context are just
first_frame / last_frame, and this node does exactly what the usual pair of
MiniMaxH3AddGuide nodes (frame_idx 0 and -1) does.
"""

from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

ANCHOR_PER_FRAME = "per frame"
ANCHOR_CLIP = "clip"


def plan_anchors(k_start, k_end, anchor_mode):
    """[(side, first_index, count, frame_idx)] for the anchors to add, in order.
    side is "start" or "end"; the slice is context[first_index:first_index + count].
    A side with 0 frames (an empty batch, e.g. the Swap Planner's pins on a side with no
    rendered neighbour) is skipped."""
    if anchor_mode == ANCHOR_CLIP:
        plan = []
        for side, k, idx in (("start", k_start, 0), ("end", k_end, -k_end)):
            if k == 0:
                continue
            if k != 1 and (k < 5 or k % 17 != 5):
                raise ValueError(
                    f"SeamStitchMiniMaxGuides: clip anchors need 1 or 5, 22, 39... (17k+5) context "
                    f"frames per side; {side}_context has {k}. Set the Loader's context_frames to 5 "
                    f"(or 22), or use anchor_mode 'per frame'.")
            plan.append((side, 0, k, idx))
        return plan
    plan = [("start", i, 1, i) for i in range(k_start)]
    plan += [("end", j, 1, -(k_end - j)) for j in range(k_end)]
    return plan


class SeamStitchMiniMaxGuides:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "start_context": ("IMAGE", {"tooltip": "SeamStitchLoader.start_context. Frame i is pinned at bridge frame i."}),
                "end_context": ("IMAGE", {"tooltip": "SeamStitchLoader.end_context. Its last frame is pinned at the bridge's last frame, the rest just before it."}),
                "anchor_mode": ([ANCHOR_PER_FRAME, ANCHOR_CLIP], {"default": ANCHOR_PER_FRAME, "tooltip":
                    "'per frame': one single-frame anchor per context frame, any K. 'clip': each side "
                    "as one multi-frame clip anchor (H3's native motion anchor, fewer tokens) - K must "
                    "be 5, 22, 39... (17k+5)."}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "Not needed for image anchors; passed through to MiniMaxH3AddGuide."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "apply"
    CATEGORY = "SeamStitch"

    def apply(self, positive, latent, vae, start_context, end_context, anchor_mode=ANCHOR_PER_FRAME,
              audio_vae=None):
        k_start, k_end = start_context.shape[0], end_context.shape[0]
        sources = {"start": start_context, "end": end_context}
        for side, first, count, idx in plan_anchors(k_start, k_end, anchor_mode):
            frames = sources[side][first:first + count]
            positive = MiniMaxH3AddGuide.execute(positive, latent, idx, vae=vae, audio_vae=audio_vae,
                                                 image=frames).args[0]
        print(f"[SeamStitch] MiniMax guides ({anchor_mode}): pinned {k_start} frame(s) at the start "
              f"and {k_end} at the end.")
        return (positive,)
