import logging

from .combine import SeamStitchCombine
from .loader import SeamStitchLoader

NODE_CLASS_MAPPINGS = {
    "SeamStitchCombine": SeamStitchCombine,
    "SeamStitchLoader": SeamStitchLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeamStitchCombine": "SeamStitch Combine",
    "SeamStitchLoader": "SeamStitch Loader",
}

# SeamStitchRecombine reuses ComfyUI-VideoHelperSuite's own ffmpeg/encode
# internals directly (see recombine.py) and raises ImportError at import
# time if VHS isn't installed alongside this pack. Guarded here so a
# missing VHS install only drops this one node instead of taking the whole
# package down with it.
try:
    from .recombine import SeamStitchRecombine
    NODE_CLASS_MAPPINGS["SeamStitchRecombine"] = SeamStitchRecombine
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchRecombine"] = "SeamStitch Recombine"
except ImportError as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] SeamStitchRecombine not loaded: {e}")

# SeamStitchLTXGuides wraps core ComfyUI's own LTXVAddGuide (comfy_extras.nodes_lt);
# a ComfyUI too old to have LTX support drops only this node.
try:
    from .ltx_guides import SeamStitchLTXGuides
    NODE_CLASS_MAPPINGS["SeamStitchLTXGuides"] = SeamStitchLTXGuides
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchLTXGuides"] = "SeamStitch LTX Guides"
except ImportError as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] SeamStitchLTXGuides not loaded: {e}")

# SeamStitchMiniMaxGuides wraps core ComfyUI's own MiniMaxH3AddGuide
# (comfy_extras.nodes_minimax_h3); a ComfyUI without H3 support drops only this node.
try:
    from .minimax_guides import SeamStitchMiniMaxGuides
    NODE_CLASS_MAPPINGS["SeamStitchMiniMaxGuides"] = SeamStitchMiniMaxGuides
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchMiniMaxGuides"] = "SeamStitch MiniMax Guides"
except ImportError as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] SeamStitchMiniMaxGuides not loaded: {e}")

# The one-track timeline (Combine + Loader in one node) and the result preview.
# Guarded so a problem here never takes the other nodes down.
try:
    from .timeline import SeamStitchTimeline
    from .result_preview import SeamStitchResultPreview
    NODE_CLASS_MAPPINGS["SeamStitchTimeline"] = SeamStitchTimeline
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchTimeline"] = "SeamStitch Timeline"
    NODE_CLASS_MAPPINGS["SeamStitchResultPreview"] = SeamStitchResultPreview
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchResultPreview"] = "SeamStitch Result Preview"
    from .batch_splice import SeamStitchBatchSplice
    NODE_CLASS_MAPPINGS["SeamStitchBatchSplice"] = SeamStitchBatchSplice
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchBatchSplice"] = "SeamStitch Batch Splice"
except Exception as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] Timeline / Result Preview not loaded: {e}")

# SeamStitch Swap (long-video chunk replacement): plan file, joins, assembly.
# Guarded so a problem here never takes the other nodes down.
try:
    from .swap_assemble import SeamStitchSwapAssemble
    NODE_CLASS_MAPPINGS["SeamStitchSwapAssemble"] = SeamStitchSwapAssemble
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapAssemble"] = "SeamStitch Swap Assemble"
except Exception as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] Swap Assemble not loaded: {e}")
try:
    from .swap_planner import SeamStitchSwapPlanner, SeamStitchSwapOption
    from .swap_take import SeamStitchSwapTake, SeamStitchSwapOutputPerson
    from .swap_mask import SeamStitchSwapMask
    NODE_CLASS_MAPPINGS["SeamStitchSwapMask"] = SeamStitchSwapMask
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapMask"] = "SeamStitch Swap Mask"
    NODE_CLASS_MAPPINGS["SeamStitchSwapPlanner"] = SeamStitchSwapPlanner
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapPlanner"] = "SeamStitch Swap Planner"
    NODE_CLASS_MAPPINGS["SeamStitchSwapOption"] = SeamStitchSwapOption
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapOption"] = "SeamStitch Swap Option"
    NODE_CLASS_MAPPINGS["SeamStitchSwapTake"] = SeamStitchSwapTake
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapTake"] = "SeamStitch Swap Take"
    NODE_CLASS_MAPPINGS["SeamStitchSwapOutputPerson"] = SeamStitchSwapOutputPerson
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapOutputPerson"] = "SeamStitch Swap Output Person"
except Exception as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] Swap Planner / Option / Take / Mask not loaded: {e}")

# SeamStitch Swap Draft Prompts calls the QwenVL and Omni custom nodes in-process (located at run time).
try:
    from .swap_draft import SeamStitchSwapDraft
    NODE_CLASS_MAPPINGS["SeamStitchSwapDraft"] = SeamStitchSwapDraft
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapDraft"] = "SeamStitch Swap Draft Prompts"
except Exception as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] Swap Draft Prompts not loaded: {e}")

# SeamStitch Timeline Prompt: the 27B drafts the Timeline's prompt (Swap Draft's GGUF runner).
try:
    from .timeline_prompt import SeamStitchTimelinePrompt
    NODE_CLASS_MAPPINGS["SeamStitchTimelinePrompt"] = SeamStitchTimelinePrompt
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchTimelinePrompt"] = "SeamStitch Timeline Prompt (27B)"
except Exception as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] Timeline Prompt not loaded: {e}")

# SeamStitch Swap Zone Latent: an object add's latent (a zone regenerated, the rest and the audio held).
# Needs core ComfyUI's MiniMax H3 support; a ComfyUI without it drops only this node.
try:
    from .swap_zone import SeamStitchSwapZoneLatent, SeamStitchSwapZoneFirst, SeamStitchSwapZoneGuides, SeamStitchAnchorHold
    NODE_CLASS_MAPPINGS["SeamStitchSwapZoneLatent"] = SeamStitchSwapZoneLatent
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapZoneLatent"] = "SeamStitch Swap Zone Latent"
    NODE_CLASS_MAPPINGS["SeamStitchSwapZoneFirst"] = SeamStitchSwapZoneFirst
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapZoneFirst"] = "SeamStitch Swap Zone First Frame"
    NODE_CLASS_MAPPINGS["SeamStitchSwapZoneGuides"] = SeamStitchSwapZoneGuides
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapZoneGuides"] = "SeamStitch Swap Zone Guides"
    NODE_CLASS_MAPPINGS["SeamStitchAnchorHold"] = SeamStitchAnchorHold
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchAnchorHold"] = "SeamStitch Anchor Hold (stop before H3)"
except Exception as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] Swap Zone Latent not loaded: {e}")

WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# SeamStitchVosrPolish: optional VOSR 2.0 sharpen step; calls the ComfyUI-VOSR2 pack's nodes at run time.
try:
    from .vosr_polish import SeamStitchVosrPolish
    NODE_CLASS_MAPPINGS["SeamStitchVosrPolish"] = SeamStitchVosrPolish
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchVosrPolish"] = "SeamStitch VOSR Polish"
except Exception as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] VOSR Polish not loaded: {e}")

# SeamStitch Frame Grab: a first and a last frame out of a video (or a connected batch) as lossless images.
try:
    from .frame_grab import SeamStitchFrameGrab
    NODE_CLASS_MAPPINGS["SeamStitchFrameGrab"] = SeamStitchFrameGrab
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchFrameGrab"] = "SeamStitch Frame Grab (first / last)"
except Exception as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] Frame Grab not loaded: {e}")
