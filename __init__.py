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
    from .swap_take import SeamStitchSwapTake
    from .swap_mask import SeamStitchSwapMask
    NODE_CLASS_MAPPINGS["SeamStitchSwapMask"] = SeamStitchSwapMask
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapMask"] = "SeamStitch Swap Mask"
    NODE_CLASS_MAPPINGS["SeamStitchSwapPlanner"] = SeamStitchSwapPlanner
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapPlanner"] = "SeamStitch Swap Planner"
    NODE_CLASS_MAPPINGS["SeamStitchSwapOption"] = SeamStitchSwapOption
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapOption"] = "SeamStitch Swap Option"
    NODE_CLASS_MAPPINGS["SeamStitchSwapTake"] = SeamStitchSwapTake
    NODE_DISPLAY_NAME_MAPPINGS["SeamStitchSwapTake"] = "SeamStitch Swap Take"
except Exception as e:
    logging.getLogger(__name__).warning(f"[SeamStitch] Swap Planner / Option / Take / Mask not loaded: {e}")

WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
