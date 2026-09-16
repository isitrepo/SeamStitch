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

WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
