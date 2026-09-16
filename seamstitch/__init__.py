from .simple_combine import SeamweaveSimpleCombine
from .load_video_ui import LoadVideoUIFirstLast

NODE_CLASS_MAPPINGS = {
    "SeamweaveSimpleCombine": SeamweaveSimpleCombine,
    "LoadVideoUIFirstLast": LoadVideoUIFirstLast,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeamweaveSimpleCombine": "Combine Clips (Simple)",
    "LoadVideoUIFirstLast": "Load Video UI (First/Last Frame)",
}

# VideoSegmentRecombine reuses ComfyUI-VideoHelperSuite's own ffmpeg/encode
# internals directly (see video_segment_recombine.py) and raises ImportError
# at import time if VHS isn't installed alongside this pack. Guarded here so
# a missing VHS install only drops this one node instead of taking the whole
# package down with it.
try:
    from .video_segment_recombine import VideoSegmentRecombine
    NODE_CLASS_MAPPINGS["VideoSegmentRecombine"] = VideoSegmentRecombine
    NODE_DISPLAY_NAME_MAPPINGS["VideoSegmentRecombine"] = "Video Combine (Segment Recombine)"
except ImportError as e:
    print(f"[SeamStitch] VideoSegmentRecombine not loaded: {e}")

WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
