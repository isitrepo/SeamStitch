"""The YUV matrix and range to decode a stream with, as PyAV reformat() arguments.

One rule for every decoder in the pack (Loader, Recombine, Timeline). The stream's own tags
win; the size guess (BT.709 from 720 lines, BT.601 below) only fills in for a stream that
has none. PyAV hands the tags back as enum members, plain ints (PyAV 17: colorspace 1 for
BT.709) or strings depending on the version. An int used to fall through to the size guess,
so a tagged BT.709 clip under 720 wide (a 624x352 Swap assembly) decoded with the BT.601
matrix: about 1-3 levels darker in the midtones, and a re-encode carried that into the result.
"""

try:
    from av.video.reformatter import Colorspace, ColorRange
except ImportError:  # pragma: no cover - very old PyAV
    Colorspace = ColorRange = None

# AVColorSpace / AVColorRange numbers (libavutil/pixfmt.h) that mean "not tagged".
_CS_UNSPECIFIED = {0, 2}       # RGB (0) is handled by the caller; 2 = unspecified
_CR_UNSPECIFIED = {0}
# AVColorSpace numbers PyAV's Colorspace has no member for, mapped to the matrix swscale uses.
_CS_ALIAS = {6: 5, 9: 1, 10: 1}   # SMPTE 170M is BT.601; BT.2020 has no PyAV member, BT.709 is nearest


def _as_int(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    val = getattr(v, "value", None)
    if isinstance(val, int):
        return val
    return None


def _named(v):
    """A string tag ('itu709', 'bt709', 'unspecified'...) lower-cased, or None."""
    if isinstance(v, str):
        return v.lower()
    name = getattr(v, "name", None)
    return name.lower() if isinstance(name, str) else None


def stream_colorspace(c_space, w, h):
    guess_709 = max(int(w or 0), int(h or 0)) >= 720
    fallback = (Colorspace.ITU709 if guess_709 else Colorspace.ITU601) if Colorspace else \
        ("itu709" if guess_709 else "itu601")
    n = _as_int(c_space)
    if n is not None:
        if n in _CS_UNSPECIFIED:
            return fallback
        n = _CS_ALIAS.get(n, n)
        if Colorspace:
            try:
                return Colorspace(n)
            except ValueError:
                return fallback
        return {1: "itu709", 5: "itu601", 4: "fcc", 7: "smpte240m"}.get(n, fallback)
    s = _named(c_space)
    if s is None or "unspecified" in s:
        return fallback
    return c_space


def stream_color_range(c_range):
    fallback = ColorRange.MPEG if ColorRange else "mpeg"
    n = _as_int(c_range)
    if n is not None:
        if n in _CR_UNSPECIFIED:
            return fallback
        if ColorRange:
            try:
                return ColorRange(n)
            except ValueError:
                return fallback
        return {1: "mpeg", 2: "jpeg"}.get(n, fallback)
    s = _named(c_range)
    if s is None or "unspecified" in s:
        return fallback
    return c_range


def color_args(cc, w, h):
    """(src_colorspace, src_color_range, dst_color_range) for frame.reformat(format='rgb24', ...)."""
    c_space = getattr(cc, "colorspace", getattr(cc, "color_space", None)) if cc is not None else None
    c_range = getattr(cc, "color_range", None) if cc is not None else None
    dst = ColorRange.JPEG if ColorRange else "jpeg"
    return stream_colorspace(c_space, w, h), stream_color_range(c_range), dst


# ---------------------------------------------------------------------------
# rotation: a phone stores the sensor's picture as shot and tags the file with a display
# matrix ("rotate 180" when held upside down) instead of turning the pixels. Players and
# ffmpeg follow the tag; PyAV hands back the stored picture, so a phone clip decoded upside
# down or on its side. Every decoder in the pack turns its frames by the tag (ffmpeg's rule)
# and reports the turned size.
# ---------------------------------------------------------------------------

_ROTATION_CACHE = {}


def frame_turns(frame):
    """Counter-clockwise quarter turns (0-3) that put a decoded frame upright: its display
    matrix's rotation / 90 (PyAV's VideoFrame.rotation, counter-clockwise degrees)."""
    try:
        r = float(getattr(frame, "rotation", 0) or 0)
    except (TypeError, ValueError):
        return 0
    return int(round(r / 90.0)) % 4


def upright(rgb, turns):
    """An HxWxC array turned upright (np.rot90 is counter-clockwise, the same as the tag)."""
    if not turns:
        return rgb
    import numpy as np
    return np.ascontiguousarray(np.rot90(rgb, turns))


def file_turns(path):
    """The quarter turns of a file's first video stream, from its first decoded frame (the
    tag rides on every frame's side data; PyAV doesn't expose it on the stream). Cached."""
    import os
    import av
    try:
        st = os.stat(path)
    except OSError:
        return 0
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key not in _ROTATION_CACHE:
        turns = 0
        try:
            with av.open(path) as c:
                if c.streams.video:
                    for frame in c.decode(c.streams.video[0]):
                        turns = frame_turns(frame)
                        break
        except Exception:
            turns = 0
        if len(_ROTATION_CACHE) > 512:
            _ROTATION_CACHE.clear()
        _ROTATION_CACHE[key] = turns
    return _ROTATION_CACHE[key]


def display_size(w, h, turns):
    """(width, height) as shown: a quarter turn swaps them."""
    return (h, w) if turns % 2 else (w, h)
