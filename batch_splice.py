"""Batch splice: several seams of one cut, each bridged on its own, joined into one video.

Result Preview writes a small record beside every result it saves (<result>.seam.json):
which cut the splice was made on, the frames it replaced and how many it put there. A batch
is a set of such results that all came from the same cut with different seams as the target
(the Timeline's "fix all open joins" queues them). This node takes the cut and the folder
and lays the bridges back into the CUT, in one go:

  picture  the cut's own frames, with each seam's range swapped for that seam's bridge
           (the frames of its result between the two joins), so the footage between the
           seams is the cut's own pixels, never a copy of a copy;
  sound    every result carries the whole track, so each seam's span is taken from its own
           result with a short margin either side (the crossfade Recombine already put at
           each join) and the stretches between seams from the nearest result.

Seams may not overlap; each result must be the one the record names (size-checked).
"""

import json
import os

import numpy as np
import torch

try:
    from . import timeline as tl
    from .audio_splice import frame_sample
except ImportError:
    import timeline as tl
    from audio_splice import frame_sample

SEAM_SUFFIX = ".seam.json"
MARGIN_S = 0.1   # audio taken from a seam's own result either side of its span (>= any join crossfade)


class BatchError(ValueError):
    pass


# ---------------------------------------------------------------------------
# the record Result Preview leaves beside each result
# ---------------------------------------------------------------------------

def write_seam_record(result_path, cut_path, cut_fps, result_fps, start, end, analysis):
    """<result>.seam.json: enough to lay the result's bridge back into its cut later."""
    st, ct = os.stat(result_path), os.stat(cut_path)
    rec = {"version": 1,
           "result": os.path.abspath(result_path), "result_size": st.st_size,
           "cut": os.path.abspath(cut_path), "cut_size": ct.st_size,
           "cut_fps": int(cut_fps), "fps": int(result_fps),
           "start": int(start), "end": int(end),
           "removed": int(analysis["removed"]), "inserted": int(analysis["inserted"]),
           "frames": int(analysis["frames"]), "cut_frames": int(analysis["source_frames"]),
           "verdicts": [r.get("verdict") for r in analysis.get("joins", [])]}
    with open(result_path + SEAM_SUFFIX, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rec


def read_seam_records(folder):
    """Every seam record in `folder`, newest result first per seam range (a re-rolled seam
    leaves a second result; the latest one wins). Records whose result is gone or changed
    are dropped and named in the second return value."""
    if not os.path.isdir(folder):
        raise BatchError(f"batch folder not found: {folder}")
    recs, skipped = [], []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(SEAM_SUFFIX):
            continue
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                rec = json.load(f)
            if os.path.getsize(rec["result"]) != rec["result_size"]:
                raise ValueError("result changed since")
        except (OSError, ValueError, KeyError) as e:
            skipped.append(f"{name}: {e}")
            continue
        rec["mtime"] = os.path.getmtime(rec["result"])
        recs.append(rec)
    latest = {}
    for r in recs:
        k = (r["cut"], r["start"], r["end"])
        if k not in latest or r["mtime"] > latest[k]["mtime"]:
            latest[k] = r
    return sorted(latest.values(), key=lambda r: r["start"]), skipped


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def check_seams(seams, cut_path=None):
    """The seams must be one cut's, in one frame rate pair, and not overlap. Returns them
    sorted by start."""
    if not seams:
        raise BatchError("no seam results to splice - run the queued seams first")
    seams = sorted(seams, key=lambda r: r["start"])
    cut = {r["cut"] for r in seams}
    if len(cut) > 1:
        raise BatchError("the results were made on different cuts: " + ", ".join(sorted(os.path.basename(c) for c in cut)))
    if cut_path is not None and os.path.normcase(os.path.abspath(cut_path)) != os.path.normcase(next(iter(cut))):
        raise BatchError(f"these results were cut from {next(iter(cut))}, not {cut_path}")
    if len({(r["cut_fps"], r["fps"]) for r in seams}) > 1:
        raise BatchError("the results differ in frame rate")
    if len({r["cut_frames"] for r in seams}) > 1:
        raise BatchError("the results disagree on how long the cut is")
    prev_end = -1
    for r in seams:
        lo = r["start"]
        if lo <= prev_end:
            raise BatchError(f"seam at frame {lo} overlaps the one before it (which reaches frame {prev_end}) - "
                             f"bridge each join with a range that stays clear of its neighbours")
        prev_end = max(r["end"], r["start"] - 1)
    return seams


def plan_frames(seams):
    """Segments of the final video, in order: ("cut", a, b) frames [a, b) of the cut, or
    ("res", k, a, b) frames [a, b) of seam k's result. Also each seam's final start frame."""
    segs, pos, shift, starts = [], 0, 0, []
    for k, r in enumerate(seams):
        if r["start"] > pos:
            segs.append(("cut", pos, r["start"]))
        starts.append(r["start"] + shift)
        if r["inserted"] > 0:
            segs.append(("res", k, r["start"], r["start"] + r["inserted"]))
        pos = r["end"] + 1
        shift += r["inserted"] - r["removed"]
    n = seams[0]["cut_frames"]
    if pos < n:
        segs.append(("cut", pos, n))
    return segs, starts, n + shift


def iter_frames(seams, decode_cut, decode_result, size=None):
    """The final video, one segment of uint8 frames at a time: the cut's own stretches and the
    seams' bridges in order. decode_*(...) -> uint8 (n, h, w, 3). A long cut never exists in
    memory whole (a minute of 1080p is ~9 GB as floats). size=(h, w) checks every piece."""
    segs, _, total = plan_frames(seams)
    seen = 0
    for s in segs:
        part = decode_cut(s[1], s[2]) if s[0] == "cut" else decode_result(s[1], s[2], s[3])
        if part.shape[0] == 0:
            continue
        if size is not None and tuple(part.shape[1:3]) != tuple(size):
            who = "the cut" if s[0] == "cut" else f"the result of the seam at frame {seams[s[1]]['start']}"
            raise BatchError(f"{who} is {part.shape[2]}x{part.shape[1]}, the batch is {size[1]}x{size[0]}")
        seen += part.shape[0]
        yield part
    if seen != total:
        raise BatchError(f"assembled {seen} frames, expected {total} - a result's frames did not decode as recorded")


def splice_frames(seams, decode_cut, decode_result):
    """The final video as one uint8 tensor (small cuts and tests; the node streams instead)."""
    return torch.cat(list(iter_frames(seams, decode_cut, decode_result)), dim=0)


class LazyFrames:
    """Just enough of an IMAGE tensor for recombine._encode_video - shape, [0] and iteration - with
    each frame made (float 0-1) only when the encoder asks for it."""

    def __init__(self, shape, make):
        self.shape = tuple(shape)
        self._make = make

    def __getitem__(self, i):
        if i != 0:
            raise IndexError("LazyFrames only gives its first frame by index; iterate for the rest")
        for f in self:
            return f

    def __iter__(self):
        for part in self._make():
            for f in part:
                yield f.to(torch.float32).div_(255.0)


def splice_batch_audio(seams, waves, sample_rate, fps):
    """The final track. waves[k]: seam k's result track (channels, samples), laid out as its
    video (frame f at frame_sample(f)).

    The final video is stretch 0, seam 0's bridge, stretch 1, ... stretch n. A stretch is the
    cut's own sound and is read from the result of the seam before it (the first from seam 0's);
    a bridge is read from its own result, with a short margin either side so the crossfade
    Recombine made at each join comes with it. The margin is taken out of the stretch, never
    past the stretch's middle, so two results never overwrite each other's crossfade."""
    _, starts, total = plan_frames(seams)
    n = len(seams)
    fs = lambda f: frame_sample(f, fps, sample_rate)   # noqa: E731
    out = torch.zeros((waves[0].shape[0], fs(total)))
    bridges = [(starts[k], starts[k] + seams[k]["inserted"]) for k in range(n)]
    # stretch i, in final frames, and the result frame its first frame is in
    stretch = []
    for i in range(n + 1):
        a = bridges[i - 1][1] if i else 0
        b = bridges[i][0] if i < n else total
        src = 0
        if i:
            r = seams[i - 1]
            src = r["start"] + r["inserted"]
        stretch.append((a, b, src))
    margin = int(round(MARGIN_S * sample_rate))
    m = [min(margin, max(0, (fs(b) - fs(a)) // 2 - 1)) for a, b, _ in stretch]   # per stretch

    def put(k, dst_a, dst_b, offset):
        """result k's samples dst+offset into out[dst_a:dst_b] (silence where it has none)."""
        dst_a, dst_b = max(dst_a, 0), min(dst_b, out.shape[-1])
        if dst_b <= dst_a:
            return
        a, b = dst_a + offset, dst_b + offset
        lo, hi = max(a, 0), min(b, waves[k].shape[-1])
        if hi > lo:
            out[:, dst_a + (lo - a):dst_a + (hi - a)] = waves[k][:, lo:hi]

    for i, (a, b, src) in enumerate(stretch):
        k = max(i - 1, 0)
        left = m[i] if i else 0
        right = m[i] if i < n else 0
        put(k, fs(a) + left, fs(b) - right, fs(src) - fs(a))
    for k, (fa, fb) in enumerate(bridges):
        put(k, fs(fa) - m[k], fs(fb) + m[k + 1], fs(seams[k]["start"]) - fs(fa))
    return out


# ---------------------------------------------------------------------------
# decoding, and the node
# ---------------------------------------------------------------------------

def read_audio(path):
    """(waveform (channels, samples) float32, sample_rate) of a file's first audio stream, or None."""
    import av
    with av.open(path) as c:
        if not c.streams.audio:
            return None
        a = c.streams.audio[0]
        rate = int(a.codec_context.sample_rate)
        rs = av.AudioResampler(format="fltp", layout=a.layout.name, rate=rate)
        chunks = []
        for fr in c.decode(a):
            for o in rs.resample(fr):
                chunks.append(o.to_ndarray())
        for o in rs.resample(None):
            chunks.append(o.to_ndarray())
    if not chunks:
        return None
    return torch.from_numpy(np.concatenate(chunks, axis=1).astype(np.float32)), rate


def run_batch(cut_path, folder, decode_range, audio_reader=read_audio):
    """Everything but the saving: (frames uint8, audio dict or None, fps, info text).
    A blank cut_path is whatever cut the records name."""
    seams, skipped = read_seam_records(folder)
    seams = check_seams(seams, cut_path or None)
    cut_path = seams[0]["cut"]
    cut_fps, fps = seams[0]["cut_fps"], seams[0]["fps"]
    info = tl.probe(cut_path, cut_fps)
    size = (info["height"], info["width"])
    total = plan_frames(seams)[2]
    frames = LazyFrames((total,) + size + (3,), lambda: iter_frames(
        seams,
        lambda a, b: decode_range(cut_path, cut_fps, a, b),
        lambda k, a, b: decode_range(seams[k]["result"], fps, a, b), size))
    audio = None
    tracks = [audio_reader(r["result"]) for r in seams]
    if all(t is not None for t in tracks):
        rate = tracks[0][1]
        if any(t[1] != rate for t in tracks):
            raise BatchError("the results' audio differ in sample rate")
        wave = splice_batch_audio(seams, [t[0] for t in tracks], rate, fps)
        audio = {"waveform": wave.unsqueeze(0), "sample_rate": rate}
    lines = [f"{len(seams)} seam(s) laid into {os.path.basename(cut_path)}: " +
             ", ".join(f"{r['start']}..{r['end']} -> {r['inserted']}f" for r in seams)]
    lines += [f"skipped {s}" for s in skipped]
    return frames, audio, fps, "\n".join(lines)


class SeamStitchBatchSplice:
    """Lays every bridge of a batch back into the cut and saves one video. Queue it AFTER the
    batch's seams (ComfyUI runs the queue in order)."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            from .result_preview import _formats
        except ImportError:
            from result_preview import _formats
        return {"required": {
            "cut_path": ("STRING", {"default": "", "multiline": False, "tooltip":
                "The cut every seam was made on. Leave blank: the seams' records name it (set it only "
                "to refuse results made on any other cut)."}),
            "batch_folder": ("STRING", {"default": "", "multiline": False, "tooltip":
                "The folder the seams' Result Preview saved into (their filename_prefix, under output). "
                "Each result there has a .seam.json record beside it."}),
            "filename_prefix": ("STRING", {"default": "seamstitch_batch_%date:yyyyMMdd_hhmmss%"}),
            "format": (_formats(), {"default": "video/h264-mp4"}),
            "crf": ("INT", {"default": 12, "min": 0, "max": 51, "step": 1}),
            "pix_fmt": (["yuv420p", "yuv420p10le"], {"default": "yuv420p"}),
            "save_metadata": ("BOOLEAN", {"default": True}),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_path",)
    OUTPUT_NODE = True
    FUNCTION = "splice"
    CATEGORY = "SeamStitch"
    DESCRIPTION = ("Joins a batch of independently bridged seams into one video: the cut's own frames "
                   "between the seams, each seam's bridge from its own result.")

    @classmethod
    def IS_CHANGED(cls, cut_path="", batch_folder="", **kw):
        h = [cut_path, batch_folder, sorted(kw.items())]
        try:
            for n in sorted(os.listdir(batch_folder)):
                st = os.stat(os.path.join(batch_folder, n))
                h.append((n, st.st_mtime_ns, st.st_size))
        except OSError:
            pass
        import hashlib
        return hashlib.sha256(json.dumps(h, default=str).encode()).hexdigest()

    def splice(self, cut_path, batch_folder, filename_prefix, format, crf, pix_fmt, save_metadata,
               prompt=None, extra_pnginfo=None):
        try:
            from . import recombine as rc
            from .result_preview import _encode, format_settings
        except ImportError:
            import recombine as rc
            from result_preview import _encode, format_settings
        cut_path = tl.resolve_path(cut_path) if str(cut_path).strip() else ""
        folder = batch_folder if os.path.isabs(batch_folder) else os.path.join(
            __import__("folder_paths").get_output_directory(), batch_folder)
        frames, audio, fps, info = run_batch(cut_path, folder, rc._decode_range)
        print(f"[SeamStitch] Batch splice: {info}")
        images = frames   # lazy: encoded a frame at a time, never held whole
        settings = format_settings(format, crf, pix_fmt, save_metadata)
        filenames = _encode(images, audio, fps, filename_prefix, format, True, settings, prompt, extra_pnginfo)
        path = filenames[1][-1]
        print(f"[SeamStitch] Batch splice: saved {path}")
        return {"ui": {"text": [info]}, "result": (path,)}
