"""SeamStitchResultPreview: the spliced result, shown around its splice, with a verdict.

Wire Recombine's Filenames in, plus the Timeline's (or Loader's) source_video_path /
start_frame / end_frame / frame_rate. After the run the node plays the result with
the regenerated span marked, and rates each of its two joins:

    ratio = the picture change across the join / the median change around it

on small grey thumbnails (codec noise averages out). ~1 means the join moves like
the footage around it; a hard cut reads many times higher. The same measurement
is taken on the ORIGINAL cut over the replaced range, so the node can say "before:
hard cut 9.8x -> after: seamless 1.1x".

"use as timeline" in the node's UI puts the result back onto the Timeline node as
its only clip, so the next splice starts from this one - one splice at a time,
as many times as it takes.

Thresholds (pixel domain): < 1.6 seamless, < 2.6 soft bump, else hard cut. They sit
a little above obvpm-timeline's latent-domain 1.2 / 1.8 because a thumbnail diff
of ordinary motion is noisier than a latent step; calibrated on the pack's real
test footage (see tests/test_result_preview.py and docs/CHANGELOG.md).
"""

import os

import av
import numpy as np
import folder_paths

try:
    from . import timeline as tl
except ImportError:
    import timeline as tl

SEAMLESS, SOFT = 1.6, 2.6
WINDOW = 24            # frames either side of a join used for the baseline
THUMB_W = 96


def verdict(ratio):
    if ratio is None:
        return "n/a"
    return "seamless" if ratio < SEAMLESS else ("soft bump" if ratio < SOFT else "hard cut")


def _thumbs(path, fr, start, end):
    """Grey float thumbnails of frames start..end-1 (clamped to the clip)."""
    start = max(0, start)
    out = []
    for f in tl._iter_frames(path, fr, start, end):
        h, w = f.shape[:2]
        step = max(1, w // THUMB_W)
        g = f[::step, ::step].astype(np.float32)
        out.append(g @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    return start, out


def join_ratio(path, fr, join, window=WINDOW):
    """Ratio of the change INTO frame `join` to the median change of the frames
    around it. None when there is not enough footage either side."""
    s, th = _thumbs(path, fr, join - window, join + window)
    k = join - s
    if k < 1 or k >= len(th):
        return None, None
    d = np.array([np.abs(th[i] - th[i - 1]).mean() for i in range(1, len(th))])
    step = float(d[k - 1])
    rest = np.delete(d, k - 1)
    base = float(np.median(rest)) if rest.size else 0.0
    return step / max(base, 0.5), step


def worst_in_range(path, fr, lo, hi, window=WINDOW):
    """Worst join ratio over frames lo..hi (the original cut's replaced span): where
    the thing being fixed was."""
    s, th = _thumbs(path, fr, lo - window, hi + window + 1)
    if len(th) < 3:
        return None, None
    d = np.array([np.abs(th[i] - th[i - 1]).mean() for i in range(1, len(th))])
    # d[i-1] is the change into absolute frame s + i
    best = (None, None)
    for j in range(max(lo, s + 1), min(hi + 1, s + len(th))):
        k = j - s
        rest = np.delete(d, k - 1)
        r = float(d[k - 1]) / max(float(np.median(rest)), 0.5)
        if best[0] is None or r > best[0]:
            best = (r, j)
    return best


def _result_path(filenames):
    files = filenames[1] if isinstance(filenames, (list, tuple)) and len(filenames) > 1 else filenames
    if isinstance(files, str):
        files = [files]
    vids = [f for f in files or [] if str(f).lower().endswith(tl._VIDEO_EXTENSIONS)]
    if not vids:
        raise ValueError("Filenames holds no video - wire SeamStitch Recombine's Filenames output")
    with_audio = [f for f in vids if os.path.splitext(f)[0].endswith("-audio")]
    return (with_audio or vids)[-1]


def analyse(result, source, fr, start, end):
    """Everything the UI shows. start/end: the replaced range on the source (end =
    start - 1 when nothing was removed)."""
    r_frames = tl.probe(result, fr)["frames"]
    s_frames = tl.probe(source, fr)["frames"]
    removed = max(0, end - start + 1)
    inserted = r_frames - (s_frames - removed)
    joins = [start, start + inserted]
    rows = []
    for name, j in (("into the new frames", joins[0]), ("back to the footage", joins[1])):
        ratio, step = join_ratio(result, fr, j) if 0 < j < r_frames else (None, None)
        rows.append({"name": name, "frame": j, "ratio": ratio, "verdict": verdict(ratio)})
    before, where = (worst_in_range(source, fr, max(1, start), min(s_frames - 1, max(start, end)))
                     if s_frames > 2 else (None, None))
    return {"frames": r_frames, "source_frames": s_frames, "inserted": inserted,
            "removed": removed, "joins": rows,
            "before": {"ratio": before, "frame": where, "verdict": verdict(before)}}


def _view_params(path):
    """/view query for a file under output or input, else None (served through
    the Loader's own file route instead)."""
    for kind, base in (("output", folder_paths.get_output_directory()),
                       ("input", folder_paths.get_input_directory())):
        base = os.path.abspath(base)
        ap = os.path.abspath(path)
        if ap.lower().startswith(base.lower() + os.sep):
            rel = os.path.relpath(ap, base)
            sub, name = os.path.split(rel)
            return {"filename": name, "subfolder": sub.replace("\\", "/"), "type": kind}
    return None


class SeamStitchResultPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filenames": ("VHS_FILENAMES", {"tooltip": "SeamStitch Recombine's Filenames output."}),
                "source_video_path": ("STRING", {"forceInput": True, "tooltip":
                    "The video the splice was cut from - the Timeline's or Loader's source_video_path."}),
                "start_frame": ("INT", {"forceInput": True}),
                "end_frame": ("INT", {"forceInput": True}),
                "frame_rate": ("INT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_path",)
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "SeamStitch"
    DESCRIPTION = ("Plays the spliced result with the regenerated span marked, rates both joins "
                   "(seamless / soft bump / hard cut) against the original, and can put the result "
                   "back on the Timeline for the next splice.")

    def preview(self, filenames, source_video_path, start_frame, end_frame, frame_rate):
        result = _result_path(filenames)
        fr = int(frame_rate) or 24
        a = analyse(result, source_video_path, fr, int(start_frame), int(end_frame))
        for row in a["joins"]:
            r = "n/a" if row["ratio"] is None else f"{row['ratio']:.2f}x"
            print(f"[SeamStitch] Result: join {row['name']} at frame {row['frame']}: {r} {row['verdict']}")
        b = a["before"]
        if b["ratio"] is not None:
            print(f"[SeamStitch] Result: original worst join in the replaced range: {b['ratio']:.2f}x "
                  f"{b['verdict']} at frame {b['frame']}")
        ui = dict(a, path=result, frame_rate=fr, start=int(start_frame), view=_view_params(result))
        return {"ui": {"seamstitch_result": [ui]}, "result": (result,)}
