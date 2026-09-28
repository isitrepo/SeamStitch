"""Sequence and splice-target arithmetic for SeamStitchTimeline.

Pure Python on purpose (no torch / ComfyUI imports) so it can be unit-tested alone,
and so js/timeline.js can mirror it line for line.

The timeline is one track. Its state is a plain text sequence, one entry per line:

    clips/a.mp4                    a whole clip
    clips/a.mp4 @ 12..119          frames 12..118 of it (enter inclusive, exit exclusive)
    clips/a.mp4 @ ..119            from its start to frame 118
    ~ 40                           a 40-frame gap
    # comment

Frame numbers are at the timeline's frame_rate, counted from the clip's own first
frame (decode order - the convention SeamStitchRecombine cuts on). The played ranges
of every clip, in order and without the gaps, make up the CUT: the one video the
node assembles and hands downstream as source_video_path. Cut frame numbers are
what Loader/Recombine see as start_frame/end_frame.

The splice target is one small JSON object (one splice at a time):

    {"mode": "replace", "start": S, "end": E}    regenerate cut frames S..E
    {"mode": "gap", "trim": N}                   bridge the timeline's one gap
                                                 (insert mode, N frames trimmed
                                                 either side of it)
    {}                                           nothing marked
"""

import json

GRID_LTX = "ltx (8k+1)"
GRID_MINIMAX = "minimax (17k+5)"
GRID_NONE = "none"


class SequenceError(ValueError):
    """The sequence text or the target cannot be turned into a cut."""


def parse_sequence(text):
    """Sequence text -> list of entries.

    Clip: {"kind": "clip", "path": str, "enter": int, "exit": int | None}
    Gap:  {"kind": "gap", "frames": int}

    A malformed line raises rather than being skipped: a skipped clip would drop
    content from the cut without anyone noticing."""
    entries = []
    for lineno, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("~"):
            try:
                n = int(line[1:].strip())
            except ValueError:
                raise SequenceError(f"line {lineno}: a gap is '~ N' with N a whole number of frames, got {raw!r}")
            if n < 1:
                raise SequenceError(f"line {lineno}: a gap needs at least 1 frame, got {n}")
            entries.append({"kind": "gap", "frames": n})
            continue
        path, enter, exit_ = line, 0, None
        at = line.rfind(" @ ")
        if at >= 0:
            path, rng = line[:at].strip(), line[at + 3:].strip()
            if ".." not in rng:
                raise SequenceError(f"line {lineno}: a range is '@ enter..exit', got {rng!r}")
            a, b = rng.split("..", 1)
            try:
                enter = int(a) if a.strip() else 0
                exit_ = int(b) if b.strip() else None
            except ValueError:
                raise SequenceError(f"line {lineno}: range numbers must be whole frames, got {rng!r}")
        if enter < 0 or (exit_ is not None and exit_ <= enter):
            raise SequenceError(f"line {lineno}: range {enter}..{exit_} is empty or negative")
        entries.append({"kind": "clip", "path": path, "enter": enter, "exit": exit_})
    return entries


def format_sequence(entries):
    """Inverse of parse_sequence, for writing back a normalised sequence."""
    lines = []
    for e in entries:
        if e["kind"] == "gap":
            lines.append(f"~ {int(e['frames'])}")
        elif e["enter"] or e["exit"] is not None:
            ex = "" if e["exit"] is None else str(int(e["exit"]))
            lines.append(f"{e['path']} @ {int(e['enter'])}..{ex}")
        else:
            lines.append(e["path"])
    return "\n".join(lines)


def resolve_cut(entries, frames_of):
    """Lay the entries out as a cut.

    frames_of(path) -> the clip's total frame count at the timeline frame rate.
    Returns {"pieces": [...], "gaps": [...], "frames": total cut frames} where each
    piece is {"path", "enter", "exit", "cut_start"} (exit resolved, clamped to the
    clip) and each gap is {"frames", "cut_pos", "index"}: cut_pos is the cut frame
    the gap sits in front of."""
    pieces, gaps, pos = [], [], 0
    for i, e in enumerate(entries):
        if e["kind"] == "gap":
            gaps.append({"frames": e["frames"], "cut_pos": pos, "index": i})
            continue
        total = int(frames_of(e["path"]))
        exit_ = total if e["exit"] is None else min(int(e["exit"]), total)
        if e["enter"] >= exit_:
            raise SequenceError(
                f"{e['path']}: enters at frame {e['enter']} but the clip only has {total} frames "
                f"at this frame rate (0..{total - 1})")
        pieces.append({"path": e["path"], "enter": e["enter"], "exit": exit_, "cut_start": pos})
        pos += exit_ - e["enter"]
    if not pieces:
        raise SequenceError("the timeline has no clips - drag a video onto it or use + add")
    return {"pieces": pieces, "gaps": gaps, "frames": pos}


def parse_target(text):
    if not text or not str(text).strip():
        return {}
    try:
        t = json.loads(text)
    except (TypeError, ValueError):
        raise SequenceError(f"splice target is not valid JSON: {text!r}")
    return t if isinstance(t, dict) else {}


def seam_positions(cut):
    """Cut frame of every join between two consecutive pieces (the first frame of
    the later piece). Gaps are not seams - they are bridge targets of their own."""
    return [p["cut_start"] for p in cut["pieces"][1:]]


def resolve_target(target, cut):
    """Splice target -> the Loader arguments that express it on the cut.

    Returns {"mode": "replace range"|"insert at join", "start": S, "end": E,
    "join": J, "trim": N, "length": L} - start/end inclusive cut frames of the
    range that is replaced (insert: removed, end = start - 1 when nothing is), join
    and length only meaningful in insert mode."""
    mode = target.get("mode")
    n_frames = cut["frames"]
    if len(cut["gaps"]) > 1:
        raise SequenceError(
            f"the timeline has {len(cut['gaps'])} gaps; SeamStitch does one splice at a time - "
            f"close all but the one you are bridging")
    if mode == "replace":
        if cut["gaps"]:
            raise SequenceError("the timeline has a gap but the target is a replace range - "
                                "close the gap, or bridge it instead (its seam menu)")
        s, e = int(target.get("start", -1)), int(target.get("end", -1))
        if s < 0 or e < s or e > n_frames - 1:
            raise SequenceError(f"replace range {s}..{e} is outside the cut (0..{n_frames - 1})")
        if s == 0 or e == n_frames - 1:
            raise SequenceError(
                f"replace range {s}..{e} touches the end of the cut (0..{n_frames - 1}); a bridge "
                f"needs a real frame either side to land on")
        return {"mode": "replace range", "start": s, "end": e, "join": 0, "trim": 0, "length": e - s + 1}
    if mode == "gap":
        if not cut["gaps"]:
            raise SequenceError("the target is a gap bridge but the timeline has no gap - open one "
                                "with a seam's menu or ctrl+drag on the strip")
        g = cut["gaps"][0]
        join, trim = g["cut_pos"], max(0, int(target.get("trim", 0)))
        if join - trim - 1 < 0 or join + trim > n_frames - 1:
            raise SequenceError(
                "a bridged gap needs real footage on both sides (and trim_each_side frames more "
                "on each) - move it between two clips")
        # The gap on the strip is the number of NEW frames. The generator also renders
        # the two kept frames either side (Recombine drops them again) and re-renders
        # the trimmed ones, so it is asked for gap + 2*trim + 2.
        return {"mode": "insert at join", "start": join - trim, "end": join + trim - 1,
                "join": join, "trim": trim, "length": g["frames"] + 2 * trim + 2}
    raise SequenceError("nothing is marked to regenerate - mark a range (I / O), click a seam and "
                        "pick 'bridge this cut', or open a gap")


def snap_up(n, grid):
    """Round a frame count UP onto the generator's grid (the Loader's
    extend_bridge rule, via SeamStitchLoader._extended_frame_count)."""
    n = max(1, int(round(n)))
    if grid == GRID_NONE:
        return n
    if grid == GRID_MINIMAX:
        n = max(5, n)
        r = (n - 5) % 17
        return n if r == 0 else n + 17 - r
    n = max(9, n)
    r = (n - 1) % 8
    return n if r == 0 else n + 8 - r


def snap_nearest(n, grid):
    """Nearest length on the grid, ties up (insert_math.snap_to_grid)."""
    if grid == GRID_NONE:
        return max(3, int(round(n)))
    if grid == GRID_MINIMAX:
        n = max(5, int(round(n)))
        lo = ((n - 5) // 17) * 17 + 5
    else:
        n = max(9, int(round(n)))
        lo = ((n - 1) // 8) * 8 + 1
    step = 17 if grid == GRID_MINIMAX else 8
    hi = lo + step
    return lo if (n - lo) < (hi - n) else hi


def generator_frames(plan, grid, context_frames=0, extend_frames=0):
    """How many frames the generator will be asked for - exactly what the Loader's
    frame_count output will say for the same plan. Shown in the node's next-run bar."""
    k = int(context_frames or 0)
    if plan["mode"] == "insert at join":
        return snap_nearest(plan["length"], grid)
    gap = plan["end"] - plan["start"] + 1
    return snap_up(gap + 2 * k + int(extend_frames or 0), grid)


def validate_context(plan, cut, context_frames):
    k = int(context_frames or 0)
    if k <= 0:
        return
    if plan["mode"] != "replace range":
        raise SequenceError("context_frames is replace mode only - a gap bridge already pins one "
                            "kept frame either side")
    if plan["start"] - k < 0 or plan["end"] + k > cut["frames"] - 1:
        raise SequenceError(
            f"context_frames {k} needs {k} real frames either side of {plan['start']}..{plan['end']} "
            f"but the cut is only 0..{cut['frames'] - 1} - lower context_frames or move the range")
