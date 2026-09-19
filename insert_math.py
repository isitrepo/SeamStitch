"""Frame arithmetic for the Loader's 'insert at join' mode.

Pure Python on purpose (no torch / ComfyUI imports) so it can be unit-tested alone.

One anchor rule: anchors are always KEPT frames just outside the removed range
[join - N, join + N - 1] (empty when N = 0)."""

MODE_REPLACE = "replace range"
MODE_INSERT = "insert at join"


def resolve_join(join_frame, seam_frame):
    """Wired seam_frame wins over the typed join_frame. Returns (join, note) where
    note is a log line when the two disagree, else None."""
    if seam_frame is None:
        return int(join_frame), None
    seam_frame = int(seam_frame)
    if int(join_frame) != seam_frame:
        return seam_frame, (f"seam_frame ({seam_frame}) is wired and differs from join_frame "
                            f"({int(join_frame)}) - using the wired seam_frame")
    return seam_frame, None


def plan_insert(join, trim_each_side, clip_frames):
    """Anchor/range arithmetic. clip_frames is the total number of frames at the
    working frame rate. Raises ValueError naming the clip length when the range
    (plus its two anchors) does not fit."""
    n = int(trim_each_side)
    join = int(join)
    if n < 0:
        raise ValueError(f"trim_each_side must be >= 0 (got {n})")
    last_idx = clip_frames - 1
    first_anchor = join - n - 1
    last_anchor = join + n
    if first_anchor < 0 or last_anchor > last_idx:
        raise ValueError(
            f"insert mode: join_frame {join} with trim_each_side {n} needs frames "
            f"{first_anchor}..{last_anchor}, but the clip has {clip_frames} frames "
            f"(0..{last_idx}). Need join_frame - trim_each_side - 1 >= 0 and "
            f"join_frame + trim_each_side <= {last_idx}.")
    return {
        "first_anchor": first_anchor,
        "last_anchor": last_anchor,
        "start_frame": join - n,
        "end_frame": join + n - 1,
    }


def snap_8n1(frames):
    """Nearest 8n+1 frame count (LTX grid), floored at 9. Ties go up."""
    n = max(9, int(round(frames)))
    lo = ((n - 1) // 8) * 8 + 1
    hi = lo + 8
    return lo if (n - lo) < (hi - n) else hi


def bridge_length(duration_sec, duration_frames, display_mode, frame_rate):
    """Editable bridge length -> (frame_count, duration_sec) on the 8n+1 grid."""
    fr = float(frame_rate) if frame_rate > 0 else 24.0
    raw = duration_frames if display_mode == "frames" else duration_sec * fr
    n = snap_8n1(raw)
    return n, n / fr
