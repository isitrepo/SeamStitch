"""SeamStitch Swap: the job's plan file, its chunk geometry and its joins.

Pure Python (no torch, no comfy, no av), so every rule here is unit-tested on its own.
The Swap nodes (Planner, Take, Assemble) all read and write one `plan.json` per job:

    <output>/seamstitch_swap/<job>/plan.json        the source of truth
    <output>/seamstitch_swap/<job>/history/         the last 50 revisions

A long source video is delivered as chunks between SPLITS. A split at frame J is where
delivery switches from the left chunk to the right one, and it is either:
  * "cut" (a straight cut): the right chunk renders from J, independently;
  * "anchored": the right chunk renders from J - overlap, its first `anchors` frames pinned
    to the left chunk's take, and the picture is spliced at J (the overlap's end).

MiniMax H3 renders 17k+5 frames, so a chunk's render length snaps UP to that grid and the
surplus is filled: past its end (tail) when the next split is anchored and no cut lies in
the extension; before its start (head) when its own split is anchored; else the guide's
last frame is held (hold) and the held frames are dropped again.

Each take records which take its pins came from and the splits it was rendered for. That
lineage decides each join's type (straight, forward, re-roll entry, re-roll exit, stale,
pending), and the type decides the splice point and the default repair (swap_join.py).

KEEP ORIGINAL (design §4.10, B3b). A chunk with `keep: true` is never rendered, marked or
drafted: it has no render range, length or fill, and the assembly delivers the source frames
there untouched. Its joins:
  * straight split: a plain cut between the source and the render (on a source cut it's clean;
    mid-shot it warns, since H3 redraws the whole room);
  * anchored onto the original, kept on the LEFT: the render starts `overlap` frames early as
    usual, its first `anchors` frames pinned to the SOURCE, and the join is the re-roll entry's
    fade from the source into the render over the overlap (splice at J);
  * anchored onto the original, kept on the RIGHT (the mirror): the render runs `overlap` frames
    past J into the kept stretch, its last `anchors` frames pinned to the source, and it fades
    out into the untouched source over [J, J + overlap - 1] (splice at J + overlap).
A take pinned to the source records `{"source": true, "frames": [a, b]}` for that side.
"""

import copy
import datetime
import json
import os
import threading
import time

FORMAT = "seamstitch_swap_plan_v1"
HISTORY_KEEP = 50
JOBS_SUBDIR = "seamstitch_swap"

MODE_CUT = "cut"
MODE_ANCHORED = "anchored"

# H3's frame grid (comfy_extras/nodes_minimax_h3.py: length snaps up, a ref video is trimmed
# down, to 17k+5).
GRID_STEP, GRID_OFFSET = 17, 5

DEFAULT_SETTINGS = {
    "overlap": 12,            # anchored overlap (0.5 s at 25 fps)
    "anchors": 5,             # pinned frames per side (5 or 22)
    "target_render": 209,     # auto-placement render length, on 17k+5
    "floor": 124,             # warning below (H3's trained minimum; 73 broke in CR-T1)
    "ceiling": 209,           # warning above: two-pass renders over 209 aren't safe on 64 GB RAM (B5a: 243 crashed, 260
                              # failed; B5b: 226 failed, then crashed ComfyUI; 209 ran 30+ times)
    "trained_max": 362,       # warning above (H3's trained range)
    "guard": 6,               # overlap guard either side of an anchored overlap (half T2's widest morph)
    "conform_to_24fps": True,
    "hand_back": 12,          # level lock / fade decay length (picked by eye over 25)
}

# join types
STRAIGHT, FORWARD, ENTRY, EXIT, STALE, PENDING = "straight", "forward", "entry", "exit", "stale", "pending"
ORIGINAL = "original"          # both sides kept: the source either side, nothing to join
SOURCE_TAKE_ID = "source"      # the "take" a kept chunk delivers: the source itself
# repairs
REPAIR_NONE, REPAIR_LOCK, REPAIR_FADE, REPAIR_CUT = "none", "lock", "fade", "cut"
DEFAULT_REPAIR = {STRAIGHT: REPAIR_NONE, FORWARD: REPAIR_LOCK, ENTRY: REPAIR_FADE, EXIT: REPAIR_LOCK,
                  STALE: REPAIR_LOCK, PENDING: REPAIR_NONE}


class PlanError(Exception):
    pass


class PlanConflict(PlanError):
    """The plan moved on since the caller read it (another writer saved first)."""


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def snap_up(n):
    """The smallest H3 length (17k+5, k >= 0) that holds n frames."""
    n = int(n)
    if n <= GRID_OFFSET:
        return GRID_OFFSET
    return GRID_OFFSET + GRID_STEP * -(-(n - GRID_OFFSET) // GRID_STEP)


def _settings(settings):
    s = dict(DEFAULT_SETTINGS)
    s.update(settings or {})
    return s


SUGGEST_NEAR = 3       # a suggested cut this close to a straight split is worth a look


def confirmed_cuts(cuts):
    """Frame numbers of the confirmed cuts (a cut at c: frame c is the first of a new shot).
    Accepts plan entries ({"frame", "confirmed"}) or bare ints (taken as confirmed)."""
    out = []
    for c in cuts or []:
        if isinstance(c, dict):
            if c.get("confirmed", True):
                out.append(int(c["frame"]))
        else:
            out.append(int(c))
    return sorted(set(out))


def suggested_cuts(cuts):
    """Frame numbers of the unconfirmed (suggested) cuts."""
    return sorted({int(c["frame"]) for c in cuts or [] if isinstance(c, dict) and c.get("confirmed", True) is False})


def _norm_splits(splits):
    out = []
    for i, s in enumerate(splits or []):
        if isinstance(s, dict):
            d = dict(s)
            d["frame"] = int(d["frame"])
            d.setdefault("mode", MODE_ANCHORED)
            d.setdefault("id", f"s{i + 1}")
        else:
            d = {"id": f"s{i + 1}", "frame": int(s), "mode": MODE_ANCHORED}
        if d["mode"] not in (MODE_CUT, MODE_ANCHORED):
            raise PlanError(f"split {d['id']}: mode must be '{MODE_CUT}' or '{MODE_ANCHORED}', not {d['mode']!r}")
        out.append(d)
    out.sort(key=lambda d: d["frame"])
    return out


def source_take(frames):
    """The pseudo-take a kept chunk delivers: the source's own frames, all of them."""
    return {"id": SOURCE_TAKE_ID, "render": [0, int(frames) - 1], "source": True, "state": "ok"}


def is_kept(chunk):
    return bool(chunk and chunk.get("keep"))


def chunk_target(plan, chunk):
    """Who the chunk replaces, as SAM3's source text and the drafter's main person: the chunk's own
    target, else the job's; "" = SAM3's "person" and the drafter's own pick."""
    return ((chunk.get("options") or {}).get("target") or plan.get("target") or "").strip()


def chunk_invert(plan, chunk):
    """Mark everything but the target (a background swap) instead of the target: the chunk's own
    setting, else the job's."""
    v = (chunk.get("options") or {}).get("invert")
    return bool(plan.get("invert")) if v is None else bool(v)


def geometry(frames, splits, cuts=(), settings=None, keep=()):
    """Chunks for a source of `frames` frames split at `splits`. `keep`: the left-split ids
    (None for the first chunk) of the chunks kept as the original (§4.10).

    Returns a list of dicts, one per chunk, in order:
      deliver [a, b]   source frames this chunk delivers (inclusive)
      render  [r0, r1] source frames it renders (overlap and tail/head fill included)
      held             guide frames held after r1 (hold fill), dropped from the take
      length           the render length H3 sees: r1 - r0 + 1 + held (always 17k+5)
      base             the unfilled render length
      fill             None or {"kind": "tail"|"head"|"hold", "frames": n}
      left / right     the bounding split dicts (None at the ends of the video)
      keep             True for a kept chunk: render None, held / length / base 0, fill None
    A rendered chunk whose anchored split borders a kept chunk on its RIGHT renders `overlap`
    frames past that split (its fade out into the original); on its left nothing changes.
    """
    s = _settings(settings)
    n_src = int(frames)
    sp = _norm_splits(splits)
    for d in sp:
        if not 0 < d["frame"] < n_src:
            raise PlanError(f"split {d['id']} at {d['frame']} is outside the video (1..{n_src - 1})")
    for a, b in zip(sp, sp[1:]):
        if a["frame"] == b["frame"]:
            raise PlanError(f"splits {a['id']} and {b['id']} are both at frame {a['frame']}")
    cut_frames = confirmed_cuts(cuts)
    ov = int(s["overlap"])
    bounds = [None] + sp + [None]
    keep = set(keep or ())
    kept = [(bounds[k]["id"] if bounds[k] else None) in keep for k in range(len(sp) + 1)]
    chunks = []
    for k in range(len(sp) + 1):
        left, right = bounds[k], bounds[k + 1]
        a = left["frame"] if left else 0
        b = right["frame"] - 1 if right else n_src - 1
        if kept[k]:
            chunks.append({"deliver": [a, b], "render": None, "held": 0, "length": 0, "base": 0, "fill": None,
                           "left": left, "right": right, "keep": True})
            continue
        r0 = max(0, a - ov) if left and left["mode"] == MODE_ANCHORED else a
        r1 = b
        if right and right["mode"] == MODE_ANCHORED and k + 1 < len(kept) and kept[k + 1]:
            r1 = min(n_src - 1, b + ov)       # the overlap runs into the kept stretch (fade out)
        base = r1 - r0 + 1
        length = snap_up(base)
        n = length - base
        fill, held = None, 0
        if n:
            tail_ok = (right is not None and right["mode"] == MODE_ANCHORED and r1 + n <= n_src - 1
                       and not any(r1 + 1 <= c <= r1 + n for c in cut_frames))
            head_ok = left is not None and left["mode"] == MODE_ANCHORED and r0 - n >= 0
            if tail_ok:
                r1 += n
                fill = {"kind": "tail", "frames": n}
            elif head_ok:
                r0 -= n
                fill = {"kind": "head", "frames": n}
            else:
                held = n
                fill = {"kind": "hold", "frames": n}
        chunks.append({"deliver": [a, b], "render": [r0, r1], "held": held, "length": length, "base": base,
                       "fill": fill, "left": left, "right": right, "keep": False})
    return chunks


def warnings(frames, splits, cuts=(), settings=None, chunks=None, keep=()):
    """Every §4.1 warning (none of them blocks). Each: {code, text, split|chunk, ...}. Kept chunks
    (§4.10) have no length warnings; a split between two kept chunks has none at all."""
    s = _settings(settings)
    cut_frames = confirmed_cuts(cuts)
    sugg = suggested_cuts(cuts)
    chunks = chunks if chunks is not None else geometry(frames, splits, cuts, s, keep)
    out = []
    for i, c in enumerate(chunks):
        if c.get("keep"):
            continue
        L = c["length"]
        where = {"chunk": i, "deliver": c["deliver"]}
        if L < s["floor"]:
            out.append(dict(where, code="floor", text=f"render {L} frames is under the floor ({s['floor']})"))
        if L > s["trained_max"]:
            out.append(dict(where, code="trained", text=f"render {L} frames is over H3's trained range ({s['trained_max']})"))
        elif L > s["ceiling"]:
            out.append(dict(where, code="ceiling", text=f"render {L} frames is over the tested ceiling ({s['ceiling']}): two-pass renders failed or crashed ComfyUI at 226 (B5b), 243 and 260 frames (B5a), at the 1 MP refine (system RAM)"))
    for i, c in enumerate(chunks[1:], start=1):
        sp = c["left"]
        J = sp["frame"]
        lk, rk = bool(chunks[i - 1].get("keep")), bool(c.get("keep"))
        if lk and rk:
            continue                                  # the source either side: nothing to join
        if sp["mode"] == MODE_ANCHORED:
            if rk:                                    # the overlap runs into the kept stretch on the right
                lo, hi = J - s["guard"], chunks[i - 1]["render"][1] + s["guard"]
            else:
                lo, hi = c["render"][0] - s["guard"], J + s["guard"]
            inside = [x for x in cut_frames if lo <= x <= hi]
            if inside:
                out.append({"split": sp.get("id"), "frame": J, "code": "guard", "cuts": inside, "guard": [lo, hi],
                            "text": (f"anchored split at {J}: cut at {', '.join(map(str, inside))} inside its overlap "
                                     f"guard [{lo}, {hi}]. Make it a straight cut on that cut, or move it clear."),
                            "remedies": [{"action": "straight_cut", "frame": inside[0]},
                                         {"action": "move", "clear": clearance(frames, splits, cuts, s, sp.get("id"),
                                                                               keep)}]})
        else:
            if not any(abs(x - J) <= 1 for x in cut_frames):
                if lk or rk:
                    side = "left" if lk else "right"
                    out.append({"split": sp.get("id"), "frame": J, "code": "no_cut", "kept": side,
                                "text": (f"straight split at {J} between the original ({side}) and a render, mid-shot: "
                                         f"the room changes here (H3 redraws the whole frame). Put it on a cut, or "
                                         f"anchor it onto the original."),
                                "remedies": [{"action": "anchor_onto_original"}]
                                + ([{"action": "move_to_cut", "frame": min(cut_frames, key=lambda x: abs(x - J))}]
                                   if cut_frames else [])})
                else:
                    out.append({"split": sp.get("id"), "frame": J, "code": "no_cut",
                                "text": f"straight split at {J}: visible jump, the source doesn't cut here"})
        # an unconfirmed suggestion where it matters: inside an anchored overlap's guard, or a few frames off a
        # straight split (B5a: an unconfirmed jump cut at 713 was copied 3 frames late). Confirm or delete it by eye.
        if sp["mode"] == MODE_ANCHORED:
            near = [x for x in sugg if lo <= x <= hi]
            where = f"inside the overlap guard [{lo}, {hi}] of the anchored split at {J}"
        else:
            near = [x for x in sugg if 0 < abs(x - J) <= SUGGEST_NEAR] if not any(abs(x - J) <= 1 for x in cut_frames) else []
            where = f"{SUGGEST_NEAR} frames or less from the straight split at {J}"
        if near:
            out.append({"split": sp.get("id"), "frame": J, "code": "suggested_cut", "cuts": near,
                        "text": (f"suggested cut at {', '.join(map(str, near))} (not confirmed) {where}: check it by eye; "
                                 f"confirm it (the split then warns or snaps) or delete it")})
    return out


def _guard_clear(chunks, k, cut_frames, guard):
    """Is the anchored split on the left of chunk k clear of every cut (after the fill)?"""
    c = chunks[k]
    sp = c["left"]
    if sp is None or sp["mode"] != MODE_ANCHORED or (c.get("keep") and chunks[k - 1].get("keep")):
        return True
    if c.get("keep"):
        lo, hi = sp["frame"] - guard, chunks[k - 1]["render"][1] + guard
    else:
        lo, hi = c["render"][0] - guard, sp["frame"] + guard
    return not any(lo <= x <= hi for x in cut_frames)


def clearance(frames, splits, cuts, settings, split_id, keep=()):
    """The nearest frames, forward and backward, an anchored split could move to so that its
    overlap guard holds no cut (after the fill), the other splits unchanged. For the strip."""
    s = _settings(settings)
    sp = _norm_splits(splits)
    idx = next((i for i, d in enumerate(sp) if d.get("id") == split_id), None)
    if idx is None:
        return {}
    cut_frames = confirmed_cuts(cuts)
    lo_lim = sp[idx - 1]["frame"] + 1 if idx > 0 else 1
    hi_lim = sp[idx + 1]["frame"] - 1 if idx + 1 < len(sp) else int(frames) - 1
    found = {}
    for name, rng in (("forward", range(sp[idx]["frame"] + 1, hi_lim + 1)),
                      ("backward", range(sp[idx]["frame"] - 1, lo_lim - 1, -1))):
        for J in rng:
            trial = [dict(d) for d in sp]
            trial[idx]["frame"] = J
            try:
                ch = geometry(frames, trial, cut_frames, s, keep)
            except PlanError:
                continue
            if _guard_clear(ch, idx + 1, cut_frames, s["guard"]):
                found[name] = J
                break
    return found


def auto_splits(frames, cuts=(), settings=None, max_shift=None, with_modes=False):
    """Split frames for target-length renders: the first at target_render, then every
    target_render - overlap (anchored). A last chunk whose render would fall under the floor merges
    into the one before, unless that passes the ceiling: then the last split moves back until the last
    render is the floor. An anchored split whose overlap guard holds a cut (checked AFTER the fill,
    since a head fill lengthens the overlap) is nudged forward, else backward, by the smallest
    shift that clears it without fouling an earlier split's guard; the splits after it are
    re-placed from its new position. If that nudge leaves a render over the ceiling and a straight
    cut on the cut in the guard doesn't (B5b: 100d's nudge to 815 made a 226-frame render, which
    failed and then crashed ComfyUI), the split becomes a straight cut there instead, and the
    splits after it are placed from the cut (its chunk renders from the cut: no overlap). If the
    straight cut passes the ceiling too, the backward nudge is taken (B7: a sitcom's shots 38 frames
    apart, where neither cleared it). Returns the frames; with_modes=True, [(frame, mode)]."""
    s = _settings(settings)
    n_src = int(frames)
    cut_frames = confirmed_cuts(cuts)
    target, ov, guard = int(s["target_render"]), int(s["overlap"]), int(s["guard"])
    step = target - ov
    max_shift = int(max_shift if max_shift is not None else step // 2)
    straight = set()                    # split frames made straight cuts

    def chain(prefix, J, cut=False):
        prefix_cut = J
        out = list(prefix)
        if J < n_src:
            out.append(J)
            J += target if cut else step
        while J < n_src:
            out.append(J)
            J += step
        # merge an under-floor last chunk into the one before, or move it back to the floor
        if out and len(out) > len(prefix) + (1 if cut else 0):
            last = out[-1]
            if snap_up(n_src - (last - ov)) < s["floor"]:
                out.pop()
                extra = [prefix_cut] if cut else []
                if out and chunks_of(out, extra)[-1]["length"] > s["ceiling"]:
                    J = last
                    while J > out[-1] + 1 and snap_up(n_src - (J - ov)) < s["floor"]:
                        J -= 1
                    ch = chunks_of(out + [J], extra)
                    if ch[-2]["length"] >= s["floor"] and not any(c["length"] > s["ceiling"] for c in ch[-2:]):
                        out.append(J)
        return out

    def chunks_of(js, extra=()):
        st = straight | set(extra)
        return geometry(n_src, [{"id": f"s{i + 1}", "frame": j, "mode": MODE_CUT if j in st else MODE_ANCHORED}
                                for i, j in enumerate(js)], cut_frames, s)

    def over(ch, upto):
        return any(c["length"] > s["ceiling"] for c in ch[:upto + 1])

    js = chain([], target) if target < n_src else []
    k = 0
    while k < len(js):
        ch = chunks_of(js)
        if js[k] in straight or _guard_clear(ch, k + 1, cut_frames, guard):
            k += 1
            continue
        found = {}
        for sign in (1, -1):
            for shift in range(1, max_shift + 1):
                J = js[k] + sign * shift
                lo_lim = js[k - 1] + 1 if k > 0 else 1
                if J < lo_lim or J >= n_src:
                    break
                trial = chain(js[:k], J)
                if len(trial) <= k or trial[k] != J:
                    continue
                tch = chunks_of(trial)
                if all(_guard_clear(tch, i + 1, cut_frames, guard) or not _guard_clear(ch, i + 1, cut_frames, guard)
                       for i in range(k)) and _guard_clear(tch, k + 1, cut_frames, guard):
                    found[sign] = trial
                    break
        best = found.get(1) or found.get(-1)
        if best is not None and over(chunks_of(best), k + 1):
            lo_g = ch[k + 1]["render"][0] - guard
            inside = [c for c in cut_frames if lo_g <= c <= js[k] + guard and (k == 0 or c > js[k - 1])]
            alt = None
            if inside:
                c0 = min(inside, key=lambda c: abs(c - js[k]))
                alt = chain(js[:k], c0, cut=True)
                if not over(chunks_of(alt, [c0]), k + 1):
                    best = alt
                    straight.add(c0)
            if best is not alt and -1 in found and not over(chunks_of(found[-1]), k + 1):
                best = found[-1]
        if best is not None:
            js = best
        k += 1
    return [(j, MODE_CUT if j in straight else MODE_ANCHORED) for j in js] if with_modes else js


# ---------------------------------------------------------------------------
# takes and lineage
# ---------------------------------------------------------------------------

def effective_take(chunk):
    """(take, state): the chosen take ("chosen"), else the chunk's first usable take
    ("unreviewed"), else (None, "pending"). A later take never becomes effective without
    a choice."""
    takes = [t for t in (chunk.get("takes") or []) if t.get("state", "ok") == "ok"]
    chosen = chunk.get("chosen")
    if chosen:
        for t in takes:
            if t.get("id") == chosen:
                return t, "chosen"
    if takes:
        return takes[0], "unreviewed"
    return None, "pending"


def take_covers(take, a, b):
    """Does the take's render range cover source frames a..b (inclusive)?"""
    if not take or not take.get("render"):
        return False
    r0, r1 = take["render"]
    return r0 <= a and b <= r1


def _pin_take(take, side):
    p = (take.get("pins") or {}).get(side) if take else None
    return p.get("take") if isinstance(p, dict) else None


def _pin_source(take, side):
    """Were the take's pins on this side the source's own frames (anchored onto the original)?"""
    p = (take.get("pins") or {}).get(side) if take else None
    return isinstance(p, dict) and bool(p.get("source"))


def _rendered_for(take, side, J):
    """Was the take rendered for an anchored split at J on this side?"""
    sp = (take.get("splits") or {}).get(side) if take else None
    return isinstance(sp, dict) and int(sp.get("frame", -1)) == int(J) and sp.get("mode") == MODE_ANCHORED


def join_info(split, a, b, settings=None):
    """The join at `split` between left take `a` and right take `b` (either may be None).

    Returns {type, splice, repair, hand_back, fade, linked, stale, override}:
      splice  the first output frame taken from the right take
      fade    [r0, splice - 1] for a fade (the right take's overlap), else None
    Types (§4.2, §4.10):
      original  both sides are kept: the source either side, nothing to join
      straight  the split's mode is cut
      pending   a side has no usable take
      entry     (kept left)  b's start pins are the source's: a fade from the source into b over
                its overlap, splice J; info["original"] = "left"
      exit      (kept right) a's end pins are the source's: a fade out of a into the source over
                [J, J + overlap - 1], splice J + overlap; info["original"] = "right"
      forward   b's start pins came from a, b has no end pins        splice J, lock
      entry     b's start pins came from a, b also has end pins      fade over b's overlap, splice J
      exit      a's end pins came from b                              splice at a's end-pin start, lock
      stale     anchored, neither take rendered against the other for this J
    A per-split override (split["repair"] = {"mode": ..., "hand_back": ...}) replaces the
    default repair ("auto" keeps it). A kept side is passed as source_take(...)."""
    s = _settings(settings)
    J = int(split["frame"])
    over = split.get("repair") or {}
    hb = int(over.get("hand_back") or s["hand_back"])
    info = {"split": split.get("id"), "frame": J, "splice": J, "fade": None, "hand_back": hb,
            "linked": False, "stale": False, "override": None, "original": None}
    a_src, b_src = bool(a and a.get("source")), bool(b and b.get("source"))
    if a_src and b_src:
        info.update(type=ORIGINAL, repair=REPAIR_NONE, original="both")
        return info
    if a_src or b_src:
        info["original"] = "left" if a_src else "right"
    if split.get("mode") == MODE_CUT:
        info["type"] = STRAIGHT
    elif a is None or b is None:
        info["type"] = PENDING
    elif a_src:                                     # anchored onto the original, kept on the left
        if _pin_source(b, "start") and _rendered_for(b, "left", J):
            info["type"] = ENTRY
            info["linked"] = True
        else:
            info["type"] = STALE
            info["stale"] = True
    elif b_src:                                     # anchored onto the original, kept on the right
        if _pin_source(a, "end") and _rendered_for(a, "right", J):
            info["type"] = EXIT
            info["linked"] = True
        else:
            info["type"] = STALE
            info["stale"] = True
    elif _pin_take(b, "start") == a.get("id") and _rendered_for(b, "left", J):
        info["type"] = ENTRY if _pin_take(b, "end") else FORWARD
        info["linked"] = True
    elif _pin_take(a, "end") == b.get("id") and _rendered_for(a, "right", J):
        info["type"] = EXIT
        info["linked"] = True
        info["splice"] = int(a["pins"]["end"]["frames"][0])
    else:
        info["type"] = STALE
        info["stale"] = True
    repair = DEFAULT_REPAIR[info["type"]]
    if info["original"] and info["type"] in (ENTRY, EXIT, STALE):
        repair = REPAIR_FADE                        # the original is never locked or re-graded: fade
    mode = over.get("mode")
    if mode and mode != "auto" and info["type"] not in (STRAIGHT, PENDING):
        if mode not in (REPAIR_LOCK, REPAIR_FADE, REPAIR_CUT):
            raise PlanError(f"split {split.get('id')}: repair must be auto, lock, fade or cut, not {mode!r}")
        info["override"] = mode
        repair = mode
    if repair == REPAIR_CUT:
        repair = REPAIR_NONE
        info["override"] = REPAIR_CUT
    if repair == REPAIR_LOCK and info["original"]:
        repair = REPAIR_FADE                        # never lock: the original side is never re-graded
    info["repair"] = repair
    if repair == REPAIR_FADE and info["original"] == "right":
        # fade OUT of the render into the original: over the render's frames past J
        r1 = int(a["render"][1])
        end = min(r1, J + int(s["overlap"]) - 1)
        if end >= J:
            info["fade"] = [J, end]
            info["splice"] = end + 1
            info["fade_dir"] = "out"
        else:
            info["repair"] = REPAIR_NONE
    elif repair == REPAIR_FADE:
        r0 = int(b["render"][0])
        if a is not None:
            r0 = max(r0, int(a["render"][0]))
        if r0 < info["splice"]:
            info["fade"] = [r0, info["splice"] - 1]
        else:                       # nothing rendered twice: a fade has nothing to cross
            info["repair"] = REPAIR_NONE
    return info


def mask_cover(plan, a, b, target=""):
    """The cached source-mask segments (plan["masks"], written by a mark run for `target`) that cover
    source frames a..b, newest first per frame: [(segment, f0, f1), ...] in frame order, or None when
    a frame is uncovered or the segments disagree on the mask size."""
    segs = sorted((m for m in plan.get("masks") or [] if m.get("target", "") == target),
                  key=lambda m: str(m.get("created", "")), reverse=True)
    out, f = [], int(a)
    while f <= b:
        seg = next((m for m in segs if m["range"][0] <= f <= m["range"][1]), None)
        if seg is None:
            return None
        f1 = min(int(b), int(seg["range"][1]))
        # a newer segment starting inside this one takes over from there
        for m in segs:
            if m is seg:
                break
            if f < m["range"][0] <= f1:
                f1 = m["range"][0] - 1
        out.append((seg, f, f1))
        f = f1 + 1
    if len({tuple(s.get("size") or ()) for s, _a, _b in out}) > 1:
        return None
    return out


def mask_coverage(plan, a, b, target=""):
    """How many of source frames a..b some cached mask segment for `target` covers (the strip's mask row)."""
    hit = set()
    for m in plan.get("masks") or []:
        if m.get("target", "") != target:
            continue
        lo, hi = max(int(a), int(m["range"][0])), min(int(b), int(m["range"][1]))
        hit.update(range(lo, hi + 1))
    return len(hit)


def join_measure(plan, j):
    """The latest measurement of join j (a plan_joins row): the assembly's join_cache entry for this
    exact pairing and repair, else a take's own review-clip measurement of it. None if never measured."""
    key = f"{j['split']}|{j.get('left_take')}|{j.get('right_take')}|{j.get('override') or j.get('repair')}|{j.get('hand_back')}"
    hit = (plan.get("join_cache") or {}).get(key)
    if hit:
        return dict(hit, source="assembly")
    for cid, tid in ((j.get("right_chunk"), j.get("right_take")), (j.get("left_chunk"), j.get("left_take"))):
        if not tid:
            continue
        for c in plan.get("chunks", []):
            if c.get("id") != cid:
                continue
            for t in c.get("takes", []):
                if t.get("id") != tid:
                    continue
                for r in t.get("joins") or []:
                    if r.get("split") == j["split"] and r.get("left_take") == j.get("left_take") \
                            and r.get("right_take") == j.get("right_take"):
                        return dict(r, source="review")
    return None


def join_flags(j, m):
    """A split's join flags (colour, motion, following, lineage) and its verdict, the worst of
    them: swap_scores.join_flags, with the thresholds B3 fitted (design §4.5, T-FLAGS)."""
    try:
        from . import swap_scores as ss
    except ImportError:  # top-level import (tests, tools)
        import swap_scores as ss
    return ss.join_flags(j, m)


def join_verdict(j, m):
    """green / amber / red / None for a split's pill (join_flags' verdict)."""
    return join_flags(j, m)["verdict"]


def chunk_take(plan, chunk):
    """(take, state) a chunk delivers: the source for a kept chunk ("original"), else its
    effective take (§4.4)."""
    if is_kept(chunk):
        return source_take(plan["source"]["frames"]), ORIGINAL
    return effective_take(chunk)


def plan_joins(plan):
    """Every join of a plan in order: join_info plus the chunks either side, their effective
    takes and states. A take that no longer covers its chunk counts as pending."""
    s = plan.get("settings")
    chunks = plan.get("chunks") or []
    out = []
    for k in range(1, len(chunks)):
        L, R = chunks[k - 1], chunks[k]
        split = next((d for d in plan.get("splits", []) if d.get("id") == R.get("left")), None)
        if split is None:
            raise PlanError(f"chunk {R.get('id')} has no left split {R.get('left')!r}")
        a, a_state = chunk_take(plan, L)
        b, b_state = chunk_take(plan, R)
        if a is not None and not take_covers(a, *L["deliver"]):
            a, a_state = None, "range changed"
        if b is not None and not take_covers(b, *R["deliver"]):
            b, b_state = None, "range changed"
        j = join_info(split, a, b, s)
        j.update(left_chunk=L.get("id"), right_chunk=R.get("id"), left_take=a and a.get("id"),
                 right_take=b and b.get("id"), left_state=a_state, right_state=b_state)
        out.append(j)
    return out


# ---------------------------------------------------------------------------
# the plan file
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()     # one process-wide lock: every read-modify-write of any plan


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def new_plan(job, source, settings=None):
    """A fresh plan. `source`: {path, frames, fps, width, height, audio, size, mtime_ns}."""
    return {"format": FORMAT, "rev": 0, "job": job, "created": now(), "source": dict(source),
            "settings": _settings(settings), "subject": "", "cuts": [], "splits": [], "chunks": [],
            "join_cache": {}, "assembled": [], "masks": [], "trash": [], "next_ids": {"split": 1, "chunk": 1}}


def _next_id(plan, kind, prefix):
    ids = plan.setdefault("next_ids", {})
    n = int(ids.get(kind, 1))
    ids[kind] = n + 1
    return f"{prefix}{n}"


def add_split(plan, frame, mode=MODE_ANCHORED, repair=None):
    sid = _next_id(plan, "split", "s")
    d = {"id": sid, "frame": int(frame), "mode": mode}
    if repair:
        d["repair"] = dict(repair)
    plan.setdefault("splits", []).append(d)
    plan["splits"].sort(key=lambda x: x["frame"])
    return d


def rebuild_chunks(plan):
    """Recompute every chunk's geometry from the splits, keeping each chunk's own data (prompt,
    options, takes, choice). A chunk is identified by the split on its left (None = the first),
    so moving a split keeps its chunks, and adding one adds a chunk. Returns the warnings."""
    src = plan["source"]
    old = {c.get("left"): c for c in plan.get("chunks", [])}
    split_ids = {d.get("id") for d in plan.get("splits", [])} | {None}
    keep = {c.get("left") for c in plan.get("chunks", []) if is_kept(c) and c.get("left") in split_ids}
    geo = geometry(src["frames"], plan.get("splits", []), plan.get("cuts", []), plan.get("settings"), keep)
    chunks = []
    for g in geo:
        left = g["left"]["id"] if g["left"] else None
        c = old.get(left)
        if c is None:
            c = {"id": _next_id(plan, "chunk", "c"), "left": left, "prompt": "", "prompt_state": "empty",
                 "draft": None, "options": {"mark": True}, "seed_mode": "new", "chosen": None, "takes": []}
        c.update(deliver=g["deliver"], render=g["render"], held=g["held"], length=g["length"], fill=g["fill"])
        chunks.append(c)
    plan["chunks"] = chunks
    plan["warnings"] = warnings(src["frames"], plan.get("splits", []), plan.get("cuts", []),
                                plan.get("settings"), geo, keep)
    return plan["warnings"]


def find_chunk(plan, chunk_id):
    for c in plan.get("chunks", []):
        if c.get("id") == chunk_id:
            return c
    raise PlanError(f"no chunk {chunk_id!r} in job {plan.get('job')!r}")


def find_take(plan, take_id):
    for c in plan.get("chunks", []):
        for t in c.get("takes", []):
            if t.get("id") == take_id:
                return c, t
    raise PlanError(f"no take {take_id!r} in job {plan.get('job')!r}")


def next_take_id(chunk):
    n = 0
    for t in chunk.get("takes", []):
        try:
            n = max(n, int(str(t["id"]).rsplit("-t", 1)[1]))
        except (KeyError, IndexError, ValueError):
            pass
    return f"{chunk['id']}-t{n + 1:03d}"


def plan_path(job_dir):
    return os.path.join(job_dir, "plan.json")


def job_dir(output_dir, job):
    if not job or any(ch in job for ch in '<>:"|?*') or job.strip(". ") != job or os.sep in job or "/" in job:
        raise PlanError(f"job name {job!r} must be a plain folder name")
    return os.path.join(output_dir, JOBS_SUBDIR, job)


def load_plan(path):
    """Read a plan. Refuses a plan written by a newer format."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except FileNotFoundError:
        raise PlanError(f"no plan file at {path}")
    except ValueError as e:
        raise PlanError(f"plan file {path} is not valid JSON: {e}")
    fmt = plan.get("format")
    if fmt != FORMAT:
        raise PlanError(f"plan file {path} is format {fmt!r}; this SeamStitch reads {FORMAT!r} "
                        f"(update the pack to open a newer plan)")
    return plan


def _atomic_write(path, text):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    for i in range(20):        # Windows: a reader holding the file open blocks the replace briefly
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    os.replace(tmp, path)


def _prune_history(hdir, keep=HISTORY_KEEP):
    revs = []
    for f in os.listdir(hdir):
        if f.startswith("plan_") and f.endswith(".json"):
            try:
                revs.append((int(f[5:-5]), f))
            except ValueError:
                pass
    revs.sort()
    for _, f in revs[:-keep] if len(revs) > keep else []:
        try:
            os.remove(os.path.join(hdir, f))
        except OSError:
            pass


def save_plan(path, plan, expect_rev=None):
    """Write the plan atomically with its revision bumped, and keep a copy in history/.
    expect_rev: the revision the caller read; a mismatch with the file raises PlanConflict
    (use update_plan to merge instead). Returns the saved plan."""
    with _LOCK:
        if expect_rev is not None and os.path.isfile(path):
            cur = load_plan(path).get("rev", 0)
            if cur != expect_rev:
                raise PlanConflict(f"plan {path} is at revision {cur}, not {expect_rev}")
        plan = dict(plan)
        plan["format"] = FORMAT
        plan["rev"] = int(plan.get("rev", 0)) + 1
        plan["saved"] = now()
        text = json.dumps(plan, indent=1, ensure_ascii=False)
        _atomic_write(path, text)
        hdir = os.path.join(os.path.dirname(os.path.abspath(path)), "history")
        os.makedirs(hdir, exist_ok=True)
        _atomic_write(os.path.join(hdir, f"plan_{plan['rev']}.json"), text)
        _prune_history(hdir)
        return plan


def update_plan(path, fn, expect_rev=None):
    """Read-modify-write under the process-wide lock: fn(plan) edits the plan in place (or
    returns a new one). Concurrent writers are serialised, never lost."""
    with _LOCK:
        plan = load_plan(path)
        if expect_rev is not None and plan.get("rev") != expect_rev:
            raise PlanConflict(f"plan {path} is at revision {plan.get('rev')}, not {expect_rev}")
        res = fn(plan)
        if isinstance(res, dict) and res.get("format") == FORMAT:
            plan = res
        return save_plan(path, plan)


def snapshot(plan):
    return copy.deepcopy(plan)
