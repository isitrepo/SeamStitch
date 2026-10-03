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
    "ceiling": 260,           # warning above (the tested 1080p ceiling, T3)
    "trained_max": 362,       # warning above (H3's trained range)
    "guard": 6,               # overlap guard either side of an anchored overlap (half T2's widest morph)
    "conform_to_24fps": True,
    "hand_back": 12,          # level lock / fade decay length (picked by eye over 25)
}

# join types
STRAIGHT, FORWARD, ENTRY, EXIT, STALE, PENDING = "straight", "forward", "entry", "exit", "stale", "pending"
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


def geometry(frames, splits, cuts=(), settings=None):
    """Chunks for a source of `frames` frames split at `splits`.

    Returns a list of dicts, one per chunk, in order:
      deliver [a, b]   source frames this chunk delivers (inclusive)
      render  [r0, r1] source frames it renders (overlap and tail/head fill included)
      held             guide frames held after r1 (hold fill), dropped from the take
      length           the render length H3 sees: r1 - r0 + 1 + held (always 17k+5)
      base             the unfilled render length
      fill             None or {"kind": "tail"|"head"|"hold", "frames": n}
      left / right     the bounding split dicts (None at the ends of the video)
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
    chunks = []
    for k in range(len(sp) + 1):
        left, right = bounds[k], bounds[k + 1]
        a = left["frame"] if left else 0
        b = right["frame"] - 1 if right else n_src - 1
        r0 = max(0, a - ov) if left and left["mode"] == MODE_ANCHORED else a
        r1 = b
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
                       "fill": fill, "left": left, "right": right})
    return chunks


def warnings(frames, splits, cuts=(), settings=None, chunks=None):
    """Every §4.1 warning (none of them blocks). Each: {code, text, split|chunk, ...}."""
    s = _settings(settings)
    cut_frames = confirmed_cuts(cuts)
    chunks = chunks if chunks is not None else geometry(frames, splits, cuts, s)
    out = []
    for i, c in enumerate(chunks):
        L = c["length"]
        where = {"chunk": i, "deliver": c["deliver"]}
        if L < s["floor"]:
            out.append(dict(where, code="floor", text=f"render {L} frames is under the floor ({s['floor']})"))
        if L > s["trained_max"]:
            out.append(dict(where, code="trained", text=f"render {L} frames is over H3's trained range ({s['trained_max']})"))
        elif L > s["ceiling"]:
            out.append(dict(where, code="ceiling", text=f"render {L} frames is over the tested ceiling ({s['ceiling']})"))
    for i, c in enumerate(chunks[1:], start=1):
        sp = c["left"]
        J = sp["frame"]
        if sp["mode"] == MODE_ANCHORED:
            lo, hi = c["render"][0] - s["guard"], J + s["guard"]
            inside = [x for x in cut_frames if lo <= x <= hi]
            if inside:
                out.append({"split": sp.get("id"), "frame": J, "code": "guard", "cuts": inside, "guard": [lo, hi],
                            "text": (f"anchored split at {J}: cut at {', '.join(map(str, inside))} inside its overlap "
                                     f"guard [{lo}, {hi}]. Make it a straight cut on that cut, or move it clear."),
                            "remedies": [{"action": "straight_cut", "frame": inside[0]},
                                         {"action": "move", "clear": clearance(frames, splits, cuts, s, sp.get("id"))}]})
        else:
            if not any(abs(x - J) <= 1 for x in cut_frames):
                out.append({"split": sp.get("id"), "frame": J, "code": "no_cut",
                            "text": f"straight split at {J}: visible jump, the source doesn't cut here"})
    return out


def _guard_clear(chunks, k, cut_frames, guard):
    """Is the anchored split on the left of chunk k clear of every cut (after the fill)?"""
    c = chunks[k]
    sp = c["left"]
    if sp is None or sp["mode"] != MODE_ANCHORED:
        return True
    lo, hi = c["render"][0] - guard, sp["frame"] + guard
    return not any(lo <= x <= hi for x in cut_frames)


def clearance(frames, splits, cuts, settings, split_id):
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
                ch = geometry(frames, trial, cut_frames, s)
            except PlanError:
                continue
            if _guard_clear(ch, idx + 1, cut_frames, s["guard"]):
                found[name] = J
                break
    return found


def auto_splits(frames, cuts=(), settings=None, max_shift=None):
    """Anchored split frames for target-length renders: the first at target_render, then every
    target_render - overlap. A last chunk whose render would fall under the floor merges into
    the one before. An anchored split whose overlap guard holds a cut (checked AFTER the fill,
    since a head fill lengthens the overlap) is nudged forward, else backward, by the smallest
    shift that clears it without fouling an earlier split's guard; the splits after it are
    re-placed from its new position."""
    s = _settings(settings)
    n_src = int(frames)
    cut_frames = confirmed_cuts(cuts)
    target, ov, guard = int(s["target_render"]), int(s["overlap"]), int(s["guard"])
    step = target - ov
    max_shift = int(max_shift if max_shift is not None else step // 2)

    def chain(prefix, J):
        out = list(prefix)
        while J < n_src:
            out.append(J)
            J += step
        # merge an under-floor last chunk into the one before
        if out and len(out) > len(prefix):
            last = out[-1]
            if snap_up(n_src - (last - ov)) < s["floor"]:
                out.pop()
        return out

    def chunks_of(js):
        return geometry(n_src, [{"id": f"s{i + 1}", "frame": j, "mode": MODE_ANCHORED} for i, j in enumerate(js)],
                        cut_frames, s)

    js = chain([], target) if target < n_src else []
    k = 0
    while k < len(js):
        ch = chunks_of(js)
        if _guard_clear(ch, k + 1, cut_frames, guard):
            k += 1
            continue
        best = None
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
                    best = trial
                    break
            if best is not None:
                break
        if best is not None:
            js = best
        k += 1
    return js


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


def _rendered_for(take, side, J):
    """Was the take rendered for an anchored split at J on this side?"""
    sp = (take.get("splits") or {}).get(side) if take else None
    return isinstance(sp, dict) and int(sp.get("frame", -1)) == int(J) and sp.get("mode") == MODE_ANCHORED


def join_info(split, a, b, settings=None):
    """The join at `split` between left take `a` and right take `b` (either may be None).

    Returns {type, splice, repair, hand_back, fade, linked, stale, override}:
      splice  the first output frame taken from the right take
      fade    [r0, splice - 1] for a fade (the right take's overlap), else None
    Types (§4.2):
      straight  the split's mode is cut
      pending   a side has no usable take
      forward   b's start pins came from a, b has no end pins        splice J, lock
      entry     b's start pins came from a, b also has end pins      fade over b's overlap, splice J
      exit      a's end pins came from b                              splice at a's end-pin start, lock
      stale     anchored, neither take rendered against the other for this J
    A per-split override (split["repair"] = {"mode": ..., "hand_back": ...}) replaces the
    default repair ("auto" keeps it)."""
    s = _settings(settings)
    J = int(split["frame"])
    over = split.get("repair") or {}
    hb = int(over.get("hand_back") or s["hand_back"])
    info = {"split": split.get("id"), "frame": J, "splice": J, "fade": None, "hand_back": hb,
            "linked": False, "stale": False, "override": None}
    if split.get("mode") == MODE_CUT:
        info["type"] = STRAIGHT
    elif a is None or b is None:
        info["type"] = PENDING
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
    mode = over.get("mode")
    if mode and mode != "auto" and info["type"] not in (STRAIGHT, PENDING):
        if mode not in (REPAIR_LOCK, REPAIR_FADE, REPAIR_CUT):
            raise PlanError(f"split {split.get('id')}: repair must be auto, lock, fade or cut, not {mode!r}")
        info["override"] = mode
        repair = mode
    if repair == REPAIR_CUT:
        repair = REPAIR_NONE
        info["override"] = REPAIR_CUT
    info["repair"] = repair
    if repair == REPAIR_FADE:
        r0 = int(b["render"][0])
        if a is not None:
            r0 = max(r0, int(a["render"][0]))
        if r0 < info["splice"]:
            info["fade"] = [r0, info["splice"] - 1]
        else:                       # nothing rendered twice: a fade has nothing to cross
            info["repair"] = REPAIR_NONE
    return info


def mask_cover(plan, a, b):
    """The cached source-mask segments (plan["masks"], written by a mark run) that cover source
    frames a..b, newest first per frame: [(segment, f0, f1), ...] in frame order, or None when a
    frame is uncovered or the segments disagree on the mask size."""
    segs = sorted(plan.get("masks") or [], key=lambda m: str(m.get("created", "")), reverse=True)
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


def mask_coverage(plan, a, b):
    """How many of source frames a..b some cached mask segment covers (for the strip's mask row)."""
    hit = set()
    for m in plan.get("masks") or []:
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


def join_verdict(j, m):
    """green / amber / red / None for a split's pill: the worst of colour (the luma jump at the
    splice, frame and character: <= 1.0, <= 2.0) and motion (Result Preview's verdict), with a stale
    join never better than amber. Provisional: B3 fits the thresholds (T-FLAGS)."""
    if j["type"] in (STRAIGHT, PENDING):
        return None
    rank = {"green": 0, "amber": 1, "red": 2}
    worst = "amber" if j.get("stale") else None
    if m:
        for key in ("frame_luma", "char_luma"):
            v = (m.get(key) or {}).get("at_splice")
            if v is None:
                continue
            v = abs(float(v))
            c = "green" if v <= 1.0 else "amber" if v <= 2.0 else "red"
            worst = c if worst is None or rank[c] > rank[worst] else worst
        mv = {"seamless": "green", "soft bump": "amber", "hard cut": "red"}.get(m.get("join_verdict"))
        if mv:
            worst = mv if worst is None or rank[mv] > rank[worst] else worst
    return worst


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
        a, a_state = effective_take(L)
        b, b_state = effective_take(R)
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
    geo = geometry(src["frames"], plan.get("splits", []), plan.get("cuts", []), plan.get("settings"))
    old = {c.get("left"): c for c in plan.get("chunks", [])}
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
                                plan.get("settings"), geo)
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
