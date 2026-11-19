"""SeamStitch Swap Planner and Swap Option: the job's plan, one queue run at a time.

The Planner is the editor of a job's plan.json (swap_plan.py) and the head of every run. Its
hidden `run` widget says what this queue item does:

    {"action": "render", "chunk": "c3", "seed": 123, "prompt": "...", "options": {...}, "nonce": "..."}
    {"action": "mark", "chunk": "c3"} / {"action": "assemble"} / {"action": "draft"}
    "" (nothing: the panel only)

A render run emits chunk k's render inputs (outputs 0-12): the source frames over its render
range (+ held frames), the audio under them (silence under held frames; on H3's 24 fps clock
when conform_to_24fps is on), the prompt snapshot, the seed, the length, and the pins, decoded
AT EXECUTION from the neighbours' effective takes (never at queue time: no file-path widget is
ever validated). When a mark run has cached the source person mask over the chunk, it comes out
too (17-18), so the render group's lazy switch skips its own SAM3 track. A mark run emits a
range's source frames (15-16) for the mark group, whose SeamStitch Swap Mask caches the mask.
Every output a run doesn't use is ExecutionBlocker(None), ComfyUI's silent block, so draft,
mark, render and assemble live in one workflow with no switches. (The strip UI also queues
draft, mark and assemble runs with ComfyUI's partial execution, aimed at their own output nodes,
so the render group's loaders never run for them.)

The routes edit the plan off the queue (js/swap_planner.js is their UI):
    GET  /seamstitch/swap/plan?job=<job>     the plan, joins + verdicts, chunk states, warnings
    POST /seamstitch/swap/op                 {job, op, ...}: create, source, settings, cuts,
                                             splits (auto too), prompts, drafts, options, seed
                                             mode, choose / delete / restore / reuse a take,
                                             render failures, masks
    POST /seamstitch/swap/detect_cuts        ffmpeg scene > 0.15, merged as suggested cuts (0.10-0.15: faint)
"""

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
import uuid

import numpy as np
import torch

try:
    from . import swap_plan as sp
    from . import swap_scores as ss
    from . import timeline as tl
except ImportError:  # imported as a top-level module (tests, tools)
    import swap_plan as sp
    import swap_scores as ss
    import timeline as tl

try:
    import folder_paths
except ImportError:  # pragma: no cover
    folder_paths = None

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:  # pragma: no cover - tests without ComfyUI
    class ExecutionBlocker:
        def __init__(self, message):
            self.message = message

CHUNK_TYPE = "SEAMSTITCH_SWAP_CHUNK"
CHUNK_FORMAT = "seamstitch_swap_chunk_v1"
MARK_TYPE = "SEAMSTITCH_SWAP_MARK"
MARK_FORMAT = "seamstitch_swap_mark_v1"
ACTION_RENDER, ACTION_ASSEMBLE, ACTION_DRAFT, ACTION_MARK, ACTION_NONE = "render", "assemble", "draft", "mark", "none"
ACTIONS = (ACTION_RENDER, ACTION_ASSEMBLE, ACTION_DRAFT, ACTION_MARK, ACTION_NONE)

# Outputs are only ever appended (links are by slot). 15-16 carry a mark run (the source's person mask,
# tracked up front and cached per range); 17-18 a render run's cached source mask, when one covers it; 19-20 the
# first pass's size in the source's shape (H3's width / height).
RETURN_TYPES = (CHUNK_TYPE, "IMAGE", "AUDIO", "STRING", "INT", "INT", "IMAGE", "IMAGE", "INT", "INT", "INT",
                "INT", "STRING", "STRING", "STRING", MARK_TYPE, "IMAGE", "MASK", "BOOLEAN", "INT", "INT", "BOOLEAN",
                "MASK", "BOOLEAN", "INT", "INT", "STRING", "IMAGE", "BOOLEAN")
RETURN_NAMES = ("chunk", "images", "audio", "prompt", "seed", "length", "start_pins", "end_pins", "width",
                "height", "frame_rate", "source_frame_rate", "options", "draft_plan", "assemble_plan",
                "mark_chunk", "mark_images", "source_mask", "has_source_mask", "render_width", "render_height",
                "object_refine", "object_zone", "zone_on", "anchor_frame", "enter_frame", "anchor_prompt", "anchor_image",
                "anchor_qwen")
OUT_OPTIONS, OUT_DRAFT, OUT_ASSEMBLE = 12, 13, 14
OUT_MARK_CHUNK, OUT_MARK_IMAGES, OUT_SOURCE_MASK, OUT_HAS_MASK = 15, 16, 17, 18
OUT_RENDER_W, OUT_RENDER_H = 19, 20
OUT_OBJECT_REFINE = 21          # an object add with a cut-out: images are the composite, for the H3 img2img refine
# 22-25, an object add with guide "zone": the zone per frame at the images' size (white = H3 redraws it), whether this
# render is one, the frame the anchor picture is drawn into, and the frame the thing enters (the zone stays empty before)
OUT_OBJECT_ZONE, OUT_ZONE_ON, OUT_ANCHOR_FRAME, OUT_ENTER_FRAME = 22, 23, 24, 25
# 26-28: what Qwen is asked to draw into the anchor frame (the chunk's anchor text, else a plain sentence), the anchor
# picture of the "composite" and "picture" anchors (a frame of the render's size), and whether Qwen draws the anchor
OUT_ANCHOR_PROMPT, OUT_ANCHOR_IMAGE, OUT_ANCHOR_QWEN = 26, 27, 28
# H3's first pass: about 0.3 MP (736 x 416 for 16:9). The example graph had 736 x 416 fixed, so a portrait source
# (the 720 x 1280 wrestling clip, 2026-10-09) was generated landscape and stretched to 9:16 by the resize.
PASS1_AREA = 736 * 416
PASS1_STEP = 32


def pass1_size(width, height, area=PASS1_AREA, step=PASS1_STEP):
    """The first pass's width and height: the source's shape at about `area` pixels, multiples of `step`
    (1920 x 1080 -> 736 x 416, 720 x 1280 -> 416 x 736, 1080 x 1080 -> 544 x 544)."""
    a = float(width) / float(height)
    h = (area / a) ** 0.5
    w = a * h
    return max(step, int(round(w / step)) * step), max(step, int(round(h / step)) * step)


def fit_size(width, height, megapixels, step=32, tolerance=0.15):
    """A single pass's width and height: multiples of `step` within `tolerance` of `megapixels` that keep the
    source's shape closest (1280 x 720 at 1.0 -> 1312 x 736, off by 0.3% where the two-sides-rounded size 1344 x 736
    is off by 2.7%; 720 x 1280 -> 736 x 1312; 1080 x 1080 -> 1024 x 1024). The pins are stretched to this size, so
    what shape error is left is undone when the render is stretched back to the source's size."""
    import math
    a = float(width) / float(height)
    area = float(megapixels) * 1e6
    best = None
    lo, hi = max(step, int(math.sqrt(area * (1 - tolerance) * a) // step) * step), int(math.sqrt(area * (1 + tolerance) * a) // step + 2) * step
    for w in range(lo, hi + 1, step):
        for h in (int(round(w / a / step)) * step + d for d in (-step, 0, step)):
            if h < step:
                continue
            miss = abs(w * h / area - 1.0)
            if miss > tolerance:
                continue
            key = (abs(math.log((w / float(h)) / a)), miss)
            if best is None or key < best[0]:
                best = (key, (w, h))
    return best[1] if best else pass1_size(width, height, area=area, step=step)
SCENE_THRESHOLD = 0.15           # ffmpeg scene score (> 0.15 found every cut of the 978-frame test clip)
# same-room jump cuts score lower: the test clip's 131, 448, 669, 713, 764 and 850 read 0.10-0.14 (B5a); they come in as
# faint suggestions only (never confirmed by "detect + plan"), for the user to confirm or delete by eye
WEAK_THRESHOLD = 0.10
WEAK_NEAR = 2                    # a weak hit this close to a stronger one is the same cut, read twice (the test clip: 902 / 903)

# widget name -> plan setting
WIDGET_SETTINGS = {"target_render_frames": "target_render", "overlap_frames": "overlap", "anchor_frames": "anchors",
                   "floor_frames": "floor", "ceiling_frames": "ceiling", "conform_to_24fps": "conform_to_24fps",
                   "first_pass_megapixels": "first_pass_mp", "forward_chain": "forward_chain"}


class PlannerError(Exception):
    pass


def _say(msg):
    """print() that can't take a run down: on a console or log file that can't encode a character (a
    cp1252 stdout redirected to a file), the line goes out with that character replaced. A failed print
    inside a node otherwise kills ComfyUI's prompt worker (found in B3b on a redirected test server)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        import sys
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        try:
            print(msg.encode(enc, errors="replace").decode(enc, errors="replace"))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# job paths
# ---------------------------------------------------------------------------

def output_dir():
    if folder_paths is None:
        raise PlannerError("no ComfyUI output folder")
    return folder_paths.get_output_directory()


def job_paths(job, out_dir=None):
    jd = sp.job_dir(out_dir or output_dir(), (job or "").strip())
    return jd, sp.plan_path(jd)


def _norm(p):
    return os.path.normcase(os.path.abspath(p or ""))


def source_facts(path):
    """The plan's source entry for a video file (its own frame rate, rounded)."""
    path = tl.resolve_path(path.strip().strip('"'))
    fr = tl.auto_frame_rate(0, path)
    info = tl.probe(path, fr)
    st = os.stat(path)
    return {"path": os.path.abspath(path).replace("\\", "/"), "frames": int(info["frames"]), "fps": fr,
            "width": int(info["width"]), "height": int(info["height"]), "audio": bool(info["has_audio"]),
            "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def create_job(job, source, settings=None, out_dir=None):
    """A new job folder with an empty plan (no splits: one chunk). Refuses an existing job."""
    jd, pp = job_paths(job, out_dir)
    if os.path.isfile(pp):
        raise PlannerError(f"job {job!r} already exists ({pp})")
    plan = sp.new_plan(job, source_facts(source), settings)
    sp.rebuild_chunks(plan)
    for sub in ("history", "chunks", "assembled", "trash", "drafts"):
        os.makedirs(os.path.join(jd, sub), exist_ok=True)
    return sp.save_plan(pp, plan)


# ---------------------------------------------------------------------------
# plan edits (pure: plan dict in, plan dict changed in place)
# ---------------------------------------------------------------------------

# what each mode keeps of its own: the subject (a character or a place) and each chunk's prompt and draft
_MODE_PLAN_KEYS = ("subject", "subject_draft", "subject_pronoun")
_MODE_CHUNK_KEYS = ("prompt", "prompt_state", "draft", "drafted", "chosen")     # each mode chooses its own take


def switch_mode(plan, mode):
    """Character replace <-> background replace. Nothing is cleared: the mode being left keeps its subject and
    every chunk's prompt (under by_mode), the other mode's come back (empty the first time: draft prompts fills
    them). Takes stay where they are; only the mode's own count (sp.effective_take). The cached masks serve both
    (the paint mask is inverted at render)."""
    old = sp.job_mode(plan)
    plan.pop("invert", None)
    plan["mode"] = mode
    for c in plan.get("chunks", []):                      # the mode is per job now (a chunk's own invert is gone)
        (c.get("options") or {}).pop("invert", None)
    if mode == old:
        return {"mode": mode, "switched": False}
    store = plan.setdefault("by_mode", {})
    store[old] = {"plan": {k: plan.get(k) for k in _MODE_PLAN_KEYS},
                  "chunks": {c["id"]: {k: c.get(k) for k in _MODE_CHUNK_KEYS} for c in plan.get("chunks", [])}}
    back = store.pop(mode, None) or {}
    for k in _MODE_PLAN_KEYS:
        plan[k] = (back.get("plan") or {}).get(k)
    plan["subject"] = plan.get("subject") or ""
    saved = back.get("chunks") or {}
    restored = 0
    for c in plan.get("chunks", []):
        s = saved.pop(c["id"], None)
        if s:
            restored += 1
        for k in _MODE_CHUNK_KEYS:
            c[k] = (s or {}).get(k)
        c["prompt"] = c.get("prompt") or ""
        c["prompt_state"] = c.get("prompt_state") or ("edited" if c["prompt"].strip() else "empty")
    if saved:                                             # chunks re-split meanwhile: their prompts are kept, unused
        plan.setdefault("by_mode_unplaced", []).extend(dict(v, chunk=k, mode=mode, at=sp.now()) for k, v in saved.items())
    return {"mode": mode, "switched": True, "from": old, "restored_prompts": restored}


def _split(plan, body):
    sid = body.get("split") or body.get("id")
    for d in plan.get("splits", []):
        if d["id"] == sid:
            return d
    if "frame" in body and sid is None:
        for d in plan.get("splits", []):
            if d["frame"] == int(body["frame"]):
                return d
    raise sp.PlanError(f"no split {sid!r}")


def _chunk(plan, ref):
    """A chunk by id ("c3"), by index (int, 0-based), or by a frame it delivers ("@412")."""
    chunks = plan.get("chunks") or []
    if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
        i = int(ref)
        if not 0 <= i < len(chunks):
            raise sp.PlanError(f"no chunk index {i} (the plan has {len(chunks)})")
        return chunks[i]
    if isinstance(ref, str) and ref.startswith("@"):
        f = int(ref[1:])
        for c in chunks:
            if c["deliver"][0] <= f <= c["deliver"][1]:
                return c
        raise sp.PlanError(f"no chunk delivers frame {f}")
    return sp.find_chunk(plan, ref)


def apply_op(plan, body):
    """One edit from the strip (or a script). Returns a small result dict. Geometry edits rebuild
    the chunks and their warnings."""
    op = body.get("op")
    geo = False
    res = {"op": op}
    if op == "settings":
        s = dict(plan.get("settings") or {})
        for k, v in (body.get("settings") or {}).items():
            if k not in sp.DEFAULT_SETTINGS:
                raise sp.PlanError(f"unknown setting {k!r}")
            s[k] = type(sp.DEFAULT_SETTINGS[k])(v)
        plan["settings"] = s
        # the render's size doesn't move a chunk: every other setting rebuilds the geometry
        geo = any(k not in ("first_pass_mp", "forward_chain") for k in (body.get("settings") or {}))
    elif op == "add_cut":
        f = int(body["frame"])
        cuts = [c for c in plan.get("cuts", []) if c["frame"] != f]
        cuts.append({"frame": f, "from": body.get("from", "manual"), "confirmed": bool(body.get("confirmed", True))})
        plan["cuts"] = sorted(cuts, key=lambda c: c["frame"])
        geo = True
    elif op == "delete_cut":
        f = int(body["frame"])
        plan["cuts"] = [c for c in plan.get("cuts", []) if c["frame"] != f]
        geo = True
    elif op == "confirm_cut":
        f = int(body["frame"])
        hit = [c for c in plan.get("cuts", []) if c["frame"] == f]
        if not hit:
            raise sp.PlanError(f"no cut at {f}")
        hit[0]["confirmed"] = bool(body.get("confirmed", True))
        geo = True
    elif op == "add_split":
        mode = body.get("mode", sp.MODE_ANCHORED)
        if mode not in (sp.MODE_CUT, sp.MODE_ANCHORED):
            raise sp.PlanError(f"mode must be {sp.MODE_CUT!r} or {sp.MODE_ANCHORED!r}")
        d = sp.add_split(plan, int(body["frame"]), mode)
        res["split"] = d["id"]
        geo = True
    elif op == "move_split":
        _split(plan, body)["frame"] = int(body["to"])
        plan["splits"].sort(key=lambda x: x["frame"])
        geo = True
    elif op == "delete_split":
        d = _split(plan, body)
        plan["splits"] = [x for x in plan["splits"] if x is not d]
        gone = [c for c in plan.get("chunks", []) if c.get("left") == d["id"]]
        if gone and gone[0].get("takes"):
            plan.setdefault("removed_chunks", []).append(dict(gone[0], removed=sp.now()))
        geo = True
    elif op == "split_mode":
        mode = body.get("mode")
        if mode not in (sp.MODE_CUT, sp.MODE_ANCHORED):
            raise sp.PlanError(f"mode must be {sp.MODE_CUT!r} or {sp.MODE_ANCHORED!r}")
        _split(plan, body)["mode"] = mode
        geo = True
    elif op == "split_repair":
        d = _split(plan, body)
        rep = {k: v for k, v in (body.get("repair") or {}).items() if v not in (None, "", "auto")}
        if rep:
            d["repair"] = rep
        else:
            d.pop("repair", None)
    elif op == "set_prompt":
        c = _chunk(plan, body["chunk"])
        c["prompt"] = body.get("prompt") or ""
        c["prompt_state"] = "edited" if c["prompt"].strip() else "empty"
    elif op == "set_anchor_text":
        # a zone add's Qwen instruction for one chunk ("" = the plain sentence swap_plan.zone_anchor_text writes)
        c = _chunk(plan, body["chunk"])
        c["anchor_text"] = (body.get("text") or "").strip()
    elif op == "set_line_speaker":
        # who says one dialogue line of the chunk's prompt (the strip's picker, where the drafter couldn't tell)
        try:
            from . import swap_draft as sd      # swap_draft imports this module
        except ImportError:
            import swap_draft as sd
        c = _chunk(plan, body["chunk"])
        try:
            c["prompt"] = sd.set_line_speaker(c.get("prompt") or "", int(body["line"]), body["speaker"])
        except sd.DraftError as e:
            raise sp.PlanError(str(e))
        c["prompt_state"] = "edited"
    elif op == "set_options":
        c = _chunk(plan, body["chunk"])
        c["options"] = dict(c.get("options") or {}, **{k: v for k, v in (body.get("options") or {}).items()
                                                      if k != "invert"})      # the mode is per job
    elif op == "seed_mode":
        c = _chunk(plan, body["chunk"])
        m = body.get("seed_mode", "new")
        if m != "new" and not (isinstance(m, dict) and isinstance(m.get("fixed"), int)):
            raise sp.PlanError("seed_mode is \"new\" or {\"fixed\": N}")
        c["seed_mode"] = m
    elif op == "set_subject":
        # the subject is shared: an edit replaces the old subject text in every chunk's prompt and draft
        old, plan["subject"] = (plan.get("subject") or "").strip(), (body.get("subject") or "").strip()
        n = 0
        if old and plan["subject"] and old != plan["subject"]:
            for c in plan.get("chunks", []):
                for k in ("prompt", "draft"):
                    if old in (c.get(k) or ""):
                        c[k] = c[k].replace(old, plan["subject"])
                        n += 1
        res["subject_replaced_in"] = n
    elif op == "set_target":
        # the job's who-to-replace (SAM3's source text, the drafter's main person; "" = "person" and the drafter's
        # pick); a chunk's own goes in its options. ("invert" from an older strip: the mode switch.)
        if "target" in body:
            plan["target"] = (body.get("target") or "").strip()
        if "invert" in body:
            res.update(switch_mode(plan, sp.MODE_BACKGROUND if body["invert"] else sp.MODE_CHARACTER))
    elif op == "set_guide":
        if body.get("guide") not in sp.GUIDES:
            raise sp.PlanError(f"guide is one of {', '.join(sp.GUIDES)}, not {body.get('guide')!r}")
        plan["guide"] = body["guide"]
    elif op == "set_mode":
        if body.get("mode") not in sp.MODES:
            raise sp.PlanError(f"mode is one of {', '.join(sp.MODES)}, not {body.get('mode')!r}")
        res.update(switch_mode(plan, body["mode"]))
    elif op == "adopt_subject_draft":
        if not (plan.get("subject_draft") or "").strip():
            raise sp.PlanError("there's no subject draft to adopt")
        res.update(apply_op(plan, {"op": "set_subject", "subject": plan["subject_draft"]}), op=op)
        plan["subject_draft"] = None
    elif op == "choose_take":
        take = body.get("take")
        if take:
            c, _t = sp.find_take(plan, take)
            if body.get("chunk") and _chunk(plan, body["chunk"]) is not c:
                raise sp.PlanError(f"take {take} isn't chunk {body['chunk']}'s")
            if sp.take_mode(_t) != sp.job_mode(plan):
                raise sp.PlanError(f"take {take} is a {sp.take_mode(_t)} replace take: switch the job to "
                                   f"{sp.take_mode(_t).capitalize()} replace to choose it")
            c["chosen"] = take
        else:
            _chunk(plan, body["chunk"])["chosen"] = None
    elif op == "confirm_cuts":
        want = body.get("frames")
        for c in plan.get("cuts", []):
            if want is None or c["frame"] in [int(x) for x in want]:
                c["confirmed"] = bool(body.get("confirmed", True))
        geo = True
    elif op == "auto_splits":
        # 209-frame renders from the confirmed cuts (§4.1). Replacing splits drops their chunks'
        # prompts and takes, so it is refused once any chunk has either, unless forced. Kept chunks
        # (§4.10) stay, with the splits either side of them; auto splits fill each rendered stretch.
        chunks = plan.get("chunks", [])
        live = [c for c in chunks if not sp.is_kept(c)]
        if not body.get("force") and any(c.get("takes") or (c.get("prompt") or "").strip() for c in live):
            raise sp.PlanError("auto splits would replace chunks that have prompts or takes (force to do it anyway)")
        if any(c.get("takes") for c in live):
            plan.setdefault("removed_chunks", []).extend(dict(c, removed=sp.now()) for c in live if c.get("takes"))
        n = int(plan["source"]["frames"])
        by_id = {d["id"]: d for d in plan.get("splits", [])}
        kept, keep_splits = [], {}
        for i, c in enumerate(chunks):
            if sp.is_kept(c):
                kept.append(c)
                for sid in (c.get("left"), chunks[i + 1].get("left") if i + 1 < len(chunks) else None):
                    if sid in by_id:
                        keep_splits[sid] = by_id[sid]
        spans, f0 = [], 0
        for c in kept:
            a, b = c["deliver"]
            if a > f0:
                spans.append((f0, a - 1))
            f0 = b + 1
        if f0 <= n - 1:
            spans.append((f0, n - 1))
        cut_frames = sp.confirmed_cuts(plan.get("cuts"))
        placed = []
        for a, b in spans:
            rel = [c - a for c in cut_frames if a < c <= b]
            placed += [(a + x, m) for x, m in sp.auto_splits(b - a + 1, rel, plan.get("settings"), with_modes=True)]
        plan["splits"] = sorted(keep_splits.values(), key=lambda d: d["frame"])
        plan["chunks"] = kept
        have = {d["frame"] for d in plan["splits"]}
        for f, m in placed:
            if f not in have:
                sp.add_split(plan, f, m)
        res["splits"] = [f for f, _ in placed]
        res["straight"] = [f for f, m in placed if m == sp.MODE_CUT]
        geo = True
    elif op == "adopt_draft":
        c = _chunk(plan, body["chunk"])
        if not (c.get("draft") or "").strip():
            raise sp.PlanError(f"chunk {c['id']} has no draft to adopt")
        c["prompt"], c["prompt_state"], c["draft"] = c["draft"], "draft", None
    elif op == "render_failed":
        c = _chunk(plan, body["chunk"])
        if body.get("nonce") and (c.get("rendering") or {}).get("nonce") not in (None, body["nonce"]):
            res["ignored"] = "a newer render of this chunk is running"
        else:
            c.pop("rendering", None)
            c["state"] = "failed"
            c["error"] = str(body.get("error") or "the render failed")[:2000]
            c["failed"] = sp.now()
    elif op == "clear_state":
        c = _chunk(plan, body["chunk"])
        c.pop("rendering", None)
        c.pop("error", None)
        c["state"] = "takes" if c.get("takes") else None
    elif op == "delete_mask":
        mid = body.get("mask")
        plan["masks"] = [m for m in plan.get("masks", []) if m.get("id") != mid]
    elif op in ("add_object_keep", "set_object_keep"):
        # an object kept from the source (swap_plan.object_keeps): {"target", "range": [a, b] | "chunk", "grow",
        # "feather", "level", "on"}; set_object_keep names its "id" and changes only what it is given
        keeps = plan.setdefault("object_keeps", [])
        if op == "add_object_keep":
            ids = [int(k["id"][1:]) for k in keeps if str(k.get("id", "")).startswith("k") and k["id"][1:].isdigit()]
            k = {"id": f"k{(max(ids) if ids else 0) + 1:03d}"}
        else:
            k = next((x for x in keeps if x.get("id") == body.get("id")), None)
            if k is None:
                raise sp.PlanError(f"no object keep {body.get('id')!r}")
        if "target" in body or op == "add_object_keep":
            t = (body.get("target") or "").strip()
            if not t:
                raise sp.PlanError("an object keep needs a target: what to keep, e.g. \"white shoe\"")
            if t == (plan.get("target") or "").strip():
                raise sp.PlanError(f"{t!r} is the job's replace target: an object keep names something else")
            k["target"] = t
        if body.get("chunk"):
            k["range"] = list(_chunk(plan, body["chunk"])["deliver"])
        elif "range" in body or op == "add_object_keep":
            r = body.get("range") or [0, int(plan["source"]["frames"]) - 1]
            a, b = int(r[0]), int(r[1])
            if not 0 <= a <= b < int(plan["source"]["frames"]):
                raise sp.PlanError(f"object keep range {a}-{b} is outside the source")
            k["range"] = [a, b]
        for key, lo, hi in (("grow", 0, 200), ("feather", 0, 100)):
            if key in body:
                k[key] = max(lo, min(hi, int(body[key])))
        if "level" in body:
            if body["level"] not in sp.KEEP_LEVELS:
                raise sp.PlanError(f"level is one of {', '.join(sp.KEEP_LEVELS)}")
            k["level"] = body["level"]
        if "on" in body:
            k["on"] = bool(body["on"])
        if op == "add_object_keep":
            keeps.append(k)
        res["keep"] = dict(k)
    elif op == "set_object":
        # an object job's settings (swap_plan.object_settings): {"action": "replace"|"add", "grow", "feather", "level"}
        o = dict(plan.get("object") or {})
        if "action" in body:
            if body["action"] not in sp.OBJECT_ACTIONS:
                raise sp.PlanError(f"action is one of {', '.join(sp.OBJECT_ACTIONS)}")
            o["action"] = body["action"]
        for key, lo, hi in (("grow", 0, 400), ("feather", 0, 100), ("paste_grow", 0, 200)):
            if key in body:
                o[key] = max(lo, min(hi, int(body[key])))
        for key, lo, hi in (("size", 0.1, 5.0), ("sink", -0.5, 0.8), ("light", 0.2, 5.0), ("shadow", 0.0, 1.0),
                            ("colour", 0.0, 1.0), ("soften", 0.0, 10.0), ("crop_pad", 0.0, 2.0)):
            if key in body:
                o[key] = max(lo, min(hi, float(body[key])))
        if "grain" in body:
            o["grain"] = bool(body["grain"])
        if "guide" in body:
            if body["guide"] not in sp.OBJECT_GUIDES:
                raise sp.PlanError(f"guide is one of {', '.join(sp.OBJECT_GUIDES)}")
            o["guide"] = body["guide"]
        if "from_take" in body:
            o["from_take"] = (body.get("from_take") or "").strip()
        if "paste" in body:
            if body["paste"] not in ("mask", "frame"):
                raise sp.PlanError("paste is mask or frame")
            o["paste"] = body["paste"]
        if "text" in body:
            o["text"] = (body.get("text") or "").strip()
        for key in ("use_picture", "keep_people", "crop", "zone_spare"):
            if key in body:
                o[key] = bool(body[key])
        for key, lo, hi in (("zone_aspect", 0.1, 10.0), ("zone_dx", -5.0, 5.0), ("zone_dy", -5.0, 5.0),
                            ("zone_touch_dx", -5.0, 5.0), ("zone_touch_dy", -5.0, 5.0), ("zone_touch_r", 0.0, 3.0),
                            ("anchor_at", 0.0, 600.0), ("enter_at", 0.0, 600.0)):
            if key in body:
                o[key] = max(lo, min(hi, float(body[key])))
        if "zone_grow" in body:
            o["zone_grow"] = max(0, min(400, int(body["zone_grow"])))
        if "brief" in body:                            # what to add, in the user's words: the drafter writes both prompts from it
            o["brief"] = " ".join(str(body.get("brief") or "").split())[:600]
        if "zone_keys" in body:                        # the zone locked to a spot from a frame on (swap_plan.zone_at)
            lim = {"size": (0.1, 5.0), "zone_aspect": (0.1, 10.0), "zone_dx": (-5.0, 5.0), "zone_dy": (-5.0, 5.0),
                   "zone_touch_dx": (-5.0, 5.0), "zone_touch_dy": (-5.0, 5.0), "zone_touch_r": (0.0, 3.0)}
            keys = {}
            for k in body["zone_keys"] or []:
                fr = max(0, int(k["frame"]))
                keys[fr] = dict({"frame": fr}, **{f: max(lim[f][0], min(lim[f][1], float(k[f]))) for f in sp.ZONE_KEY_FIELDS if f in k})
            o["zone_keys"] = [keys[f] for f in sorted(keys)]
        if "zone_shape" in body:
            if body["zone_shape"] not in sp.ZONE_SHAPES:
                raise sp.PlanError(f"zone_shape is one of {', '.join(sp.ZONE_SHAPES)}")
            o["zone_shape"] = body["zone_shape"]
        if "anchor_picture" in body:
            o["anchor_picture"] = (body.get("anchor_picture") or "").strip()
        if "anchor" in body:
            if body["anchor"] not in sp.ANCHOR_MODES:
                raise sp.PlanError(f"anchor is one of {', '.join(sp.ANCHOR_MODES)}")
            o["anchor"] = body["anchor"]
        if "level" in body:
            if body["level"] not in sp.KEEP_LEVELS:
                raise sp.PlanError(f"level is one of {', '.join(sp.KEEP_LEVELS)}")
            o["level"] = body["level"]
        if "fill" in body:
            if body["fill"] not in sp.OBJECT_FILLS:
                raise sp.PlanError(f"fill is one of {', '.join(sp.OBJECT_FILLS)}")
            o["fill"] = body["fill"]
        if "picture" in body:
            o["picture"] = (body.get("picture") or "").strip()
        plan["object"] = o
        res["object"] = sp.object_settings(plan)
    elif op == "remove_object_keep":
        plan["object_keeps"] = [k for k in plan.get("object_keeps") or [] if k.get("id") != body.get("id")]
    elif op == "clear_all":
        # start fresh: every cut, split, kept stretch and prompt goes (one chunk over the whole source).
        # Nothing is erased: chunks with takes or prompts are listed under removed_chunks (their take
        # files stay on disk), and the cached masks stay (they're per source frame, still valid).
        busy = [c["id"] for c in plan.get("chunks", []) if c.get("state") == "rendering"]
        if busy:
            raise sp.PlanError(f"chunk {', '.join(busy)} is rendering: clear once it's in")
        gone = [c for c in plan.get("chunks", []) if c.get("takes") or (c.get("prompt") or "").strip()]
        if gone:
            plan.setdefault("removed_chunks", []).extend(dict(c, removed=sp.now(), reason="clear all") for c in gone)
        res["cleared"] = {"cuts": len(plan.get("cuts", [])), "splits": len(plan.get("splits", [])),
                          "kept": sum(1 for c in plan.get("chunks", []) if sp.is_kept(c)),
                          "prompts": sum(1 for c in plan.get("chunks", []) if (c.get("prompt") or "").strip()),
                          "takes_listed": sum(len(c.get("takes") or []) for c in gone),
                          "masks_kept": len(plan.get("masks") or [])}
        plan["cuts"], plan["splits"], plan["chunks"] = [], [], []
        geo = True
    elif op == "keep":
        # keep the original (§4.10): {"chunk", "keep": bool}, or a trim handle {"edge": "start"|"end",
        # "frame": J (0 or the frame count = no trim), "mode": optional}
        if body.get("edge"):
            res.update(_keep_edge(plan, body["edge"], body.get("frame"), body.get("mode")))
        else:
            c = _chunk(plan, body["chunk"])
            if bool(body.get("keep", True)):
                if c.get("state") == "rendering":
                    raise sp.PlanError(f"chunk {c['id']} is rendering: keep it once the render is in")
                c["keep"] = True
            else:
                c.pop("keep", None)
            res["chunk"] = c["id"]
        geo = True
    else:
        raise sp.PlanError(f"unknown op {op!r}")
    if geo:
        res["warnings"] = sp.rebuild_chunks(plan)
    return res


def _new_chunk_record(plan, left, **kw):
    c = {"id": sp._next_id(plan, "chunk", "c"), "left": left, "prompt": "", "prompt_state": "empty", "draft": None,
         "options": {"mark": True}, "seed_mode": "new", "chosen": None, "takes": []}
    c.update(kw)
    return c


def _keep_edge(plan, edge, frame, mode=None):
    """A trim handle (§4.10): keep the source's start (or end) up to (or from) frame J as the original,
    in one write. A split goes at J (a straight cut, snapped to a confirmed cut within a frame, unless
    `mode` says otherwise); the outer piece is kept. Dragging an existing handle moves its split;
    J = 0 (start) or the frame count (end) removes the trim and gives the frames back to the next chunk."""
    if edge not in ("start", "end"):
        raise sp.PlanError("edge is 'start' or 'end'")
    n = int(plan["source"]["frames"])
    chunks = plan.get("chunks") or []
    if not chunks:
        sp.rebuild_chunks(plan)
        chunks = plan["chunks"]
    J = None if frame is None else int(frame)
    off = J is None or (edge == "start" and J <= 0) or (edge == "end" and J >= n)
    outer = chunks[0] if edge == "start" else chunks[-1]
    has = sp.is_kept(outer) and len(chunks) > 1
    splits = {d["id"]: d for d in plan.get("splits", [])}
    if off:
        if not has:
            if sp.is_kept(outer):
                outer.pop("keep", None)
            return {"edge": edge, "trim": None}
        if edge == "start":                       # the next chunk takes the frames back, its data kept
            nxt = chunks[1]
            d = splits[nxt["left"]]
            plan["splits"] = [x for x in plan["splits"] if x is not d]
            plan["chunks"] = [c for c in chunks if c is not outer]
            nxt["left"] = None
        else:
            d = splits[outer["left"]]
            plan["splits"] = [x for x in plan["splits"] if x is not d]
            plan["chunks"] = [c for c in chunks if c is not outer]
        return {"edge": edge, "trim": None}
    if not 0 < J < n:
        raise sp.PlanError(f"frame {J} is outside the video (1..{n - 1})")
    cuts = sp.confirmed_cuts(plan.get("cuts"))
    near = [c for c in cuts if abs(c - J) <= 1]
    if near:
        J = near[0]
    want_mode = mode or sp.MODE_CUT
    if has:                                       # move the handle's split
        d = splits[chunks[1]["left"]] if edge == "start" else splits[outer["left"]]
        others = [x["frame"] for x in plan["splits"] if x is not d]
        if edge == "start" and any(f <= J for f in others):
            raise sp.PlanError(f"the start trim can't reach {J}: a split sits at or before it ({min(others)})")
        if edge == "end" and any(f >= J for f in others):
            raise sp.PlanError(f"the end trim can't reach {J}: a split sits at or after it ({max(others)})")
        d["frame"] = J
        if mode:
            d["mode"] = mode
        plan["splits"].sort(key=lambda x: x["frame"])
        return {"edge": edge, "trim": J, "split": d["id"]}
    existing = next((x for x in plan.get("splits", []) if x["frame"] == J), None)
    if edge == "start":
        if existing is None and any(x["frame"] < J for x in plan.get("splits", [])):
            raise sp.PlanError(f"the start trim at {J} would pass split(s) before it: move or delete them first")
        if existing is not None:                  # the first chunk already ends there: keep it
            outer["keep"] = True
            return {"edge": edge, "trim": J, "split": existing["id"]}
        d = sp.add_split(plan, J, want_mode)
        # the rendered remainder keeps the old first chunk's data (prompt, takes); the kept piece is new
        outer["left"] = d["id"]
        plan["chunks"] = [_new_chunk_record(plan, None, keep=True)] + [c for c in chunks]
    else:
        if existing is None and any(x["frame"] > J for x in plan.get("splits", [])):
            raise sp.PlanError(f"the end trim at {J} would pass split(s) after it: move or delete them first")
        d = existing or sp.add_split(plan, J, want_mode)
        hit = next((c for c in chunks if c.get("left") == d["id"]), None)
        if hit is None:
            plan["chunks"] = chunks + [_new_chunk_record(plan, d["id"], keep=True)]
        else:
            hit["keep"] = True
    return {"edge": edge, "trim": J, "split": d["id"]}


def draft_chunks(plan, which=None):
    """The chunks a draft run writes prompts for: never a kept chunk (§4.10). which: chunk ids, or
    None for every rendered chunk."""
    return [c for c in plan.get("chunks", []) if not sp.is_kept(c) and (which is None or c["id"] in which)]


# ---------------------------------------------------------------------------
# edits that move files (run under the plan lock, inside update_plan)
# ---------------------------------------------------------------------------

def _to_trash(jd, rel_dir, label):
    """Move a job sub-folder into trash/ (never erased by a node). Returns its new relative path."""
    src = os.path.join(jd, rel_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(jd, "trash", f"{label}_{stamp}")
    i = 1
    while os.path.exists(dst):
        dst = os.path.join(jd, "trash", f"{label}_{stamp}_{i}")
        i += 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.move(src, dst)
    return os.path.relpath(dst, jd).replace("\\", "/")


def _op_delete_take(plan, body, jd):
    c, t = sp.find_take(plan, body["take"])
    if (c.get("rendering") or {}).get("take") == t["id"]:
        raise sp.PlanError(f"{t['id']} is rendering")
    rel = os.path.dirname(t["file"])
    where = _to_trash(jd, rel, f"{c['id']}-{os.path.basename(rel)}")
    c["takes"] = [x for x in c["takes"] if x is not t]
    if c.get("chosen") == t["id"]:
        c["chosen"] = None
    plan.setdefault("trash", []).append({"kind": "take", "chunk": c["id"], "take": t, "dir": rel, "trash": where,
                                         "trashed": sp.now()})
    return {"op": "delete_take", "take": t["id"], "trash": where}


def _op_restore_take(plan, body, jd):
    tid = body["take"]
    hit = [e for e in plan.get("trash", []) if e.get("kind") == "take" and e["take"]["id"] == tid]
    if not hit:
        raise sp.PlanError(f"take {tid} isn't in the job's trash")
    e = hit[-1]
    c = sp.find_chunk(plan, e["chunk"])
    if any(t["id"] == tid for t in c.get("takes", [])):
        raise sp.PlanError(f"chunk {c['id']} already has a take {tid}")
    dst = os.path.join(jd, e["dir"])
    if os.path.exists(dst):
        raise sp.PlanError(f"{e['dir']} is in use: can't restore {tid} there")
    src = os.path.join(jd, e["trash"])
    if os.path.isdir(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    c.setdefault("takes", []).append(e["take"])
    c["takes"].sort(key=lambda t: t["id"])
    plan["trash"] = [x for x in plan["trash"] if x is not e]
    return {"op": "restore_take", "take": tid}


def _op_use_take(plan, body, jd):
    """'use this prompt and seed': the take's prompt into its chunk, the seed fixed to the take's."""
    c, t = sp.find_take(plan, body["take"])
    pf = os.path.join(jd, os.path.dirname(t["file"]), "prompt.txt")
    if not os.path.isfile(pf):
        raise sp.PlanError(f"{t['id']} has no prompt.txt")
    with open(pf, encoding="utf-8") as f:
        c["prompt"] = f.read()
    c["prompt_state"] = "edited" if c["prompt"].strip() else "empty"
    c["seed_mode"] = {"fixed": int(t["seed"])}
    return {"op": "use_take", "take": t["id"], "seed": int(t["seed"])}


def _op_delete_mask(plan, body, jd):
    hit = [m for m in plan.get("masks", []) if m.get("id") == body.get("mask")]
    if not hit:
        raise sp.PlanError(f"no mask {body.get('mask')!r}")
    m = hit[0]
    rel = os.path.dirname(m["file"])
    where = _to_trash(jd, rel, f"mask-{m['id']}")
    plan["masks"] = [x for x in plan["masks"] if x is not m]
    plan.setdefault("trash", []).append({"kind": "mask", "mask": m, "dir": rel, "trash": where, "trashed": sp.now()})
    return {"op": "delete_mask", "mask": m["id"], "trash": where}


def _stash_takes(c):
    """A chunk's takes out of play: listed under past_takes (next_take_id counts those, so no id is reused)."""
    have = {t["id"] for t in c.get("past_takes", [])}
    c.setdefault("past_takes", []).extend(t for t in c.get("takes") or [] if t["id"] not in have)
    c["takes"] = []


# ---------------------------------------------------------------------------
# a chunk's take as that chunk's own source. The job keeps ONE source file (everything downstream reads it), so the
# edit is a patched copy of the original: lossless, the original's audio, and the chosen chunks' frames replaced by
# the take's. Built once per set of edits (kept by a hash of them: flipping back and forth reuses the files), with
# no joins, grades or fades: that is assembly's work, on the takes, as ever.
# ---------------------------------------------------------------------------

def _patched_key(orig, overrides):
    h = hashlib.sha1(json.dumps([orig["path"], orig.get("mtime_ns"), orig.get("size"),
                                 sorted((cid, o["file"], o["render"][0], o["range"]) for cid, o in overrides.items())],
                                default=str).encode()).hexdigest()
    return h[:12]


def build_patched_source(jd, orig, overrides, progress=None):
    """The original with each override's frames (take file, offset = its render start) in its range. Returns the new
    source facts; a file already built for the same edits is reused."""
    try:
        from . import swap_assemble as sa
    except ImportError:
        import swap_assemble as sa
    out_dir = os.path.join(jd, "source")
    os.makedirs(out_dir, exist_ok=True)
    key = _patched_key(orig, overrides)
    for f in os.listdir(out_dir):
        if f.startswith(f"patched_{key}.") and ".part" not in f:
            return source_facts(os.path.join(out_dir, f))
    fr, N = int(round(float(orig["fps"]))), int(orig["frames"])
    w, h = int(orig["width"]), int(orig["height"])
    spec = sa._format_spec("video/ffv1-mkv", 0, "yuv420p")
    final = os.path.join(out_dir, f"patched_{key}.{spec['extension']}")
    video_only = os.path.join(out_dir, f"patched_{key}.video.part.{spec['extension']}")
    rd_orig = sa.FrameReader(orig["path"], fr, 0, (w, h))
    spans = sorted((int(o["range"][0]), int(o["range"][1]), sa.FrameReader(os.path.join(jd, o["file"]), fr, int(o["render"][0]), (w, h)))
                   for o in overrides.values())
    enc = sa.StreamEncoder(video_only, w, h, fr, spec)
    try:
        for f in range(N):
            rd = next((r for a, b, r in spans if a <= f <= b), rd_orig)
            enc.write(rd.read(f))
            if progress and f % 50 == 0:
                progress(f, N)
        enc.close()
        if orig.get("audio"):
            sa._mux_audio(video_only, orig["path"], final, spec)
        else:
            os.replace(video_only, final)
    except BaseException:
        for p in (video_only, final):
            try:
                os.remove(p)
            except OSError:
                pass
        raise
    finally:
        if os.path.exists(video_only):
            try:
                os.remove(video_only)
            except OSError:
                pass
    facts = source_facts(final)
    if facts["frames"] != N:
        os.replace(final, final + ".refused")
        raise PlannerError(f"the patched source has {facts['frames']} frames, not {N}: refused")
    return facts


def do_chunk_source(body, out_dir=None):
    """op chunk_source {chunk, take?}: the chunk's take (default: the one in use) becomes the picture the chunk renders
    from, so the next change builds on it. op chunk_source_revert {chunk}: back to the original for that chunk. chunk
    "all": every chunk that has a take (or every edited chunk, for the revert) in one go. The
    chunk's earlier takes, prompt and masks are put aside (nothing erased) and come back on a revert. The patched file
    is built outside the plan lock (a minute for a long clip); the plan edit after it is quick."""
    jd, pp = job_paths(body.get("job", ""), out_dir)
    revert = body["op"] == "chunk_source_revert"
    plan = sp.load_plan(pp)
    if any(c.get("state") == "rendering" for c in plan.get("chunks", [])):
        raise sp.PlanError("a chunk is rendering: change a source once it's in")
    base = dict(plan.get("overrides") or {})
    orig = plan.get("source_orig") or plan["source"]
    mode = sp.job_mode(plan)
    every = body["chunk"] == "all"
    cids, skipped = [], {}
    if every:
        for c in plan.get("chunks", []):
            if revert:
                if c["id"] in base:
                    cids.append(c["id"])
            elif sp.is_kept(c):
                skipped[c["id"]] = "kept as the original"
            elif sp.effective_take(c, mode)[0] is None:
                skipped[c["id"]] = "no take"
            else:
                cids.append(c["id"])
        if not cids:
            raise sp.PlanError("no chunk to change" + (f" ({skipped})" if skipped else ""))
    else:
        cids = [_chunk(plan, body["chunk"])["id"]]
    for cid in cids:
        c = sp.find_chunk(plan, cid)
        if revert:
            if cid not in base:
                raise sp.PlanError(f"chunk {cid} renders from the original already")
            base.pop(cid)
            continue
        take = None
        if body.get("take") and not every:
            take = next((t for t in c.get("takes", []) if t["id"] == body["take"]), None)
        else:
            take = sp.effective_take(c, mode)[0]
        why = None
        if take is None:
            why = "has no take to use as its source"
        elif not sp.take_covers(take, *c["deliver"]):
            why = f"take {take['id']} renders {take['render']}, which doesn't cover the chunk {c['deliver']}"
        elif not os.path.isfile(os.path.join(jd, take["file"])):
            why = f"take {take['id']}'s file is missing"
        if why:
            if not every:
                raise sp.PlanError(f"chunk {cid} {why}")
            skipped[cid] = why
            continue
        base[cid] = {"take": take["id"], "file": take["file"], "render": list(take["render"]), "range": list(c["deliver"])}
    cids = [cid for cid in cids if revert or cid in base]
    facts = build_patched_source(jd, orig, base) if base else orig

    def upd(p):
        if (p.get("overrides") or {}) != (plan.get("overrides") or {}):
            raise sp.PlanError("another chunk's source changed while this one was built: try again")
        stash = p.setdefault("override_stash", {})
        for cid in cids:
            ch = sp.find_chunk(p, cid)
            r0, r1 = ch["render"]
            if revert:
                back = stash.pop(cid, {})
                mine = {t["id"] for t in back.get("takes") or []}                 # (older flips stashed the takes: they come back)
                ch["past_takes"] = [t for t in ch.get("past_takes", []) if t["id"] not in mine]
                ch["takes"] = sorted(list(back.get("takes") or []) + list(ch.get("takes") or []), key=lambda t: t["id"])
                if back.get("takes"):
                    ch["chosen"] = back.get("chosen")
                ch.update(draft=back.get("draft"), prompt=back.get("prompt") or "")
                if back.get("drafted"):                                   # the draft's own record (warnings, time) comes back too
                    ch["drafted"] = back["drafted"]
                ch["state"] = "takes" if ch["takes"] else None
                ch["prompt_state"] = back.get("prompt_state") or ("edited" if ch["prompt"].strip() else "empty")
                p["masks"] = (p.get("masks") or []) + [m for m in back.get("masks") or [] if m not in (p.get("masks") or [])]
            else:
                drop = [m for m in p.get("masks") or [] if m["range"][0] <= r1 and r0 <= m["range"][1]]
                # the takes stay listed and the choice stays (user: "not giving the option for the previous images"): only the
                # prompt, the draft and the cached masks (they outline the old picture) are put aside
                stash[cid] = {"takes": [], "chosen": ch.get("chosen"), "prompt": ch.get("prompt"),
                              "prompt_state": ch.get("prompt_state"), "draft": ch.get("draft"), "drafted": ch.get("drafted"),
                              "masks": drop,
                              "stashed": sp.now()}
                p["masks"] = [m for m in p.get("masks") or [] if m not in drop]
                ch.update(prompt="", prompt_state="empty", draft=None, drafted=None)
            for k in ("rendering", "error", "failed", "anchor_text"):
                ch.pop(k, None)
        if base:
            p["overrides"], p["source_orig"] = base, orig
            p["source"] = dict(facts, patched=True)
        else:
            p.pop("overrides", None)
            p.pop("source_orig", None)
            p["source"] = orig
        p["assembled"] = p.get("assembled") or []
        sp.rebuild_chunks(p)
    plan = sp.update_plan(pp, upd)
    return {"ok": True, "rev": plan["rev"], "result": {"op": body["op"], "chunks": cids, "skipped": skipped,
                                                       "overrides": sorted(base), "source": plan["source"]["path"]},
            "warnings": plan.get("warnings", []), "plan": plan, "text": plan_text(plan)}


FILE_OPS = {"delete_take": _op_delete_take, "restore_take": _op_restore_take, "use_take": _op_use_take,
            "delete_mask": _op_delete_mask}


# ---------------------------------------------------------------------------
# cut detection: "detect + plan" (the default) brings the cuts in confirmed and places the auto
# splits from them in one step, for one review; "detect" alone adds faint suggestions
# ---------------------------------------------------------------------------

def detect_scenes(path, fps, low=WEAK_THRESHOLD):
    """ffmpeg's scene score for every frame scoring > low, as [(frame, score)] (frame c = the first of a new shot,
    numbered on the decoder's timeline at `fps`, the frames a chunk is cut from). Every frame's time is read, so
    the score lands where the frame before it stops showing (tl.first_index_after), not at round(t * fps): a
    29.97 fps source planned at 30 had every cut past frame ~500 one frame late."""
    ff = tl._ffmpeg_exe()
    p = subprocess.run([ff, "-hide_banner", "-nostats", "-i", path, "-an", "-vf",
                        "select='gte(scene,0)',metadata=print:key=lavfi.scene_score", "-f", "null", "-"],
                       capture_output=True)
    if p.returncode != 0:
        raise PlannerError(f"cut detection failed: {p.stderr.decode(errors='replace')[-600:]}")
    info = tl.probe(path, fps)
    base, tb = float(info.get("base_time") or 0.0), info.get("time_base")
    out, t, prev = {}, None, None
    for line in p.stderr.decode(errors="replace").splitlines():
        m = re.search(r"pts_time:\s*(-?[0-9.]+)", line)
        if m:
            t = float(m.group(1))
            continue
        m = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if m and t is not None:
            sc = float(m.group(1))
            if prev is not None and sc > float(low):
                f = tl.first_index_after(prev, base, fps, tb)
                if f > 0:
                    out[f] = max(out.get(f, 0.0), sc)
            prev, t = t, None
    return sorted(out.items())


def split_scenes(scored, threshold=SCENE_THRESHOLD, low=WEAK_THRESHOLD, near=WEAK_NEAR):
    """(cuts, weak): frames over `threshold`, and frames in (low, threshold] not within `near` frames of a
    stronger hit (a cut read on two neighbouring frames is one cut)."""
    strong = [f for f, sc in scored if sc > threshold]
    weak = []
    for f, sc in scored:
        if low < sc <= threshold and not any(abs(f - g) <= near and sg > sc for g, sg in scored if g != f):
            weak.append((f, round(sc, 3)))
    return strong, weak


def detect_cuts(path, fps, threshold=SCENE_THRESHOLD):
    """ffmpeg's scene score > threshold, as source frame numbers (frame c = the first of a new shot)."""
    return split_scenes(detect_scenes(path, fps, min(threshold, WEAK_THRESHOLD)), threshold)[0]


def merge_detected(plan, found, confirmed=False, weak=()):
    """Detected cuts in, as suggestions (or confirmed). Confirmed or hand-placed cuts stay; earlier
    unconfirmed suggestions are replaced; a detection on a frame that already has a cut is skipped
    (confirmed=True confirms a suggestion already there). `weak`: [(frame, score)] under the threshold, always
    added as suggestions with their score, never confirmed here."""
    keep = [c for c in plan.get("cuts", []) if c.get("confirmed", True) or c.get("from") != "detected"
            or (confirmed and c["frame"] in found)]
    if confirmed:
        for c in keep:
            if c["frame"] in found:
                c["confirmed"] = True
    have = {c["frame"] for c in keep}
    added = [f for f in found if f not in have]
    new = [{"frame": f, "from": "detected", "confirmed": bool(confirmed)} for f in added]
    have |= set(added)
    new += [{"frame": int(f), "from": "detected", "confirmed": False, "weak": True, "score": float(sc)}
            for f, sc in weak if int(f) not in have]
    plan["cuts"] = sorted(keep + new, key=lambda c: c["frame"])
    sp.rebuild_chunks(plan)
    return added


def detect_and_plan(plan, found, weak=()):
    """One step: the detected cuts confirmed, then the auto splits placed from every confirmed cut.
    Splits are only replaced while no chunk has a prompt or a take; otherwise they're kept. Weak hits come in
    as suggestions only, so they never move the auto splits until the user confirms them."""
    added = merge_detected(plan, found, confirmed=True, weak=weak)
    busy = any(c.get("takes") or (c.get("prompt") or "").strip() for c in plan.get("chunks", []) if not sp.is_kept(c))
    res = {"added": added, "splits": None, "kept_splits": busy}
    if not busy:
        res["splits"] = apply_op(plan, {"op": "auto_splits"})["splits"]
    return res


def do_detect_cuts(body, out_dir=None):
    jd, pp = job_paths(body.get("job", ""), out_dir)
    plan = sp.load_plan(pp)
    src = plan["source"]
    thr = float(body.get("threshold", SCENE_THRESHOLD))
    low = float(body.get("weak_threshold", WEAK_THRESHOLD))
    found, weak = split_scenes(detect_scenes(src["path"], int(round(float(src["fps"]))), min(thr, low)), thr, low)
    out = {"added": [], "splits": None, "kept_splits": False}
    if body.get("plan"):
        plan = sp.update_plan(pp, lambda p: out.update(detect_and_plan(p, found, weak)), body.get("rev"))
    else:
        plan = sp.update_plan(pp, lambda p: out["added"].extend(merge_detected(p, found, weak=weak)), body.get("rev"))
    return dict(out, ok=True, found=found, weak=[f for f, _ in weak], rev=plan["rev"])


def list_jobs(out_dir=None):
    """Every job under output/seamstitch_swap/, newest first: name, source, frames, chunks, takes."""
    base = os.path.join(out_dir or output_dir(), sp.JOBS_SUBDIR)
    jobs = []
    if not os.path.isdir(base):
        return jobs
    for name in os.listdir(base):
        pp = sp.plan_path(os.path.join(base, name))
        if not os.path.isfile(pp):
            continue
        try:
            p = sp.load_plan(pp)
        except sp.PlanError:
            continue
        src = p.get("source") or {}
        jobs.append({"job": name, "source": src.get("path"), "frames": src.get("frames"), "fps": src.get("fps"),
                     "chunks": len(p.get("chunks") or []), "takes": sum(len(c.get("takes") or []) for c in p.get("chunks") or []),
                     "saved": p.get("saved") or p.get("created"), "mtime": os.path.getmtime(pp)})
    jobs.sort(key=lambda j: j["mtime"], reverse=True)
    return jobs


def free_job_name(stem, out_dir=None):
    """A job name for a video: its stem made folder-safe, with _2, _3... when that job exists."""
    base = re.sub(r"[^\w.-]+", "_", stem).strip("._") or "swap_job"
    have = {j["job"] for j in list_jobs(out_dir)}
    name, i = base, 2
    while name in have:
        name = f"{base}_{i}"
        i += 1
    return name


def mark_size(source):
    """The size a mark run's mask comes out at: the render group scales the source to 1 MP (2^20 pixels) for SAM3."""
    s = (float(2 ** 20) / (int(source["width"]) * int(source["height"]))) ** 0.5
    return int(round(int(source["width"]) * s)), int(round(int(source["height"]) * s))


def do_empty_mask(body, out_dir=None):
    """A "nobody here" mark over a range (the strip's Mark menu): an empty mask, cached as a mark run's is, for a shot
    the target isn't in. SAM3 run on such a shot marks something anyway (test clip, 2026-10-17: the dark background
    behind a held ticket, a shadow on a wall, both of which the render would have repainted as the character); with
    an empty mark the shot is left as filmed. Sized like the job's other masks, so the stitched cover holds."""
    try:
        from . import swap_mask as sm       # swap_mask imports this module
    except ImportError:
        import swap_mask as sm
    jd, pp = job_paths(body.get("job", ""), out_dir)
    plan = sp.load_plan(pp)
    a, b = int(body["range"][0]), int(body["range"][1])
    if not 0 <= a <= b < int(plan["source"]["frames"]):
        raise sp.PlanError(f"range {a}-{b} is outside the source (0-{int(plan['source']['frames']) - 1})")
    sizes = [tuple(m["size"]) for m in plan.get("masks", []) if m.get("size")]
    w, h = sizes[-1] if sizes else mark_size(plan["source"])
    c = sp.find_chunk(plan, body["chunk"]) if body.get("chunk") else \
        next((x for x in plan["chunks"] if x["deliver"][0] <= a <= x["deliver"][1]), None)
    n = b - a + 1
    mark = {"format": MARK_FORMAT, "plan": pp, "job": plan["job"], "range": [a, b], "frames": n,
            "source": plan["source"], "chunk": c["id"] if c else None,
            "target": sp.chunk_target(plan, c) if c else (plan.get("target") or ""), "nonce": uuid.uuid4().hex,
            "started": time.time()}
    entry = sm.save_mask(mark, torch.zeros((n, h, w)), save_preview=False, fill_holes_frames=0)
    plan = sp.load_plan(pp)
    return {"ok": True, "rev": plan["rev"], "result": {"op": "empty_mask", "mask": entry["id"]},
            "warnings": plan.get("warnings", []), "plan": plan, "text": plan_text(plan)}


UNDO_KEEP = 50
# Not edits of the strip: they never go on the undo stack.
UNDO_SKIP = {"create", "undo", "render_failed", "rescore", "empty_mask", "clear_state", "source"}


def _undo_file(jd):
    return os.path.join(jd, "undo.json")


def undo_stack(jd):
    try:
        with open(_undo_file(jd), encoding="utf8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except (OSError, ValueError):
        return []


def _undo_save(jd, stack):
    sp._atomic_write(_undo_file(jd), json.dumps(stack[-UNDO_KEEP:], indent=1))


def _op_label(body):
    """A short line for the undo button: what the edit was."""
    bits = [str(body.get("op"))]
    for k in ("split", "chunk", "cut", "mask", "take", "mode", "frame", "to"):
        if body.get(k) is not None:
            bits.append(f"{k} {body[k]}")
    return " ".join(bits)[:120]


def do_undo(jd, pp):
    """Put the plan back as it was before the last strip edit: the plan saved in history/ before that edit, with every take the
    chunks have now (renders since stay; a chunk the old plan did not have is kept under removed_chunks, as delete_split does)."""
    stack = undo_stack(jd)
    if not stack:
        raise PlannerError("nothing to undo")
    ent = stack[-1]
    hf = os.path.join(jd, "history", f"plan_{ent['rev']}.json")
    if not os.path.isfile(hf):
        raise PlannerError(f"revision {ent['rev']} is no longer in history (only the last {sp.HISTORY_KEEP} saves are kept)")
    with open(hf, encoding="utf8") as f:
        prev = json.load(f)

    def upd(p):
        cur = {c["id"]: c for c in p.get("chunks", [])}
        for c in prev.get("chunks", []):
            cc = cur.pop(c["id"], None)
            if cc is not None:
                c["takes"] = cc.get("takes") or []
                ids = {t["id"] for t in c["takes"]}
                if c.get("chosen") not in ids:
                    c["chosen"] = cc.get("chosen") if cc.get("chosen") in ids else c.get("chosen")
        for cc in cur.values():
            if cc.get("takes"):
                prev.setdefault("removed_chunks", []).append(dict(cc, removed=sp.now()))
        prev["rev"] = p.get("rev", 0)
        return prev
    plan = sp.update_plan(pp, upd)
    _undo_save(jd, stack[:-1])
    return {"ok": True, "rev": plan["rev"], "result": {"op": "undo", "undid": ent.get("label"), "restored_rev": ent["rev"]},
            "warnings": plan.get("warnings", []), "plan": plan, "text": plan_text(plan)}


def do_op(body, out_dir=None):
    """The op route's body: create a job, or apply one edit under the plan lock."""
    job = body.get("job", "")
    jd, pp = job_paths(job, out_dir)
    if body.get("op") == "create":
        plan = create_job(job, body.get("source", ""), body.get("settings"), out_dir)
        return {"ok": True, "rev": plan["rev"], "plan": plan, "text": plan_text(plan)}
    if body.get("op") == "undo":
        return do_undo(jd, pp)
    if body.get("op") in ("chunk_source", "chunk_source_revert"):
        return do_chunk_source(body, out_dir)
    if body.get("op") == "rescore":
        return do_rescore(body, out_dir)
    if body.get("op") == "empty_mask":
        return do_empty_mask(body, out_dir)
    if body.get("op") == "source":
        def upd(p):
            if any(c.get("takes") for c in p.get("chunks", [])):
                raise sp.PlanError("the job has takes: start a new job for another source")
            p["source"] = source_facts(body.get("source", ""))
            sp.rebuild_chunks(p)
        plan = sp.update_plan(pp, upd, body.get("rev"))
        return {"ok": True, "rev": plan["rev"], "plan": plan, "text": plan_text(plan)}
    out = {}
    before = {}

    def upd(p):
        before["rev"] = p.get("rev")
        if body.get("op") in FILE_OPS:
            out.update(FILE_OPS[body["op"]](p, body, jd))
        else:
            out.update(apply_op(p, body))
    plan = sp.update_plan(pp, upd, body.get("rev"))
    if body.get("op") not in UNDO_SKIP and before.get("rev") is not None:
        _undo_save(jd, undo_stack(jd) + [{"rev": before["rev"], "op": body.get("op"), "label": _op_label(body), "at": sp.now()}])
    return {"ok": True, "rev": plan["rev"], "result": out, "warnings": plan.get("warnings", []), "plan": plan,
            "text": plan_text(plan)}


_ZONE_GUIDES = {}


def do_zone_preview(job, frame, out_dir=None):
    """Where an object add's zone sits on one source frame, in source pixels, for the strip's zone editor: the zone's shape and
    geometry, the touch circle, the target's anchor point and `fw` (one target-width in px: the unit of every zone number). The
    guide (the target's cached masks over the chunk's render range) is built once per chunk and masks version; the settings are read
    fresh, so a number typed in the strip moves the picture without a rebuild."""
    try:
        from . import swap_keep as sk
    except ImportError:
        import swap_keep as sk
    jd, pp = job_paths(job, out_dir)
    plan = sp.load_plan(pp)
    if not sp.object_zone(plan):
        raise PlannerError("the job is not an object add that redraws a zone")
    frame = int(frame)
    c = next((x for x in plan.get("chunks", []) if x["deliver"][0] <= frame <= x["deliver"][1]), None)
    if c is None:
        raise PlannerError(f"frame {frame} is in no chunk")
    src = plan["source"]
    fr, (w, h) = int(round(float(src["fps"]))), (int(src["width"]), int(src["height"]))
    target = (sp.chunk_target(plan, c) or "").strip()
    masks_key = tuple((m.get("id"), m.get("created")) for m in plan.get("masks") or [])
    key = (os.path.abspath(jd), c["id"], target, tuple(c["render"]), masks_key)
    g = _ZONE_GUIDES.get(key)
    if g is None:
        if len(_ZONE_GUIDES) > 8:
            _ZONE_GUIDES.clear()
        g = _ZONE_GUIDES[key] = sk.ZoneGuide(plan, jd, fr, (w, h), int(c["render"][0]), int(c["render"][1]), target)
    g.o = sp.object_settings(plan)
    o = g.o
    out = {"ok": True, "frame": frame, "chunk": c["id"], "w": w, "h": h, "target": target, "found": False,
           "settings": {k: o.get(k) for k in ("size", "zone_aspect", "zone_grow", "zone_dx", "zone_dy", "zone_shape", "zone_spare",
                                              "zone_touch_dx", "zone_touch_dy", "zone_touch_r", "sink", "anchor_at", "enter_at")}}
    ok = sp.zone_key_at(o, frame)
    out["keys"] = [k["frame"] for k in sorted(o.get("zone_keys") or [], key=lambda x: x["frame"])]
    out["key"] = ok["frame"] if ok else None
    out["settings"].update({f: sp.zone_at(o, frame).get(f) for f in sp.ZONE_KEY_FIELDS})
    a = g.anchors.get(frame)
    if a is None or not g.first_width:
        out["why"] = ("no mask of " + (repr(target) if target else "the target") + " on this frame: mark the chunk first"
                      if not g.anchors else "the target is not found on this frame")
        return out
    fw = float(g.first_width)
    out.update(found=True, fw=fw, anchor=[float(a[0]), float(a[1]), float(a[2])])
    if o.get("zone_shape") == "mask":
        out["zone"] = {"shape": "mask"}
    else:
        oe = sp.zone_at(o, frame)
        cx, cy, rx, ry = sk.zone_geometry(a, fw, oe)
        out["zone"] = {"shape": o.get("zone_shape") or "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry}
    oe = sp.zone_at(o, frame)
    r = float(oe.get("zone_touch_r", 0.0))
    if r > 0:
        out["touch"] = {"x": a[0] + float(oe.get("zone_touch_dx", 0.0)) * fw, "y": a[1] + float(oe.get("zone_touch_dy", 0.0)) * fw, "r": r * fw}
    return out



def do_rescore(body, out_dir=None):
    """Recompute the scores of saved takes from their files (CPU, seconds per take; outside the
    plan lock, which only guards each write): body take = one take, chunk = its takes, or neither =
    every take of the job. Takes saved before B3 carry only pose IoU."""
    try:
        from . import swap_take as stk
    except ImportError:
        import swap_take as stk
    jd, pp = job_paths(body.get("job", ""), out_dir)
    plan = sp.load_plan(pp)
    if body.get("take"):
        ids = [body["take"]]
    else:
        ids = [t["id"] for c in plan.get("chunks", []) if not body.get("chunk") or c["id"] == body["chunk"]
               for t in c.get("takes") or []]
    done, failed = {}, {}
    for tid in ids:
        try:
            done[tid] = stk.rescore_take(pp, tid, mouth=bool(body.get("mouth", True)))
        except Exception as e:
            failed[tid] = str(e)
    plan = sp.load_plan(pp)
    return {"ok": not failed, "rev": plan["rev"], "result": {"op": "rescore", "scored": sorted(done), "failed": failed},
            "warnings": plan.get("warnings", []), "plan": plan, "text": plan_text(plan)}


# ---------------------------------------------------------------------------
# the read-only summary (the panel until B2's strip)
# ---------------------------------------------------------------------------

def plan_text(plan):
    src = plan.get("source") or {}
    lines = [f"job {plan.get('job')} · {os.path.basename(src.get('path', '?'))} · {src.get('frames')} frames @ "
             f"{src.get('fps')} fps · rev {plan.get('rev')}"]
    if plan.get("cuts"):
        lines.append("cuts: " + " ".join(str(c["frame"]) + ("" if c.get("confirmed", True) else "?")
                                         for c in plan["cuts"]))
    warn_splits = {w.get("split") for w in plan.get("warnings", []) if w.get("split")}
    if plan.get("splits"):
        lines.append("splits: " + " · ".join(
            f"{d['id']} {d['frame']} {'cut' if d['mode'] == sp.MODE_CUT else 'anch'}"
            + (" ⚠" if d["id"] in warn_splits else "") for d in plan["splits"]))
    lines.append("chunks:")
    for c in plan.get("chunks", []):
        t, state = sp.effective_take(c, sp.job_mode(plan))
        fill = c.get("fill")
        fill_s = f" +{fill['kind']} {fill['frames']}" if fill else ""
        takes = ", ".join(x["id"].split("-")[-1] + ("*" if x["id"] == c.get("chosen") else "")
                          for x in c.get("takes", [])) or "-"
        if sp.is_kept(c):
            lines.append(f" {c['id']} {c['deliver'][0]}-{c['deliver'][1]} · original (kept: never rendered)")
            continue
        pr = "prompt ✓" if (c.get("prompt") or "").strip() else "NO PROMPT"
        st = c.get("state") if c.get("state") == "rendering" else state
        lines.append(f" {c['id']} {c['deliver'][0]}-{c['deliver'][1]} · render {c['render'][0]}-{c['render'][1]}"
                     f"{fill_s} = {c.get('length')} · {pr} · takes {takes} · {st}")
    try:
        joins = sp.plan_joins(plan)
    except sp.PlanError as e:
        joins = []
        lines.append(f"joins: {e}")
    if joins:
        lines.append("joins:")
        for j in joins:
            lines.append(f" {j['split']} @{j['frame']}: {j['type']}"
                         + (f" {j['left_take']} | {j['right_take']}, splice {j['splice']}, {j['repair']}"
                            if j["type"] not in (sp.STRAIGHT, sp.PENDING, sp.ORIGINAL) else "")
                         + (f" (original on the {j['original']})" if j.get("original") in ("left", "right") else ""))
    for w in plan.get("warnings", []):
        lines.append(f" ⚠ {w['text']}")
    return "\n".join(lines)


_EMPTY_CACHE = {}


def mask_empty_frames(jd, seg, fps):
    """Source frames of a cached mask segment where the mask is empty (SAM3 lost the person). Swap
    Mask records them; a segment cached before it did is scanned once per file version."""
    if "empty" in seg:
        return seg["empty"]
    path = os.path.join(jd, seg["file"])
    try:
        st = os.stat(path)
    except OSError:
        return []
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key not in _EMPTY_CACHE:
        a = int(seg["range"][0])
        _EMPTY_CACHE[key] = [a + i for i, f in enumerate(tl._iter_frames(path, fps, 0, None)) if not (f[..., 0] > 127).any()]
    return _EMPTY_CACHE[key]


def chunk_status(plan, jd):
    """What the strip draws for each chunk (§5.1 UI states), from the plan and the files on disk:
    state = pending | unreviewed | chosen | range changed | missing file, plus rendering / failed
    when the plan says so, the takes count, and how much of its render range the mask cache covers."""
    out = []
    fps = int(round(float((plan.get("source") or {}).get("fps") or 25)))
    # a frame counts only if the segment that covers it (the newest, as renders read it) is empty there
    segs = sorted(plan.get("masks") or [], key=lambda m: str(m.get("created", "")), reverse=True)
    wins = lambda f, m: next(s for s in segs if s["range"][0] <= f <= s["range"][1]) is m  # noqa: E731
    empty = sorted(f for m in segs for f in mask_empty_frames(jd, m, fps) if wins(f, m))
    filled = sorted(f for m in segs for f in m.get("filled") or [] if wins(f, m))
    for c in plan.get("chunks", []):
        if sp.is_kept(c):
            out.append({"chunk": c["id"], "state": "original", "keep": True, "take": None,
                        "takes": len(c.get("takes") or []), "prompt": "not needed", "draft": False, "mask": None,
                        "mask_empty": [], "mask_filled": [], "take_flags": {}, "rank": []})
            continue
        t, st = sp.effective_take(c, sp.job_mode(plan))
        if t is not None and not sp.take_covers(t, *c["deliver"]):
            st = "range changed"
        elif t is not None and not os.path.isfile(os.path.join(jd, t["file"])):
            st = "missing file"
        r0, r1 = c["render"]
        d = {"chunk": c["id"], "state": st, "take": t and t["id"], "takes": len(c.get("takes") or []),
             "prompt": "empty" if not (c.get("prompt") or "").strip() else c.get("prompt_state") or "edited",
             "draft": bool((c.get("draft") or "").strip()),
             "mask": round(sp.mask_coverage(plan, r0, r1, sp.chunk_target(plan, c)) / float(r1 - r0 + 1), 4),
             "mask_empty": [f for f in empty if r0 <= f <= r1], "mask_filled": [f for f in filled if r0 <= f <= r1]}
        if c.get("state") == "rendering":
            d["rendering"] = c.get("rendering")
        elif c.get("state") == "failed":
            d["failed"] = c.get("error") or "failed"
        if t is not None:
            d["scores"] = t.get("scores") or {}
            d["flags"] = ss.chunk_flags(d["scores"])
            d["proxy"] = t.get("proxy")
            d["render"] = t.get("render")
        # every take's flags, and the display-only order (following, then lost cuts; never a pick)
        takes = c.get("takes") or []
        d["take_flags"] = {x["id"]: ss.chunk_flags(x.get("scores")) for x in takes}
        d["rank"] = [x["id"] for x in sorted(takes, key=ss.rank_key)]
        out.append(d)
    return out


BROWSER_CODECS = {"h264", "vp8", "vp9", "av1"}      # what Chrome decodes in a <video>
_CODEC_CACHE = {}


def video_codec(path):
    """The source's video codec name (PyAV), cached per file version; None when unreadable."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _CODEC_CACHE:
        try:
            import av
            with av.open(path) as c:
                _CODEC_CACHE[key] = c.streams.video[0].codec_context.name
        except Exception:
            _CODEC_CACHE[key] = None
    return _CODEC_CACHE[key]


def source_view(plan, jd):
    """What the strip plays for the source: the source itself when the browser can decode its video, else
    an H.264 copy in the job's cache/ at the plan's frame rate (the frames the Planner counts), made once and
    remade when the source changes. A 2007 phone clip in MPEG-4 Part 2 played black in the strip."""
    src = (plan.get("source") or {}).get("path") or ""
    codec = video_codec(src)
    if not src or codec is None or codec in BROWSER_CODECS:
        return src
    out = os.path.join(jd, "cache", "source_view.mp4")
    if os.path.isfile(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        return out.replace("\\", "/")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fr = int(round(float(plan["source"]["fps"])))
    tmp = out + ".part.mp4"
    subprocess.run([tl._ffmpeg_exe(), "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", f"fps={fr},scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-g", str(fr), "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", tmp], check=True, capture_output=True)
    os.replace(tmp, out)
    _say(f"[SeamStitch] Swap Planner: {os.path.basename(src)} is {codec} (the browser can't play it): "
         f"made a preview copy for the strip")
    return out.replace("\\", "/")


def plan_view(job, out_dir=None):
    jd, pp = job_paths(job, out_dir)
    plan = sp.load_plan(pp)
    try:
        joins = sp.plan_joins(plan)
    except sp.PlanError:
        joins = []
    for j in joins:
        m = sp.join_measure(plan, j)
        j["measure"] = m
        j["flags"] = sp.join_flags(j, m)
        j["verdict"] = j["flags"]["verdict"]
    return {"path": pp, "job_dir": jd, "subfolder": os.path.relpath(jd, output_dir_or(out_dir)).replace("\\", "/"),
            "plan": plan, "joins": joins, "status": chunk_status(plan, jd), "warnings": plan.get("warnings", []),
            "quality": ss.legend(), "text": plan_text(plan), "source_view": _source_view_safe(plan, jd),
            "undo": (lambda st: {"n": len(st), "last": st[-1].get("label") if st else None})(undo_stack(jd))}


def _source_view_safe(plan, jd):
    try:
        return source_view(plan, jd)
    except Exception as e:
        _say(f"[SeamStitch] Swap Planner: no preview copy for the strip: {e}")
        return (plan.get("source") or {}).get("path")


def output_dir_or(out_dir):
    return out_dir or output_dir()


# ---------------------------------------------------------------------------
# a render run
# ---------------------------------------------------------------------------

def newest_take(chunk, mode, frames):
    """(take, "newest"): the chunk's most recent usable take in the job's mode that covers the frames, or (None,
    "pending"). The forward chain pins to it, chosen or not."""
    ok = [t for t in (chunk.get("takes") or []) if t.get("state", "ok") == "ok" and sp.take_mode(t) == mode
          and sp.take_covers(t, *frames)]
    if not ok:
        return None, "pending"
    return max(ok, key=lambda t: (t.get("created") or "", t.get("id") or "")), "newest"


def resolve_pins(plan, k):
    """Which pins chunk k gets, from the lineage at this moment: every anchored side whose
    neighbour has an effective take covering the K frames. {side: {"take", "frames"} | None},
    plus notes on the sides that get none."""
    chunks = plan["chunks"]
    c = chunks[k]
    K = int(sp._settings(plan.get("settings"))["anchors"])
    splits = {d["id"]: d for d in plan.get("splits", [])}
    pins, notes = {"start": None, "end": None}, []
    forward = bool(sp._settings(plan.get("settings")).get("forward_chain"))
    if K <= 0:
        return pins, ["anchor_frames is 0: no pins"]
    if sp.is_kept(c):
        return pins, ["kept as the original: never rendered"]
    r0, r1 = c["render"]
    left = splits.get(c.get("left"))
    if k > 0 and left and left["mode"] == sp.MODE_ANCHORED and sp.is_kept(chunks[k - 1]):
        pins["start"] = {"source": True, "frames": [r0, r0 + K - 1]}       # anchored onto the original
    elif k > 0 and left and left["mode"] == sp.MODE_ANCHORED:
        f = [r0, r0 + K - 1]
        t, _st = (newest_take(chunks[k - 1], sp.job_mode(plan), f) if forward
                  else sp.effective_take(chunks[k - 1], sp.job_mode(plan)))
        if t is not None and sp.take_covers(t, *f):
            pins["start"] = {"take": t["id"], "frames": f}
        else:
            notes.append(f"start: {chunks[k - 1]['id']} has no take covering {f[0]}-{f[1]}: free start")
    if forward:
        notes.append("end: forward chain is on: no end pin (the chunk after is rendered from this one)")
    elif k + 1 < len(chunks):
        right = splits.get(chunks[k + 1].get("left"))
        if right and right["mode"] == sp.MODE_ANCHORED:
            f = [r1 - K + 1, r1]
            t, _st = sp.effective_take(chunks[k + 1], sp.job_mode(plan))
            if int(c.get("held") or 0):
                notes.append("end: the render ends in held frames, so it can't be pinned at its end")
            elif sp.is_kept(chunks[k + 1]):
                pins["end"] = {"source": True, "frames": f}                    # anchored onto the original
            elif t is not None and sp.take_covers(t, *f):
                pins["end"] = {"take": t["id"], "frames": f}
            elif t is not None and sp.take_covers(t, r1 + 1, r1 + K):
                # the take after was rendered from its own first frame (its left split was a cut then), so it has no
                # picture under this chunk's last frame: pin the end to the frames just after it, one frame late
                pins["end"] = {"take": t["id"], "frames": [r1 + 1, r1 + K]}
                notes.append(f"end: {chunks[k + 1]['id']}'s take starts after this chunk, so the end is pinned to the frame after it")
            elif t is not None:
                notes.append(f"end: {chunks[k + 1]['id']}'s take doesn't cover {f[0]}-{f[1]}: free end")
    return pins, notes


def render_descriptor(plan, plan_file, run, settings=None):
    """The render run's chunk descriptor (no decoding): what the Take node needs, and what the
    Planner decodes. Refuses an empty prompt."""
    c = _chunk(plan, run.get("chunk"))
    if sp.is_kept(c):
        raise PlannerError(f"chunk {c['id']} {c['deliver']} is kept as the original: it's never rendered "
                           f"(switch 'keep original' off in its panel to render it)")
    k = plan["chunks"].index(c)
    prompt = run["prompt"] if run.get("prompt") is not None else c.get("prompt", "")
    if not (prompt or "").strip():
        raise PlannerError(f"chunk {c['id']} {c['deliver']} has no prompt: a render needs one (write it in the plan)")
    seed = run.get("seed")
    if seed is None:
        m = c.get("seed_mode", "new")
        seed = int(m["fixed"]) if isinstance(m, dict) else random.randint(0, 2 ** 32 - 1)
    pins, notes = resolve_pins(plan, k)
    # the effective target and invert reach the render group as options ("person" when no one is named)
    options = dict(c.get("options") or {}, **(run.get("options") or {}))
    options.pop("target", None)
    if sp.chunk_target(plan, c):
        options["target"] = sp.chunk_target(plan, c)
    options.pop("invert", None)
    if sp.chunk_invert(plan, c):
        options["invert"] = True
    options["mode"] = sp.job_mode(plan)             # the take's mode, whichever way the guide marks
    splits = {d["id"]: d for d in plan.get("splits", [])}
    left = splits.get(c.get("left"))
    right = splits.get(plan["chunks"][k + 1].get("left")) if k + 1 < len(plan["chunks"]) else None
    s = sp._settings(plan.get("settings"))
    src = plan["source"]
    fr = int(round(float(src["fps"])))
    return {"format": CHUNK_FORMAT, "job": plan["job"], "plan": plan_file, "plan_rev": plan.get("rev"),
            "chunk": c["id"], "index": k, "take": sp.next_take_id(c),
            "deliver": list(c["deliver"]), "render": list(c["render"]), "held": int(c.get("held") or 0),
            "length": int(c["length"]), "fill": c.get("fill"),
            "splits": {"left": left and {"frame": left["frame"], "mode": left["mode"]},
                       "right": right and {"frame": right["frame"], "mode": right["mode"]}},
            "pins": pins, "pin_notes": notes, "seed": int(seed), "prompt": prompt,
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
            "options": options,
            "source": {k2: src[k2] for k2 in ("path", "frames", "fps", "width", "height", "audio")},
            "conform": bool(s["conform_to_24fps"]), "render_fps": tl.conform_rate(fr, s["conform_to_24fps"]),
            "anchors": int(s["anchors"]), "hand_back": int(s["hand_back"]),
            "nonce": run.get("nonce") or uuid.uuid4().hex, "started": time.time()}


def decode_frames(path, fr, first, count, size=None):
    """`count` frames of a file from its frame `first` (at fr), as a float IMAGE batch."""
    out = None
    n = 0
    for i, f in enumerate(tl._iter_frames(path, fr, first, first + count)):
        if size and (f.shape[1], f.shape[0]) != tuple(size):
            import cv2
            f = cv2.resize(f, tuple(size), interpolation=cv2.INTER_AREA if f.shape[1] > size[0] else cv2.INTER_CUBIC)
        if out is None:
            out = torch.empty((count, f.shape[0], f.shape[1], 3), dtype=torch.float32)
        out[i] = torch.from_numpy(f).to(torch.float32).div_(255.0)
        n += 1
    if n != count:
        raise PlannerError(f"{os.path.basename(path)}: decoded {n} frames from {first}, expected {count}")
    return out


def object_crop(plan, job_dir, r0, r1, target, w, h, extra=None, pad=None):
    """An object job's render box in source pixels [x0, y0, x1, y1] (even sides): the union of the target's cached
    masks over r0..r1, each side padded by sp.CROP_PAD of its own length (at least sp.CROP_MIN), inside the frame. None when no
    mask covers the range or every one is empty."""
    cover = sp.mask_cover(plan, r0, r1, target)
    if not cover:
        return None
    fr = int(round(float(plan["source"]["fps"])))
    box = None
    for seg, f0, f1 in cover:
        path = os.path.join(job_dir, seg["file"])
        if not os.path.isfile(path):
            return None
        for f in tl._iter_frames(path, fr, f0 - int(seg["range"][0]), f1 - int(seg["range"][0]) + 1):
            m = f[..., 0] > 127
            if not m.any():
                continue
            mh, mw = m.shape
            ys, xs = np.nonzero(m)
            b = (xs.min() * w / mw, ys.min() * h / mh, (xs.max() + 1) * w / mw, (ys.max() + 1) * h / mh)
            box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    for b in extra or []:                       # an add's cut-out boxes: the crop holds the added object too
        if b is not None:
            box = tuple(b) if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    if box is None:
        return None
    box = (max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(h), box[3]))
    bw, bh = box[2] - box[0], box[3] - box[1]
    # each side padded by its own size: a strip along the frame's edge (a painting cut off by the framing) stays a strip
    pad = sp.CROP_PAD if pad is None else float(pad)
    cw = min(w, max(sp.CROP_MIN, bw * (1 + 2 * pad)))
    ch = min(h, max(sp.CROP_MIN, bh * (1 + 2 * pad)))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0 = int(max(0, min(w - cw, cx - cw / 2))) // 2 * 2
    y0 = int(max(0, min(h - ch, cy - ch / 2))) // 2 * 2
    x1, y1 = min(w, x0 + int(cw) // 2 * 2), min(h, y0 + int(ch) // 2 * 2)
    return [x0, y0, x1, y1]


def render_inputs(desc, job_dir, plan):
    """Decode the render's frames (+ held), its audio and its pins. All at execution. An object job renders a crop
    around its target (desc["crop"], object_crop): the frames, the size and the cached mask are cropped, and no pins
    (outside the target the assembly is the source)."""
    src = desc["source"]
    path = src["path"]
    if not os.path.isfile(path):
        raise PlannerError(f"the source video is missing: {path}")
    fr = int(round(float(src["fps"])))
    info = tl.probe(path, fr)
    if int(info["frames"]) != int(src["frames"]):
        raise PlannerError(f"the source now has {info['frames']} frames at {fr} fps; the plan was made for {src['frames']}")
    w, h = int(info["width"]), int(info["height"])
    src_w, src_h = w, h
    r0, r1 = desc["render"]
    n = r1 - r0 + 1
    held = int(desc["held"])
    frames = decode_frames(path, fr, r0, n)
    desc["crop"] = None
    desc["object_refine"] = False
    guide = None
    zone = None
    pin_frames = None
    if sp.object_composite(plan):
        try:
            from . import swap_keep as sk
        except ImportError:
            import swap_keep as sk
        o = sp.object_settings(plan)
        kind = o.get("guide") if o.get("guide") in sp.OBJECT_GUIDES else "composite"
        guide = sk.CompositeGuide(plan, job_dir, fr, (w, h), r0, r1, (desc["options"] or {}).get("target", ""))
        if kind == "take":                      # a second pass: this chunk's own take, refined again
            _c, t = sp.find_take(plan, o.get("from_take") or "")
            if t is None or [int(x) for x in t["render"]] != [r0, r1]:
                raise PlannerError(f"object guide 'take': {o.get('from_take')!r} isn't a take of this chunk's render {r0}-{r1}")
            frames = decode_frames(os.path.join(job_dir, t["file"]), fr, 0, n)
            desc["object_refine"], desc["composite_frames"] = True, n
        else:
            comp, placed = frames.clone(), 0
            for i in range(n):
                f8 = (frames[i].numpy() * 255 + 0.5).astype(np.uint8)
                out, a = guide.apply(r0 + i, f8)
                if a is not None:
                    comp[i] = torch.from_numpy(out).to(torch.float32).div_(255.0)
                    placed += 1
            desc["composite_frames"] = placed
            if kind == "placeholder":           # a grey silhouette, marked: H3 replaces it with the object
                grey = np.full_like(guide.cut_rgb, 128)
                masks = torch.zeros((n, h, w), dtype=torch.float32)
                for i in range(n):
                    p_ = guide.place(r0 + i)
                    if p_ is None:
                        continue
                    f8 = (frames[i].numpy() * 255 + 0.5).astype(np.uint8)
                    out, a = sk.place_cutout(f8, grey, guide.cut_a, p_)
                    frames[i] = torch.from_numpy(out).to(torch.float32).div_(255.0)
                    masks[i] = torch.from_numpy((a > 0.5).astype(np.float32))
                desc["placeholder_mask"] = masks
            elif kind == "composite":
                frames = comp
                desc["object_refine"] = placed > 0
            else:                               # "pins": the source is the guide; the composite pins its ends
                k = max(1, int(desc.get("anchors") or 1))
                pin_frames = (comp[:k], comp[n - k:])
    elif sp.object_zone(plan):                  # a zone redrawn, no cut-out: the frames stay as filmed
        try:
            from . import swap_keep as sk
        except ImportError:
            import swap_keep as sk
        zone = guide = sk.ZoneGuide(plan, job_dir, fr, (w, h), r0, r1, (desc["options"] or {}).get("target", ""))
        if not zone.anchors:
            raise PlannerError(f"chunk {desc['chunk']}: object add (zone) found no cached mask of its target over frames "
                               f"{r0}-{r1}: mark the chunk first")
        desc["object_zone"] = True
    if sp.job_mode(plan) == sp.MODE_OBJECT and sp.object_settings(plan).get("crop", True):
        desc["crop"] = object_crop(plan, job_dir, r0, r1, (desc["options"] or {}).get("target", ""), w, h,
                                   extra=[guide.box(f) for f in range(r0, r1 + 1)] if guide is not None else None,
                                   pad=float(sp.object_settings(plan).get("crop_pad", sp.CROP_PAD)))
        if desc["crop"]:
            x0, y0, x1, y1 = desc["crop"]
            frames = frames[:, y0:y1, x0:x1, :].contiguous()
            w, h = x1 - x0, y1 - y0
            if any(desc["pins"].get(s) for s in ("start", "end")):
                desc["pin_notes"].append("object job, cropped render: no pins (outside the target the assembly is the source)")
            desc["pins"] = {"start": None, "end": None}
            if pin_frames is not None:
                pin_frames = tuple(p_[:, y0:y1, x0:x1, :].contiguous() for p_ in pin_frames)
    if zone is not None:                        # the zone per frame, at the (cropped) frames' size
        cx0, cy0, cx1, cy1 = desc["crop"] or (0, 0, src_w, src_h)
        zmask = torch.zeros((n, cy1 - cy0, cx1 - cx0), dtype=torch.float32)
        found = 0
        for i in range(n):
            m = zone.mask(r0 + i)
            if m is not None:
                zmask[i] = torch.from_numpy(m[cy0:cy1, cx0:cx1])
                found += 1
        if held:
            zmask = torch.cat([zmask, zmask[-1:].expand(held, -1, -1)], 0)
        desc["zone_mask"] = zmask.contiguous()
        desc["zone_frames"] = found
        o_ = sp.object_settings(plan)
        a_at = max(0, min(n - 1, int(round(float(o_["anchor_at"]) * float(desc["render_fps"])))))
        if o_["anchor"] == "picture":                       # your own finished anchor frame, fitted over the crop
            try:
                from . import swap_keep as sk
            except ImportError:
                import swap_keep as sk
            try:
                pic = sk.load_picture(o_.get("anchor_picture"), job_dir)
            except ValueError:
                raise PlannerError(f"chunk {desc['chunk']}: the anchor picture is not set: pick a file for the object's anchor "
                                   f"picture (a file in ComfyUI's input folder)")
            pic = sk.fit_cover(pic, cx1 - cx0, cy1 - cy0)
            desc["anchor_image"] = torch.from_numpy(np.ascontiguousarray(pic)).to(torch.float32).div_(255.0)[None]
        elif o_["anchor"] == "composite":                   # the cut-out of Picture 1 placed on the anchor frame, no model
            if not (o_.get("cutout") or {}).get("rgb"):
                raise PlannerError(f"chunk {desc['chunk']}: the composite anchor needs Picture 1's cut-out (the BiRefNet group): "
                                   f"it has none")
            try:
                from . import swap_keep as sk
            except ImportError:
                import swap_keep as sk
            cg = sk.CompositeGuide(plan, job_dir, fr, (src_w, src_h), r0, r1, (desc["options"] or {}).get("target", ""))
            f8 = (decode_frames(path, fr, r0 + a_at, 1)[0].numpy() * 255 + 0.5).astype(np.uint8)    # the whole frame, before the crop
            comp, _a = cg.apply(r0 + a_at, f8)
            comp = comp[cy0:cy1, cx0:cx1]
            desc["anchor_image"] = torch.from_numpy(np.ascontiguousarray(comp)).to(torch.float32).div_(255.0)[None]
    if held:
        frames = torch.cat([frames, frames[-1:].expand(held, -1, -1, -1)], 0).contiguous()
    total = n + held
    if total != desc["length"]:
        raise PlannerError(f"chunk {desc['chunk']}: {total} frames for a {desc['length']}-frame render")
    cut = {"pieces": [{"path": path, "enter": r0, "exit": r1 + 1}], "frames": n}
    wav, sr = tl.cut_audio(cut, fr)
    want = int(round(total / float(fr) * sr))
    if wav.shape[1] < want:                      # silence under the held frames
        wav = np.pad(wav, ((0, 0), (0, want - wav.shape[1])))
    wav = wav[:, :want]
    if desc["conform"] and desc["render_fps"] != fr:
        wav = tl.conform_audio(wav, sr, total, fr)
    audio = {"waveform": torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0), "sample_rate": sr}
    # the render's size. first_pass_mp 0: the two-pass first pass (~0.3 MP) and pins at the source's size, which the
    # refine pass's own guides resize. > 0: a single pass at that size, and the pins arrive STRETCHED to it: the guides
    # node would otherwise centre-crop each pin to the canvas's shape, and a canvas a hair off the source's shape
    # (1344 x 768 against 1280 x 720) cost a strip of the picture per chunk, chunk after chunk (the couch got longer).
    mp = float(sp._settings(plan.get("settings")).get("first_pass_mp") or 0.0)
    desc["pass1"] = list(fit_size(w, h, mp) if mp > 0 else pass1_size(w, h))
    pin_size = tuple(desc["pass1"]) if mp > 0 else (w, h)
    pins = {}
    if pin_frames is not None:                  # an object add's composite, pinned at both ends (guide "pins")
        import torch.nn.functional as F_
        for side, pf in zip(("start", "end"), pin_frames):
            pins[side] = F_.interpolate(pf.permute(0, 3, 1, 2), size=(pin_size[1], pin_size[0]), mode="area").permute(0, 2, 3, 1).contiguous()
        desc["pins"] = {"start": {"object_composite": True, "frames": [r0, r0 + int(pin_frames[0].shape[0]) - 1]},
                        "end": {"object_composite": True, "frames": [r1 - int(pin_frames[1].shape[0]) + 1, r1]}}
        return frames, audio, pins, (w, h, fr)
    for side in ("start", "end"):
        p = desc["pins"].get(side)
        if not p:
            pins[side] = torch.zeros((0, pin_size[1], pin_size[0], 3), dtype=torch.float32)
            continue
        if p.get("source"):                      # anchored onto the original: the source's own frames
            f0, f1 = p["frames"]
            pins[side] = decode_frames(path, fr, f0, f1 - f0 + 1, pin_size)
            continue
        _c, t = sp.find_take(plan, p["take"])
        tp = os.path.join(job_dir, t["file"])
        if not os.path.isfile(tp):
            raise PlannerError(f"pins for {desc['chunk']} ({side}): take {t['id']}'s file is missing: {tp}")
        f0, f1 = p["frames"]
        pins[side] = decode_frames(tp, fr, f0 - int(t["render"][0]), f1 - f0 + 1, pin_size)
    return frames, audio, pins, (w, h, fr)


def save_cutout(plan_file, job_dir, image, mask):
    """The graph's cut-out of Picture 1 into the job (object/cutout_rgb.png, object/cutout_mask.png) when it changed;
    the plan's object.cutout names them. Returns the updated plan, or None when nothing changed."""
    import cv2
    rgb = (image[0].cpu().numpy() * 255 + 0.5).clip(0, 255).astype(np.uint8)
    m = mask[0].cpu().numpy() if mask.dim() == 3 else mask.cpu().numpy()
    a = (m * 255 + 0.5).clip(0, 255).astype(np.uint8)
    sha = hashlib.sha1(rgb.tobytes() + a.tobytes()).hexdigest()
    cur = (sp.load_plan(plan_file).get("object") or {}).get("cutout") or {}
    if cur.get("sha1") == sha and os.path.isfile(os.path.join(job_dir, cur.get("rgb", ""))):
        return None
    d = os.path.join(job_dir, "object")
    os.makedirs(d, exist_ok=True)
    cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tofile(os.path.join(d, "cutout_rgb.png"))
    cv2.imencode(".png", a)[1].tofile(os.path.join(d, "cutout_mask.png"))

    def upd(p):
        p["object"] = dict(p.get("object") or {}, cutout={"rgb": "object/cutout_rgb.png", "mask": "object/cutout_mask.png",
                                                         "sha1": sha, "size": [int(rgb.shape[1]), int(rgb.shape[0])]})
    return sp.update_plan(plan_file, upd)


def mark_descriptor(plan, plan_file, run):
    """A mark run's descriptor: the range whose source mask is tracked and cached: a chunk's render
    range, one shot of it (the strip marks shot by shot: SAM3 doesn't find the target again after a
    cut), or an explicit [a, b]. A chunk's run tracks the chunk's target; an object keep's run
    ({"keep": id}) tracks the keep's target over the given range, else the keep's own."""
    src = plan["source"]
    keep = None
    if run.get("keep"):
        keep = next((k for k in plan.get("object_keeps") or [] if k.get("id") == run["keep"]), None)
        if keep is None:
            raise PlannerError(f"no object keep {run['keep']!r}")
    c = _chunk(plan, run["chunk"]) if run.get("chunk") and keep is None else None
    if c is not None and sp.is_kept(c):
        raise PlannerError(f"chunk {c['id']} {c['deliver']} is kept as the original: it needs no mask")
    if run.get("range"):
        r0, r1 = int(run["range"][0]), int(run["range"][1])
    else:
        r0, r1 = keep["range"] if keep is not None else c["render"]
    cid = c and c["id"]
    if not 0 <= r0 <= r1 < int(src["frames"]):
        raise PlannerError(f"mark range {r0}-{r1} is outside the source (0-{int(src['frames']) - 1})")
    d = {"format": MARK_FORMAT, "job": plan["job"], "plan": plan_file, "plan_rev": plan.get("rev"),
         "chunk": cid, "range": [r0, r1], "frames": r1 - r0 + 1,
         "target": keep["target"] if keep is not None else
                   sp.chunk_target(plan, c) if cid else (plan.get("target") or "").strip(),
         "source": {k: src[k] for k in ("path", "frames", "fps", "width", "height", "audio")},
         "nonce": run.get("nonce") or uuid.uuid4().hex, "started": time.time()}
    if keep is not None:
        d.update(kind="keep", keep=keep["id"])
    return d


def cached_mask(plan, job_dir, r0, r1, held=0, target=""):
    """The cached source mask of `target` over r0..r1 (+ held repeats) as a MASK [n, h, w], or (None, why)."""
    cover = sp.mask_cover(plan, r0, r1, target)
    if not cover:
        return None, f"no cached mask of {target or 'person'} covers {r0}-{r1}"
    fr = int(round(float(plan["source"]["fps"])))
    parts = []
    for seg, f0, f1 in cover:
        path = os.path.join(job_dir, seg["file"])
        if not os.path.isfile(path):
            return None, f"mask {seg['id']}'s file is missing: {path}"
        n = 0
        for f in tl._iter_frames(path, fr, f0 - int(seg["range"][0]), f1 - int(seg["range"][0]) + 1):
            parts.append(torch.from_numpy(f[..., 0].astype(np.float32) / 255.0))
            n += 1
        if n != f1 - f0 + 1:
            return None, f"mask {seg['id']}: decoded {n} frames of {f1 - f0 + 1}"
    m = torch.stack(parts, 0)
    if held:
        m = torch.cat([m, m[-1:].expand(held, -1, -1)], 0)
    return m.contiguous(), None


def _blocked(n=len(RETURN_TYPES)):
    return [ExecutionBlocker(None) for _ in range(n)]


def parse_run(run):
    run = (run or "").strip()
    if not run:
        return {"action": ACTION_NONE}
    try:
        d = json.loads(run)
    except ValueError as e:
        raise PlannerError(f"run is not JSON: {e}")
    if not isinstance(d, dict) or d.get("action", ACTION_NONE) not in ACTIONS:
        raise PlannerError(f"run.action must be render, mark, assemble, draft or none: {run[:200]}")
    d.setdefault("action", ACTION_NONE)
    return d


class SeamStitchSwapPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job": ("STRING", {"default": "", "tooltip":
                    "The job: a folder under output/seamstitch_swap/ holding plan.json, the takes and the "
                    "assemblies. Created on the first edit."}),
                "source": ("STRING", {"default": "", "tooltip":
                    "The source video (a path; input/output folder paths work too). Fixed once a chunk "
                    "has takes."}),
                "target_render_frames": ("INT", {"default": 209, "min": 22, "max": 1000, "step": 1, "tooltip":
                    "Auto-placement render length (on H3's 17k+5 grid). 209 = about 8 s."}),
                "overlap_frames": ("INT", {"default": 12, "min": 0, "max": 100, "step": 1, "tooltip":
                    "An anchored split's overlap: the right chunk renders from J - overlap."}),
                "anchor_frames": ("INT", {"default": 5, "min": 0, "max": 39, "step": 1, "tooltip":
                    "Pins per side (clip anchors: 5 or 22)."}),
                "floor_frames": ("INT", {"default": 124, "min": 5, "max": 1000, "step": 1, "tooltip":
                    "Warn below this render length (H3's trained minimum)."}),
                "ceiling_frames": ("INT", {"default": 209, "min": 5, "max": 2000, "step": 1, "tooltip":
                    "Warn above this render length: two-pass renders over 209 frames failed or crashed ComfyUI at the "
                    "1 MP refine on a 32 GB card with 64 GB RAM (B5b: 226 failed, then crashed; B5a: 243 crashed, 260 "
                    "failed)."}),
                "conform_to_24fps": ("BOOLEAN", {"default": True, "tooltip":
                    "The render's reference audio on H3's 24 fps clock (frames stay 1:1). Takes keep "
                    "the original audio."}),
                "run": ("STRING", {"default": "", "tooltip":
                    "Set by the Planner's UI (or a script) before each queue: what this run does."}),
                "ui_state": ("STRING", {"default": "", "tooltip": "The strip's zoom and selection."}),
            },
            "optional": {
                # appended last, so a saved workflow's widget values keep their places
                "first_pass_megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip":
                    "0: the two-pass first pass (about 0.3 MP in the source's shape; the pins arrive at the source's "
                    "size, for the refine pass). Above 0: a single pass at this many megapixels (1.0 = H3's own canvas) "
                    "in the source's shape, a multiple of 32 each way, wired from render_width / render_height into "
                    "the H3 node: the pins arrive stretched to exactly that size, so the guides node doesn't crop a "
                    "strip off every pin (it did, 1.6% of the width per chunk at 1344 x 768 on a 1280 x 720 source, "
                    "and the picture crept wider chunk by chunk)."}),
                "forward_chain": ("BOOLEAN", {"default": False, "tooltip":
                    "On: a render pins to the NEWEST take of the chunk before it (not the chosen one) and never to the "
                    "chunk after it, so renders queued in order (c6, c7, c8 ...) each start from the take just made, "
                    "and an old take further down the chain can't pull a chunk towards it. Your choices in the strip "
                    "stay yours: this only decides what a render pins to. Off: pins follow the chosen takes on both "
                    "sides, for re-rolling one chunk between two takes you've chosen."}),
                "object_cutout": ("IMAGE", {"tooltip":
                    "Object jobs that add a thing: the sheet's object cut out (BiRefNet of Picture 1, background black). "
                    "With its mask, the render's guide is the cut-out placed on the target in every frame."}),
                "object_cutout_mask": ("MASK", {"tooltip": "The cut-out's mask (BiRefNet's mask output)."}),
            },
        }

    RETURN_TYPES = RETURN_TYPES
    RETURN_NAMES = RETURN_NAMES
    FUNCTION = "plan"
    CATEGORY = "SeamStitch/Swap"
    DESCRIPTION = ("The job's plan: splits, chunks, prompts and takes. Each run emits one chunk's render inputs "
                   "(pins decoded from the neighbours' takes at execution), or routes an assemble or draft run.")

    @classmethod
    def IS_CHANGED(cls, job="", source="", run="", **kw):
        h = hashlib.sha256(json.dumps([job, source, run, sorted(kw.items())], default=str).encode())
        try:
            _jd, pp = job_paths(job)
            h.update(str(sp.load_plan(pp).get("rev")).encode())
            src = sp.load_plan(pp)["source"]["path"]
            st = os.stat(src)
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
        except Exception:
            pass
        return h.hexdigest()

    def plan(self, job, source, target_render_frames=209, overlap_frames=12, anchor_frames=5, floor_frames=124,
             ceiling_frames=209, conform_to_24fps=True, run="", ui_state="", first_pass_megapixels=0.0,
             forward_chain=False, object_cutout=None, object_cutout_mask=None):
        widgets = {"target_render_frames": target_render_frames, "overlap_frames": overlap_frames,
                   "anchor_frames": anchor_frames, "floor_frames": floor_frames, "ceiling_frames": ceiling_frames,
                   "conform_to_24fps": conform_to_24fps, "first_pass_megapixels": first_pass_megapixels,
                   "forward_chain": bool(forward_chain)}
        settings = {WIDGET_SETTINGS[k]: v for k, v in widgets.items()}
        r = parse_run(run)
        jd, pp = job_paths(job)
        if not os.path.isfile(pp):
            if not (source or "").strip():
                raise PlannerError(f"job {job!r} has no plan yet: set source to create it")
            create_job(job, source, settings)
        plan = sp.load_plan(pp)
        if (source or "").strip():
            try:
                want = _norm(tl.resolve_path(source.strip().strip('"')))
                # a job with a chunk "edit this" runs on a patched file; the widget still names the clip it started from
                same = want in (_norm(plan["source"]["path"]), _norm((plan.get("source_orig") or {}).get("path") or ""))
            except Exception:
                same = False
            if not same:
                raise PlannerError(f"job {job!r} is on {plan['source']['path']}, not {source!r}: clear source, "
                                   f"or start a new job")
        cur = sp._settings(plan.get("settings"))
        changed = {k: v for k, v in settings.items() if cur.get(k) != type(sp.DEFAULT_SETTINGS[k])(v)}
        if changed:
            _say(f"[SeamStitch] Swap Planner: {job}: settings from the widgets {changed}")
            plan = sp.update_plan(pp, lambda p: apply_op(p, {"op": "settings", "settings": changed}))

        out = _blocked()
        action = r["action"]
        if action == ACTION_RENDER and object_cutout is not None and object_cutout_mask is not None \
                and sp.job_mode(plan) == sp.MODE_OBJECT:
            plan = save_cutout(pp, jd, object_cutout, object_cutout_mask) or plan
        if action == ACTION_RENDER:
            desc = render_descriptor(plan, pp, r)
            frames, audio, pins, (w, h, fr) = render_inputs(desc, jd, plan)

            def mark(p):
                c = sp.find_chunk(p, desc["chunk"])
                c["state"] = "rendering"
                c["rendering"] = {k: desc[k] for k in ("take", "seed", "pins", "nonce", "prompt_sha1", "render",
                                                       "held", "length")}
                c["rendering"]["started"] = sp.now()
            plan = sp.update_plan(pp, mark)
            desc["plan_rev"] = plan["rev"]
            out[OUT_RENDER_W], out[OUT_RENDER_H] = desc["pass1"]          # set by render_inputs, with the pins' size
            out[OUT_OBJECT_REFINE] = bool(desc.get("object_refine"))
            zmask = desc.pop("zone_mask", None)
            aimg = desc.pop("anchor_image", None)
            o_ = sp.object_settings(plan)
            out[OUT_ANCHOR_PROMPT] = sp.zone_anchor_text(plan, sp.find_chunk(plan, desc["chunk"])) if zmask is not None else ""
            out[OUT_ANCHOR_QWEN] = zmask is not None and o_["anchor"] == "qwen"
            out[OUT_ANCHOR_IMAGE] = aimg if aimg is not None else torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            tot_ = int(desc["length"])
            out[OUT_ZONE_ON] = zmask is not None
            out[OUT_OBJECT_ZONE] = zmask if zmask is not None else torch.zeros((1, 64, 64), dtype=torch.float32)
            out[OUT_ANCHOR_FRAME] = max(0, min(tot_ - 1, int(round(float(o_["anchor_at"]) * desc["render_fps"]))))
            out[OUT_ENTER_FRAME] = max(0, min(tot_ - 1, int(round(float(o_["enter_at"]) * desc["render_fps"]))))
            if zmask is not None:
                _say(f"[SeamStitch] Swap Planner:   object add (zone): the zone found on {desc['zone_frames']} frames; the anchor is "
                     f"drawn into frame {out[OUT_ANCHOR_FRAME]}, the thing enters at frame {out[OUT_ENTER_FRAME]} "
                     f"(the zone is empty before it)")
            if desc.get("object_refine"):
                _say(f"[SeamStitch] Swap Planner:   object add: the cut-out composited on {desc['composite_frames']} "
                     f"frames (the refine group renders it img2img)")
            out[:13] = [desc, frames, audio, desc["prompt"], desc["seed"], desc["length"], pins["start"], pins["end"],
                        w, h, desc["render_fps"], fr, json.dumps(desc["options"])]
            m, why = cached_mask(plan, jd, desc["render"][0], desc["render"][1], desc["held"],
                                 desc["options"].get("target", ""))
            ph = desc.pop("placeholder_mask", None)
            if ph is not None:                  # the placeholder is what the guide marks, at the cached masks' size
                size = next((tuple(x["size"]) for x in reversed(plan.get("masks") or []) if x.get("size")), None)
                if size:
                    import torch.nn.functional as F_
                    ph = F_.interpolate(ph[:, None], size=(size[1], size[0]), mode="nearest")[:, 0]
                held = int(desc["held"])
                m = torch.cat([ph, ph[-1:].expand(held, -1, -1)], 0).contiguous() if held else ph
                why = None
                _say("[SeamStitch] Swap Planner:   object placeholder: the grey silhouette is the marked target")
            if m is not None and desc.get("crop"):           # the object job's crop, in the mask's own size
                x0, y0, x1, y1 = desc["crop"]
                sw, sh = int(desc["source"]["width"]), int(desc["source"]["height"])
                mh, mw = int(m.shape[1]), int(m.shape[2])
                m = m[:, int(y0 * mh / sh):max(int(y0 * mh / sh) + 1, int(round(y1 * mh / sh))),
                      int(x0 * mw / sw):max(int(x0 * mw / sw) + 1, int(round(x1 * mw / sw)))].contiguous()
                _say(f"[SeamStitch] Swap Planner:   object crop {desc['crop']} ({x1 - x0}x{y1 - y0} of {sw}x{sh})")
            out[OUT_SOURCE_MASK] = m if m is not None else torch.zeros((1, 64, 64), dtype=torch.float32)
            out[OUT_HAS_MASK] = m is not None
            desc["source_mask"] = "cached" if m is not None else None
            _say(f"[SeamStitch] Swap Planner:   source mask: " + (f"cached ({m.shape[2]}x{m.shape[1]})" if m is not None
                                                                    else f"none ({why}): the render group tracks it"))
            pin_s = ", ".join(f"{s} {'the source' if p.get('source') else 'the object composite' if p.get('object_composite') else p['take']} {p['frames'][0]}-{p['frames'][1]}"
                              for s, p in desc["pins"].items() if p) or "none"
            _say(f"[SeamStitch] Swap Planner: {job}: render {desc['chunk']} (take {desc['take']}) frames "
                  f"{desc['render'][0]}-{desc['render'][1]}" + (f" + {desc['held']} held" if desc["held"] else "")
                  + f" = {desc['length']}, seed {desc['seed']}, pins {pin_s}"
                  + (f", audio on H3's {desc['render_fps']} fps clock" if desc["render_fps"] != fr else ""))
            for n_ in desc["pin_notes"]:
                _say(f"[SeamStitch] Swap Planner:   {n_}")
        elif action == ACTION_MARK:
            desc = mark_descriptor(plan, pp, r)
            src = desc["source"]
            if not os.path.isfile(src["path"]):
                raise PlannerError(f"the source video is missing: {src['path']}")
            fr = int(round(float(src["fps"])))
            out[OUT_MARK_CHUNK] = desc
            out[OUT_MARK_IMAGES] = decode_frames(src["path"], fr, desc["range"][0], desc["frames"])
            out[OUT_OPTIONS] = json.dumps({"target": desc["target"]} if desc["target"] else {})
            _say(f"[SeamStitch] Swap Planner: {job}: mark {desc['chunk'] or ''} frames {desc['range'][0]}-"
                  f"{desc['range'][1]} ({desc['frames']}): the source mask of {desc['target'] or 'person'}, tracked and cached")
        elif action == ACTION_ASSEMBLE:
            out[OUT_ASSEMBLE] = pp
        elif action == ACTION_DRAFT:
            # a redraft names its chunk(s): Draft Prompts drafts those, whatever its chunks widget says
            named = [x for x in ([r["chunk"]] if r.get("chunk") else []) + list(r.get("chunks") or []) if x]
            out[OUT_DRAFT] = json.dumps({"plan": pp, "chunks": named}) if named else pp
        for w_ in plan.get("warnings", []):
            _say(f"[SeamStitch] Swap Planner: warning: {w_['text']}")
        return {"ui": {"seamstitch_swap_plan": [{"job": plan["job"], "rev": plan["rev"], "text": plan_text(plan)}]},
                "result": tuple(out)}


# ---------------------------------------------------------------------------
# Swap Option: one per-chunk option, so the Planner stays generic
# ---------------------------------------------------------------------------

def _parse_value(v):
    if isinstance(v, str):
        t = v.strip()
        if t.lower() in ("true", "false"):
            return t.lower() == "true"
        try:
            return json.loads(t)
        except ValueError:
            return v
    return v


def option_value(options, key, default):
    try:
        d = json.loads(options) if isinstance(options, str) and options.strip() else (options or {})
    except ValueError:
        raise PlannerError(f"options is not JSON: {options[:200]}")
    v = d.get(key, _parse_value(default)) if isinstance(d, dict) else _parse_value(default)
    v = _parse_value(v)
    as_bool = bool(v) if not isinstance(v, str) else v.strip().lower() not in ("", "0", "false", "no", "off")
    try:
        as_float = float(v)
    except (TypeError, ValueError):
        as_float = float(as_bool)
    return as_bool, int(as_float), as_float, v if isinstance(v, str) else json.dumps(v)


class SeamStitchSwapOption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "options": ("STRING", {"forceInput": True, "tooltip": "The Swap Planner's options output."}),
                "key": ("STRING", {"default": "mark", "tooltip": "The option to read, e.g. mark."}),
                "default": ("STRING", {"default": "true", "tooltip": "Its value when the chunk doesn't set it."}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("boolean", "int", "float", "string")
    FUNCTION = "read"
    CATEGORY = "SeamStitch/Swap"
    DESCRIPTION = "Reads one per-chunk option from the Swap Planner (e.g. mark) as a boolean, int, float and string."

    def read(self, options, key="mark", default="true"):
        return option_value(options, key, default)


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/seamstitch/swap/plan")
    async def _plan_route(request):
        try:
            return web.json_response(plan_view(request.query.get("job", "")))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/seamstitch/swap/zone_preview")
    async def _zone_preview_route(request):
        try:
            import asyncio
            q = request.query
            res = await asyncio.get_event_loop().run_in_executor(None, do_zone_preview, q.get("job", ""), q.get("frame", "0"))
            return web.json_response(res)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/seamstitch/swap/op")
    async def _op_route(request):
        try:
            body = await request.json()
            import asyncio
            res = await asyncio.get_event_loop().run_in_executor(None, do_op, body)
            try:
                PromptServer.instance.send_sync("seamstitch_swap_plan", {"job": body.get("job"), "rev": res.get("rev")})
            except Exception:
                pass
            return web.json_response(res)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.get("/seamstitch/swap/jobs")
    async def _jobs_route(request):
        try:
            stem = request.query.get("name_for")
            return web.json_response({"jobs": list_jobs(), "free_name": free_job_name(stem) if stem else None})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/seamstitch/swap/detect_cuts")
    async def _detect_cuts_route(request):
        try:
            body = await request.json()
            import asyncio
            res = await asyncio.get_event_loop().run_in_executor(None, do_detect_cuts, body)
            try:
                PromptServer.instance.send_sync("seamstitch_swap_plan", {"job": body.get("job"), "rev": res.get("rev")})
            except Exception:
                pass
            return web.json_response(res)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
except Exception:  # pragma: no cover - no server (tools)
    pass
