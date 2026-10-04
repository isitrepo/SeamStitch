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
    POST /seamstitch/swap/detect_cuts        ffmpeg scene > 0.15, merged as suggested cuts
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
# tracked up front and cached per range); 17-18 a render run's cached source mask, when one covers it.
RETURN_TYPES = (CHUNK_TYPE, "IMAGE", "AUDIO", "STRING", "INT", "INT", "IMAGE", "IMAGE", "INT", "INT", "INT",
                "INT", "STRING", "STRING", "STRING", MARK_TYPE, "IMAGE", "MASK", "BOOLEAN")
RETURN_NAMES = ("chunk", "images", "audio", "prompt", "seed", "length", "start_pins", "end_pins", "width",
                "height", "frame_rate", "source_frame_rate", "options", "draft_plan", "assemble_plan",
                "mark_chunk", "mark_images", "source_mask", "has_source_mask")
OUT_DRAFT, OUT_ASSEMBLE = 13, 14
OUT_MARK_CHUNK, OUT_MARK_IMAGES, OUT_SOURCE_MASK, OUT_HAS_MASK = 15, 16, 17, 18
SCENE_THRESHOLD = 0.15           # ffmpeg scene score (> 0.15 found every cut of the 978-frame test clip)

# widget name -> plan setting
WIDGET_SETTINGS = {"target_render_frames": "target_render", "overlap_frames": "overlap", "anchor_frames": "anchors",
                   "floor_frames": "floor", "ceiling_frames": "ceiling", "conform_to_24fps": "conform_to_24fps"}


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
        geo = True
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
    elif op == "set_options":
        c = _chunk(plan, body["chunk"])
        c["options"] = dict(c.get("options") or {}, **(body.get("options") or {}))
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
        # who in the source is replaced (the drafter's main person), when its own pick is wrong; "" = let it pick
        plan["target"] = (body.get("target") or "").strip()
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
        frames = []
        for a, b in spans:
            rel = [c - a for c in cut_frames if a < c <= b]
            frames += [a + x for x in sp.auto_splits(b - a + 1, rel, plan.get("settings"))]
        plan["splits"] = sorted(keep_splits.values(), key=lambda d: d["frame"])
        plan["chunks"] = kept
        have = {d["frame"] for d in plan["splits"]}
        for f in frames:
            if f not in have:
                sp.add_split(plan, f, sp.MODE_ANCHORED)
        res["splits"] = frames
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


FILE_OPS = {"delete_take": _op_delete_take, "restore_take": _op_restore_take, "use_take": _op_use_take,
            "delete_mask": _op_delete_mask}


# ---------------------------------------------------------------------------
# cut detection: "detect + plan" (the default) brings the cuts in confirmed and places the auto
# splits from them in one step, for one review; "detect" alone adds faint suggestions
# ---------------------------------------------------------------------------

def detect_cuts(path, fps, threshold=SCENE_THRESHOLD):
    """ffmpeg's scene score > threshold, as source frame numbers (frame c = the first of a new shot)."""
    ff = tl._ffmpeg_exe()
    p = subprocess.run([ff, "-hide_banner", "-nostats", "-i", path, "-an", "-vf",
                        f"select='gt(scene,{float(threshold)})',showinfo", "-f", "null", "-"],
                       capture_output=True)
    if p.returncode != 0:
        raise PlannerError(f"cut detection failed: {p.stderr.decode(errors='replace')[-600:]}")
    base = float(tl.probe(path, fps).get("base_time") or 0.0)
    out = []
    for m in re.finditer(r"pts_time:\s*([0-9.]+)", p.stderr.decode(errors="replace")):
        f = int(round((float(m.group(1)) - base) * float(fps)))
        if f > 0 and f not in out:
            out.append(f)
    return sorted(out)


def merge_detected(plan, found, confirmed=False):
    """Detected cuts in, as suggestions (or confirmed). Confirmed or hand-placed cuts stay; earlier
    unconfirmed suggestions are replaced; a detection on a frame that already has a cut is skipped
    (confirmed=True confirms a suggestion already there)."""
    keep = [c for c in plan.get("cuts", []) if c.get("confirmed", True) or c.get("from") != "detected"
            or (confirmed and c["frame"] in found)]
    if confirmed:
        for c in keep:
            if c["frame"] in found:
                c["confirmed"] = True
    have = {c["frame"] for c in keep}
    added = [f for f in found if f not in have]
    plan["cuts"] = sorted(keep + [{"frame": f, "from": "detected", "confirmed": bool(confirmed)} for f in added],
                          key=lambda c: c["frame"])
    sp.rebuild_chunks(plan)
    return added


def detect_and_plan(plan, found):
    """One step: the detected cuts confirmed, then the auto splits placed from every confirmed cut.
    Splits are only replaced while no chunk has a prompt or a take; otherwise they're kept."""
    added = merge_detected(plan, found, confirmed=True)
    busy = any(c.get("takes") or (c.get("prompt") or "").strip() for c in plan.get("chunks", []) if not sp.is_kept(c))
    res = {"added": added, "splits": None, "kept_splits": busy}
    if not busy:
        res["splits"] = apply_op(plan, {"op": "auto_splits"})["splits"]
    return res


def do_detect_cuts(body, out_dir=None):
    jd, pp = job_paths(body.get("job", ""), out_dir)
    plan = sp.load_plan(pp)
    src = plan["source"]
    found = detect_cuts(src["path"], int(round(float(src["fps"]))), float(body.get("threshold", SCENE_THRESHOLD)))
    out = {"added": [], "splits": None, "kept_splits": False}
    if body.get("plan"):
        plan = sp.update_plan(pp, lambda p: out.update(detect_and_plan(p, found)), body.get("rev"))
    else:
        plan = sp.update_plan(pp, lambda p: out["added"].extend(merge_detected(p, found)), body.get("rev"))
    return dict(out, ok=True, found=found, rev=plan["rev"])


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


def do_op(body, out_dir=None):
    """The op route's body: create a job, or apply one edit under the plan lock."""
    job = body.get("job", "")
    jd, pp = job_paths(job, out_dir)
    if body.get("op") == "create":
        plan = create_job(job, body.get("source", ""), body.get("settings"), out_dir)
        return {"ok": True, "rev": plan["rev"], "plan": plan, "text": plan_text(plan)}
    if body.get("op") == "rescore":
        return do_rescore(body, out_dir)
    if body.get("op") == "source":
        def upd(p):
            if any(c.get("takes") for c in p.get("chunks", [])):
                raise sp.PlanError("the job has takes: start a new job for another source")
            p["source"] = source_facts(body.get("source", ""))
            sp.rebuild_chunks(p)
        plan = sp.update_plan(pp, upd, body.get("rev"))
        return {"ok": True, "rev": plan["rev"], "plan": plan, "text": plan_text(plan)}
    out = {}

    def upd(p):
        if body.get("op") in FILE_OPS:
            out.update(FILE_OPS[body["op"]](p, body, jd))
        else:
            out.update(apply_op(p, body))
    plan = sp.update_plan(pp, upd, body.get("rev"))
    return {"ok": True, "rev": plan["rev"], "result": out, "warnings": plan.get("warnings", []), "plan": plan,
            "text": plan_text(plan)}


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
        t, state = sp.effective_take(c)
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
        t, st = sp.effective_take(c)
        if t is not None and not sp.take_covers(t, *c["deliver"]):
            st = "range changed"
        elif t is not None and not os.path.isfile(os.path.join(jd, t["file"])):
            st = "missing file"
        r0, r1 = c["render"]
        d = {"chunk": c["id"], "state": st, "take": t and t["id"], "takes": len(c.get("takes") or []),
             "prompt": "empty" if not (c.get("prompt") or "").strip() else c.get("prompt_state") or "edited",
             "draft": bool((c.get("draft") or "").strip()),
             "mask": round(sp.mask_coverage(plan, r0, r1) / float(r1 - r0 + 1), 4),
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
            "quality": ss.legend(), "text": plan_text(plan), "source_view": _source_view_safe(plan, jd)}


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

def resolve_pins(plan, k):
    """Which pins chunk k gets, from the lineage at this moment: every anchored side whose
    neighbour has an effective take covering the K frames. {side: {"take", "frames"} | None},
    plus notes on the sides that get none."""
    chunks = plan["chunks"]
    c = chunks[k]
    K = int(sp._settings(plan.get("settings"))["anchors"])
    splits = {d["id"]: d for d in plan.get("splits", [])}
    pins, notes = {"start": None, "end": None}, []
    if K <= 0:
        return pins, ["anchor_frames is 0: no pins"]
    if sp.is_kept(c):
        return pins, ["kept as the original: never rendered"]
    r0, r1 = c["render"]
    left = splits.get(c.get("left"))
    if k > 0 and left and left["mode"] == sp.MODE_ANCHORED and sp.is_kept(chunks[k - 1]):
        pins["start"] = {"source": True, "frames": [r0, r0 + K - 1]}       # anchored onto the original
    elif k > 0 and left and left["mode"] == sp.MODE_ANCHORED:
        t, _st = sp.effective_take(chunks[k - 1])
        f = [r0, r0 + K - 1]
        if t is not None and sp.take_covers(t, *f):
            pins["start"] = {"take": t["id"], "frames": f}
        else:
            notes.append(f"start: {chunks[k - 1]['id']} has no take covering {f[0]}-{f[1]}: free start")
    if k + 1 < len(chunks):
        right = splits.get(chunks[k + 1].get("left"))
        if right and right["mode"] == sp.MODE_ANCHORED:
            f = [r1 - K + 1, r1]
            t, _st = sp.effective_take(chunks[k + 1])
            if int(c.get("held") or 0):
                notes.append("end: the render ends in held frames, so it can't be pinned at its end")
            elif sp.is_kept(chunks[k + 1]):
                pins["end"] = {"source": True, "frames": f}                    # anchored onto the original
            elif t is not None and sp.take_covers(t, *f):
                pins["end"] = {"take": t["id"], "frames": f}
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
            "options": dict(c.get("options") or {}, **(run.get("options") or {})),
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


def render_inputs(desc, job_dir, plan):
    """Decode the render's frames (+ held), its audio and its pins. All at execution."""
    src = desc["source"]
    path = src["path"]
    if not os.path.isfile(path):
        raise PlannerError(f"the source video is missing: {path}")
    fr = int(round(float(src["fps"])))
    info = tl.probe(path, fr)
    if int(info["frames"]) != int(src["frames"]):
        raise PlannerError(f"the source now has {info['frames']} frames at {fr} fps; the plan was made for {src['frames']}")
    w, h = int(info["width"]), int(info["height"])
    r0, r1 = desc["render"]
    n = r1 - r0 + 1
    held = int(desc["held"])
    frames = decode_frames(path, fr, r0, n)
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
    pins = {}
    for side in ("start", "end"):
        p = desc["pins"].get(side)
        if not p:
            pins[side] = torch.zeros((0, h, w, 3), dtype=torch.float32)
            continue
        if p.get("source"):                      # anchored onto the original: the source's own frames
            f0, f1 = p["frames"]
            pins[side] = decode_frames(path, fr, f0, f1 - f0 + 1, (w, h))
            continue
        _c, t = sp.find_take(plan, p["take"])
        tp = os.path.join(job_dir, t["file"])
        if not os.path.isfile(tp):
            raise PlannerError(f"pins for {desc['chunk']} ({side}): take {t['id']}'s file is missing: {tp}")
        f0, f1 = p["frames"]
        pins[side] = decode_frames(tp, fr, f0 - int(t["render"][0]), f1 - f0 + 1, (w, h))
    return frames, audio, pins, (w, h, fr)


def mark_descriptor(plan, plan_file, run):
    """A mark run's descriptor: the range whose source person mask is tracked and cached (a chunk's
    render range, or an explicit [a, b])."""
    src = plan["source"]
    if run.get("range"):
        r0, r1 = int(run["range"][0]), int(run["range"][1])
        cid = None
    else:
        c = _chunk(plan, run.get("chunk"))
        if sp.is_kept(c):
            raise PlannerError(f"chunk {c['id']} {c['deliver']} is kept as the original: it needs no mask")
        r0, r1 = c["render"]
        cid = c["id"]
    if not 0 <= r0 <= r1 < int(src["frames"]):
        raise PlannerError(f"mark range {r0}-{r1} is outside the source (0-{int(src['frames']) - 1})")
    return {"format": MARK_FORMAT, "job": plan["job"], "plan": plan_file, "plan_rev": plan.get("rev"),
            "chunk": cid, "range": [r0, r1], "frames": r1 - r0 + 1,
            "source": {k: src[k] for k in ("path", "frames", "fps", "width", "height", "audio")},
            "nonce": run.get("nonce") or uuid.uuid4().hex, "started": time.time()}


def cached_mask(plan, job_dir, r0, r1, held=0):
    """The cached source person mask over r0..r1 (+ held repeats) as a MASK [n, h, w], or (None, why)."""
    cover = sp.mask_cover(plan, r0, r1)
    if not cover:
        return None, f"no cached mask covers {r0}-{r1}"
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
                "ceiling_frames": ("INT", {"default": 260, "min": 5, "max": 2000, "step": 1, "tooltip":
                    "Warn above this render length (the tested ceiling at 1080p)."}),
                "conform_to_24fps": ("BOOLEAN", {"default": True, "tooltip":
                    "The render's reference audio on H3's 24 fps clock (frames stay 1:1). Takes keep "
                    "the original audio."}),
                "run": ("STRING", {"default": "", "tooltip":
                    "Set by the Planner's UI (or a script) before each queue: what this run does."}),
                "ui_state": ("STRING", {"default": "", "tooltip": "The strip's zoom and selection."}),
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
             ceiling_frames=260, conform_to_24fps=True, run="", ui_state=""):
        widgets = {"target_render_frames": target_render_frames, "overlap_frames": overlap_frames,
                   "anchor_frames": anchor_frames, "floor_frames": floor_frames, "ceiling_frames": ceiling_frames,
                   "conform_to_24fps": conform_to_24fps}
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
                same = _norm(tl.resolve_path(source.strip().strip('"'))) == _norm(plan["source"]["path"])
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
            out[:13] = [desc, frames, audio, desc["prompt"], desc["seed"], desc["length"], pins["start"], pins["end"],
                        w, h, desc["render_fps"], fr, json.dumps(desc["options"])]
            m, why = cached_mask(plan, jd, desc["render"][0], desc["render"][1], desc["held"])
            out[OUT_SOURCE_MASK] = m if m is not None else torch.zeros((1, 64, 64), dtype=torch.float32)
            out[OUT_HAS_MASK] = m is not None
            desc["source_mask"] = "cached" if m is not None else None
            _say(f"[SeamStitch] Swap Planner:   source mask: " + (f"cached ({m.shape[2]}x{m.shape[1]})" if m is not None
                                                                    else f"none ({why}): the render group tracks it"))
            pin_s = ", ".join(f"{s} {'the source' if p.get('source') else p['take']} {p['frames'][0]}-{p['frames'][1]}"
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
            _say(f"[SeamStitch] Swap Planner: {job}: mark {desc['chunk'] or ''} frames {desc['range'][0]}-"
                  f"{desc['range'][1]} ({desc['frames']}): the source person mask, tracked and cached")
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
