"""SeamStitch Swap Mask: closes a mark run. Caches the source's person mask over a range, so the
marking can be checked on the Planner's mask row before anything renders, and render runs reuse
it instead of tracking again.

    masks/<id>_<a>-<b>/
        mask.mkv       the mask, lossless (FFV1, white = person), frame i = source frame a + i,
                       at the size it was tracked at (the render group's 1 MP SAM3 copy)
        preview.mp4    what to look at: the marked guide (the person inverted), or the mask itself

Frames must come back 1:1 with the range or the mask is refused. Registered in plan.json under
"masks"; the newest segment wins where two overlap (swap_plan.mask_cover).
"""

import os
import shutil
import time

try:
    from . import swap_plan as sp
    from . import swap_planner as spl
    from . import swap_take as st
except ImportError:  # imported as a top-level module (tests, tools)
    import swap_plan as sp
    import swap_planner as spl
    import swap_take as st


class MaskError(Exception):
    pass


def save_mask(mark, mask, preview=None, save_preview=True):
    """Everything the node does. Returns the plan's mask entry."""
    if not isinstance(mark, dict) or mark.get("format") != spl.MARK_FORMAT:
        raise MaskError("mark_chunk is not a Swap Planner mark run: wire the Planner's mark_chunk output")
    plan_file = mark["plan"]
    plan = sp.load_plan(plan_file)
    if plan.get("job") != mark["job"]:
        raise MaskError(f"this mark run is from job {mark['job']!r}, the plan at {plan_file} is {plan.get('job')!r}")
    a, b = mark["range"]
    n = int(mark["frames"])
    if int(mask.shape[0]) != n:
        raise MaskError(f"1:1 broken: {int(mask.shape[0])} mask frames for the {n}-frame range {a}-{b}: refused")
    job = os.path.dirname(plan_file)
    fr = int(round(float(mark["source"]["fps"])))
    h, w = int(mask.shape[1]), int(mask.shape[2])
    with sp._LOCK:
        ids = [int(str(m.get("id", "m0"))[1:]) for m in plan.get("masks", []) + [e["mask"] for e in plan.get("trash", [])
                                                                                 if e.get("kind") == "mask"]
               if str(m.get("id", "")).startswith("m") and str(m.get("id"))[1:].isdigit()]
        mid = f"m{(max(ids) if ids else 0) + 1:03d}"
        d = os.path.join(job, "masks", f"{mid}_{a}-{b}")
        while os.path.exists(d):
            mid = f"m{int(mid[1:]) + 1:03d}"
            d = os.path.join(job, "masks", f"{mid}_{a}-{b}")
        os.makedirs(d)
    rel = os.path.relpath(d, job).replace("\\", "/")
    try:
        wrote = st.encode(st.mask_frames_u8(mask, n), w, h, fr, os.path.join(d, "mask.mkv"))
        if wrote != n:
            raise MaskError(f"wrote {wrote} mask frames of {n}")
        prev = None
        if save_preview:
            if preview is not None and int(preview.shape[0]) == n:
                pw, ph = int(preview.shape[2]) // 2 * 2, int(preview.shape[1]) // 2 * 2
                frames = (st.to_u8(preview[i])[:ph, :pw] for i in range(n))
            else:
                pw, ph = w // 2 * 2, h // 2 * 2
                frames = (f[:ph, :pw] for f in st.mask_frames_u8(mask, n))
            st.encode(frames, pw, ph, fr, os.path.join(d, "preview.mp4"), st.CODEC_H264, 20)
            prev = f"{rel}/preview.mp4"
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    # frames where SAM3 found nobody: the mask row marks them, so a lost track is seen before a render
    empty = [a + i for i in range(n) if float(mask[i].max()) <= 0.5]
    entry = {"id": mid, "range": [a, b], "frames": n, "file": f"{rel}/mask.mkv", "preview": prev, "size": [w, h],
             "empty": empty,
             "chunk": mark.get("chunk"), "created": sp.now(), "nonce": mark.get("nonce"),
             "seconds": round(time.time() - float(mark.get("started") or time.time()))}

    def add(p):
        p.setdefault("masks", []).append(entry)
    saved = sp.update_plan(plan_file, add)
    entry["plan_rev"] = saved["rev"]
    return entry


class SeamStitchSwapMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mark_chunk": (spl.MARK_TYPE, {"tooltip": "The Swap Planner's mark_chunk output."}),
                "mask": ("MASK", {"tooltip": "The source person mask over the run's range (SAM3), frames 1:1."}),
                "save_preview": ("BOOLEAN", {"default": True, "tooltip":
                    "preview.mp4 for the Planner's mask row: the marked guide if wired, else the mask."}),
            },
            "optional": {
                "preview": ("IMAGE", {"tooltip": "What to show on the mask row: the marked guide (the person inverted)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mask_id",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "SeamStitch/Swap"
    DESCRIPTION = ("Caches a mark run's source person mask in the job (lossless), so the Planner's mask row shows "
                   "it before any render and render runs reuse it instead of tracking again.")

    def save(self, mark_chunk, mask, save_preview=True, preview=None):
        e = save_mask(mark_chunk, mask, preview, save_preview)
        print(f"[SeamStitch] Swap Mask: {mark_chunk['job']}: {e['id']} frames {e['range'][0]}-{e['range'][1]} "
              f"({e['size'][0]}x{e['size'][1]}) cached")
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("seamstitch_swap_plan", {"job": mark_chunk["job"], "rev": e["plan_rev"]})
        except Exception:
            pass
        return {"ui": {"seamstitch_swap_mask": [{"job": mark_chunk["job"], "mask": e["id"], "range": e["range"]}]},
                "result": (e["id"],)}
