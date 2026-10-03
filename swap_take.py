"""SeamStitch Swap Take: closes every render. Saves the take, records its lineage, scores it,
builds its review clip and registers it in the job's plan.

    chunks/<chunk>/<take>/
        take.mkv          FFV1 RGB + FLAC at the SOURCE frame rate, with the chunk's ORIGINAL audio
                          (or take.mp4, h264, by choice)
        proxy.mp4         H.264 for the browser
        review.mp4        the chunk +- 2 s with both joins, as if this take were chosen
        take.json         a copy of the take's plan entry (plan.json can be rebuilt from these)
        prompt.txt, api_prompt.json, workflow.json
        mask_src.mkv, mask_out.mkv      SAM3 person masks (source, output), for scoring
        marked_guide.mp4  what H3 saw, when marking was on

Frames must come back 1:1 (output frames = the render length) or the take is refused; held
frames (the hold fill) are dropped. Nothing here picks a take: the panel offers "choose this
take" and "keep current", and Kay chooses.
"""

import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
import torch

try:
    from . import swap_plan as sp
    from . import swap_planner as spl
    from . import swap_assemble as sa
    from . import timeline as tl
    from . import result_preview as rp
    from . import swap_scores as ss
except ImportError:  # imported as a top-level module (tests, tools)
    import swap_plan as sp
    import swap_planner as spl
    import swap_assemble as sa
    import timeline as tl
    import result_preview as rp
    import swap_scores as ss

CODEC_LOSSLESS = tl.CODEC_LOSSLESS
CODEC_H264 = tl.CODEC_H264
REVIEW_SECONDS = 2


class TakeError(Exception):
    pass


def to_u8(img):
    """One IMAGE frame (float HxWx3) as uint8 the way VHS writes it (x*255 + 0.5, clipped)."""
    a = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _ff():
    return tl._ffmpeg_exe()


def encode(frames, w, h, fr, path, codec=CODEC_LOSSLESS, crf=12, wav=None, sr=48000):
    """uint8 RGB frames -> path, with the Timeline's cut arguments: FFV1 RGB planes + FLAC
    (lossless), or BT.709 h264 + AAC. Returns the frame count written."""
    lossless = codec != CODEC_H264
    cmd = [_ff(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fr), "-i", "-"]
    apath = None
    if wav is not None:
        apath = tl._write_f32(wav, os.path.dirname(os.path.abspath(path)))
        cmd += ["-f", "f32le", "-ar", str(int(sr)), "-ac", str(int(np.asarray(wav).shape[0])), "-i", apath,
                "-map", "0:v", "-map", "1:a"]
    if lossless:
        cmd += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", "-g", "1", "-slices", "16", "-slicecrc", "1",
                "-color_primaries", "bt709", "-color_trc", "bt709"] + (["-c:a", "flac"] if apath else [])
    else:
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264",
                "-crf", str(int(crf)), "-preset", "medium", "-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-color_range", "tv"] + (["-c:a", "aac", "-b:a", "256k"] if apath else []) \
            + ["-movflags", "+faststart"]
    cmd.append(path)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    try:
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
            n += 1
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
        if apath:
            try:
                os.remove(apath)
            except OSError:
                pass
    if proc.returncode != 0:
        raise TakeError(f"ffmpeg failed writing {os.path.basename(path)}: {err.strip()[-800:]}")
    return n


def proxy(src, dst):
    """The browser copy (timeline.preview_copy's arguments)."""
    subprocess.run([_ff(), "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-color_range", "tv", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", dst],
                   check=True, capture_output=True)
    return dst


def mask_frames_u8(mask, n):
    """MASK [N, h, w] (0..1) -> n uint8 RGB frames (white = person), for a lossless mask video."""
    for i in range(n):
        m = mask[i].detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask[i])
        g = (np.clip(m, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        yield np.repeat(g[..., None], 3, axis=2)


def mask_list(mask, n):
    """MASK [N, h, w] (0..1) -> the first n frames as bool arrays."""
    out = []
    for i in range(n):
        m = mask[i].detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask[i])
        out.append(m > 0.5)
    return out


def _mouth_cache(job, r0, r1):
    return os.path.join(job, "cache", f"mouth_src_{r0:05d}-{r1:05d}.json")


def compute_scores(job, plan, r0, r1, out_frames, src_masks=None, out_masks=None, mouth=True):
    """Every chunk score of a take delivering source frames r0..r1 (swap_scores, design §4.5):
    following (with both masks), lost cuts per confirmed cut inside, the scene alarm (with the
    source mask), and mouth sync (with mediapipe and models/mediapipe/face_landmarker.task; the
    source's mouth series is cached in the job's cache/ by range). out_frames: RGB uint8,
    frame i = source frame r0 + i."""
    src = plan["source"]
    fr = int(round(float(src["fps"])))
    cuts = [c for c in sp.confirmed_cuts(plan.get("cuts")) if r0 < c <= r1]
    cache = _mouth_cache(job, r0, r1)
    ms = None
    if mouth and os.path.isfile(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                ms = json.load(f)
            if len(ms) != r1 - r0 + 1:
                ms = None
        except (OSError, ValueError):
            ms = None
    src_frames = tl._iter_frames(src["path"], fr, r0, r1 + 1)
    scores, extras = ss.score_frames(out_frames, src_frames, src_masks, out_masks, cuts, r0, fr, mouth=mouth,
                                     mouth_src=ms)
    if extras.get("mouth_src") is not None and ms is None:
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(extras["mouth_src"], f)
        except OSError:
            pass
    scores["scored"] = sp.now()
    return scores


def _take_masks(tdir, n, fr):
    """The take's saved SAM3 masks (source, output) as bool frames, or None where missing."""
    out = []
    for name in ("mask_src.mkv", "mask_out.mkv"):
        p = os.path.join(tdir, name)
        out.append([f[..., 0] > 127 for f in tl._iter_frames(p, fr, 0, n)] if os.path.isfile(p) else None)
    return out


def rescore_take(plan_file, take_id, mouth=True):
    """Recompute a saved take's scores from its own files (the take, its masks, the source), e.g.
    for takes saved before B3. Writes them into plan.json and take.json; returns the scores."""
    plan = sp.load_plan(plan_file)
    job = os.path.dirname(plan_file)
    _c, t = sp.find_take(plan, take_id)
    r0, r1 = (int(x) for x in t["render"])
    n = r1 - r0 + 1
    path = os.path.join(job, t["file"])
    tdir = os.path.dirname(path)
    fr = int(round(float(plan["source"]["fps"])))
    sm, om = _take_masks(tdir, n, fr)
    scores = compute_scores(job, plan, r0, r1, tl._iter_frames(path, fr, 0, n), sm, om, mouth)

    def upd(p):
        _cc, tt = sp.find_take(p, take_id)
        tt["scores"] = scores
    sp.update_plan(plan_file, upd)
    tj = os.path.join(tdir, "take.json")
    try:
        with open(tj, encoding="utf-8") as f:
            d = json.load(f)
        d["scores"] = scores
        with open(tj, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
    except (OSError, ValueError):
        pass
    return scores


def settings_summary(api_prompt):
    """What the render group ran with, read off the API prompt (a summary; api_prompt.json has all)."""
    s = {}
    if not isinstance(api_prompt, dict):
        return s
    loras, steps = [], []
    for n in api_prompt.values():
        ct, ins = n.get("class_type"), n.get("inputs", {})
        if ct == "UNETLoader":
            s["model"] = ins.get("unet_name")
        elif ct in ("LoraLoaderModelOnly", "LoraLoader"):
            loras.append(ins.get("lora_name"))
        elif ct == "BasicScheduler":
            steps.append([ins.get("scheduler"), ins.get("steps"), ins.get("denoise")])
        elif ct == "MiniMaxH3ReferenceToVideo":
            s["pass1"] = [ins.get("width"), ins.get("height")]
            s["ref_image_size"] = ins.get("ref_image_size")
        elif ct == "BlockSparseAttention":
            s["sla_keep"] = ins.get("selection.keep_percent")
        elif ct == "MinimaxH3LatentUpscaler3D":
            s["upscale_mp"] = ins.get("mode.megapixels")
    if loras:
        s["loras"] = loras
    if steps:
        s["schedulers"] = steps
    return s


def _reserve_dir(plan_file, chunk_id, want):
    """The take id and its folder: the Planner's id unless something took it meanwhile."""
    job = os.path.dirname(plan_file)
    with sp._LOCK:
        plan = sp.load_plan(plan_file)
        c = sp.find_chunk(plan, chunk_id)
        used = {t["id"] for t in c.get("takes", [])}
        tid = want
        while tid in used or os.path.exists(os.path.join(job, "chunks", chunk_id, tid.split("-")[-1])):
            n = int(tid.rsplit("-t", 1)[1]) + 1
            tid = f"{chunk_id}-t{n:03d}"
        d = os.path.join(job, "chunks", chunk_id, tid.split("-")[-1])
        os.makedirs(d)
    return tid, d


def save_take(chunk, images, source_mask=None, output_mask=None, marked_guide=None, take_codec=CODEC_LOSSLESS,
              crf=12, score=True, mouth_sync=True, review_clip=True, prompt=None, extra_pnginfo=None, progress=None):
    """Everything the node does. Returns (take entry, take folder, report)."""
    t0 = time.time()
    desc = chunk
    if not isinstance(desc, dict) or desc.get("format") != spl.CHUNK_FORMAT:
        raise TakeError("chunk is not a Swap Planner chunk: wire the Planner's chunk output")
    plan_file = desc["plan"]
    plan = sp.load_plan(plan_file)
    if plan.get("job") != desc["job"]:
        raise TakeError(f"this chunk is from job {desc['job']!r}, the plan at {plan_file} is {plan.get('job')!r}")
    c = sp.find_chunk(plan, desc["chunk"])
    job = os.path.dirname(plan_file)

    n = int(images.shape[0])
    L, held = int(desc["length"]), int(desc["held"])
    if n != L:
        raise TakeError(f"1:1 broken: the render gave {n} frames for chunk {desc['chunk']}'s {L}-frame render "
                        f"(frames {desc['render'][0]}-{desc['render'][1]}" + (f" + {held} held" if held else "")
                        + "): the take is refused")
    keep = L - held
    r0, r1 = desc["render"]
    assert keep == r1 - r0 + 1
    src = desc["source"]
    fr = int(round(float(src["fps"])))
    W, H = int(src["width"]), int(src["height"])
    flags = []
    ih, iw = int(images.shape[1]), int(images.shape[2])
    resize = (iw, ih) != (W, H)
    if resize:
        flags.append({"code": "resized", "text": f"the render came back {iw}x{ih}; resized to the source's {W}x{H}"})
        print(f"[SeamStitch] Swap Take: warning - {flags[-1]['text']}")

    def frames_u8():
        import cv2
        for i in range(keep):
            f = to_u8(images[i])
            if resize:
                f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA if iw > W else cv2.INTER_CUBIC)
            yield f

    tid, tdir = _reserve_dir(plan_file, desc["chunk"], desc["take"])
    rel = os.path.relpath(tdir, job).replace("\\", "/")
    lossless = take_codec != CODEC_H264
    take_name = "take.mkv" if lossless else "take.mp4"
    take_path = os.path.join(tdir, take_name)
    try:
        cut = {"pieces": [{"path": src["path"], "enter": r0, "exit": r1 + 1}], "frames": keep}
        wav, sr = tl.cut_audio(cut, fr) if src.get("audio") else (None, 48000)
        wrote = encode(frames_u8(), W, H, fr, take_path, take_codec, crf, wav, sr)
        got = tl.probe(take_path, fr)["frames"]
        if wrote != keep or got != keep:
            raise TakeError(f"take {tid}: wrote {wrote} frames, the file holds {got}, expected {keep}")
        prox = os.path.join(tdir, "proxy.mp4")
        if lossless:
            proxy(take_path, prox)
        else:
            shutil.copy2(take_path, prox)

        # the extras: a failure here is flagged, never allowed to cost the render
        files = {}
        for name, m in (("mask_src.mkv", source_mask), ("mask_out.mkv", output_mask)):
            if m is None:
                continue
            if int(m.shape[0]) != L:
                flags.append({"code": "mask_frames", "text": f"{name}: {int(m.shape[0])} mask frames for a {L}-frame render"})
                continue
            try:
                encode(mask_frames_u8(m, keep), int(m.shape[2]), int(m.shape[1]), fr, os.path.join(tdir, name))
                files[name.split(".")[0]] = f"{rel}/{name}"
            except Exception as e:
                flags.append({"code": "mask_failed", "text": f"{name} not saved: {e}"})
        opts = desc.get("options") or {}
        if marked_guide is not None and spl._parse_value(opts.get("mark", True)) is True:
            try:
                g = marked_guide
                gw, gh = int(g.shape[2]) // 2 * 2, int(g.shape[1]) // 2 * 2
                encode((to_u8(g[i])[:gh, :gw] for i in range(min(keep, int(g.shape[0])))), gw, gh, fr,
                       os.path.join(tdir, "marked_guide.mp4"), CODEC_H264, 16)
                files["marked_guide"] = f"{rel}/marked_guide.mp4"
            except Exception as e:
                flags.append({"code": "marked_guide_failed", "text": f"marked_guide.mp4 not saved: {e}"})

        scores = {}
        if score:
            try:
                sm = mask_list(source_mask, keep) if source_mask is not None and "mask_src" in files else None
                om = mask_list(output_mask, keep) if output_mask is not None and "mask_out" in files else None
                scores = compute_scores(job, plan, r0, r1, frames_u8(), sm, om, bool(mouth_sync))
            except Exception as e:
                flags.append({"code": "score_failed", "text": f"scores not computed: {e}"})
                print(f"[SeamStitch] Swap Take: scoring failed: {e}")

        with open(os.path.join(tdir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(desc["prompt"])
        if prompt is not None:
            with open(os.path.join(tdir, "api_prompt.json"), "w", encoding="utf-8") as f:
                json.dump(prompt, f)
        wf = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        if wf is not None:
            with open(os.path.join(tdir, "workflow.json"), "w", encoding="utf-8") as f:
                json.dump(wf, f)

        settings = settings_summary(prompt)
        settings.update(mark=spl._parse_value(opts.get("mark", True)), conform=bool(desc.get("conform")),
                        options=opts, anchors=desc.get("anchors"))
        take = {"id": tid, "created": sp.now(), "state": "ok", "seed": desc["seed"], "render": [r0, r1],
                "held": held, "length": L, "frames": keep, "splits": desc["splits"], "pins": desc["pins"],
                "file": f"{rel}/{take_name}", "proxy": f"{rel}/proxy.mp4", "take_codec": take_codec,
                "prompt_sha1": desc["prompt_sha1"], "settings": settings, "files": files, "scores": scores,
                "gpu_s": round(t0 - float(desc.get("started") or t0)), "save_s": None, "flags": flags,
                "nonce": desc.get("nonce"), "planner_rev": desc.get("plan_rev")}
    except BaseException:
        shutil.rmtree(tdir, ignore_errors=True)
        raise

    def append(p):
        cc = sp.find_chunk(p, desc["chunk"])
        if list(cc["render"]) != [r0, r1]:
            take["flags"].append({"code": "range_changed", "text": f"the chunk renders {cc['render']} now; this take "
                                                                    f"renders {[r0, r1]} (a split moved during the render)"})
        cc.setdefault("takes", []).append(take)
        if (cc.get("rendering") or {}).get("nonce") == desc.get("nonce"):
            cc.pop("rendering", None)
            cc["state"] = "takes"
    sp.update_plan(plan_file, append)

    report = {"take": tid, "chunk": desc["chunk"], "job": desc["job"], "frames": keep, "file": take_path,
              "scores": scores, "flags": take["flags"], "pins": desc["pins"], "joins": []}
    if review_clip:
        try:
            p2 = sp.load_plan(plan_file)
            sp.find_chunk(p2, desc["chunk"])["chosen"] = tid
            N = int(src["frames"])
            d0, d1 = desc["deliver"]
            win = (max(0, d0 - REVIEW_SECONDS * fr), min(N - 1, d1 + REVIEW_SECONDS * fr))
            rv = sa.assemble(plan_file, int(desc.get("hand_back") or 12), "video/h264-mp4", 12, "yuv420p",
                             window=win, out_file=os.path.join(tdir, "review.mp4"), plan=p2, write_plan=False,
                             progress=progress)
            report["joins"] = rv["joins"]
            report["review"] = {"file": f"{rel}/review.mp4", "window": rv["window"], "flags": rv["flags"]}
        except Exception as e:
            report["review_error"] = str(e)
            print(f"[SeamStitch] Swap Take: review clip failed: {e}")
    take["save_s"] = round(time.time() - t0)
    take["review"] = report.get("review")
    take["joins"] = [{k: r.get(k) for k in ("split", "frame", "type", "splice", "repair", "override", "left_take",
                                             "right_take", "stale", "frame_luma", "char_luma", "join_ratio",
                                             "join_verdict", "follow", "flags", "verdict")} for r in report["joins"]]

    def finish(p):
        _c, t = sp.find_take(p, tid)
        t.update(review=take["review"], joins=take["joins"], save_s=take["save_s"])
    plan = sp.update_plan(plan_file, finish)
    with open(os.path.join(tdir, "take.json"), "w", encoding="utf-8") as f:
        json.dump(take, f, indent=1)
    report["plan_rev"] = plan["rev"]
    return take, tdir, report


class SeamStitchSwapTake:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk": (spl.CHUNK_TYPE, {"tooltip": "The Swap Planner's chunk output."}),
                "images": ("IMAGE", {"tooltip": "The render, at the source size, frames 1:1 with the chunk's render length."}),
                "take_codec": ([CODEC_LOSSLESS, CODEC_H264], {"default": CODEC_LOSSLESS, "tooltip":
                    "lossless (ffv1): RGB planes + FLAC, about 0.4-0.5 GB per 209 frames at 1080p; pins and the "
                    "assembly read the exact model output. h264: crf below, one lossy generation."}),
                "crf": ("INT", {"default": 12, "min": 0, "max": 51, "step": 1}),
                "score": ("BOOLEAN", {"default": True, "tooltip":
                    "Score the take (design §4.5): following (pose IoU of the two SAM3 person masks), lost cuts and "
                    "the scene alarm. Information for you: nothing picks a take."}),
                "mouth_sync": ("BOOLEAN", {"default": True, "tooltip":
                    "Mouth sync, a secondary signal (never a gate). Needs mediapipe and "
                    "models/mediapipe/face_landmarker.task; n/a without them."}),
                "review_clip": ("BOOLEAN", {"default": True, "tooltip":
                    "review.mp4: the chunk +- 2 s with both joins as if this take were chosen."}),
            },
            "optional": {
                "source_mask": ("MASK", {"tooltip": "SAM3 person mask of the source frames (the marking group's)."}),
                "output_mask": ("MASK", {"tooltip": "SAM3 person mask of the render (group Swap Score)."}),
                "marked_guide": ("IMAGE", {"tooltip": "What H3 saw as its guide; saved when the chunk's mark option is on."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("take_path", "take_id", "report")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "SeamStitch/Swap"
    DESCRIPTION = ("Saves a render as a take of its chunk (lossless, source frame rate, original audio), with its "
                   "lineage, masks, scores and a review clip of both joins. Refuses anything that isn't 1:1.")

    def save(self, chunk, images, take_codec=CODEC_LOSSLESS, crf=12, score=True, mouth_sync=True, review_clip=True,
             source_mask=None, output_mask=None, marked_guide=None, prompt=None, extra_pnginfo=None):
        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pass

        def progress(i, n):
            if pbar is not None and i % 10 == 0:
                pbar.update_absolute(int(100 * i / n), 100)

        take, tdir, report = save_take(chunk, images, source_mask, output_mask, marked_guide, take_codec, crf, score,
                                       mouth_sync, review_clip, prompt, extra_pnginfo, progress)
        lines = [f"{take['id']}: {take['frames']} frames {take['render'][0]}-{take['render'][1]}, seed {take['seed']}"
                 + (f", pose IoU {take['scores']['pose_iou']}" if take["scores"].get("pose_iou") is not None else "")
                 + (f", mouth {take['scores']['mouth']}" if take["scores"].get("mouth") is not None else "")]
        quality = ss.chunk_flags(take["scores"])
        for j in report["joins"]:
            fl = j.get("frame_luma") or {}
            cl = j.get("char_luma") or {}
            lines.append(f"  {j['split']} @ {j['frame']}: {j['type']}, splice {j['splice']}, {j.get('override') or j['repair']}; "
                         f"jump {fl.get('at_splice')} / char {cl.get('at_splice')}, ratio {j.get('join_ratio')}")
        print("[SeamStitch] Swap Take: " + "\n[SeamStitch] Swap Take: ".join(lines))
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("seamstitch_swap_plan", {"job": chunk["job"], "rev": report.get("plan_rev")})
        except Exception:
            pass
        rv = os.path.join(tdir, "review.mp4")
        ui = {"job": chunk["job"], "chunk": chunk["chunk"], "take": take["id"], "seed": take["seed"],
              "review": rp._view_params(rv) if os.path.isfile(rv) else None, "review_path": rv,
              "proxy": rp._view_params(os.path.join(tdir, "proxy.mp4")), "scores": take["scores"], "quality": quality,
              "joins": take["joins"], "flags": take["flags"], "fps": int(round(float(chunk["source"]["fps"]))),
              "window": (report.get("review") or {}).get("window"), "deliver": list(chunk["deliver"]),
              "text": "\n".join(lines)}
        return {"ui": {"seamstitch_swap_take": [ui]},
                "result": (os.path.join(tdir, os.path.basename(take["file"])), take["id"], json.dumps(report))}
