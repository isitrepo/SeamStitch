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
take" and "keep current", and the user chooses.
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


# pin level match (2026-10-15): a pinned frame comes back out of H3 about 2.3 levels darker than the take it was pinned
# to (the 60 s test job, c2 -> c7: -2.7, -2.0, -2.2, -2.4, -2.2 at each join, flat inside each chunk; the VAE round trip
# SatoDive reports as ~2.4% per hop), and each chunk pins to the last, so the picture slid ~15 levels by c7. The take's
# pinned frames show the same moment as the pin, so the take is scaled, per channel, to match them before it is saved:
# the next chunk then pins to a levelled frame.
LEVEL_CLAMP = (0.8, 1.25)       # a larger difference is not a pin round trip: left alone, and flagged
LEVEL_FLAG = 1.0                # luma levels: a take starting further off its pin than this, uncorrected, is flagged


def _luma(rgb_mean):
    return float(0.299 * rgb_mean[0] + 0.587 * rgb_mean[1] + 0.114 * rgb_mean[2])


def pin_frames_u8(plan, job, desc, side, size):
    """The frames a side of the take was pinned to, decoded from the pinned take at the source size (uint8), with the
    take-relative frames they pin; None for no pin, a pin onto the original (anchored onto the source, not a take) or a
    missing file."""
    p = (desc.get("pins") or {}).get(side)
    if not p or p.get("source") or not p.get("take"):
        return None
    _c, t = sp.find_take(plan, p["take"])
    if t is None or not os.path.isfile(os.path.join(job, t["file"])):
        return None
    f0, f1 = (int(x) for x in p["frames"])
    fr = int(round(float(desc["source"]["fps"])))
    out = []
    for f in tl._iter_frames(os.path.join(job, t["file"]), fr, f0 - int(t["render"][0]), f1 - int(t["render"][0]) + 1):
        if (f.shape[1], f.shape[0]) != tuple(size):
            import cv2
            f = cv2.resize(f, tuple(size), interpolation=cv2.INTER_AREA)
        out.append(f)
    r0 = int(desc["render"][0])
    return (out, list(range(f0 - r0, f1 - r0 + 1))) if out else None


def level_gains(out_frames, pin_frames, masks=None, contrast=True):
    """The per-channel transform that brings the take's pinned frames to the pin's: T = [scale, offset] (3 each), a
    pixel becoming pixel * scale + offset. With contrast, the mean AND the spread (standard deviation) are matched
    (scale = pin spread / take spread): the 60 s test job's c3 -> c8 (2026-10-16) kept its level with the mean alone but
    the couch's contrast still rose ~1.5-2 a chunk, darker lines and muddier weave, each chunk sharpening its pin a
    little more. Without, the mean only (scale = pin mean / take mean, offset 0). Returns (T, luma step take - pin,
    clamped). masks: one bool array per frame, the pixels to measure (one region: the room or the person)."""
    def stats(frames):
        mu, sd = [], []
        for k, f in enumerate(frames):
            a = np.asarray(f, np.float64).reshape(-1, 3)
            if masks is not None:
                a = a[np.asarray(masks[k]).reshape(-1)]
            mu.append(a.mean(0))
            sd.append(a.std(0))
        return np.mean(mu, 0), np.mean(sd, 0)
    (o, so), (q, sq) = stats(out_frames), stats(pin_frames)
    step = _luma(o) - _luma(q)
    mean_raw = q / np.maximum(o, 1.0)
    if contrast:
        raw = sq / np.maximum(so, 0.5)
        scale = np.clip(raw, *LEVEL_CLAMP)
        offset = q - scale * o
        clamped = bool(np.any(scale != raw) or np.any(np.clip(mean_raw, *LEVEL_CLAMP) != mean_raw))
    else:
        scale = np.clip(mean_raw, *LEVEL_CLAMP)
        offset = np.zeros(3)
        clamped = bool(np.any(scale != mean_raw))
    return np.stack([scale, offset]), step, clamped


# pin detail match (2026-10-29): the same pin round trip that darkens a frame also softens it. The 2642-frame test job
# (14 chunks, 0.75 MP): frame 0 of every take kept only ~0.78 of its pin's fine detail (Laplacian variance, room patch;
# 0.64-0.94 over 13 joins), flat inside each chunk, so the room's texture fell ~75% by the last chunk while the mean and
# the spread (level_gains) held. The take's fine band (luma minus its own blur) is boosted so its pinned frames carry the
# pin's band energy; the next chunk then pins to a frame that kept its detail.
DETAIL_SIGMA = 1.2              # px: the blur that splits the fine band off
DETAIL_CLAMP = (0.0, 0.8)       # k of f + k * band(f): at most ~1.8x the band, never softer than rendered
DETAIL_MIN = 0.01               # a k below this is not worth a pass
DETAIL_LOCAL = 48               # px: the blur of the local band-energy ratio (the room's wall lost less than its sofa:
DETAIL_LOCAL_SPAN = (0.5, 1.5)  # one gain for the room over-sharpened the wall 25-45% while the sofa came back ~8% short)
# the anchor (2026-10-29): matching each chunk to its pin only stops the loss at the join, not the shortfall left at every
# join (the 14-chunk test job, pin-matched: wall 76 -> 48, sofa 209 -> 105 by the last chunk). The target is then the job's
# first full chunk (the one rendered with no pin, so no round trip in it): its room's band energy per pixel.
ANCHOR_MIN_LENGTH = 100         # frames: the reference is the first chunk with a chosen take at least this long and no pins at all
PERSON_Q = (1, 5, 15, 30, 50, 70, 85, 95, 99)   # the person's colour anchor: per channel, her pixels' brightness quantiles
PERSON_MAX = 24                 # levels: the most a quantile is moved (a pose change is not a fade)
PERSON_MIN_PX = 3000            # her pixels needed on each side to measure
ANCHOR_SAMPLES = 8              # frames of the reference take the target is measured on


def person_lut(frames, masks, ref_frames, ref_masks):
    """The per-channel LUT [3, 256] that moves the person's brightness quantiles (per channel) in frames toward the
    reference's, each by at most PERSON_MAX levels, and the mean absolute move; None when too few of her pixels are
    measured. Her colour drifts down a chain of pinned chunks (the 14-chunk test job: her red coat fell ~56 -> ~40 and her
    saturation 141 -> 108 by the last chunk) in a way a mean + spread match to the pin cannot hold; the quantiles
    separate the dark reds from the skin, so they are moved apart."""
    import cv2

    def pix(fs, ms):
        out = []
        for f, m in zip(fs, ms):
            sel = cv2.erode(np.asarray(m).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            if sel.any():
                out.append(f[sel][:: max(1, int(sel.sum()) // 40000)])
        return np.concatenate(out) if out else np.zeros((0, 3), np.uint8)

    a, r = pix(frames, masks), pix(ref_frames, ref_masks)
    if len(a) < PERSON_MIN_PX or len(r) < PERSON_MIN_PX:
        return None
    x = np.arange(256, dtype=np.float32)
    lut = np.zeros((3, 256), np.float32)
    moves = []
    for ch in range(3):
        qa = np.percentile(a[:, ch], PERSON_Q).astype(np.float32)
        qr = np.percentile(r[:, ch], PERSON_Q).astype(np.float32)
        d = np.clip(qr - qa, -PERSON_MAX, PERSON_MAX)
        qa, ui = np.unique(qa, return_index=True)
        d = d[ui]
        if len(qa) < 2:
            lut[ch] = x
            continue
        lut[ch] = np.clip(x + np.interp(x, qa, d), 0, 255)
        moves.append(float(np.abs(d).mean()))
    return np.round(lut).astype(np.uint8), float(np.mean(moves)) if moves else 0.0


def anchor_sample(plan, job, size):
    """(frames, room masks) sampled from the job's reference take for the detail anchor, or None: the first chunk (by
    render start) whose chosen take is >= ANCHOR_MIN_LENGTH frames, has no start pin and has a saved source mask."""
    import cv2
    best = None
    for c in plan.get("chunks") or []:
        if sp.is_kept(c) or not c.get("chosen"):
            continue
        _c, t = sp.find_take(plan, c["chosen"])
        if t is None or int(t.get("frames") or 0) < ANCHOR_MIN_LENGTH                 or (t.get("pins") or {}).get("start") or (t.get("pins") or {}).get("end"):
            continue
        if not (t.get("files") or {}).get("mask_src"):
            continue
        if best is None or int(t["render"][0]) < int(best["render"][0]):
            best = t
    if best is None:
        return None
    fr = int(round(float(plan["source"]["fps"])))
    n = int(best["frames"])
    pick = sorted({int(round(x)) for x in np.linspace(0, n - 1, ANCHOR_SAMPLES + 2)[1:-1]})
    frames, masks = [], []
    for i in pick:
        f = next(iter(tl._iter_frames(os.path.join(job, best["file"]), fr, i, i + 1)), None)
        m = next(iter(tl._iter_frames(os.path.join(job, best["files"]["mask_src"]), fr, i, i + 1)), None)
        if f is None or m is None:
            continue
        f = cv2.resize(f, tuple(size), interpolation=cv2.INTER_AREA) if (f.shape[1], f.shape[0]) != tuple(size) else f
        m = cv2.resize(m, tuple(size), interpolation=cv2.INTER_AREA) if (m.shape[1], m.shape[0]) != tuple(size) else m
        frames.append(f)
        masks.append(m[..., 0] < 128)               # the source mask is the person: the room is the rest
    return (frames, masks, best["id"], [~x for x in masks]) if frames else None


def _fine_band(lum, sigma=DETAIL_SIGMA):
    import cv2
    return lum - cv2.GaussianBlur(lum, (0, 0), sigma)


def detail_gain(out_frames, pin_frames, masks=None, sigma=DETAIL_SIGMA, spatial=False, anchor=None):
    """k such that out + k * band(out) has the pin's fine-band energy (luma, over the masked pixels of the given frames).
    The band of the boosted frame is band + k * band(band), so the energy is a quadratic in k, solved exactly. Returns
    (k clamped to DETAIL_CLAMP, the unclamped amplitude ratio pin / take, a map), or None when no pixel is measured. A
    take already as detailed as its pin gets k = 0. The map is None unless spatial and masks are given: then it is an
    [h, w] float32 k per pixel from the LOCAL ratio of the two band energies (a DETAIL_LOCAL px blur, over the masked
    pixels only), scaled so its centre is k and held within DETAIL_LOCAL_SPAN x k (the camera may move: the map is
    measured on the pinned frames and applied to the whole take). With anchor = (frames, room masks) the target energy is
    theirs, per pixel, instead of the pin's (the map's shape still comes from the pin)."""
    import cv2
    a = b = c = ep = 0.0
    n = 0
    eo_loc = ep_loc = den_loc = None
    for i, (o, p) in enumerate(zip(out_frames, pin_frames)):
        lo = cv2.cvtColor(np.ascontiguousarray(o), cv2.COLOR_RGB2GRAY).astype(np.float32)
        lp = cv2.cvtColor(np.ascontiguousarray(p), cv2.COLOR_RGB2GRAY).astype(np.float32)
        sel = slice(None)
        if masks is not None:
            # eroded: the person's edge moves between the pin and the render and is not texture
            sel = cv2.erode(np.asarray(masks[i]).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        bo = _fine_band(lo, sigma)
        bb = _fine_band(bo, sigma)
        bp = _fine_band(lp, sigma)
        if spatial and masks is not None:
            mf = sel.astype(np.float32)
            blur = lambda x: cv2.GaussianBlur(x.astype(np.float32), (0, 0), DETAIL_LOCAL)
            e1, e2, e3 = blur(bo * bo * mf), blur(bp * bp * mf), blur(mf)
            eo_loc = e1 if eo_loc is None else eo_loc + e1
            ep_loc = e2 if ep_loc is None else ep_loc + e2
            den_loc = e3 if den_loc is None else den_loc + e3
        bo, bb, bp = bo[sel].astype(np.float64), bb[sel].astype(np.float64), bp[sel].astype(np.float64)
        n += int(bo.size)
        a += float(np.sum(bo ** 2))
        b += float(np.sum(bo * bb))
        c += float(np.sum(bb ** 2))
        ep += float(np.sum(bp ** 2))
    if n == 0 or a <= 0 or c <= 0:
        return None
    ratio_pin = float(np.sqrt(ep / a))
    if anchor is not None:
        e_ref, n_ref = 0.0, 0
        for f, m in zip(anchor[0], anchor[1]):
            bf = _fine_band(cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY).astype(np.float32), sigma)
            sel = cv2.erode(np.asarray(m).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            e_ref += float(np.sum(bf[sel].astype(np.float64) ** 2))
            n_ref += int(sel.sum())
        if n_ref:
            ep = e_ref / n_ref * n
    ratio = float(np.sqrt(ep / a))
    if ep <= a:
        return 0.0, ratio, None
    k = float(np.clip((-b + np.sqrt(b * b - c * (a - ep))) / c, *DETAIL_CLAMP))
    kmap = None
    if eo_loc is not None and ratio_pin > 1.001 and k > 0:
        r = np.sqrt((ep_loc + 1.0) / (eo_loc + 1.0))
        r = np.where(den_loc < 0.02 * len(out_frames), ratio_pin, r)       # too little room near here: the centre
        kmap = np.clip(k * (r - 1.0) / (ratio_pin - 1.0), DETAIL_LOCAL_SPAN[0] * k, DETAIL_LOCAL_SPAN[1] * k)
        kmap = np.minimum(kmap, DETAIL_CLAMP[1]).astype(np.float32)
    return k, ratio, kmap


def boost_detail(f, k):
    """uint8 RGB frame f with its luma fine band boosted by k (a scalar or an [h, w] map): the same amount added to every
    channel, so colour is untouched and chroma noise is not amplified."""
    import cv2
    lum = cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY).astype(np.float32)
    delta = _fine_band(lum) * np.asarray(k, np.float32)
    return np.clip(f.astype(np.float32) + delta[..., None] + 0.5, 0, 255).astype(np.uint8)


LEVEL_MIN_REGION = 0.02         # a region smaller than this share of the frame takes the other region's gains


def _soft_mask(mask, i, size):
    """Frame i of a MASK [N, h, w] at size (w, h) as float32 0..1 (the last frame for held frames)."""
    import cv2
    m = mask[min(i, int(mask.shape[0]) - 1)]
    m = m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
    m = np.clip(m.astype(np.float32), 0, 1)
    if (m.shape[1], m.shape[0]) != tuple(size):
        m = cv2.resize(m, tuple(size), interpolation=cv2.INTER_LINEAR)
    return m


def gain_at(i, n, start, end):
    """The gain for take frame i of n: the start pin's at the start, the end pin's at the end, a ramp between when
    both are given, one constant when only one side is (save_take gives the start pin's only)."""
    if start is not None and end is not None:
        a = i / float(max(1, n - 1))
        return start + (end - start) * a
    return start if start is not None else end


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


def _mouth_cache(job, r0, r1, src_masks=None):
    """The source's mouth series cache. With a source mask (B8: her face only) it is keyed by the mask too, so
    a re-marked range or another target never reads a series of someone else's face."""
    if src_masks is None:
        return os.path.join(job, "cache", f"mouth_src_{r0:05d}-{r1:05d}.json")
    h = hashlib.sha1()
    for m in src_masks:
        h.update(np.packbits(ss.mask_thumb(m, 96, 54)).tobytes())
    return os.path.join(job, "cache", f"mouth_her_{r0:05d}-{r1:05d}_{h.hexdigest()[:10]}.json")


def compute_scores(job, plan, r0, r1, out_frames, src_masks=None, out_masks=None, mouth=True):
    """Every chunk score of a take delivering source frames r0..r1 (swap_scores, design §4.5):
    following (with both masks), lost cuts per confirmed cut inside, the scene alarm (with the
    source mask), and mouth sync (with mediapipe and models/mediapipe/face_landmarker.task; the
    source's mouth series is cached in the job's cache/ by range). out_frames: RGB uint8,
    frame i = source frame r0 + i."""
    src = plan["source"]
    fr = int(round(float(src["fps"])))
    cuts = [c for c in sp.confirmed_cuts(plan.get("cuts")) if r0 < c <= r1]
    cache = _mouth_cache(job, r0, r1, src_masks)
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


def output_mask_is_picked(api_prompt):
    """Does the API prompt's Swap Take get its output_mask from Swap Output Person (B8)? True when it can't be
    told (no prompt: a direct call)."""
    if not isinstance(api_prompt, dict):
        return True
    for n in api_prompt.values():
        if isinstance(n, dict) and n.get("class_type") == "SeamStitchSwapTake":
            src = (n.get("inputs") or {}).get("output_mask")
            if not isinstance(src, list) or not src:
                return True
            up = api_prompt.get(str(src[0])) or {}
            return up.get("class_type") == "SeamStitchSwapOutputPerson"
    return True


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
              crf=12, score=True, mouth_sync=True, review_clip=True, prompt=None, extra_pnginfo=None, progress=None,
              level_match=False, detail_match=False, detail_anchor=False, person_anchor=False):
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
    if sp.is_kept(c):
        raise TakeError(f"chunk {desc['chunk']} is kept as the original (switched on after this render was queued?): "
                        f"a kept chunk takes no takes; switch 'keep original' off in its panel first")
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
    crop = desc.get("crop")                 # an object job's cropped render: pasted back into the source frame here
    RW, RH = (crop[2] - crop[0], crop[3] - crop[1]) if crop else (W, H)
    flags = []
    ih, iw = int(images.shape[1]), int(images.shape[2])
    resize = (iw, ih) != (RW, RH)
    if resize:
        flags.append({"code": "resized", "text": f"the render came back {iw}x{ih}; resized to {RW}x{RH}"})
        print(f"[SeamStitch] Swap Take: warning - {flags[-1]['text']}")
    src_frames = None
    if crop:
        try:
            from . import swap_keep as sk
        except ImportError:
            import swap_keep as sk
        src_frames = sk._SeqFrames(src["path"], fr, 0)

    def raw_u8(i):
        import cv2
        f = to_u8(images[i])
        if resize:
            f = cv2.resize(f, (RW, RH), interpolation=cv2.INTER_AREA if iw > RW else cv2.INTER_CUBIC)
        if crop:
            full = src_frames.read(r0 + i).copy()
            full[crop[1]:crop[3], crop[0]:crop[2]] = f
            f = full
        return f

    # the pin level match: measured on every pinned take, applied when level_match is on
    level, gains, person_gains = {}, {"start": None, "end": None}, {"start": None, "end": None}
    detail = {"k": 0.0, "person_k": 0.0, "map": None}
    plut = None
    for side in ("start", "end"):
        try:
            pf = pin_frames_u8(plan, job, desc, side, (W, H))
        except Exception as e:
            print(f"[SeamStitch] Swap Take: level match ({side}) skipped: {e}")
            pf = None
        if not pf:
            continue
        pins_u8, idx = pf
        idx = [i for i in idx if 0 <= i < keep][:len(pins_u8)]
        if not idx:
            continue
        outs, pins_ = [raw_u8(i) for i in idx], pins_u8[:len(idx)]
        # by region when the person's mask is here: H3 darkens the room more than the kept person when it redraws a
        # pinned frame (the 60 s test job's c5, c6: one whole-frame gain matched the room and left her 1.4-1.5 lighter
        # each chunk), so the room and the person each get the gains of their own pixels
        g_person = None
        if source_mask is not None and int(source_mask.shape[0]) >= 1:
            pm = [_soft_mask(source_mask, i, (W, H)) > 0.5 for i in idx]
            share = float(np.mean([m.mean() for m in pm]))
            if LEVEL_MIN_REGION <= share <= 1 - LEVEL_MIN_REGION:
                g, step, clamped = level_gains(outs, pins_, [~m for m in pm])
                gp, step_p, clamped_p = level_gains(outs, pins_, pm)
                g_person = None if clamped_p else gp
            else:
                g, step, clamped = level_gains(outs, pins_)
        else:
            g, step, clamped = level_gains(outs, pins_)
        # the start pin only: the take is levelled to the chunk before it. An end pin is the next chunk's take, which can
        # be an older one from before a fix (the 60 s test job's c7, 2026-10-16: ramping to a stale, drifted c8 darkened c7's
        # end 12-18% and muddied the couch); its step is measured and flagged, and Assemble's lock smooths that join
        # (2026-10-30: a chunk with NO start pin, a backward chain from a free chunk, is levelled to its end pin: that pin is
        # the take just rendered and chosen, not a stale one)
        applied = bool(level_match) and not clamped and (side == "start" or "start" not in level)
        level[side] = {"step": round(step, 2), "gains": [round(float(x), 4) for x in g[0]],
                       "offsets": [round(float(x), 2) for x in g[1]], "applied": applied}
        if g_person is not None:
            level[side].update(person_step=round(step_p, 2), person_gains=[round(float(x), 4) for x in g_person[0]],
                               person_offsets=[round(float(x), 2) for x in g_person[1]])
        if applied:
            gains[side] = g
            person_gains[side] = g_person if g_person is not None else g
        # the fine detail the same round trip took, measured on the same pinned frames
        if side == "start":
            anch = None
            try:
                dm = None
                if source_mask is not None and int(source_mask.shape[0]) >= 1:
                    pm = [_soft_mask(source_mask, i, (W, H)) > 0.5 for i in idx]
                    share = float(np.mean([m.mean() for m in pm]))
                    if LEVEL_MIN_REGION <= share <= 1 - LEVEL_MIN_REGION:
                        dm = (pm, [~m for m in pm])
                if (detail_anchor or person_anchor) and dm:
                    try:
                        anch = anchor_sample(plan, job, (W, H))
                    except Exception as e:
                        print(f"[SeamStitch] Swap Take: detail anchor skipped: {e}")
                d_room = detail_gain(outs, pins_, dm[1] if dm else None, spatial=True,
                                     anchor=anch[:2] if anch and detail_anchor else None)
                d_person = detail_gain(outs, pins_, dm[0]) if dm else None
                if d_room is not None:
                    kr = d_room[0]
                    kp = d_person[0] if d_person is not None else kr
                    level[side]["detail"] = {"ratio": round(d_room[1], 3), "k": round(kr, 3), "person_k": round(kp, 3),
                                             "applied": bool(detail_match) and max(kr, kp) >= DETAIL_MIN}
                    if anch and detail_anchor:
                        level[side]["detail"]["anchor"] = anch[2]
                    if level[side]["detail"]["applied"]:
                        detail["k"], detail["person_k"], detail["map"] = kr, kp, d_room[2]
            except Exception as e:
                print(f"[SeamStitch] Swap Take: detail match skipped: {e}")
            if person_anchor and anch and source_mask is not None:
                try:
                    pick = sorted({int(round(v)) for v in np.linspace(0, keep - 1, ANCHOR_SAMPLES + 2)[1:-1]})
                    pg = person_gains["start"] if person_gains["start"] is not None else gains["start"]
                    fs, ms = [], []
                    for i in pick:
                        f = raw_u8(i)
                        if pg is not None:
                            f = np.clip(f.astype(np.float32) * pg[0].astype(np.float32) + pg[1].astype(np.float32) + 0.5,
                                        0, 255).astype(np.uint8)
                        fs.append(f)
                        ms.append(_soft_mask(source_mask, i, (W, H)) > 0.5)
                    res = person_lut(fs, ms, anch[0], anch[3])
                    if res is not None:
                        plut = res[0] if res[1] >= 1.0 else None
                        level[side]["person_anchor"] = {"moved": round(res[1], 2), "applied": plut is not None,
                                                        "ref": anch[2]}
                except Exception as e:
                    print(f"[SeamStitch] Swap Take: person colour anchor skipped: {e}")
        if side == "end" and abs(step) > LEVEL_FLAG and not applied:
            flags.append({"code": "level_end", "text": f"ends {abs(step):.1f} levels {'darker' if step < 0 else 'brighter'} "
                          f"than the next chunk's take it's pinned to: that take may be from before a fix; re-roll "
                          f"the next chunk after this one"})
        elif abs(step) > LEVEL_FLAG and not applied:
            flags.append({"code": "level", "text": f"started {abs(step):.1f} levels {'darker' if step < 0 else 'brighter'} "
                          f"than its {side} pin and wasn't levelled ("
                          + ("level_match is off" if not level_match else "too large a difference to be the pin's")
                          + "): the next chunk pinned to it carries the step"})

    # no start pin (a straight-cut chunk, or one pinned onto the original): the anchors still hold it to the job's first
    # full chunk, measured on frames sampled across the take (there is no pinned frame to measure)
    if "start" not in level and source_mask is not None and int(source_mask.shape[0]) >= 1 \
            and ((detail_match and detail_anchor) or person_anchor):
        try:
            anch = anchor_sample(plan, job, (W, H))
            if anch and anch[2].split("-")[0] != desc["chunk"]:         # the reference chunk is not matched to itself
                pick = sorted({int(round(v)) for v in np.linspace(0, keep - 1, ANCHOR_SAMPLES + 2)[1:-1]})
                outs = [raw_u8(i) for i in pick]
                pm = [_soft_mask(source_mask, i, (W, H)) > 0.5 for i in pick]
                share = float(np.mean([m.mean() for m in pm]))
                if LEVEL_MIN_REGION <= share <= 1 - LEVEL_MIN_REGION:
                    free = {"ref": anch[2]}
                    if detail_match and detail_anchor:
                        d = detail_gain(outs, outs, [~m for m in pm], anchor=anch[:2])
                        if d is not None:
                            free["detail"] = {"ratio": round(d[1], 3), "k": round(d[0], 3),
                                              "applied": d[0] >= DETAIL_MIN}
                            if free["detail"]["applied"]:
                                detail["k"] = detail["person_k"] = d[0]
                    if person_anchor:
                        res = person_lut(outs, pm, anch[0], anch[3])
                        if res is not None:
                            plut = res[0] if res[1] >= 1.0 else None
                            free["person_anchor"] = {"moved": round(res[1], 2), "applied": plut is not None}
                    level["anchor"] = free
        except Exception as e:
            print(f"[SeamStitch] Swap Take: anchor (no pin) skipped: {e}")

    regional = any(person_gains[k] is not None and gains[k] is not None and not np.array_equal(person_gains[k], gains[k])
                   for k in gains)

    def frames_u8():
        for i in range(keep):
            f = raw_u8(i)
            g = gain_at(i, keep, gains["start"], gains["end"])
            if g is not None:
                if regional:
                    gp = gain_at(i, keep, person_gains["start"], person_gains["end"])
                    m = _soft_mask(source_mask, i, (W, H))[..., None]
                    sc = g[0].astype(np.float32)[None, None, :] * (1 - m) + gp[0].astype(np.float32)[None, None, :] * m
                    of = g[1].astype(np.float32)[None, None, :] * (1 - m) + gp[1].astype(np.float32)[None, None, :] * m
                    f = np.clip(f.astype(np.float32) * sc + of + 0.5, 0, 255).astype(np.uint8)
                else:
                    f = np.clip(f.astype(np.float32) * g[0].astype(np.float32) + g[1].astype(np.float32) + 0.5, 0,
                                255).astype(np.uint8)
            if plut is not None and source_mask is not None:
                m = _soft_mask(source_mask, i, (W, H))[..., None]
                g2 = np.stack([plut[c][f[..., c]] for c in range(3)], axis=-1).astype(np.float32)
                f = np.clip(f.astype(np.float32) + m * (g2 - f.astype(np.float32)) + 0.5, 0, 255).astype(np.uint8)
            if detail["k"] >= DETAIL_MIN or detail["person_k"] >= DETAIL_MIN:
                room_k = detail["map"] if detail["map"] is not None else detail["k"]
                if source_mask is not None and (detail["person_k"] != detail["k"] or detail["map"] is not None):
                    m = _soft_mask(source_mask, i, (W, H))
                    kmap = room_k * (1 - m) + detail["person_k"] * m
                else:
                    kmap = room_k
                f = boost_detail(f, kmap)
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
        if output_mask is not None and not output_mask_is_picked(prompt):
            flags.append({"code": "output_mask_any_person", "text":
                          "the output mask doesn't come from Swap Output Person: with other people in the shot, "
                          "following can read someone else (the example workflow wires it)"})
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
        if desc.get("pass1") and not all(isinstance(v, int) for v in settings.get("pass1") or [None]):
            settings["pass1"] = desc["pass1"]       # H3's size linked from the Planner: the size it got
        settings.update(mark=spl._parse_value(opts.get("mark", True)), conform=bool(desc.get("conform")),
                        options=opts, anchors=desc.get("anchors"))
        take = {"id": tid, "created": sp.now(), "state": "ok", "seed": desc["seed"], "render": [r0, r1],
                "held": held, "length": L, "frames": keep, "splits": desc["splits"], "pins": desc["pins"],
                "file": f"{rel}/{take_name}", "proxy": f"{rel}/proxy.mp4", "take_codec": take_codec,
                "prompt_sha1": desc["prompt_sha1"], "settings": settings, "files": files, "scores": scores,
                "gpu_s": round(t0 - float(desc.get("started") or t0)), "save_s": None, "flags": flags, "level": level,
                "nonce": desc.get("nonce"), "planner_rev": desc.get("plan_rev")}
        if crop:
            take["crop"] = list(crop)
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
                "free_memory_after": ("BOOLEAN", {"default": True, "tooltip":
                    "When this run finishes, ComfyUI drops its models and its cache, as its own /free does, so the next "
                    "chunk starts on a clear machine (each render reloads its models, about half a minute). Off: the "
                    "models stay loaded, and on a 64 GB PC the second render in a row can page to disk (2026-10-15: "
                    "282 s a step against 76, ComfyUI holding 67.7 GB)."}),
                "level_match": ("BOOLEAN", {"default": True, "tooltip":
                    "Before the take is saved, match its brightness and contrast, per colour channel, to the frame it "
                    "was pinned to at its start (the same moment): a pinned frame comes out of H3 2-3 levels darker and "
                    "a little harsher, and with each chunk pinned to the last the picture slid darker and muddier chunk "
                    "by chunk. With the person's mask wired (source_mask) the room and the person are matched apart. "
                    "Only the start pin is followed (a chunk with no start pin follows its end pin instead): an end that lands more than 1 level off the next chunk's take gets "
                    "a level_end flag (that take may be older), and a start too far off to be the pin's (or level_match "
                    "off) gets a level flag."}),
                "detail_match": ("BOOLEAN", {"default": True, "tooltip":
                    "The pin round trip also softens a frame: each chunk kept only ~0.78 of its pin's fine detail, and "
                    "with each chunk pinned to the last the room's texture faded ~75% over a 2642-frame job (it looked "
                    "washed out) while the level match held. Before the take is saved, its fine detail (the band a "
                    "1-2 px blur removes, luma only) is boosted, up to 1.6x, until its pinned frames carry the pin's "
                    "detail; the room and the person (source_mask) are measured apart. It restores crispness, it "
                    "cannot redraw lost texture. Only a take that has a start pin is touched."}),
                "detail_anchor": ("BOOLEAN", {"default": True, "tooltip":
                    "Aim detail_match at the job's first full chunk (rendered with no pin) instead of at the pin. Matching "
                    "only the pin stops the loss at each join but not the shortfall left at every join, which still "
                    "added up (wall 76 -> 48, sofa 209 -> 105 over 14 chunks); matched to the first chunk it cannot "
                    "build up. The room is measured by the source mask, so a job without one falls back to the pin."}),
                "person_anchor": ("BOOLEAN", {"default": True, "tooltip":
                    "Her colour fades down a chain of pinned chunks (the red coat went from ~56 to ~40 and her saturation "
                    "fell 141 -> 108 over 14 chunks) in a way matching each chunk to its pin cannot hold. The brightness "
                    "quantiles of her pixels (per channel, by the source mask) are moved toward the job's first full "
                    "chunk's, by at most 24 levels. Needs source_mask and a reference chunk (rendered with no pin)."}),
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
             source_mask=None, output_mask=None, marked_guide=None, free_memory_after=True, level_match=True,
             detail_match=True, detail_anchor=True, person_anchor=True,
             prompt=None, extra_pnginfo=None):
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
                                       mouth_sync, review_clip, prompt, extra_pnginfo, progress, level_match, detail_match, detail_anchor, person_anchor)
        lines = [f"{take['id']}: {take['frames']} frames {take['render'][0]}-{take['render'][1]}, seed {take['seed']}"
                 + (f", pose IoU {take['scores']['pose_iou']}" if take["scores"].get("pose_iou") is not None else "")
                 + (f", mouth {take['scores']['mouth']}" if take["scores"].get("mouth") is not None else "")]
        lv = take.get("level") or {}
        for side in ("start", "end"):
            if lv.get(side):
                x = lv[side]
                lines.append(f"  level ({side} pin): started {x['step']:+.1f} levels off its pin"
                             + (f" (the person {x['person_step']:+.1f})" if "person_step" in x else "") + "; "
                             + ("levelled" if x["applied"] else "not levelled"))
                pa = x.get("person_anchor")
                if pa:
                    lines.append(f"  person colour ({side} pin): " + (f"moved toward {pa['ref']} by {pa['moved']:.1f} levels on "
                                 f"average" if pa["applied"] else "already within 1 level of it"))
                dt = x.get("detail")
                if dt:
                    lines.append(f"  detail ({side} pin): kept {dt['ratio']:.2f} of its pin's fine detail; "
                                 + (f"aimed at {dt['anchor']}, " if dt.get("anchor") else "")
                                 + (f"boosted x{1 + dt['k']:.2f} (the person x{1 + dt['person_k']:.2f})"
                                    if dt["applied"] else "not boosted"))
        fa = lv.get("anchor")
        if fa:
            if fa.get("detail"):
                lines.append(f"  detail (no pin): {fa['detail']['ratio']:.2f}x the room detail of {fa['ref']}; "
                             + (f"boosted x{1 + fa['detail']['k']:.2f}" if fa["detail"]["applied"] else "not boosted"))
            if fa.get("person_anchor"):
                lines.append("  person colour (no pin): " + (f"moved toward {fa['ref']} by {fa['person_anchor']['moved']:.1f} "
                             "levels on average" if fa["person_anchor"]["applied"] else "already within 1 level of it"))
        quality = ss.chunk_flags(take["scores"])
        for j in report["joins"]:
            fl = j.get("frame_luma") or {}
            cl = j.get("char_luma") or {}
            lines.append(f"  {j['split']} @ {j['frame']}: {j['type']}, splice {j['splice']}, {j.get('override') or j['repair']}; "
                         f"jump {fl.get('at_splice')} / char {cl.get('at_splice')}, ratio {j.get('join_ratio')}")
        if free_memory_after and request_free():
            lines.append("  memory: ComfyUI drops its models and cache when this run finishes")
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


def request_free():
    """Ask ComfyUI to unload its models and drop its cache once the running prompt is done: the flags its /free route
    sets, which the prompt worker applies between queue items (dropping them mid-run would only make later nodes
    reload). Measured idle after a render: 54.6 -> 11.1 GB private, the card 31.4 -> 1.7 GB. -> True when asked."""
    try:
        from server import PromptServer
        q = PromptServer.instance.prompt_queue
        q.set_flag("unload_models", True)
        q.set_flag("free_memory", True)
        return True
    except Exception as e:
        print(f"[SeamStitch] Swap Take: couldn't ask ComfyUI to free its memory: {e}")
        return False


def output_person(desc, track_data, source_mask, images=None, sheet=None):
    """Everything Swap Output Person does: (MASK [n, H, W], report). The replacement's mask in the render, picked
    per shot from everyone SAM3 Track Output followed (swap_scores.pick_person): by the character sheet's colours
    with the render's frames and the sheet, else by the overlap with her source mask; cuts are the chunk's
    confirmed cuts inside its render."""
    import torch.nn.functional as F
    if not isinstance(desc, dict) or desc.get("format") != spl.CHUNK_FORMAT:
        raise TakeError("chunk is not a Swap Planner chunk: wire the Planner's chunk output")
    plan = sp.load_plan(desc["plan"])
    r0, r1 = (int(x) for x in desc["render"])
    cuts = [c - r0 for c in sp.confirmed_cuts(plan.get("cuts")) if r0 < c <= r1]
    H, W = (int(x) for x in track_data["orig_size"])
    n = int(track_data["n_frames"])
    packed = track_data.get("packed_masks")
    if packed is None:
        return torch.zeros((n, H, W), dtype=torch.float32), {"objects": 0, "picks": [], "cuts": cuts}
    objs = ss.unpack_bits(packed.detach().cpu().numpy() if isinstance(packed, torch.Tensor) else packed)
    sm = source_mask.detach().cpu().numpy() if isinstance(source_mask, torch.Tensor) else np.asarray(source_mask)
    m = len(sm)
    # held frames repeat the last delivered one: they take its mask
    src = [sm[min(i, m - 1)] for i in range(n)]
    frames = sh = None
    if images is not None and sheet is not None:
        frames = _LazyFrames(images)
        sh = to_u8(sheet[0])
    hers, picks = ss.pick_person(objs, src, cuts, frames, sh)
    out = torch.from_numpy(hers.astype(np.float32)).unsqueeze(1)
    out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    return out, {"objects": int(objs.shape[1]), "picks": picks, "cuts": cuts,
                 "by": "sheet" if sh is not None else "overlap"}


class _LazyFrames:
    """An IMAGE batch read as RGB uint8 frames on demand (the pick samples a few per shot)."""

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return int(self.images.shape[0])

    def __getitem__(self, i):
        return to_u8(self.images[min(int(i), len(self) - 1)])


class SeamStitchSwapOutputPerson:
    """B8: the replaced person in the render, for the take's scores, when other people share the shot."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk": (spl.CHUNK_TYPE, {"tooltip": "The Swap Planner's chunk output (for the chunk's cuts)."}),
                "track_data": ("SAM3_TRACK_DATA", {"tooltip":
                    "SAM3 Track Output over the render, tracking every person (max_objects 0)."}),
                "source_mask": ("MASK", {"tooltip":
                    "The replaced person's source mask (the same one Swap Take gets)."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip":
                    "The render (the same frames Swap Take gets): with the sheet, people are told apart by colour."}),
                "sheet": ("IMAGE", {"tooltip":
                    "The character sheet (Picture 1): per shot, the tracked person who looks most like it is the "
                    "replacement, wherever H3 put her. Without it: the person overlapping her source mask most."}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("output_mask", "report")
    FUNCTION = "pick"
    CATEGORY = "SeamStitch/Swap"
    DESCRIPTION = ("Finds the replacement among everyone SAM3 tracked in the render, per shot (split at the chunk's "
                   "confirmed cuts): the tracked person who looks most like the character sheet, else the one "
                   "overlapping her source mask most. Frames where she isn't in the source get no mask, so "
                   "following skips them. Feed its output_mask to Swap Take.")

    def pick(self, chunk, track_data, source_mask, images=None, sheet=None):
        mask, rep = output_person(chunk, track_data, source_mask, images, sheet)
        shots = ", ".join(f"{p['shot'][0]}-{p['shot'][1] - 1}: " + (
            f"person {p['object']} (by {p['by']}: sheet {p['sheet']}, IoU {p['iou']})"
            if p["object"] is not None else "nobody") for p in rep["picks"])
        print(f"[SeamStitch] Swap Output Person: {chunk.get('chunk')}: {rep['objects']} tracked; {shots or 'nothing tracked'}")
        return (mask, json.dumps(rep))
