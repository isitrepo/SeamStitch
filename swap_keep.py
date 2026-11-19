"""SeamStitch Swap object keep: the source's own pixels of an object pasted over the assembled takes.

A swap re-draws the whole frame, and H3 is free to change what it wasn't asked to (test clip, 2026-10-20:
white shoes drawn black in c10, as a pink sock in c11). An object keep (swap_plan.object_keeps) names the thing
("white shoe") and a source range; a mark run with the keep caches its SAM3 mask, and the assembly pastes the source
through that mask, grown (to cover where the take drew it a little off) and feathered:

    out = take * (1 - a) + source * gain * a          a = feather(grow(mask))

gain ("level": "ring", optional; the default is "none", the source as filmed): per channel, the take's median over a ring just outside the grown mask against the
source's over the same ring, clamped to GAIN_CLAMP, so the pasted object sits in the take's grade (the takes are
level-matched to their pins, not to the source). It misreads where the take drew something else right next to
the object (the ring then measures that), so it is off by default.

Streaming: frames are asked for in order (the assembly's loop), so the masks and the source are read sequentially.
"""

import numpy as np

try:
    from . import timeline as tl
    from . import swap_plan as sp
except ImportError:  # imported as a top-level module (tests, tools)
    import timeline as tl
    import swap_plan as sp

RING = 10                     # px: the ring the level is measured on, outside the grown mask
RING_MIN = 200                # px in the ring, else no level (gain 1)
GAIN_CLAMP = (0.85, 1.18)
SPARE_GROW = 9                # px: how far a zone keeps off its spared target


def _cv2():
    import cv2
    return cv2


def _ellipse(r):
    cv2 = _cv2()
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def alpha(mask, w, h, grow, feather):
    """A cached mask frame (uint8 or bool, any size) -> (alpha float32 [h, w] in 0..1, grown bool mask)."""
    cv2 = _cv2()
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 127) if m.dtype != bool else m
    m = m.astype(np.uint8) * 255
    if m.shape[1] != w or m.shape[0] != h:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    g = m > 127
    if grow > 0 and g.any():
        g = cv2.dilate(g.astype(np.uint8), _ellipse(int(grow))) > 0
    a = g.astype(np.float32)
    if feather > 0 and g.any():
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=float(feather))
    return a, g


def ring_gain(take, source, grown):
    """Per-channel gain that brings the source to the take's grade, from their medians on a ring outside `grown`."""
    if not grown.any():
        return np.ones(3, np.float32)
    cv2 = _cv2()
    ring = (cv2.dilate(grown.astype(np.uint8), _ellipse(RING)) > 0) & ~grown
    if int(ring.sum()) < RING_MIN:
        return np.ones(3, np.float32)
    t = np.median(take[ring].astype(np.float32), axis=0)
    s = np.median(source[ring].astype(np.float32), axis=0)
    g = np.where(s > 4, t / np.maximum(s, 1e-3), 1.0)
    return np.clip(g, *GAIN_CLAMP).astype(np.float32)


def paste(take, source, a, gain=None):
    """take, source uint8 [h, w, 3]; a float [h, w] -> uint8."""
    s = source.astype(np.float32)
    if gain is not None:
        s = s * gain[None, None, :]
    t = take.astype(np.float32)
    out = t + (np.clip(s, 0, 255) - t) * a[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class _SeqFrames:
    """Frames of one file by source frame number, read in order (restarts on a jump back)."""

    def __init__(self, path, fr, offset):
        self.path, self.fr, self.offset = path, fr, int(offset)
        self.it, self.next_f = None, None

    def read(self, f):
        if self.it is None or f != self.next_f:
            if self.it is not None and f > self.next_f:
                for _ in range(f - self.next_f):            # a short skip ahead: read through
                    next(self.it)
            else:
                self.it = tl._iter_frames(self.path, self.fr, f - self.offset, None)
            self.next_f = f
        frame = next(self.it)
        self.next_f = f + 1
        return frame


# A take's own cut can land a few frames off the source's (H3 cuts early or late inside a chunk; test clip c10-t006
# cut to the close-up at 1667, the source at 1666, so frame 1666 showed the wide shot's black shoe with no mask, the
# source being on the close-up already). Within CUT_LAG frames of a confirmed cut, which side of it the take is on is
# read off small grey thumbnails: the take frame against the source's last frame before the cut and its first after.
# Still on the old shot after the cut (late): the last pasted frame before the cut is held. On the new shot before it
# (early): nothing is pasted (the source's object belongs to the shot the take has left).
CUT_LAG = 8
SIDE_MARGIN = 0.75            # the take must be this much closer to one side's thumbnail to count as on that side


def thumb(frame):
    cv2 = _cv2()
    g = cv2.resize(np.asarray(frame, np.float32).mean(-1), (48, 27), interpolation=cv2.INTER_AREA)
    return (g - g.mean()) / max(float(g.std()), 1e-3)


def take_side(take_thumb, before, after):
    """'before' / 'after': the side of a cut a take frame shows, or None when it isn't clearly either."""
    db, da = float(np.abs(take_thumb - before).mean()), float(np.abs(take_thumb - after).mean())
    if db < SIDE_MARGIN * da:
        return "before"
    if da < SIDE_MARGIN * db:
        return "after"
    return None


class KeepPaster:
    """The job's object keeps over source frames lo..hi, applied frame by frame in order."""

    def __init__(self, plan, job_dir, source_path, fr, size, lo, hi, source_reader=None, cuts=None):
        import os
        self.w, self.h = size
        self.items = []
        for k in sp.object_keeps(plan):
            for seg, f0, f1 in sp.keep_segments(plan, k, lo, hi):
                path = os.path.join(job_dir, seg["file"])
                if not os.path.isfile(path):
                    continue
                self.items.append({"keep": k, "seg": seg, "f0": f0, "f1": f1,
                                   "reader": _SeqFrames(path, fr, int(seg["range"][0]))})
        self.source = source_reader or _SeqFrames(source_path, fr, 0)
        self.stats = {}
        self.lag, self.early = [], []
        self.held, self.held_at = [], None
        # the source's thumbnails either side of each confirmed cut a keep reaches
        cuts = sp.confirmed_cuts(plan.get("cuts")) if cuts is None else cuts
        n = int(plan["source"]["frames"])
        self.cut_thumbs = {}
        if self.items:
            for c in sorted(set(int(c) for c in cuts)):
                if not 0 < c < n or not any(it["f0"] - CUT_LAG <= c <= it["f1"] + CUT_LAG for it in self.items):
                    continue
                if not lo - CUT_LAG < c <= hi + CUT_LAG:
                    continue
                pair = [x.copy() for x in tl._iter_frames(source_path, fr, c - 1, c + 1)]   # the decoder reuses its buffer
                if len(pair) == 2:
                    self.cut_thumbs[c] = (thumb(pair[0]), thumb(pair[1]))

    def __bool__(self):
        return bool(self.items)

    def _near_cut(self, f):
        for c in self.cut_thumbs:
            if c - CUT_LAG <= f < c + CUT_LAG:
                return c
        return None

    def covers(self, f):
        return any(it["f0"] <= f <= it["f1"] for it in self.items) or (self.held_at is not None and
                                                                        self._near_cut(f) is not None)

    def apply(self, f, frame):
        c = self._near_cut(f)
        if c is not None:
            side = take_side(thumb(frame), *self.cut_thumbs[c])
            if f >= c and side == "before":                      # the take hasn't cut yet: hold the shot's last paste
                if self.held and self.held_at is not None and self.held_at >= c - 1 - CUT_LAG:
                    for a, src, gain in self.held:
                        frame = paste(frame, src, a, gain)
                    self.lag.append(f)
                return frame
            if f < c and side == "after":                        # the take cut early: the source's object isn't in it
                self.early.append(f)
                return frame
        hit = [it for it in self.items if it["f0"] <= f <= it["f1"]]
        if not hit:
            return frame
        src = self.source.read(f)
        if src.shape[:2] != frame.shape[:2]:
            src = _cv2().resize(src, (frame.shape[1], frame.shape[0]), interpolation=_cv2().INTER_CUBIC)
        pasted = []
        for it in hit:
            k = it["keep"]
            a, grown = alpha(it["reader"].read(f), self.w, self.h, int(k["grow"]), float(k["feather"]))
            st = self.stats.setdefault(k["id"], {"target": k["target"], "frames": 0, "empty": 0, "gain_sum": np.zeros(3)})
            if not grown.any():
                st["empty"] += 1
                continue
            gain = ring_gain(frame, src, grown) if k["level"] == sp.KEEP_LEVEL_RING else None
            frame = paste(frame, src, a, gain)
            pasted.append((a, src, gain))
            st["frames"] += 1
            st["gain_sum"] += gain if gain is not None else 1.0
        if pasted:
            self.held, self.held_at = pasted, f
        return frame

    def report(self):
        out = {}
        for kid, st in self.stats.items():
            n = max(1, st["frames"])
            out[kid] = {"target": st["target"], "frames": st["frames"], "empty_mask_frames": st["empty"],
                        "mean_gain": [round(float(x), 4) for x in st["gain_sum"] / n]}
        if self.lag or self.early:
            out["cut_timing"] = {"held_over_a_late_take_cut": self.lag, "skipped_before_an_early_take_cut": self.early}
        return out


class ObjectCompositor:
    """An object job's assembly (swap_plan.MODE_OBJECT): the source everywhere, the take inside the target's cached
    mask (grown and feathered; a chunk's own target where it has one). "ring" level: the take is scaled to the
    source's grade on a ring just outside the grown mask, where both show the same untouched picture.
    Frames with no mask cached, or an empty one, are the source."""

    def __init__(self, plan, job_dir, source_path, fr, size, lo, hi):
        import os
        self.w, self.h = size
        self.o = sp.object_settings(plan)
        self.items = []
        targets = {sp.chunk_target(plan, c) for c in plan.get("chunks") or [] if not sp.is_kept(c)}
        for t in targets:
            k = {"target": t, "range": [lo, hi]}
            for seg, f0, f1 in sp.keep_segments(plan, k, lo, hi):
                path = os.path.join(job_dir, seg["file"])
                if os.path.isfile(path):
                    self.items.append({"target": t, "f0": f0, "f1": f1, "reader": _SeqFrames(path, fr, int(seg["range"][0]))})
        # a target text that matches no cached mask (a blank one, or "black shoe" for a mask marked "black shoes") gives the
        # source back for its chunks without a word: name it (Q4 fault 2, 2026-11-17). Matching stays exact.
        have = {it["target"] for it in self.items}
        marked = sorted({m.get("target", "") for m in plan.get("masks") or []})
        plain = self.o.get("paste") == "frame" and self.o.get("fill") != "picture"      # the take as it is: no mask is used
        self.unmatched = [] if plain else [{"target": t, "marked": marked} for t in sorted(targets) if t not in have]
        self.chunk_of = {}
        for c in plan.get("chunks") or []:
            for f in range(max(lo, c["deliver"][0]), min(hi, c["deliver"][1]) + 1):
                self.chunk_of[f] = c
        self.plan = plan
        self.source = _SeqFrames(source_path, fr, 0)
        self.stats = {"frames": 0, "source_only": 0, "gain_sum": np.zeros(3)}
        self.guide = None
        self.zone = None
        if sp.object_composite(plan):                   # an add with a cut-out: paste the cut-out's own area back
            t = next(iter(targets)) if len(targets) == 1 else (plan.get("target") or "")
            self.guide = CompositeGuide(plan, job_dir, fr, size, lo, hi, t)
        elif sp.object_zone(plan):                      # an add by redrawing a zone: paste the zone back
            t = next(iter(targets)) if len(targets) == 1 else (plan.get("target") or "")
            self.zone = ZoneGuide(plan, job_dir, fr, size, lo, hi, t)
        self.picture = None
        if self.o.get("fill") == "picture":
            self.picture = load_picture(self.o.get("picture"), job_dir)
            self.boxes = shot_boxes(self.items, plan, fr, (self.w, self.h), lo, hi)
            self.tone = None

    def apply(self, f, take):
        src = self.source.read(f)
        if src.shape[:2] != take.shape[:2]:
            src = _cv2().resize(src, (take.shape[1], take.shape[0]), interpolation=_cv2().INTER_CUBIC)
        c = self.chunk_of.get(f)
        if self.o.get("paste") == "frame" and c is not None and not sp.is_kept(c) and self.picture is None:
            self.stats["frames"] += 1                   # the plain way: the take as it is
            return take
        t = sp.chunk_target(self.plan, c) if c is not None else None
        it = next((x for x in self.items if x["target"] == t and x["f0"] <= f <= x["f1"]), None)
        if c is None or sp.is_kept(c) or it is None:
            self.stats["source_only"] += 1
            return src
        if self.picture is not None:
            return self._picture(f, src, it)
        if self.guide is not None:
            _fr, ca = self.guide.apply(f, src)
            if ca is None:
                self.stats["source_only"] += 1
                return src
            m = (ca > 0.5).astype(np.uint8) * 255
            a, grown = alpha(m, self.w, self.h, int(self.o["paste_grow"]), float(self.o["feather"]))
        elif self.zone is not None:
            zm = self.zone.mask(f)
            if zm is None:
                self.stats["source_only"] += 1
                return src
            a, grown = alpha((zm > 0.5).astype(np.uint8) * 255, self.w, self.h, 0, float(self.o["feather"]))
        else:
            a, grown = alpha(it["reader"].read(f), self.w, self.h, int(self.o["grow"]), float(self.o["feather"]))
        if not grown.any():
            self.stats["source_only"] += 1
            return src
        gain = ring_gain(src, take, grown) if self.o["level"] == sp.KEEP_LEVEL_RING else None
        self.stats["frames"] += 1
        self.stats["gain_sum"] += gain if gain is not None else 1.0
        return paste(src, take, a, gain)

    def _picture(self, f, src, it):
        """fill "picture": the picture fitted over the shot's box, toned to the target's own light, through its mask."""
        a, grown = alpha(it["reader"].read(f), self.w, self.h, sp.PICTURE_GROW, sp.PICTURE_FEATHER)
        box = self.boxes.get(f)
        if box is None or not grown.any():
            self.stats["source_only"] += 1
            return src
        x0, y0, x1, y1 = box
        canvas = np.zeros_like(src)
        canvas[y0:y1, x0:x1] = fit_cover(self.picture, x1 - x0, y1 - y0)
        # the light: the picture's luma brought to the target's own luma (mean and spread) under the mask, smoothed
        # over frames so the old painting's own pattern doesn't flicker into it
        tl_ = luma(src)[grown]
        m, s = float(tl_.mean()), float(tl_.std())
        self.tone = (m, s) if self.tone is None else (0.8 * self.tone[0] + 0.2 * m, 0.8 * self.tone[1] + 0.2 * s)
        pic = canvas[y0:y1, x0:x1].astype(np.float32)
        pl = luma(pic)
        pm, ps = float(pl.mean()), max(float(pl.std()), 1.0)
        new_l = (pl - pm) / ps * max(self.tone[1], 1.0) + self.tone[0]
        pic = np.clip(pic + (new_l - pl)[..., None], 0, 255)
        if sp.PICTURE_SOFTEN > 0:
            pic = _cv2().GaussianBlur(pic, (0, 0), sigmaX=sp.PICTURE_SOFTEN)
        canvas[y0:y1, x0:x1] = (pic + 0.5).astype(np.uint8)
        inside = np.zeros_like(a)
        inside[y0:y1, x0:x1] = a[y0:y1, x0:x1]                 # the picture only inside its shot's box
        a = inside
        self.stats["frames"] += 1
        return paste(src, canvas, a)

    def warnings(self):
        """One sentence per target text that matched no cached mask (those chunks come out as the source)."""
        out = []
        for u in self.unmatched:
            t = u["target"]
            have = ", ".join(repr(m) for m in u["marked"]) or "none"
            out.append(f"object target {t!r} matches no cached mask (masks are marked for: {have}): its chunks come out as "
                       f"the source. The target text must be the same as the one the mask was marked with")
        return out

    def report(self):
        n = max(1, self.stats["frames"])
        out = {"settings": self.o, "frames_with_object": self.stats["frames"], "source_only_frames": self.stats["source_only"],
               "mean_take_gain": [round(float(x), 4) for x in self.stats["gain_sum"] / n]}
        if self.picture is not None:
            out["shot_boxes"] = sorted({tuple(b) for b in self.boxes.values()})
        return out


def luma(img):
    img = np.asarray(img, np.float32)
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def fit_cover(picture, w, h):
    """The picture scaled to cover w x h (its shape kept) and centre-cropped to it."""
    cv2 = _cv2()
    ph, pw = picture.shape[:2]
    s = max(w / float(pw), h / float(ph))
    nw, nh = max(w, int(round(pw * s))), max(h, int(round(ph * s)))
    r = cv2.resize(picture, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x, y = (nw - w) // 2, (nh - h) // 2
    return r[y:y + h, x:x + w]


def load_picture(name, job_dir):
    """The picture for fill "picture": a path, or a file name in ComfyUI's input folder (or the job's)."""
    import os
    cv2 = _cv2()
    cands = [name]
    try:
        import folder_paths
        cands.append(os.path.join(folder_paths.get_input_directory(), name))
    except Exception:
        pass
    cands.append(os.path.join(job_dir, name))
    for p in cands:
        if name and os.path.isfile(p):
            img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise ValueError(f"fill picture: no picture {name!r} (set the object's picture: a file in ComfyUI's input folder)")


def shot_boxes(items, plan, fr, size, lo, hi):
    """Per frame, the box of its shot: the union of the target's masks over the shot's frames (a still camera: the
    painting's whole visible extent, people in front of it or not). {frame: [x0, y0, x1, y1]} in source pixels."""
    w, h = size
    cuts = sorted(set(sp.confirmed_cuts(plan.get("cuts"))) | {lo, hi + 1})
    shots = [(a, b - 1) for a, b in zip(cuts, cuts[1:]) if lo <= a <= hi]
    masks = {}
    for it in items:
        rd = _SeqFrames(it["reader"].path, it["reader"].fr, it["reader"].offset)
        for f in range(it["f0"], it["f1"] + 1):
            m = rd.read(f)
            m = (m[..., 0] if m.ndim == 3 else m) > 127
            if m.any():
                ys, xs = np.nonzero(m)
                mh, mw = m.shape
                masks[f] = (xs.min() * w / mw, ys.min() * h / mh, (xs.max() + 1) * w / mw, (ys.max() + 1) * h / mh)
    out = {}
    for a, b in shots:
        bs = [masks[f] for f in range(a, b + 1) if f in masks]
        if not bs:
            continue
        box = [int(min(x[0] for x in bs)), int(min(x[1] for x in bs)), int(np.ceil(max(x[2] for x in bs))),
               int(np.ceil(max(x[3] for x in bs)))]
        box = [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])]
        for f in range(a, b + 1):
            out[f] = box
    return out


# ---------------------------------------------------------------------------
# object add, composite guide (2026-10-21): H3 Ref2VA, asked to add a blue jay on the test subject's shoulder, put a giant jay
# in his place (marked or not, Eros or the plain model). The cut-out of the sheet's object (BiRefNet, in the graph) is
# placed on the target in every frame instead: its feet on the top of the target's mask, its height a multiple of
# the mask's width (the first frame's), following the mask's top-centre (smoothed), toned to the scene. That
# composite is what H3 refines (img2img at a partial denoise, the Face Refine wiring), and its own alpha is what
# Swap Assemble pastes back.
# ---------------------------------------------------------------------------

ANCHOR_SMOOTH = 7             # frames: a moving average over the mask's top-centre
ANCHOR_TOP_BAND = 0.2         # the top fifth of the mask's rows gives its top-centre x
JUMP = 0.5                    # a frame-to-frame move beyond this many target widths is a stray patch, held


def load_cutout(rgb_path, mask_path):
    """The cut-out as (rgb uint8 [h, w, 3], alpha float [h, w]), cropped to its alpha's box."""
    cv2 = _cv2()
    rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(rgb_path, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    a = cv2.imdecode(np.fromfile(mask_path, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    if a.shape != rgb.shape[:2]:
        a = cv2.resize(a, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(a > 0.05)
    if not len(ys):
        raise ValueError("the cut-out's mask is empty")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return rgb[y0:y1, x0:x1], a[y0:y1, x0:x1]


def anchors(mask_frames, size):
    """{frame: (x, y, width)} in source pixels from {frame: mask}: the top-centre and width of the target's piece of
    each mask, followed frame to frame (the piece nearest the last anchor; the largest on the first frame), a jump of
    more than JUMP x its width held at the last place, x and y smoothed over ANCHOR_SMOOTH frames. SAM3's shoulder came
    with stray patches (the other shoulder, his chin when he raised an arm: test clip, 2026-10-21)."""
    cv2 = _cv2()
    w, h = size
    raw, last = {}, None
    for f in sorted(mask_frames):
        m = mask_frames[f]
        m = (m[..., 0] if m.ndim == 3 else m) > 127
        if not m.any():
            continue
        mh, mw = m.shape
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        pieces = []
        for k in range(1, n):
            ys, xs = np.nonzero(lab == k)
            top, bot = ys.min(), ys.max()
            band = ys <= top + max(1, int((bot - top + 1) * ANCHOR_TOP_BAND))
            pieces.append((float(np.median(xs[band])) * w / mw, float(top) * h / mh,
                           float(xs.max() - xs.min() + 1) * w / mw, int(stats[k, cv2.CC_STAT_AREA])))
        if last is None:
            pick = max(pieces, key=lambda q: q[3])
        else:
            pick = min(pieces, key=lambda q: (q[0] - last[0]) ** 2 + (q[1] - last[1]) ** 2)
            if ((pick[0] - last[0]) ** 2 + (pick[1] - last[1]) ** 2) ** 0.5 > JUMP * max(last[2], 1.0):
                pick = last                                     # a jump to another patch: hold
        raw[f] = pick[:3]
        last = pick
    out, fs = {}, sorted(raw)
    for f in fs:
        near = [raw[g] for g in fs if abs(g - f) <= ANCHOR_SMOOTH // 2]
        out[f] = (float(np.mean([q[0] for q in near])), float(np.mean([q[1] for q in near])), raw[f][2])
    return out


def placement(cut_a_shape, anchor, first_width, settings):
    """(x0, y0, scale) of the cut-out for one frame: its height `size` x the first frame's target width, its feet
    `sink` of its height below the anchor (resting into the shoulder, not floating on its edge)."""
    ch, cw = cut_a_shape
    hgt = float(settings.get("size", 1.3)) * first_width
    s = hgt / float(ch)
    x0 = anchor[0] - cw * s / 2.0
    y0 = anchor[1] - ch * s + float(settings.get("sink", 0.12)) * ch * s
    return x0, y0, s


def place_cutout(frame, cut_rgb, cut_a, place, tone=None, look=None, seed=0):
    """The cut-out pasted onto frame at place = (x0, y0, scale). Returns (frame, alpha [h, w]): the alpha covers the
    cut-out and, with a shadow, the shadow (what changed). tone: the cut-out's luma brought to this mean (the scene's
    light); None = as it is. look (all optional, 2026-10-21 "it looks pasted on"):
      shadow  0..1   a soft contact shadow under its feet and a faint drop shadow of its shape, this dark at most
      colour  0..1   its colour pulled this far towards the scene's own cast (the room's light, not the photo's sun)
      soften  px     a Gaussian blur, to the film's softness
      grain   bool   the source's own noise level added (measured around the anchor)"""
    cv2 = _cv2()
    look = look or {}
    H, W = frame.shape[:2]
    x0, y0, s = place
    nw, nh = max(1, int(round(cut_rgb.shape[1] * s))), max(1, int(round(cut_rgb.shape[0] * s)))
    rgb = cv2.resize(cut_rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC).astype(np.float32)
    a = cv2.resize(cut_a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ix, iy = int(round(x0)), int(round(y0))
    # the scene around it: a box twice its size, for the colour cast and the grain
    bx0, by0 = max(0, ix - nw // 2), max(0, iy - nh // 4)
    bx1, by1 = min(W, ix + nw + nw // 2), min(H, iy + nh + nh // 4)
    scene = frame[by0:by1, bx0:bx1].astype(np.float32)
    if float(look.get("colour", 0)) > 0 and scene.size:
        cast = scene.reshape(-1, 3).mean(0)
        cast = cast / max(float(cast.mean()), 1.0)                     # the room's tint, luma-neutral
        rgb = rgb * (1 - float(look["colour"]) + float(look["colour"]) * cast[None, None, :])
    if tone is not None:
        lm = float((luma(rgb) * a).sum() / max(a.sum(), 1e-3))
        rgb = np.clip(rgb * (tone / max(lm, 1.0)), 0, 255)
    if float(look.get("soften", 0)) > 0:
        rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=float(look["soften"]))
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=max(0.5, float(look["soften"]) * 0.7))
    if look.get("grain") and scene.size:
        g = luma(scene)
        sigma = float(np.std(g - cv2.GaussianBlur(g, (0, 0), sigmaX=1.5)))
        rng = np.random.default_rng(seed)
        rgb = np.clip(rgb + rng.normal(0, sigma, rgb.shape[:2])[..., None], 0, 255)
    full_a = np.zeros((H, W), np.float32)
    fx0, fy0, fx1, fy1 = max(0, ix), max(0, iy), min(W, ix + nw), min(H, iy + nh)
    if fx1 <= fx0 or fy1 <= fy0:
        return frame, full_a
    sub = (slice(fy0 - iy, fy1 - iy), slice(fx0 - ix, fx1 - ix))
    full_a[fy0:fy1, fx0:fx1] = a[sub]
    out = frame.astype(np.float32)
    changed = full_a.copy()
    if float(look.get("shadow", 0)) > 0:
        k = float(look["shadow"])
        sh = np.zeros((H, W), np.float32)
        # contact: a flat ellipse where its feet meet the target (the bottom of its alpha)
        ys, xs = np.nonzero(full_a > 0.3)
        if len(ys):
            fy, fx = int(ys.max()), int(np.median(xs[ys >= ys.max() - max(2, nh // 12)]))
            cv2.ellipse(sh, (fx, fy), (max(4, int(nw * 0.35)), max(2, int(nh * 0.05))), 0, 0, 360, 1.0, -1)
        # drop: its own shape, a little down and to the side, faint
        dx, dy = int(round(nw * 0.06)), int(round(nh * 0.04))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        sh = np.maximum(sh, 0.45 * cv2.warpAffine(full_a, M, (W, H)))
        sh = cv2.GaussianBlur(sh, (0, 0), sigmaX=max(2.0, nw * 0.05))
        out = out * (1 - k * sh[..., None])
        changed = np.maximum(changed, np.clip(sh * 2.0, 0, 1))
    out[fy0:fy1, fx0:fx1] = out[fy0:fy1, fx0:fx1] * (1 - a[sub][..., None]) + rgb[sub] * a[sub][..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), changed


def scene_tone(frame, anchor, width, settings):
    """The luma the cut-out takes: `light` x the scene's mean luma in a box around the anchor (a bird in a dim room
    is dim, its white feathers still the brightest thing on it)."""
    H, W = frame.shape[:2]
    r = int(max(16, width))
    x, y = int(anchor[0]), int(anchor[1])
    box = frame[max(0, y - r):min(H, y + r), max(0, x - r):min(W, x + r)]
    return float(settings.get("light", 1.8)) * float(luma(box).mean()) if box.size else None


class CompositeGuide:
    """An object add job's composite: the cut-out on the target, frame by frame, for source frames lo..hi."""

    def __init__(self, plan, job_dir, fr, size, lo, hi, target):
        import os
        self.o = sp.object_settings(plan)
        c = self.o.get("cutout") or {}
        self.cut_rgb, self.cut_a = load_cutout(os.path.join(job_dir, c["rgb"]), os.path.join(job_dir, c["mask"]))
        masks = {}
        for seg, f0, f1 in sp.keep_segments(plan, {"target": target, "range": [lo, hi]}, lo, hi):
            rd = _SeqFrames(os.path.join(job_dir, seg["file"]), fr, int(seg["range"][0]))
            for f in range(f0, f1 + 1):
                masks[f] = rd.read(f).copy()
        self.size = size
        self.anchors = anchors(masks, size)
        # the size from the median width over the range (the first frame's mask can be a poor one)
        self.first_width = float(np.median([a[2] for a in self.anchors.values()])) if self.anchors else None
        self.tone = None

    def place(self, f):
        a = self.anchors.get(f)
        return None if a is None or self.first_width is None else placement(self.cut_a.shape, a, self.first_width, self.o)

    def apply(self, f, frame):
        """(composited frame, alpha) for source frame f; (frame, None) where the target isn't found."""
        p = self.place(f)
        if p is None:
            return frame, None
        t = scene_tone(frame, self.anchors[f], self.first_width, self.o)
        if t is not None:
            self.tone = t if self.tone is None else 0.85 * self.tone + 0.15 * t
        look = {k: self.o.get(k) for k in ("shadow", "colour", "soften", "grain") if self.o.get(k)}
        return place_cutout(frame, self.cut_rgb, self.cut_a, p, self.tone, look, seed=int(f))

    def box(self, f, pad=0):
        """The cut-out's box in source pixels on frame f, or None."""
        p = self.place(f)
        if p is None:
            return None
        x0, y0, s = p
        return [x0 - pad, y0 - pad, x0 + self.cut_a.shape[1] * s + pad, y0 + self.cut_a.shape[0] * s + pad]


def zone_geometry(anchor, first_width, o):
    """(cx, cy, rx, ry) of an object add's zone for one frame, in source pixels, from the target's anchor (x, y, width).
    The zone is `size` target widths tall (the first frame's width, so it doesn't breathe with the mask) and
    `zone_aspect` x that wide, its bottom edge `sink` of its height below the anchor like a cut-out's feet (a bird
    resting on a shoulder); zone_dx / zone_dy move it, in target widths, for a thing that stands beside the target
    instead of on it. zone_grow pads it."""
    hgt = float(o.get("size", 1.3)) * first_width
    wid = hgt * float(o.get("zone_aspect", 1.0))
    g = float(o.get("zone_grow", 40))
    cx = anchor[0] + float(o.get("zone_dx", 0.0)) * first_width
    cy = anchor[1] - hgt * (0.5 - float(o.get("sink", 0.12))) + float(o.get("zone_dy", 0.0)) * first_width
    return cx, cy, wid / 2.0 + g, hgt / 2.0 + g


class ZoneGuide:
    """An object add job's zone (guide "zone"): where H3 redraws, frame by frame, for source frames lo..hi. Built from the
    target's cached masks the way CompositeGuide places a cut-out, so the Planner (the render) and the assembly (the
    paste-back) rebuild exactly the same zone from the same plan."""

    def __init__(self, plan, job_dir, fr, size, lo, hi, target):
        import os
        self.o = sp.object_settings(plan)
        masks = {}
        for seg, f0, f1 in sp.keep_segments(plan, {"target": target, "range": [lo, hi]}, lo, hi):
            rd = _SeqFrames(os.path.join(job_dir, seg["file"]), fr, int(seg["range"][0]))
            for f in range(f0, f1 + 1):
                masks[f] = rd.read(f).copy()
        self.size = size
        self.masks = masks
        self.anchors = anchors(masks, size)
        self.first_width = float(np.median([a[2] for a in self.anchors.values()])) if self.anchors else None

    def geometry(self, f):
        a = self.anchors.get(f)
        return None if a is None or not self.first_width else zone_geometry(a, self.first_width, sp.zone_at(self.o, f))

    def silhouette(self, f):
        """Zone shape "mask": the target's own mask on frame f, grown by zone_grow px, as uint8 [h, w] (or None where the
        target isn't found). For taking something OUT: H3 redraws exactly where the thing was, the rest is held."""
        t = self.masks.get(f)
        if t is None:
            return None
        cv2 = _cv2()
        w, h = self.size
        t = (t[..., 0] if t.ndim == 3 else t) > 127
        if not t.any():
            return None
        t = cv2.resize(t.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        g = max(0, int(round(float(self.o.get("zone_grow", 40)))))
        return cv2.dilate(t, _ellipse(g)) if g else t

    def box(self, f, pad=0):
        """The zone's box [x0, y0, x1, y1] in source pixels on frame f (clipped to the frame), or None."""
        if self.o.get("zone_shape") == "mask":
            m = self.silhouette(f)
            if m is None:
                return None
            ys, xs = np.nonzero(m)
            w, h = self.size
            return [max(0.0, float(xs.min() - pad - 1)), max(0.0, float(ys.min() - pad - 1)),
                    min(float(w), float(xs.max() + pad + 2)), min(float(h), float(ys.max() + pad + 2))]
        g = self.geometry(f)
        if g is None:
            return None
        cx, cy, rx, ry = g
        w, h = self.size
        pad += 1                                # the drawn zone rounds out to whole pixels: the box always holds it
        return [max(0.0, cx - rx - pad), max(0.0, cy - ry - pad), min(float(w), cx + rx + pad), min(float(h), cy + ry + pad)]

    def mask(self, f):
        """The zone on frame f: float32 [h, w], 1 inside; None where the target isn't found."""
        if self.o.get("zone_shape") == "mask":
            m = self.silhouette(f)
            return None if m is None else m.astype(np.float32)
        g = self.geometry(f)
        if g is None:
            return None
        cv2 = _cv2()
        w, h = self.size
        m = np.zeros((h, w), np.uint8)
        cx, cy, rx, ry = g
        if self.o.get("zone_shape") == "box":
            cv2.rectangle(m, (int(round(cx - rx)), int(round(cy - ry))), (int(round(cx + rx)), int(round(cy + ry))), 1, -1)
        else:
            cv2.ellipse(m, (int(round(cx)), int(round(cy))), (max(1, int(round(rx))), max(1, int(round(ry)))), 0, 0, 360, 1, -1)
        if self.o.get("zone_spare"):                        # the target stays as filmed, apart from where the new thing touches it
            t = self.masks.get(f)
            if t is not None:
                t = (t[..., 0] if t.ndim == 3 else t) > 127
                t = cv2.resize(t.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                m[cv2.dilate(t, _ellipse(SPARE_GROW)) > 0] = 0
            oz = sp.zone_at(self.o, f)
            r = float(oz.get("zone_touch_r", 0.0)) * self.first_width
            if r > 0:
                a = self.anchors[f]
                tx = a[0] + float(oz.get("zone_touch_dx", 0.0)) * self.first_width
                ty = a[1] + float(oz.get("zone_touch_dy", 0.0)) * self.first_width
                cv2.circle(m, (int(round(tx)), int(round(ty))), max(1, int(round(r))), 1, -1)
        return m.astype(np.float32)
