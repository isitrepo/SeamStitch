"""SeamStitch Swap Assemble: the job's effective takes, joined, over the source's own audio.

Streams chunk by chunk, so the whole video is never in memory (978 float frames at 1080p
would be 24 GB):
  1. each chunk's effective take (the chosen one, else its first) is decoded with the
     Timeline's decoder (timeline._iter_frames: PyAV, colour-tag aware);
  2. each join is typed from the takes' lineage (swap_plan.join_info) and repaired at its
     splice (swap_join: the measured lock, the level-matched fade, or a plain cut);
  3. the frames go straight into ONE ffmpeg encode with VHS's own format arguments, the
     same arguments recombine._encode_video (SeamStitch Result Preview's save) builds, so
     the colours are Result Preview's;
  4. the source's original audio is muxed in whole (stream copy): joins are picture-only;
  5. output frames must equal source frames (1:1) and the audio must be as long as the
     source's, else the assembly is refused;
  6. every join is measured (luma jumps, seam score, join ratio) and the report and the
     assembly record are written into plan.json.

A chunk with no usable take is filled with source frames and flagged (or refused, by choice).
"""

import json
import os
import subprocess
import threading
import time

import av
import numpy as np

try:
    from . import timeline as tl
    from . import swap_plan as sp
    from . import swap_join as sj
    from . import result_preview as rp
except ImportError:  # imported as a top-level module (tests, tools)
    import timeline as tl
    import swap_plan as sp
    import swap_join as sj
    import result_preview as rp

try:
    import folder_paths
except ImportError:  # pragma: no cover
    folder_paths = None

PENDING_PREVIEW = "source frames (preview)"
PENDING_REFUSE = "refuse"
DEFAULT_PREFIX = "swap_%date:yyyyMMdd_hhmmss%"
SEAM_SIDE = 5                     # frames either side of a splice for the seam score (bridge_join)


class AssembleError(Exception):
    pass


def formats():
    """Result Preview's format list (VHS's own, recombine.pack_video_formats), minus any `.json`
    entry another pack adds (they can save untagged)."""
    return [f for f in rp._formats() if not f.endswith(".json")] or ["video/h264-mp4"]


def _recombine():
    try:
        from . import recombine as rc
    except ImportError:
        import recombine as rc
    return rc


def _format_spec(fmt, crf, pix_fmt):
    """VHS's resolved format (main_pass, extension, audio_pass...), with the node's settings
    applied as Result Preview applies them (format_settings), metadata off."""
    rc = _recombine()
    kw = rp.format_settings(fmt, crf, pix_fmt, save_metadata=False)
    return rc.apply_format_widgets(fmt.split("/")[-1], dict(kw))


def _ffmpeg_path():
    try:
        return _recombine().ffmpeg_path or tl._ffmpeg_exe()
    except Exception:
        return tl._ffmpeg_exe()


def _merge_filter_args(args, ftype="-vf"):
    """VHS utils.merge_filter_args: fold repeated -vf into one chain."""
    try:
        start = args.index(ftype) + 1
        i = start
        while True:
            i = args.index(ftype, i)
            args[start] += "," + args[i + 1]
            args.pop(i)
            args.pop(i)
    except ValueError:
        pass


class StreamEncoder:
    """recombine._encode_video's ffmpeg command, fed one uint8 RGB frame at a time."""

    def __init__(self, path, w, h, fr, spec, ffmpeg=None):
        self.path = path
        self.depth16 = spec.get("input_color_depth", "8bit") == "16bit"
        self.align = int(spec.get("dim_alignment", 2))
        self.pad = (-w % self.align, -h % self.align)
        W, H = w + self.pad[0], h + self.pad[1]
        args = [ffmpeg or _ffmpeg_path(), "-v", "error", "-f", "rawvideo", "-pix_fmt",
                "rgb48" if self.depth16 else "rgb24",
                "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
                "-color_trc", spec.get("fake_trc", "iec61966-2-1"),
                "-s", f"{W}x{H}", "-r", str(fr), "-i", "-"]
        if "inputs_main_pass" in spec:
            n = args.index("-i") + 2
            args = args[:n] + list(spec["inputs_main_pass"]) + args[n:]
        bitrate = spec.get("bitrate")
        args += list(spec["main_pass"])
        if bitrate is not None:
            args += ["-b:v", str(bitrate) + ("M" if spec.get("megabit") == "True" else "K")]
        _merge_filter_args(args)
        self.args = args + [path]
        env = os.environ.copy()
        env.update(spec.get("environment", {}))
        self.proc = subprocess.Popen(self.args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self._err = []
        self._t = threading.Thread(target=lambda: self._err.append(self.proc.stderr.read()), daemon=True)
        self._t.start()
        self.frames = 0

    def write(self, frame):
        if self.pad != (0, 0):
            p = ((self.pad[1] // 2, self.pad[1] - self.pad[1] // 2), (self.pad[0] // 2, self.pad[0] - self.pad[0] // 2), (0, 0))
            frame = np.pad(frame, p, mode="edge")
        if self.depth16:
            frame = frame.astype(np.uint16) * 257          # == VHS tensor_to_shorts(u8 / 255)
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            self.close()
            raise AssembleError(f"ffmpeg stopped while encoding: {self.error()}")
        self.frames += 1

    def error(self):
        return b"".join(e or b"" for e in self._err).decode(errors="replace").strip()[-800:]

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait()
        self._t.join(timeout=10)
        if self.proc.returncode != 0:
            raise AssembleError(f"ffmpeg failed encoding {os.path.basename(self.path)}: {self.error()}")


class FrameReader:
    """Sequential frames of one file by SOURCE frame number: frame f is the file's frame
    f - offset (a take's offset is its render start; the source's is 0)."""

    def __init__(self, path, fr, offset, size):
        self.path, self.fr, self.offset, self.size = path, fr, int(offset), size
        self.next_f = None
        self.it = None
        self.resized = False

    def read(self, f):
        if self.it is None or f != self.next_f:
            if self.it is not None and f < self.next_f:
                raise AssembleError(f"{os.path.basename(self.path)}: frame {f} requested after {self.next_f - 1}")
            self.it = tl._iter_frames(self.path, self.fr, f - self.offset, None)
            self.next_f = f
        try:
            frame = next(self.it)
        except StopIteration:
            raise AssembleError(f"{os.path.basename(self.path)} ended before source frame {f} "
                                f"(file frame {f - self.offset})")
        self.next_f = f + 1
        w, h = self.size
        if frame.shape[1] != w or frame.shape[0] != h:
            import cv2
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if frame.shape[1] > w else cv2.INTER_CUBIC)
            self.resized = True
        return frame


def read_range(path, fr, offset, a, b, size):
    r = FrameReader(path, fr, offset, size)
    return [r.read(f) for f in range(a, b + 1)]


def resolve_plan(value):
    """assemble_plan -> plan.json path: a plan file, a job folder, or a job name under
    output/seamstitch_swap/."""
    v = (value or "").strip().strip('"')
    if not v:
        raise AssembleError("assemble_plan is empty: wire the Planner's assemble_plan, or give a job's plan.json")
    if os.path.isfile(v):
        return os.path.abspath(v)
    if os.path.isdir(v) and os.path.isfile(sp.plan_path(v)):
        return os.path.abspath(sp.plan_path(v))
    if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
        p = sp.plan_path(sp.job_dir(folder_paths.get_output_directory(), v))
        if os.path.isfile(p):
            return p
    raise AssembleError(f"no plan found for {value!r}")


def _unique(path):
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{stem}_{i:05d}{ext}"):
        i += 1
    return f"{stem}_{i:05d}{ext}"


def _audio_seconds(path):
    """Duration of the first audio stream from its packets (exact for a stream copy)."""
    with av.open(path) as c:
        if not c.streams.audio:
            return None
        a = c.streams.audio[0]
        tb = float(a.time_base)
        first, end = None, 0.0
        for p in c.demux(a):
            if p.pts is None:
                continue
            t = p.pts * tb
            first = t if first is None else min(first, t)
            end = max(end, t + (p.duration or 0) * tb)
        return None if first is None else end - first


def _mux_audio(video, source, out, spec):
    """The source's first audio track, whole, beside the assembled picture: stream copy, or the
    format's own audio codec when the container won't take the source's."""
    ff = _ffmpeg_path()
    flags = ["-movflags", "+faststart"] if out.lower().endswith((".mp4", ".mov", ".m4v")) else []
    base = [ff, "-v", "error", "-y", "-i", video, "-i", source, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
    p = subprocess.run(base + ["-c:a", "copy"] + flags + [out], capture_output=True)
    if p.returncode == 0:
        return "copy"
    p2 = subprocess.run(base + list(spec.get("audio_pass", ["-c:a", "aac"])) + flags + [out], capture_output=True)
    if p2.returncode != 0:
        raise AssembleError(f"ffmpeg failed muxing the source audio: {p2.stderr.decode(errors='replace')[-600:]}")
    return " ".join(spec.get("audio_pass", ["-c:a", "aac"]))


def _mux_audio_window(video, source, out, spec, t0, dur):
    """The source's audio from t0 for dur seconds beside the picture, re-encoded with the format's
    own audio codec (a stream copy can only cut on packet boundaries)."""
    ff = _ffmpeg_path()
    flags = ["-movflags", "+faststart"] if out.lower().endswith((".mp4", ".mov", ".m4v")) else []
    a_pass = list(spec.get("audio_pass", ["-c:a", "aac"]))
    p = subprocess.run([ff, "-v", "error", "-y", "-i", video, "-ss", f"{max(0.0, t0):.6f}", "-t", f"{dur:.6f}",
                        "-i", source, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"] + a_pass
                       + ["-af", f"apad=whole_dur={dur:.6f}", "-t", f"{dur:.6f}"] + flags + [out], capture_output=True)
    if p.returncode != 0:
        raise AssembleError(f"ffmpeg failed muxing the source audio: {p.stderr.decode(errors='replace')[-600:]}")
    return " ".join(a_pass)


FOLLOW_SIDE = 25                 # frames either side of a splice for the join's following (pose IoU)


def _src_mask_path(job, take):
    for name in ("mask_src.mkv", "mask_src.mp4"):
        p = os.path.join(job, os.path.dirname(take["file"]), name)
        if os.path.isfile(p):
            return p
    return None


def _join_follow(job, a, b, splice, lo, hi):
    """Pose IoU (the SAM3 person masks, source vs output: r10's measure) over source frames
    lo..hi around a splice: the left take's frames before it, the right take's from it.
    None when either take lacks its masks."""
    if a is None or b is None:
        return None
    paths = [(_mask_path(job, t), _src_mask_path(job, t)) for t in (a, b)]
    if not all(p for pair in paths for p in pair):
        return None
    vals = []
    for (mo, ms), t, f0, f1 in ((paths[0], a, lo, splice - 1), (paths[1], b, splice, hi)):
        if f1 < f0:
            continue
        out = _mask_frames(mo, int(t["render"][0]), f0, f1)
        src = _mask_frames(ms, int(t["render"][0]), f0, f1)
        for f in range(f0, f1 + 1):
            if f in out and f in src:
                v = sj.iou(src[f], out[f])
                if v is not None:
                    vals.append(v)
    if not vals:
        return None
    return {"pose_iou": round(float(np.mean(vals)), 4), "pose_iou_p10": round(float(np.percentile(vals, 10)), 4),
            "frames": len(vals), "range": [lo, hi]}


def _mask_path(job, take):
    for name in ("mask_out.mkv", "mask_out.mp4"):
        p = os.path.join(job, os.path.dirname(take["file"]), name)
        if os.path.isfile(p):
            return p
    return None


def _mask_frames(path, offset, a, b):
    """Bool masks for source frames a..b from a take's mask video (frame i = take frame i)."""
    with av.open(path) as c:
        rate = c.streams.video[0].average_rate
    fr = float(rate) if rate else 25.0
    out = {}
    for i, m in enumerate(tl._iter_frames(path, fr, a - offset, b - offset + 1)):
        out[a + i] = m[..., 0] > 127
    return out


def assemble(plan_file, hand_back=12, fmt="video/h264-mp4", crf=12, pix_fmt="yuv420p", filename_prefix=DEFAULT_PREFIX,
             pending_chunks=PENDING_PREVIEW, require_reviewed=False, lock_options=None, measure=True,
             write_plan=True, progress=None, window=None, out_file=None, plan=None, lock_regions=True):
    """Build the assembly. Returns the report (also written beside the video and into plan.json).

    window=(lo, hi): only source frames lo..hi (inclusive), with the source's audio under them,
    into out_file (the Take's review clip: the same joins, typed and repaired the same way, on
    a stretch of the video). plan: a plan dict to use instead of the file's (e.g. with a take
    marked chosen, "as if this take were chosen"); plan_file still locates the job.
    lock_regions: the regional lock (character and background each locked, through the right
    take's feathered SAM3 mask) wherever both takes have their masks; a split's repair
    {"regions": false} turns it off there. The lock is always cut-aware (swap_join)."""
    t_start = time.time()
    plan_file = resolve_plan(plan_file)
    job = os.path.dirname(plan_file)
    plan = plan if plan is not None else sp.load_plan(plan_file)
    src = plan["source"]
    source = src["path"]
    if not os.path.isfile(source):
        raise AssembleError(f"the source video is missing: {source}")
    fr = int(round(float(src["fps"])))
    info = tl.probe(source, fr)
    N = int(info["frames"])
    if N != int(src["frames"]):
        raise AssembleError(f"the source now has {N} frames at {fr} fps; the plan was made for {src['frames']}")
    w, h = int(info["width"]), int(info["height"])
    settings = dict(plan.get("settings") or {})
    settings["hand_back"] = int(hand_back)
    chunks = plan.get("chunks") or []
    if not chunks:
        raise AssembleError("the plan has no chunks")
    if chunks[0]["deliver"][0] != 0 or chunks[-1]["deliver"][1] != N - 1 or any(
            chunks[i]["deliver"][0] != chunks[i - 1]["deliver"][1] + 1 for i in range(1, len(chunks))):
        raise AssembleError("the plan's chunks don't tile the source: open it in the Planner to rebuild them")

    flags = []
    eff = []
    for c in chunks:
        t, state = sp.effective_take(c)
        if t is not None and not sp.take_covers(t, *c["deliver"]):
            flags.append({"chunk": c["id"], "code": "range_changed",
                          "text": f"chunk {c['id']}: take {t['id']} renders {t['render']}, which no longer "
                                  f"covers the chunk {c['deliver']} (a split moved)"})
            t, state = None, "range changed"
        if t is not None:
            path = os.path.join(job, t["file"])
            if not os.path.isfile(path):
                raise AssembleError(f"chunk {c['id']}: take {t['id']}'s file is missing: {path}")
            t = dict(t, _path=path)
        if t is None:
            if pending_chunks == PENDING_REFUSE:
                raise AssembleError(f"chunk {c['id']} {c['deliver']} has no take (pending_chunks is 'refuse')")
            flags.append({"chunk": c["id"], "code": "pending",
                          "text": f"chunk {c['id']} {c['deliver']}: no take, filled with source frames"})
        elif state == "unreviewed":
            if require_reviewed:
                raise AssembleError(f"chunk {c['id']}: take {t['id']} is unreviewed (require_reviewed is on)")
            flags.append({"chunk": c["id"], "code": "unreviewed", "text": f"chunk {c['id']}: {t['id']} is unreviewed"})
        eff.append((t, state))

    splits = {d["id"]: d for d in plan.get("splits", [])}
    joins = []
    for k in range(1, len(chunks)):
        split = splits.get(chunks[k].get("left"))
        if split is None:
            raise AssembleError(f"chunk {chunks[k]['id']} has no left split")
        j = sp.join_info(split, eff[k - 1][0], eff[k][0], settings)
        j.update(left_chunk=chunks[k - 1]["id"], right_chunk=chunks[k]["id"],
                 left_take=eff[k - 1][0] and eff[k - 1][0]["id"], right_take=eff[k][0] and eff[k][0]["id"])
        opts = dict(lock_options or {})
        opts.update({k2: bool(v) for k2, v in (split.get("repair") or {}).items() if k2 in ("gate", "clamp", "swing")})
        j["lock_options"] = {k2: v for k2, v in opts.items() if v}
        if j["stale"]:
            flags.append({"split": j["split"], "code": "stale",
                          "text": f"split {j['split']} at {j['frame']}: stale join ({j['left_take']} | {j['right_take']}), "
                                  f"repair {j['override'] or j['repair']}: re-roll to fit"})
        joins.append(j)

    # each chunk's output range
    starts = [0] + [j["splice"] for j in joins]
    ends = [j["splice"] - 1 for j in joins] + [N - 1]
    for k, (a, b) in enumerate(zip(starts, ends)):
        if b < a:
            raise AssembleError(f"chunk {chunks[k]['id']}: its joins leave it no frames ({a}..{b})")
        t = eff[k][0]
        if t is not None and not sp.take_covers(t, a, b):
            raise AssembleError(f"chunk {chunks[k]['id']}: take {t['id']} {t['render']} doesn't cover its output {a}..{b}")

    f_lo, f_hi = (0, N - 1) if window is None else (max(0, int(window[0])), min(N - 1, int(window[1])))
    if f_hi < f_lo:
        raise AssembleError(f"window {window} holds no frames of the {N}-frame source")
    n_out = f_hi - f_lo + 1

    def reader(k, f0=None):
        t = eff[k][0]
        if t is None:
            return FrameReader(source, fr, 0, (w, h))
        return FrameReader(t["_path"], fr, int(t["render"][0]), (w, h))

    # repairs, measured on the takes' own frames before the stream
    cut_frames = sp.confirmed_cuts(plan.get("cuts"))
    for k, j in enumerate(joins):
        a, b = eff[k][0], eff[k + 1][0]
        s, H = j["splice"], j["hand_back"]
        j["gains"] = None
        j["decay"] = None
        j["soft"] = None
        if j["repair"] == sp.REPAIR_LOCK:
            lt = read_range(a["_path"], fr, a["render"][0], max(s - sj.FIT, int(a["render"][0])), s - 1, (w, h))
            rh_end = min(s + H - 1, ends[k + 1], int(b["render"][1]))
            rh = read_range(b["_path"], fr, b["render"][0], s, rh_end, (w, h))
            lm, rm = sj.means(lt), sj.means(rh)
            n = sj.opening_frames(rm, s, H, cut_frames)
            nl = sj.heading_frames(lm, s, cut_frames)
            j["lock_fit"] = {"left": nl, "right": n, "hand_back": H}
            if n == 0:
                j["repair"] = sp.REPAIR_NONE
                j["lock_mode"] = "none: the splice is a cut"
                continue
            lt, rh, lm, rm = lt[-nl:], rh[:n], lm[-nl:], rm[:n]
            split = splits.get(chunks[k + 1].get("left")) or {}
            ma, mb = _mask_path(job, a), _mask_path(job, b)
            if lock_regions and (split.get("repair") or {}).get("regions", True) and ma and mb:
                try:
                    ml = _mask_frames(ma, int(a["render"][0]), s - nl, s - 1)
                    mr = _mask_frames(mb, int(b["render"][0]), s, s + n - 1)
                    lc, lb = sj.region_means(lt, [ml[f] for f in range(s - nl, s)])
                    rc, rb = sj.region_means(rh, [mr[f] for f in range(s, s + n)])
                    if np.isfinite(lc).all() and np.isfinite(lb).all() and np.isfinite(rc).all() and np.isfinite(rb).all():
                        gch = sj.lock_gains(lc, rc, n, **j["lock_options"])
                        gbg = sj.lock_gains(lb, rb, n, **j["lock_options"])
                        soft = [sj.soft_mask(mr[f], (w, h)) for f in range(s, s + n)]
                        # each region's mean lands exactly where its own lock puts it, feather included
                        solved = [sj.solve_field_gains(rh[i], mr[s + i], soft[i], rc[i] * gch[i], rb[i] * gbg[i])
                                  for i in range(n)]
                        j["gains_char"] = [x[0] for x in solved]
                        j["gains_bg"] = [x[1] for x in solved]
                        j["soft"] = soft
                        j["gains"] = gch
                        j["lock_mode"] = "regional"
                except Exception as e:              # masks unreadable: the global lock
                    j["lock_mode_note"] = f"regional lock skipped: {e}"
            if j["soft"] is None:
                j["gains"] = sj.lock_gains(lm, rm, n, **j["lock_options"])
                j["lock_mode"] = "global"
        elif j["repair"] == sp.REPAIR_FADE:
            f0, f1 = j["fade"]
            lm = sj.means(read_range(a["_path"], fr, a["render"][0], f0, f1, (w, h)))
            rm = sj.means(read_range(b["_path"], fr, b["render"][0], f0, f1, (w, h)))
            j["ratios"] = sj.fade_ratios(lm, rm)
            j["weights"] = sj.fade_weights(f1 - f0 + 1)
            j["decay"] = j["ratios"][-1]

    try:
        spec = _format_spec(fmt, crf, pix_fmt)
    except Exception as e:
        raise AssembleError(f"format {fmt}: {e}")
    if out_file:
        out_dir = os.path.dirname(os.path.abspath(out_file))
        os.makedirs(out_dir, exist_ok=True)
        final = os.path.abspath(out_file)
    else:
        out_dir = os.path.join(job, "assembled")
        os.makedirs(out_dir, exist_ok=True)
        stem = rp._expand_date(filename_prefix or DEFAULT_PREFIX)
        final = _unique(os.path.join(out_dir, f"{stem}.{spec['extension']}"))
    stem = os.path.splitext(os.path.basename(final))[0]
    video_only = os.path.join(out_dir, f"{stem}.video.part.{spec['extension']}")
    enc = StreamEncoder(video_only, w, h, fr, spec)

    lum = np.zeros(N, np.float64)
    thumbs = {}
    seam_frames = {f for j in joins for f in range(j["splice"] - SEAM_SIDE, j["splice"] + SEAM_SIDE)}
    readers = {}
    resized = set()
    try:
        for k, c in enumerate(chunks):
            left = joins[k - 1] if k > 0 else None
            right = joins[k] if k < len(joins) else None
            a_k, b_k = max(starts[k], f_lo), min(ends[k], f_hi)
            if b_k < a_k:
                readers.pop(k, None)
                continue
            rd = readers.pop(k, None) or reader(k)
            for f in range(a_k, b_k + 1):
                frame = rd.read(f)
                if left is not None:
                    i = f - left["splice"]
                    if left.get("soft") is not None and i < len(left["soft"]):
                        frame = sj.apply_gain_field(frame, left["gains_char"][i], left["gains_bg"][i], left["soft"][i])
                    elif left["gains"] is not None and i < len(left["gains"]):
                        frame = sj.apply_gain(frame, left["gains"][i])
                    elif left["decay"] is not None:
                        g = sj.decay_gain(left["decay"], i, left["hand_back"])
                        if g is not None:
                            frame = sj.apply_gain(frame, g)
                if right is not None and right["repair"] == sp.REPAIR_FADE and f >= right["fade"][0]:
                    if k + 1 not in readers:
                        readers[k + 1] = reader(k + 1)
                    r = readers[k + 1].read(f)
                    i = f - right["fade"][0]
                    frame = sj.fade_frame(frame, r, right["ratios"][i], right["weights"][i])
                enc.write(frame)
                lum[f] = sj.luma(frame)
                if f in seam_frames:
                    thumbs[f] = sj.gray(sj.thumb(frame))
                if progress:
                    progress(f - f_lo + 1, n_out)
            if rd.resized:
                resized.add(c["id"])
        enc.close()
    except BaseException:
        try:
            enc.close()
        except Exception:
            pass
        try:
            os.remove(video_only)
        except OSError:
            pass
        raise
    for cid in sorted(resized):
        flags.append({"chunk": cid, "code": "resized", "text": f"chunk {cid}: take frames resized to {w}x{h}"})

    # the original audio, whole (or the window's stretch of it)
    try:
        if info["has_audio"] and window is not None:
            audio_mode = _mux_audio_window(video_only, source, final, spec, f_lo / fr, n_out / fr)
        elif info["has_audio"]:
            audio_mode = _mux_audio(video_only, source, final, spec)
        else:
            os.replace(video_only, final)
            audio_mode = None
    finally:
        if os.path.exists(video_only):
            try:
                os.remove(video_only)
            except OSError:
                pass

    # 1:1 and audio checks
    out_info = tl.probe(final, fr)
    checks = {"frames": out_info["frames"], "source_frames": N, "frames_ok": out_info["frames"] == n_out == enc.frames,
              "frame_rate": fr, "audio": audio_mode}
    if window is not None:
        checks["window"] = [f_lo, f_hi]
    if info["has_audio"]:
        a_out = _audio_seconds(final)
        a_src = _audio_seconds(source) if window is None else n_out / fr
        checks.update(audio_seconds=round(a_out or 0, 4), source_audio_seconds=round(a_src or 0, 4),
                      audio_ok=a_out is not None and a_src is not None and abs(a_out - a_src) <= 1.0 / fr)
    else:
        checks["audio_ok"] = True
    if not checks["frames_ok"] or not checks["audio_ok"]:
        bad = final + ".refused"
        os.replace(final, _unique(bad))
        raise AssembleError(f"assembly refused: {checks} (kept as {os.path.basename(bad)})")

    # measurements
    rows = []
    for k, j in enumerate(joins):
        if not f_lo < j["splice"] <= f_hi:
            continue
        a, b = eff[k][0], eff[k + 1][0]
        row = {key: j[key] for key in ("split", "frame", "type", "splice", "repair", "override", "hand_back", "fade",
                                       "linked", "stale", "left_chunk", "right_chunk", "left_take", "right_take",
                                       "lock_options")}
        if j["gains"] is not None:
            row["lock_gain_first"] = [round(float(x), 5) for x in j["gains"][0]]
        for key in ("lock_mode", "lock_fit", "lock_mode_note"):
            if j.get(key) is not None:
                row[key] = j[key]
        if j.get("gains_bg") is not None:
            row["lock_gain_first_bg"] = [round(float(x), 5) for x in j["gains_bg"][0]]
        if j.get("ratios") is not None:
            row["fade_ratio_last"] = [round(float(x), 5) for x in j["ratios"][-1]]
        if measure:
            lo_t = max(int(a["render"][0]) if a else starts[k], f_lo)
            hi_t = min(int(b["render"][1]) if b else ends[k + 1], f_hi)
            r_start = int(b["render"][0]) if b else j["splice"]
            lo, hi = sj.measure_window(r_start, j["splice"], j["hand_back"])
            row["frame_luma"] = sj.jumps(lum[lo_t:hi_t + 1], lo_t, j["splice"], max(lo, lo_t), min(hi, hi_t + 1))
            ma = _mask_path(job, a) if a else None
            mb = _mask_path(job, b) if b else None
            if ma and mb:
                lo_c, hi_c = max(lo, lo_t), min(hi, hi_t + 1)
                # one track across the splice where it can (r16's method: one SAM3 track over the
                # joined file): the right take's mask over its own overlap, the left take's before it
                b0 = max(lo_c, int(b["render"][0]))
                masks = _mask_frames(ma, int(a["render"][0]), lo_c, b0 - 1) if b0 > lo_c else {}
                masks.update(_mask_frames(mb, int(b["render"][0]), b0, hi_c - 1))
                fr_out = FrameReader(final, fr, f_lo, (w, h))
                cl = [sj.luma(fr_out.read(f), masks.get(f)) for f in range(lo_c, hi_c)]
                row["char_luma"] = sj.jumps(cl, lo_c, j["splice"], lo_c, hi_c)
            s = j["splice"]
            if all(f in thumbs for f in range(s - SEAM_SIDE, s + SEAM_SIDE)):
                row["seam"] = sj.seam_score([thumbs[f] for f in range(s - SEAM_SIDE, s)],
                                            [thumbs[f] for f in range(s, s + SEAM_SIDE)])
            ratio, step = rp.join_ratio(final, fr, s - f_lo)
            row["join_ratio"] = None if ratio is None else round(float(ratio), 3)
            row["join_verdict"] = rp.verdict(ratio)
            follow = _join_follow(job, a, b, s, max(f_lo, s - FOLLOW_SIDE), min(f_hi, s + FOLLOW_SIDE - 1))
            if follow is not None:
                row["follow"] = follow
        rows.append(row)

    report = {"file": final, "frames": out_info["frames"], "window": None if window is None else [f_lo, f_hi], "fps": fr, "size": [w, h], "format": fmt, "crf": int(crf),
              "pix_fmt": pix_fmt, "plan_rev": plan.get("rev"), "job": plan.get("job"),
              "takes": {c["id"]: {"take": t and t["id"], "state": st} for c, (t, st) in zip(chunks, eff)},
              "joins": rows, "flags": flags, "checks": checks, "seconds": round(time.time() - t_start, 1),
              "created": sp.now()}
    with open(os.path.join(out_dir, f"{stem}.report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    if write_plan and window is None:
        def upd(p):
            cache = p.setdefault("join_cache", {})
            for r in rows:
                key = f"{r['split']}|{r['left_take']}|{r['right_take']}|{r['override'] or r['repair']}|{r['hand_back']}"
                cache[key] = {kk: r.get(kk) for kk in ("type", "splice", "frame_luma", "char_luma", "seam", "join_ratio",
                                                        "join_verdict", "follow", "lock_options") if r.get(kk) is not None}
            p.setdefault("assembled", []).append({
                "file": os.path.relpath(final, job).replace("\\", "/"), "frames": report["frames"],
                "takes": {cid: v["take"] for cid, v in report["takes"].items()}, "created": report["created"],
                "format": fmt, "flags": [fl["code"] for fl in flags]})
        sp.update_plan(plan_file, upd)
    return report


# ---------------------------------------------------------------------------
# the node and its route
# ---------------------------------------------------------------------------

def _summary(report):
    lines = [f"{report['frames']} frames at {report['fps']} fps -> {report['file']}"]
    for r in report["joins"]:
        fl = r.get("frame_luma") or {}
        lines.append(f"  {r['split']} @ {r['frame']}: {r['type']}, splice {r['splice']}, "
                     f"{r['override'] or r['repair']}; jump {fl.get('at_splice')} (max {fl.get('max')}), "
                     f"ratio {r.get('join_ratio')} {r.get('join_verdict', '')}")
    for f in report["flags"]:
        lines.append(f"  ! {f['text']}")
    return "\n".join(lines)


class SeamStitchSwapAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "assemble_plan": ("STRING", {"default": "", "tooltip":
                    "The job's plan: wire the Swap Planner's assemble_plan, or give a plan.json path, a job "
                    "folder, or a job name under output/seamstitch_swap/."}),
                "hand_back_frames": ("INT", {"default": 12, "min": 2, "max": 100, "step": 1, "tooltip":
                    "Frames the level lock (and the fade's level match) take to hand back to the right take's "
                    "own grade, where a split has no override. 12 (picked by eye); 25 leaves a smaller dip."}),
                "format": (formats(), {"default": "video/h264-mp4", "tooltip":
                    "VHS's own formats and encode path, as Result Preview. video/ffv1-mkv for a lossless master."}),
                "crf": ("INT", {"default": 12, "min": 0, "max": 51, "step": 1}),
                "pix_fmt": (["yuv420p", "yuv420p10le"], {"default": "yuv420p"}),
                "filename_prefix": ("STRING", {"default": DEFAULT_PREFIX, "tooltip":
                    "Written to the job's assembled/ folder. %date:yyyyMMdd_hhmmss% is replaced."}),
                "pending_chunks": ([PENDING_PREVIEW, PENDING_REFUSE], {"default": PENDING_PREVIEW, "tooltip":
                    "A chunk with no take: fill it with source frames (flagged), or refuse to assemble."}),
                "require_reviewed": ("BOOLEAN", {"default": False, "tooltip":
                    "Off: an unreviewed take (a chunk's first, never chosen) is used and flagged. On: refuse."}),
            },
        }

    RETURN_TYPES = ("STRING", "VHS_FILENAMES", "STRING")
    RETURN_NAMES = ("video_path", "Filenames", "report")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "SeamStitch/Swap"
    DESCRIPTION = ("Joins every chunk's effective take (lock, fade or cut per its lineage), over the source's "
                   "original audio, in one streamed encode. CPU only.")

    def run(self, assemble_plan, hand_back_frames=12, format="video/h264-mp4", crf=12, pix_fmt="yuv420p",
            filename_prefix=DEFAULT_PREFIX, pending_chunks=PENDING_PREVIEW, require_reviewed=False):
        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pass

        def progress(i, n):
            if pbar is not None and i % 10 == 0:
                pbar.update_absolute(int(100 * i / n), 100)

        report = assemble(assemble_plan, hand_back_frames, format, crf, pix_fmt, filename_prefix, pending_chunks,
                          require_reviewed, progress=progress)
        text = _summary(report)
        print("[SeamStitch] Swap Assemble: " + text.replace("\n", "\n[SeamStitch] Swap Assemble: "))
        view = rp._view_params(report["file"]) if folder_paths is not None else None
        ui = {"file": report["file"], "view": view, "frames": report["frames"], "fps": report["fps"],
              "joins": [{k: r.get(k) for k in ("split", "frame", "splice", "type", "repair", "override", "frame_luma", "char_luma",
                                                "join_ratio", "join_verdict")} for r in report["joins"]],
              "flags": report["flags"]}
        return {"ui": {"seamstitch_swap_assemble": [ui]},
                "result": (report["file"], (True, [report["file"]]), json.dumps(report))}


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/seamstitch/swap/assemble")
    async def _assemble_route(request):
        """The Assemble node's code, off the queue (the Planner's assemble button). CPU only."""
        try:
            body = await request.json()
            import asyncio
            loop = asyncio.get_event_loop()
            report = await loop.run_in_executor(None, lambda: assemble(
                body.get("plan", ""), int(body.get("hand_back_frames", 12)), body.get("format", "video/h264-mp4"),
                int(body.get("crf", 12)), body.get("pix_fmt", "yuv420p"), body.get("filename_prefix", DEFAULT_PREFIX),
                body.get("pending_chunks", PENDING_PREVIEW), bool(body.get("require_reviewed", False))))
            return web.json_response(report)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
except Exception:  # pragma: no cover - no server (tools)
    pass
