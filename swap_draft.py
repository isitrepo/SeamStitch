"""SeamStitch Swap Draft Prompts: the captioner drafts every chunk's prompt before anything renders.

One queue item (design §4.6 / §5.5), one model on the GPU at a time:
  1. free ComfyUI's models (unload_all_models, soft_empty_cache);
  2. Omni Captioner Transcribe on each chunk's original audio: the words and the audio events
     (its own llama.cpp process per chunk);
  3. faster-whisper large-v3-turbo, loaded once: the word timings. Omni's words are placed on
     Whisper's timings by sequence alignment and grouped into lines with frame ranges;
  4. QwenVL (the AILab_QwenVL node, called in-process), loaded once: the subject from the sheet,
     then per chunk the objects it sees and the draft (video_1 / shots / sounds) from 16 frames over
     the render range plus the shots, the dialogue lines and the previous chunk's shot blocks;
     then unloaded;
  5. the node fills the six-section Ref2VA template and writes each chunk's draft.
Qwen needs the dialogue to place each line at its action, so Omni and Whisper run before it (the
design lists Qwen first; the dependency decides the order).

Each stage writes its files under drafts/<chunk>/ before the next starts (omni.txt, whisper.json,
qwen.txt, draft.txt, check.json); free VRAM is logged between stages and the device-wide peak is
sampled throughout. A draft never overwrites a prompt: it fills an empty prompt, otherwise it goes
into the chunk's `draft` field (checked under the plan lock at write time). Kept chunks are never
drafted (swap_planner.draft_chunks).
"""

import difflib
import gc
import importlib
import importlib.util
import json
import os
import re
import sys
import threading
import time

import numpy as np
import torch

try:
    from . import swap_plan as sp
    from . import swap_planner as spl
    from . import timeline as tl
except ImportError:  # imported as a top-level module (tests, tools)
    import swap_plan as sp
    import swap_planner as spl
    import timeline as tl

_say = spl._say

QWEN_CLASS = "AILab_QwenVL"
OMNI_CLASS = "OmniCaptionerTranscribe"
DEFAULT_QWEN = "Qwen3-VL-8B-Unredacted-MAX (Captioner)"
QUANTIZATIONS = ["None (FP16)", "8-bit (Balanced)", "4-bit (VRAM-friendly)"]
CHUNK_MODES = ["empty only", "all (into the draft field)", "selected"]
TEMPLATES = ["character replace (Ref2VA)", "character replace (Ref2VA, per shot)"]
# per shot: each shot described from its own frames (B4 workshop: the one-call form copied across shots and
# chunks, or looped; per shot followed the foam, card, booklet and stickies of 400-608 shot by shot)
DEFAULT_TEMPLATE = TEMPLATES[1]
DEFAULT_FRAMES = 24             # spread over the shots (B4 workshop: 24 named the foam sheet and the booklet, 16 didn't)
WHISPER_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
            "overall_soundscape", "non_diegetic_music")
# Qwen sampling: the QwenVL node's temperature / top_p; repetition penalty 1.15 (its process() fixes 1.2). At 1.05
# Qwen looped one sentence to the token limit (B4 workshop A1); the speech labels are the node's, not Qwen's.
QWEN_SAMPLING = {"temperature": 0.6, "top_p": 0.9, "num_beams": 1, "repetition_penalty": 1.15, "seed": 1}
QWEN_FRAME_MAX_SIDE = 1024     # pre-shrink only; QwenVL's own auto budget (16 frames, 8k ctx) takes them to ~530
LANGS = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
         "nl": "Dutch", "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "ru": "Russian"}

_CUSTOM_NODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DraftError(Exception):
    pass


# ---------------------------------------------------------------------------
# locating the captioner nodes (in-process, the way Recombine locates VHS)
# ---------------------------------------------------------------------------

def _registered(name):
    """A node class ComfyUI has already loaded (the normal case inside ComfyUI)."""
    try:
        import nodes
        return nodes.NODE_CLASS_MAPPINGS.get(name)
    except Exception:
        return None


def _scan_custom_nodes(marker, names=()):
    """The custom_nodes sibling holding `marker` (a relative file path): the usual folder names
    first, then every sibling (a renamed folder)."""
    for n in names:
        cand = os.path.join(_CUSTOM_NODES_DIR, n)
        if os.path.isfile(os.path.join(cand, marker)):
            return cand
    try:
        for entry in os.listdir(_CUSTOM_NODES_DIR):
            cand = os.path.join(_CUSTOM_NODES_DIR, entry)
            if os.path.isfile(os.path.join(cand, marker)):
                return cand
    except OSError:
        pass
    return None


def locate_qwen():
    """AILab_QwenVL's class, or None. Outside ComfyUI's registry: ComfyUI-QwenVL's py/ folder is put
    on sys.path (its modules import each other as top-level modules) and AILab_QwenVL imported."""
    cls = _registered(QWEN_CLASS)
    if cls is not None:
        return cls
    d = _scan_custom_nodes(os.path.join("py", "AILab_QwenVL.py"), ("ComfyUI-QwenVL",))
    if d is None:
        return None
    # py/ must come first: the pack's root can hold stale copies of its py/ modules (its __init__ does the same)
    for p in (d, os.path.join(d, "py")):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    try:
        return getattr(importlib.import_module("AILab_QwenVL"), QWEN_CLASS)
    except Exception as e:
        _say(f"[SeamStitch] Swap Draft: ComfyUI-QwenVL found at {d} but didn't import: {e}")
        return None


def locate_omni():
    cls = _registered(OMNI_CLASS)
    if cls is not None:
        return cls
    d = _scan_custom_nodes("omni_captioner_node.py", ("ComfyUI-OmniCaptioner-GGUF",))
    if d is None:
        return None
    try:
        spec = importlib.util.spec_from_file_location("seamstitch_omni_captioner_node",
                                                      os.path.join(d, "omni_captioner_node.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, OMNI_CLASS)
    except Exception as e:
        _say(f"[SeamStitch] Swap Draft: the Omni captioner found at {d} but didn't import: {e}")
        return None


def qwen_models():
    """The QwenVL model list (its HF VL models), for the qwen_model widget."""
    cls = locate_qwen()
    mod = sys.modules.get(getattr(cls, "__module__", ""), None) if cls else None
    try:
        mod.load_model_configs()
        names = list(mod.HF_VL_MODELS.keys())
    except Exception:
        names = []
    return names or [DEFAULT_QWEN]


# ---------------------------------------------------------------------------
# VRAM: free memory between stages, and the device-wide peak (Omni is another process)
# ---------------------------------------------------------------------------

def free_vram_gb():
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return round(free / 2 ** 30, 2), round(total / 2 ** 30, 2)
    except Exception:
        pass
    return None, None


class VramSampler(threading.Thread):
    """Device-wide used VRAM (NVML), sampled every `interval` s, with the peak per stage."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.stage = "setup"
        self.peak = 0.0
        self.by_stage = {}
        self._stop = threading.Event()
        self.ok = False

    def run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.ok = True
        except Exception:
            return
        while not self._stop.is_set():
            try:
                used = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2 ** 30
            except Exception:
                break
            self.peak = max(self.peak, used)
            self.by_stage[self.stage] = max(self.by_stage.get(self.stage, 0.0), used)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        return {"device_peak_gb": round(self.peak, 2) if self.ok else None,
                "device_peak_by_stage_gb": {k: round(v, 2) for k, v in self.by_stage.items()} if self.ok else {}}


# ---------------------------------------------------------------------------
# geometry: shots, frames
# ---------------------------------------------------------------------------

def chunk_shots(plan, r0, r1):
    """The chunk's shots over its render range r0..r1 (source frames), split at the confirmed cuts
    (a cut at c: frame c starts a new shot). Returns [(a, b)] in source frames."""
    cuts = [c for c in sp.confirmed_cuts(plan.get("cuts")) if r0 < c <= r1]
    edges = [r0] + cuts + [r1 + 1]
    return [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]


def sample_indices(n, k=16):
    """k frame indices spread evenly over 0..n-1, as QwenVL's own sample_video_frames picks them."""
    if n <= k:
        return list(range(n))
    return [int(i) for i in np.linspace(0, n - 1, k, dtype=int)]


def decode_sampled(path, fr, r0, idx, max_side=QWEN_FRAME_MAX_SIDE):
    """The frames r0 + idx of a video as a float IMAGE batch, shrunk to max_side (one decode pass)."""
    want = set(idx)
    last = max(idx)
    got = {}
    for i, f in enumerate(tl._iter_frames(path, fr, r0, r0 + last + 1)):
        if i in want:
            h, w = f.shape[:2]
            if max(h, w) > max_side:
                import cv2
                s = max_side / float(max(h, w))
                f = cv2.resize(f, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            got[i] = torch.from_numpy(np.ascontiguousarray(f)).to(torch.float32).div_(255.0)
    if len(got) != len(want):
        raise DraftError(f"{os.path.basename(path)}: decoded {len(got)} of {len(want)} sampled frames from {r0}")
    return torch.stack([got[i] for i in idx], 0)


def chunk_audio(path, fr, r0, r1):
    """The render range's ORIGINAL audio (the source clock: Omni and Whisper hear it as it is), as
    (wav [C, n] float32, sample_rate), or (None, None) without an audio track."""
    try:
        wav, sr = tl.cut_audio({"pieces": [{"path": path, "enter": r0, "exit": r1 + 1}], "frames": r1 - r0 + 1}, fr)
    except Exception as e:
        _say(f"[SeamStitch] Swap Draft: no audio for {r0}-{r1}: {e}")
        return None, None
    if wav is None or not np.size(wav):
        return None, None
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


def to_mono_16k(wav, sr):
    x = torch.from_numpy(wav).float()
    if x.dim() == 2 and x.shape[0] > 1:
        x = x.mean(0, keepdim=True)
    elif x.dim() == 1:
        x = x.unsqueeze(0)
    if sr != 16000:
        import torchaudio
        x = torchaudio.functional.resample(x, sr, 16000)
    return x.squeeze(0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# dialogue: Omni's words on Whisper's timings
# ---------------------------------------------------------------------------

_BRACKET = re.compile(r"\[([^\]]+)\]")


def parse_omni(text):
    """Omni's answer to the node's default structured prompt -> {language, words (str), notes, events, speech}.
    Tolerates the label-less form (the quoted words then)."""
    t = " ".join((text or "").split())
    m = re.search(r"transcript\s*:\s*(.*?)\s*(?:audio[_ ]events\s*:\s*(.*))?$", t, re.I)
    structured = bool(m)
    if m:
        tr, ev = m.group(1).strip(), (m.group(2) or "").strip()
    else:
        # free prose (Omni ignores the structured prompt now and then): only its quoted words, no events
        quotes = re.findall(r'["“]([^"”]+)["”]', t)
        tr, ev = " ".join(quotes), ""
    lang, notes = None, []
    for b in _BRACKET.findall(tr):
        if lang is None and b.strip().istitle() and len(b.split()) == 1:
            lang = b.strip()
        else:
            notes.append(b.strip())
    words = " ".join(_BRACKET.sub(" ", tr).replace('"', " ").replace("“", " ").replace("”", " ").split())
    speech = bool(words) and words.strip(" .").lower() not in ("none", "no speech", "n/a")
    return {"language": lang, "words": words if speech else "", "notes": notes, "events": ev, "speech": speech,
            "structured": structured}


def _norm_tok(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def align_words(omni_words, whisper_words):
    """Omni's tokens (as written, punctuation kept) with times from Whisper's words, by sequence
    alignment on the normalised tokens: matched tokens take their Whisper word's times, a replaced
    run spreads its Whisper span over Omni's tokens, an Omni-only token is interpolated between its
    neighbours. Returns [{"word", "start", "end", "matched"}]."""
    toks = (omni_words or "").split()
    if not toks:
        return []
    ww = [w for w in whisper_words or [] if _norm_tok(w["word"])]
    a = [_norm_tok(t) for t in toks]
    b = [_norm_tok(w["word"]) for w in ww]
    out = [{"word": t, "start": None, "end": None, "matched": False} for t in toks]
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[i1 + k].update(start=ww[j1 + k]["start"], end=ww[j1 + k]["end"], matched=True)
        elif tag == "replace":
            s, e = ww[j1]["start"], ww[j2 - 1]["end"]
            n = i2 - i1
            for k in range(n):
                out[i1 + k].update(start=s + (e - s) * k / n, end=s + (e - s) * (k + 1) / n)
    # Omni-only tokens: between the known neighbours
    known = [i for i, o in enumerate(out) if o["start"] is not None]
    if not known:
        return out
    for i, o in enumerate(out):
        if o["start"] is not None:
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        if prev is None:
            o["start"] = o["end"] = out[nxt]["start"]
        elif nxt is None:
            o["start"] = o["end"] = out[prev]["end"]
        else:
            s, e = out[prev]["end"], out[nxt]["start"]
            span = nxt - prev
            o["start"] = s + (e - s) * (i - prev - 0.5) / span
            o["end"] = s + (e - s) * (i - prev + 0.5) / span
    return out


def dialogue_lines(aligned, fps, n_frames, gap_s=0.45, comma_gap_s=0.25):
    """Group timed words into spoken lines: a break after . ? ! …, after a comma with a pause, or at
    any longer pause. Frames are chunk-relative (0..n_frames-1) at the source rate."""
    lines, cur = [], []

    def close():
        if cur:
            s = cur[0]["start"]
            e = cur[-1]["end"]
            text = " ".join(w["word"] for w in cur)
            d = {"text": text}
            if s is not None:
                d.update(start=round(s, 3), end=round(e, 3),
                         frames=[max(0, min(n_frames - 1, int(round(s * fps)))),
                                 max(0, min(n_frames - 1, int(round(e * fps))))])
            lines.append(d)
            cur.clear()

    for i, w in enumerate(aligned):
        cur.append(w)
        nxt = aligned[i + 1] if i + 1 < len(aligned) else None
        if nxt is None:
            break
        pause = (nxt["start"] - w["end"]) if (nxt.get("start") is not None and w.get("end") is not None) else 0.0
        word = w["word"].rstrip('"”')
        if word.endswith((".", "?", "!", "…")) or pause >= gap_s or (word.endswith((",", ";", ":")) and pause >= comma_gap_s):
            close()
    close()
    return lines


def lines_from_text(words):
    """Lines without timings (Whisper missing): split at sentence ends."""
    parts = re.split(r"(?<=[.?!…])\s+", (words or "").strip())
    return [{"text": p} for p in parts if p]


# ---------------------------------------------------------------------------
# Qwen's instructions
# ---------------------------------------------------------------------------

SUBJECT_INSTRUCTION = """This picture is a character sheet: one character, shown from one or more angles. Write exactly three labelled lines and nothing else:

name: a short descriptive name for the character, 2-4 words, lower case (for example "clown angel girl" or "bearded sailor").
appearance: ONE sentence, a comma-separated list from head to toe: apparent age and build, face and make-up, hair (colour, length, style), anything worn on the head, then each piece of clothing with its colours and details, gloves, footwear, and anything attached to the body (for example wings). Start with "a young woman", "an old man", "a boy", or similar.
pronoun: she, he or they.

Rules: describe only the character's own body, clothes and what is worn on the body. Never name an object held in the hands or lying near the character: no props. Concrete visual detail only (colour, material, shape, pattern), no mood, personality or story."""

OBJECTS_INSTRUCTION = """These are {n} frames sampled in order from one video clip. Write exactly two labelled lines and nothing else:

person: yes if a person is visible in any frame, otherwise no.
objects: a comma-separated list of the distinct objects you can see (furniture, boxes, packaging, papers, devices, tools, anything held or handled), each ONCE, as a short plain noun phrase with its colour. Include only what is clearly visible. Do not list the person, their clothes or body parts, the walls, the floor or the light."""

# One call per chunk (the design's form): the shots listed with the pictures that show them and the words
# spoken in them. B4's first run handed Qwen the previous chunk's shot blocks and a worked example: it
# copied both, word for word, into every chunk. Neither is given now.
DRAFT_INSTRUCTION = """You are writing part of a prompt for a video-edit model. You are shown {n} pictures, frames sampled in order from one clip of a source video ({secs:.1f} seconds). In the edit, the person in the clip is replaced by a character, called "{subj}" below, who does exactly what the person does.

The clip has {k} shot{k_s}. Shot by shot, the pictures that show it and the words spoken in it:
{shot_lines}
{objects_line}{prev_line}
Write exactly three labelled parts and nothing else.

video_1: ONE sentence about the source clip as a whole: where the person is, the setting and background, the light, what the person is doing overall, and the camera's angle and framing. Don't describe the person's face, hair or clothes.

shots: {k} block{k_s}, one per shot, in order, each on its own line starting with "[Shot n]". For each shot describe only what its own pictures show: how {subj} sits or stands and leans, where {subj} looks, what each hand does, which object {subj} picks up, holds, shows or puts down, and how that changes through the shot. Shots differ: never repeat a sentence from one shot in another. {dialogue_rule}Present tense, 1 to 4 sentences per shot.

sounds: ONE sentence: the sounds through the clip, in order{sounds_hint}.

Rules:
- Describe only what the pictures show. Name an object only in a shot whose pictures show it, by what it looks like (colour, shape, size). Never quote printed text, logos or brand names.
- {Subj} never moves to the centre of the frame and never poses for the camera.
- {Poss} hands hold only what the person's hands hold.
- Don't describe {poss} appearance or costume, and don't mention pictures, frames, times, "the person" or the replacement.{extra}"""

# One call per shot (template "per shot"): each shot described from its own frames only, then one call
# over the whole chunk for video_1 and sounds.
SHOT_INSTRUCTION = """You are shown {n} pictures, frames sampled in order from one continuous shot ({secs:.1f} seconds) of a video. In the edit, the person in it is replaced by a character, called "{subj}" below, who does exactly what the person does.
{objects_line}{lines_block}
Write 1 to 5 sentences, present tense, describing only what these pictures show {subj} doing, in the order it happens from the first picture to the last (the last pictures matter as much as the first): how {subj} sits or stands and leans, where {subj} looks, what each hand does, which object {subj} picks up, holds up, shows or puts down, and where {subj} holds it (for example close to the face). {dialogue_rule}

Rules: name an object only if these pictures show it, by what it looks like (colour, shape, size); never quote printed text, logos or brand names; {subj} never moves to the centre of the frame and never poses for the camera; {poss} hands hold only what the person's hands hold; don't describe {poss} appearance or costume; don't mention pictures, frames, times, "the person" or the replacement.{extra} Output only the sentences."""

CHUNK_INSTRUCTION = """You are shown {n} pictures, frames sampled in order from one clip of a source video ({secs:.1f} seconds). Write exactly two labelled lines and nothing else:

video_1: ONE sentence about the clip as a whole: where the person is, the setting and background, the light, what the person is doing overall, and the camera's angle and framing. Don't describe the person's face, hair or clothes.
sounds: ONE sentence: the sounds through the clip, in order{sounds_hint}."""


PRONOUNS = {"she": {"subj": "she", "Subj": "She", "poss": "her", "Poss": "Her", "does": "does", "moves": "moves",
                    "poses": "poses", "stays": "stays", "handles": "handles"},
            "he": {"subj": "he", "Subj": "He", "poss": "his", "Poss": "His", "does": "does", "moves": "moves",
                   "poses": "poses", "stays": "stays", "handles": "handles"},
            "they": {"subj": "they", "Subj": "They", "poss": "their", "Poss": "Their", "does": "do", "moves": "move",
                     "poses": "pose", "stays": "stay", "handles": "handle"}}


def pronoun_of(plan_or_subject):
    """The subject's pronoun: the drafted one in the plan, else read off the subject sentence."""
    if isinstance(plan_or_subject, dict):
        p = (plan_or_subject.get("subject_pronoun") or "").strip().lower()
        if p in PRONOUNS:
            return p
        text = plan_or_subject.get("subject") or ""
    else:
        text = plan_or_subject or ""
    t = " " + text.lower() + " "
    if re.search(r"\b(woman|girl|lady|female|she|her)\b", t):
        return "she"
    if re.search(r"\b(man|boy|gentleman|male|he|his)\b", t):
        return "he"
    return "they"


def subject_name(subject):
    m = re.search(r"is the (.+?) whose motion", subject or "")
    return m.group(1).strip() if m else "character"


def fmt_frames(a, b, fps):
    return f"frames {a}-{b} ({a / fps:.2f}-{(b + 1) / fps:.2f} s)"


def neutral_events(events):
    """Omni's audio events without the voice's gender (it hears the source person; the subject may differ)."""
    t = re.sub(r"\b(?:an?\s+)?(?:[a-z]+\s+)?(?:male|female|man's|woman's|masculine|feminine)\s+voice", "a voice",
               events or "", flags=re.I)
    return re.sub(r"\b(?:a|the)\s+(?:man|woman)\s+(speaks|says|talks)", r"a voice \1", t, flags=re.I)


def _sounds_hint(events, lines):
    if events:
        return f", taken only from this description of the audio: {neutral_events(events)}"
    return " (room tone, the sounds of objects being handled" + (", and the speech" if lines else "") + ")"


def _line_where(ln, fps, origin=0):
    if "frames" not in ln:
        return f'"{ln["text"]}"'
    a, b = ln["frames"][0] - origin, ln["frames"][1] - origin
    return f'"{ln["text"]}" ({a / fps:.1f}-{(b + 1) / fps:.1f} s)'


def lines_in_shot(lines, shot_rel, k_shots, i):
    """The lines whose start falls in shot i (untimed lines: all in shot 1, Qwen places them)."""
    a, b = shot_rel
    out = []
    for ln in lines:
        if "frames" not in ln:
            if i == 0:
                out.append(ln)
        elif a <= ln["frames"][0] <= b or (i == k_shots - 1 and ln["frames"][0] > b):
            out.append(ln)
    return out


def _dialogue_rule(any_lines, per="that shot's"):
    if not any_lines:
        return ""
    return (f'Put each of {per} spoken lines at the action it goes with, written as SAYS "exact words" (copy the '
            f'words and punctuation exactly; every line once). ')


def _extra(extra):
    return ("\n- " + extra.strip()) if (extra or "").strip() else ""


def draft_instruction(*, n_frames, fps, sample_idx, shots_rel, lines, language, prev_names, events, pronoun,
                      objects=(), extra=""):
    """The one-call instruction: each shot with its pictures (1-based, of the ones shown) and its lines."""
    pr = PRONOUNS[pronoun]
    k = len(shots_rel)
    rows = []
    for i, (a, b) in enumerate(shots_rel):
        pics = [j + 1 for j, f in enumerate(sample_idx) if a <= f <= b]
        pic = (f"pictures {pics[0]}-{pics[-1]}" if len(pics) > 1 else f"picture {pics[0]}") if pics else \
            "no picture (a brief moment between pictures)"
        said = lines_in_shot(lines, (a, b), k, i)
        sp_ = ("spoken: " + ", ".join(_line_where(ln, fps) for ln in said)) if said else "nothing spoken"
        rows.append(f"Shot {i + 1} ({pic}, {a / fps:.1f}-{(b + 1) / fps:.1f} s): {sp_}.")
    if lines and not all("frames" in ln for ln in lines):
        rows.append("(The spoken lines have no times: place each where it fits the action.)")
    objs = ", ".join(objects)
    return DRAFT_INSTRUCTION.format(
        n=len(sample_idx), secs=n_frames / float(fps), subj=pr["subj"], Subj=pr["Subj"], poss=pr["poss"],
        Poss=pr["Poss"], k=k, k_s="" if k == 1 else "s", shot_lines="\n".join(rows),
        objects_line=f"Objects seen in the pictures: {objs}.\n" if objs else "",
        prev_line=(f"Names used for objects in the previous clip (reuse a name only for the same object, and only "
                   f"if you see it): {', '.join(prev_names)}.\n") if prev_names else "",
        dialogue_rule=_dialogue_rule(bool(lines)), sounds_hint=_sounds_hint(events, lines), extra=_extra(extra))


def shot_instruction(*, n, secs, lines_rel, fps, pronoun, objects=(), extra=""):
    pr = PRONOUNS[pronoun]
    objs = ", ".join(objects)
    if lines_rel:
        lb = "Words spoken in this shot, in order (times from the shot's start): " + ", ".join(
            _line_where(ln, fps) for ln in lines_rel) + ".\n"
    else:
        lb = "Nothing is spoken in this shot.\n"
    return SHOT_INSTRUCTION.format(n=n, secs=secs, subj=pr["subj"], poss=pr["poss"],
                                   objects_line=f"Objects seen in this part of the video: {objs}.\n" if objs else "",
                                   lines_block=lb, dialogue_rule=_dialogue_rule(bool(lines_rel), "the"),
                                   extra=_extra(extra).replace("\n- ", " "))


def shot_frames(shots_rel, n_frames, total=16, floor=4):
    """Per-shot sample indices (chunk-relative) for the per-shot template: each shot gets frames in
    proportion to its length, at least `floor` (or all its frames when shorter)."""
    out = []
    for a, b in shots_rel:
        n = b - a + 1
        k = min(n, max(floor, int(round(total * n / float(n_frames)))))
        out.append([a + i for i in sample_indices(n, k)])
    return out


def tidy_shots(blocks):
    """[Shot 1] never opens on a jump cut; later shots (each after a source cut) always do."""
    out = []
    for i, b in enumerate(blocks):
        m = re.match(r"(\[Shot\s*\d+\])\s*(.*)", b, re.S)
        head, body = (m.group(1), m.group(2)) if m else (f"[Shot {i + 1}]", b)
        body = re.sub(r"^(?:Jump cut\.\s*)+", "", body.strip(), flags=re.I)
        out.append(f"{head} " + ("Jump cut. " if i else "") + body)
    return out


# ---------------------------------------------------------------------------
# parsing Qwen's answers
# ---------------------------------------------------------------------------

_LABEL = re.compile(r"(?im)^[ \t]*[*#>\- \t]*(video[_ ]?1|shots|sounds|name|appearance|pronoun|person|objects)"
                    r"[ \t]*\**[ \t]*:[ \t]*\**")


def _labelled(text):
    """{label: text} for "label: ..." parts (markdown decorations tolerated)."""
    text = text or ""
    marks = [(m.start(), m.end(), re.sub(r"[ _]", "_", m.group(1).lower())) for m in _LABEL.finditer(text)]
    out = {}
    for i, (s, e, lab) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.setdefault(lab.replace("video1", "video_1"), text[e:end].strip().strip("*").strip())
    return out


def parse_subject(text):
    d = _labelled(text)
    name = re.sub(r"[\"'.]", "", (d.get("name") or "").splitlines()[0] if d.get("name") else "").strip().lower()
    app = " ".join((d.get("appearance") or "").split()).strip().strip('"')
    pron = (d.get("pronoun") or "").strip().lower().split()[0:1]
    pron = re.sub(r"[^a-z]", "", pron[0]) if pron else ""
    return {"name": name or "character", "appearance": app, "pronoun": pron if pron in PRONOUNS else None}


def subject_sentence(name, appearance):
    app = appearance.rstrip(" .")
    return (f"<Subject 1> (S1) is the {name} whose motion comes from <Video 1> and whose appearance comes from "
            f"<Picture 1>: {app}.")


def parse_objects(text):
    d = _labelled(text)
    person = (d.get("person") or "").strip().lower()
    objs = []
    for o in re.split(r"[,;\n]", d.get("objects") or ""):
        o = o.strip(" .")
        if o and o.lower() not in (x.lower() for x in objs):     # a light repetition penalty lets it loop
            objs.append(o)
    return {"person": None if not person else person.startswith("y"), "objects": objs[:40]}


def _says_repl(language):
    def repl(m):
        words = m.group(1).strip()
        return f"<Subject 1> (S1) says <d>[{language}]{words}</d>"
    return repl


# SAYS "words", or the forms Qwen slips into: she says "words", She says SAYS "words", (S1) says "words"
_SAYS = re.compile(r"(?:<Subject 1>\s*)?(?:\b(?:she|he|they)\s+)?(?:\(S1\)\s*)?(?:\bsays\s+)?\bsays\s*:?\s*"
                   r"[\"“]([^\"”]+)[\"”]", re.I)


def convert_says(text, language="English"):
    """Qwen writes speech as SAYS "words"; H3 reads <Subject 1> (S1) says <d>[Language]words</d>."""
    return _SAYS.sub(_says_repl(language), text)


# any quoted span, with the speech verb Qwen put before it ("saying", "stating", "remarks", ...)
_QUOTED = re.compile(r"(?:,?\s*(?:and\s+)?(?:(?:she|he|they)\s+)?(?:says|saying|said|states|stating|remarks|remarking|"
                     r"adds|adding|exclaims|exclaiming|asks|asking|announces|announcing|notes|noting|continues|"
                     r"replies|finally remarks|then)\s*,?\s*)?[\"“]([^\"”]+)[\"”]", re.I)
_DTAG = re.compile(r"\s*<Subject 1> \(S1\) says <d>\[[^\]]*\](.*?)</d>")


def _match_line(words, lines, used):
    """The index of the dialogue line these words are (ratio >= 0.6 on normalised tokens), or None."""
    a = [_norm_tok(w) for w in words.split() if _norm_tok(w)]
    best, bi = 0.0, None
    for i, ln in enumerate(lines):
        if i in used:
            continue
        b = [_norm_tok(w) for w in ln["text"].split() if _norm_tok(w)]
        r = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0
        if r > best:
            best, bi = r, i
    return bi if best >= 0.6 else None


def place_dialogue(block, lines, language="English"):
    """One shot block with exactly its own dialogue lines, each once, in the transcription's words: a
    tag or quote that matches one of them becomes its tag; a tag that matches none (a line Qwen put in
    the wrong shot) goes; a line Qwen dropped is added at the end. Returns (block, added count)."""
    used = set()

    def tag(i):
        used.add(i)
        return f" <Subject 1> (S1) says <d>[{language}]{lines[i]['text']}</d>"

    def from_tag(m):
        i = _match_line(m.group(1), lines, used)
        return tag(i) if i is not None else ""

    def from_quote(m):
        i = _match_line(m.group(1), lines, used)
        return tag(i) if i is not None else m.group(0)

    out = _DTAG.sub(from_tag, block)
    out = _QUOTED.sub(from_quote, out)
    missing = [i for i in range(len(lines)) if i not in used]
    if missing:
        out = out.rstrip() + "".join(tag(i) for i in missing)
    out = re.sub(r"\s+([.,;])", r"\1", re.sub(r"[ \t]{2,}", " ", out)).strip()
    return out, len(missing)


def dedupe_sentences(text, cap=8):
    """A shot block without repeated sentences: at most `cap` of them (a loop's tail cut off), plus any later
    one that carries speech."""
    out, seen = [], set()
    for snt in re.split(r"(?<=[.!?])\s+(?=[A-Z<])", (text or "").strip()):
        key = re.sub(r"[^a-z]", "", snt.lower())
        if key and key not in seen:
            seen.add(key)
            if len(out) < cap or re.search(r'["“]|<d>', snt):
                out.append(snt)
    return " ".join(out)


def parse_qwen(text, n_shots, language="English"):
    """Qwen's draft -> {video_1, shots [str], sounds, warnings}."""
    d = _labelled(text)
    warns = []
    v1 = re.split(r"\[Shot\s*\d+\]", d.get("video_1") or "")[0]     # shots run on without their label
    v1 = " ".join(v1.split()).strip().strip('"')
    if not v1:
        warns.append("Qwen gave no video_1 line")
    sounds = " ".join((d.get("sounds") or "").split()).strip().strip('"')
    if not sounds:
        warns.append("Qwen gave no sounds line")
    body = d.get("shots") or ""
    if not body.strip():
        # no label: take whatever [Shot n] blocks the answer has
        m = re.search(r"\[Shot\s*\d+\]", text or "")
        body = (text or "")[m.start():] if m else ""
        body = re.split(r"(?i)(?:^|\s)[*#]*sounds[*]*\s*:", body)[0]      # the sounds part after the shots
    parts = [p.strip() for p in re.split(r"(?=\[Shot\s*\d+\])", body) if p.strip()]
    blocks = []
    for p in parts:
        m = re.match(r"\[Shot\s*\d+\]\s*(.*)", p, re.S)
        if m:
            blocks.append(" ".join(m.group(1).split()))
        elif not blocks and p:
            blocks.append(" ".join(p.split()))
    if not blocks:
        warns.append("Qwen gave no [Shot n] blocks")
    elif len(blocks) != n_shots:
        warns.append(f"Qwen gave {len(blocks)} shot block(s) for {n_shots} shot(s)")
    blocks = [f"[Shot {i + 1}] " + convert_says(dedupe_sentences(b), language) for i, b in enumerate(blocks)]
    return {"video_1": v1.rstrip(".") + "." if v1 else "", "shots": blocks, "sounds": sounds, "warnings": warns}


# ---------------------------------------------------------------------------
# the six-section Ref2VA template ("character replace (Ref2VA)")
# ---------------------------------------------------------------------------

def _cuts_phrase(n_cuts):
    return {0: "", 1: "the jump cut, ", 2: "both jump cuts, "}.get(n_cuts, f"all {n_cuts} jump cuts, ")


def fill_ref2va(*, subject, video_1, shots, sounds, n_shots, dialogue, audio, pronoun):
    """The six sections, filled from the job's subject and Qwen's three parts (design §4.6)."""
    pr = PRONOUNS[pronoun]
    name = subject_name(subject)
    n_cuts = max(0, n_shots - 1)
    cp = _cuts_phrase(n_cuts)
    tag = "[video editing + reference generation" + (" + audio reuse]" if audio else "]")
    v1 = video_1 or "the source video, with a person in it."
    summary = (f"{tag} The target video is an edited version of <Video 1> in which the person is replaced by "
               f"<Subject 1>, the {name} styled from <Picture 1>. Everything else in <Video 1> is kept exactly: every "
               f"movement, lean, reach, hand position, head turn, mouth movement and expression timing, {cp}the camera "
               f"framing, the background, the lighting and shadows, and every object {pr['subj']} {pr['handles']}. The edit "
               f"runs continuously from the first frame to the last without deviation.")
    ret = [f"<Subject 1> (appears throughout): partially_preserved - {pr['poss']} face, hair, costume and accessories "
           f"from <Picture 1> are retained; {pr['poss']} body position, lean, pose, arm and hand actions, head "
           f"direction and how much of {pr['poss']} head is in frame, mouth movements and timing from <Video 1> are "
           f"retained at every moment.",
           f"<Video 1> (whole-video temporal structure, cuts, camera framing, background, props, lighting): "
           f"fully_preserved - the shot structure{' and every jump cut' if n_cuts else ''}, the camera framing, the "
           f"background, the lighting and shadows, and every object the person handles are preserved in every frame "
           f"without deviation. Nothing new is added to the scene."]
    if audio:
        ret.append("<Audio 1>: fully_copy - the original speech and room sound of <Video 1> are kept as they are"
                   + (f", and {pr['poss']} lip movements follow the speech." if dialogue else "."))
    style = (f"The target video matches the source footage exactly in style: the same camera, framing, lighting, "
             f"shadows, colour grading and contrast as <Video 1>. {pr['Subj']} {pr['does']} exactly what the person "
             f"does in <Video 1>, frame by frame, framed exactly the same way; {pr['subj']} never {pr['moves']} to "
             f"the centre of the frame, never {pr['poses']} for the camera, and {pr['poss']} hands only hold what the "
             f"person's hands hold.")
    blocks = list(shots) or ["[Shot 1] " + f"{pr['Subj']} {pr['does']} exactly what the person does."]
    tail = []
    if dialogue:
        tail.append(f"{pr['Poss']} lips move with every word, in time with the speech in <Audio 1>, and close between "
                    f"sentences.")
    if n_cuts:
        tail.append("Every jump cut happens at exactly the same moment as in <Video 1>, with no transition, dissolve "
                    "or morph.")
    if tail:
        blocks[-1] = blocks[-1].rstrip() + " " + " ".join(tail)
    sound = ("The original sound of <Video 1>: " + (sounds or "the room tone and the sounds of the action.")) if audio \
        else (sounds or "Quiet room tone.")
    return ("subject_definitions:\n" + subject.strip() + "\n<Video 1> is the source video for the target video edit: "
            + v1 + "\n\nsummary:\n" + summary + "\n\nretention_analysis:\n" + "\n".join(ret)
            + "\n\ndetailed_description:\n" + style + "\n" + "\n".join(blocks)
            + "\n\noverall_soundscape:\n" + sound + "\n\nnon_diegetic_music:\nN/A\n")


def parse_sections(prompt):
    """A six-section prompt -> {section: text}, in order. Raises DraftError on a missing, extra or
    out-of-order section."""
    heads = [(m.start(), m.end(), m.group(1)) for m in re.finditer(r"(?m)^([a-z_]+):[ \t]*$", prompt or "")]
    names = [h[2] for h in heads]
    if tuple(names) != SECTIONS:
        raise DraftError(f"sections {names} are not the six Ref2VA sections in order")
    out = {}
    for i, (s, e, name) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(prompt)
        out[name] = prompt[e:end].strip()
    return out


def shots_of(prompt):
    """The [Shot n] blocks of a prompt's detailed_description (the previous chunk's context)."""
    try:
        dd = parse_sections(prompt)["detailed_description"]
    except DraftError:
        dd = prompt or ""
    return [" ".join(p.split()) for p in re.split(r"(?=\[Shot\s*\d+\])", dd) if re.match(r"\[Shot\s*\d+\]", p.strip())]


# ---------------------------------------------------------------------------
# the prop check (finding d): nouns the draft names that the chunk's own frames didn't show
# ---------------------------------------------------------------------------

_DETS = {"a", "an", "the", "her", "his", "their", "its", "some", "another", "two", "three", "each", "both", "this",
         "that", "these", "those"}
_STOPS = {"and", "or", "with", "to", "up", "out", "of", "in", "into", "on", "onto", "at", "from", "by", "for", "over",
          "under", "behind", "beside", "near", "as", "while", "then", "that", "which", "who", "inside", "above",
          "below", "off", "down", "toward", "towards", "across", "through", "around", "so", "but", "is", "are",
          "she", "he", "they", "it", "again", "back", "before", "after", "between", "against", "says", "said"}
_NOT_PROPS = {"hand", "hands", "head", "face", "eye", "eyes", "mouth", "lip", "lips", "arm", "arms", "finger",
              "fingers", "body", "shoulder", "shoulders", "chin", "knee", "knees", "lap", "leg", "legs", "hair",
              "glove", "gloves", "halo", "wing", "wings", "dress", "skirt", "nose", "cheek", "cheeks", "fringe",
              "frame", "camera", "shot", "video", "edge", "top", "bottom", "side", "front", "centre", "center",
              "left", "right", "moment", "time", "way", "cut", "speech", "word", "words", "line", "view", "angle",
              "light", "lighting", "shadow", "shadows", "wall", "floor", "room", "background", "scene", "start",
              "end", "middle", "position", "pose", "direction", "gaze", "expression", "sentence", "sentences",
              "subject", "audio", "sound", "sounds", "tone", "hum", "rustle", "rustling", "click", "thud", "tap",
              "voice", "person", "girl", "woman", "man", "character", "motion", "movement", "action", "rest",
              "other", "same", "one", "first", "last", "next", "inner", "outer", "part", "half", "corner"}


def _sing(w):
    w = re.sub(r"[^a-z]", "", w.lower())
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


_AFTER = {"steady", "still", "upright", "open", "closed", "level", "flat", "tight", "up", "aside", "away", "forward",
          "remains", "rests", "stays", "moves", "turns", "holds", "lifts", "reaches", "lowers", "raises", "shows",
          "held", "without", "free", "aloft", "toward", "inside", "back"}


def named_nouns(text):
    """Head nouns of the noun phrases a text names (a determiner, up to 4 words, a stop): a heuristic
    for the prop check, not a parser."""
    text = re.sub(r"<d>.*?</d>", " ", text or "", flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    toks = re.findall(r"[A-Za-z][A-Za-z\-']*|[.,;:!?]", text)
    out = []
    for i, t in enumerate(toks):
        if t.lower() not in _DETS:
            continue
        phrase = []
        for u in toks[i + 1:i + 6]:
            lu = u.lower()
            if u in ".,;:!?" or lu in _STOPS or lu in _DETS or lu.endswith("ly") or lu in _AFTER:
                break
            phrase.append(u)
            if _sing(u) in _NOT_PROPS:          # "her left hand remains": the phrase ends at the body word
                break
        if phrase:
            head = _sing(phrase[-1])
            if head and head not in _NOT_PROPS and len(head) > 2:
                out.append((head, " ".join(phrase)))
    return out


def prop_check(shots, objects, prev_shots=()):
    """Nouns named in the shots that the chunk's own object list doesn't contain, and of those, the
    ones the previous chunk named (a likely carry-over). A warning list to check by eye, never a fix."""
    seen = set()
    for o in objects or []:
        for w in re.findall(r"[A-Za-z]+", o):
            seen.add(_sing(w))
    stems = {w[:4] for w in seen if len(w) >= 4}          # "stickies" / "sticker", "foams" / "foam"
    prev = {h for h, _ in named_nouns(" ".join(prev_shots or []))}
    unseen, carried = {}, {}
    for h, phrase in named_nouns(" ".join(shots or [])):
        if h in seen or (len(h) >= 4 and h[:4] in stems):
            continue
        unseen.setdefault(h, phrase)
        if h in prev:
            carried.setdefault(h, phrase)
    return {"unseen": sorted(unseen.items()), "carried_over": sorted(carried.items())}


def dialogue_check(shots_text, lines):
    """Each dialogue line's words should appear in the shots' <d> tags."""
    said = " ".join(re.findall(r"<d>\[[^\]]*\](.*?)</d>", shots_text or "", re.S))
    got = [_norm_tok(w) for w in said.split() if _norm_tok(w)]
    want = [_norm_tok(w) for ln in lines for w in ln["text"].split() if _norm_tok(w)]
    if not want:
        return None
    sm = difflib.SequenceMatcher(None, want, got, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return round(matched / float(len(want)), 3)


# ---------------------------------------------------------------------------
# the models
# ---------------------------------------------------------------------------

class QwenRunner:
    """AILab_QwenVL, in-process: loaded once, asked many times, then cleared. Uses the node's
    own load_model / run (no hand-rolled loader). Refuses a model that isn't on disk: the pack never
    downloads."""

    def __init__(self, cls, model, quantization, attention="auto"):
        mod = sys.modules.get(cls.__module__)
        info = {}
        try:
            mod.load_model_configs()
            info = mod.HF_ALL_MODELS.get(model) or {}
        except Exception:
            pass
        if not info:
            raise DraftError(f"QwenVL doesn't list a model {model!r}")
        try:
            local = mod.resolve_hf_model_path(info.get("repo_id", ""))
        except Exception:
            local = None
        if local is None:
            raise DraftError(f"QwenVL model {model!r} isn't on disk (models/LLM): Swap Draft never downloads; "
                             f"place it first or pick another")
        self.node = cls()
        self.model, self.quant, self.attention = model, quantization, attention
        self.path = str(local)

    def load(self):
        self.node.load_model(self.model, self.quant, self.attention, False, "auto", True)

    def ask(self, prompt, image=None, video=None, max_tokens=2048, frame_count=16):
        s = QWEN_SAMPLING
        out = self.node.run(self.model, self.quant, "", prompt, image, video, frame_count, max_tokens,
                            s["temperature"], s["top_p"], s["num_beams"], s["repetition_penalty"], s["seed"], True,
                            self.attention, False, "auto")
        return out[0] if isinstance(out, (tuple, list)) else str(out)

    def close(self):
        try:
            self.node.clear()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class OmniRunner:
    """The Omni Captioner Transcribe node, in-process (it starts its own llama.cpp process per call),
    with its default structured prompt and the full answer (transcript + audio events)."""

    def __init__(self, cls):
        self.node = cls()
        mod = sys.modules.get(cls.__module__)
        self.prompt = getattr(mod, "DEFAULT_PROMPT_WIDGET_TEXT", None) or "Describe this audio."
        missing = [p for p in (getattr(mod, "DEFAULT_CLI", None), getattr(mod, "DEFAULT_MODEL", None),
                               getattr(mod, "DEFAULT_MMPROJ", None)) if p and not os.path.exists(p)]
        if missing:
            raise DraftError(f"the Omni captioner's files are missing: {missing}")

    def transcribe(self, wav, sr):
        audio = {"waveform": torch.from_numpy(wav).unsqueeze(0), "sample_rate": sr}
        out = self.node.run(audio, prompt=self.prompt, transcript_only=False, max_tokens=448, timeout_seconds=300)
        return out[0] if isinstance(out, (tuple, list)) else str(out)

    def close(self):
        pass


class WhisperRunner:
    """faster-whisper large-v3-turbo from ComfyUI's models/stt (local files only), word timings."""

    def __init__(self, root=None):
        from faster_whisper import WhisperModel   # noqa: F401  (ImportError = Whisper missing)
        if root is None:
            import folder_paths
            root = os.path.join(folder_paths.models_dir, "stt", "faster-whisper")
        if not os.path.isdir(os.path.join(root, "models--" + WHISPER_REPO.replace("/", "--"))):
            raise DraftError(f"faster-whisper {WHISPER_REPO} isn't under {root}: Swap Draft never downloads")
        self.root = root
        self.model = None

    def load(self):
        from faster_whisper import WhisperModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(WHISPER_REPO, device=dev, compute_type="float16" if dev == "cuda" else "int8",
                                  download_root=self.root, local_files_only=True)

    def words(self, audio16k, language=None):
        segs, info = self.model.transcribe(audio16k, language=language, word_timestamps=True, beam_size=5,
                                           vad_filter=False, condition_on_previous_text=False)
        out = []
        for s in segs:
            for w in s.words or []:
                out.append({"word": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                            "p": round(float(w.probability), 3)})
        return out, getattr(info, "language", None)

    def close(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def free_comfy_models():
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as e:
        _say(f"[SeamStitch] Swap Draft: couldn't free ComfyUI's models: {e}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def parse_draft_plan(draft_plan):
    """The Planner's draft_plan: the plan path, or {"plan": path, "chunks": [...]} when the run names
    chunks (the panel's redraft)."""
    s = (draft_plan or "").strip()
    if s.startswith("{"):
        d = json.loads(s)
        return d.get("plan", ""), list(d.get("chunks") or [])
    return s, []


def select_chunks(plan, mode, named=()):
    """The chunks this run drafts. Named chunks (the panel's redraft) win over the widget; kept chunks
    never draft."""
    rendered = spl.draft_chunks(plan)
    if named:
        ids = set(named)
        return [c for c in rendered if c["id"] in ids]
    if mode == "selected":
        raise DraftError("chunks = selected drafts the chunk a redraft names: use a chunk's redraft button "
                         "(or set chunks to 'empty only' / 'all')")
    if mode == CHUNK_MODES[0]:
        return [c for c in rendered if not (c.get("prompt") or "").strip()]
    return rendered


def write_chunk_draft(plan_file, chunk_id, text, info):
    """Writes one chunk's draft under the plan lock: into the prompt only if it's empty at this
    moment, otherwise into the draft field. Returns where it went."""
    where = {}

    def upd(p):
        c = sp.find_chunk(p, chunk_id)
        if not (c.get("prompt") or "").strip():
            c["prompt"], c["prompt_state"] = text, "draft"
            where["into"] = "prompt"
        else:
            c["draft"] = text
            where["into"] = "draft"
        c["drafted"] = dict(info, at=sp.now(), into=where["into"])
    plan = sp.update_plan(plan_file, upd)
    return where["into"], plan["rev"]


def write_subject(plan_file, sentence, pronoun):
    where = {}

    def upd(p):
        if not (p.get("subject") or "").strip():
            p["subject"] = sentence
            where["into"] = "subject"
        else:
            p["subject_draft"] = sentence
            where["into"] = "subject_draft"
        if pronoun and not p.get("subject_pronoun"):
            p["subject_pronoun"] = pronoun
    sp.update_plan(plan_file, upd)
    return where["into"]


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_json(path, obj):
    _write(path, json.dumps(obj, indent=1, ensure_ascii=False))


def run_draft(plan_file, *, sheet=None, mode=CHUNK_MODES[0], named=(), qwen_model=DEFAULT_QWEN,
              quantization=QUANTIZATIONS[0], frames_per_chunk=DEFAULT_FRAMES, transcribe=True, word_timings=True,
              extra_instructions="", max_tokens=2048, template=None, qwen_cls=None, omni_cls=None, whisper=None,
              free_models=True,
              sample_vram=True):
    """The whole draft run (one queue item). qwen_cls / omni_cls / whisper: injected for tests;
    None = locate the installed nodes (False = treat as missing). Returns the report dict."""
    t_run = time.time()
    template = template or DEFAULT_TEMPLATE
    plan = sp.load_plan(plan_file)
    jd = os.path.dirname(os.path.abspath(plan_file))
    src = plan["source"]
    path = src["path"]
    fps = int(round(float(src["fps"])))
    if not os.path.isfile(path):
        raise DraftError(f"the source video is missing: {path}")
    chunks = select_chunks(plan, mode, named)
    warnings = []
    report = {"job": plan["job"], "mode": mode, "named": list(named), "chunks": {}, "warnings": warnings,
              "stages": {}, "vram": {"free_gb": []}, "qwen_model": qwen_model, "quantization": quantization,
              "frames_per_chunk": int(frames_per_chunk), "sampling": dict(QWEN_SAMPLING), "template": template}
    if named:
        kept = [x for x in named if sp.is_kept(next((c for c in plan["chunks"] if c["id"] == x), None))]
        if kept:
            warnings.append(f"kept chunk(s) {', '.join(kept)} are never drafted")
    need_subject = not (plan.get("subject") or "").strip() or mode == CHUNK_MODES[1]
    if not chunks and not need_subject:
        report["text"] = f"nothing to draft ({mode}): every rendered chunk has a prompt"
        _say(f"[SeamStitch] Swap Draft: {plan['job']}: {report['text']}")
        return report

    # the models: QwenVL refuses the run when missing; Omni and Whisper degrade
    qcls = locate_qwen() if qwen_cls is None else (qwen_cls or None)
    if qcls is None:
        raise DraftError("SeamStitch Swap Draft Prompts needs ComfyUI-QwenVL (1038lab) in custom_nodes, with the "
                         f"model {qwen_model!r}: install it (Manager: 'ComfyUI-QwenVL'), restart ComfyUI")
    qwen = qcls if hasattr(qcls, "ask") else QwenRunner(qcls, qwen_model, quantization)
    omni = None
    if transcribe:
        try:
            ocls = locate_omni() if omni_cls is None else (omni_cls or None)
            if ocls is None:
                warnings.append("Omni Captioner Transcribe isn't installed: dialogue words from Whisper alone, "
                                "no audio events")
            else:
                omni = OmniRunner(ocls) if not hasattr(ocls, "transcribe") else ocls
        except Exception as e:
            warnings.append(f"Omni unavailable ({e}): dialogue words from Whisper alone, no audio events")
    wh = None
    if word_timings:
        try:
            wh = WhisperRunner() if whisper is None else (whisper or None)
            if wh is None:
                warnings.append("faster-whisper isn't available: dialogue drafted without timings")
        except Exception as e:
            warnings.append(f"faster-whisper unavailable ({e}): dialogue drafted without timings")
            wh = None
    if not transcribe and not word_timings:
        warnings.append("transcribe and word_timings are off: no dialogue drafted")

    sampler = VramSampler() if sample_vram else None
    if sampler:
        sampler.start()
    stages = report["stages"]

    def stage(name):
        if sampler:
            sampler.stage = name
        free, total = free_vram_gb()
        report["vram"]["free_gb"].append({"before": name, "free": free, "total": total})
        if free is not None:
            _say(f"[SeamStitch] Swap Draft: free VRAM {free:.1f} / {total:.1f} GB before {name}")

    info = {}
    try:
        t = time.time()
        stage("free ComfyUI's models")
        if free_models:
            free_comfy_models()
        stages["free_s"] = round(time.time() - t, 2)

        # per chunk geometry + audio
        for c in chunks:
            r0, r1 = c["render"]
            n = r1 - r0 + 1
            shots = chunk_shots(plan, r0, r1)
            info[c["id"]] = {"render": [r0, r1], "n": n, "shots": shots,
                             "shots_rel": [(a - r0, b - r0) for a, b in shots], "dir": os.path.join(jd, "drafts", c["id"]),
                             "warnings": []}
            os.makedirs(info[c["id"]]["dir"], exist_ok=True)
        audio_ok = bool(src.get("audio"))
        wavs = {}
        if audio_ok and (omni or wh):
            for c in chunks:
                wavs[c["id"]] = chunk_audio(path, fps, *info[c["id"]]["render"])

        # 2. Omni per chunk
        stages["omni_s"] = {}
        omni_parsed = {}
        if omni is not None and audio_ok:
            stage("Omni")
            for c in chunks:
                wav, sr = wavs.get(c["id"], (None, None))
                if wav is None:
                    continue
                t = time.time()
                raws = []
                for attempt in range(2):          # one retry when it answers in free prose
                    try:
                        raws.append(omni.transcribe(wav, sr))
                    except Exception as e:
                        raws.append("")
                        info[c["id"]]["warnings"].append(f"Omni failed: {e}")
                    po = parse_omni(raws[-1])
                    if not raws[-1] or po["structured"]:
                        break
                raw = raws[-1]
                stages["omni_s"][c["id"]] = round(time.time() - t, 2)
                if not raw:
                    info[c["id"]]["warnings"].append("Omni returned nothing (timed out or failed): words from Whisper")
                    po = None
                elif not po["structured"]:
                    # its quoted fragments drop words: Whisper's words, and no audio events
                    info[c["id"]]["warnings"].append("Omni answered twice without its transcript / audio_events "
                                                     "sections: words from Whisper, no audio events")
                    po = None
                raw = "\n\n---- retry ----\n".join(raws)
                omni_parsed[c["id"]] = po
                po = po or parse_omni("")
                _write(os.path.join(info[c["id"]]["dir"], "omni.txt"),
                       raw + "\n\n---- parsed ----\n" + json.dumps(po, indent=1, ensure_ascii=False))
                _say(f"[SeamStitch] Swap Draft: {c['id']} Omni {stages['omni_s'][c['id']]} s: {po['words'][:120]!r}")
            omni.close()

        # 3. Whisper, loaded once
        stages["whisper_s"] = {}
        whisper_words = {}
        if wh is not None and audio_ok:
            stage("Whisper")
            t = time.time()
            try:
                wh.load()
                stages["whisper_load_s"] = round(time.time() - t, 2)
                for c in chunks:
                    wav, sr = wavs.get(c["id"], (None, None))
                    if wav is None:
                        continue
                    t = time.time()
                    po = omni_parsed.get(c["id"])
                    lang = None
                    if po and po.get("language"):
                        lang = next((k for k, v in LANGS.items() if v.lower() == po["language"].lower()), None)
                    words, wlang = wh.words(to_mono_16k(wav, sr), lang)
                    whisper_words[c["id"]] = (words, wlang)
                    stages["whisper_s"][c["id"]] = round(time.time() - t, 2)
            except Exception as e:
                warnings.append(f"faster-whisper failed ({e}): dialogue drafted without timings")
            finally:
                wh.close()

        # dialogue lines per chunk
        for c in chunks:
            ci = info[c["id"]]
            po = omni_parsed.get(c["id"])
            words, wlang = whisper_words.get(c["id"], (None, None))
            lang = (po or {}).get("language") or LANGS.get(wlang or "", None) or "English"
            if po is not None and not po["speech"]:
                lines, aligned = [], []
            elif po is not None and po["words"]:
                aligned = align_words(po["words"], words) if words else []
                lines = dialogue_lines(aligned, fps, ci["n"]) if words else lines_from_text(po["words"])
                if not words and (wh is not None or word_timings):
                    ci["warnings"].append("no word timings: the lines go to Qwen without frames")
            elif words:
                good = [w for w in words if w.get("p", 1) >= 0.4]
                aligned = [dict(w, matched=True) for w in good]
                lines = dialogue_lines(aligned, fps, ci["n"])
                if omni is not None or transcribe:
                    ci["warnings"].append("dialogue from Whisper alone (no Omni words)")
            else:
                lines, aligned = [], []
            ci.update(lines=lines, language=lang, events=(po or {}).get("events") or "",
                      notes=(po or {}).get("notes") or [])
            if words is not None or po is not None:
                _write_json(os.path.join(ci["dir"], "whisper.json"),
                            {"language": wlang, "words": words or [], "aligned": aligned, "lines": lines,
                             "fps": fps, "render": ci["render"]})

        # 4. Qwen, loaded once: the subject, then each chunk
        stage("Qwen")
        t = time.time()
        qwen.load()
        stages["qwen_load_s"] = round(time.time() - t, 2)
        subject = (plan.get("subject") or "").strip()
        if need_subject:
            if sheet is None:
                if not subject:
                    warnings.append("no sheet wired: the subject wasn't drafted (wire the render group's Load Image "
                                    "into sheet, or write the subject)")
            else:
                t = time.time()
                raw = qwen.ask(SUBJECT_INSTRUCTION, image=sheet[:1], max_tokens=512)
                stages["subject_s"] = round(time.time() - t, 2)
                ps = parse_subject(raw)
                sentence = subject_sentence(ps["name"], ps["appearance"]) if ps["appearance"] else ""
                sdir = os.path.join(jd, "drafts", "subject")
                _write(os.path.join(sdir, "qwen.txt"), SUBJECT_INSTRUCTION + "\n\n---- Qwen ----\n" + raw)
                if sentence:
                    _write(os.path.join(sdir, "draft.txt"), sentence + "\n")
                    into = write_subject(plan_file, sentence, ps["pronoun"])
                    report["subject"] = {"into": into, "text": sentence, "pronoun": ps["pronoun"]}
                    if into == "subject":
                        subject = sentence
                else:
                    warnings.append("Qwen gave no appearance for the subject")
        plan = sp.load_plan(plan_file)
        pronoun = pronoun_of(plan)
        if not subject:
            subject = ("<Subject 1> (S1) is the character whose motion comes from <Video 1> and whose appearance "
                       "comes from <Picture 1>.")
        k_frames = max(1, int(frames_per_chunk))
        per_shot = template == TEMPLATES[1]
        order = [c["id"] for c in plan["chunks"]]
        drafted_shots, drafted_objects = {}, {}
        stages["qwen_s"] = {}
        for c in chunks:
            ci = info[c["id"]]
            r0, r1 = ci["render"]
            idx = sample_indices(ci["n"], k_frames)
            video = decode_sampled(path, fps, r0, idx)
            # the previous chunk: its shot blocks only for the carry-over check, its object names for Qwen
            k = order.index(c["id"])
            prev = plan["chunks"][k - 1] if k > 0 else None
            prev_shots, prev_names = [], []
            if prev is not None and not sp.is_kept(prev):
                prev_shots = drafted_shots.get(prev["id"]) or shots_of(prev.get("prompt") or prev.get("draft") or "")
                prev_names = drafted_objects.get(prev["id"]) or []
            t = time.time()
            oraw = qwen.ask(OBJECTS_INSTRUCTION.format(n=len(idx)), video=video, max_tokens=384, frame_count=len(idx))
            t_obj = round(time.time() - t, 2)
            objs = parse_objects(oraw)
            log = ["==== objects: instruction ====", OBJECTS_INSTRUCTION.format(n=len(idx)), "", "==== objects: Qwen ====",
                   oraw, ""]
            t = time.time()
            if not per_shot:
                instr = draft_instruction(n_frames=ci["n"], fps=fps, sample_idx=idx, shots_rel=ci["shots_rel"],
                                          lines=ci["lines"], language=ci["language"], prev_names=prev_names,
                                          events=ci["events"], pronoun=pronoun, objects=objs["objects"],
                                          extra=extra_instructions)
                draw = qwen.ask(instr, video=video, max_tokens=int(max_tokens), frame_count=len(idx))
                log += ["==== draft: instruction ====", instr, "", "==== draft: Qwen ====", draw, ""]
                parts = parse_qwen(draw, len(ci["shots"]), ci["language"])
            else:
                # each shot from its own frames, then video_1 and sounds over the chunk
                blocks = []
                k_sh = len(ci["shots_rel"])
                for i, (sh, fr_idx) in enumerate(zip(ci["shots_rel"], shot_frames(ci["shots_rel"], ci["n"], k_frames))):
                    said = lines_in_shot(ci["lines"], sh, k_sh, i)
                    rel = [dict(ln, frames=[ln["frames"][0] - sh[0], ln["frames"][1] - sh[0]]) if "frames" in ln else ln
                           for ln in said]
                    instr = shot_instruction(n=len(fr_idx), secs=(sh[1] - sh[0] + 1) / float(fps), lines_rel=rel,
                                             fps=fps, pronoun=pronoun, objects=objs["objects"], extra=extra_instructions)
                    sv = decode_sampled(path, fps, r0, fr_idx)
                    sraw = qwen.ask(instr, video=sv, max_tokens=512, frame_count=len(fr_idx))
                    log += [f"==== shot {i + 1}: instruction ====", instr, "", f"==== shot {i + 1}: Qwen ====", sraw, ""]
                    body = " ".join(re.sub(r"^\s*(?:\[Shot\s*\d+\]|shot\s*\d+\s*:)\s*", "", sraw, flags=re.I).split())
                    blocks.append(f"[Shot {i + 1}] {body}")
                cinstr = CHUNK_INSTRUCTION.format(n=len(idx), secs=ci["n"] / float(fps),
                                                  sounds_hint=_sounds_hint(ci["events"], ci["lines"]))
                craw = qwen.ask(cinstr, video=video, max_tokens=384, frame_count=len(idx))
                log += ["==== chunk: instruction ====", cinstr, "", "==== chunk: Qwen ====", craw, ""]
                parts = parse_qwen(craw + "\nshots:\n" + "\n".join(blocks), len(ci["shots"]), ci["language"])
            t_draft = round(time.time() - t, 2)
            stages["qwen_s"][c["id"]] = {"objects": t_obj, "draft": t_draft}
            _write(os.path.join(ci["dir"], "qwen.txt"), "\n".join(log))
            parts["shots"] = tidy_shots(parts["shots"])
            # each shot carries exactly its own lines, in the transcription's words
            k_sh, added = len(ci["shots_rel"]), 0
            for i in range(min(k_sh, len(parts["shots"]))):
                said = lines_in_shot(ci["lines"], ci["shots_rel"][i], k_sh, i)
                parts["shots"][i], n_add = place_dialogue(parts["shots"][i], said, ci["language"])
                added += n_add
            if added:
                parts["warnings"].append(f"{added} spoken line(s) Qwen left out were added at the end of their shot")
            if not ci["events"]:
                # Qwen can't hear: without Omni's events it invents sounds (B4 workshop: "muffled voices")
                parts["sounds"] = ("the room tone and the sounds of objects being handled"
                                   + (f", and {PRONOUNS[pronoun]['poss']} speech." if ci["lines"] else "."))
            elif not parts["sounds"]:
                parts["sounds"] = neutral_events(ci["events"])
                parts["warnings"] = [w for w in parts["warnings"] if "sounds" not in w] + [
                    "Qwen gave no sounds line: Omni's audio events used"]
            drafted_objects[c["id"]] = objs["objects"]
            ci["warnings"] += parts["warnings"]
            if objs["person"] is False:
                ci["warnings"].append("Qwen saw no person in this chunk")
            text = fill_ref2va(subject=subject, video_1=parts["video_1"], shots=parts["shots"], sounds=parts["sounds"],
                               n_shots=len(ci["shots"]), dialogue=bool(ci["lines"]), audio=audio_ok, pronoun=pronoun)
            parse_sections(text)            # the template must parse into the six sections
            drafted_shots[c["id"]] = parts["shots"]
            props = prop_check(parts["shots"], objs["objects"], prev_shots)
            dcov = dialogue_check(" ".join(parts["shots"]), ci["lines"])
            if props["unseen"]:
                ci["warnings"].append("named but not in the frames' object list (check by eye): "
                                      + ", ".join(p for _, p in props["unseen"]))
            if props["carried_over"]:
                ci["warnings"].append("also named in the previous chunk (carried over?): "
                                      + ", ".join(p for _, p in props["carried_over"]))
            if dcov is not None and dcov < 0.9:
                ci["warnings"].append(f"only {dcov:.0%} of the dialogue's words made it into the <d> lines")
            _write(os.path.join(ci["dir"], "draft.txt"), text)
            check = {"objects": objs, "props": props, "dialogue_coverage": dcov, "warnings": ci["warnings"],
                     "sample_frames": [r0 + i for i in idx], "shots": ci["shots"], "lines": ci["lines"]}
            _write_json(os.path.join(ci["dir"], "check.json"), check)
            into, rev = write_chunk_draft(plan_file, c["id"], text,
                                          {"warnings": ci["warnings"], "files": f"drafts/{c['id']}/"})
            report["chunks"][c["id"]] = {"into": into, "render": ci["render"], "shots": len(ci["shots"]),
                                         "lines": len(ci["lines"]), "warnings": ci["warnings"],
                                         "dialogue_coverage": dcov, "unseen": [p for _, p in props["unseen"]],
                                         "carried_over": [p for _, p in props["carried_over"]], "plan_rev": rev,
                                         "chars": len(text)}
            _say(f"[SeamStitch] Swap Draft: {c['id']} {r0}-{r1}: objects {t_obj} s, draft {t_draft} s -> {into}"
                 + (f" · {len(ci['warnings'])} warning(s)" if ci["warnings"] else ""))
        stage("Qwen unload")
    finally:
        t = time.time()
        try:
            qwen.close()
        except Exception:
            pass
        stages["qwen_unload_s"] = round(time.time() - t, 2)
        free, total = free_vram_gb()
        report["vram"]["free_gb"].append({"before": "end", "free": free, "total": total})
        if sampler:
            report["vram"].update(sampler.stop())
    report["total_s"] = round(time.time() - t_run, 2)
    lines = [f"job {plan['job']}: drafted {len(report['chunks'])} chunk(s) in {report['total_s']} s ({mode})"]
    if report.get("subject"):
        lines.append(f"subject -> {report['subject']['into']}")
    for cid, r in report["chunks"].items():
        lines.append(f" {cid} {r['render'][0]}-{r['render'][1]}: -> {r['into']}, {r['shots']} shot(s), "
                     f"{r['lines']} line(s)" + (f", {len(r['warnings'])} warning(s)" if r["warnings"] else ""))
    lines += [f" ⚠ {w}" for w in warnings]
    if report["vram"].get("device_peak_gb") is not None:
        lines.append(f"VRAM peak (device) {report['vram']['device_peak_gb']} GB")
    report["text"] = "\n".join(lines)
    _write_json(os.path.join(jd, "drafts", f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"), report)
    return report


class SeamStitchSwapDraft:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "draft_plan": ("STRING", {"forceInput": True, "tooltip":
                    "The Planner's draft_plan output: carries the plan only in a draft run."}),
                "chunks": (CHUNK_MODES, {"default": CHUNK_MODES[0], "tooltip":
                    "empty only: chunks with no prompt. all: every rendered chunk (a chunk with a prompt gets the "
                    "draft in its draft field, never over the prompt). selected: the chunk a redraft names (a "
                    "redraft always drafts its chunk, whatever this says)."}),
                "qwen_model": (qwen_models(), {"default": DEFAULT_QWEN, "tooltip": "A QwenVL model on disk."}),
                "quantization": (QUANTIZATIONS, {"default": QUANTIZATIONS[0]}),
                "frames_per_chunk": ("INT", {"default": DEFAULT_FRAMES, "min": 2, "max": 48, "tooltip":
                    "Frames Qwen sees, spread over the chunk's render range (per shot: shared out by shot length, "
                    "at least 4 each). The QwenVL node itself uses 16."}),
                "transcribe": ("BOOLEAN", {"default": True, "tooltip":
                    "Omni Captioner Transcribe on each chunk's audio: the words and the audio events."}),
                "word_timings": ("BOOLEAN", {"default": True, "tooltip":
                    "faster-whisper large-v3-turbo: word timings, so each line lands in its shot and moment."}),
                "template": (TEMPLATES, {"default": DEFAULT_TEMPLATE, "tooltip":
                    "per shot: each shot from its own frames, then video_1 and sounds over the chunk (the default). "
                    "The other: one call for the whole chunk."}),
                "extra_instructions": ("STRING", {"default": "", "multiline": True, "tooltip":
                    "Added to Qwen's rules for every chunk."}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 4096}),
            },
            "optional": {
                "sheet": ("IMAGE", {"tooltip": "The character sheet (the render group's Load Image): the subject is "
                                               "drafted from it once per job."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "draft"
    OUTPUT_NODE = True
    CATEGORY = "SeamStitch/Swap"
    DESCRIPTION = ("Drafts the subject from the sheet and every chunk's prompt from its frames and dialogue "
                   "(QwenVL + Omni + faster-whisper, one model at a time). A draft fills an empty prompt; "
                   "otherwise it waits in the chunk's draft field.")

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")          # a draft run always runs (the plan path alone never changes)

    def draft(self, draft_plan, chunks=CHUNK_MODES[0], qwen_model=DEFAULT_QWEN, quantization=QUANTIZATIONS[0],
              frames_per_chunk=DEFAULT_FRAMES, transcribe=True, word_timings=True, template=DEFAULT_TEMPLATE,
              extra_instructions="",
              max_tokens=2048, sheet=None):
        plan_file, named = parse_draft_plan(draft_plan)
        if not plan_file:
            raise DraftError("draft_plan is empty: queue a draft run from the Planner")
        rep = run_draft(plan_file, sheet=sheet, mode=chunks, named=named, qwen_model=qwen_model,
                        quantization=quantization, frames_per_chunk=frames_per_chunk, transcribe=transcribe,
                        word_timings=word_timings, extra_instructions=extra_instructions, max_tokens=max_tokens,
                        template=template)
        _say("[SeamStitch] Swap Draft: " + rep["text"].replace("\n", "\n[SeamStitch] Swap Draft: "))
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("seamstitch_swap_plan", {"job": rep["job"]})
        except Exception:
            pass
        return {"ui": {"text": [rep["text"]]}, "result": (rep["text"],)}
