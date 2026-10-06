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
    from . import swap_scores as ss
    from . import timeline as tl
except ImportError:  # imported as a top-level module (tests, tools)
    import swap_plan as sp
    import swap_planner as spl
    import swap_scores as ss
    import timeline as tl

_say = spl._say

QWEN_CLASS = "AILab_QwenVL"
OMNI_CLASS = "OmniCaptionerTranscribe"
DEFAULT_QWEN = "Qwen3-VL-8B-Unredacted-MAX (Captioner)"
QUANTIZATIONS = ["None (FP16)", "8-bit (Balanced)", "4-bit (VRAM-friendly)"]
CHUNK_MODES = ["empty only", "all (into the draft field)", "selected"]
TEMPLATES = ["character replace (Ref2VA)", "character replace (Ref2VA, per shot)",
             "character replace (Ref2VA, timeline)"]
MOMENT_S = 0.5                  # timeline: one picture per this many seconds of each shot
# timeline (B4 render rounds 1-2, 400-608 against X10 at X10's settings, 2 seeds): pose IoU 0.733 / 0.729 against
# the hand prompt's 0.645 / 0.727 through the same nodes, mouth sync 0.67 / 0.69 against 0.26 / 0.14 (second
# half 0.59 / 0.61 against -0.34 / -0.16). The per-shot draft (T-DRAFT) tied on pose and lost lip sync (-0.07).
DEFAULT_TEMPLATE = TEMPLATES[2]
# B4 workshop C2: parts of a long shot (4 s) and a 6-frame scene call made the shots wordier and more invented,
# and video_1 no better, than whole shots (C1). Both kept as options, off by default.
SEGMENT_S = 0.0                 # per shot: > 0 describes a longer shot in parts of at most this many seconds
SCENE_FRAMES = 0                # per shot: > 0 = video_1 and sounds from this many frames (0: the chunk's frames)
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
        sys.modules[spec.name] = mod
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

QWEN_RAM_GB = 24.0              # Qwen3-VL-8B FP16 staged through RAM (16 GB of weights) plus headroom


def ram_gb():
    """Available RAM and free swap (GB), or None without psutil."""
    try:
        import psutil
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        return {"available": round(vm.available / 2 ** 30, 1), "total": round(vm.total / 2 ** 30, 1),
                "swap_free": round(sw.free / 2 ** 30, 1)}
    except Exception:
        return None


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


_TIMESTAMP = re.compile(r"[\[(]?\s*\d{1,2}:\d{2}(?:[.:]\d{1,3})?\s*(?:-|–|to)\s*\d{1,2}:\d{2}(?:[.:]\d{1,3})?\s*[\])]?")
_SPEAKER = re.compile(r"(?:^|(?<=[\s.!?\"”]))-?\s*((?:[Mm]ale|[Ff]emale)\s+speaker|Man|Woman|Boy|Girl|Male|Female|Child|"
                      r"Narrator|Singer|Speaker\s*\d+|Person\s*\d+|Voice\s*\d*)\s*:\s*")
INTERJECTIONS = {"oh", "ah", "aah", "ahh", "yeah", "yea", "yes", "woo", "whoo", "wooo", "wow", "hey", "ha", "haha", "hahaha",
                 "ho", "hoo", "yay", "mm", "mmm", "hmm", "ooh", "oo", "uh", "um", "huh", "whoa", "yo", "la", "na", "da",
                 "ya", "eh", "oi", "hah", "hee", "aw", "aww", "boo", "yah", "ay", "ayy", "ole", "olé"}


def is_looping(text, min_tokens=12, unique_ratio=0.3):
    """A transcript that is mostly one phrase over and over (Omni's failure on drums, cheering and
    instrumentals: "Oh, yeah!" x 100, "Pika..." x 100)."""
    toks = [_norm_tok(w) for w in (text or "").split() if _norm_tok(w)]
    return len(toks) >= min_tokens and len(set(toks)) / float(len(toks)) < unique_ratio


def interjections_only(text):
    """Only cheers and exclamations ("Woo!", "Oh, yeah", "Hey hey"): shouting, not lines anyone says."""
    toks = [_norm_tok(w) for w in (text or "").split() if _norm_tok(w)]
    return bool(toks) and all(t in INTERJECTIONS for t in toks)


def drop_hum_lines(lines, min_words=3):
    """Lines that are only a hum or an exclamation said over and over ("Mmm mmm mmm ..." x 14, B5b: a man humming
    while he mimes licking a packet) aren't words she says: they go into the soundscape. A short "Yeah." or "Wow!"
    stays a line. -> (the lines kept, the hums' text)."""
    hums = [ln for ln in lines if len(ln["text"].split()) >= min_words and interjections_only(ln["text"])]
    return [ln for ln in lines if ln not in hums], " ".join(ln["text"] for ln in hums)


def parse_omni(text):
    """Omni's answer to the node's default structured prompt -> {language, words (str), speakers (one label or
    None per word), notes, events, speech, structured, sung, looping}. Cleans what Omni wraps around the
    words: markdown, timestamps, speaker labels (kept per word), "(sung)" and other stage notes. The
    label-less form gives the quoted words and no events."""
    t = " ".join((text or "").replace("**", " ").split())
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
        elif not _TIMESTAMP.fullmatch(f"[{b}]"):
            notes.append(b.strip())
    tr = _TIMESTAMP.sub(" ", tr)
    notes += [x.strip() for x in re.findall(r"\(([^)]*)\)", tr)]        # "(voice trails off)", "(sung)": notes
    sung = any(re.search(r"\bsung|singing|sings\b", n_, re.I) for n_ in notes) or bool(
        re.search(r"\b(sings|singing|song|lyrics|chant(s|ing)?)\b", ev, re.I)
        and not re.search(r"\b(speaks|speaking|says|talks|talking|conversation)\b", ev, re.I))
    words, speakers, flags = [], [], []
    for text, is_sung in _sung_marked(tr):
        w_, s_ = _words_speakers(text)
        words += w_
        speakers += s_
        flags += [is_sung] * len(w_)
    if sung and not any(flags):
        flags = [True] * len(words)           # sung, but no note says where: the whole transcript (as before)
    wtext = " ".join(words)
    speech = bool(wtext) and wtext.strip(" .").lower() not in ("none", "no speech", "n/a", "silence")
    spoken = [(w, sp_) for w, sp_, f in zip(words, speakers, flags) if not f]
    return {"language": lang, "words": wtext if speech else "", "speakers": speakers if speech else [],
            "notes": notes, "events": ev, "speech": speech, "structured": structured, "sung": sung and speech,
            "looping": is_looping(wtext),
            # the parts: "put the tray in. [sings] La la la..." is a line, then a song (a spoken line lost in B5a)
            "spoken": " ".join(w for w, _ in spoken) if speech else "",
            "spoken_speakers": [sp_ for _, sp_ in spoken] if speech else [],
            "sung_words": " ".join(w for w, f in zip(words, flags) if f) if speech else ""}


_SUNG_NOTE = re.compile(r"(?i)\bsung|singing|sings\b")
_SPOKEN_NOTE = re.compile(r"(?i)\b(speaks|speaking|says|spoken|talks|talking)\b")
_NOTE = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def _sung_marked(tr):
    """The transcript cut at its stage notes into [(text, sung)], notes removed. A sung note ("[sings]",
    "(singing)") marks the text after it, up to a spoken note; a sung note with no text after it before the next
    note ("... (sung)") marks the text before it instead."""
    pieces, last, sung_next = [], 0, False
    notes = list(_NOTE.finditer(tr))
    for i, m in enumerate(notes):
        pieces.append([tr[last:m.start()], sung_next])
        note = m.group(0)[1:-1]
        last = m.end()
        if _SUNG_NOTE.search(note):
            after = tr[last:notes[i + 1].start() if i + 1 < len(notes) else len(tr)]
            if after.strip(" .,;:-*\"“”"):
                sung_next = True
            else:
                for pc in reversed(pieces):
                    if pc[0].strip(" .,;:-*\"“”"):
                        pc[1] = True
                        break
        elif _SPOKEN_NOTE.search(note):
            sung_next = False
    pieces.append([tr[last:], sung_next])
    return [(t, f) for t, f in pieces if t.strip()]


def _words_speakers(text):
    """Words, and the speaker label for each ("Man: What?" -> ["What?"], ["Man"])."""
    words, speakers = [], []
    parts = _SPEAKER.split(" " + text)
    label = None
    for k, part in enumerate(parts):
        if k % 2 == 1:
            label = part.strip()
            continue
        for w in part.replace('"', " ").replace("“", " ").replace("”", " ").split():
            w = w.strip("-*")
            if w:
                words.append(w)
                speakers.append(label)
    return words, speakers


def _norm_tok(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def unstretch(words, longest=0.45, keep=0.4):
    """A word over `longest` s is Whisper stretching it over the silence before it: keep its last `keep` s."""
    return [dict(w, start=max(w["start"], w["end"] - keep)) if w["end"] - w["start"] > longest else w
            for w in words or []]


def align_words(omni_words, whisper_words, speakers=None):
    """Omni's tokens (as written, punctuation kept) with times from Whisper's words, by sequence
    alignment on the normalised tokens: matched tokens take their Whisper word's times, a replaced
    run spreads its Whisper span over Omni's tokens, an Omni-only token is interpolated between its
    neighbours. Returns [{"word", "start", "end", "matched"}]."""
    toks = (omni_words or "").split()
    if not toks:
        return []
    ww = [w for w in unstretch(whisper_words) if _norm_tok(w["word"])]
    a = [_norm_tok(t) for t in toks]
    b = [_norm_tok(w["word"]) for w in ww]
    out = [{"word": t, "start": None, "end": None, "matched": False,
            "speaker": (speakers[i] if speakers and i < len(speakers) else None)} for i, t in enumerate(toks)]
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


def dialogue_lines(aligned, fps, n_frames, gap_s=0.8, comma_gap_s=0.25):
    """Group timed words into spoken lines: a break after . ? ! …, after a comma with a pause, or at
    any longer pause. Frames are chunk-relative (0..n_frames-1) at the source rate."""
    lines, cur = [], []

    def close():
        if cur:
            s = cur[0]["start"]
            e = cur[-1]["end"]
            text = " ".join(w["word"] for w in cur)
            d = {"text": text}
            if cur[0].get("speaker"):
                d["speaker"] = cur[0]["speaker"]
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
        if word.endswith((".", "?", "!", "…")) or pause >= gap_s or (word.endswith((",", ";", ":")) and pause >= comma_gap_s) \
                or nxt.get("speaker") != w.get("speaker"):
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

name: a short descriptive name for the character, 2-4 words, lower case (for example "bearded sailor" or "silver robot knight").
appearance: ONE sentence, a comma-separated list from head to toe that names every visible detail, in this order: apparent age and build; face and make-up (skin, cheeks, nose, lips, eyes); hair (colour, length, style, fringe); anything on or above the head, with its colours; anything at the neck; the top or dress (colour, neckline, sleeves, trims, buttons, bows, belts); the skirt or trousers (layers, colours); gloves or arm pieces (length, colours, cuffs); legwear; footwear; anything attached to the body (for example wings or a tail). Start with "a young woman", "an old man", "a boy", or similar.
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
SHOT_INSTRUCTION = """You are shown {n} pictures, frames sampled in order from one continuous shot ({secs:.1f} seconds) of a video. In the edit, the person in it is replaced by a character, called "{subj}" below, who does exactly what the person does.{part}
{objects_line}{lines_block}
Write 1 to 5 sentences, present tense, describing only what these pictures show {subj} doing, in the order it happens from the first picture to the last (the last pictures matter as much as the first): how {subj} sits or stands and leans, where {subj} looks, what each hand does, which object {subj} picks up, holds up, shows or puts down, and where {subj} holds it (for example close to the face). {dialogue_rule}

Rules: name an object only if these pictures show it, by what it looks like (colour, shape, size); never quote printed text, logos or brand names; {subj} never moves to the centre of the frame and never poses for the camera; {poss} hands hold only what the person's hands hold; don't describe {poss} appearance or costume; don't mention pictures, frames, times, "the person" or the replacement.{extra} Output only the sentences."""

CHUNK_INSTRUCTION = """You are shown {n} pictures, frames sampled in order from one clip of a source video ({secs:.1f} seconds). Write exactly two labelled lines and nothing else:

video_1: ONE sentence about the clip as a whole: where the person is and what they sit or stand on, the background, the light and where it comes from, what the person is doing overall, and the camera: whether it looks down on the person from above, is level with them or looks up at them, whether it moves, and where the person's head sits in the frame (for example near the top edge, or in the centre). Don't describe the person's face, hair or clothes.
sounds: ONE sentence: the sounds through the clip, in order{sounds_hint}."""


# timeline: one full-size picture per moment, one sentence each; then the shot written from those
# sentences as text (single images are this captioner's strength: ~1280 px each, where 16-24 frames
# in one video call get ~460 px)
MOMENT_INSTRUCTION = """This is one frame from a video. The main person in this video is {who}. Answer in exactly six short labelled lines and nothing else:

person: "yes" if that main person is visible in this frame, otherwise "no".
pose: how that person sits, stands or moves, in a few words (for example "kneeling on the grass").
look: where that person looks: "at the camera", "down", "to the side", or at what (for example "at the object in their hands").
hands: what that person's hands do and hold, and where a held object is, as one full phrase with "a" and "the", starting with a verb (for example "holds a small black device with a red ring up beside the face with the right hand"). Name every object by what it looks like: colour, shape, size and any coloured parts.
others: anyone else in the frame and what they do, in a few words, or "none".
frame: what the frame shows, in a few words.

Describe only the main person in pose, look and hands, never anyone else. Never quote printed text or brand names."""

COMPOSE_INSTRUCTION = """Here is what a person does in one continuous shot of a video ({secs:.1f} seconds), moment by moment{and_says}:
{moments}

In an edit of this video the person is replaced by a character, called "{subj}" below, who does exactly what the person does. Rewrite the moments as {n_sent} short sentences, present tense, in the same order: merge moments that repeat, give the same object the same name every time, and keep every object {subj} picks up, holds up, shows or puts down, where {subj} holds it, and where {subj} looks (say "looking at the camera" when the moments say so). An object that the moments name differently from one moment to the next keeps its most frequent name. {dialogue_rule}Use only what the moments say: add nothing. Don't mention times, moments, frames or "the person".{extra} Write only the sentences."""

# asked only when a shot would be written as empty although a moment describes a pose or hands (below)
RECHECK_INSTRUCTION = """This is one frame from a video. Is {who} in this frame at all, even partly: only the top of the head, the back, a shoulder, an arm or the hands, seen from above, from behind or turned away? Answer only "yes" or "no"."""

WHO_INSTRUCTION = """These are {n} frames from one video clip. Who is the main person or character in it: the one most in focus, the largest, or the one doing the most? Prefer a person or a human-like character over an animal or a creature. Answer in a few words that tell them apart from anyone else in the clip by how they look (build, hair, clothes), never by what they hold or do, which changes from shot to shot; for example "a man in a black t-shirt", "a girl in a blue dress" or "a cartoon boy in a white hat". Answer "none" only if no person or character appears in any of the frames."""

SCENE_INSTRUCTION = """These are {n} frames from one clip of a video{talk}. The main person is {who}. Write exactly three labelled lines and nothing else:

video_1: ONE sentence about the main person, not anyone else: where the main person is and what they sit or stand on, what is in front of them, the background and its colour, the light (hard or soft, and the side it comes from, judged by where the shadows fall), the visual style (for example live-action phone video, or 2D cartoon animation), what the main person is doing, and the camera: whether it looks down on them, is level with them or looks up at them, whether it moves, and where their head sits in the frame. Don't describe their face, hair or clothes.
sounds: ONE sentence: the sounds through the clip, in order{sounds_hint}.
others: everyone else who appears in these frames, each in a few words by how they look and which side of the frame they are on, saying left, centre or right (for example "a man in a striped jacket on the right"), separated by semicolons; or "none"."""

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
    """The lines whose middle falls in shot i (untimed lines: all in shot 1, Qwen places them). The middle,
    not the start: Whisper stretches a line's first word back over the silence before it (B4: "So" at
    0.76-1.26 s put "So what have we got here?" in the shot before the one it's spoken in)."""
    a, b = shot_rel
    out = []
    for ln in lines:
        if "frames" not in ln:
            if i == 0:
                out.append(ln)
            continue
        mid = (ln["frames"][0] + ln["frames"][1]) // 2
        if a <= mid <= b or (i == k_shots - 1 and mid > b) or (i == 0 and mid < a):
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


def shot_instruction(*, n, secs, lines_rel, fps, pronoun, objects=(), extra="", part=None):
    pr = PRONOUNS[pronoun]
    objs = ", ".join(objects)
    if lines_rel:
        lb = "Words spoken in this shot, in order (times from the shot's start): " + ", ".join(
            _line_where(ln, fps) for ln in lines_rel) + ".\n"
    else:
        lb = "Nothing is spoken in this shot.\n"
    pt = (f" These pictures are part {part[0]} of {part[1]} of a longer shot: describe only this part, as it "
          f"continues from the part before.") if part and part[1] > 1 else ""
    return SHOT_INSTRUCTION.format(n=n, secs=secs, subj=pr["subj"], poss=pr["poss"], part=pt,
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


def shot_segments(shots_rel, fps, seg_s=None):
    """[(shot index, (a, b), part j, parts m)]: each shot, a long one cut into equal parts of at most seg_s."""
    seg_s = SEGMENT_S if seg_s is None else seg_s
    out = []
    for i, (a, b) in enumerate(shots_rel):
        n = b - a + 1
        m = max(1, -(-n // max(1, int(seg_s * fps)))) if seg_s else 1
        edges = [a + int(round(n * j / float(m))) for j in range(m + 1)]
        out += [(i, (edges[j], edges[j + 1] - 1), j, m) for j in range(m)]
    return out


def segment_lines(shot_lines, segs, i, j):
    """Of a shot's lines, those whose start falls in its part j (untimed lines: the first part)."""
    parts = [g for g in segs if g[0] == i]
    a, b = parts[j][1]
    last = j == len(parts) - 1
    return [ln for ln in shot_lines if ("frames" not in ln and j == 0) or
            ("frames" in ln and (a <= ln["frames"][0] <= b or (last and ln["frames"][0] > b) or (j == 0 and ln["frames"][0] < a)))]


def moment_frames(shot, fps, step_s=None, cap=16):
    """Timeline pictures for a shot (render-relative frames): one per step_s, 2 frames clear of the shot's
    edges (a cut's neighbour frames can blend), at least 2 and at most cap."""
    step_s = MOMENT_S if step_s is None else step_s
    a, b = shot
    lo, hi = (a + 2, b - 2) if b - a >= 6 else (a, b)
    n = max(2, min(cap, int(round((hi - lo + 1) / float(fps) / step_s)) + 1)) if hi > lo else 1
    return sorted({int(round(x)) for x in np.linspace(lo, hi, n)})


def compose_instruction(*, moments, secs, lines_rel, fps, pronoun, extra=""):
    """moments: [(seconds, sentence)]. The shot's lines go into the list at their start times, marked
    SAYS, so the rewrite keeps each where it's spoken (T1: given apart, Qwen left every line out)."""
    pr = PRONOUNS[pronoun]
    rows = [(t, f"{t:.1f} s: {m}") for t, m in moments]
    untimed = []
    for ln in lines_rel:
        if "frames" in ln:
            t = ln["frames"][0] / float(fps)
            rows.append((t + 1e-3, f'{t:.1f} s: SAYS "{ln["text"]}"'))
        else:
            untimed.append(ln["text"])
    rows.sort(key=lambda r: r[0])
    text = "\n".join(r for _, r in rows)
    if untimed:
        text += "\n(also spoken somewhere in the shot: " + ", ".join(f'SAYS "{u}"' for u in untimed) + ")"
    rule = ('Keep every SAYS "words" line exactly as written, at its place in the order, inside or between your '
            'sentences. ') if lines_rel else ""
    n_sent = "2 to 6" if len(moments) > 3 else "1 to 3"
    return COMPOSE_INSTRUCTION.format(secs=secs, moments=text, and_says=" (SAYS marks what is said, and when)"
                                      if lines_rel else "", subj=pr["subj"], n_sent=n_sent, dialogue_rule=rule,
                                      extra=_extra(extra).replace("\n- ", " "))


def parse_moment(text):
    """A moment answer -> {person, pose, look, hands, others, frame} (missing labels: the whole answer as hands)."""
    d = {}
    for lab in ("person", "pose", "look", "hands", "others", "frame"):
        m = re.search(rf"(?im)^[ \t*-]*{lab}[ \t*]*:[ \t]*(.+)$", text or "")
        d[lab] = " ".join(m.group(1).split()).strip(" .*") if m else ""
    if not any(d.values()):
        d["hands"] = " ".join((text or "").split()).strip(" .")
    for lab in ("pose", "look", "hands"):          # "none", "n/a": nothing to say
        if d[lab].lower().strip(" .") in ("none", "n/a", "na", "-", "not visible", "unknown"):
            d[lab] = ""
    return d


def present(m):
    """Whether the main person is in a moment's frame (no answer: assume yes)."""
    v = (m.get("person") or "").lower()
    return not v.startswith("no")


_ACTION = re.compile(r"(?i)(?:,\s*|\s+)(?:who is\s+|who's\s+|that is\s+)?\b(?!wearing\b)[a-z]+ing\b.*$")


def who_lasting(who):
    """The main person by what lasts: Qwen's "a bald man holding a knife" (100d chunk 1, B5a) made every later
    question about a shot without the knife answer "no". Drops the first "-ing" clause on (holding, carrying,
    sitting ...; "wearing" stays: clothes last) and anything after it."""
    out = _ACTION.sub("", who or "").strip(" ,.")
    return out if len(out.split()) >= 2 else (who or "")


def _describes_someone(m):
    """A moment whose pose or hands caption describes a body (not "none visible", "not visible ...")."""
    for lab in ("pose", "hands"):
        v = (m.get(lab) or "").lower().strip(" .")
        if v and not v.startswith(("none", "no ", "not ", "nobody", "n/a", "nothing", "empty")):
            return True
    return False


def recheck_absent(moments, pics, who, ask):
    """Qwen can answer person "no" while describing that person's pose and hands: 100d 186-208, the man bent over
    the box, seen from above (B5a; the shot was written as empty and H3 rendered an empty room). When no moment of
    a shot says yes, each "no" moment that describes a pose or hands gets one single-purpose question on its own
    picture (B4: those work where the six-line answer doesn't); a "yes" makes the moment present. Returns the
    number of moments turned present. Moments that describe no one aren't asked (the empty landscapes)."""
    if any(present(m) for _, m in moments):
        return 0
    n = 0
    for k, (_, m) in enumerate(moments):
        if not _describes_someone(m) or k >= len(pics):
            continue
        ans = (ask(RECHECK_INSTRUCTION.format(who=who or "the person most in focus"), pics[k]) or "").strip().lower()
        m["rechecked"] = ans.strip(" .*\"'").startswith("yes")
        if m["rechecked"]:
            m["person"] = "yes"
            n += 1
    return n


def _hands_clause(hands, pr):
    """A hands caption as a clause after the pronoun: "hold" -> "holds", "resting ..." -> "is resting ...";
    empty when it says the hands hold nothing or can't be seen ("not visible holding anything")."""
    h = hands.strip(" .")
    low = h.lower()
    if not h or low.startswith(("not ", "no ", "none", "nothing", "holds no", "holding no", "empty", "hands not",
                               "hands are not", "both hands not")):
        return ""
    w = h.split()
    if w[0].lower().endswith("ing"):
        return f"{'are' if pr['subj'] == 'they' else 'is'} {h[0].lower() + h[1:]}"
    if w[0].lower() in ("arms", "arm", "hands", "hand", "palms", "palm", "both", "left", "right", "fingers", "elbows"):
        return f"{'have' if pr['subj'] == 'they' else 'has'} {pr['poss']} {h[0].lower() + h[1:]}"
    return " ".join([_verb3(w[0].lower(), pr)] + w[1:])


def _third_to(pr, t):
    """The person / they / their -> the subject's pronoun, for a caption written about the person."""
    t = re.sub(r"\b[Tt]he person['’]s\b", pr["poss"], t)
    t = re.sub(r"\b[Tt]he person\b", pr["subj"], t)
    t = re.sub(r"\btheir\b", pr["poss"], t)
    t = re.sub(r"\bthemselves\b", "herself" if pr["subj"] == "she" else "himself" if pr["subj"] == "he" else "themselves", t)
    t = re.sub(r"\bthey\b", pr["subj"], t)
    # Qwen describes the source person ("near his mouth", "in his hands", B5a): the caption is about the main
    # person, so the subject's pronoun wins
    for a, b in _PRONOUN_SWAP[pr["subj"]]:
        t = re.sub(rf"\b{a}\b", b, t)
        t = re.sub(rf"\b{a.capitalize()}\b", b.capitalize(), t)
    return t


_PRONOUN_SWAP = {"she": [("himself", "herself"), ("his", "her"), ("him", "her"), ("he", "she")],
                 "he": [("herself", "himself"), ("her", "his"), ("she", "he")],
                 "they": [("himself", "themselves"), ("herself", "themselves"), ("his", "their"), ("him", "them"),
                          ("her", "their"), ("he", "they"), ("she", "they")]}


def say_line(ln, language):
    """A dialogue line as H3 reads it, by its speaker: <Subject 1> (S1) unless speaker_ids gave it another."""
    return f"{ln.get('tag') or '<Subject 1> (S1)'} says <d>[{language}]{ln['text']}</d>"


def speaker_ids(lines):
    """Each line's tag: its speaker ("by": a subject label or a voice description, else <Subject 1>) with the
    (Sx) H3's guide gives it, numbered in the order the voices first speak (ref guide §5.4)."""
    ids = {}
    for ln in sorted(lines, key=lambda x: x["frames"][0] if "frames" in x else float("inf")):
        who = ln.get("speaker_label") or "<Subject 1>"
        ids.setdefault(who, f"S{len(ids) + 1}")
    for ln in lines:
        who = ln.get("speaker_label") or "<Subject 1>"
        ln["tag"] = f"{who} ({ids[who]})"
    return ids


def attribute_lines(lines, frames, masks, model_path, fps):
    """Marks the timed lines another face says ("by": "other", and "by_x": where that face is across the
    frame, 0 left - 1 right), from whose mouth moves over each line's frames (swap_scores.face_mouths /
    speaker_of). frames and masks: the chunk's render range. -> count."""
    tgt, oth, oth_x = ss.face_mouths(frames, masks, model_path, fps)
    n = 0
    for ln in lines:
        a, b = ln["frames"] if "frames" in ln else (0, -1)
        if "frames" in ln and ss.speaker_of(tgt[a:b + 1], oth[a:b + 1]) == "other":
            xs = [x for x in oth_x[a:b + 1] if x is not None]
            ln["by"] = "other"
            ln["by_x"] = float(np.median(xs)) if xs else None
            n += 1
    return n


_SIDES = {"left": ("left",), "right": ("right",), "centre": ("centre", "center", "middle")}


def subject_at(x, people):
    """The subject the talking face is: the one described on its side of the frame ("on the right" for a
    face right of centre), else the one left when those described on another side are ruled out; None when
    none or several fit. people: [(label, description)]."""
    if x is None:
        return None
    side = "left" if x < 0.4 else "right" if x > 0.6 else "centre"

    def sides(desc):
        return {k for k, ws in _SIDES.items() if any(re.search(rf"\b{w}\b", desc.lower()) for w in ws)}
    fit = [lab for lab, desc in people if side in sides(desc)]
    if not fit:
        fit = [lab for lab, desc in people if not sides(desc)]
    return fit[0] if len(fit) == 1 else None


def _verb3(v, pr):
    """'hold' -> 'holds' for she / he (captions sometimes come in the base form)."""
    if pr["subj"] == "they" or not v or v.endswith("s"):
        return v
    return v + ("es" if v.endswith(("sh", "ch", "x")) else "s")


def merge_moments(moments, lines_rel, fps, pronoun, language="English", similar=0.85):
    """A shot block from its moments, by the node: one sentence per moment whose hands differ from
    the last kept one (a repeated hand with a new gaze adds the gaze), the shot's lines placed at their
    times. moments: [(seconds, {pose, look, hands})]. Nothing observed is dropped, nothing is added."""
    pr = PRONOUNS[pronoun]
    out, last_hands, last_look = [], None, None
    ev = [(t, "m", m) for t, m in moments]
    ev += [(ln["frames"][0] / float(fps) + 1e-3, "l", ln) for ln in lines_rel if "frames" in ln]
    ev.sort(key=lambda e: e[0])
    first = True
    for _, kind, x in ev:
        if kind == "l":
            out.append(say_line(x, language))
            continue
        if not present(x):
            continue                    # the main person isn't in this frame
        hands = _hands_clause(re.sub(r"\s*;\s*", ", and ", _third_to(pr, x.get("hands") or "")), pr)
        look = _third_to(pr, (x.get("look") or "").lower().strip(" ."))
        same = last_hands is not None and difflib.SequenceMatcher(None, hands.lower(), last_hands.lower()).ratio() >= similar
        if first:
            pose = _third_to(pr, x.get("pose") or "").strip(" .")
            w0 = pose.split()[0].lower() if pose else ""
            if not pose:
                lead = pr["Subj"]
            elif w0.endswith("ing") and w0 != "leaning":    # "sitting on the floor" -> "She is sitting ..."
                lead = f"{pr['Subj']} {'are' if pr['subj'] == 'they' else 'is'} {pose}"
            elif w0.endswith("s"):              # "sits on the floor ..."
                lead = f"{pr['Subj']} {pose}"
            else:                               # "leaning forward ...", "on the floor ..." -> "She sits leaning ..."
                lead = f"{pr['Subj']} sits {pose}"
            out.append(lead + (f" and {hands}" if hands else "")
                       + (f", looking {look}" if look.startswith(("at", "down", "up", "into", "toward", "to the")) else "")
                       + ".")
            first = False
        elif same:
            if look and look != last_look:
                out.append(f"{pr['Subj']} looks {look}.")
        elif hands:
            out.append(f"{pr['Subj']} {hands}" + (f", looking {look}" if look and look != last_look else "") + ".")
        last_hands, last_look = hands or last_hands, look or last_look
    untimed = [ln for ln in lines_rel if "frames" not in ln]
    out += [say_line(ln, language) for ln in untimed]
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def empty_shot(moments):
    """A shot in which the main person never appears: what it shows, and that it stays as it is."""
    seen = []
    for _, m in moments:
        f = (m.get("frame") or "").strip(" .")
        if f and all(difflib.SequenceMatcher(None, f.lower(), x.lower()).ratio() < 0.7 for x in seen):
            seen.append(f)
    what = "; then ".join(seen[:3]) or "the scene"
    return f"No one is replaced in this shot: {what[0].lower() + what[1:]}. It stays exactly as in <Video 1>."


def gender_of(text):
    t = " " + (text or "").lower() + " "
    if re.search(r"\b(woman|girl|lady|female|she|her|mother|princess)\b", t):
        return "f"
    if re.search(r"\b(man|boy|gentleman|male|he|his|father|guy)\b", t):
        return "m"
    return None


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

_LABEL = re.compile(r"(?im)^[ \t]*[*#>\- \t]*(video[_ ]?1|shots|sounds|name|appearance|pronoun|person|objects|others)"
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
    if not d.get("appearance"):
        # the three lines without their labels (B4 workshop S1): name, appearance, pronoun, in that order
        rows = [r.strip(" *-") for r in (text or "").splitlines() if r.strip(" *-")]
        if len(rows) >= 2:
            d = {"name": rows[0], "appearance": max(rows[1:], key=len),
                 "pronoun": next((r for r in rows[1:] if r.lower().strip(" .") in PRONOUNS), "")}
    name =re.sub(r"[\"'.]", "", (d.get("name") or "").splitlines()[0] if d.get("name") else "").strip().lower()
    app = " ".join((d.get("appearance") or "").split()).strip().strip('"')
    pron = (d.get("pronoun") or "").strip().lower().split()[0:1]
    pron = re.sub(r"[^a-z]", "", pron[0]) if pron else ""
    return {"name": name or "character", "appearance": app, "pronoun": pron if pron in PRONOUNS else None}


def subject_sentence(name, appearance):
    app = appearance.rstrip(" .")
    return (f"<Subject 1> is the {name} whose motion comes from <Video 1> and whose appearance comes from "
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
                     r"replies|mutters|muttering|murmurs|murmuring|whispers|whispering|declares|comments|commenting|"
                     r"finally remarks|then)\s*,?\s*)?[\"“]([^\"”]+)[\"”]", re.I)
_DTAG = re.compile(r"\s*(?:<Subject \d+>|\b[a-z][a-z' -]*? voice) \(S\d+\) says <d>\[[^\]]*\](.*?)</d>")


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
        return " " + say_line(lines[i], language)

    def from_tag(m):
        i = _match_line(m.group(1), lines, used)
        return tag(i) if i is not None else ""

    def from_quote(m):
        i = _match_line(m.group(1), lines, used)
        return tag(i) if i is not None else m.group(0)

    out = re.sub(r"(?:,?\s*(?:(?:she|he|they)\s+)?(?:says|saying|mutters|muttering|murmurs|whispers|states|"
                 r"stating|remarks|adds|exclaims|asks|notes|comments)\s*,?)\s*(?=(?:<Subject \d+>|\b[a-z][a-z' -]*? voice) \(S\d+\) says <d>)", " ", block,
                 flags=re.I)
    out = _DTAG.sub(from_tag, out)
    out = _QUOTED.sub(from_quote, out)
    missing = [i for i in range(len(lines)) if i not in used]
    for i in missing:
        # back in order: right after the line before it when that one is placed, else at the end
        t, at = tag(i), -1
        for k in range(i - 1, -1, -1):
            key = f"<d>[{language}]{lines[k]['text']}</d>"
            j = out.find(key)
            if j >= 0:
                at = j + len(key)
                break
        out = out[:at] + t + out[at:] if at >= 0 else out.rstrip() + t
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


def parse_qwen(text, n_shots, language="English", cap=8):
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
    blocks = [f"[Shot {i + 1}] " + convert_says(dedupe_sentences(b, cap), language) for i, b in enumerate(blocks)]
    return {"video_1": v1.rstrip(".") + "." if v1 else "", "shots": blocks, "sounds": sounds, "warnings": warns}


# ---------------------------------------------------------------------------
# the six-section Ref2VA template ("character replace (Ref2VA)")
# ---------------------------------------------------------------------------

def _cuts_phrase(n_cuts):
    return {0: "", 1: "the jump cut, ", 2: "both jump cuts, "}.get(n_cuts, f"all {n_cuts} jump cuts, ")


def fill_ref2va(*, subject, video_1, shots, sounds, n_shots, dialogue, audio, pronoun, who=None, others=False,
                people=(), speaking=False):
    """The six sections, filled from the job's subject and Qwen's three parts (design §4.6), after H3's
    reference guide (MiniMax-H3 skills/h3-prompt-writing/references/ref-en.txt). people: everyone else the
    scene names, [(label, description, shots they're in)], each a <Subject N> kept as it is; speaking: one of
    them, or an off-screen voice, says a line."""
    pr = PRONOUNS[pronoun]
    name = subject_name(subject)
    n_cuts = max(0, n_shots - 1)
    cp = _cuts_phrase(n_cuts)
    tag = "[video editing + reference generation" + (" + audio reuse]" if audio else "]")
    v1 = video_1 or "the source video, with a person in it."
    the_person = f"the main person ({who})" if who else "the person"
    summary = (f"{tag} The target video is an edited version of <Video 1> in which {the_person} is replaced by "
               f"<Subject 1>, the {name} styled from <Picture 1>. Everything else in <Video 1> is kept exactly: every "
               f"movement, lean, reach, hand position, head turn, mouth movement and expression timing, {cp}the camera "
               f"framing, the background, the lighting and shadows, and every object {pr['subj']} {pr['handles']}. The edit "
               f"runs continuously from the first frame to the last without deviation."
               + (f" {_and_list([lab for lab, _, _ in people])} {'keeps' if len(people) == 1 else 'keep'} exactly "
                  f"{'their' if len(people) == 1 else 'their'} own appearance and movements: only the main person is "
                  f"replaced." if people else
                  " Everyone else in <Video 1> keeps exactly their own appearance and movements: only the main person "
                  "is replaced." if others else ""))
    ret = [f"<Subject 1> (appears throughout): partially_preserved - {pr['poss']} face, hair, costume and accessories "
           f"from <Picture 1> are retained; {pr['poss']} body position, lean, pose, arm and hand actions, head "
           f"direction and how much of {pr['poss']} head is in frame, mouth movements and timing from <Video 1> are "
           f"retained at every moment."]
    for lab, desc, in_shots in people:
        where = "throughout" if len(in_shots) == n_shots else "in " + ", ".join(f"[Shot {k}]" for k in in_shots)
        ret.append(f"{lab} (appears {where}): fully_preserved - {desc} keeps exactly their own appearance, position "
                   f"and movements from <Video 1>.")
    ret += [f"<Video 1> (whole-video temporal structure, cuts, camera framing, background, props, lighting): "
           f"fully_preserved - the shot structure{' and every jump cut' if n_cuts else ''}, the camera framing, the "
           f"background, the lighting and shadows, and every object the person handles are preserved in every frame "
           f"without deviation. Nothing new is added to the scene."]
    if audio:
        ret.append("<Audio 1>: fully_copy - the original speech and room sound of <Video 1> are kept as they are"
                   + (", and each speaker's lips follow their own lines." if dialogue and speaking else
                      f", and {pr['poss']} lip movements follow the speech." if dialogue else "."))
    style = (f"The target video matches the source footage exactly in style: the same camera, framing, lighting, "
             f"shadows, colour grading and contrast as <Video 1>. {pr['Subj']} {pr['does']} exactly what the person "
             f"does in <Video 1>, frame by frame, framed exactly the same way; {pr['subj']} never {pr['moves']} to "
             f"the centre of the frame, never {pr['poses']} for the camera, and {pr['poss']} hands only hold what the "
             f"person's hands hold.")
    blocks = list(shots) or ["[Shot 1] " + f"{pr['Subj']} {pr['does']} exactly what the person does."]
    tail = []
    if dialogue and speaking:
        tail.append(f"{pr['Poss']} lips move only with {pr['poss']} own lines, in time with the speech in <Audio 1>; "
                    f"everyone else's lips move only with their own.")
    elif dialogue:
        tail.append(f"{pr['Poss']} lips move with every word, in time with the speech in <Audio 1>, and close between "
                    f"sentences.")
    if n_cuts:
        tail.append("Every jump cut happens at exactly the same moment as in <Video 1>, with no transition, dissolve "
                    "or morph.")
    if tail:
        blocks[-1] = blocks[-1].rstrip() + " " + " ".join(tail)
    sound = ("The original sound of <Video 1>: " + (sounds or "the room tone and the sounds of the action.")) if audio \
        else (sounds or "Quiet room tone.")
    defs = "".join(f"\n{lab} is {desc} in <Video 1>, who keeps their own appearance and movements." for lab, desc, _ in people)
    a1 = ("\n<Audio 1> is the synchronized audio track of <Video 1>, and is reused in the target video." if audio else "")
    return ("subject_definitions:\n" + subject.strip() + defs + "\n<Video 1> is the source video for the target video edit: "
            + v1 + a1 + "\n\nsummary:\n" + summary + "\n\nretention_analysis:\n" + "\n".join(ret)
            + "\n\ndetailed_description:\n" + style + "\n" + "\n".join(blocks)
            + "\n\noverall_soundscape:\n" + sound + "\n\nnon_diegetic_music:\nN/A\n")


_LINE = re.compile(r"(<Subject \d+>|\b[a-z][a-z' -]*? voice) \((S\d+)\) says <d>\[([^\]]*)\](.*?)</d>")
_VOICE = re.compile(r"[a-z][a-z' -]* voice")
_LIPS_ALL = re.compile(r"(Her|His|Their) lips move with every word, in time with the speech in <Audio 1>, and close between "
                       r"sentences\.")
_LIPS_OWN = re.compile(r"(Her|His|Their) lips move only with (her|his|their) own lines, in time with the speech in <Audio 1>; "
                       r"everyone else's lips move only with their own\.")


def set_line_speaker(prompt, index, speaker):
    """The prompt with its index-th dialogue line said by `speaker` (a <Subject N> the prompt defines, or a
    voice description such as "an off-screen male voice"): speaker IDs renumbered in the order the voices
    first speak, and the lips lines saying whose lips move (H3's ref guide §5.4). The strip's per-line
    speaker picker, for the lines the drafter can't tell (no face seen)."""
    sec = parse_sections(prompt)
    defined = re.findall(r"(?m)^(<Subject \d+>) is ", sec["subject_definitions"])
    if speaker not in defined and not _VOICE.fullmatch(speaker):
        raise DraftError(f"{speaker!r} isn't a subject this prompt defines ({', '.join(defined)}) or a voice")
    d = sec["detailed_description"]
    ms = list(_LINE.finditer(d))
    if not 0 <= int(index) < len(ms):
        raise DraftError(f"the prompt has {len(ms)} dialogue line(s), not a line {int(index) + 1}")
    who = [m.group(1) for m in ms]
    who[int(index)] = speaker
    ids = {}
    for w in who:
        ids.setdefault(w, f"S{len(ids) + 1}")
    out, at = [], 0
    for m, w in zip(ms, who):
        out += [d[at:m.start()], f"{w} ({ids[w]}) says <d>[{m.group(3)}]{m.group(4)}</d>"]
        at = m.end()
    d2 = "".join(out) + d[at:]
    ret = sec["retention_analysis"]
    poss = (re.search(r"<Subject 1> \([^)]*\): \w+ - (her|his|their) ", ret) or [None, "her"])[1]
    if any(w != "<Subject 1>" for w in who):
        d2 = _LIPS_ALL.sub(lambda m: f"{m.group(1)} lips move only with {poss} own lines, in time with the speech in "
                                     f"<Audio 1>; everyone else's lips move only with their own.", d2)
        ret2 = ret.replace(f", and {poss} lip movements follow the speech.", ", and each speaker's lips follow their own lines.")
    else:
        d2 = _LIPS_OWN.sub(lambda m: f"{m.group(1)} lips move with every word, in time with the speech in <Audio 1>, and "
                                     f"close between sentences.", d2)
        ret2 = ret.replace(", and each speaker's lips follow their own lines.", f", and {poss} lip movements follow the speech.")
    return prompt.replace(d, d2, 1).replace(ret, ret2, 1)


def _and_list(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


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
          "held", "without", "free", "aloft", "toward", "inside", "back", "angled", "tilted", "propped"}


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


def drop_unseen_hands(moments, objects):
    """Blank each moment's hands that name a prop the chunk's object list doesn't contain, so the shot
    block never puts it in her hands (B5b c2: a knife read as a "cigar" and a "box lid", rendered as
    such on two seeds). Returns the dropped phrases for the chunk's warnings."""
    dropped = []
    for _, m in moments:
        unseen = prop_check([m.get("hands") or ""], objects)["unseen"]
        if unseen:
            dropped += [p for _, p in unseen if p not in dropped]
            m["hands"] = ""
    return dropped


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
        """One question to the loaded model through the node's own generate() (what its run() calls after
        loading). run() re-resolves the attention backend and logs it on every call (two console lines per
        question, ~17 questions a chunk with the timeline template), so it's used only to load."""
        s = QWEN_SAMPLING
        if getattr(self.node, "model", None) is None or not hasattr(self.node, "generate"):
            out = self.node.run(self.model, self.quant, "", prompt, image, video, frame_count, max_tokens,
                                s["temperature"], s["top_p"], s["num_beams"], s["repetition_penalty"], s["seed"], True,
                                self.attention, False, "auto")
            return out[0] if isinstance(out, (tuple, list)) else str(out)
        torch.manual_seed(s["seed"])
        return self.node.generate(prompt, image, video, frame_count, max_tokens, s["temperature"], s["top_p"],
                                  s["num_beams"], s["repetition_penalty"], video_frame_size="auto")

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
        g = getattr(getattr(cls, "run", None), "__globals__", None) or vars(sys.modules.get(cls.__module__) or object)
        # its structured prompt: without it Omni writes free prose (B4: a module loaded from file isn't in
        # sys.modules, so the lookup there found nothing and every answer came back as prose)
        self.prompt = g.get("DEFAULT_PROMPT_WIDGET_TEXT") or "Describe this audio."
        if self.prompt == "Describe this audio.":
            _say("[SeamStitch] Swap Draft: the Omni node's structured prompt wasn't found: its answers may be prose")
        missing = [p for p in (g.get("DEFAULT_CLI"), g.get("DEFAULT_MODEL"), g.get("DEFAULT_MMPROJ"))
                   if p and not os.path.exists(p)]
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
            song = cheer = ""
            if po is not None and po.get("looping"):
                ci["warnings"].append("Omni repeated one phrase over and over (a loop on music, drums or cheering): its "
                                      "words dropped")
                po = dict(po, words="", speech=bool(words), speakers=[])
            if words and is_looping(" ".join(w["word"] for w in words)):
                ci["warnings"].append("Whisper repeated one phrase over and over: its words dropped")
                words = []
            if po is not None and po.get("sung") and po["words"]:
                song = po.get("sung_words") or po["words"]
                spoken = po.get("spoken") or ""
                if spoken and not interjections_only(spoken):
                    # a line, then a song (B5a): the line stays, the song goes into the soundscape
                    ci["warnings"].append(f"part of the audio is sung ({song[:40]}...): into the soundscape; the "
                                          f"spoken words kept as lines")
                    po = dict(po, words=spoken, speakers=po.get("spoken_speakers") or [None] * len(spoken.split()))
                else:
                    ci["warnings"].append("the words are sung (a song, not lines she says): into the soundscape")
                    po = dict(po, words="", speakers=[], speech=False)
                    words = []                  # Whisper heard the same lyrics
            if po is not None and po["words"] and words is not None and len(words) <= 2 and \
                    len(po["words"].split()) >= 6:
                if not words and not is_looping(po["words"]):
                    # Whisper can return nothing at all on a stretch Omni hears clearly (B5a 785-891)
                    ci["warnings"].append(f"Whisper heard nothing; Omni's {len(po['words'].split())} spoken words "
                                          f"kept without timings: check them against the clip")
                else:
                    ci["warnings"].append(f"Omni heard {len(po['words'].split())} words where Whisper heard "
                                          f"{len(words)}: Omni's words dropped (likely invented)")
                    po = dict(po, words="", speakers=[])
            if po is not None and not po["speech"] and not words:
                lines, aligned = [], []
            elif po is not None and po["words"]:
                aligned = align_words(po["words"], words, po.get("speakers")) if words else []
                lines = dialogue_lines(aligned, fps, ci["n"]) if words else lines_from_text(po["words"])
                if not words and (wh is not None or word_timings):
                    ci["warnings"].append("no word timings: the lines go to Qwen without frames")
            elif words:
                good = [w for w in words if w.get("p", 1) >= 0.4]
                aligned = [dict(w, matched=True) for w in unstretch(good)]
                lines = dialogue_lines(aligned, fps, ci["n"])
                if omni is not None or transcribe:
                    ci["warnings"].append("dialogue from Whisper alone (no Omni words)")
            else:
                lines, aligned = [], []
            said = " ".join(ln["text"] for ln in lines)
            if lines and interjections_only(said):
                cheer = said
                ci["warnings"].append(f"only cheers or exclamations heard ({said[:60]}): into the soundscape, not lines")
                lines = []
            lines, hum = drop_hum_lines(lines)
            if hum:
                ci["warnings"].append(f"a hum or exclamation, not words ({hum[:60]}): into the soundscape, not a line")
            ci.update(lines=lines, language=lang, events=(po or {}).get("events") or "",
                      notes=(po or {}).get("notes") or [], song=song, cheer=cheer, hum=hum)
            if words is not None or po is not None:
                _write_json(os.path.join(ci["dir"], "whisper.json"),
                            {"language": wlang, "words": words or [], "aligned": aligned, "lines": lines,
                             "fps": fps, "render": ci["render"]})

        # 4. Qwen, loaded once: the subject, then each chunk
        stage("Qwen")
        ram = ram_gb()
        report["ram_gb_before_qwen"] = ram
        if ram and ram["available"] is not None and ram["available"] + ram["swap_free"] < QWEN_RAM_GB:
            warnings.append(f"low memory before Qwen ({ram['available']:.0f} GB RAM + {ram['swap_free']:.0f} GB swap free): "
                            f"ComfyUI still holds the last render's models; the strip's draft buttons free them first "
                            f"(queued another way: POST /free with free_memory, then queue the draft)")
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
            subject = ("<Subject 1> is the character whose motion comes from <Video 1> and whose appearance "
                       "comes from <Picture 1>.")
        if re.search(r"<Subject 1>\s*\(S\d+\)", subject):
            warnings.append("the subject carries a speaker ID ('(S1)'): H3's guide keeps IDs out of subject_definitions; "
                            "delete it in the strip's subject box (every prompt follows)")
        k_frames = max(1, int(frames_per_chunk))
        per_shot = template == TEMPLATES[1]
        timeline = template == TEMPLATES[2]
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
            if not per_shot and not timeline:
                instr = draft_instruction(n_frames=ci["n"], fps=fps, sample_idx=idx, shots_rel=ci["shots_rel"],
                                          lines=ci["lines"], language=ci["language"], prev_names=prev_names,
                                          events=ci["events"], pronoun=pronoun, objects=objs["objects"],
                                          extra=extra_instructions)
                draw = qwen.ask(instr, video=video, max_tokens=int(max_tokens), frame_count=len(idx))
                log += ["==== draft: instruction ====", instr, "", "==== draft: Qwen ====", draw, ""]
                parts = parse_qwen(draw, len(ci["shots"]), ci["language"])
            elif per_shot:
                # each shot from its own frames, then video_1 and sounds over the chunk
                blocks = []
                k_sh = len(ci["shots_rel"])
                segs = shot_segments(ci["shots_rel"], fps)
                seg_frames = shot_frames([g[1] for g in segs], ci["n"], k_frames)
                bodies = [[] for _ in ci["shots_rel"]]
                for (i, sg, j, m), fr_idx in zip(segs, seg_frames):
                    said = segment_lines(lines_in_shot(ci["lines"], ci["shots_rel"][i], k_sh, i), segs, i, j)
                    rel = [dict(ln, frames=[ln["frames"][0] - sg[0], ln["frames"][1] - sg[0]]) if "frames" in ln else ln
                           for ln in said]
                    instr = shot_instruction(n=len(fr_idx), secs=(sg[1] - sg[0] + 1) / float(fps), lines_rel=rel,
                                             fps=fps, pronoun=pronoun, objects=objs["objects"], extra=extra_instructions,
                                             part=(j + 1, m))
                    sv = decode_sampled(path, fps, r0, fr_idx)
                    sraw = qwen.ask(instr, video=sv, max_tokens=512, frame_count=len(fr_idx))
                    tag = f"shot {i + 1}" + (f" part {j + 1}/{m}" if m > 1 else "")
                    log += [f"==== {tag}: instruction ====", instr, "", f"==== {tag}: Qwen ====", sraw, ""]
                    bodies[i].append(" ".join(re.sub(r"^\s*(?:\[Shot\s*\d+\]|shot\s*\d+\s*:)\s*", "", sraw,
                                                     flags=re.I).split()))
                blocks = [f"[Shot {i + 1}] " + " ".join(b) for i, b in enumerate(bodies)]
                sidx = sample_indices(ci["n"], min(SCENE_FRAMES, k_frames)) if SCENE_FRAMES > 0 else idx
                cinstr = CHUNK_INSTRUCTION.format(n=len(sidx), secs=ci["n"] / float(fps),
                                                  sounds_hint=_sounds_hint(ci["events"], ci["lines"]))
                craw = qwen.ask(cinstr, video=decode_sampled(path, fps, r0, sidx), max_tokens=384,
                                frame_count=len(sidx))
                log += ["==== chunk: instruction ====", cinstr, "", "==== chunk: Qwen ====", craw, ""]
                parts = parse_qwen(craw + "\nshots:\n" + "\n".join(blocks), len(ci["shots"]), ci["language"])
            elif timeline:
                # the scene first (it names the main person every caption is about), from 3 frames
                k_sh = len(ci["shots_rel"])
                sidx = sorted({0, ci["n"] // 2, ci["n"] - 1})
                talk = ", in which someone speaks" if ci["lines"] else ""
                widx = sample_indices(ci["n"], 5)
                wraw = qwen.ask(WHO_INSTRUCTION.format(n=len(widx)), video=decode_sampled(path, fps, r0, widx,
                                                                                          max_side=1280),
                                max_tokens=64, frame_count=len(widx))
                log += ["==== who: Qwen ====", wraw, ""]
                who = " ".join(re.sub(r"(?i)^\s*(main person|person|answer)\s*:\s*", "", wraw).split()).strip(' ."')
                if who.lower().startswith(("none", "n/a", "no person", "no one")) or len(who) > 160:
                    who = ""
                who = who_lasting(who)
                if sp.chunk_target(plan, c):
                    who = sp.chunk_target(plan, c)      # who to replace, set by hand (the chunk's, else the job's), wins
                ci["who"] = who
                # the scene knows who it's about (B7: a two-person shot's line was written around the man) and names
                # everyone else, whom the prompt then keeps as they are
                sinstr = SCENE_INSTRUCTION.format(n=len(sidx), talk=talk, who=who or "the person most in focus",
                                                  sounds_hint=_sounds_hint(ci["events"], ci["lines"]))
                craw = qwen.ask(sinstr, video=decode_sampled(path, fps, r0, sidx, max_side=1280), max_tokens=384,
                                frame_count=len(sidx))
                log += ["==== scene: instruction ====", sinstr, "", "==== scene: Qwen ====", craw, ""]
                named = " ".join((_labelled(craw).get("others") or "").split()).strip(' ."')
                if named.lower().startswith(("none", "no one", "nobody", "n/a")) or len(named) > 240:
                    named = ""
                # everyone else the scene names is a <Subject N> of their own (ref guide §2.3)
                people = [x.strip(' .') for x in named.split(";") if x.strip(' .')][:3]
                people = [x[0].lower() + x[1:] for x in people]
                labels = [f"<Subject {k + 2}>" for k in range(len(people))]
                # who says each line, when someone else is in the clip: whose mouth moves (the target's face is the
                # one inside the chunk's cached mask). One other person speaks as their own subject; with more, as
                # "another person's voice" (B7: every line of a two-person scene went to her)
                if named and any("frames" in ln for ln in ci["lines"]):
                    ok, why = ss.mouth_available()
                    m, _ = spl.cached_mask(plan, jd, r0, r0 + ci["n"] - 1, 0, sp.chunk_target(plan, c))
                    if not ok:
                        ci["warnings"].append(f"who says each line wasn't checked ({why}): every line is hers")
                    elif m is None:
                        ci["warnings"].append("who says each line wasn't checked: mark this chunk first (the mask "
                                              "tells her face from the others'), then redraft; every line is hers")
                    else:
                        k = attribute_lines(ci["lines"], list(tl._iter_frames(path, fps, r0, r0 + ci["n"])), m.numpy(),
                                            why, fps)
                        for ln in ci["lines"]:
                            if ln.get("by") == "other":
                                # one other person: theirs; several: whoever the talking face's place in frame fits
                                ln["speaker_label"] = (labels[0] if len(labels) == 1 else
                                                       subject_at(ln.get("by_x"), list(zip(labels, people)))
                                                       or "another person's voice")
                        if k:
                            ci["warnings"].append(f"{k} line(s) said by {people[0] if len(people) == 1 else 'someone else'}"
                                                  f": not hers (check them against the clip)")
                # lines from a voice that isn't the main person's (Omni's speaker labels): an off-screen voice
                g_who = gender_of(who)
                for ln in ci["lines"]:
                    g = gender_of(ln.get("speaker"))
                    if not ln.get("speaker_label") and ln.get("speaker") and g_who and g and g != g_who:
                        ln["speaker_label"] = f"an off-screen {'male' if g == 'm' else 'female'} voice"
                        ci["warnings"].append(f"a line by another voice ({ln['speaker']}): an off-screen voice, not hers")
                blocks, others, seen_in = [], [], []
                for i, sh in enumerate(ci["shots_rel"]):
                    mf = moment_frames(sh, fps)
                    pics = decode_sampled(path, fps, r0, mf, max_side=1280)
                    moments = []
                    minstr = MOMENT_INSTRUCTION.format(who=who or "the person most in focus")
                    for f, pic in zip(mf, pics):
                        mraw = qwen.ask(minstr, image=pic[None], max_tokens=240)
                        moments.append(((f - sh[0]) / float(fps), parse_moment(mraw)))
                    here = [m["others"] for _, m in moments if m.get("others") and
                            m["others"].lower().strip(" .") not in ("none", "no one", "nobody", "n/a")]
                    others += here
                    if here or not any(present(m) for _, m in moments):
                        seen_in.append(i + 1)
                    said = lines_in_shot(ci["lines"], sh, k_sh, i)
                    rel = [dict(ln, frames=[ln["frames"][0] - sh[0], ln["frames"][1] - sh[0]]) if "frames" in ln else ln
                           for ln in said]
                    recheck_absent(moments, pics, who, lambda q, pic: qwen.ask(q, image=pic[None], max_tokens=8))
                    if not any(present(m) for _, m in moments):
                        body = empty_shot(moments)          # she isn't in it: its lines are someone else's
                        for ln in said:
                            ln.setdefault("speaker_label", labels[0] if len(labels) == 1 else "another person's voice")
                        speaker_ids(ci["lines"])
                        body += "".join(" " + say_line(ln, ci["language"]) for ln in said)
                        ci.setdefault("empty_shots", []).append(i)
                    else:
                        dropped = drop_unseen_hands(moments, objs["objects"])
                        if dropped:
                            ci["warnings"].append(f"shot {i + 1}: left out of her hands (not in the frames' object "
                                                  f"list): " + ", ".join(dropped))
                        speaker_ids(ci["lines"])
                        rel = [dict(ln, frames=[ln["frames"][0] - sh[0], ln["frames"][1] - sh[0]]) if "frames" in ln
                               else ln for ln in said]
                        body = merge_moments(moments, rel, fps, pronoun, ci["language"])
                    log += [f"==== shot {i + 1}: moments (frames {[r0 + f for f in mf]}) ====",
                            "\n".join(f"{t:.1f} s: {json.dumps(m, ensure_ascii=False)}" for t, m in moments), "",
                            f"==== shot {i + 1}: merged ====", body, ""]
                    if people and here and not any(f"{labels[0]}," in b for b in blocks):
                        # the others at their first appearance (ref guide §5.3)
                        body = " ".join(f"{lab}, {desc}, stays exactly as in <Video 1>."
                                        for lab, desc in zip(labels, people)) + " " + body
                    blocks.append(f"[Shot {i + 1}] {body}")
                ci["others"] = bool(named or others)
                ci["people"] = [(lab, desc, seen_in or list(range(1, k_sh + 1))) for lab, desc in zip(labels, people)]
                parts = parse_qwen(craw + "\nshots:\n" + "\n".join(blocks), len(ci["shots"]), ci["language"], cap=24)
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
            extra_sounds = []
            if ci.get("song"):
                extra_sounds.append(f'a voice sings "{ci["song"][:160]}"')
            if ci.get("cheer"):
                extra_sounds.append(f'voices shout "{ci["cheer"][:80]}"')
            if ci.get("hum"):
                extra_sounds.append(f'a voice hums "{ci["hum"][:40]}"')
            for ln in ci.get("offscreen", []):
                extra_sounds.append(f'{("a " + ln["speaker"].lower() + "\'s voice") if ln.get("speaker") else "a voice"} '
                                    f'off camera says "{ln["text"]}"')
            if extra_sounds:
                parts["sounds"] = parts["sounds"].rstrip(" .") + "; " + "; ".join(extra_sounds) + "."
            drafted_objects[c["id"]] = objs["objects"]
            ci["warnings"] += parts["warnings"]
            if objs["person"] is False:
                ci["warnings"].append("Qwen saw no person in this chunk")
            text = fill_ref2va(subject=subject, video_1=parts["video_1"], shots=parts["shots"], sounds=parts["sounds"],
                               n_shots=len(ci["shots"]), dialogue=bool(ci["lines"]), audio=audio_ok, pronoun=pronoun,
                               who=ci.get("who"), others=ci.get("others", False), people=ci.get("people", ()),
                               speaking=any(ln.get("speaker_label") for ln in ci["lines"]))
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
                    "timeline (the default): one full-size picture every half second, a caption each, merged into "
                    "each shot with its lines at their times. per shot: each shot from its own frames in one call. "
                    "The first: one call for the whole chunk."}),
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
