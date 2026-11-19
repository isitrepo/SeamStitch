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

import contextvars
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
             "character replace (Ref2VA, timeline)", "character replace (Ref2VA, timeline, lean)",
             "character replace (Ref2VA, timeline, lean, 2 references)",
             "character replace (Ref2VA, lean, large model)",
             "background replace (Ref2VA)", "background replace (Ref2VA, 2 references)",
             "background replace (Ref2VA, lean)", "background replace (Ref2VA, lean, 2 references)",
             "background replace (Ref2VA, lean, large model)",
             "character replace (BFS body_swap, large model)",
             "object replace / add (Ref2VA)",
             "background replace (Ref2VA, lean, large model, 2 characters)"]
# lean (2026-10-09): the timeline's questions and placement, written short, after his pre-Swap prompt that worked:
# one look sentence, attribute_transfer onto the person, the source's background / camera / light named once, no
# per-moment hand lines unless she handles something. Wrestling c4, same seed: pose 0.49 against the timeline's 0.37.
LEAN_TEMPLATE = TEMPLATES[3]
# the lean template with a second reference picture (the node's sheet_2): <Picture 2> is a second view of the same
# character (a face close-up, another angle), described once per job and defined in the subject
LEAN_2REF = TEMPLATES[4]
# the lean template for a large GGUF model (2026-10-16, Qwen3.8-27B): the lean sections and scene questions, but each
# shot in one call (its pictures in order with their times, the cast by label, its lines) written by the model,
# where lean captions one picture at a time and merges the captions in code (test clip c5 on the 27B: the merge kept
# the first picture's pose, and the relation question put him across the wrong person's lap)
LARGE_TEMPLATE = TEMPLATES[5]
LARGE_SHOT_FRAMES = 8           # pictures per shot call (one per half second, at most this many)
LARGE_FRAME_SIDE = 896          # their long side: 8 x ~350 tokens fits GGUF_CTX with the instruction and the answer
LARGE_THINK = True              # its calls reason first (thinking mode): test clip c5, 3 seeds each, thinking caught the
                                # crossed legs 3/3 (instruct 0/2) and each head turn; ~225 s a chunk against ~130 s
# background replace (the strip's mode switch): the sheet is a place, the person stays (<Subject 2>). The job's mode
# decides the kind of template: a background job always drafts with one of these two, a character job with one of
# the character templates, whatever the widget says (the report says which was used).
BACKGROUND = TEMPLATES[6]
# the same with a second reference picture: <Picture 2> is a second view of the same place (a closer look at a
# surface, another angle). A background job takes it when sheet_2 is wired, or when the widget asks for it.
BACKGROUND_2REF = TEMPLATES[7]
# the lean forms of the two: the same one-call draft of the kept person, the six sections written short (one place
# sentence, the retention lines in one clause each, no style paragraph). Not the character lean's half-second
# moments and mask placement: a background job keeps the person as filmed, so what the prompt adds is the place.
# Picked from the widget only (a character template in the widget still gets the full form).
BACKGROUND_LEAN = TEMPLATES[8]
BACKGROUND_LEAN_2REF = TEMPLATES[9]
# the background lean form for a large GGUF model (2026-10-17): the lean sections, placement lines and place sentence
# unchanged, but each shot of the kept person in one call (its pictures in order with their times), the clip's camera
# and sounds from one picture per shot, and the who / place / object questions through the same calls, as the
# character large template does
LARGE_BG_TEMPLATE = TEMPLATES[10]
# the same, for a clip with TWO people kept as filmed (2026-10-28): the place is <Subject 1>, the first person
# <Subject 2>, the second <Subject 3>; each shot is described for both, and how they sit or stand against each other
LARGE_BG2_TEMPLATE = TEMPLATES[13]
LARGE_TEMPLATES = (TEMPLATES[5], LARGE_BG_TEMPLATE, LARGE_BG2_TEMPLATE)
# the BFS H3 Character Swap LoRA's prompt (2026-10-18): for the workflow's BFS switch (BFS LoRA + an aligned guide of
# the plain source, no <Video 1>). The LoRA wants its own body_swap template with a head-to-toe description of the
# character; the drafter asks the model ONCE, from the sheet, for that description (BFS's own instruction, from their
# TextGenerate node), fills the template (the {target} form when the job has a target) and writes the same prompt
# into every chunk drafted. No dialogue and no per-shot lines: the guide carries the motion. Tested on an unlike swap:
# a full swap; on look-alike swaps (a look-alike test clip) the LoRA passes the source actor through whatever the prompt.
# The text is BFS's v1.1 template (Civitai 3412620, for minimax_h3_character_swap_v1.1_ref2va_fro): "complete
# identity ... facial structure" where v1.0 said "complete look ... head, face", and v1.0's "no trace of the original
# subject" / "real, photographic video" lines are gone.
BFS_TEMPLATE = TEMPLATES[11]
BFS_BODY_SWAP = (
    "body_swap: subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity, face, head, "
    "hair or fur, skin, body shape, proportions and clothing define the character in the target video. The guide video "
    "provides the scene and the performance: its background, environment, lighting, camera, framing, body pose, hand "
    "and arm movement, head rotation, gaze direction, lip movement and facial expression, frame by frame; only {who} "
    "is replaced: {their} face, hair, body and clothing become <Subject 1>.  summary: [video generation + reference] "
    "Replace {who_short} with <Subject 1>, keeping the scene, camera and the complete motion and performance of the "
    "guide video.  detailed_description: [Shot 1] <Subject 1> takes the exact place of {who_place} and moves with the "
    "same timing, body pose, gestures, head motion, gaze, speech and facial expressions, while keeping the complete "
    "identity from <Picture 1>: facial structure, hair or fur, skin, body shape and clothing stay stable in every "
    "frame. The background, lighting, colors and camera of the guide video are preserved. Prevent identity drift, "
    "clothing changes, texture crawling and flicker.")
BFS_DETAILS_INSTRUCTION = """Look at the character in this picture (a character sheet: one character, possibly shown from several angles) and describe their complete visible appearance from head to toe as ONE detailed, comma-separated attribute list.

Include, in this exact order:
1. Gender, apparent height, body type, physique, shoulder width, waist, and body proportions.
2. Head and face shape, jawline, facial structure, skin tone, eye color, and distinctive facial characteristics.
3. Hair color, hairstyle, length, texture, beard, mustache, and other visible facial hair.
4. Upper-body clothing, including garment types, exact colors, fit, fabric appearance, textures, collars, sleeves, buttons, zippers, pockets, logos, patterns, and clothing layers.
5. Lower-body clothing, including pants, jeans, shorts, skirts or other garments, with colors, materials, fit, and visible design details.
6. Footwear, including shoes, boots, or sneakers, with colors, materials, style, and visible details.
7. All visible accessories, including hats, glasses, earrings, necklaces, chains, watches, bracelets, rings, belts, bags, scarves, wings, halos and piercings.
8. Distinctive features, including tattoos, scars, wrinkles, makeup, and other identifying characteristics.

Rules: start with the gender ("Male, ..." or "Female, ..."), with no article before it; describe only what is visible; never invent unseen parts; omit what can't be told from the picture; concise, visually specific English attributes; no background, camera, lighting or composition; one line.

Answer with the attribute list ONLY (it is inserted after "<Subject 1> is the" and before "shown in <Picture 1>"). Example of the format only: Male, average height, sturdy build, broad shoulders, oval face, light brown skin, blue eyes, grayish-white tousled hair, long thick beard, dark gray flat cap, dark jacket, black inner shirt, silver wristwatch"""
# object job (swap_plan.MODE_OBJECT, 2026-10-20): the sheet is a thing; the job's target is what it replaces (or, for
# "add", where it goes). The model is asked twice: once about the sheet (what the object looks like, as BFS's sheet
# question), once per chunk about a few of its frames (the scene, and where the target is and how it moves). The six
# sections are H3's (the same order as the character prompts); retention for the object is attribute_transfer (the
# guide's visual relationships: fully_preserved, partially_preserved, attribute_transfer, weak_reference), the rest of
# <Video 1> partially_preserved. Only the inside of the target's mask is kept at assembly, so the prompt needn't hold
# the rest of the frame perfectly; it just mustn't move the target.
OBJECT_TEMPLATE = TEMPLATES[12]
OBJECT_DETAILS_INSTRUCTION = """Look at the main object in this picture and describe how it looks as ONE comma-separated attribute list: what it is, its colours, its materials and finish, its shape and proportions, any markings or details on it, and what it shows if it is a picture or a screen. Describe only what is visible; no background, camera or lighting; one line.

Answer with the attribute list ONLY (it is inserted after "<Subject 1> is the" and before "shown in <Picture 1>"). Example of the format only: framed oil painting of a ship at sea, dark wooden frame, landscape format, muted blue and grey tones"""
# a flat thing that shows a picture (2026-10-20, test clip: "replace the painting with this photo of a car"): the
# sheet is what the picture shows, not an object to stand in its place, so the model describes it as a picture
# (asked of a car photo as an object, the 27B wrote "vintage four-door sedan", and the prompt then put a car there)
OBJECT_PICTURE_TARGET = re.compile(r"(?i)\b(painting|picture|photo(graph)?|poster|print|portrait|canvas|frame|screen|"
                                   r"monitor|tv|television|sign|billboard|mural|artwork|tapestry)s?\b")
OBJECT_PICTURE_INSTRUCTION = """This picture will be shown as the image inside a flat picture in a video (a painting, a framed photo, a poster or a screen on a wall). Describe it as that image in ONE comma-separated attribute list: what kind of image it is (photograph or painting, colour or black-and-white), what it shows and where in the image, its tones and colours, its overall shape (landscape or portrait). One line, no camera or lighting of the room.

Answer with the attribute list ONLY (it is inserted after "<Subject 1> is the" and before "shown in <Picture 1>"). Example of the format only: black-and-white photograph of a sailing ship on a calm sea, ship in the centre, grey sky, landscape format"""
OBJECT_SCENE_INSTRUCTION = """You are shown {n} pictures, frames sampled in order from one clip of a video ({secs:.1f} seconds). Answer in exactly three lines:
video_1: ONE sentence about the clip as a whole: the setting, who is in it and what they do, the light, and the camera (whether it moves, its angle and framing).
placement: ONE sentence about {the_target}: where it is in the frame (left, centre or right; top, middle or bottom), how large it is in the frame, what is right around it, and whether it stays still or moves (and how) during the clip. If it isn't in some pictures, say when it is out of frame.
people: every whole person you can see, each in a few words by how they look and where they are in the frame (left, centre or right), separated by semicolons (for example "a man in a red jacket in the centre; a woman with a skull face on the left"); or "none". Leave out {the_target} and anything worn on it or attached to it, and never list a body part on its own (a leg, a foot, a hand): a person who is only partly in frame is listed as that person.
Write about what you see, not about an edit."""
# the person making the edit says what to add, in their own words (the job's brief); the model, who sees the frames, turns it into the
# sentence the prompt needs: where the new subject is, what it does, who it is next to. The zone never feeds the drafter.
OBJECT_ADDITION_LINE = """
addition: ONE sentence saying how the new subject is in this clip, following this request from the person making the edit: "{brief}". Say where in the frame the new subject is (left, centre or right; near or far), its position or action (standing, sitting, walking in), who it is next to or touching, and when it comes into the clip if the request says so. This is an edit still to be made, so the new subject is NOT in these pictures yet: never say it is absent, unseen or not visible. Describe where it will be, as the request says, using what the pictures show of the room (the seat, the space beside the people) and say the side explicitly if the request does; call it "the new subject"."""
OBJECT_BRIEF_NOTE = """

The person making the edit describes the picture's subject as: "{brief}". Use their words for who or what it is where they fit what is visible, and do not compare it to any named character, film or style."""
_OBJ_HEAD = ("subject_definitions:\n<Subject 1> is the {details} shown in <Picture 1>.\n"
             "<Video 1> is the source video for the target video edit: {video_1}\n"
             "<Audio 1> is the synchronized audio track of <Video 1>, and is reused in the target video.\n\n")
_OBJ_TAIL = ("<Audio 1>: fully_copy - the original sound of <Video 1> is kept as it is.\n\n")
_OBJ_END = "\n\noverall_soundscape:\nThe original sound of <Video 1>.\n\nnon_diegetic_music:\nN/A"
OBJECT_FORMS = {
    "replace": (
        _OBJ_HEAD
        + "summary:\n[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. "
          "<Subject 1> replaces {the_target} at its exact place, size and perspective in every frame, while the people, the "
          "rest of the background, the camera and the action remain as recorded.\n\n"
          "retention_analysis:\n<Subject 1> (appears wherever {the_target} is in frame): attribute_transfer - {the_target} "
          "takes the look of <Subject 1> from <Picture 1>, keeping its place, size, angle and perspective in the frame.\n"
          "<Video 1> (whole-video temporal structure, people, camera, background, lighting): partially_preserved - preserve "
          "every person and their movements, everything around {the_target}, the camera, the framing, every cut and the "
          "lighting.\n"
        + _OBJ_TAIL
        + "detailed_description:\nThe target video keeps the camera, framing, lighting and colour of <Video 1>, with "
          "<Subject 1> in place of {the_target}. {placement} <Subject 1> is lit by the scene's own light and keeps the "
          "place, size and movement of {the_target} in every frame; everything else stays exactly as in <Video 1>."
        + _OBJ_END),
    "add": (
        _OBJ_HEAD
        + "summary:\n[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. "
          "<Subject 1> is added on {the_target} in every frame, at a natural size for the scene, while the people, the "
          "background, the camera and the action remain as recorded.\n\n"
          "retention_analysis:\n<Subject 1> (appears wherever {the_target} is in frame): attribute_transfer - <Subject 1> "
          "from <Picture 1> rests on {the_target} with its own look, at a natural size, and moves with it.\n"
          "<Video 1> (whole-video temporal structure, people, camera, background, lighting): partially_preserved - preserve "
          "every person, their look and their movements, the background, the camera, the framing, every cut and the "
          "lighting.\n"
        + _OBJ_TAIL
        + "detailed_description:\nThe target video keeps the camera, framing, lighting and colour of <Video 1>, with "
          "<Subject 1> added on {the_target}. {placement} <Subject 1> stays on {the_target} and moves with it in every "
          "frame, lit by the scene's own light; everything else stays exactly as in <Video 1>."
        + _OBJ_END),
}
OBJECT_FRAMES = 4

# the Qwen instruction of a zone add (the anchor frame: the new thing drawn into one frame), written by the large model from the
# anchor frame and the sheet, to Qwen-Image-2.1's own edit-rewriting rules (QwenLM/Qwen-Image-2.1, prompt_rewrite/prompts/
# system_prompt_edit.txt, read 2026-10-25): lead with the operation; refer to the pictures only as <image1> / <image2> and state each
# one's role; point to the reference for identity instead of describing features; edit only what is asked, push it to a clearly
# visible degree, one blanket preservation clause; observable properties only; one paragraph, affirmative, no hedging or ratios.
QWEN_ANCHOR_INSTRUCTION = """You write the instruction for Qwen-Image-2.1's image-editing model. It will draw a new subject into one frame of a video. The pictures that follow are <image1> (the frame of the video: the canvas) and <image2> (the reference picture of the new subject).

The job: add the person or thing shown in <image2> into <image1>, {where}.{contact}{size}{lightsrc}

Write ONE paragraph of about five plain sentences, with no line breaks, no JSON and no quotation marks around it, following Qwen's own rewriting rules:
- Lead with the operation ("Add the ... from <image2> into <image1> ..."), written as someone who sees only these pictures. Refer to the pictures only as <image1> and <image2>, never "image 1" or "the first picture".
- State each picture's role: <image1> is the canvas whose composition, room and people survive; <image2> is the source of the new subject's identity. Point to <image2> for how he or she looks (face, hair, costume), and do not describe those features in words.
- Edit only what is asked, and make the addition clearly visible: the whole subject inside the frame, a natural size for the room, feet or base resting where they would, cropped by the frame edge only where a real camera would crop. Hold everything else at input fidelity with ONE blanket clause naming what stays unchanged ({target}, the room, the furniture, the framing and the lighting).
- Lighting (the part that decides whether the new subject looks like it was filmed there): look at the people already in <image1> (or at <image3> when there is one), not at the furniture or the walls, and name the colour of the light on their skin, hair and clothes in plain words that say both the colour and the temperature, as one would tell a colourist (for example "warm orange light from the right with soft grey shadows", or "cool blue and purple light with deep dark shadows"; use the colours you actually see, not these). Also say how bright those people are (for example "dim, mostly in shadow, with light only on one side", or "evenly and brightly lit") and where the light falls on them. Then write one sentence saying the new subject is lit by the same light as those people, with the same colour tint on his or her skin and clothes (for example "a pale warm tint on his skin and jacket"), exactly as bright and as saturated as they are and no more (skin and clothes no brighter than theirs), with highlights on the same side and the same shadows where he or she touches the room. Do not call the room's own colours the light, and do not invent light you cannot see.
- Base every claim on what is visible in <image1> and leave out what you cannot see. No hedging, no alternatives, no ellipses, no resolution or aspect-ratio words; say what to do rather than what not to do.
Answer with the paragraph only."""

BACKGROUND_TEMPLATES = (BACKGROUND, BACKGROUND_2REF, BACKGROUND_LEAN, BACKGROUND_LEAN_2REF, LARGE_BG_TEMPLATE,
                        LARGE_BG2_TEMPLATE)
_BG_FORMS = {(False, False): BACKGROUND, (False, True): BACKGROUND_2REF,
             (True, False): BACKGROUND_LEAN, (True, True): BACKGROUND_LEAN_2REF}   # (lean, two references)
MOMENT_S = 0.5                  # timeline: one picture per this many seconds of each shot
# timeline (B4 render rounds 1-2, 400-608 against X10 at X10's settings, 2 seeds): pose IoU 0.733 / 0.729 against
# the hand prompt's 0.645 / 0.727 through the same nodes, mouth sync 0.67 / 0.69 against 0.26 / 0.14 (second
# half 0.59 / 0.61 against -0.34 / -0.16). The per-shot draft (T-DRAFT) tied on pose and lost lip sync (-0.07).
# lean became the default (2026-10-09): with the kept opponent and the relation sentence, wrestling c4 on one seed:
# lean 0.43 / timeline 0.40 pose (a hand-written lean 0.49), and a third shorter.
DEFAULT_TEMPLATE = LEAN_TEMPLATE

# ---------------------------------------------------------------------------
# the user's own templates (2026-10-29: save / delete from the node; for character, background and object jobs alike).
# A template is a LAYOUT for the draft: the drafter writes the prompt as it always does (its questions, its frames, the pipeline of
# the built-in template it is "built on"), and the layout puts the pieces where you want them and adds your own text around them.
# The pieces are the prompt's six sections, as {blanks}: {summary}, {subject_definitions}, {retention_analysis},
# {detailed_description}, {overall_soundscape}, {non_diegetic_music}, and {prompt} for the whole draft in one. Saved by name in one
# JSON file, listed in the dropdown as "my: <name>", deleted from the node.
# ---------------------------------------------------------------------------
USER_PREFIX = "my: "
SECTION_NAMES = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music")
USER_BLANKS = ("prompt",) + SECTION_NAMES
AUTO_BASE = "(automatic: the job's own template)"
USER_FORM_SKELETON = """subject_definitions:
{subject_definitions}

summary:
{summary}

retention_analysis:
{retention_analysis}

detailed_description:
{detailed_description}

overall_soundscape:
{overall_soundscape}

non_diegetic_music:
{non_diegetic_music}
"""
_SECTION_RE = re.compile(r"(?m)^(" + "|".join(SECTION_NAMES) + r"):[ \t]*$")
# what the preview shows in place of each blank
SAMPLE_SECTIONS = {n: f"[the drafter's {n.replace('_', ' ')} goes here]" for n in SECTION_NAMES}


def user_templates_path():
    try:
        import folder_paths
        base = os.path.join(folder_paths.get_user_directory(), "default", "SeamStitch")
    except Exception:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "draft_templates.json")


def load_user_templates():
    """{name: {"text": layout, "base": a built-in template name or ""}}; {} when there is no file or it can't be read."""
    try:
        with open(user_templates_path(), encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return {}
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        if isinstance(v, dict):
            out[str(k)] = {"text": str(v.get("text") or ""), "base": str(v.get("base") or "")}
        else:
            out[str(k)] = {"text": str(v), "base": ""}
    return out


def _write_user_templates(d):
    p = user_templates_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    os.replace(tmp, p)


def clean_template_name(name):
    n = " ".join((name or "").split())
    if n.lower().startswith(USER_PREFIX.strip().lower()):
        n = n[len(USER_PREFIX.strip()):].lstrip(": ").strip()
    if not n or len(n) > 60:
        raise DraftError("give the template a name of 1-60 characters")
    return n


def check_user_layout(text):
    """Why a layout can't be saved, or None. Every {word} must be one of the blanks (a typo like {sumary} would leave a hole in every
    prompt), and at least one blank must be there (or every chunk would get the same text)."""
    used = re.findall(r"\{([A-Za-z_][A-Za-z_0-9]*)\}", text or "")
    bad = sorted({u for u in used if u not in USER_BLANKS})
    ok = ", ".join("{" + b + "}" for b in USER_BLANKS)
    if bad:
        return "these are not blanks: " + ", ".join("{" + b + "}" for b in bad) + ". The blanks are " + ok
    if not used:
        return "the template has no blanks, so every chunk would get the same text. Add at least one of " + ok
    return None


def save_user_template(name, text, base=""):
    """Saves (or replaces) a template; returns the dropdown entry. base: the built-in template whose questions it drafts with ("" =
    the job's own)."""
    n = clean_template_name(name)
    text = (text or "").replace("\r\n", "\n").strip("\n")
    if not text.strip():
        raise DraftError("the template box is empty: nothing to save")
    why = check_user_layout(text)
    if why:
        raise DraftError(why)
    base = base if base in TEMPLATES else ""
    d = load_user_templates()
    d[n] = {"text": text + "\n", "base": base}
    _write_user_templates(d)
    return USER_PREFIX + n


def delete_user_template(entry):
    n = (entry or "")[len(USER_PREFIX):] if (entry or "").startswith(USER_PREFIX) else (entry or "")
    d = load_user_templates()
    if n not in d:
        raise DraftError(f"there is no saved template {n!r}")
    del d[n]
    _write_user_templates(d)
    return n


def template_choices():
    return list(TEMPLATES) + [USER_PREFIX + n for n in sorted(load_user_templates())]


def base_choices():
    return [AUTO_BASE] + list(TEMPLATES)


def is_user_template(template):
    return bool(template) and template.startswith(USER_PREFIX) and template[len(USER_PREFIX):] in load_user_templates()


def split_sections(text):
    """A prompt's six sections as {name: body}; a section the prompt doesn't have is missing."""
    ms = list(_SECTION_RE.finditer(text or ""))
    out = {}
    for n, m in enumerate(ms):
        end = ms[n + 1].start() if n + 1 < len(ms) else len(text)
        out[m.group(1)] = text[m.end():end].strip()
    return out


def apply_layout(layout, text):
    """The user's layout with each {blank} replaced by that piece of the drafted prompt `text`."""
    pieces = dict(split_sections(text), prompt=(text or "").strip())
    out = re.sub(r"\{(" + "|".join(USER_BLANKS) + r")\}", lambda m: pieces.get(m.group(1), ""), layout)
    return re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"


_EX_CHARACTER = ("<Subject 1> is the young woman whose motion comes from <Video 1> and whose appearance comes from <Picture 1>: "
                 "long pink hair, a white halo, a pale blue dress.")
_EX_VIDEO = "a woman sits on a brown couch in a dim room under a framed painting, filmed in a static medium shot."
_EX_SHOTS = ["[Shot 1] She sits and laughs, turns her head to the left and claps her hands."]


def sample_prompt(base):
    """An example of the prompt a built-in template drafts (stand-in answers, no model), so a layout can be tried against each kind
    before it is saved. Returns (text, note)."""
    name = base if base in TEMPLATES else TEMPLATES[3]
    note = "" if base in TEMPLATES else "automatic = the job's own template; shown here with the character one. "
    common = dict(video_1=_EX_VIDEO, shots=_EX_SHOTS, sounds="soft room hum", n_shots=1, dialogue=False, audio=True, pronoun="she")
    scene = scene_sentence("living room", "a brown couch, a dark textured wall and a framed landscape painting")
    try:
        if name == TEMPLATES[11]:
            return fill_bfs_body_swap("Male, slim build, green hair, burgundy suit", "the man in the white suit"), note
        if name == OBJECT_TEMPLATE:
            return fill_object("man with green hair and a burgundy suit", _EX_VIDEO, "She sits in the centre of the frame.", "the woman", sp.OBJECT_ADD,
                               people=[], addition="The new subject sits on the couch to her left."), note
        if name in BACKGROUND_TEMPLATES:
            if name == TEMPLATES[13]:
                return fill_bg2_ref2va_lean(subject=scene, whos=["the woman", "the man"], pronouns=["she", "he"], **{k: v for k, v in common.items() if k != "pronoun"}), note
            f = fill_bg_ref2va_lean if name in (BACKGROUND_LEAN, BACKGROUND_LEAN_2REF, LARGE_BG_TEMPLATE) else fill_bg_ref2va
            return f(subject=scene, who="the woman", **common), note
        f = fill_ref2va_lean if name in (LEAN_TEMPLATE, LEAN_2REF, LARGE_TEMPLATE) else fill_ref2va
        return f(subject=_EX_CHARACTER, who="the woman", **common), note
    except Exception as e:                                 # an example is a courtesy: never fail a preview over it
        return "\n\n".join(f"{n}:\n{SAMPLE_SECTIONS[n]}" for n in SECTION_NAMES), note + f"(no example could be built for this template: {e}) "


def preview_layout(layout, base=""):
    """(the base's example prompt, the layout applied to it, a note): what a layout gives for the chosen built-in template, so it can be
    read before it is saved."""
    why = check_user_layout(layout)
    if why:
        raise DraftError(why)
    example, note = sample_prompt(base)
    return example, apply_layout(layout, example), note


# the layout the run in progress applies to every chunk it writes (set by run_draft for a "my: ..." template, cleared after)
_LAYOUT = {"text": None}
# B4 workshop C2: parts of a long shot (4 s) and a 6-frame scene call made the shots wordier and more invented,
# and video_1 no better, than whole shots (C1). Both kept as options, off by default.
SEGMENT_S = 0.0                 # per shot: > 0 describes a longer shot in parts of at most this many seconds
SCENE_FRAMES = 0                # per shot: > 0 = video_1 and sounds from this many frames (0: the chunk's frames)
DEFAULT_FRAMES = 24             # spread over the shots (B4 workshop: 24 named the foam sheet and the booklet, 16 didn't)
WHISPER_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
WHISPER_ALONE_P = 0.5           # Whisper's words with Omni hearing no speech: kept only at this mean confidence or more
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


GGUF_PREFIX = "GGUF: "          # qwen_model entries run through QwenVL's GGUF (llama.cpp) side, not its HF loader


def locate_qwen_gguf():
    """AILab_QwenVL_GGUF's module (its catalog, loader and image helpers), or None. locate_qwen() puts the
    pack's py/ folder on sys.path first."""
    cls = _registered("AILab_QwenVL_GGUF")
    if cls is not None:
        return sys.modules.get(cls.__module__)
    if locate_qwen() is None:
        return None
    try:
        return importlib.import_module("AILab_QwenVL_GGUF")
    except Exception as e:
        _say(f"[SeamStitch] Swap Draft: QwenVL's GGUF side didn't import: {e}")
        return None


def gguf_model_file(mod, name):
    """The GGUF catalog entry's model and mmproj files on disk, or (None, None). Never downloads."""
    try:
        r = mod._resolve_model_entry(name)
        d = mod.resolve_base_dir(mod.GGUF_VL_CATALOG.get("base_dir") or "LLM/GGUF") \
            / mod.safe_dirname(r.author or "") / mod.safe_dirname(r.repo_dirname)
        model = mod.find_local_gguf_file(r.model_filename, d)
        mmproj = mod.find_local_gguf_file(r.mmproj_filename, d, allow_recursive=False) if r.mmproj_filename else None
        return (model, mmproj) if model and os.path.isfile(model) else (None, None)
    except Exception:
        return None, None


def gguf_models():
    """The GGUF catalog's vision models that are on disk (a model with its mmproj), as qwen_model entries."""
    mod = locate_qwen_gguf()
    if mod is None:
        return []
    try:
        names = sorted((mod._load_gguf_vl_catalog().get("models") or {}).keys())
    except Exception:
        return []
    return [GGUF_PREFIX + n for n in names if all(gguf_model_file(mod, n))]


def qwen_models():
    """The QwenVL model list (its HF VL models, then the GGUF vision models on disk), for the qwen_model widget."""
    cls = locate_qwen()
    mod = sys.modules.get(getattr(cls, "__module__", ""), None) if cls else None
    try:
        mod.load_model_configs()
        names = list(mod.HF_VL_MODELS.keys())
    except Exception:
        names = []
    return (names or [DEFAULT_QWEN]) + gguf_models()


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
appearance: ONE sentence, a comma-separated list from head to toe that names every visible detail, in this order: apparent age and build; face and make-up (skin, cheeks, nose, lips, eyes; name make-up and face paint as make-up, for example "white face paint" or "black eye make-up", never as skin colour or as shadows or circles under the eyes); hair (colour, length, style, fringe); anything on or above the head, with its colours; anything at the neck; the top or dress (colour, neckline, sleeves, trims, buttons, bows, belts); the skirt or trousers (layers, colours); gloves or arm pieces (length, colours, cuffs); legwear; footwear; anything attached to the body (for example wings or a tail). Start with "a young woman", "an old man", "a boy", or similar.
pronoun: she, he or they.

Rules: describe only the character's own body, clothes and what is worn on the body. Never name an object held in the hands or lying near the character: no props. Concrete visual detail only (colour, material, shape, pattern), no mood, personality or story."""
# 2026-10-17, test clip c14: a cosplay sheet's white face paint and black eye make-up came back as "pale skin with
# heavy dark circles around the eyes", which fits the source actor as he is; H3 kept him on every seed and model, and
# the same chunk swapped with the line that said "make-up" (the sheet's picture unchanged)
_SUBJECT_BEFORE_MAKEUP = SUBJECT_INSTRUCTION.replace(
    '; name make-up and face paint as make-up, for example "white face paint" or "black eye make-up", never as skin '
    'colour or as shadows or circles under the eyes', "")

SUBJECT_2_INSTRUCTION = """This picture is a second reference of the same character: a close-up of the face, or another view. Write exactly one labelled line and nothing else:

detail: ONE sentence saying what this picture shows of the character that a full-length view shows less clearly: the face (skin, eyes, lips), the hair, or a close look at the costume or an accessory, with colours, materials and shapes.

Rules: describe only the character's own face, body and clothes. Never name an object held in the hands or lying near the character. Concrete visual detail only (colour, material, shape, pattern), no mood or story."""

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
- {Subj} is wherever the person is in the frame, as the pictures show it, and looks where the person looks.
- {Poss} hands hold only what the person's hands hold.
- Don't describe {poss} appearance or costume, and don't mention pictures, frames, times, "the person" or the replacement.{extra}"""

# One call per shot (template "per shot"): each shot described from its own frames only, then one call
# over the whole chunk for video_1 and sounds.
SHOT_INSTRUCTION = """You are shown {n} pictures, frames sampled in order from one continuous shot ({secs:.1f} seconds) of a video. In the edit, the person in it is replaced by a character, called "{subj}" below, who does exactly what the person does.{part}
{objects_line}{lines_block}
Write 1 to 5 sentences, present tense, describing only what these pictures show {subj} doing, in the order it happens from the first picture to the last (the last pictures matter as much as the first): how {subj} sits or stands and leans, where {subj} looks, what each hand does, which object {subj} picks up, holds up, shows or puts down, and where {subj} holds it (for example close to the face). {dialogue_rule}

Rules: name an object only if these pictures show it, by what it looks like (colour, shape, size); never quote printed text, logos or brand names; {subj} is wherever the person is in the frame, as these pictures show it, and looks where the person looks; {poss} hands hold only what the person's hands hold; don't describe {poss} appearance or costume; don't mention pictures, frames, times, "the person" or the replacement.{extra} Output only the sentences."""

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

# with someone else in the clip (the scene question named them), the moment question gets a seventh line: where the
# main person and the others touch (2026-10-08: partner dances, combat). The main person is "the person" and the
# others go by how they look, so the node turns only "the person" into her pronoun (the pronoun swap would make the
# partner's "his hand" hers)
MOMENT_TOUCH_LINE = """touch: where the main person and anyone else in the frame touch each other, as one phrase that calls the main person "the person" and the other by how they look (for example "the person's right hand holds the left hand of the man in the suit" or "the hand of the man in the suit rests on the person's back"), or "none" if they don't touch or no one else is in the frame."""


def moment_instruction(who, touch=False):
    """The moment question; with touch, its "touch" line goes in after "others" (six labelled lines become seven)."""
    q = MOMENT_INSTRUCTION.format(who=who)
    if not touch:
        return q
    q = q.replace("Answer in exactly six short labelled lines", "Answer in exactly seven short labelled lines")
    return q.replace("\nframe: ", "\n" + MOMENT_TOUCH_LINE + "\nframe: ")


# once per shot with someone else in it (2026-10-09): the hand prompts that beat the drafts said how the body sits
# against the other person ("lies across her, pinning her to the mat"); the moment captions only give a still pose
RELATION_INSTRUCTION = """These are {n} frames from one continuous shot of a video, in order. The main person is {who}; the other person is {other}. In ONE sentence, say how the main person's body is placed against the other person through the shot (for example lying across them, kneeling over them, beside them, behind them, facing them, holding them), calling the main person "the person" and the other "the other person". Answer only "none" if they stay apart and don't touch. Never name anyone."""

# the large-model template: one call per shot. The pictures come in order, each after its time (llama.cpp gives a
# GGUF model separate pictures with no clock of their own); the cast comes by label, so the model says who is beside
# whom instead of the code guessing it; the answer is JSON (a 27B keeps the keys; the code reads what it needs)
LARGE_SHOT_INSTRUCTION = """This is one continuous shot ({secs:.1f} seconds) from a film clip. In an edit of the clip, one person in it is replaced by a new character; everyone else stays exactly as filmed. The pictures below are from this shot, in time order, each after its time in the shot.

The person who is replaced: {who}. Below this person is called "{subj}".
The other people in the clip, by label (use only these labels for them):
{cast}
{objects}{lines}
Look at every picture before you answer: a pose that changes partway through the shot matters as much as the first picture. Then answer with one JSON object and nothing else:

{{"main_visible": true or false (is {who} in any of the pictures, even only partly: the hands, an arm, the back of the head),
"frame": "what the shot shows, in a few words",
"others_visible": [the labels of the other people visible in any picture of this shot: check each listed person against every picture],
"beside": "who is directly beside {subj} and on which side of the frame, by label, or none",
"description": "2 to 4 sentences, present tense, about {subj} only, in the order things happen: how {subj} sits, stands, slumps or leans (and against what or whom), each change of pose and roughly when it comes (at the start, then, by the end), where {subj} looks, and what each hand does and holds (an object by its colour, shape and size; hands out of view get no sentence, and a hand holds only what you can see in it)"}}

Rules for "description": start with "{Subj}"; call {subj} only "{subj}" and the other people only by their labels; {dialogue_rule}describe only what the pictures show; give each object one name and keep it (an object you can't make out is named by its colour and shape, never guessed); keep each sentence under 30 words; don't describe {poss} face, hair or clothes; don't name anyone or quote printed text; don't mention pictures, frames, times, "the person" or the replacement.{extra}"""


# the large background template: the kept person is <Subject 2> from the first word (no relabelling afterwards), and
# the rules are the background lean draft's (no backdrop, no shot size, only what happens)
LARGE_BG_SHOT_INSTRUCTION = """This is one continuous shot ({secs:.1f} seconds) from a video. In an edit of the video, the person in it, {who}, is kept exactly as filmed and only the background behind them is replaced by a new place. Below, this person is called "<Subject 2>". The pictures below are from this shot, in time order, each after its time in the shot.
{objects}{lines}
Look at every picture before you answer: a movement partway through the shot matters as much as the first picture. Then answer with one JSON object and nothing else:

{{"main_visible": true or false (is <Subject 2> in any of the pictures, even only partly),
"frame": "what the shot shows, in a few words",
"description": "1 to 4 sentences, present tense, about <Subject 2> only, in the order things happen: when <Subject 2> enters or leaves the frame and from which side, where in the frame <Subject 2> is, which way <Subject 2> faces and moves, how <Subject 2> sits, stands or walks, what each hand does and holds (an object by its colour, shape and size; hands out of view get no sentence), and where <Subject 2> looks"}}

Rules for "description": start with "<Subject 2>", and after that call <Subject 2> "{subj}" or "<Subject 2>"; {dialogue_rule}describe only what the pictures show; never describe the backdrop, the walls, the floor or the light behind <Subject 2>: they are replaced; what <Subject 2> sits, lies or leans on is replaced too, so name it without its colour, pattern or material ("the sofa", "a cushion"); don't say how close the shot is or how much of <Subject 2> shows (no close-up, medium shot, wide shot, full view, waist up); write only what happens (nothing like "no one enters" or "no significant motion"); don't name anyone or quote printed text; don't mention pictures, frames, times or the replacement.{extra}"""

# the clip's camera, pronoun and sounds, from one picture per shot (the background lean draft's own wording for each)
LARGE_BG_CLIP_INSTRUCTION = """These are {n} pictures from one clip of a video, one from each shot, in order{talk}. In an edit of the clip, {who} is kept exactly as filmed and only the background behind them is replaced by a new place. Write exactly three labelled lines and nothing else:

pronoun: she, he or they, for {who}.
video_1: ONE sentence about the clip as a whole: the camera's angle and height, whether the camera moves at all (and how, if it does), where in the frame {who} is, and where {who} moves through the frame (enters, crosses, leaves). Don't say how close the shot is or how much of {who} shows (no close-up, medium shot, wide shot, full view, waist up). Don't describe the backdrop, the wall or the floor: they are replaced.
sounds: ONE sentence: the sounds through the clip, in order{sounds_hint}.

Say only what is there: leave out anything the lines ask about that the pictures don't show."""


# the two-character large background template (LARGE_BG2_TEMPLATE): who the two people are, then the clip, then each shot for both.
# The first person is <Subject 2>, the second <Subject 3>, from the first word of every answer.
WHO2_INSTRUCTION = """These are {n} frames from one video clip. Who are the TWO main people or characters in it: the two most in focus, the largest, or doing the most? Prefer people or human-like characters over animals or creatures. Never name them, even when a name is written on screen. Answer with exactly two short descriptions separated by a semicolon, each telling that person apart from the other by how they look (build, hair, clothes), never by what they hold or do, which changes from shot to shot, and the first one the person who is further to the left in the frames; for example "a man in a black t-shirt; a woman in a red dress". Answer "none" only if fewer than two people or characters appear in the frames."""

LARGE_BG2_CLIP_INSTRUCTION = """These are {n} pictures from one clip of a video, one from each shot, in order{talk}. In an edit of the clip, two people, {who2} and {who3}, are kept exactly as filmed and only the background behind them is replaced by a new place. Write exactly three labelled lines and nothing else:

pronoun: two words, one for {who2} and then one for {who3}, each she, he or they (for example "he she").
video_1: ONE sentence about the clip as a whole: the camera's angle and height, whether the camera moves at all (and how, if it does), where in the frame each of the two people is, and where each moves through the frame (enters, crosses, leaves). Don't say how close the shot is or how much of them shows (no close-up, medium shot, wide shot, full view, waist up). Don't describe the backdrop, the wall or the floor: they are replaced.
sounds: ONE sentence: the sounds through the clip, in order{sounds_hint}.

Say only what is there: leave out anything the lines ask about that the pictures don't show."""

LARGE_BG2_SHOT_INSTRUCTION = """This is one continuous shot ({secs:.1f} seconds) from a video. In an edit of the video, the two people in it are kept exactly as filmed and only the background behind them is replaced by a new place. Below, the first person, {who2}, is called "<Subject 2>" and the second person, {who3}, is called "<Subject 3>". The pictures below are from this shot, in time order, each after its time in the shot.
{objects}{lines}
Look at every picture before you answer: a movement partway through the shot matters as much as the first picture. Then answer with one JSON object and nothing else:

{{"visible_2": true or false (is <Subject 2> in any of the pictures, even only partly),
"visible_3": true or false (is <Subject 3> in any of the pictures, even only partly),
"frame": "what the shot shows, in a few words",
"description_2": "1 to 3 sentences, present tense, about <Subject 2> only, in the order things happen: when <Subject 2> enters or leaves the frame and from which side, where in the frame <Subject 2> is, which way <Subject 2> faces and moves, how <Subject 2> sits, stands or walks, what each hand does and holds (an object by its colour, shape and size; hands out of view get no sentence), and where <Subject 2> looks",
"description_3": "the same for <Subject 3>, about <Subject 3> only",
"together": "ONE sentence on how <Subject 2> and <Subject 3> are placed against each other through the shot (for example side by side with <Subject 2> on the left, facing each other, one behind the other, touching, one holding the other), or \"none\" if they stay apart and don't touch"}}

Rules for the descriptions: start each with its own label, "<Subject 2>" or "<Subject 3>", and call each person only by that label; {dialogue_rule}describe only what the pictures show; never describe the backdrop, the walls, the floor or the light behind them: they are replaced; what they sit, lie or lean on is replaced too, so name it without its colour, pattern or material ("the sofa", "a cushion"); don't say how close the shot is or how much of them shows (no close-up, medium shot, wide shot, full view, waist up); write only what happens (nothing like "no one enters" or "no significant motion"); don't name anyone or quote printed text; don't mention pictures, frames, times or the replacement.{extra}"""


def large_cast(labels, people):
    """The shot call's cast lines: "- <Subject 2>: a man in a tan jacket", or a line saying there is no one else."""
    return "\n".join(f"- {lab}: {desc}" for lab, desc in zip(labels, people)) or "- none: no one else takes part"


def large_objects(objects):
    """The shot call's object names: the chunk's object list, so a shot doesn't rename the same thing (test clip
    c5: one ticket strip was "a card", "a rod", "a scroll" and "a cigar" in four shots)."""
    names = [o for o in (objects or []) if o]
    return ("Objects seen in this clip, by the names to use for them: " + ", ".join(names) + ".\n") if names else ""


def large_lines(lines_rel, fps):
    """The shot call's spoken lines, with their times in the shot."""
    if not lines_rel:
        return ""
    return "Words spoken in this shot: " + "; ".join(_line_where(ln, fps) for ln in lines_rel) + "\n"


def parse_large_shot(text):
    """The shot call's JSON object -> dict ({} if none parses). A reply without JSON is taken as the description."""
    t = (text or "").strip()
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        try:
            d = json.loads(t[a:b + 1])
            if isinstance(d, dict):
                return d
        except ValueError:
            pass
    return {"main_visible": True, "description": t} if t else {}


def large_visible(value):
    return str(value).strip().lower() not in ("false", "no", "0", "none", "")


# when the scene named no one but the moments keep seeing someone (others_from_moments): how they look, so the prompt
# keeps "a woman in a blue singlet" rather than "the opponent"
OTHER_LOOK_INSTRUCTION = """These are {n} frames from one video clip. Besides the main person ({who}), there is {other}. Describe {other} in a few words by how they look: clothes, colours and hair (for example "a woman in a blue singlet" or "a man in a grey suit"). Never name anyone. Answer only with the description."""

# asked only when a shot would be written as empty although a moment describes a pose or hands (below)
RECHECK_INSTRUCTION = """This is one frame from a video. Is {who} in this frame at all, even partly: only the top of the head, the back, a shoulder, an arm or the hands, seen from above, from behind or turned away? Answer only "yes" or "no"."""

WHO_INSTRUCTION = """These are {n} frames from one video clip. Who is the main person or character in it: the one most in focus, the largest, or the one doing the most? Prefer a person or a human-like character over an animal or a creature. Never name them, even when a name is written on screen. Answer in a few words that tell them apart from anyone else in the clip by how they look (build, hair, clothes), never by what they hold or do, which changes from shot to shot; for example "a man in a black t-shirt", "a girl in a blue dress" or "a cartoon boy in a white hat". Answer "none" only if no person or character appears in any of the frames."""

SCENE_INSTRUCTION = """These are {n} frames from one clip of a video{talk}. The main person is {who}. Write exactly four labelled lines and nothing else:

video_1: ONE sentence about the main person, not anyone else: where the main person is and what they sit or stand on, what is in front of them, the background and its colour (and what is in it: a crowd or audience, seating, screens, a scoreboard, banners, and any caption or logo laid over the picture, said as "a caption" or "a logo" without its words), the light (hard or soft, and the side it comes from, judged by where the shadows fall), the visual style (for example live-action phone video, or 2D cartoon animation), what the main person is doing, and the camera: whether it looks down on them, is level with them or looks up at them, whether it moves, and where their head sits in the frame. Don't describe their face, hair or clothes.
sounds: ONE sentence: the sounds through the clip, in order{sounds_hint}.
background: what is behind the people, as a short list separated by semicolons (for example "a crowd in tiered seating; an electronic scoreboard; a caption"), a caption or logo without its words, or "none".
others: every other person who takes part in what happens (not a crowd or spectators in the background, and not a screen, sign, scoreboard or object), each in a few words by how they look and which side of the frame they are on, saying left, centre or right (for example "a man in a striped jacket on the right"), separated by semicolons; or "none".

Never name anyone, even when a name is written on screen: describe people by how they look."""

# the scene's "others" are kept as <Subject N> people: a crowd or a scoreboard isn't one (wrestling c4 kept
# "scoreboard visible upper-left" and "blurred audience members" as subjects)
NOT_PEOPLE = re.compile(r"(?i)\b(audience|crowd|spectators|onlookers|scoreboard|banner|logo|caption|on-screen|graphic)\b")

PRONOUNS = {"she": {"subj": "she", "Subj": "She", "poss": "her", "Poss": "Her", "does": "does", "moves": "moves",
                    "poses": "poses", "stays": "stays", "handles": "handles"},
            "he": {"subj": "he", "Subj": "He", "poss": "his", "Poss": "His", "does": "does", "moves": "moves",
                   "poses": "poses", "stays": "stays", "handles": "handles"},
            "they": {"subj": "they", "Subj": "They", "poss": "their", "Poss": "Their", "does": "do", "moves": "move",
                     "poses": "pose", "stays": "stay", "handles": "handle"}}


# background replace: Picture 1 is a place. The subject is the environment, in H3's reference guide's own words
# ("<Subject N> is the coffee-shop environment in <Picture N>, featuring ..."); the person in the clip is kept.
SCENE_SHEET_INSTRUCTION = """This picture shows a place: a set, a room or an outdoor scene. Write exactly two labelled lines and nothing else:

name: a short name for the place, 2-4 words, lower case (for example "coffee-shop" or "ship-themed diner").
features: ONE sentence, a comma-separated list of what the place is made of, from back to front: the walls (colours, stripes, panels), windows and doors, fixtures and fittings, furniture and counters, signs and decorations, and the light (its colour and where it comes from).

Rules: concrete visual detail only (colour, material, shape, pattern), no mood or story. Never quote printed text, logos or brand names. Don't describe any person."""

# what the second picture is, read off it, not assumed (2026-10-13: the test job's second picture was the same room from the same
# view; the old question asked for "the surface or object seen close up", so Qwen described a close-up that wasn't
# there, and the prompts called it one)
SCENE_SHEET_2_INSTRUCTION = """This picture is a second reference of the same place as the first reference. Write exactly two labelled lines and nothing else:

view: "closer" if it shows part of the place close up (a surface, a piece of furniture, a fitting), "angle" if it shows the place from another side or angle, or "same" if it shows the place from about the same view as a whole.
detail: ONE sentence saying what this picture shows of the place: for a closer view, the surface or object seen close up, its colours, materials and pattern, and the light on it; for another angle, what that side of the place has; for the same view, the walls, furniture, fittings and light it shows most clearly.

Rules: concrete visual detail only (colour, material, shape, pattern), no mood or story. Never quote printed text, logos or brand names. Don't describe any person."""

BG_DRAFT_INSTRUCTION = """You are writing part of a prompt for a video-edit model. You are shown {n} pictures, frames sampled in order from one clip of a source video ({secs:.1f} seconds). In the edit, the person in the clip, called "{who}" below, is kept exactly as they are: only the background behind them is replaced by a new place. In some pictures {who} may be only partly in frame or not in frame at all.

The clip has {k} shot{k_s}. Shot by shot, the pictures that show it and the words spoken in it:
{shot_lines}
Write exactly four labelled parts and nothing else.

pronoun: she, he or they, for {who}.

video_1: ONE sentence about the source clip as a whole: the camera's angle and height, whether the camera moves at all (and how, if it does), where in the frame {who} is, and where {who} moves through the frame (enters, crosses, leaves). Don't say how close the shot is or how much of {who} shows (no close-up, medium shot, wide shot, full view, waist up). Don't describe the backdrop, the wall or the floor: they are replaced.

shots: {k} block{k_s}, one per shot, in order, each on its own line starting with "[Shot n]". For each shot describe, in the order it happens, only what its own pictures show: when {who} enters or leaves the frame and from which side, where in the frame {who} is, which way {who} faces and walks, what each hand does, where {who} looks, and the stretches where {who} is out of frame. {dialogue_rule}Present tense, 1 to 5 sentences per shot, each sentence about {who} by name.

sounds: ONE sentence: the sounds through the clip, in order{sounds_hint}.

Rules:
- Describe only what the pictures show. Never quote printed text, logos or brand names.
- Never describe the backdrop, the wall, the floor or the light behind {who}: the new place replaces them.
- Write only what happens: leave out what doesn't happen (nothing like "no one enters" or "no significant motion").
- Don't mention pictures, images, frames, times or the replacement.{extra}"""


# ---------------------------------------------------------------------------
# Qwen's questions, editable from the node (2026-10-09: "a text box ... that shows what the qwen prompt is,
# so I can see it and tweak it"). One "### name" section per question; the node's default is the built-in text.
# ---------------------------------------------------------------------------
QUESTIONS = (("subject", "SUBJECT_INSTRUCTION", "the character sheet -> the subject line (once per job)"),
             ("subject_2", "SUBJECT_2_INSTRUCTION", "the 2 references template: the second picture of the character "
                                                    "(once per job)"),
             ("objects", "OBJECTS_INSTRUCTION", "what the person handles in the chunk"),
             ("who", "WHO_INSTRUCTION", "timeline: who the main person is"),
             ("scene", "SCENE_INSTRUCTION", "timeline: the clip, the camera, the others, the sounds"),
             ("moment", "MOMENT_INSTRUCTION", "timeline: one picture every half second"),
             ("touch", "MOMENT_TOUCH_LINE", "timeline: the moment question's extra line when someone else is in the clip "
                                            "(it goes in before the moment question's \"frame:\" line)"),
             ("recheck", "RECHECK_INSTRUCTION", "timeline: is the person there at all"),
             ("relation", "RELATION_INSTRUCTION", "timeline: how her body sits against the other person, once a shot"),
             ("other_look", "OTHER_LOOK_INSTRUCTION", "timeline: how the person the scene missed looks"),
             ("large_shot", "LARGE_SHOT_INSTRUCTION", "the large-model template: each shot in one call"),
             ("large_bg_shot", "LARGE_BG_SHOT_INSTRUCTION", "the large-model background template: each shot of the "
                                                            "kept person in one call"),
             ("large_bg_clip", "LARGE_BG_CLIP_INSTRUCTION", "the large-model background template: the camera, "
                                                            "pronoun and sounds"),
             ("who2", "WHO2_INSTRUCTION", "the 2 characters background template: who the two people are"),
             ("large_bg2_clip", "LARGE_BG2_CLIP_INSTRUCTION", "the 2 characters background template: the camera, "
                                                              "pronouns and sounds"),
             ("large_bg2_shot", "LARGE_BG2_SHOT_INSTRUCTION", "the 2 characters background template: each shot, "
                                                              "both people"),
             ("one_call", "DRAFT_INSTRUCTION", "the one-call template"),
             ("per_shot", "SHOT_INSTRUCTION", "the per-shot template: each shot"),
             ("per_shot_chunk", "CHUNK_INSTRUCTION", "the per-shot template: the clip and its sounds"),
             ("bg_place", "SCENE_SHEET_INSTRUCTION", "background replace: the place on the sheet"),
             ("bg_place_2", "SCENE_SHEET_2_INSTRUCTION", "background replace with 2 references: the second picture "
                                                         "of the place (once per job)"),
             ("bg_draft", "BG_DRAFT_INSTRUCTION", "background replace: the chunk"))
QUESTIONS_HEAD = ("Qwen's questions, one per ### section. Edit a section to change that question for every draft "
                  "run of this node; delete a section, or clear the whole box, to use the built-in one. Keep the "
                  "{names in braces}: the node fills them in (an edit with a name the built-in doesn't have is "
                  "ignored, with a warning). Text above the first ### is ignored.")
_BUILTIN_QUESTIONS = {}
# earlier built-in texts: a workflow saved with one of these in its box hasn't edited that question, so it gets
# today's built-in (2026-10-09: SCENE_INSTRUCTION's video_1 now asks for the crowd, screens and overlays)
_SCENE_3_LINES = SCENE_INSTRUCTION.replace("Write exactly four labelled lines", "Write exactly three labelled lines").replace(
    'background: what is behind the people, as a short list separated by semicolons (for example "a crowd in tiered seating; an electronic scoreboard; a caption"), a caption or logo without its words, or "none".\n', "")
OLD_SCENE_SHEET_2 = "This picture is a second view of the same place as the first reference: a closer look at a surface, another angle, or a detail. Write exactly one labelled line and nothing else:\n\ndetail: ONE sentence saying what this picture shows that a description of the whole place would miss: the surface or object seen close up (for example the weave of a fabric, a wall's texture, a pattern, a fitting), its colours, materials and pattern, and the light on it.\n\nRules: concrete visual detail only (colour, material, shape, pattern), no mood or story. Never quote printed text, logos or brand names. Don't describe any person."
_LEGACY_QUESTIONS = {"SUBJECT_INSTRUCTION": [_SUBJECT_BEFORE_MAKEUP],
                     "SCENE_INSTRUCTION": [_SCENE_3_LINES, _SCENE_3_LINES.replace(
    "the background and its colour (and what is in it: a crowd or audience, seating, screens, a scoreboard, banners, "
    "and any caption or logo laid over the picture, said as \"a caption\" or \"a logo\" without its words)",
    "the background and its colour")],
    # 2026-10-13 template review: the "never moves to the centre" rule (a centred person was written off-centre), and
    # the background question's framing words and "only what happens"
    "DRAFT_INSTRUCTION": [DRAFT_INSTRUCTION.replace(
        "- {Subj} is wherever the person is in the frame, as the pictures show it, and looks where the person looks.\n",
        "- {Subj} never moves to the centre of the frame and never poses for the camera.\n")],
    "SHOT_INSTRUCTION": [SHOT_INSTRUCTION.replace(
        "{subj} is wherever the person is in the frame, as these pictures show it, and looks where the person looks; ",
        "{subj} never moves to the centre of the frame and never poses for the camera; ")],
    "SCENE_SHEET_2_INSTRUCTION": [OLD_SCENE_SHEET_2],
    "BG_DRAFT_INSTRUCTION": [BG_DRAFT_INSTRUCTION.replace(
        "where in the frame {who} is, and where",
        "how much of {who} shows (for example the whole figure, or from the waist up) and where in the frame {who} is, "
        "and where").replace(
        "Don't say how close the shot is or how much of {who} shows (no close-up, medium shot, wide shot, full view, "
        "waist up).", "Don't name a shot size (close-up, medium shot, wide shot)."),
        BG_DRAFT_INSTRUCTION.replace(
        "the camera's angle and height, whether the camera moves at all (and how, if it does), where in the frame "
        "{who} is, and where",
        "the camera's angle, height and movement, how {who} is framed, and where").replace(
        " Don't say how close the shot is or how much of {who} shows (no close-up, medium shot, wide shot, full view, "
        "waist up).", "").replace(
        "- Write only what happens: leave out what doesn't happen (nothing like \"no one enters\" or \"no significant "
        "motion\").\n", "").replace("pictures, images, frames", "pictures, frames")]}


def questions_text():
    """The box's default: every built-in question under its ### heading."""
    g = globals()
    parts = [QUESTIONS_HEAD]
    for name, const, what in QUESTIONS:
        parts.append(f"### {name} ({what})\n{_BUILTIN_QUESTIONS.get(const, g[const])}")
    return "\n\n".join(parts) + "\n"


def _fields(s):
    import string
    return {f.split(".")[0].split("[")[0] for _, f, _, _ in string.Formatter().parse(s) if f}


def question_overrides(text):
    """The box -> ({constant: text} for the sections that differ from the built-in, [warnings])."""
    if not (text or "").strip():
        return {}, []
    by_name = {name: const for name, const, _ in QUESTIONS}
    out, warn, cur, buf = {}, [], None, []

    def close():
        if cur is None:
            return
        body = "\n".join(buf).strip("\n")
        const = by_name.get(cur)
        if const is None:
            warn.append(f"Qwen's questions: unknown section ### {cur}: ignored")
            return
        base = _BUILTIN_QUESTIONS.get(const, globals()[const])
        if body.strip() == base.strip() or body.strip() in (x.strip() for x in _LEGACY_QUESTIONS.get(const, ())):
            return
        if not body.strip():
            return
        try:
            extra = _fields(body) - _fields(base)
        except ValueError as e:
            warn.append(f"Qwen's questions: ### {cur} has a stray brace ({e}): the built-in one is used")
            return
        if extra:
            warn.append(f"Qwen's questions: ### {cur} uses {', '.join('{' + x + '}' for x in sorted(extra))}, which the "
                        f"node doesn't fill: the built-in one is used")
            return
        out[const] = body

    for line in text.splitlines():
        if line.startswith("### "):
            close()
            cur, buf = line[4:].split(" (", 1)[0].strip(), []
        elif cur is not None:
            buf.append(line)
    close()
    return out, warn


class edited_questions:
    """Within: the module's question constants are the box's edits; restored after (one draft run at a time)."""

    def __init__(self, overrides):
        self.overrides = overrides

    def __enter__(self):
        g = globals()
        for const, body in self.overrides.items():
            _BUILTIN_QUESTIONS.setdefault(const, g[const])
            g[const] = body
        return self

    def __exit__(self, *a):
        g = globals()
        for const in self.overrides:
            g[const] = _BUILTIN_QUESTIONS[const]
        return False


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
    for lab in ("person", "pose", "look", "hands", "others", "touch", "frame"):
        m = re.search(rf"(?im)^[ \t*-]*{lab}[ \t*]*:[ \t]*(.+)$", text or "")
        d[lab] = " ".join(m.group(1).split()).strip(" .*") if m else ""
    if not any(d.values()):
        d["hands"] = " ".join((text or "").split()).strip(" .")
    for lab in ("pose", "look", "hands", "touch"):          # "none", "n/a": nothing to say
        if d[lab].lower().strip(" .") in ("none", "n/a", "na", "-", "not visible", "unknown"):
            d[lab] = ""
    return d


def present(m):
    """Whether the main person is in a moment's frame (no answer: assume yes)."""
    v = (m.get("person") or "").lower()
    return not v.startswith("no")


_ACTION = re.compile(r"(?i)(?:,\s*|\s+)(?:who is\s+|who's\s+|that is\s+)?\b(?!wearing\b)[a-z]+ing\b.*$")


def who_lasting(who):
    """The main person by what lasts: Qwen's "a bald man holding a knife" (the test clip's chunk 1, B5a) made every later
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
    """Qwen can answer person "no" while describing that person's pose and hands: the test clip's frames 186-208, the man bent over
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


# placement, from the chunk's cached source mask (2026-10-08, the green-screen walk-past c6: Qwen's draft said only
# "walking sideways", the test character posed full-body centre-frame, dx 0.284; the same chunk with timed position / size / exit
# beats and a framing lock followed him unpinned, dx 0.035, pose 0.629). Where the person is, how big and how
# framed, and when they leave and come back is what H3 couldn't read off the guide on its own.
PLACE_MIN_COVER = 0.004          # less of the frame than this: out of frame
PLACE_EDGE_SHARE = 0.5           # touching an edge with under this share of the chunk's usual size: going out there
PLACE_BACK_SHARE = 0.75          # ... and back in only once over this share (c6's sliver at the edge hovered at 0.5)
PLACE_MAX_BEATS = 8
PLACE_SLOW, PLACE_FAST = 0.10, 0.35   # pace, in frame widths a second (c6's walk-past: ~0.25)


def mask_place(m):
    """One mask frame (h x w, 0-1) -> where the person is: {x centre, cover, left/right edge touched, top, bottom,
    height} as shares of the frame, or None when they're out of it."""
    b = np.asarray(m) > 0.5
    h, w = b.shape
    cover = float(b.mean())
    if cover < PLACE_MIN_COVER:
        return None
    xs = np.nonzero(b.any(axis=0))[0]
    ys = np.nonzero(b.any(axis=1))[0]
    return {"x": (xs[0] + xs[-1] + 1) / 2.0 / w, "cover": cover, "left": xs[0] <= 0.01 * w,
            "right": xs[-1] >= 0.99 * w - 1, "top": ys[0] / float(h), "bottom": (ys[-1] + 1) / float(h),
            "height": (ys[-1] - ys[0] + 1) / float(h)}


def place_away(p, ref_cover):
    """Whether a mask frame shows the person out of shot: gone, or only a sliver past an edge. Qwen can answer
    "person: yes" for that sliver and describe gestures nobody sees (c6, 3.4 s: "gestures with both hands near
    chest"), so those moments' pose and hands lines are dropped (the placement beat says where they are)."""
    return p is None or ((p["left"] or p["right"]) and p["cover"] < PLACE_EDGE_SHARE * ref_cover)


def away_frames(mask, frames, ref_cover):
    """The frames (of `frames`, in order) where the person is out of shot, by the same rule as place_beats: gone, or
    a sliver past an edge, and once there, back only when clearly in again (PLACE_BACK_SHARE). So a moment's line and
    the beats agree (c6: frame 280 counted as in for the lines but still out for the beat, so a gesture came before
    "comes back into the frame")."""
    out, edge = set(), False
    for f in frames:
        p = mask_place(mask[f]) if 0 <= f < len(mask) else None
        share = PLACE_BACK_SHARE if edge else PLACE_EDGE_SHARE
        edge = p is not None and (p["left"] or p["right"]) and p["cover"] < share * ref_cover
        if p is None or edge:
            out.add(f)
    return out


def split_others(named, cap=3):
    """The scene's "others" -> (people, kept as <Subject N>; the rest, which belongs to the background).
    "none" entries go (c4-t002's "<Subject 3> is none")."""
    people, behind = [], []
    for x in (named or "").split(";"):
        x = x.strip(' ."')
        if not x or x.lower() in ("none", "no one", "nobody", "n/a", "nothing"):
            continue
        x = x[0].lower() + x[1:]
        (behind if NOT_PEOPLE.search(x) else people).append(x)
    return people[:cap], behind


def others_from_moments(moment_others, at_least=2):
    """The moments' "others" answers -> one person to keep, when the scene question named nobody (wrestling c4: the
    scene said "none" for the opponent in every frame, while the moments said "opponent lying pinned, spectators
    watching"). Each answer is split at commas and semicolons and the crowd / screen parts dropped; the person's
    words up to what they're doing ("opponent lying pinned" -> "the opponent"), the most frequent; None when fewer
    than at_least answers see a person."""
    found = []
    for x in moment_others:
        x = " ".join((x or "").split()).strip(' ."')
        if not x or x.lower().startswith(("none", "no one", "nobody", "n/a")):
            continue
        part = next((q.strip(' ."') for q in re.split(r"[;,]", x) if q.strip(' ."') and not NOT_PEOPLE.search(q)), "")
        if not part:
            continue
        words = part.split()
        cut = next((k for k, w in enumerate(words) if k and re.fullmatch(r"(?i)[a-z]+(ing|ed)|beneath|below|under", w)),
                   len(words))
        d = " ".join(words[:cut]).strip(" ,")
        d = d[0].lower() + d[1:]
        if not re.match(r"(?i)(the|a|an)\b", d):
            d = "the " + d
        found.append(d)
    if len(found) < at_least:
        return None
    return max(set(found), key=lambda d: (found.count(d), -len(d)))


def relation_sentence(answer, pr):
    """The relation answer as the shot's opening sentence: "the person" becomes her, "the other person" stays the
    other; "" for none."""
    t = " ".join((answer or "").split()).strip(' ."')
    if not t or t.lower().startswith(("none", "n/a", "they stay apart", "no ")):
        return ""
    t = re.sub(r"\bthe other person\b", "THE_OTHER", t, flags=re.I)
    t = touch_sentence(t, pr)
    return t.replace("THE_OTHER", "the other person")


def with_background(video_1, behind):
    """The source clip's sentence, with what the scene saw behind the people added: kept as the background."""
    v = (video_1 or "").strip()
    if v and not v.endswith("."):
        v += "."
    return (v + " " if v else "") + "Behind them, kept as they are: " + "; ".join(behind) + "."


def _zone(x):
    if x < 1 / 3.0:
        return "in the left third of the frame"
    if x > 2 / 3.0:
        return "in the right third of the frame"
    if x < 0.42:
        return "just left of the centre of the frame"
    if x > 0.58:
        return "just right of the centre of the frame"
    return "in the centre of the frame"


def _framing(p, pr, kept=False):
    """How the person is framed, from the mask's top and bottom. kept: the person stays (background replace), so
    there is no "person" to be framed as close as."""
    share = p["height"]
    size = ("most of" if share > 0.85 else "about three quarters of" if share > 0.68 else
            "about two thirds of" if share > 0.58 else "about half of" if share > 0.4 else "about a third of"
            if share > 0.25 else "a small part of")
    if p["bottom"] >= 0.97:
        # touching the bottom edge: close, or a seated whole figure with the feet on the edge (the 60 s test job's c3,
        # 2026-10-14: "framed close ... cutting across her body" on a wide shot, and H3 pushed in). Only what the mask
        # measures: the share of the height, and the head near the top
        return (f"filling {size} the frame's height"
                + (f", {pr['poss']} head near the top of the frame" if p["top"] <= 0.12 else ""))
    return f"{pr['poss']} whole figure inside the frame, filling {size} the frame's height"


def place_beats(mask, frames, clock_fps, pronoun, ref_cover=None, kept=False):
    """Timed placement sentences for one shot. mask: the chunk's source mask [n, h, w]; frames: chunk-relative frames
    to read (the shot's moment pictures); clock_fps: the render's rate (H3's 24 when conforming), so the times are
    the ones H3 plays. A sentence is written when the place changes: first place + framing + direction, a move to
    another part of the frame, going out past an edge or out of frame, coming back in. kept: the person stays
    (background replace; see _framing). -> [(frame, sentence)]."""
    pr = PRONOUNS[pronoun]
    n = len(mask)
    frames = [f for f in frames if 0 <= f < n]
    if not frames:
        return []
    places = {f: mask_place(mask[f]) for f in frames}
    if ref_cover is None:
        covers = [p["cover"] for p in places.values() if p]
        ref_cover = float(np.median(covers)) if covers else 0.0

    def state(p, last):
        if p is None:
            return ("out",)
        share = PLACE_BACK_SHARE if last and last[0] == "edge" else PLACE_EDGE_SHARE
        if (p["left"] or p["right"]) and p["cover"] < share * ref_cover:
            return ("edge", "left" if p["left"] and (not p["right"] or p["x"] < 0.5) else "right")
        return ("in", _zone(p["x"]))

    def heading(k, first=False):
        """Which way and how fast the person moves around frames[k], from the centre's drift over about a second:
        "already moving ..." at the start of the shot (the hand prompt that followed said "mid-stride")."""
        f = frames[k]
        a, b = places.get(f), None
        for g in frames[k + 1:]:
            if g - f >= clock_fps * 0.4 and places.get(g):
                b, f2 = places[g], g
                break
        if not a or not b:
            return ""
        v = (b["x"] - a["x"]) / ((f2 - f) / float(clock_fps))
        if abs(v) <= 0.04:
            return ""
        pace = "slowly " if abs(v) < PLACE_SLOW else "quickly " if abs(v) > PLACE_FAST else ""
        return f", {'already ' if first else ''}moving {pace}towards the {'left' if v < 0 else 'right'}"

    out, last, framed = [], None, False
    for k, f in enumerate(frames):
        p, st = places[f], state(places[f], last)
        if st == last:
            continue
        if last is not None and st[0] == last[0] == "in":
            # a move from one part of the frame to another counts only when the next picture shows it too: a person
            # sitting on a zone's edge (the 60 s test job's c3, x 0.66-0.70) wobbled across it and the draft said "just
            # right of the centre, moving towards the right" then "in the right third" a tenth of a second later
            nxt = frames[k + 1] if k + 1 < len(frames) else None
            if nxt is None or state(places[nxt], st) != st:
                continue
        if k and last is not None:
            # the change happened somewhere since the last picture: time it at the first frame that shows it
            for g in range(frames[k - 1] + 1, f):
                pg = mask_place(mask[g])
                if state(pg, last) == st:
                    f, p = g, pg or p
                    break
        t = f / float(clock_fps)
        at = "At the start" if f == frames[0] and t < 0.3 else f"At about {t:.1f} seconds"
        if st[0] == "out":
            s = (f"{at} {pr['subj']} is out of the frame, and only the background shows." if last is None else
                 f"{at} {pr['subj']} {'has' if pr['subj'] != 'they' else 'have'} gone out of the frame, and only the "
                 f"background shows.")
        elif st[0] == "edge":
            s = (f"{at} {pr['subj']} {'has' if pr['subj'] != 'they' else 'have'} gone out past the {st[1]} edge of the "
                 f"frame, with only part of {pr['poss']} body still showing at that edge.")
        else:
            back = last is not None and last[0] in ("out", "edge")
            side = (last[1] if last[0] == "edge" else "left" if p["x"] < 0.5 else "right") if back else ""
            verb = "is" if pr["subj"] != "they" else "are"
            s = (f"{at} {pr['subj']} comes back into the frame from the {side} and {verb} {st[1]}" if back
                 else f"{at} {pr['subj']} {verb} {st[1]}")
            # a kept person (background replace) gets no framing words: the mask can't tell a close shot from a seated
            # whole figure whose feet touch the bottom edge (the 60 s test job's c3, 2026-10-14: a wide shot of her on the
            # couch was written "framed close ... the bottom edge cutting across her body", and H3 pushed in to her
            # face). Her size comes from <Video 1>; the hold sentence says it stays.
            if not framed and not kept:
                s += ", " + _framing(p, pr, kept)
                framed = True
            s += heading(k, first=last is None) + "."
        out.append((f, s[0].upper() + s[1:]))
        last = st
        if len(out) >= PLACE_MAX_BEATS:
            break
    return out


def touch_sentence(touch, pr):
    """A moment's touch answer as a sentence about the main person: only "the person" becomes her pronoun (the
    others keep theirs); "" for none."""
    t = " ".join((touch or "").split()).strip(" .")
    if not t or t.lower().startswith(("none", "no ", "not ", "nothing", "n/a", "they don't", "they do not")):
        return ""
    t = re.sub(r"\b[Tt]he person['’]s\b", pr["poss"], t)
    t = re.sub(r"\b[Tt]he person\b", pr["subj"], t)
    return t[0].upper() + t[1:] + "."


_HANDLING = re.compile(r"(?i)\b(hold|holds|holding|held|pick|picks|picking|put|puts|putting|show|shows|showing|"
                       r"lift|lifts|lifting|raise|raises|raising|carry|carries|carrying|hands? (?:\w+ ){0,3}over|pass|passes|"
                       r"open|opens|opening|pour|pours|pouring|drink|drinks|eat|eats|write|writes|type|types)\b")


def merge_moments(moments, lines_rel, fps, pronoun, language="English", similar=0.85, placements=(), lean=False):
    """A shot block from its moments, by the node: one sentence per moment whose hands differ from
    the last kept one (a repeated hand with a new gaze adds the gaze), the shot's lines placed at their
    times. moments: [(seconds, {pose, look, hands})]; placements: [(seconds, sentence)] from place_beats, each
    put just before the moment at its time. Nothing observed is dropped, nothing is added."""
    pr = PRONOUNS[pronoun]
    out, last_hands, last_look, last_touch, no_touch = [], None, None, "", 0
    ev = [(t, "m", m) for t, m in moments]
    ev += [(ln["frames"][0] / float(fps) + 1e-3, "l", ln) for ln in lines_rel if "frames" in ln]
    ev += [(t - 1e-3, "p", s) for t, s in placements]     # where she is, just before what she does there
    ev.sort(key=lambda e: e[0])
    first = True
    for _, kind, x in ev:
        if kind == "l":
            out.append(say_line(x, language))
            continue
        if kind == "p":
            out.append(x)
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
            if look and look != last_look and not lean:
                out.append(f"{pr['Subj']} looks {look}.")
        elif hands and (not lean or _HANDLING.search(hands)):
            out.append(f"{pr['Subj']} {hands}" + (f", looking {look}" if look and look != last_look and not lean
                                                    else "") + ".")
        last_hands, last_look = hands or last_hands, look or last_look
        touch = touch_sentence(x.get("touch") or "", pr)
        if touch:
            no_touch = 0
            if not last_touch or difflib.SequenceMatcher(None, touch.lower(), last_touch.lower()).ratio() < similar:
                out.append(touch)
            last_touch = touch
        elif last_touch:
            # one "none" in the middle of a hold is Qwen missing it (wrestling c4): ended only after two in a row
            no_touch += 1
            if no_touch >= 2:
                out.append(f"{pr['Subj']} and the others no longer touch.")
                last_touch, no_touch = "", 0
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


def shot_time(sec):
    """Seconds as H3's guide writes a cut time: MM:SS.mmm."""
    sec = max(0.0, float(sec))
    mm = int(sec // 60)
    return f"{mm:02d}:{sec - 60 * mm:06.3f}"


_CUT = re.compile(r"^(?:(?:Jump cut\.|At \d\d:\d\d\.\d{3}, the shot cuts to the next part of <Video 1>\.)\s*)+", re.I)


def tidy_shots(blocks, starts=None):
    """[Shot 1] never opens on a cut; later shots (each after a source cut) always do. With starts (each shot's start
    in seconds on the render's clock), in H3's guide's form: "[Shot 2] At 00:03.500, the shot cuts to ..." (base guide
    §4.2, ref guide §5.1: later shots mark their cut time); without, "Jump cut."."""
    out = []
    for i, b in enumerate(blocks):
        m = re.match(r"(\[Shot\s*\d+\])\s*(.*)", b, re.S)
        head, body = (m.group(1), m.group(2)) if m else (f"[Shot {i + 1}]", b)
        body = _CUT.sub("", body.strip())
        cut = ""
        if i:
            cut = (f"At {shot_time(starts[i])}, the shot cuts to the next part of <Video 1>. "
                   if starts and i < len(starts) else "Jump cut. ")
        out.append(f"{head} " + cut + body)
    return out


# H3's own camera words (base guide §4.3: "Static Shot - the camera position and lens remain still"; its example
# "The camera holds a static shot as ..."). Lean background drafts with "stays fixed at the same distance and angle"
# still zoomed slowly on some seeds (2026-10-12); the guide's term is what its captions use.
STATIC_CAMERA = "The camera holds a static shot"
# the full templates' cut sentence, said positively (it read "with no transition, dissolve or morph": a negation names
# what it forbids, and H3 morphs across source cuts anyway, CR-T1 T2)
HARD_CUTS = "Every jump cut is a hard cut at exactly the same moment as in <Video 1>."


def camera_in_shots(blocks, video_1, lead=""):
    """The shots with the camera written inside them, as the guide asks: when Qwen's clip sentence says the camera
    holds still (camera_is_static), each shot says so in the guide's words ("throughout" for a single shot); a moving
    or unknown camera gets nothing added. lead: a sentence to open the first shot with, before the camera (the
    background templates' establishing line)."""
    static = camera_is_static(video_1)
    out = []
    for i, b in enumerate(blocks):
        m = re.match(r"(\[Shot\s*\d+\]\s*)(.*)", b, re.S)
        head, body = (m.group(1), m.group(2)) if m else ("", b)
        cut = _CUT.match(body)
        head, body = head + (cut.group(0) if cut else ""), body[cut.end():] if cut else body
        add = []
        if i == 0 and lead:
            add.append(lead)
        if static:
            add.append(STATIC_CAMERA + (" throughout." if len(blocks) == 1 else "."))
        out.append(head + " ".join(add + ([body] if body.strip() else [])))
    return out


def hold_sentence(beats, who):
    """When a shot's placement is one place, held (one beat, not at an edge or out, not moving): say it holds to the end
    (the c6 hand prompt that followed best: "She stays framed from the waist up for the whole shot, at the man's size
    and position at every moment"; on a still background shot a slow zoom set in partway, 2026-10-12)."""
    if len(beats) != 1:
        return ""
    s = beats[0][1]
    if re.search(r"out of the frame|out past|moving ", s):
        return ""
    return f"{who} stays at this place and size in the frame to the end of the shot."


# ---------------------------------------------------------------------------
# parsing Qwen's answers
# ---------------------------------------------------------------------------

_LABEL = re.compile(r"(?im)^[ \t]*[*#>\- \t]*(video[_ ]?1|shots|sounds|name|appearance|features|pronoun|person|objects|others|background)"
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
    blocks = [f"[Shot {i + 1}] " + convert_says(dedupe_sentences(_META.sub("", b), cap), language)
              for i, b in enumerate(blocks)]
    return {"video_1": v1.rstrip(".") + "." if v1 else "", "shots": blocks, "sounds": sounds, "warnings": warns}


# Qwen talking about its own input ("she stays fully visible across all images", 2026-10-12): H3 gets no images list
# ("in the frame" / "in every frame" stay: they say where she is, not what Qwen was shown)
_META = re.compile(r",?\s+(?:across|in|throughout|over|through)\s+(?:all\s+|every\s+|each\s+|the\s+)(?:of\s+the\s+)?"
                   r"(?:sampled\s+)?(?:images?|pictures?|stills?)\b"
                   r"|,?\s+(?:during|throughout|for|over|across|in)\s+(?:this|the)\s+(?:entire\s+|whole\s+)?"
                   r"(?:segment|clip|sequence|excerpt)\b", re.I)


# ---------------------------------------------------------------------------
# the six-section Ref2VA template ("character replace (Ref2VA)")
# ---------------------------------------------------------------------------

def _cuts_phrase(n_cuts):
    return {0: "", 1: "the jump cut, ", 2: "both jump cuts, "}.get(n_cuts, f"all {n_cuts} jump cuts, ")


def soundscape(sounds, audio):
    """overall_soundscape: the copied ambience of <Audio 1>'s source (ref guide §6), ending on a full stop."""
    s = (sounds or "").strip().rstrip(" ;,")
    if audio:
        s = "The original sound of <Video 1>: " + (s or "the room tone and the sounds of the action")
    else:
        s = s or "Quiet room tone"
    return s if s.endswith((".", "!", "?")) else s + "."


def fill_ref2va(*, subject, video_1, shots, sounds, n_shots, dialogue, audio, pronoun, who=None, others=False,
                people=(), speaking=False, placed=False, exits=False):
    """The six sections, filled from the job's subject and Qwen's three parts (design §4.6), after H3's
    reference guide (MiniMax-H3 skills/h3-prompt-writing/references/ref-en.txt). people: everyone else the
    scene names, [(label, description, shots they're in)], each a <Subject N> kept as it is; speaking: one of
    them, or an off-screen voice, says a line. placed: the shots carry placement beats from the source mask
    (place_beats), so the summary and retention lock the place, size and framing too; exits: the person goes out
    of the frame and comes back in this chunk."""
    subject = fold_picture_2(subject, False)        # a second picture is cited inside the subject's line
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
               + (f" {pr['Subj']} {'takes' if pr['subj'] != 'they' else 'take'} the person's exact place in every "
                  f"frame: the same position across the frame, the same size and the same framing"
                  + (", and the same moments of leaving the frame and coming back into it." if exits else ".")
                  if placed else "")
               + (f" {_and_list([lab for lab, _, _ in people])} {'keeps' if len(people) == 1 else 'keep'} exactly "
                  f"{'their' if len(people) == 1 else 'their'} own appearance and movements: only the main person is "
                  f"replaced." if people else
                  " Everyone else in <Video 1> keeps exactly their own appearance and movements: only the main person "
                  "is replaced." if others else ""))
    ret = [f"<Subject 1> (appears throughout): partially_preserved - {pr['poss']} face, hair, costume and accessories "
           f"from <Picture 1> are retained; "
           + (f"{pr['poss']} position across the frame, size and framing"
              + (", the moments of leaving and coming back into the frame" if exits else "") + ", "
              if placed else "")
           + f"{pr['poss']} body position, lean, pose, arm and hand actions, head "
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
             f"does in <Video 1>, frame by frame, "
             # said positively: a negated "never moves to the centre" names the centre (the test character kept landing there on the
             # green-screen clip; a negated "no pan, zoom or tilt" brought the zoom back on the background drafts)
             + f"framed exactly the same way and in the same place in the frame, looking where the person looks, "
             + f"and {pr['poss']} hands only hold what the person's hands hold.")
    blocks = list(shots) or ["[Shot 1] " + f"{pr['Subj']} {pr['does']} exactly what the person does."]
    blocks = camera_in_shots(blocks, video_1)
    tail = []
    if dialogue and speaking:
        tail.append(f"{pr['Poss']} lips move only with {pr['poss']} own lines, in time with the speech in <Audio 1>; "
                    f"everyone else's lips move only with their own.")
    elif dialogue:
        tail.append(f"{pr['Poss']} lips move with every word, in time with the speech in <Audio 1>, and close between "
                    f"sentences.")
    if n_cuts:
        tail.append(HARD_CUTS)
    if tail:
        blocks[-1] = blocks[-1].rstrip() + " " + " ".join(tail)
    sound = soundscape(sounds, audio)
    defs = "".join(f"\n{lab} is {desc} in <Video 1>, who keeps their own appearance and movements." for lab, desc, _ in people)
    a1 = ("\n<Audio 1> is the synchronized audio track of <Video 1>, and is reused in the target video." if audio else "")
    return ("subject_definitions:\n" + subject.strip() + defs + "\n<Video 1> is the source video for the target video edit: "
            + v1 + a1 + "\n\nsummary:\n" + summary + "\n\nretention_analysis:\n" + "\n".join(ret)
            + "\n\ndetailed_description:\n" + style + "\n" + "\n".join(blocks)
            + "\n\noverall_soundscape:\n" + sound + "\n\nnon_diegetic_music:\nN/A\n")


# small details the look sentence leaves to the sheet (the hand lean that won had 7 items; the sheet's list has 15)
_DETAIL = re.compile(r"(?i)\b(eyes?|lips?|eyebrows?|lashes|make-?up|choker|necklace|earrings?|bow|bows|buttons?|"
                     r"ribbons?|bracelets?|rings?|nails?)\b")
_LEGWEAR = re.compile(r"(?i)\b(socks?|stockings?|tights|leggings|heels|shoes?|boots?|sandals?|sneakers|trainers|"
                      r"slippers|footwear)\b")
_ABSENT = re.compile(r"(?i)(nothing|no|none)\b")
_EXPRESSION = re.compile(r"(?i)\b(open|opened|closed|parted)\s+mouth\b|\bsmil(e|es|ing)\b|\bfrown")
# "red lips", "and orange-tinged lips", "dark lipstick": up to two describing words, never "and" / "with" themselves
_LIPS = re.compile(r"(?i)\s*(?:\b(?:and|with)\b\s*)?(?:(?!(?:and|with)\b)[\w-]+\s+){0,2}lip(?:s|stick)?\b")


def full_look_subject(subject):
    """The large template's look sentence: the sheet's items whole (face, make-up, accessories and their "with ..."
    details), minus legwear, what isn't there and the sheet's own expression. test clip c5 t001 (2026-10-16):
    lean_subject left "pale skin, greenish hair, a dark red suit", which fits the source man as well as the
    cosplayer; the eye make-up, the lips and the bow tie that tell them apart were the details it drops, and H3
    rebuilt the source actor from the marked guide."""
    first, _, rest = (subject or "").strip().partition("\n")
    m = re.match(r"<Subject 1> is the (.+?) whose motion comes from <Video 1> and whose appearance comes from "
                 r"<Picture 1>:\s*(.+?)\.?\s*$", first)
    if not m:
        return fold_picture_2(subject.strip(), False)
    # the lips' colour is left to the picture: "red lips" (the second test sheet, muted pink-red lips) came
    # out saturated lipstick red on every frame of c4 / c15 t002
    items = [_LIPS.sub("", x).strip(" ,") for x in re.split(r"[,;]", m.group(2))
             if x.strip() and not _LEGWEAR.search(x) and not _ABSENT.match(x.strip()) and not _EXPRESSION.search(x)]
    items = [x for x in items if x]
    line = f"<Subject 1> is the {m.group(1)} in <Picture 1>: {', '.join(items)}."
    return fold_picture_2(line + ("\n" + rest.strip() if rest.strip() else ""), False)


def lean_subject(subject):
    """The job's subject as one short look sentence: "<Subject 1> is the X in <Picture 1>: ..." with the sheet's items
    minus footwear and legwear (head-to-toe detail pulls the framing to full body; a cap cut the wings) and small
    details (eyes, lips, choker, bows), each without its "with ..." tail ("white halo with yellow trim" -> "white
    halo"). A
    subject that isn't in the drafter's form is kept as it is."""
    first, _, rest = (subject or "").strip().partition("\n")
    m = re.match(r"<Subject 1> is the (.+?) whose motion comes from <Video 1> and whose appearance comes from "
                 r"<Picture 1>:\s*(.+?)\.?\s*$", first)
    if not m:
        return fold_picture_2(subject.strip(), False)
    # the items split at commas and semicolons (the 27B lists them with "; ", 2026-10-16: split at commas only, its
    # whole list after "a young, thin man" was one item, cut at its first "with"); "nothing on the head" and
    # "no visible trousers" say what isn't there
    items = [re.split(r"\s+with\s+", x.strip(), 1)[0] for x in re.split(r"[,;]", m.group(2))
             if x.strip() and not _LEGWEAR.search(x) and not _ABSENT.match(x.strip())
             and not _DETAIL.search(re.split(r"\s+with\s+", x.strip(), 1)[0])]
    line = f"<Subject 1> is the {m.group(1)} in <Picture 1>: {', '.join(items)}."
    return fold_picture_2(line + ("\n" + rest.strip() if rest.strip() else ""), False)


def fill_ref2va_lean(*, subject, video_1, shots, sounds, n_shots, dialogue, audio, pronoun, who=None, others=False,
                     people=(), speaking=False, placed=False, exits=False, behind=(), full_look=False):
    """The lean template's six sections: the same parts as fill_ref2va, written short (LEAN_TEMPLATE). full_look:
    the subject's look sentence keeps the sheet's details (the large template)."""
    pr = PRONOUNS[pronoun]
    n_cuts = max(0, n_shots - 1)
    tag = "[video editing + reference generation" + (" + audio reuse]" if audio else "]")
    v1 = video_1 or "the source video, with a person in it."
    the_person = ("the " + re.sub(r"(?i)^(the|a|an)\s+", "", who.strip())) if who else "the person"
    kept = [lab for lab, _, _ in people] + (["everyone else"] if others and not people else [])
    summary = (f"{tag} The target video is an edited version of <Video 1>. <Subject 1>'s appearance replaces "
               f"{the_person} at their exact place, size and framing in every frame"
               + (", leaving the frame and coming back at the same moments" if exits else "")
               + f", while {_and_list(kept + ['the background', 'the camera', 'the action'])} remain as recorded.")
    pics = "<Picture 1> and <Picture 2>" if has_picture_2(subject) else "<Picture 1>"
    # the guide's retention lines carry where each label appears ("(appears in [Shot 1], [Shot 3])", ref guide §4.1);
    # the lean background form that dropped them was the one that showed the marked backdrop (2026-10-11)
    ret = [f"<Subject 1> (appears throughout, wherever {the_person} is in frame): attribute_transfer - transfer "
           f"{pr['poss']} face, hair, costume and accessories from {pics} onto {the_person}'s position, size, framing "
           f"and motion" + (", with the mouth moving with the speech." if dialogue else ".")]
    for lab, desc, in_shots in people:
        where = "throughout" if len(in_shots) >= n_shots else "in " + ", ".join(f"[Shot {k}]" for k in in_shots)
        ret.append(f"{lab} (appears {where}): fully_preserved - {desc} keeps their own appearance, position and "
                   f"movements from the source.")
    ret.append("<Video 1> (whole-video temporal structure, camera, background, lighting): partially_preserved - "
               "preserve the background"
               + (f" ({'; '.join(behind)})" if behind else "")
               + ", the camera position and angle, the framing" + (", every jump cut" if n_cuts else "")
               + " and the lighting.")
    if audio:
        ret.append("<Audio 1>: fully_copy - the original sound of <Video 1> is kept as it is.")
    blocks = list(shots) or [f"[Shot 1] {pr['Subj']} {pr['does']} what the person does."]
    tail = []
    if dialogue and speaking:
        tail.append(f"{pr['Poss']} lips move only with {pr['poss']} own lines; everyone else's only with theirs.")
    elif dialogue:
        tail.append(f"{pr['Poss']} lips move with every word and close between sentences.")
    if n_cuts:
        tail.append("Every jump cut happens at the same moment as in <Video 1>.")
    if tail:
        blocks[-1] = blocks[-1].rstrip() + " " + " ".join(tail)
    blocks = camera_in_shots(blocks, video_1)
    # the guide's opening before [Shot 1] (ref guide §5.2); both hand prompts that beat the drafts opened on the camera,
    # the framing and the place (c6 dx 0.035, wrestling c4 pose 0.49)
    opening = (f"The target video keeps the camera, framing, lighting and colour of <Video 1>, with <Subject 1> filling "
               f"exactly the place of {the_person} in it.")
    sound = soundscape(sounds, audio)
    defs = "".join(f"\n{lab} is {desc} in <Video 1>, unchanged from the source." for lab, desc, _ in people)
    a1 = ("\n<Audio 1> is the synchronized audio track of <Video 1>, and is reused in the target video." if audio else "")
    return ("subject_definitions:\n" + (full_look_subject if full_look else lean_subject)(subject) + defs
            + "\n<Video 1> is the source video for the target video edit: " + v1 + a1
            + "\n\nsummary:\n" + summary + "\n\nretention_analysis:\n" + "\n".join(ret)
            + "\n\ndetailed_description:\n" + opening + "\n" + "\n".join(blocks)
            + "\n\noverall_soundscape:\n" + sound + "\n\nnon_diegetic_music:\nN/A\n")


# ---------------------------------------------------------------------------
# the background template ("background replace (Ref2VA)"): the Planner's invert
# ---------------------------------------------------------------------------

def parse_scene_sheet(text):
    """Qwen's two lines about the place in Picture 1 -> {name, features}."""
    d = _labelled(text)
    if not d.get("features"):
        rows = [r.strip(" *-") for r in (text or "").splitlines() if r.strip(" *-")]
        if len(rows) >= 2:
            d = {"name": rows[0], "features": max(rows[1:], key=len)}
    name = re.sub(r"[\"'.]", "", (d.get("name") or "").splitlines()[0] if d.get("name") else "").strip().lower()
    name = re.sub(r"\s+environment$", "", name)
    feats = " ".join((d.get("features") or "").split()).strip().strip('"').rstrip(" .")
    return {"name": name or "new", "features": feats}


def scene_sentence(name, features):
    """H3's reference guide's pattern for a place: "<Subject N> is the [location] environment in <Picture N>,
    featuring [specific visual elements]." """
    return f"<Subject 1> is the {name} environment in <Picture 1>" + (f", featuring {features}." if features else ".")


def is_scene_subject(subject):
    return bool(re.match(r"\s*<Subject 1> is the .* environment in <Picture 1>", subject or ""))


def parse_detail(text):
    """Qwen's one "detail:" line about a second reference picture -> the sentence's body: no label, no final
    period. Without the label, the answer's own text."""
    t = text or ""
    m = re.search(r"(?im)^[ \t]*[*#>\- \t]*detail[ \t]*\**[ \t]*:[ \t]*\**[ \t]*(.+)$", t)
    body = m.group(1) if m else " ".join(r.strip(" *-") for r in t.splitlines() if r.strip(" *-"))
    return " ".join(body.split()).strip().strip('"*').rstrip(" .")


def parse_view(text):
    """Qwen's "view:" line about a second picture of the place -> "closer", "angle" or "same" ("same" when missing)."""
    m = re.search(r"(?im)^[ \t]*[*#>\- \t]*view[ \t]*\**[ \t]*:[ \t]*\**[ \t]*\"?(\w+)", text or "")
    v = (m.group(1).lower() if m else "")
    return v if v in ("closer", "angle") else "same"


_P2_KIND = {"closer": "a closer view of the same environment", "angle": "the same environment seen from another angle",
            "same": "a second view of the same environment"}


def picture_2_sentence(detail, scene, view="same"):
    """The line that defines <Picture 2> in the subject. It goes on its own line under the subject sentence, so the
    lean template's look sentence (which reads the first line only) keeps it as it is. A place's line says what kind of
    view it is (parse_view), which the prompts read back (picture_2_view)."""
    if not detail:
        return ""
    if scene:
        return f"<Picture 2> is {_P2_KIND.get(view, _P2_KIND['same'])}: {detail}."
    return f"<Picture 2> is a second reference of <Subject 1>: {detail}."


def has_picture_2(subject):
    return "<Picture 2>" in (subject or "")


def picture_2_view(subject):
    """The kind of view the subject's <Picture 2> line declares: "closer", "angle" or "same" (a line from before the
    view was asked says "a second view": "same", the reading that claims nothing about scale)."""
    m = re.search(r"(?m)^<Picture 2> is ([^:\n]*):", subject or "")
    k = m.group(1) if m else ""
    return "closer" if "closer view" in k else "angle" if "another angle" in k else "same"


def picture_2_retention(subject):
    """The place's retention clause for <Picture 2>: only a closer view is "surface detail only"; any view leaves the
    camera's distance and framing to <Video 1>."""
    if picture_2_view(subject) == "closer":
        return "<Picture 2> adds only surface detail, and the camera's distance and framing come only from <Video 1>."
    return "<Picture 2> adds detail of the same place, and the camera's position and framing come only from <Video 1>."


_P2_LINE = re.compile(r"(?m)^<Picture 2> is [^:\n]*:[ \t]*(.*?)\.?[ \t]*$")


def drop_picture_2(text):
    """The subject without its <Picture 2> line (the stored form), for a draft that has no second picture."""
    kept = [ln for ln in (text or "").split("\n") if not ln.startswith("<Picture 2>")]
    return "\n".join(kept).strip()


def fold_picture_2(text, scene):
    """The subject with its <Picture 2> line folded into <Subject 1>'s own line. H3's prompt guide keeps a standalone
    <Picture N> line for an image that is a target frame or a shot-planning anchor (a close-up read that way
    zoomed the shot in); an image that only defines a scene or a character is cited inside that subject's
    definition, with what it provides. The job's subject keeps the line as drafted; the prompts get this."""
    m = _P2_LINE.search(text or "")
    if not m:
        return text
    detail = m.group(1)
    view = picture_2_view(text)
    lines = ((text[:m.start()] + text[m.end():]).strip("\n")).split("\n")
    first = lines[0].rstrip().rstrip(".")
    if scene and view == "closer":
        first += (f"; <Picture 2> provides only its surface texture, colours and light, never the camera's distance "
                  f"or framing: {detail}.")
    elif scene:
        first += (f"; <Picture 2> shows the same environment{' from another angle' if view == 'angle' else ''}, adding "
                  f"its detail: {detail}.")
    else:
        first += f"; <Picture 2> provides only a closer look at the same character: {detail}."
    return "\n".join([first] + [ln for ln in lines[1:] if ln.strip()])


def bg_instruction(*, n_frames, fps, sample_idx, shots_rel, lines, events, who, extra=""):
    """The background template's one call: each shot with its pictures and its lines, about the kept person."""
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
    return BG_DRAFT_INSTRUCTION.format(
        n=len(sample_idx), secs=n_frames / float(fps), who=who, k=k, k_s="" if k == 1 else "s",
        shot_lines="\n".join(rows), dialogue_rule=_dialogue_rule(bool(lines)),
        sounds_hint=_sounds_hint(events, lines), extra=_extra(extra))


def label_person(block, who, pronoun):
    """A background shot block with the kept person called by their label, as H3's guide keeps labels
    consistent: the who phrase and its short form ("the man") become <Subject 2>, and so does a sentence that
    opens with their pronoun ("He walks" -> "<Subject 2> walks", "His hands" -> "<Subject 2>'s hands"). A
    pronoun inside a sentence stays ("<Subject 2> lifts his arm"); "they" stays (the verb would disagree)."""
    m = re.match(r"(\[Shot\s*\d+\]\s*)(.*)", block, re.S)
    head, body = (m.group(1), m.group(2)) if m else ("", block)
    w = re.sub(r"(?i)^(the|a|an)\s+", "", (who or "").strip())
    names = [w] if w else []
    core = re.split(r"\s+(?:in|with|wearing|holding|on|at)\s+", w, maxsplit=1)[0] if w else ""
    names += [core, core.split()[-1]] if core else []      # "bald man", "man"
    names.append("person")
    for nm in sorted(set(names), key=len, reverse=True):
        body = re.sub(rf"(?i)\b(?:the|a|an)\s+{re.escape(nm)}\b", "<Subject 2>", body)
    if pronoun in ("he", "she"):
        subj, poss = ("He", "His") if pronoun == "he" else ("She", "Her")
        start = r"(^|[.!?]\s+|;\s+)"
        body = re.sub(start + poss + r"\b", lambda x: x.group(1) + "<Subject 2>'s", body)
        body = re.sub(start + subj + r"\b", lambda x: x.group(1) + "<Subject 2>", body)
    return head + body


_AT_CENTRE = re.compile(r"\bis in the centre of the frame\b")
_CENTRE_WORDS = re.compile(r",?\s+(?:centered|centred|centrally|center-frame|centre-frame)(?:\s+(?:in|within)\s+the\s+frame)?"
                           r"|,?\s+in\s+the\s+(?:centre|center|middle)\s+of\s+the\s+frame", re.I)


def drop_centre(text):
    """Qwen's "sits centered in the frame" taken out where the mask puts the person elsewhere: the 60 s test job's c8
    (2026-10-13) had her at x 0.70 by the mask, Qwen said "centered", and the prompt said both."""
    return _CENTRE_WORDS.sub("", text or "")


def bg_place_shots(blocks, mask, shots_rel, fps, clock, pronoun):
    """Background shots with the kept person's placement put in front of Qwen's description: place_beats over the
    shot's moment frames (where in the frame, how much of the frame they fill, when they leave and come back), the
    pronoun that opens each beat as <Subject 2> (label_person's rule), and the hold sentence when the place holds."""
    pr = PRONOUNS[pronoun]
    covers = [float((mask[k] > 0.5).mean()) for k in range(0, len(mask), max(1, int(fps) // 4))]
    covers = [v for v in covers if v >= PLACE_MIN_COVER]
    ref = float(np.median(covers)) if covers else 0.0
    out = []
    for i, b in enumerate(blocks):
        if i >= len(shots_rel):
            out.append(b)
            continue
        sh = shots_rel[i]
        beats = place_beats(mask, moment_frames(sh, fps), clock, pronoun, ref, kept=True)
        if not beats:
            out.append(b)
            continue
        said = []
        for _, s in beats:
            if pronoun != "they":
                s = re.sub(rf"^(At the start|At about [\d.]+ seconds) {pr['subj']} ", r"\1 <Subject 2> ", s)
            said.append(s)
        hold = hold_sentence(beats, "<Subject 2>")
        m = re.match(r"(\[Shot\s*\d+\]\s*)(.*)", b, re.S)
        head, body = (m.group(1), m.group(2)) if m else ("", b)
        if not _AT_CENTRE.search(beats[0][1]):
            body = drop_centre(body)
        out.append(head + " ".join(said + [body.strip()] + ([hold] if hold else [])).strip())
    return out


def fill_bg_ref2va(*, subject, who, video_1, shots, sounds, n_shots, dialogue, audio, pronoun):
    """The six sections for a background swap, after H3's reference guide: the place is <Subject 1> (from
    <Picture 1>), the person is <Subject 2> (from <Video 1>, fully_preserved), the backdrop of <Video 1> is
    not kept. Task tag: video editing (the source is edited) + reference generation (the picture guides the
    new background without anchoring a frame) + audio reuse."""
    p2_ret = picture_2_retention(subject)          # read off the subject's own <Picture 2> line, before the fold
    subject = fold_picture_2(subject, True)         # a second picture is cited inside the place's line
    pr = PRONOUNS[pronoun]
    is_ = "are" if pronoun == "they" else "is"
    m = re.match(r"\s*<Subject 1> is the (.*?) environment in <Picture 1>", subject or "")
    name = m.group(1) if m else "new"
    n_cuts = max(0, n_shots - 1)
    cp = _cuts_phrase(n_cuts)
    who = who or "the person"
    tag = "[video editing + reference generation" + (" + audio reuse]" if audio else "]")
    where = "throughout" if n_shots == 1 else "in " + ", ".join(f"[Shot {k}]" for k in range(1, n_shots + 1))
    v1 = drop_shot_size(video_1) or f"{who} filmed in front of a plain backdrop."
    p2 = has_picture_2(subject)          # a second reference picture of the same place
    summary = (f"{tag} The target video is an edited version of <Video 1> in which the backdrop behind <Subject 2> is "
               f"replaced by <Subject 1>, the {name} environment from <Picture 1>{' and <Picture 2>' if p2 else ''}. "
               f"<Subject 2> is kept exactly: "
               f"{pr['poss']} appearance and clothing, every step, turn, hand movement, head turn, mouth movement and "
               f"expression timing, {cp}the camera's angle and movement and how {pr['subj']} {is_} framed. "
               f"Wherever <Subject 2> is out of frame, the frame shows only <Subject 1>. Nothing of the original "
               f"backdrop remains. The edit runs continuously from the first frame to the last without deviation.")
    ret = [f"<Subject 1> (appears throughout): partially_preserved - the layout, colours, fixtures, furniture and "
           f"decorations of the environment in <Picture 1> are retained as the background, seen from the camera's "
           f"angle in <Video 1> and moving with the camera."
           # the lean form's wording, the one a render confirmed (2026-10-11); "close detail" read as a closer view
           + (" " + p2_ret if p2 else ""),
           f"<Subject 2> (appears {where}, whenever {pr['subj']} {is_} in frame): fully_preserved - {pr['poss']} "
           f"identity, face, hair, build and clothing are retained, with every movement, position in the frame and "
           f"its timing from <Video 1>.",
           # partially_preserved, as the guide defines it ("still used, but some defined characteristics are
           # changed"): the backdrop is replaced. fully_preserved said to keep all of <Video 1>, marked backdrop and all
           f"<Video 1> (whole-video temporal structure, camera angle and movement, the person): partially_preserved - "
           f"the shot structure{' and every jump cut' if n_cuts else ''}, the camera's angle and movement and "
           f"<Subject 2>'s every movement are preserved in every frame; its original backdrop is replaced by "
           f"<Subject 1>."]
    if audio:
        ret.append("<Audio 1>: fully_copy - the original speech and room sound of <Video 1> are kept as they are"
                   + (f", and {pr['poss']} lip movements follow the speech." if dialogue else "."))
    style = ("The target video keeps the camera, framing and timing of <Video 1>, with <Subject 1> behind "
             "<Subject 2> in place of the original backdrop. <Subject 2> is lit to match <Subject 1>, with natural "
             f"shadows and contact with whatever {pr['subj']} {'sit' if pronoun == 'they' else 'sits'} or "
             f"{'stand' if pronoun == 'they' else 'stands'} on, and the background moves with the camera as the "
             "original backdrop did.")
    blocks = list(shots) or ["[Shot 1] <Subject 2> moves exactly as in <Video 1>."]
    tail = []
    if dialogue:
        tail.append(f"{pr['Poss']} lips move with every word, in time with the speech in <Audio 1>, and close between "
                    f"sentences.")
    if n_cuts:
        tail.append(HARD_CUTS)
    if tail:
        blocks[-1] = blocks[-1].rstrip() + " " + " ".join(tail)
    blocks = camera_in_shots(blocks, video_1)
    sound = soundscape(sounds, audio)
    a1 = ("\n<Audio 1> is the synchronized audio track of <Video 1>, and is reused in the target video." if audio else "")
    return ("subject_definitions:\n" + subject.strip()
            + f"\n<Subject 2> is {who} in <Video 1>, who keeps {pr['poss']} own appearance and movements."
            + "\n<Video 1> is the source video for the target video edit: " + v1 + a1
            + "\n\nsummary:\n" + summary + "\n\nretention_analysis:\n" + "\n".join(ret)
            + "\n\ndetailed_description:\n" + style + "\n" + "\n".join(blocks)
            + "\n\noverall_soundscape:\n" + sound + "\n\nnon_diegetic_music:\nN/A\n")


_PLACE_ITEMS = 8
_CAMERA_STILL = re.compile(r"\b(holds steady|remains steady|stays steady|stationary|static|fixed|locked[- ]off|"
                           r"holds still|remains still|stays still|does not move|doesn't move|motionless)\b", re.I)
_CAMERA_MOVES = re.compile(r"\b(follows|tracks?|pans?|tilts?|zooms?|dolly|handheld|hand-held|moves?|moving|drifts?|"
                           r"pushes|pulls|sweeps|orbits|circles|glides)\b", re.I)


_SHOT_SIZE = re.compile(r"\s+(?:in|at)\s+(?:an?\s+)?(?:(?:extreme|medium|tight|wide|full|close|long)[ -])*"
                        r"(?:close-up|closeup|shot)\b"
                        # how much of the person shows: "in full view", "from the waist up", "full-length" (2026-10-14:
                        # "capturing her seated in full view" sat beside a close placement; framing words are camera cues)
                        r"|\s+(?:in\s+full\s+(?:view|frame)|from\s+(?:the\s+)?(?:waist|chest|knees|hips)\s+up|"
                        r"from\s+head\s+to\s+(?:toe|foot))\b"
                        r"|\s*,?\s*\b(?:full[- ](?:length|body|figure)|waist[- ]up|head[- ]to[- ]toe)\b", re.I)


def drop_shot_size(video_1):
    """Qwen's video sentence without its shot-size words ("keeping her centered in medium close-up as ..." ->
    "keeping her centered as ..."). Three zooming renders had "medium close-up" in <Video 1>'s line and the
    ones that held had "framing her seated centrally"; with the words taken out the zoom came on slower and
    smaller. The framing itself stays: only the label for how close the shot is goes."""
    return _SHOT_SIZE.sub("", video_1 or "")


def camera_is_static(video_1):
    """Qwen's video_1 sentence says the camera holds still ("The camera holds steady at eye level", "a static
    camera") and nothing in it says the camera moves."""
    sents = [x for x in re.split(r"[.;]", video_1 or "") if re.search(r"\bcamera\b", x, re.I)]
    return bool(sents) and any(_CAMERA_STILL.search(x) for x in sents) and not any(_CAMERA_MOVES.search(x) for x in sents)


def place_items(subject):
    """The features in the place sentence, each without its "with ..." tail: [] when the subject isn't in the
    drafter's form."""
    first = (subject or "").strip().partition("\n")[0]
    m = re.match(r"<Subject 1> is the .+? environment in <Picture 1>, featuring (.+?)\.?\s*$", first)
    return [re.split(r"\s+with\s+", x.strip(), maxsplit=1)[0] for x in m.group(1).split(",") if x.strip()] if m else []


def lean_place(subject):
    """The job's place subject with a short feature list: "<Subject 1> is the X environment in <Picture 1>,
    featuring a, b, c." keeps its first items, each without its "with ..." tail. A subject that isn't in the
    drafter's form is kept as it is. A <Picture 2> line below it is folded into the first line (fold_picture_2)."""
    text = (subject or "").strip()
    first, _, rest = text.partition("\n")
    m = re.match(r"(<Subject 1> is the .+? environment in <Picture 1>), featuring (.+?)\.?\s*$", first)
    if m:
        text = (f"{m.group(1)}, featuring {', '.join(place_items(text)[:_PLACE_ITEMS])}."
                + ("\n" + rest.strip() if rest.strip() else ""))
    return fold_picture_2(text, True)


def fill_bg_ref2va_lean(*, subject, who, video_1, shots, sounds, n_shots, dialogue, audio, pronoun):
    """The background template's six sections written short (BACKGROUND_LEAN): the same parts and labels as
    fill_bg_ref2va, one clause each, and one lead sentence in place of the style paragraph (its lighting and shadows
    ride in the place's line). It keeps the labels the full form has ("appears throughout", "whenever she is in frame"):
    the first lean draft dropped them and the two renders made from it showed the marked backdrop."""
    pr = PRONOUNS[pronoun]
    m = re.match(r"\s*<Subject 1> is the (.*?) environment in <Picture 1>", subject or "")
    name = m.group(1) if m else "new"
    n_cuts = max(0, n_shots - 1)
    who = who or "the person"
    tag = "[video editing + reference generation" + (" + audio reuse]" if audio else "]")
    pics = "<Picture 1> and <Picture 2>" if has_picture_2(subject) else "<Picture 1>"
    v1 = drop_shot_size(video_1) or f"{who} filmed in front of a plain backdrop."
    is_ = "are" if pronoun == "they" else "is"
    where = "throughout" if n_shots == 1 else "in " + ", ".join(f"[Shot {k}]" for k in range(1, n_shots + 1))
    summary = (f"{tag} The target video is an edited version of <Video 1>. The backdrop behind <Subject 2> is replaced by "
               f"<Subject 1>, the {name} environment from {pics}, while <Subject 2>, the camera and the action remain "
               f"as recorded" + (", every jump cut at the same moment" if n_cuts else "")
               + "; wherever <Subject 2> is out of frame, only <Subject 1> shows. Nothing of the original backdrop "
               "remains. The edit runs continuously from the first frame to the last without deviation.")
    # the labels the guide's format (and the full template) carry: <Subject 1> appears throughout, <Subject 2> whenever
    # in frame. The first lean draft dropped them; the two renders that showed the marked backdrop were those drafts.
    # the layout comes from <Picture 1> only, as in the full form: <Picture 2> is a close detail, and "the layout ... in
    # <Picture 1> and <Picture 2>" asks H3 to show the room at that close-up's scale too (see the shot below)
    ret = [f"<Subject 1> (appears throughout): partially_preserved - the layout, colours, fixtures and furniture of the "
           f"environment in <Picture 1> "
           f"are the background, seen from the camera's angle in <Video 1> and moving with the camera; <Subject 2> is "
           f"lit to match it, with natural shadows"
           + ("; " + picture_2_retention(subject) if has_picture_2(subject) else "."),
           f"<Subject 2> (appears {where}, whenever {pr['subj']} {is_} in frame): fully_preserved - {pr['poss']} "
           f"identity, face, hair, build and clothing, and every movement, position in the frame and timing, from "
           f"<Video 1>.",
           # partially_preserved: the backdrop is replaced (see fill_bg_ref2va)
           "<Video 1> (whole-video temporal structure, camera angle and movement, the person): partially_preserved - "
           "the camera's angle and movement, the framing" + (", every jump cut" if n_cuts else "")
           + "; its original backdrop is replaced by <Subject 1>."]
    if audio:
        ret.append("<Audio 1>: fully_copy - the original sound of <Video 1> is kept as it is.")
    blocks = list(shots) or ["[Shot 1] <Subject 2> moves exactly as in <Video 1>."]
    tail = []
    if dialogue:
        tail.append(f"{pr['Poss']} lips move with every word and close between sentences.")
    if n_cuts:
        tail.append("Every jump cut happens at the same moment as in <Video 1>.")
    if tail:
        blocks[-1] = blocks[-1].rstrip() + " " + " ".join(tail)
    # the guide opens the first shot by establishing the environment with its features, then the person in it; the
    # camera goes inside each shot in the guide's words (camera_in_shots: a still camera said outright, a moving one
    # left to <Video 1>)
    # <Picture 2> stays out of the shots: a picture cited in a shot is a frame anchor in H3's guide ("the shot's keyframe
    # corresponds to <Picture 2>", ref guide §5.3). Same chunk, same seed (2026-10-13): lean + "its surfaces
    # detailed from <Picture 2>" in Shot 1 zoomed and panned fast, twice; lean with one picture and the full form with
    # two (which names <Picture 2> only in the subject, summary and retention) both held the camera
    feats = ", ".join(place_items(subject)[:4])
    est = (f"The shot establishes <Subject 1>, the {name} environment" + (f" with its {feats}" if feats else "")
           + ", behind <Subject 2>.")
    blocks = camera_in_shots(blocks, video_1, lead=est)
    sound = soundscape(sounds, audio)
    a1 = ("\n<Audio 1> is the synchronized audio track of <Video 1>, and is reused in the target video." if audio else "")
    return ("subject_definitions:\n" + lean_place(subject)
            + f"\n<Subject 2> is {who} in <Video 1>, who keeps {pr['poss']} own appearance and movements."
            + "\n<Video 1> is the source video for the target video edit: " + v1 + a1
            + "\n\nsummary:\n" + summary + "\n\nretention_analysis:\n" + "\n".join(ret)
            + "\n\ndetailed_description:\n"
            # the full form's opening, which says the framing is kept (ref guide §5.2: one or two sentences before
            # [Shot 1])
            + "The target video keeps the camera, framing and timing of <Video 1>, with <Subject 1> behind <Subject 2> "
              "in place of the original backdrop, moving with the camera as the original backdrop did.\n"
            + "\n".join(blocks)
            + "\n\noverall_soundscape:\n" + sound + "\n\nnon_diegetic_music:\nN/A\n")


def split_two_people(text):
    """Two people's short descriptions out of an answer ("a man in a tan jacket; a woman in a red dress") or a hand-set
    target ("the man and the woman"): [first, second] with their articles ("the man"), or fewer if the text names fewer."""
    t = " ".join(re.sub(r"(?i)^\s*(main people|people|answer)\s*:\s*", "", text or "").split()).strip(' ."')
    if not t or t.lower().startswith(("none", "n/a", "no person", "no one")) or len(t) > 320:
        return []
    bits = [b.strip(' ."') for b in re.split(r"\s*;\s*|\s+\band\b\s+(?=(?:the|a|an)\b)", t) if b.strip(' ."')]
    out = []
    for b in bits[:2]:
        b = who_lasting(b) or b
        out.append(re.sub(r"(?i)^(a|an)\b", "the", b) if re.match(r"(?i)(the|a|an)\b", b) else "the " + b)
    return out


def fill_bg2_ref2va_lean(*, subject, whos, pronouns, video_1, shots, sounds, n_shots, dialogue, audio):
    """fill_bg_ref2va_lean for TWO kept people: the place is <Subject 1>, the first person <Subject 2>, the second
    <Subject 3> (both from <Video 1>, fully_preserved). Same sections and labels; the backdrop behind both is replaced."""
    pr2, pr3 = PRONOUNS[pronouns[0]], PRONOUNS[pronouns[1]]
    m = re.match(r"\s*<Subject 1> is the (.*?) environment in <Picture 1>", subject or "")
    name = m.group(1) if m else "new"
    n_cuts = max(0, n_shots - 1)
    who2, who3 = (list(whos) + ["the person", "the other person"])[:2]
    tag = "[video editing + reference generation" + (" + audio reuse]" if audio else "]")
    pics = "<Picture 1> and <Picture 2>" if has_picture_2(subject) else "<Picture 1>"
    v1 = drop_shot_size(video_1) or f"{who2} and {who3} filmed in front of a plain backdrop."
    where = "throughout" if n_shots == 1 else "in " + ", ".join(f"[Shot {k}]" for k in range(1, n_shots + 1))
    summary = (f"{tag} The target video is an edited version of <Video 1>. The backdrop behind <Subject 2> and <Subject 3> "
               f"is replaced by <Subject 1>, the {name} environment from {pics}, while <Subject 2>, <Subject 3>, the camera "
               f"and the action remain as recorded" + (", every jump cut at the same moment" if n_cuts else "")
               + "; wherever both are out of frame, only <Subject 1> shows. Nothing of the original backdrop "
               "remains. The edit runs continuously from the first frame to the last without deviation.")
    ret = [f"<Subject 1> (appears throughout): partially_preserved - the layout, colours, fixtures and furniture of the "
           f"environment in <Picture 1> are the background, seen from the camera's angle in <Video 1> and moving with the "
           f"camera; <Subject 2> and <Subject 3> are lit to match it, with natural shadows"
           + ("; " + picture_2_retention(subject) if has_picture_2(subject) else "."),
           f"<Subject 2> (appears {where}, whenever {pr2['subj']} {'are' if pronouns[0] == 'they' else 'is'} in frame): "
           f"fully_preserved - {pr2['poss']} identity, face, hair, build and clothing, and every movement, position in the "
           f"frame and timing, from <Video 1>.",
           f"<Subject 3> (appears {where}, whenever {pr3['subj']} {'are' if pronouns[1] == 'they' else 'is'} in frame): "
           f"fully_preserved - {pr3['poss']} identity, face, hair, build and clothing, and every movement, position in the "
           f"frame and timing, from <Video 1>, and where {pr3['subj']} {'stand' if pronouns[1] == 'they' else 'stands'} "
           f"or {'sit' if pronouns[1] == 'they' else 'sits'} against <Subject 2>.",
           "<Video 1> (whole-video temporal structure, camera angle and movement, the people): partially_preserved - "
           "the camera's angle and movement, the framing" + (", every jump cut" if n_cuts else "")
           + "; its original backdrop is replaced by <Subject 1>."]
    if audio:
        ret.append("<Audio 1>: fully_copy - the original sound of <Video 1> is kept as it is.")
    blocks = list(shots) or ["[Shot 1] <Subject 2> and <Subject 3> move exactly as in <Video 1>."]
    tail = []
    if dialogue:
        tail.append("Their lips move only with their own lines, in time with the speech in <Audio 1>; everyone else's lips "
                    "move only with their own.")
    if n_cuts:
        tail.append("Every jump cut happens at the same moment as in <Video 1>.")
    if tail:
        blocks[-1] = blocks[-1].rstrip() + " " + " ".join(tail)
    feats = ", ".join(place_items(subject)[:4])
    est = (f"The shot establishes <Subject 1>, the {name} environment" + (f" with its {feats}" if feats else "")
           + ", behind <Subject 2> and <Subject 3>.")
    blocks = camera_in_shots(blocks, video_1, lead=est)
    sound = soundscape(sounds, audio)
    a1 = ("\n<Audio 1> is the synchronized audio track of <Video 1>, and is reused in the target video." if audio else "")
    return ("subject_definitions:\n" + lean_place(subject)
            + f"\n<Subject 2> is {who2} in <Video 1>, who keeps {pr2['poss']} own appearance and movements."
            + f"\n<Subject 3> is {who3} in <Video 1>, who keeps {pr3['poss']} own appearance and movements."
            + "\n<Video 1> is the source video for the target video edit: " + v1 + a1
            + "\n\nsummary:\n" + summary + "\n\nretention_analysis:\n" + "\n".join(ret)
            + "\n\ndetailed_description:\n"
            + "The target video keeps the camera, framing and timing of <Video 1>, with <Subject 1> behind <Subject 2> "
              "and <Subject 3> in place of the original backdrop, moving with the camera as the original backdrop did.\n"
            + "\n".join(blocks)
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
            self.node = None                # the node object too: nothing of Qwen's left to hold on to
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


GGUF_CTX = 16384               # 16 frames at ~530 px plus the long draft instruction and a 2048-token answer
GGUF_TEXT_RESERVE = 4096       # of GGUF_CTX, kept for the instruction and the answer when the frames are sized
GGUF_IMAGE_MAX_TOKENS = 1280   # one picture's cap (the sheet at 1280 px); video frames are shrunk to fit first
GGUF_SYSTEM = ("You are a helpful vision-language assistant. "
               "Answer directly with the final answer only. No <think> and no reasoning.")   # the GGUF node's own
# the large-model template's calls: the Qwen3.8 model card's two sampling sets (instruct / thinking), not the 8B's
# 0.6 / 0.9 / repetition 1.15 (the card: repetition_penalty 1.0; mixing the sets "causes mushy shorts or rambling loops")
LARGE_SAMPLING = {"instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                               "repeat_penalty": 1.0},
                  "think": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                            "repeat_penalty": 1.0}}
LARGE_THINK_TOKENS = 4096       # added to a thinking call's answer budget for the reasoning
LARGE_SYSTEM = ("You describe film footage precisely for a video-editing pipeline. You look at every picture you are "
                "given before answering, report only what is visible, and follow the requested output format exactly.")


def _gguf_arch(path):
    """general.architecture from a GGUF file's header, or ""."""
    try:
        import gguf
        f = gguf.GGUFReader(str(path)).fields.get("general.architecture")
        return bytes(f.parts[f.data[0]]).decode() if f else ""
    except Exception:
        return ""


class GgufQwenRunner:
    """A QwenVL GGUF catalog model (llama.cpp, in-process), with QwenRunner's load / ask / close. Uses the GGUF
    node's own loader, _invoke and image helpers; only the chat handler is picked here: the node always takes
    Qwen3VLChatHandler, but a qwen35-architecture model (Qwen3.5 / 3.6 / 3.8) wants Qwen35ChatHandler, whose
    template closes an empty <think> block when thinking is off (under the Qwen3-VL template it thinks first and
    spends the answer's tokens on it). Refuses a model that isn't on disk: never downloads."""

    def __init__(self, mod, model):
        self.mod = mod
        self.name = model[len(GGUF_PREFIX):] if model.startswith(GGUF_PREFIX) else model
        path, mmproj = gguf_model_file(mod, self.name)
        if path is None or mmproj is None:
            raise DraftError(f"GGUF model {self.name!r} isn't on disk with its mmproj (models/LLM/GGUF): Swap Draft "
                             f"never downloads; place it first or pick another")
        self.model, self.path, self.mmproj = model, str(path), str(mmproj)
        self.arch = _gguf_arch(path)
        self.node = mod.QwenVLGGUFBase()

    def _handler(self):
        """The chat handler class the node's loader should build, as a stand-in for Qwen3VLChatHandler."""
        if self.arch != "qwen35":
            return None
        from llama_cpp.llama_chat_format import Qwen35ChatHandler

        class NoThink(Qwen35ChatHandler):
            def __init__(self, clip_model_path=None, image_max_tokens=-1, verbose=False, **_):
                super().__init__(mmproj_path=clip_model_path, image_max_tokens=image_max_tokens, verbose=verbose,
                                 enable_thinking=False)
        return NoThink

    def load(self):
        import llama_cpp.llama_chat_format as fmt
        h, old = self._handler(), getattr(fmt, "Qwen3VLChatHandler", None)
        if h is not None:
            fmt.Qwen3VLChatHandler = h
        try:
            self.node._load_model(model_name=self.name, device="auto", ctx=GGUF_CTX, n_batch=None, gpu_layers=-1,
                                  image_max_tokens=GGUF_IMAGE_MAX_TOKENS, top_k=None, pool_size=None)
        finally:
            if h is not None:
                fmt.Qwen3VLChatHandler = old
        if self.node.chat_handler is None:
            raise DraftError(f"GGUF model {self.name!r} loaded without its mmproj: it can't see the frames")

    def ask(self, prompt, image=None, video=None, max_tokens=2048, frame_count=16):
        m = self.mod
        pics = []
        if image is not None:
            pics += [m.tensor_to_base64_png(f, max_side=1280) for f in m.sample_video_frames(image, image.shape[0])]
        if video is not None:
            side = m.resolve_safe_video_max_side(video, frame_count=int(frame_count),
                                                 ctx=GGUF_CTX - GGUF_TEXT_RESERVE + 1024, video_frame_size="auto")
            pics += [m.tensor_to_base64_png(f, max_side=side) for f in m.sample_video_frames(video, int(frame_count))]
        s = QWEN_SAMPLING
        return self.node._invoke(system_prompt=GGUF_SYSTEM, user_prompt=prompt, images_b64=[p for p in pics if p],
                                 max_tokens=max_tokens, temperature=s["temperature"], top_p=s["top_p"],
                                 repetition_penalty=s["repetition_penalty"], seed=s["seed"])

    def ask_parts(self, parts, max_tokens=1024, think=False, max_side=1280):
        """One question as text and pictures in order (a str is text, a [H, W, 3] tensor a picture; the template
        puts "Picture n:" before each), with the model card's sampling for the mode (LARGE_SAMPLING). think: the
        model reasons first; its <think> part is dropped from the answer."""
        content = []
        for p in parts:
            if isinstance(p, str):
                content.append({"type": "text", "text": p})
            else:
                b = self.mod.tensor_to_base64_png(p, max_side=max_side)
                if b:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
        h = self.node.chat_handler
        args = getattr(h, "extra_template_arguments", None)
        if args is not None and "enable_thinking" in args:
            args["enable_thinking"] = bool(think)
        s = LARGE_SAMPLING["think" if think else "instruct"]
        try:
            res = self.node.llm.create_chat_completion(
                messages=[{"role": "system", "content": LARGE_SYSTEM}, {"role": "user", "content": content}],
                max_tokens=int(max_tokens) + (LARGE_THINK_TOKENS if think else 0), seed=int(QWEN_SAMPLING["seed"]),
                stop=["<|im_end|>", "<|im_start|>"], **s)
        finally:
            if args is not None and "enable_thinking" in args:
                args["enable_thinking"] = False
        msg = (res.get("choices") or [{}])[0].get("message", {})
        text = msg.get("content") or ""
        return text.split("</think>")[-1].strip()

    def close(self):
        try:
            self.node.clear()
        finally:
            self.node = None
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


ECHO_WORDS = 6        # an answer that repeats the question's first 6+ words is the question handed back, not an answer


def _norm_words(s):
    return re.sub(r"[^\w<>]+", " ", str(s or "").lower()).split()


def echoes_instruction(instruction, answer):
    """True when `answer` contains the opening words of `instruction` (the model handed the question back). Instructions
    shorter than ECHO_WORDS words are never judged: a short answer can legitimately repeat them."""
    opening = _norm_words(instruction)[:ECHO_WORDS]
    if len(opening) < ECHO_WORDS:
        return False
    return " ".join(opening) in " ".join(_norm_words(answer))


class EchoGuard:
    """Wraps a Qwen runner (Q4 fault 5, 2026-11-17: Qwen sometimes answers with its own instruction). An answer that
    contains the instruction's opening words is asked once more; if it echoes again the answer is "" and `echoed`
    counts it, so the draft can leave that chunk's prompt empty and say why. Everything else goes to the runner."""

    def __init__(self, inner, warnings=None):
        self._inner, self._warnings, self.echoed = inner, warnings, 0
        if hasattr(inner, "ask_parts"):
            self.ask_parts = self._ask_parts

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _guard(self, instruction, call):
        out = call()
        if not echoes_instruction(instruction, out):
            return out
        out = call()                                    # redraft once
        if not echoes_instruction(instruction, out):
            return out
        self.echoed += 1
        if self._warnings is not None:
            self._warnings.append(f"Qwen answered with its own instruction twice ({str(instruction)[:50]!r}...): no answer taken")
        return ""

    def ask(self, prompt, *a, **kw):
        return self._guard(prompt, lambda: self._inner.ask(prompt, *a, **kw))

    def _ask_parts(self, parts, *a, **kw):
        first = next((p for p in parts if isinstance(p, str)), "")
        return self._guard(first, lambda: self._inner.ask_parts(parts, *a, **kw))


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


def _process_ram_gb():
    """This process's private memory (what ComfyUI holds, page file included), or None."""
    try:
        import psutil
        mi = psutil.Process().memory_full_info() if hasattr(psutil.Process(), "memory_full_info") else None
        mi = mi or psutil.Process().memory_info()
        return round(float(getattr(mi, "private", None) or getattr(mi, "uss", None) or mi.rss) / 1024 ** 3, 1)
    except Exception:
        return None


def release_memory():
    """After a draft (2026-10-09: the render after a draft ran at 105 s a step with ComfyUI holding 56 GB):
    unload everything ComfyUI has loaded, then let PyTorch hand back its cached GPU memory and its pinned
    system memory. -> {"vram_free": [before, after], "ram": [before, after]} in GB."""
    v0, r0 = free_vram_gb()[0], _process_ram_gb()
    free_comfy_models()
    for _ in range(2):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        host = getattr(torch._C, "_host_emptyCache", None)     # pinned (page-locked) system memory
        if host:
            try:
                host()
            except Exception:
                pass
    return {"vram_free": [v0, free_vram_gb()[0]], "ram": [r0, _process_ram_gb()]}


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


_DRAFT_MODE = contextvars.ContextVar("seamstitch_draft_mode", default=None)


def update_plan_in_draft_mode(plan_file, fn):
    """sp.update_plan(plan_file, fn) for a draft's writes into a mode's own state (subject, a chunk's prompt / draft).
    A draft belongs to the mode the job was in when it started (_DRAFT_MODE, set by _run_draft). If the strip switched
    the job's mode while it ran, fn edits that mode's saved copy under plan["by_mode"] (what switch_mode keeps for a mode
    that is not current) instead of the live fields, so the finished draft can't write into the new mode (Q4 fault 1,
    2026-11-17). Same mode, or no run in progress: the live plan, as before."""
    mode = _DRAFT_MODE.get()

    def upd(p):
        if mode is None or sp.job_mode(p) == mode:
            return fn(p)
        store = p.setdefault("by_mode", {}).setdefault(mode, {"plan": {}, "chunks": {}})
        saved = store.setdefault("chunks", {})
        spl = _planner()
        view = {"job": p.get("job"), "chunks": [dict({"id": c_["id"]}, **(saved.get(c_["id"]) or {}))
                                                 for c_ in p.get("chunks", [])]}
        view.update(store.get("plan") or {})
        res = fn(view)
        for k in spl._MODE_PLAN_KEYS:
            if k in view:
                store.setdefault("plan", {})[k] = view[k]
        for c_ in view["chunks"]:
            saved[c_["id"]] = {k: c_.get(k) for k in spl._MODE_CHUNK_KEYS}
        return res
    return sp.update_plan(plan_file, upd)


def write_chunk_draft(plan_file, chunk_id, text, info):
    """Writes one chunk's draft under the plan lock: into the prompt only if it's empty at this
    moment, otherwise into the draft field. Returns where it went."""
    where = {}
    if _LAYOUT["text"]:
        text = apply_layout(_LAYOUT["text"], text)

    def upd(p):
        c = sp.find_chunk(p, chunk_id)
        if not (c.get("prompt") or "").strip():
            c["prompt"], c["prompt_state"] = text, "draft"
            where["into"] = "prompt"
        else:
            c["draft"] = text
            where["into"] = "draft"
        c["drafted"] = dict(info, at=sp.now(), into=where["into"])
    plan = update_plan_in_draft_mode(plan_file, upd)
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
    update_plan_in_draft_mode(plan_file, upd)
    return where["into"]


def write_mode_subject(plan_file, sentence, pronoun, bg):
    """A drafted subject of the job's kind (a place for background, a character otherwise): it replaces a subject
    of the other kind, which moves to the subject draft (the strip's adopt brings it back); one of the right kind
    already there is kept and the new one waits as the draft."""
    where = {}

    def upd(p):
        cur = (p.get("subject") or "").strip()
        if cur and is_scene_subject(cur) == bg:
            p["subject_draft"] = sentence
            where["into"] = "subject_draft"
        else:
            if cur:
                p["subject_draft"] = cur
            p["subject"] = sentence
            where["into"] = "subject"
            if not bg and pronoun:
                p["subject_pronoun"] = pronoun
    update_plan_in_draft_mode(plan_file, upd)
    return where["into"]


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_json(path, obj):
    _write(path, json.dumps(obj, indent=1, ensure_ascii=False))


def run_draft(plan_file, *, template=None, **kw):
    """The whole draft run. A "my: <name>" template drafts exactly as its base built-in template would (the job's own when none is set)
    and then lays each chunk's prompt out as the user's layout says."""
    mine = None
    if is_user_template(template):
        mine = load_user_templates()[template[len(USER_PREFIX):]]
        template = mine["base"] if mine["base"] in TEMPLATES else DEFAULT_TEMPLATE
    elif (template or "").startswith(USER_PREFIX):          # deleted since the workflow was saved
        template = DEFAULT_TEMPLATE
    _LAYOUT["text"] = mine["text"] if mine else None
    try:
        rep = _run_draft(plan_file, template=template, **kw)
    finally:
        _LAYOUT["text"] = None
    if mine:
        rep["template_info"] = "your template's layout was applied. " + (rep.get("template_info") or "")
    return rep


def _run_draft(plan_file, **kw):
    """The whole draft run, filed under the mode the job is in when it starts (see update_plan_in_draft_mode)."""
    tok = _DRAFT_MODE.set(sp.job_mode(sp.load_plan(plan_file)))
    try:
        return _run_draft_unscoped(plan_file, **kw)
    finally:
        _DRAFT_MODE.reset(tok)


def _run_draft_unscoped(plan_file, *, sheet=None, mode=CHUNK_MODES[0], named=(), qwen_model=DEFAULT_QWEN,
              quantization=QUANTIZATIONS[0], frames_per_chunk=DEFAULT_FRAMES, transcribe=True, word_timings=True,
              extra_instructions="", max_tokens=2048, template=None, qwen_cls=None, omni_cls=None, whisper=None,
              free_models=True,
              sample_vram=True, sheet_2=None):
    """The whole draft run (one queue item). qwen_cls / omni_cls / whisper: injected for tests;
    None = locate the installed nodes (False = treat as missing). Returns the report dict."""
    t_run = time.time()
    plan = sp.load_plan(plan_file)
    # the job's mode picks the kind of prompt; the widget only picks the character template's style
    bg = sp.job_mode(plan) == sp.MODE_BACKGROUND
    if sp.job_mode(plan) == sp.MODE_OBJECT:          # an object job drafts with the object template, whatever's picked
        template = OBJECT_TEMPLATE
    two = sheet_2 is not None                       # a second reference picture is wired
    template_notes = []
    asked, template_info = template, ""
    bg_lean = False
    if bg:
        # the job's mode picks the kind: a character template in the widget (its default is one) gets the full form
        # and leaves the choice between one and two references to the picture; a background one is taken as asked
        if asked in BACKGROUND_TEMPLATES:
            bg_lean = asked in (BACKGROUND_LEAN, BACKGROUND_LEAN_2REF, LARGE_BG_TEMPLATE, LARGE_BG2_TEMPLATE)
            wants_two = asked in (BACKGROUND_2REF, BACKGROUND_LEAN_2REF)
            if wants_two and not two:
                template = _BG_FORMS[(bg_lean, False)]
                template_notes.append("the 2 references template needs a second picture: wire the render group's second "
                                      f"Load Image into sheet_2. Drafted as {template!r}.")
            elif two and not wants_two:
                template = asked
                template_notes.append("sheet_2 is wired but this template doesn't use a second picture: pick "
                                      f"{_BG_FORMS[(bg_lean, True)]!r} to define <Picture 2> in the prompts.")
                two = False
            else:
                template = asked
        else:
            template = _BG_FORMS[(False, two)]
        if asked not in BACKGROUND_TEMPLATES:
            template_info = (f"background job: drafted with {template!r} (the job's mode picks the template; the "
                             f"template widget picks the style of a character job)")
    elif template == OBJECT_TEMPLATE:
        if asked != OBJECT_TEMPLATE:
            template_info = f"object job: drafted with {OBJECT_TEMPLATE!r} (the job's mode picks the template)"
        two = False
    else:
        if asked in BACKGROUND_TEMPLATES or asked == OBJECT_TEMPLATE:
            template_info = (f"character job: {asked!r} is for background jobs, drafted with {DEFAULT_TEMPLATE!r}")
            asked = DEFAULT_TEMPLATE
        template = asked if asked in TEMPLATES else DEFAULT_TEMPLATE
        if template == LEAN_2REF and not two:
            template_notes.append("the 2 references template needs a second picture: wire the render group's second "
                                  "Load Image into sheet_2. Drafted as the lean template.")
            template = LEAN_TEMPLATE
        elif two and template != LEAN_2REF:
            template_notes.append("sheet_2 is wired but this template doesn't use a second picture: pick 'character "
                                  "replace (Ref2VA, timeline, lean, 2 references)' to define <Picture 2> in the prompts.")
            two = False
    jd = os.path.dirname(os.path.abspath(plan_file))
    src = plan["source"]
    path = src["path"]
    fps = int(round(float(src["fps"])))
    if not os.path.isfile(path):
        raise DraftError(f"the source video is missing: {path}")
    chunks = select_chunks(plan, mode, named)
    warnings = list(template_notes)
    report = {"job": plan["job"], "mode": mode, "named": list(named), "chunks": {}, "warnings": warnings,
              "stages": {}, "vram": {"free_gb": []}, "qwen_model": qwen_model, "quantization": quantization,
              "frames_per_chunk": int(frames_per_chunk), "sampling": dict(QWEN_SAMPLING), "template": template,
              "template_info": template_info}
    if named:
        kept = [x for x in named if sp.is_kept(next((c for c in plan["chunks"] if c["id"] == x), None))]
        if kept:
            warnings.append(f"kept chunk(s) {', '.join(kept)} are never drafted")
    # a subject of the other kind (a place on a character job, a character on a background job) is redrafted
    need_subject = (not (plan.get("subject") or "").strip() or mode == CHUNK_MODES[1]
                    or bg != is_scene_subject(plan.get("subject")))
    # a second picture is wired but the subject in use doesn't define it: its line is added under the subject
    need_p2 = two and bool((plan.get("subject") or "").strip()) and not has_picture_2(plan.get("subject"))
    if not chunks and not need_subject and not need_p2:
        report["text"] = f"nothing to draft ({mode}): every rendered chunk has a prompt"
        _say(f"[SeamStitch] Swap Draft: {plan['job']}: {report['text']}")
        return report

    # the models: QwenVL refuses the run when missing; Omni and Whisper degrade
    qcls = locate_qwen() if qwen_cls is None else (qwen_cls or None)
    if qcls is None:
        raise DraftError("SeamStitch Swap Draft Prompts needs ComfyUI-QwenVL (1038lab) in custom_nodes, with the "
                         f"model {qwen_model!r}: install it (Manager: 'ComfyUI-QwenVL'), restart ComfyUI")
    if hasattr(qcls, "ask"):
        qwen = qcls
    elif qwen_model.startswith(GGUF_PREFIX):
        gmod = locate_qwen_gguf()
        if gmod is None:
            raise DraftError(f"{qwen_model!r} needs ComfyUI-QwenVL's GGUF side (llama-cpp-python), which didn't load")
        qwen = GgufQwenRunner(gmod, qwen_model)
    else:
        qwen = QwenRunner(qcls, qwen_model, quantization)
    qwen = EchoGuard(qwen, warnings)
    if template == OBJECT_TEMPLATE:
        return run_object_draft(plan_file, plan, chunks, qwen, sheet, report, warnings, free_models=free_models,
                                t_run=t_run, jd=jd)
    if template == BFS_TEMPLATE:
        return run_bfs_draft(plan_file, plan, chunks, qwen, sheet, report, warnings, free_models=free_models,
                             t_run=t_run, jd=jd)
    if template in LARGE_TEMPLATES and not hasattr(qwen, "ask_parts"):
        fallback = BACKGROUND_LEAN if template in (LARGE_BG_TEMPLATE, LARGE_BG2_TEMPLATE) else LEAN_TEMPLATE
        warnings.append(f"the large-model template needs a GGUF: model; {qwen_model!r} drafted with {fallback!r}")
        template = report["template"] = fallback
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
                # both transcribers inventing over music, apart (test clip c14/c5 2026-10-17: Omni "The
                # reference-to-video generation process is now complete.", Whisper "I'm going to take a look at it."
                # at mean confidence 0.29): no word in common and an unsure Whisper means no speech
                invented = bool(words and aligned and not any(a.get("matched") for a in aligned) and
                                sum(w.get("p", 1) for w in words) / float(len(words)) < WHISPER_ALONE_P)
                if invented:
                    ci["warnings"].append(f"Omni and Whisper heard different words ({po['words'][:40]!r} / "
                                          f"{' '.join(w['word'] for w in words)[:40]!r}), Whisper unsure: dropped "
                                          f"as invented over the music")
                    lines, aligned = [], []
                else:
                    lines = dialogue_lines(aligned, fps, ci["n"]) if words else lines_from_text(po["words"])
                if not words and (wh is not None or word_timings):
                    ci["warnings"].append("no word timings: the lines go to Qwen without frames")
            elif words:
                # a "word" with no letters is punctuation Whisper made up; with Omni hearing no speech at all, two
                # words or fewer are Whisper inventing over music (test clip c5, 2026-10-16: "go." and ". . ." on
                # an instrumental, each written into shot 1 as a line he says)
                # and when their mean confidence is under WHISPER_ALONE_P (c5's inventions, 4 runs: 0.18-0.46; 6 to 12
                # words, "I will do it first. I'm going to get it.")
                good = [w for w in words if w.get("p", 1) >= 0.4 and re.search(r"[^\W\d_]", w.get("word") or "")]
                mean_p = sum(w.get("p", 1) for w in words) / float(len(words))
                if po is not None and not po["words"] and (len(good) <= 2 or mean_p < WHISPER_ALONE_P):
                    if good:
                        ci["warnings"].append(f"Whisper heard {len(good)} word(s) where Omni heard no speech: "
                                              f"dropped (likely invented)")
                    good = []
                aligned = [dict(w, matched=True) for w in unstretch(good)]
                lines = dialogue_lines(aligned, fps, ci["n"]) if good else []
                if good and (omni is not None or transcribe):
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
                s_instr = SCENE_SHEET_INSTRUCTION if bg else SUBJECT_INSTRUCTION
                if template in LARGE_TEMPLATES:
                    raw = qwen.ask_parts([s_instr, sheet[0]], max_tokens=512, think=LARGE_THINK)
                else:
                    raw = qwen.ask(s_instr, image=sheet[:1], max_tokens=512)
                stages["subject_s"] = round(time.time() - t, 2)
                if bg:
                    ps = parse_scene_sheet(raw)
                    sentence = scene_sentence(ps["name"], ps["features"]) if ps["features"] else ""
                    ps["pronoun"] = None
                else:
                    ps = parse_subject(raw)
                    sentence = subject_sentence(ps["name"], ps["appearance"]) if ps["appearance"] else ""
                qlog = s_instr + "\n\n---- Qwen ----\n" + raw
                if sentence and two:
                    # the second reference picture: one more question, its answer on its own line under the subject
                    t2 = time.time()
                    p2_instr = SCENE_SHEET_2_INSTRUCTION if bg else SUBJECT_2_INSTRUCTION
                    raw2 = qwen.ask(p2_instr, image=sheet_2[:1], max_tokens=256)
                    stages["subject_2_s"] = round(time.time() - t2, 2)
                    qlog += "\n\n==== picture 2: instruction ====\n" + p2_instr + "\n\n---- Qwen ----\n" + raw2
                    s2 = picture_2_sentence(parse_detail(raw2), bg, parse_view(raw2))
                    if s2:
                        sentence = sentence + "\n" + s2
                    else:
                        warnings.append("Qwen gave no detail for the second picture: <Picture 2> isn't defined in the "
                                        "subject (write its line there, or redraft)")
                sdir = os.path.join(jd, "drafts", "subject")
                _write(os.path.join(sdir, "qwen.txt"), qlog)
                if sentence:
                    _write(os.path.join(sdir, "draft.txt"), sentence + "\n")
                    into = write_mode_subject(plan_file, sentence, ps["pronoun"], bg)
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
        plan_subject = (plan.get("subject") or "").strip()
        if two and plan_subject and not has_picture_2(plan_subject):
            # the subject in use is kept as it is and gains one line under it: <Picture 2>, as Qwen sees it
            t2 = time.time()
            p2_instr = SCENE_SHEET_2_INSTRUCTION if bg else SUBJECT_2_INSTRUCTION
            raw2 = qwen.ask(p2_instr, image=sheet_2[:1], max_tokens=256)
            stages["picture_2_s"] = round(time.time() - t2, 2)
            _write(os.path.join(jd, "drafts", "subject", "qwen_picture_2.txt"),
                   p2_instr + "\n\n---- Qwen ----\n" + raw2)
            s2 = picture_2_sentence(parse_detail(raw2), bg, parse_view(raw2))
            if s2:
                def add_p2(p):
                    cur = (p.get("subject") or "").strip()
                    if cur and not has_picture_2(cur):
                        p["subject"] = cur + "\n" + s2
                update_plan_in_draft_mode(plan_file, add_p2)
                plan = sp.load_plan(plan_file)
                subject = (plan.get("subject") or "").strip()
                report["picture_2"] = {"into": "subject", "text": s2}
        if two and not has_picture_2(subject):
            warnings.append("a second picture is wired, but the subject in use doesn't define <Picture 2> (there is no "
                            "subject yet, or Qwen gave no detail for the picture), so these prompts don't mention it: "
                            "write its line in the strip's subject box, or draft again.")
        if not two and has_picture_2(subject):
            # the job's subject still carries the picture's line, but no second picture goes to H3 (sheet_2 isn't
            # wired, or the template doesn't use it): a prompt can't name a picture that isn't an input. The job's own
            # subject is left as it is; only the prompts leave the picture out.
            subject = drop_picture_2(subject)
            warnings.append("the job's subject defines <Picture 2>, but no second picture is in use (sheet_2 isn't wired, "
                            "or this template doesn't use it): these prompts leave it out. The subject keeps its line; "
                            "delete it in the strip's subject box if you won't use a second picture.")
        k_frames = max(1, int(frames_per_chunk))
        per_shot = template == TEMPLATES[1]
        timeline = template in (TEMPLATES[2], LEAN_TEMPLATE, LEAN_2REF, LARGE_TEMPLATE)
        lean = template in (LEAN_TEMPLATE, LEAN_2REF, LARGE_TEMPLATE)
        large = template == LARGE_TEMPLATE
        large_bg = template in (LARGE_BG_TEMPLATE, LARGE_BG2_TEMPLATE)
        two_chars = template == LARGE_BG2_TEMPLATE
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
            echoed_before = qwen.echoed
            t = time.time()
            if template in LARGE_TEMPLATES:
                # the shot calls name every object by this list, so it gets their pictures and their thinking: from
                # 24 small frames it called c14's rolled ticket strip "bone", and the shots then wrote "a long bone"
                oidx = sample_indices(ci["n"], LARGE_SHOT_FRAMES)
                oinstr = OBJECTS_INSTRUCTION.format(n=len(oidx))
                oraw = qwen.ask_parts([oinstr] + list(decode_sampled(path, fps, r0, oidx, max_side=LARGE_FRAME_SIDE)),
                                      max_tokens=384, think=LARGE_THINK, max_side=LARGE_FRAME_SIDE)
            else:
                oinstr = OBJECTS_INSTRUCTION.format(n=len(idx))
                oraw = qwen.ask(oinstr, video=video, max_tokens=384, frame_count=len(idx))
            t_obj = round(time.time() - t, 2)
            objs = parse_objects(oraw)
            log = ["==== objects: instruction ====", oinstr, "", "==== objects: Qwen ====", oraw, ""]
            t = time.time()
            # the render's clock (H3's 24 when conforming): placement times and cut times are the ones H3 plays
            clock = tl.conform_rate(fps, sp._settings(plan.get("settings"))["conform_to_24fps"])
            if bg:
                widx = sample_indices(ci["n"], 5)
                wq, wvid = (WHO2_INSTRUCTION if two_chars else WHO_INSTRUCTION).format(n=len(widx)), \
                    decode_sampled(path, fps, r0, widx, max_side=1280)
                wraw = (qwen.ask_parts([wq] + list(wvid), max_tokens=96 if two_chars else 64, think=LARGE_THINK, max_side=1024) if large_bg
                        else qwen.ask(wq, video=wvid, max_tokens=64, frame_count=len(widx)))
                log += ["==== who: Qwen ====", wraw, ""]
                if two_chars:
                    # the two kept people: the chunk's hand-set target ("the man and the woman") wins, else what the model saw
                    by_hand = split_two_people(sp.chunk_target(plan, c) or "")
                    whos = by_hand if len(by_hand) == 2 else (split_two_people(wraw) or by_hand)
                    if len(whos) < 2:
                        ci["warnings"].append("two characters were not found in this chunk (the model named "
                                              f"{len(whos)}): set the chunk's target to both, e.g. 'the man and the woman', then redraft")
                        whos = (whos + ["the first person", "the second person"])[:2]
                    ci["whos"] = whos
                    who = f"{whos[0]} and {whos[1]}"
                    ci["who"] = who
                else:
                    who = " ".join(re.sub(r"(?i)^\s*(main person|person|answer)\s*:\s*", "", wraw).split()).strip(' ."')
                    if who.lower().startswith(("none", "n/a", "no person", "no one")) or len(who) > 160:
                        who = ""
                    who = who_lasting(who)
                    if sp.chunk_target(plan, c):
                        who = sp.chunk_target(plan, c)
                    who = who or "the person"
                    who = who if re.match(r"(?i)(the|a|an)\b", who) else "the " + who
                    who = re.sub(r"(?i)^(a|an)\b", "the", who)
                    ci["who"] = who
                if large_bg and two_chars:
                    # the clip first (each person's pronoun), from one picture per shot, then each shot for both people
                    sidx = sorted({(a + b) // 2 for a, b in ci["shots_rel"]})[:LARGE_SHOT_FRAMES]
                    talk = ", in which someone speaks" if ci["lines"] else ""
                    cinstr = LARGE_BG2_CLIP_INSTRUCTION.format(n=len(sidx), talk=talk, who2=whos[0], who3=whos[1],
                                                               sounds_hint=_sounds_hint(ci["events"], ci["lines"]))
                    draw = qwen.ask_parts([cinstr] + list(decode_sampled(path, fps, r0, sidx, max_side=1024)),
                                          max_tokens=448, think=LARGE_THINK, max_side=1024)
                    log += ["==== clip: instruction ====", cinstr, "", "==== clip: Qwen ====", draw, ""]
                    found = re.findall(r"(?i)\b(she|he|they)\b", _labelled(draw).get("pronoun") or "")
                    ci["pronouns"] = [f.lower() for f in found[:2]] + [pronoun_of(w) for w in whos[len(found[:2]):]]
                    blocks = []
                    k_sh = len(ci["shots_rel"])
                    for i, sh in enumerate(ci["shots_rel"]):
                        mf = moment_frames(sh, fps, cap=LARGE_SHOT_FRAMES)
                        pics = decode_sampled(path, fps, r0, mf, max_side=LARGE_FRAME_SIDE)
                        said = lines_in_shot(ci["lines"], sh, k_sh, i)
                        rel = [dict(ln, frames=[ln["frames"][0] - sh[0], ln["frames"][1] - sh[0]]) if "frames" in ln
                               else ln for ln in said]
                        q = LARGE_BG2_SHOT_INSTRUCTION.format(
                            secs=(sh[1] - sh[0] + 1) / float(fps), who2=whos[0], who3=whos[1],
                            objects=large_objects(objs["objects"]), lines=large_lines(rel, fps),
                            dialogue_rule=_dialogue_rule(bool(rel), "this shot's"), extra=_extra(extra_instructions))
                        ask = [q]
                        for f, pic in zip(mf, pics):
                            ask += [f"\n{(f - sh[0]) / float(fps):.1f} s:", pic]
                        sraw = qwen.ask_parts(ask, max_tokens=800, think=LARGE_THINK, max_side=LARGE_FRAME_SIDE)
                        sj = parse_large_shot(sraw)
                        log += [f"==== shot {i + 1}: large, two characters (frames {[r0 + f for f in mf]}) ====", sraw, ""]
                        d2 = " ".join(str(sj.get("description_2") or sj.get("description") or "").split())
                        d3 = " ".join(str(sj.get("description_3") or "").split())
                        tog = " ".join(str(sj.get("together") or "").split())
                        if not large_visible(sj.get("visible_2", True)) or not d2:
                            d2 = "<Subject 2> is out of frame in this shot."
                        if not large_visible(sj.get("visible_3", True)) or not d3:
                            d3 = "<Subject 3> is out of frame in this shot."
                        if tog and not large_visible(tog):
                            tog = ""
                        blocks.append(f"[Shot {i + 1}] " + " ".join(x for x in (d2, d3, tog) if x))
                    parts = parse_qwen(draw + "\nshots:\n" + "\n".join(blocks), len(ci["shots"]), ci["language"], cap=24)
                elif large_bg:
                    # the clip first (its pronoun goes into every shot question), from one picture per shot
                    sidx = sorted({(a + b) // 2 for a, b in ci["shots_rel"]})[:LARGE_SHOT_FRAMES]
                    talk = ", in which someone speaks" if ci["lines"] else ""
                    cinstr = LARGE_BG_CLIP_INSTRUCTION.format(n=len(sidx), talk=talk, who=who,
                                                              sounds_hint=_sounds_hint(ci["events"], ci["lines"]))
                    draw = qwen.ask_parts([cinstr] + list(decode_sampled(path, fps, r0, sidx, max_side=1024)),
                                          max_tokens=384, think=LARGE_THINK, max_side=1024)
                    log += ["==== clip: instruction ====", cinstr, "", "==== clip: Qwen ====", draw, ""]
                    lp = re.sub(r"[^a-z]", "", ((_labelled(draw).get("pronoun") or "").strip().lower().split() or [""])[0])
                    lpr = PRONOUNS[lp if lp in PRONOUNS else pronoun_of(who)]
                    blocks = []
                    k_sh = len(ci["shots_rel"])
                    for i, sh in enumerate(ci["shots_rel"]):
                        mf = moment_frames(sh, fps, cap=LARGE_SHOT_FRAMES)
                        pics = decode_sampled(path, fps, r0, mf, max_side=LARGE_FRAME_SIDE)
                        said = lines_in_shot(ci["lines"], sh, k_sh, i)
                        rel = [dict(ln, frames=[ln["frames"][0] - sh[0], ln["frames"][1] - sh[0]]) if "frames" in ln
                               else ln for ln in said]
                        q = LARGE_BG_SHOT_INSTRUCTION.format(
                            secs=(sh[1] - sh[0] + 1) / float(fps), who=who, subj=lpr["subj"],
                            objects=large_objects(objs["objects"]), lines=large_lines(rel, fps),
                            dialogue_rule=_dialogue_rule(bool(rel), "this shot's"), extra=_extra(extra_instructions))
                        ask = [q]
                        for f, pic in zip(mf, pics):
                            ask += [f"\n{(f - sh[0]) / float(fps):.1f} s:", pic]
                        sraw = qwen.ask_parts(ask, max_tokens=600, think=LARGE_THINK, max_side=LARGE_FRAME_SIDE)
                        sj = parse_large_shot(sraw)
                        log += [f"==== shot {i + 1}: large (frames {[r0 + f for f in mf]}) ====", sraw, ""]
                        desc = " ".join(str(sj.get("description") or "").split())
                        if not large_visible(sj.get("main_visible", True)) or not desc:
                            # the place still shows: only the person is away (the mask's beats say when, if marked)
                            desc = "<Subject 2> is out of frame in this shot."
                        blocks.append(f"[Shot {i + 1}] {desc}")
                    parts = parse_qwen(draw + "\nshots:\n" + "\n".join(blocks), len(ci["shots"]), ci["language"],
                                       cap=24)
                else:
                    instr = bg_instruction(n_frames=ci["n"], fps=fps, sample_idx=idx, shots_rel=ci["shots_rel"],
                                           lines=ci["lines"], events=ci["events"], who=who, extra=extra_instructions)
                    draw = qwen.ask(instr, video=video, max_tokens=int(max_tokens), frame_count=len(idx))
                    log += ["==== draft: instruction ====", instr, "", "==== draft: Qwen ====", draw, ""]
                    parts = parse_qwen(draw, len(ci["shots"]), ci["language"])
                if two_chars:
                    # both people are labelled by the shot answers already (<Subject 2>, <Subject 3>); the placement beats
                    # of the mask are for one person, so none are written here and the people say where they are
                    ci["pronoun"] = "they"
                    for ln in ci["lines"]:
                        ln["tag"] = "<Subject 2> (S1)"
                    if ci["lines"]:
                        ci["warnings"].append("who says each line isn't worked out for two characters: every line is "
                                              "<Subject 2>'s; set each line's speaker in the strip")
                else:
                    pron = re.sub(r"[^a-z]", "", (_labelled(draw).get("pronoun") or "").strip().lower().split(" ")[0]) \
                        if (_labelled(draw).get("pronoun") or "").strip() else ""
                    # Qwen's pronoun line is sometimes missing: read it off who they are, then off the shots, so the
                    # prompt doesn't say "their" in one section and "her" in the next
                    fallback = pronoun_of(who)
                    if fallback == "they":
                        fallback = pronoun_of(" ".join(parts["shots"]))
                    ci["pronoun"] = pron if pron in PRONOUNS else fallback
                    # the kept person says the lines: <Subject 2> (S1)
                    for ln in ci["lines"]:
                        ln["tag"] = "<Subject 2> (S1)"
                    # Qwen writes about "the man ..." and "He ..."; the prompt's label is <Subject 2>
                    parts["shots"] = [label_person(s, who, ci["pronoun"]) for s in parts["shots"]]
                    # where the kept person is and how big, from the chunk's source mask (place_beats, as the timeline
                    # templates do), and that a held place holds to the end of the shot: on a still shot H3 slowly zoomed
                    # partway through the chunk with nothing in the shot saying the framing holds (2026-10-12)
                    bmask, bwhy = spl.cached_mask(plan, jd, r0, r0 + ci["n"] - 1, 0, sp.chunk_target(plan, c))
                    if bmask is None:
                        ci["warnings"].append(f"no placement lines ({bwhy}): mark this chunk, then redraft")
                    else:
                        parts["shots"] = bg_place_shots(parts["shots"], bmask.numpy(), ci["shots_rel"], fps, clock,
                                                        ci["pronoun"])
                        if parts["shots"] and "At the start" in parts["shots"][0] and \
                                not _AT_CENTRE.search(parts["shots"][0]):
                            parts["video_1"] = drop_centre(parts["video_1"])
            elif not per_shot and not timeline:
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
                if large:
                    # one picture from the middle of each shot, not the chunk's first / middle / last (test clip
                    # c5: those were his close-up, a hand-held ticket and a slump, so video_1 missed the couch scene)
                    sidx = sorted({(a + b) // 2 for a, b in ci["shots_rel"]})[:LARGE_SHOT_FRAMES]
                sinstr = SCENE_INSTRUCTION.format(n=len(sidx), talk=talk, who=who or "the person most in focus",
                                                  sounds_hint=_sounds_hint(ci["events"], ci["lines"]))
                if large:
                    # its first run wrote "with no caption or logo overlaid" into video_1: the question's list, echoed;
                    # its c5 draft said "holding a cigarette" here (the shots say what the hands hold, this doesn't)
                    sinstr += ("\nThe pictures are one from each shot of the clip, in order. Say only what is there: "
                               "leave out anything the list asks about that the pictures don't show, and don't name "
                               "anything held in the hands.")
                    craw = qwen.ask_parts([sinstr] + list(decode_sampled(path, fps, r0, sidx, max_side=1024)),
                                          max_tokens=384, think=LARGE_THINK, max_side=1024)
                else:
                    craw = qwen.ask(sinstr, video=decode_sampled(path, fps, r0, sidx, max_side=1280), max_tokens=384,
                                    frame_count=len(sidx))
                log += ["==== scene: instruction ====", sinstr, "", "==== scene: Qwen ====", craw, ""]
                named = " ".join((_labelled(craw).get("others") or "").split()).strip(' ."')
                if named.lower().startswith(("none", "no one", "nobody", "n/a")) or len(named) > 240:
                    named = ""
                # everyone else the scene names is a <Subject N> of their own (ref guide §2.3); what isn't a person
                # (a crowd, a scoreboard) goes into the background instead (wrestling c4-t002: with them dropped and
                # Qwen's video_1 saying only "a white mat under bright lights", H3 put her in an empty studio)
                people, behind = split_others(named)
                # the scene's own list of what's behind them (wrestling c4: the retention said only "the background")
                for x in re.split(r"[;]", _labelled(craw).get("background") or ""):
                    x = " ".join(x.split()).strip(' ."')
                    # "no caption or logo overlaid" says what isn't there (test clip c5, the 27B)
                    if x and x.lower() not in ("none", "n/a", "nothing") and not _ABSENT.match(x) \
                            and x.lower() not in (b.lower() for b in behind):
                        behind.append(x[0].lower() + x[1:])
                ci["behind"] = behind
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
                # where the person is, how big, and when they leave and come back: from the chunk's cached source
                # mask, on the render's clock (place_beats)
                pmask, pwhy = spl.cached_mask(plan, jd, r0, r0 + ci["n"] - 1, 0, sp.chunk_target(plan, c))
                if pmask is None:
                    ci["warnings"].append(f"no placement lines ({pwhy}): mark this chunk, then redraft")
                    pmask = None
                else:
                    pmask = pmask.numpy()
                    pcov = [float((pmask[k] > 0.5).mean()) for k in range(0, len(pmask), max(1, fps // 4))]
                    pcov = [v for v in pcov if v >= PLACE_MIN_COVER]
                    pref = float(np.median(pcov)) if pcov else 0.0
                ci["placed"] = pmask is not None
                large_seen, introduced = {lab: [] for lab in labels}, []
                for i, sh in enumerate(ci["shots_rel"]):
                    if large:
                        # the whole shot in one call: its pictures in order with their times, the cast by label
                        mf = moment_frames(sh, fps, cap=LARGE_SHOT_FRAMES)
                        pics = decode_sampled(path, fps, r0, mf, max_side=LARGE_FRAME_SIDE)
                        said = lines_in_shot(ci["lines"], sh, k_sh, i)
                        rel = [dict(ln, frames=[ln["frames"][0] - sh[0], ln["frames"][1] - sh[0]]) if "frames" in ln
                               else ln for ln in said]
                        pr = PRONOUNS[pronoun]
                        q = LARGE_SHOT_INSTRUCTION.format(
                            secs=(sh[1] - sh[0] + 1) / float(fps), who=who or "the person most in focus",
                            subj=pr["subj"], Subj=pr["Subj"], poss=pr["poss"], cast=large_cast(labels, people),
                            objects=large_objects(objs["objects"]), lines=large_lines(rel, fps), dialogue_rule=_dialogue_rule(bool(rel), "this shot's"),
                            extra=_extra(extra_instructions))
                        ask = [q]
                        for f, pic in zip(mf, pics):
                            ask += [f"\n{(f - sh[0]) / float(fps):.1f} s:", pic]
                        sraw = qwen.ask_parts(ask, max_tokens=600, think=LARGE_THINK, max_side=LARGE_FRAME_SIDE)
                        sj = parse_large_shot(sraw)
                        log += [f"==== shot {i + 1}: large (frames {[r0 + f for f in mf]}) ====", sraw, ""]
                        # who is in the shot: its list, and anyone it puts beside him or names in the description
                        # (c5's shot 2: both kept people in every picture, an empty list, "beside" naming one)
                        seen_here = json.dumps([sj.get("others_visible"), sj.get("beside"), sj.get("description")])
                        vis = [lab for lab in labels if lab in seen_here]
                        for lab in vis:
                            large_seen[lab].append(i + 1)
                        if vis:
                            seen_in.append(i + 1)
                        speaker_ids(ci["lines"])
                        desc = " ".join(str(sj.get("description") or "").split())
                        if not large_visible(sj.get("main_visible", True)) or not desc:
                            body = empty_shot([(0.0, {"frame": " ".join(str(sj.get("frame") or "").split())})])
                            for ln in said:
                                ln.setdefault("speaker_label", labels[0] if len(labels) == 1 else "another person's voice")
                            speaker_ids(ci["lines"])
                            body += "".join(" " + say_line(ln, ci["language"]) for ln in said)
                            ci.setdefault("empty_shots", []).append(i)
                        else:
                            body = desc
                        # each other person at their first shot (ref guide §5.3), only where the model sees them
                        intro = [lab for lab in vis if lab not in introduced]
                        if intro:
                            body = " ".join(f"{lab}, {people[labels.index(lab)]}, stays exactly as in <Video 1>."
                                            for lab in intro) + " " + body
                            introduced += intro
                        blocks.append(f"[Shot {i + 1}] {body}")
                        continue
                    mf = moment_frames(sh, fps)
                    pics = decode_sampled(path, fps, r0, mf, max_side=1280)
                    moments = []
                    minstr = moment_instruction(who or "the person most in focus", touch=bool(people))
                    for f, pic in zip(mf, pics):
                        mraw = qwen.ask(minstr, image=pic[None], max_tokens=240)
                        moments.append(((f - sh[0]) / float(fps), parse_moment(mraw)))
                    here = [m["others"] for _, m in moments if m.get("others") and
                            m["others"].lower().strip(" .") not in ("none", "no one", "nobody", "n/a")]
                    others += here
                    if not people:
                        seen = others_from_moments(here)
                        if seen:
                            lidx = mf if len(mf) <= 3 else [mf[int(round(k))] for k in np.linspace(0, len(mf) - 1, 3)]
                            lraw = qwen.ask(OTHER_LOOK_INSTRUCTION.format(n=len(lidx), who=who or "the person most in focus",
                                                                          other=seen),
                                            video=decode_sampled(path, fps, r0, lidx, max_side=1280), max_tokens=40,
                                            frame_count=len(lidx))
                            log += [f"==== shot {i + 1}: other_look ({seen}) ====", lraw, ""]
                            look = " ".join((lraw or "").split()).strip(' ."')
                            if 2 <= len(look.split()) <= 12 and not look.lower().startswith(("none", "n/a", "i ", "the main")):
                                seen = look[0].lower() + look[1:]
                            people, labels = [seen], ["<Subject 2>"]
                            ci["warnings"].append(f"the scene named no one else, but {len(here)} moment(s) see "
                                                  f"{seen}: kept as <Subject 2>")
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
                        shot_mask_ok = pmask is not None
                        if shot_mask_ok:
                            # a mask with no one in it where Qwen sees the person (c6: a new chunk took frames 154-196
                            # from an old mark that never found the man) is the mask's fault: warn, and don't place by it
                            blind = [r0 + f for (_, m), f in zip(moments, mf)
                                     if present(m) and f < len(pmask) and mask_place(pmask[f]) is None]
                            if len(blind) >= 2:
                                shot_mask_ok = False
                                ci["warnings"].append(f"shot {i + 1}: the mask is empty at frames {blind[0]}-{blind[-1]} "
                                                      f"where the person is in the picture: re-mark this chunk, then "
                                                      f"redraft (no placement lines for this shot)")
                        if shot_mask_ok:
                            # out of shot by the mask: no pose / hands / gaze line (the beat says where they went)
                            gone = away_frames(pmask, mf, pref)
                            away = [k for k, f in enumerate(mf) if f in gone]
                            if any(present(m) for k, (_, m) in enumerate(moments) if k not in away):
                                for k in away:
                                    moments[k][1]["person"] = "no"
                        dropped = drop_unseen_hands(moments, objs["objects"])
                        if dropped:
                            ci["warnings"].append(f"shot {i + 1}: left out of her hands (not in the frames' object "
                                                  f"list): " + ", ".join(dropped))
                        speaker_ids(ci["lines"])
                        rel = [dict(ln, frames=[ln["frames"][0] - sh[0], ln["frames"][1] - sh[0]]) if "frames" in ln
                               else ln for ln in said]
                        beats = place_beats(pmask, mf, clock, pronoun, pref) if shot_mask_ok else []
                        if any(b[1].lower().find("gone out") >= 0 for b in beats):
                            ci["exits"] = True
                        body = merge_moments(moments, rel, fps, pronoun, ci["language"],
                                             placements=[((f - sh[0]) / float(fps), b) for f, b in beats], lean=lean)
                        hold = hold_sentence(beats, PRONOUNS[pronoun]["Subj"])
                        if hold:
                            body = body.rstrip() + " " + hold
                        if people and here:
                            ridx = mf if len(mf) <= 6 else [mf[int(round(k))] for k in np.linspace(0, len(mf) - 1, 6)]
                            rinstr = RELATION_INSTRUCTION.format(n=len(ridx), who=who or "the person most in focus",
                                                                 other=people[0])
                            rraw = qwen.ask(rinstr, video=decode_sampled(path, fps, r0, ridx, max_side=1280),
                                            max_tokens=96, frame_count=len(ridx))
                            log += [f"==== shot {i + 1}: relation ====", rraw, ""]
                            rs = relation_sentence(rraw, PRONOUNS[pronoun]).replace("the other person", labels[0])
                            if rs:
                                body = rs + " " + body
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
                if large:
                    # each person where the shot calls saw them (a person no call saw: kept everywhere, as before)
                    ci["people"] = [(lab, desc, large_seen[lab] or list(range(1, k_sh + 1)))
                                    for lab, desc in zip(labels, people)]
                parts = parse_qwen(craw + "\nshots:\n" + "\n".join(blocks), len(ci["shots"]), ci["language"], cap=24)
                if behind:
                    parts["video_1"] = with_background(parts.get("video_1") or "", behind)
            t_draft = round(time.time() - t, 2)
            stages["qwen_s"][c["id"]] = {"objects": t_obj, "draft": t_draft}
            _write(os.path.join(ci["dir"], "qwen.txt"), "\n".join(log))
            if qwen.echoed > echoed_before:
                # Qwen handed an instruction back twice for this chunk: no prompt from it (Q4 fault 5)
                ci["warnings"].append("Qwen answered with its own instruction twice: nothing was drafted for this chunk; "
                                      "redraft it, or write its prompt by hand")
                report["chunks"][c["id"]] = {"into": "nothing", "render": ci["render"], "shots": len(ci["shots"]),
                                             "lines": len(ci["lines"]), "warnings": ci["warnings"],
                                             "dialogue_coverage": None, "unseen": [], "carried_over": [],
                                             "plan_rev": None, "chars": 0}
                _say(f"[SeamStitch] Swap Draft: {c['id']} {r0}-{r1}: Qwen echoed its instruction twice, nothing drafted")
                continue
            parts["shots"] = tidy_shots(parts["shots"], [a / float(clock) for a, _ in ci["shots_rel"]])
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
            if bg and two_chars:
                text = fill_bg2_ref2va_lean(subject=subject, whos=ci["whos"], pronouns=ci["pronouns"],
                                            video_1=parts["video_1"], shots=parts["shots"], sounds=parts["sounds"],
                                            n_shots=len(ci["shots"]), dialogue=bool(ci["lines"]), audio=audio_ok)
            else:
                text = (fill_bg_ref2va_lean if bg_lean else fill_bg_ref2va)(
                                      subject=subject, who=ci["who"], video_1=parts["video_1"], shots=parts["shots"],
                                      sounds=parts["sounds"], n_shots=len(ci["shots"]), dialogue=bool(ci["lines"]),
                                      audio=audio_ok, pronoun=ci["pronoun"]) if bg else \
                    (fill_ref2va_lean if lean else fill_ref2va)(
                                   subject=subject, video_1=parts["video_1"], shots=parts["shots"], sounds=parts["sounds"],
                                   n_shots=len(ci["shots"]), dialogue=bool(ci["lines"]), audio=audio_ok, pronoun=pronoun,
                                   who=ci.get("who"), others=ci.get("others", False), people=ci.get("people", ()),
                                   speaking=any(ln.get("speaker_label") for ln in ci["lines"]),
                                   placed=bool(ci.get("placed")), exits=bool(ci.get("exits")),
                                   **({"behind": ci.get("behind", ())} if lean else {}),
                                   **({"full_look": True} if large else {}))
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
    if template_info:
        lines.append(template_info)
    if report.get("subject"):
        lines.append(f"subject -> {report['subject']['into']}")
    if report.get("picture_2"):
        lines.append("subject: added the <Picture 2> line under it (the rest of the subject is as it was)")
    for cid, r in report["chunks"].items():
        lines.append(f" {cid} {r['render'][0]}-{r['render'][1]}: -> {r['into']}, {r['shots']} shot(s), "
                     f"{r['lines']} line(s)" + (f", {len(r['warnings'])} warning(s)" if r["warnings"] else ""))
    lines += [f" ⚠ {w}" for w in warnings]
    if report["vram"].get("device_peak_gb") is not None:
        lines.append(f"VRAM peak (device) {report['vram']['device_peak_gb']} GB")
    report["text"] = "\n".join(lines)
    _write_json(os.path.join(jd, "drafts", f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"), report)
    return report


def clean_bfs_details(raw):
    """The model's attribute list as one clause for "<Subject 1> is the ... shown in <Picture 1>"."""
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S | re.I).strip()
    m = re.search(r"is the (.+?) shown in <Picture 1>", t, flags=re.S)   # it wrote the whole sentence anyway
    if m:
        t = m.group(1)
    t = t.strip().strip("`").strip()
    t = re.sub(r"^(?:\*\*)?(?:description|attributes?|answer)\s*:?\s*(?:\*\*)?\s*", "", t, flags=re.I)
    lines = [ln.strip(" -*\t").rstrip(" .;") for ln in t.splitlines() if ln.strip(" -*\t")]
    t = ", ".join(lines)
    t = re.sub(r"\s+", " ", t).strip(" .,;")
    t = re.sub(r"^(?:a|an|the)\s+", "", t, flags=re.I)
    return t


def fill_bfs_body_swap(details, target=""):
    """BFS's body_swap template: the {target} form (Civitai trainedWords[1]) when the job names who to replace, the
    'main subject of the guide video' form (trainedWords[0]) otherwise."""
    target = (target or "").strip()
    if target and target.lower() not in ("person", "the person", "a person"):
        who = target if re.match(r"(?i)^(the|a|an)\s", target) else "the " + target
        return BFS_BODY_SWAP.format(details=details, who=who, their="their", who_short=f"{who} in the guide video",
                                    who_place=who)
    return BFS_BODY_SWAP.format(details=details, who="the main subject of the guide video", their="its",
                                who_short="the guide subject", who_place="the guide subject")


def run_bfs_draft(plan_file, plan, chunks, qwen, sheet, report, warnings, *, free_models=True, t_run=None, jd=None):
    """The BFS template's draft: one question to the model about the sheet, then the same body_swap prompt into every
    chunk (into the prompt when it's empty, else into the draft field, as the other templates)."""
    t_run = time.time() if t_run is None else t_run
    stages = report["stages"]
    if sheet is None:
        raise DraftError("the BFS template describes the character from the sheet: wire the sheet into the Draft node")
    try:
        t = time.time()
        if free_models:
            free_comfy_models()
        stages["free_s"] = round(time.time() - t, 2)
        t = time.time()
        qwen.load()                     # the GGUF runner's ask_parts needs the model in memory (as run_object_draft)
        stages["qwen_load_s"] = round(time.time() - t, 2)
        t = time.time()
        img = sheet[0]
        if hasattr(qwen, "ask_parts"):
            raw = qwen.ask_parts([BFS_DETAILS_INSTRUCTION, img], max_tokens=768, think=LARGE_THINK)
        else:
            raw = qwen.ask(BFS_DETAILS_INSTRUCTION, image=sheet, max_tokens=768)
        stages["bfs_details_s"] = round(time.time() - t, 2)
        details = clean_bfs_details(raw)
        if len(details.split(",")) < 6:
            warnings.append(f"the model's description is short ({details[:120]!r}): check it, or re-draft")
        text = fill_bfs_body_swap(details, plan.get("target") or "")
        os.makedirs(os.path.join(jd, "drafts"), exist_ok=True)
        _write(os.path.join(jd, "drafts", "bfs_details_raw.txt"), raw or "")
        _say(f"[SeamStitch] Swap Draft: BFS description ({stages['bfs_details_s']} s): {details[:200]}")
        report["bfs_details"] = details
        for c in chunks:
            into, rev = write_chunk_draft(plan_file, c["id"], text,
                                          {"warnings": [], "files": "drafts/bfs_details_raw.txt", "template": BFS_TEMPLATE})
            report["chunks"][c["id"]] = {"into": into, "render": list(c["render"]), "shots": 0, "lines": 0,
                                         "warnings": [], "dialogue_coverage": None, "unseen": [], "carried_over": [],
                                         "plan_rev": rev, "chars": len(text)}
    finally:
        try:
            qwen.close()
        except Exception:
            pass
    report["total_s"] = round(time.time() - t_run, 2)
    lines = [f"job {plan['job']}: BFS body_swap prompt into {len(report['chunks'])} chunk(s) in {report['total_s']} s",
             f"description: {report.get('bfs_details', '')}"]
    lines += [f" {cid} {r['render'][0]}-{r['render'][1]}: -> {r['into']}" for cid, r in report["chunks"].items()]
    lines += [f" \u26a0 {w}" for w in warnings]
    report["text"] = "\n".join(lines)
    _write_json(os.path.join(jd, "drafts", f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"), report)
    return report


def object_details_question(target, action):
    """The sheet question for a thing (not a flat picture). On a replace it says what the thing stands in for, so a
    dark, cropped photo of a white boot is described as footwear and not as "the bandaged limb" (2026-10-21, the c10
    shoe job: Picture 1 was a low-resolution crop of a white boot with a frilled sock cuff)."""
    if action == sp.OBJECT_ADD or not (target or "").strip():
        return OBJECT_DETAILS_INSTRUCTION
    return (OBJECT_DETAILS_INSTRUCTION.replace(
        "describe how it looks as ONE comma-separated attribute list: what it is,",
        f"describe how it looks as ONE comma-separated attribute list. This picture shows what will take the place of "
        f"{the_target(target)} in a video, so name what it is as that kind of thing (even if the picture is dark, small "
        f"or only part of it is visible) and describe it as such, not as something else it resembles. Start with what it is,"))


def the_target(target):
    t = (target or "").strip() or "the marked object"
    return t if re.match(r"(?i)^(the|a|an|his|her|their|its)\s", t) else "the " + t


def parse_object_addition(raw):
    """The `addition:` line of the scene answer ("" when the model didn't write one)."""
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S | re.I).strip()
    m = re.search(r"(?im)^[ \t*#>-]*addition[ \t*]*:[ \t*]*(.+)$", t)
    return re.sub(r"\s+", " ", m.group(1)).strip(" *") if m else ""


def parse_object_scene(raw, with_people=False):
    """The model's lines -> (video_1, placement) or, with_people, (video_1, placement, [people]); "" / [] when it
    didn't write one."""
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S | re.I).strip()
    if with_people:
        m = re.search(r"(?im)^[ \t*#>-]*people[ \t*]*:[ \t*]*(.+)$", t)
        people = [] if not m else [x.strip(" .*") for x in m.group(1).split(";")
                                   if x.strip(" .*") and x.strip(" .*").lower() not in ("none", "no one", "nobody")]
        return parse_object_scene(raw) + (people,)
    out = {}
    for key in ("video_1", "placement"):
        m = re.search(rf"(?im)^[ \t*#>-]*{key}[ \t*]*:[ \t*]*(.+)$", t)
        out[key] = re.sub(r"\s+", " ", m.group(1)).strip(" *") if m else ""
    lines = [re.sub(r"\s+", " ", ln).strip(" *-") for ln in t.splitlines() if ln.strip(" *-")]
    if not out["video_1"] and not out["placement"] and len(lines) == 2:    # the two lines, unlabelled (the 27B, c2)
        out["video_1"], out["placement"] = lines
    return out["video_1"], out["placement"]


_BODY_PART_FIRST = re.compile(r"(?i)^\s*(?:(?:a|an|the|one|two|his|her|their)\s+)?(?:(?:bare|human|left|right|lower|upper)\s+)*"
                              r"(?:legs?|foot|feet|ankles?|knees?|hands?|arms?|fingers?|toes?)\b")
_TARGET_STOP = {"the", "and", "with", "from", "that", "this", "black", "white", "red", "blue", "green", "grey", "gray",
                "brown", "pink", "yellow", "orange", "purple", "dark", "light", "old", "new", "small", "large", "big"}


def object_people(people, target, replace=True):
    """The scene answer's people without the thing being replaced, as (kept, dropped): a person line that is only a
    body part ("a bare leg ending in a black shoe") or, on a replace, that names the target ("black shoes" -> shoe)
    is dropped, so the prompt never declares as kept exactly what the edit replaces (2026-10-21, the c10 shoe job:
    Subject 2 was "a bare leg ending in a black shoe", fully_preserved, beside "Subject 3 replaces the black shoes").
    On an add the target is a place on a person ("his shoulder"), so naming it never drops that person."""
    words = [w[:-1] if w.endswith("s") and len(w) > 4 else w
             for w in re.findall(r"[a-z]{4,}", (target or "").lower()) if w not in _TARGET_STOP] if replace else []
    kept, dropped = [], []
    for p in people or []:
        low = p.lower()
        (dropped if _BODY_PART_FIRST.match(p) or any(w in low for w in words) else kept).append(p)
    return kept, dropped


def fill_object(details, video_1, placement, target, action, people=None, picture=True, addition=""):
    """The object prompt. people None: the first form (the object the only subject). people given (2026-10-21, after
    H3 cast the added jay as THE man): every person the model saw is declared first as kept exactly as filmed
    (fully_preserved), as the character prompts declare the others, and the object is the next subject. picture
    False: the object is defined by the text alone (no <Picture 1>, no retention line for it)."""
    v1, pl = (video_1 or "").strip(), (placement or "").strip()
    pl = pl[0].upper() + pl[1:] if pl else ""
    dot = lambda x: x if not x or x.endswith(".") else x + "."     # noqa: E731
    addition = (addition or "").strip()
    if addition and action == sp.OBJECT_ADD and people is None:
        people = []
    if people is None:
        form = OBJECT_FORMS["add" if action == sp.OBJECT_ADD else "replace"]
        return form.format(details=details, video_1=dot(v1), placement=dot(pl), the_target=the_target(target)).replace("  ", " ")
    tt = the_target(target)
    add = action == sp.OBJECT_ADD
    k = len(people) + 1
    obj = f"<Subject {k}>"
    subj = [f"<Subject {i + 1}> is {p if re.match(r'(?i)^(a|an|the)\s', p) else 'the ' + p} in <Video 1>, kept exactly as filmed."
            for i, p in enumerate(people)]
    subj.append(f"{obj} is the {details} shown in <Picture 1>." if picture else f"{obj} is {details}.")
    tag = "[video editing + reference generation + audio reuse]" if picture else "[video editing + audio reuse]"
    mine = bool(addition) and add                       # the user's own brief: a new subject in the scene, not a thing resting on the target
    what = (f"{obj} is added to the scene" if mine else f"{obj} is added on {tt}") if add else f"{obj} replaces {tt}"
    labels = [f"<Subject {i + 1}>" for i in range(len(people))]
    ret = [f"<Subject {i + 1}> (appears throughout): fully_preserved - {p} keeps their own face, hair, clothes, body, "
           f"position and movements from the source in every frame." for i, p in enumerate(people)]
    if picture:
        ret.append(f"{obj} (appears {'from when it enters the scene' if mine else 'wherever ' + tt + ' is in frame'}): attribute_transfer - {obj} keeps the look of <Picture 1>"
                   + (", at a natural size for the scene, as described below." if mine else f", at a natural size for the scene, resting on {tt} and moving with it." if add
                      else f", in the place, size and perspective of {tt}."))
    ret.append(f"<Video 1> (whole-video temporal structure, camera, background, lighting): partially_preserved - "
               f"everything in <Video 1> stays as filmed except the {'added' if add else 'replaced'} {obj}: every cut, "
               f"the camera, the framing, the background and the lighting.")
    ret.append("<Audio 1>: fully_copy - the original sound of <Video 1> is kept as it is.")
    body = (f"The target video is <Video 1> as filmed, with one change: {what}. {dot(pl)} "
            + ((f"{dot(addition.replace('the new subject', obj).replace('The new subject', obj))} {obj} is at a natural size for the room, lit by "
                f"the scene's own light, and moves naturally. " if mine else
                f"{obj} rests on {tt} and moves with it in every frame, at a natural size, lit by the scene's own light, "
                f"and it moves a little on its own as a living thing would. ") if add else
               f"{obj} keeps the place, size and movement of {tt} in every frame, lit by the scene's own light. ")
            + (f"{_and_list(labels)} {'stays' if len(people) == 1 else 'stay'} exactly as in <Video 1>: the "
               f"same face, hair, clothes, expressions and movements." if people else ""))
    return ("subject_definitions:\n" + "\n".join(subj) + "\n"
            + f"<Video 1> is the source video for the target video edit: {dot(v1)}\n"
            + "<Audio 1> is the synchronized audio track of <Video 1>, and is reused in the target video.\n\n"
            + f"summary:\n{tag} The target video is an edited version of <Video 1>. {what}{'' if mine else ' in every frame'}, while "
            + _and_list(labels + ["the background", "the camera", "the action"]) + " remain as recorded.\n\n"
            + "retention_analysis:\n" + "\n".join(ret) + "\n\n"
            + "detailed_description:\n" + body.strip() + "\n\n"
            + "overall_soundscape:\nThe original sound of <Video 1>.\n\nnon_diegetic_music:\nN/A").replace("  ", " ")


def _planner():
    try:
        from . import swap_planner as m
    except ImportError:
        import swap_planner as m
    return m


def _shrink(frames, max_side):
    """A float IMAGE batch with its longer side at most max_side."""
    h, w = int(frames.shape[1]), int(frames.shape[2])
    if max(h, w) <= max_side:
        return frames.contiguous()
    import torch.nn.functional as F
    s = max_side / float(max(h, w))
    out = F.interpolate(frames.permute(0, 3, 1, 2), size=(max(1, int(round(h * s))), max(1, int(round(w * s)))),
                        mode="area")
    return out.permute(0, 2, 3, 1).contiguous()


def person_light_crop(plan, jd, c, frame, full):
    """The person already in the frame, alone: a close crop of the target's mask on source frame `frame` from `full` (a float
    IMAGE [1, H, W, 3] of that whole frame), everything outside the mask greyed, so the model reads the light's colour from skin,
    hair and clothes and not from the furniture. None when the target has no mask on that frame."""
    import numpy as np
    import torch
    try:
        from . import swap_keep as sk
    except ImportError:
        import swap_keep as sk
    import cv2
    src = plan["source"]
    fr, (w, h) = int(round(float(src["fps"]))), (int(src["width"]), int(src["height"]))
    target = (sp.chunk_target(plan, c) or "").strip()
    g = sk.ZoneGuide(plan, jd, fr, (w, h), int(frame), int(frame), target)
    m = g.masks.get(int(frame))
    if m is None:
        return None
    m = (m[..., 0] if m.ndim == 3 else m) > 127
    H, W = int(full.shape[1]), int(full.shape[2])
    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    pad = int(0.06 * max(xs.max() - xs.min(), ys.max() - ys.min())) + 4
    x0, x1, y0, y1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1), max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    k = cv2.dilate(m, np.ones((5, 5), np.uint8))[y0:y1, x0:x1].astype(np.float32)[None, :, :, None]
    img = full[:, y0:y1, x0:x1, :].cpu().numpy()
    out = img * k + 0.5 * (1.0 - k)
    return torch.from_numpy(out.astype(np.float32)).contiguous()


def qwen_anchor_question(plan, c, with_crop=False):
    """The large model's question for one chunk's Qwen instruction: where the new thing goes and how it touches, from the zone settings."""
    o = sp.object_settings(plan)
    target = (sp.chunk_target(plan, c) or "the person").strip()
    tgt = target if target.lower().startswith(("the ", "a ", "an ", "his ", "her ")) else f"the {target}"
    brief = (o.get("brief") or "").strip().rstrip(".")
    # where the new thing goes is the person's own request, seen against the frame; the zone only says where H3 redraws and is not told to the model
    where = (f"exactly as the person making the edit asks: \"{brief}\" (follow its placement and action, and make it fit what is visible in <image1>)"
             if brief else "at a natural place in the room, all of it inside the picture")
    lightsrc = (f" A third picture, <image3>, shows only {tgt} as lit in the scene (everything else greyed out): name the colour of the "
                f"light from <image3> alone, but never mention <image3> in the paragraph you write (the editing model is not given it): state the "
                f"colours of the light directly." if with_crop else "")
    return QWEN_ANCHOR_INSTRUCTION.format(where=where, contact="", size="", target=tgt, lightsrc=lightsrc)


def clean_anchor_text(raw):
    """The model's answer as the one-paragraph instruction, or None when it is not usable: a JSON {"rewritten_prompt": ...} or a code
    fence is unwrapped, lines are joined, and it must name both <image1> and <image2> (Qwen's own tags)."""
    t = (raw or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        t = t[4:].strip() if t.lower().startswith("json") else t
    if t.startswith("{"):
        try:
            t = str(json.loads(t).get("rewritten_prompt") or "")
        except ValueError:
            m = re.search(r'"rewritten_prompt"\s*:\s*"(.*?)"\s*[,}]', t, flags=re.S)
            t = m.group(1) if m else t
    t = " ".join(t.split()).strip('" ')
    return t if "<image1>" in t and "<image2>" in t and "<image3>" not in t and len(t) > 120 else None


def write_anchor_text(plan_file, chunk_id, text):
    """Into the chunk's Qwen instruction box when it is empty; a typed one is never replaced. Returns where it went."""
    box = {}

    def upd(p):
        c = sp.find_chunk(p, chunk_id)
        if (c.get("anchor_text") or "").strip():
            box["into"] = "nothing (the chunk already has its own Qwen instruction)"
            return
        c["anchor_text"] = text
        box["into"] = "the chunk's Qwen instruction box"
    sp.update_plan(plan_file, upd)
    return box["into"]


def run_object_draft(plan_file, plan, chunks, qwen, sheet, report, warnings, *, free_models=True, t_run=None, jd=None):
    """An object job's draft: the sheet described once, each chunk's scene and the target's placement from a few of
    its frames, into OBJECT_FORMS (into the prompt when it's empty, else into the draft field)."""
    t_run = time.time() if t_run is None else t_run
    stages = report["stages"]
    o = sp.object_settings(plan)
    text_only = bool((o.get("text") or "").strip()) and not o.get("use_picture", True)
    if sheet is None and not text_only:
        raise DraftError("an object job describes the new object from the sheet: wire the sheet into the Draft node, "
                         "or give the object as text (object settings: text, use_picture false)")
    action = o["action"]
    brief = (o.get("brief") or "").strip()
    keep_people = bool(o.get("keep_people", True))
    src = plan["source"]
    path, fps = src["path"], int(round(float(src["fps"])))
    large = hasattr(qwen, "ask_parts")
    try:
        t = time.time()
        if free_models:
            free_comfy_models()
        stages["free_s"] = round(time.time() - t, 2)
        t = time.time()
        qwen.load()
        stages["qwen_load_s"] = round(time.time() - t, 2)
        t = time.time()
        if text_only:                           # the object as you wrote it ("a blue jay"), no picture
            raw = details = o["text"].strip()
            report["object_question"] = "text"
        else:
            dq = OBJECT_PICTURE_INSTRUCTION if OBJECT_PICTURE_TARGET.search(plan.get("target") or "") else OBJECT_DETAILS_INSTRUCTION
            report["object_question"] = "picture" if dq is OBJECT_PICTURE_INSTRUCTION else "object"
            if dq is OBJECT_DETAILS_INSTRUCTION:
                dq = object_details_question(plan.get("target"), action)
            if brief:
                dq += OBJECT_BRIEF_NOTE.format(brief=brief)
            raw = (qwen.ask_parts([dq, sheet[0]], max_tokens=512, think=LARGE_THINK) if large
                   else qwen.ask(dq, image=sheet, max_tokens=512))
            details = clean_bfs_details(raw)
        stages["object_details_s"] = round(time.time() - t, 2)
        if not text_only and len(details.split(",")) < 3:
            warnings.append(f"the model's object description is short ({details[:120]!r}): check it, or re-draft")
        os.makedirs(os.path.join(jd, "drafts"), exist_ok=True)
        _write(os.path.join(jd, "drafts", "object_details_raw.txt"), raw or "")
        report["object_details"] = details
        _say(f"[SeamStitch] Swap Draft: object ({stages['object_details_s']} s): {details[:200]}")
        for c in chunks:
            r0, r1 = c["render"]
            n = r1 - r0 + 1
            idx = sorted({int(round(i * (n - 1) / max(1, OBJECT_FRAMES - 1))) for i in range(OBJECT_FRAMES)})
            target = sp.chunk_target(plan, c)
            # the model sees what H3 will: the render's crop around the target (2026-10-20, test clip c2: shown the
            # whole frame, with the painting a strip along the top, it called the skull-faced figure the painting)
            crop = None
            if sp.object_settings(plan).get("crop", True):
                try:
                    crop = _planner().object_crop(plan, jd, r0, r1, target, int(src["width"]), int(src["height"]))
                except Exception as e:
                    warnings.append(f"{c['id']}: the crop couldn't be read ({e}); the model saw the whole frame")
            frames = decode_sampled(path, fps, r0, idx, max_side=100000)
            if crop:
                x0, y0, x1, y1 = crop
                frames = frames[:, y0:y1, x0:x1, :]
            frames = _shrink(frames, LARGE_FRAME_SIDE if large else QWEN_FRAME_MAX_SIDE)
            q = OBJECT_SCENE_INSTRUCTION.format(n=len(idx), secs=n / float(fps), the_target=the_target(target))
            if brief and action == sp.OBJECT_ADD:
                q = q.replace("Answer in exactly three lines:", "Answer in exactly four lines:", 1)
                q = q.replace("\nWrite about what you see, not about an edit.", OBJECT_ADDITION_LINE.format(brief=brief.replace('"', "'"))
                              + "\nWrite about what you see, not about an edit (except the addition line).", 1)
            t = time.time()
            sraw = (qwen.ask_parts([q] + list(frames), max_tokens=400, think=LARGE_THINK) if large
                    else qwen.ask(q, video=frames, max_tokens=400, frame_count=len(idx)))
            video_1, placement, people = parse_object_scene(sraw, with_people=True)
            addition = parse_object_addition(sraw) if brief and action == sp.OBJECT_ADD else ""
            people, dropped = object_people(people, target, replace=action != sp.OBJECT_ADD)
            cw = [] if video_1 and placement else ["the model's scene answer lacked a line: check the prompt"]
            if addition and re.search(r"(?i)(does not|doesn't|do not|is not|isn't|not)[^.]{0,40}(appear|visible|seen|present|shown|in the (frames|pictures|clip))", addition):
                cw.append(f"the model described the new subject as absent ({addition[:100]!r}): dropped; your own words stand in for it. Redraft, or write the line by hand")
                addition = f"As requested: {brief.rstrip('.')}."
            if brief and action == sp.OBJECT_ADD and not addition:
                cw.append("the model wrote no addition line: the prompt says the new subject is added on the target; check it")
            if not brief and action == sp.OBJECT_ADD:
                cw.append("no brief: write what to add in the zone row (e.g. 'a man with green hair sitting next to her'), then redraft")
            if dropped:
                cw.append(f"left out of the kept people (a body part, or the target itself): {'; '.join(dropped)}")
            p_target, p_action = target, action
            if o.get("guide") == "placeholder":     # H3 replaces the marked grey placeholder drawn on the target
                p_target = "marked grey placeholder shape on " + the_target(target)
                p_action = sp.OBJECT_REPLACE
            text = fill_object(details, video_1, placement, p_target, p_action, people=people if keep_people else None,
                               picture=not text_only, addition=addition)
            _write(os.path.join(jd, "drafts", f"{c['id']}_object_scene_raw.txt"), sraw or "")
            into, rev = write_chunk_draft(plan_file, c["id"], text, {"warnings": cw, "template": OBJECT_TEMPLATE,
                                                                     "files": f"drafts/{c['id']}_object_scene_raw.txt"})
            if sp.object_zone(plan) and (o.get("anchor") or "qwen") == "qwen" and sheet is not None and not text_only:
                try:
                    a_i = max(0, min(n - 1, int(round(float(o.get("anchor_at", 0.0)) * fps))))
                    af = decode_sampled(path, fps, r0, [a_i], max_side=100000)
                    pcrop = None
                    try:
                        pcrop = person_light_crop(plan, jd, c, r0 + a_i, af)
                    except Exception as e:
                        cw.append(f"the person's light crop couldn't be made ({e}): the model reads the light from the whole frame")
                    if pcrop is not None:
                        pcrop = _shrink(pcrop, LARGE_FRAME_SIDE if large else QWEN_FRAME_MAX_SIDE)
                        try:                     # what the model was shown to read the light from, to look at
                            import numpy as _np
                            import cv2 as _cv2
                            _cv2.imwrite(os.path.join(jd, "drafts", f"{c['id']}_light_crop.png"),
                                         (_np.clip(pcrop[0].cpu().numpy(), 0, 1)[:, :, ::-1] * 255).astype(_np.uint8))
                        except Exception:
                            pass
                    if crop:
                        x0, y0, x1, y1 = crop
                        af = af[:, y0:y1, x0:x1, :]
                    af = _shrink(af, LARGE_FRAME_SIDE if large else QWEN_FRAME_MAX_SIDE)
                    qq = qwen_anchor_question(plan, c, with_crop=pcrop is not None and large)
                    t_a = time.time()
                    parts = [qq, "<image1> (the frame):", af[0], "<image2> (the reference):", sheet[0]]
                    if pcrop is not None and large:
                        parts += ["<image3> (the person already in the frame, alone, as lit by the scene):", pcrop[0]]
                    araw = (qwen.ask_parts(parts, max_tokens=700,
                                           think=LARGE_THINK) if large
                            else qwen.ask(qq, image=sheet, max_tokens=700))
                    atext = clean_anchor_text(araw)
                    _write(os.path.join(jd, "drafts", f"{c['id']}_qwen_anchor_raw.txt"), araw or "")
                    if atext:
                        where_to = write_anchor_text(plan_file, c["id"], atext)
                        cw.append(f"Qwen instruction written ({round(time.time() - t_a, 1)} s) into {where_to}")
                    else:
                        cw.append("the model's Qwen instruction was unusable (no <image1>/<image2>, or too short): the default sentence "
                                  "stays; see drafts/" + c["id"] + "_qwen_anchor_raw.txt")
                except Exception as e:                    # the H3 prompt is already written: never lose it over the anchor text
                    cw.append(f"the Qwen instruction could not be written: {e}")
            def _keep_notes(p, cid=c["id"], notes=list(cw)):         # what the draft said about this chunk, for the strip to show beside "redraft"
                ch = sp.find_chunk(p, cid)
                ch["drafted"] = dict(ch.get("drafted") or {}, warnings=notes)
            try:
                sp.update_plan(plan_file, _keep_notes)
            except Exception:
                pass
            report["chunks"][c["id"]] = {"into": into, "render": [r0, r1], "shots": 1, "lines": 0, "warnings": cw,
                                         "dialogue_coverage": None, "unseen": [], "carried_over": [], "plan_rev": rev,
                                         "chars": len(text), "seconds": round(time.time() - t, 2)}
    finally:
        try:
            qwen.close()
        except Exception:
            pass
    report["total_s"] = round(time.time() - t_run, 2)
    lines = [f"job {plan['job']}: object {action} prompt into {len(report['chunks'])} chunk(s) in {report['total_s']} s",
             f"object: {report.get('object_details', '')}"]
    lines += [f" {cid} {r['render'][0]}-{r['render'][1]}: -> {r['into']}" for cid, r in report["chunks"].items()]
    lines += [f" \u26a0 {w}" for w in warnings]
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
                "qwen_model": (qwen_models(), {"default": DEFAULT_QWEN, "tooltip":
                    "A QwenVL model on disk. GGUF: entries are QwenVL's GGUF vision models (llama.cpp, with their "
                    "mmproj); quantization doesn't apply to them."}),
                "quantization": (QUANTIZATIONS, {"default": QUANTIZATIONS[0]}),
                "frames_per_chunk": ("INT", {"default": DEFAULT_FRAMES, "min": 2, "max": 48, "tooltip":
                    "Frames Qwen sees, spread over the chunk's render range (per shot: shared out by shot length, "
                    "at least 4 each). The QwenVL node itself uses 16."}),
                "transcribe": ("BOOLEAN", {"default": True, "tooltip":
                    "Omni Captioner Transcribe on each chunk's audio: the words and the audio events."}),
                "word_timings": ("BOOLEAN", {"default": True, "tooltip":
                    "faster-whisper large-v3-turbo: word timings, so each line lands in its shot and moment."}),
                "template": (template_choices(), {"default": DEFAULT_TEMPLATE, "tooltip":
                    "timeline: one full-size picture every half second, a caption each, merged into "
                    "each shot with its lines at their times; with the chunk marked, timed lines from its mask say "
                    "where the person is, how framed, and when they leave and come back, and with someone else in "
                    "the clip each picture is also asked where they touch. per shot: each shot from its own frames in one call. "
                    "lean (the default): the timeline's questions and placement, written short (one look sentence, the "
                    "character's look moved onto the person, the background named once, hand lines only when she "
                    "handles something). lean + 2 references: lean with a second picture (the sheet_2 socket) defined "
                    "as <Picture 2>. The first: one call for the whole chunk. The job's mode (the strip's switch) "
                    "decides the kind of template, whatever this says: a background replace job always drafts with "
                    "a background template (pick 'background replace' here to match; its lean form writes the six "
                    "sections short; the 2 references ones need sheet_2, and a wired sheet_2 picks the full 2 "
                    "references one when this isn't a background template): the sheet is a "
                    "place (<Subject 1>), the person is kept (<Subject 2>); a chunk with nobody in it still drafts "
                    "the person in, so write that prompt by hand. A character job given a background template "
                    "drafts with the default character one. The draft's report says which was used."}),
                "extra_instructions": ("STRING", {"default": "", "multiline": True, "tooltip":
                    "Added to Qwen's rules for every chunk."}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 4096}),
            },
            "optional": {
                "sheet": ("IMAGE", {"tooltip": "The character sheet (the render group's Load Image): the subject is "
                                               "drafted from it once per job."}),
                # appended after the sheet socket, so a saved workflow's inputs and widget values keep their places
                "free_memory_after": ("BOOLEAN", {"default": True, "tooltip":
                    "When the draft is done: unload everything ComfyUI has loaded and hand back PyTorch's cached GPU "
                    "and pinned memory, so the next render starts clean (Qwen, Omni and Whisper are closed either "
                    "way). The node's output says how much was freed."}),
                "qwen_questions": ("STRING", {"default": questions_text(), "multiline": True, "tooltip":
                    "What the draft asks Qwen, one ### section per question, pre-filled with the built-in text. Edit "
                    "a section to change that question; delete it, or clear the box, for the built-in one. The "
                    "questions actually sent are in each chunk's drafts/<chunk>/qwen.txt."}),
                "template_form": ("STRING", {"default": USER_FORM_SKELETON, "multiline": True, "tooltip":
                    "Your own template, for any job: how the prompt is laid out. Write {summary}, {subject_definitions}, {retention_analysis}, "
                    "{detailed_description}, {overall_soundscape}, {non_diegetic_music} where the drafter's piece should go ({prompt} = the whole draft), "
                    "and add your own words anywhere. Give it a name below and press 'save template': it joins the template list as 'my: <name>'. "
                    "Pick a saved one to load it here."}),
                "template_name": ("STRING", {"default": "", "tooltip": "The name to save the template box under (the same name replaces that template)."}),
                "template_example": ("STRING", {"default": "", "multiline": True, "tooltip":
                    "Read-only: an example of what your layout gives for the 'template_base' picked, with stand-in answers. It updates when you change the base, "
                    "the layout or the template. Nothing you type here is used."}),
                "template_base": (base_choices(), {"default": AUTO_BASE, "tooltip":
                    "Which built-in template your layout is built on: it decides the questions the model is asked and how the prompt is worked out; your layout then "
                    "only places the pieces. 'automatic' uses the job's own (character, background or object)."}),
                "sheet_2": ("IMAGE", {"tooltip":
                    "A second reference picture (the render group's second Load Image, the one in the H3 node's "
                    "ref_image_1). Qwen describes it once per job and the subject defines it as <Picture 2>: a second "
                    "view of the same place on a background job, of the same character on a character job (pick the "
                    "'2 references' template). The subject gains the <Picture 2> line the first time a draft runs "
                    "with it wired (the rest of the subject is kept as it is)."}),
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
              max_tokens=2048, sheet=None, qwen_questions="", free_memory_after=True, sheet_2=None,
              template_form="", template_name="", template_base="", template_example=""):
        plan_file, named = parse_draft_plan(draft_plan)
        if not plan_file:
            raise DraftError("draft_plan is empty: queue a draft run from the Planner")
        overrides, qwarn = question_overrides(qwen_questions)
        for w in qwarn:
            _say("[SeamStitch] Swap Draft: " + w)
        if overrides:
            names = [n for n, const, _ in QUESTIONS if const in overrides]
            _say("[SeamStitch] Swap Draft: using your edited questions: " + ", ".join(names))
        with edited_questions(overrides):
            rep = run_draft(plan_file, sheet=sheet, mode=chunks, named=named, qwen_model=qwen_model,
                            quantization=quantization, frames_per_chunk=frames_per_chunk, transcribe=transcribe,
                            word_timings=word_timings, extra_instructions=extra_instructions, max_tokens=max_tokens,
                            template=template, sheet_2=sheet_2)
        notes = qwarn + ([f"edited questions: {', '.join(names)}"] if overrides else [])
        if free_memory_after:
            fr = release_memory()
            v, r = fr["vram_free"], fr["ram"]
            msg = ("memory freed after the draft: VRAM free "
                   + (f"{v[0]:.1f} -> {v[1]:.1f} GB" if None not in v else "?")
                   + (f", ComfyUI's RAM {r[0]} -> {r[1]} GB" if None not in r else ""))
            _say("[SeamStitch] Swap Draft: " + msg)
            notes.append(msg)
        if notes:
            rep["text"] = "\n".join(notes) + "\n" + rep["text"]
        _say("[SeamStitch] Swap Draft: " + rep["text"].replace("\n", "\n[SeamStitch] Swap Draft: "))
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("seamstitch_swap_plan", {"job": rep["job"]})
        except Exception:
            pass
        return {"ui": {"text": [rep["text"]]}, "result": (rep["text"],)}


# the node's save / delete template buttons (js/draft_templates.js)
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/seamstitch/draft_template/list")
    async def _draft_templates_list(request):
        return web.json_response({"templates": load_user_templates(), "choices": template_choices()})

    @PromptServer.instance.routes.post("/seamstitch/draft_template/preview")
    async def _draft_template_preview(request):
        try:
            body = await request.json()
            ex, res, note = preview_layout(body.get("text", ""), body.get("base", ""))
            return web.json_response({"ok": True, "example": ex, "text": res, "note": note})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/seamstitch/draft_template/save")
    async def _draft_template_save(request):
        try:
            body = await request.json()
            entry = save_user_template(body.get("name", ""), body.get("text", ""), body.get("base", ""))
            return web.json_response({"ok": True, "entry": entry, "choices": template_choices()})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    @PromptServer.instance.routes.post("/seamstitch/draft_template/delete")
    async def _draft_template_delete(request):
        try:
            body = await request.json()
            name = delete_user_template(body.get("entry", ""))
            return web.json_response({"ok": True, "deleted": name, "choices": template_choices()})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
except Exception:                                    # no server (tests): the routes are only for the node's buttons
    pass
