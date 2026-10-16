"""The drafter's second reference picture (swap_draft sheet_2): <Picture 2> is defined in the job's subject, once per
job; in the prompts it is cited inside <Subject 1>'s own line (H3's guide keeps a standalone <Picture N> line for a
target frame or a shot-planning anchor), and the summary and retention name it. The lean background templates keep
the guide's labels and establish the place in the first shot."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_plan as sp  # noqa: E402
import swap_draft as sd  # noqa: E402
from test_swap import job  # noqa: E402,F401
from test_swap_draft import BgQwen, FakeQwen, SUBJECT, _draft, _sheet  # noqa: E402

DETAIL = "a close view of a brown herringbone fabric sofa with a dark navy wall behind it"
P2_PLACE = ("; <Picture 2> provides only its surface texture, colours and light, never the camera's distance or "
            "framing: ")
P2_CHAR = "; <Picture 2> provides only a closer look at the same character: "


class BgQwen2(BgQwen):
    """The place, then (once) the second picture of the place."""

    def ask(self, prompt, image=None, video=None, max_tokens=2048, frame_count=16):
        if "second reference of the same place" in prompt:
            self.calls.append({"prompt": prompt, "image": image is not None, "frames": None})
            return f"view: closer\ndetail: {DETAIL}."
        return super().ask(prompt, image=image, video=video, max_tokens=max_tokens, frame_count=frame_count)


class CharQwen2(FakeQwen):
    def ask(self, prompt, image=None, video=None, max_tokens=2048, frame_count=16):
        if "second reference of the same character" in prompt:
            self.calls.append({"prompt": prompt, "image": image is not None, "frames": None})
            return "**detail:** pink wavy hair with bangs, red lips and a round red clown nose"
        if "Who is the main person" in prompt:
            self.calls.append({"prompt": prompt})
            return "a woman in a red dress"
        if "The main person is" in prompt:
            self.calls.append({"prompt": prompt})
            return "video_1: She sits in a diner booth.\nsounds: chatter.\nothers: none"
        if prompt.startswith("This is one frame"):
            self.calls.append({"prompt": prompt})
            return "person: yes\npose: sitting in a booth\nlook: to the side\nhands: rests them on the table\nframe: a booth"
        if prompt.startswith("Is the") or prompt.startswith("Is there"):
            self.calls.append({"prompt": prompt})
            return "yes"
        return super().ask(prompt, image=image, video=video, max_tokens=max_tokens, frame_count=frame_count)


def _bg(job):
    sp.update_plan(job["plan"], lambda p: p.update(mode="background"))


def _no_standalone_picture_2(text):
    """The guide: an image that only defines a scene or a character is cited inside the subject's line."""
    return not any(ln.startswith("<Picture 2>") for ln in text.splitlines())


# ---------------------------------------------------------------- the pieces

def test_parse_detail_takes_the_label_or_the_whole_answer():
    assert sd.parse_detail("detail: a brown sofa, a navy wall.") == "a brown sofa, a navy wall"
    assert sd.parse_detail("**Detail:** a brown sofa") == "a brown sofa"
    assert sd.parse_detail("a brown sofa, a navy wall") == "a brown sofa, a navy wall"
    assert sd.parse_detail("") == ""


def test_picture_2_is_defined_in_the_kind_of_subject_it_belongs_to():
    assert sd.picture_2_sentence("a close view", True) == "<Picture 2> is a second view of the same environment: a close view."
    assert sd.picture_2_sentence("her face", False) == "<Picture 2> is a second reference of <Subject 1>: her face."
    assert sd.picture_2_sentence("", True) == ""
    assert sd.has_picture_2("x\n<Picture 2> is a second view") and not sd.has_picture_2(SUBJECT)


def test_the_picture_2_line_is_folded_into_the_subjects_own_line():
    place = "<Subject 1> is the old diner environment in <Picture 1>, featuring red stools."
    out = sd.fold_picture_2(place + "\n<Picture 2> is a closer view of the same environment: a sofa.", True)
    assert out == place.rstrip(".") + P2_PLACE + "a sofa."
    out = sd.fold_picture_2(SUBJECT + "\n<Picture 2> is a second reference of <Subject 1>: her face.\n<Subject 2> is x.", False)
    assert out.splitlines() == [SUBJECT.rstrip(".") + P2_CHAR + "her face.", "<Subject 2> is x."]
    assert sd.fold_picture_2(place, True) == place                       # no picture 2: nothing changes
    assert sd.fold_picture_2("", True) == ""


def test_the_two_reference_templates_are_registered_and_the_default_is_unchanged():
    assert sd.TEMPLATES[4] == sd.LEAN_2REF and tuple(sd.TEMPLATES[5:]) == sd.BACKGROUND_TEMPLATES
    assert sd.BACKGROUND_TEMPLATES == (sd.BACKGROUND, sd.BACKGROUND_2REF, sd.BACKGROUND_LEAN, sd.BACKGROUND_LEAN_2REF)
    assert sd.DEFAULT_TEMPLATE == sd.LEAN_TEMPLATE
    assert ("bg_place_2" in [n for n, _, _ in sd.QUESTIONS]) and ("subject_2" in [n for n, _, _ in sd.QUESTIONS])
    assert list(sd.SeamStitchSwapDraft.INPUT_TYPES()["optional"])[-1] == "sheet_2"


def test_lean_subject_folds_the_picture_2_line():
    subj = SUBJECT + "\n<Picture 2> is a second reference of <Subject 1>: her face."
    out = sd.lean_subject(subj)
    assert out == "<Subject 1> is the clown angel girl in <Picture 1>: a young woman" + P2_CHAR + "her face."


def test_the_fills_name_picture_2_only_when_the_subject_defines_it():
    kw = dict(video_1="a booth.", shots=["[Shot 1] She sits."], sounds="room", n_shots=1, dialogue=False, audio=False,
              pronoun="she", who="a woman in a red dress")
    plain = sd.fill_ref2va_lean(subject=SUBJECT, **kw)
    two = sd.fill_ref2va_lean(subject=SUBJECT + "\n<Picture 2> is a second reference of <Subject 1>: her face.", **kw)
    sd.parse_sections(two)
    assert "<Picture 2>" not in plain and "from <Picture 1> and <Picture 2> onto" in two
    sub = sd.parse_sections(two)["subject_definitions"]
    assert P2_CHAR + "her face." in sub and _no_standalone_picture_2(sub)
    # the full character form folds it too
    full = sd.fill_ref2va(subject=SUBJECT + "\n<Picture 2> is a second reference of <Subject 1>: her face.", **kw)
    assert P2_CHAR + "her face." in sd.parse_sections(full)["subject_definitions"]
    place = "<Subject 1> is the old diner environment in <Picture 1>, featuring red stools."
    bkw = dict(who="a man in a black t-shirt", video_1="a handheld camera.", shots=["[Shot 1] <Subject 2> walks."],
               sounds="steps", n_shots=1, dialogue=False, audio=False, pronoun="he")
    b1 = sd.fill_bg_ref2va(subject=place, **bkw)
    b2 = sd.fill_bg_ref2va(subject=place + "\n<Picture 2> is a closer view of the same environment: a sofa.", **bkw)
    assert "<Picture 2>" not in b1
    s = sd.parse_sections(b2)
    assert "environment from <Picture 1> and <Picture 2>." in s["summary"]
    assert "<Picture 2> adds only surface detail" in s["retention_analysis"]
    assert P2_PLACE + "a sofa." in s["subject_definitions"] and _no_standalone_picture_2(s["subject_definitions"])


# ---------------------------------------------------------------- end to end

def test_background_draft_with_a_second_picture(job):
    _bg(job)
    q = BgQwen2()
    rep = _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=q, template=sd.TEMPLATES[0])
    p = sp.load_plan(job["plan"])
    first, second = p["subject"].split("\n")          # the job's subject keeps the line as drafted
    assert sd.is_scene_subject(first) and second == f"<Picture 2> is a closer view of the same environment: {DETAIL}."
    assert rep["template"] == sd.BACKGROUND_2REF and rep["subject"]["into"] == "subject"
    asked = [c for c in q.calls if "second reference of the same place" in c["prompt"]]
    assert len(asked) == 1 and asked[0]["image"]
    s = sd.parse_sections(p["chunks"][0]["prompt"])
    assert P2_PLACE + DETAIL + "." in s["subject_definitions"] and _no_standalone_picture_2(s["subject_definitions"])
    assert "environment from <Picture 1> and <Picture 2>." in s["summary"]
    assert "<Picture 2> adds only surface detail" in s["retention_analysis"]
    # the questions Qwen was sent are kept with the subject's other files
    log = open(os.path.join(job["dir"], "drafts", "subject", "qwen.txt"), encoding="utf-8").read()
    assert "==== picture 2: instruction ====" in log and DETAIL in log


def test_without_a_second_picture_nothing_changes(job):
    _bg(job)
    q = BgQwen2()
    rep = _draft(job, sheet=_sheet(), qwen_cls=q, template=sd.TEMPLATES[0])
    p = sp.load_plan(job["plan"])
    assert "\n" not in p["subject"] and rep["template"] == sd.BACKGROUND
    assert not any("second reference of the same place" in c["prompt"] for c in q.calls)
    assert "<Picture 2>" not in p["chunks"][0]["prompt"]


def test_an_existing_subject_gains_the_picture_2_line_and_nothing_else_changes(job):
    place = "<Subject 1> is the old diner environment in <Picture 1>, featuring red stools."
    sp.update_plan(job["plan"], lambda p: p.update(subject=place, mode="background"))
    q = BgQwen2()
    rep = _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=q)
    p = sp.load_plan(job["plan"])
    assert p["subject"] == place + f"\n<Picture 2> is a closer view of the same environment: {DETAIL}."
    assert rep["picture_2"]["into"] == "subject" and not rep["warnings"]
    assert "subject: added the <Picture 2> line" in rep["text"]
    # the prompts drafted in the same run already name it, inside the place's line
    s = sd.parse_sections(p["chunks"][0]["prompt"])
    assert P2_PLACE + DETAIL + "." in s["subject_definitions"]
    assert "environment from <Picture 1> and <Picture 2>." in s["summary"]
    # one question for the picture, no subject call (the place was already there)
    assert len([c for c in q.calls if "second reference of the same place" in c["prompt"]]) == 1
    assert not any("shows a place" in c["prompt"] for c in q.calls)
    assert os.path.isfile(os.path.join(job["dir"], "drafts", "subject", "qwen_picture_2.txt"))


def test_the_picture_2_line_is_added_once(job):
    place = "<Subject 1> is the old diner environment in <Picture 1>, featuring red stools."
    sp.update_plan(job["plan"], lambda p: p.update(subject=place, mode="background"))
    _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=BgQwen2())
    q = BgQwen2()
    sp.update_plan(job["plan"], lambda p: [c.update(prompt="", prompt_state="empty") for c in p["chunks"]])
    rep = _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=q)
    p = sp.load_plan(job["plan"])
    assert p["subject"].count("<Picture 2>") == 1 and "picture_2" not in rep
    assert not any("second reference of the same place" in c["prompt"] for c in q.calls)


def test_a_wired_picture_alone_updates_the_subject_when_every_chunk_has_a_prompt(job):
    # the one-step fix for prompts drafted before the picture was wired: queue a draft, the subject gets its line
    place = "<Subject 1> is the old diner environment in <Picture 1>, featuring red stools."
    sp.update_plan(job["plan"], lambda p: p.update(subject=place, mode="background"))
    sp.update_plan(job["plan"], lambda p: [c.update(prompt="kept as is", prompt_state="edited") for c in p["chunks"]])
    rep = _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=BgQwen2())
    p = sp.load_plan(job["plan"])
    assert "<Picture 2>" in p["subject"] and rep["picture_2"]["into"] == "subject"
    assert all(c["prompt"] == "kept as is" for c in p["chunks"])          # a draft never touches a prompt
    assert "nothing to draft" not in rep["text"] and "added the <Picture 2> line" in rep["text"]


def test_no_detail_from_qwen_leaves_the_subject_alone_with_a_warning(job):
    class NoDetail(BgQwen2):
        def ask(self, prompt, image=None, video=None, max_tokens=2048, frame_count=16):
            if "second reference of the same place" in prompt:
                self.calls.append({"prompt": prompt, "image": image is not None, "frames": None})
                return ""
            return super().ask(prompt, image=image, video=video, max_tokens=max_tokens, frame_count=frame_count)
    place = "<Subject 1> is the old diner environment in <Picture 1>, featuring red stools."
    sp.update_plan(job["plan"], lambda p: p.update(subject=place, mode="background"))
    rep = _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=NoDetail())
    assert sp.load_plan(job["plan"])["subject"] == place
    assert any("doesn't define <Picture 2>" in w for w in rep["warnings"])


def test_a_character_draft_with_the_two_reference_template(job):
    q = CharQwen2()
    rep = _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=q, template=sd.LEAN_2REF)
    p = sp.load_plan(job["plan"])
    assert rep["template"] == sd.LEAN_2REF
    first, second = p["subject"].split("\n")
    assert first.startswith("<Subject 1> is the clown angel girl") and second == (
        "<Picture 2> is a second reference of <Subject 1>: pink wavy hair with bangs, red lips and a round red clown nose.")
    assert len([c for c in q.calls if "second reference of the same character" in c["prompt"]]) == 1
    s = sd.parse_sections(p["chunks"][0]["prompt"])
    assert (P2_CHAR + "pink wavy hair with bangs, red lips and a round red clown nose.") in s["subject_definitions"]
    assert _no_standalone_picture_2(s["subject_definitions"])
    assert "from <Picture 1> and <Picture 2> onto" in s["retention_analysis"]


def test_the_template_and_the_socket_must_agree_on_a_character_job(job):
    # the template without a picture: the lean template, with a note
    rep = _draft(job, sheet=_sheet(), qwen_cls=CharQwen2(), template=sd.LEAN_2REF)
    assert rep["template"] == sd.LEAN_TEMPLATE and any("needs a second picture" in w for w in rep["warnings"])
    assert "<Picture 2>" not in sp.load_plan(job["plan"])["subject"]
    # a picture with another template: not used, with a note
    sp.update_plan(job["plan"], lambda p: [p.update(subject="", subject_draft="")] +
                   [c.update(prompt="", prompt_state="empty") for c in p["chunks"]])
    q = CharQwen2()
    rep = _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=q, template=sd.LEAN_TEMPLATE)
    assert any("doesn't use a second picture" in w for w in rep["warnings"])
    assert not any("second reference of the same character" in c["prompt"] for c in q.calls)
    assert "<Picture 2>" not in sp.load_plan(job["plan"])["subject"]


# ---------------------------------------------------------------- the dropdown lists the background templates too

def test_the_dropdown_lists_both_kinds_and_the_default_stays_a_character_one():
    opt = sd.SeamStitchSwapDraft.INPUT_TYPES()["required"]["template"]
    assert opt[0] == sd.TEMPLATES and opt[1]["default"] == sd.LEAN_TEMPLATE
    assert sd.BACKGROUND in opt[0] and sd.BACKGROUND_2REF in opt[0]


def test_a_background_job_says_which_template_it_used_when_the_widget_is_a_character_one(job):
    _bg(job)
    rep = _draft(job, sheet=_sheet(), qwen_cls=BgQwen(), template=sd.LEAN_TEMPLATE)
    assert rep["template"] == sd.BACKGROUND and f"background job: drafted with {sd.BACKGROUND!r}" in rep["template_info"]
    assert rep["text"].splitlines()[1] == rep["template_info"] and not rep["warnings"]


def test_a_background_template_in_the_widget_is_taken_as_asked_without_a_note(job):
    _bg(job)
    rep = _draft(job, sheet=_sheet(), qwen_cls=BgQwen(), template=sd.BACKGROUND)
    assert rep["template"] == sd.BACKGROUND and rep["template_info"] == "" and not rep["warnings"]


def test_the_2_references_background_template_needs_the_picture(job):
    _bg(job)
    rep = _draft(job, sheet=_sheet(), qwen_cls=BgQwen2(), template=sd.BACKGROUND_2REF)
    assert rep["template"] == sd.BACKGROUND and any("needs a second picture" in w for w in rep["warnings"])
    assert "<Picture 2>" not in sp.load_plan(job["plan"])["subject"]


def test_the_plain_background_template_leaves_a_wired_picture_unused_with_a_note(job):
    _bg(job)
    q = BgQwen2()
    rep = _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=q, template=sd.BACKGROUND)
    assert rep["template"] == sd.BACKGROUND and any("doesn't use a second picture" in w for w in rep["warnings"])
    assert not any("second reference of the same place" in c["prompt"] for c in q.calls)
    assert "<Picture 2>" not in sp.load_plan(job["plan"])["subject"]


def test_a_character_job_given_a_background_template_drafts_the_default_character_one(job):
    for t in sd.BACKGROUND_TEMPLATES:
        sp.update_plan(job["plan"], lambda p: [p.update(subject="", subject_draft="")] +
                       [c.update(prompt="", prompt_state="empty") for c in p["chunks"]])
        rep = _draft(job, sheet=_sheet(), template=t, qwen_cls=CharQwen2())
        assert rep["template"] == sd.DEFAULT_TEMPLATE
        assert f"character job: {t!r} is for background jobs" in rep["template_info"]
        assert sp.load_plan(job["plan"])["subject"].startswith("<Subject 1> is the clown angel girl")


# ---------------------------------------------------------------- the lean background templates

PLACE = ("<Subject 1> is the ship-themed diner environment in <Picture 1>, featuring navy and white striped walls, a frosted "
         "window with red spindles, a white counter with a blue edge, red stools, a jukebox with chrome trim, a round wall "
         "lamp, framed photos, a checkered floor, a coat rack, a menu board.")
BKW = dict(who="a man in a black t-shirt", video_1="a handheld camera follows him.", shots=["[Shot 1] <Subject 2> walks in."],
           sounds="footsteps", n_shots=1, dialogue=False, audio=True, pronoun="he")
LEAN_PLACE_LINE = ("<Subject 1> is the ship-themed diner environment in <Picture 1>, featuring navy and white striped "
                   "walls, a frosted window, a white counter, red stools, a jukebox, a round wall lamp, framed photos, "
                   "a checkered floor")


def test_lean_place_keeps_the_first_items_without_their_with_tails():
    assert sd.lean_place(PLACE) == LEAN_PLACE_LINE + "."
    assert sd.place_items(PLACE)[:4] == ["navy and white striped walls", "a frosted window", "a white counter", "red stools"]
    assert sd.place_items("<Subject 1> is some place.") == []
    assert sd.lean_place("<Subject 1> is some place.") == "<Subject 1> is some place."
    # a <Picture 2> line below it is folded into the first line, as the guide asks
    out = sd.lean_place(PLACE + "\n<Picture 2> is a closer view of the same environment: a sofa.")
    assert out == LEAN_PLACE_LINE + P2_PLACE + "a sofa."


def test_the_lean_background_fill_is_short_and_keeps_the_six_sections():
    full = sd.fill_bg_ref2va(subject=PLACE, **BKW)
    lean = sd.fill_bg_ref2va_lean(subject=PLACE, **BKW)
    s = sd.parse_sections(lean)
    assert len(lean) < len(full)          # still shorter, but only by the place list and the retention wording
    assert "environment from <Picture 1>, while <Subject 2>, the camera and the action remain as recorded" in s["summary"]
    assert "wherever <Subject 2> is out of frame, only <Subject 1> shows" in s["summary"]
    ret = s["retention_analysis"]
    assert "<Subject 1> (appears throughout): partially_preserved" in ret
    assert "<Subject 2> (appears throughout, whenever he is in frame): fully_preserved - his identity" in ret
    assert "<Video 1> (whole-video temporal structure, camera angle and movement, the person): partially_preserved" in ret
    assert "its original backdrop is replaced by <Subject 1>" in ret and "<Audio 1>: fully_copy" in ret and "lit to match it" in ret
    assert "<Picture 2>" not in lean
    # the guide: a one or two sentence opening before [Shot 1], then the first shot establishes the place
    dd = s["detailed_description"]
    assert dd.startswith("The target video keeps the camera, framing and timing of <Video 1>, with <Subject 1> behind "
                      "<Subject 2> in place of the original backdrop, moving with the camera")
    assert ("[Shot 1] The shot establishes <Subject 1>, the ship-themed diner environment with its navy and white striped "
            "walls, a frosted window, a white counter, red stools, behind <Subject 2>. <Subject 2> walks in.") in dd
    assert "The edit runs continuously from the first frame to the last without deviation." in s["summary"]
    # the backdrop is replaced outright; the "colour-marked area" sentence is gone (ours, it named the marker; 2026-10-14)
    assert "Nothing of the original backdrop remains" in s["summary"] and "colour-marked" not in s["summary"]


def test_the_lean_background_fill_names_picture_2_only_when_defined():
    two = sd.fill_bg_ref2va_lean(subject=PLACE + "\n<Picture 2> is a closer view of the same environment: a sofa.", **BKW)
    s = sd.parse_sections(two)
    assert "environment from <Picture 1> and <Picture 2>," in s["summary"]
    # the layout from <Picture 1> only, as in the full form; <Picture 2> named for surface detail
    assert "environment in <Picture 1> are the background" in s["retention_analysis"]
    assert "<Picture 2> adds only surface detail" in s["retention_analysis"]
    assert ("<Picture 2> adds only surface detail, and the camera's distance and framing come only from <Video 1>."
            in s["retention_analysis"])
    sub = s["subject_definitions"]
    assert LEAN_PLACE_LINE + P2_PLACE + "a sofa." in sub and _no_standalone_picture_2(sub) and "second view" not in sub


def test_a_lean_background_draft_end_to_end(job):
    _bg(job)
    full = _draft(job, sheet=_sheet(), qwen_cls=BgQwen(), template=sd.BACKGROUND)
    full_prompt = sp.load_plan(job["plan"])["chunks"][0]["prompt"]
    sp.update_plan(job["plan"], lambda p: [p.update(subject="", subject_draft="")] +
                   [c.update(prompt="", prompt_state="empty") for c in p["chunks"]])
    rep = _draft(job, sheet=_sheet(), qwen_cls=BgQwen(), template=sd.BACKGROUND_LEAN)
    p = sp.load_plan(job["plan"])
    assert full["template"] == sd.BACKGROUND and rep["template"] == sd.BACKGROUND_LEAN
    assert rep["template_info"] == "" and not rep["warnings"]
    lean_prompt = p["chunks"][0]["prompt"]
    s = sd.parse_sections(lean_prompt)
    assert len(lean_prompt) < len(full_prompt)
    # the full form's opening, which says the framing is kept (2026-10-13)
    assert "The target video keeps the camera, framing and timing of <Video 1>" in lean_prompt
    assert "<Subject 1> (appears throughout): partially_preserved" in s["retention_analysis"]
    assert "<Subject 2> (appears throughout, whenever he is in frame): fully_preserved" in s["retention_analysis"]
    dd = s["detailed_description"]
    assert "[Shot 1] The shot establishes <Subject 1>, the ship-themed diner environment with its" in dd
    assert "behind <Subject 2>. <Subject 2> walks in from the left" in dd


def test_a_lean_background_draft_with_a_second_picture(job):
    _bg(job)
    rep = _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=BgQwen2(), template=sd.BACKGROUND_LEAN_2REF)
    p = sp.load_plan(job["plan"])
    assert rep["template"] == sd.BACKGROUND_LEAN_2REF and "\n<Picture 2> is a closer view" in p["subject"]
    s = sd.parse_sections(p["chunks"][0]["prompt"])
    assert "environment from <Picture 1> and <Picture 2>," in s["summary"]
    assert "<Subject 1> (appears throughout): partially_preserved" in s["retention_analysis"]
    # the job's subject keeps the line as drafted; the prompt cites the picture inside the place's line
    sub = s["subject_definitions"]
    assert P2_PLACE + DETAIL + "." in sub and _no_standalone_picture_2(sub) and "second view" not in sub
    assert "Nothing of the original backdrop remains" in s["summary"] and not rep["warnings"]


def test_the_lean_background_templates_follow_the_same_picture_rules(job):
    _bg(job)
    rep = _draft(job, sheet=_sheet(), qwen_cls=BgQwen2(), template=sd.BACKGROUND_LEAN_2REF)
    assert rep["template"] == sd.BACKGROUND_LEAN and any("needs a second picture" in w for w in rep["warnings"])
    q = BgQwen2()
    sp.update_plan(job["plan"], lambda p: [p.update(subject="", subject_draft="")] +
                   [c.update(prompt="", prompt_state="empty") for c in p["chunks"]])
    rep = _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=q, template=sd.BACKGROUND_LEAN)
    assert rep["template"] == sd.BACKGROUND_LEAN
    assert any("doesn't use a second picture" in w and sd.BACKGROUND_LEAN_2REF in w for w in rep["warnings"])
    assert not any("second reference of the same place" in c["prompt"] for c in q.calls)


def test_a_character_widget_on_a_background_job_still_gets_the_full_form(job):
    _bg(job)
    rep = _draft(job, sheet=_sheet(), qwen_cls=BgQwen(), template=sd.LEAN_TEMPLATE)
    assert rep["template"] == sd.BACKGROUND


# ---------------------------------------------------------------- a still camera is said in the shot; pronouns agree

def test_camera_is_static_reads_qwens_video_sentence():
    assert sd.camera_is_static("The camera holds steady at eye level, framing her seated centrally on the couch throughout; "
                               "she remains still.")
    assert sd.camera_is_static("A static camera shows the man at a desk.")
    assert sd.camera_is_static("a person sitting behind an open box, filmed from a fixed high angle with the camera locked off")
    assert not sd.camera_is_static("A handheld camera follows the man as he crosses the frame.")
    assert not sd.camera_is_static("The camera holds steady, then pans left to follow her.")
    assert not sd.camera_is_static("A man walks across a room.")            # nothing said about the camera
    assert not sd.camera_is_static("")


def test_the_lean_background_first_shot_says_a_still_camera_and_names_picture_2():
    still = dict(BKW, video_1="The camera holds steady at eye level, framing him throughout.")
    moving = dict(BKW, video_1="A handheld camera follows him across the room.")
    two = PLACE + "\n<Picture 2> is a closer view of the same environment: a sofa."
    s1 = sd.parse_sections(sd.fill_bg_ref2va_lean(subject=PLACE, **still))["detailed_description"]
    # said positively: a negated "no pan, zoom or tilt" put those words in the prompt and the zoom came back
    # in H3's own camera words (base guide §4.3 "Static Shot")
    assert ("behind <Subject 2>. The camera holds a static shot throughout. "
            "<Subject 2> walks in.") in s1 and "<Picture 2>" not in s1
    assert "zoom" not in s1 and "pan" not in s1.replace("Japan", "")
    s2 = sd.parse_sections(sd.fill_bg_ref2va_lean(subject=PLACE, **moving))["detailed_description"]
    assert "static shot" not in s2
    s3 = sd.parse_sections(sd.fill_bg_ref2va_lean(subject=two, **still))["detailed_description"]
    # <Picture 2> stays out of the shots (a picture in a shot is a frame anchor): with it there, the same seed zoomed
    # and panned fast; without, it held (2026-10-13)
    assert ("red stools, behind <Subject 2>. The camera holds a static shot") in s3 and "<Picture 2>" not in s3


def test_a_missing_pronoun_line_is_read_off_who_the_person_is(job):
    class NoPronoun(BgQwen):
        def ask(self, prompt, image=None, video=None, max_tokens=2048, frame_count=16):
            out = super().ask(prompt, image=image, video=video, max_tokens=max_tokens, frame_count=frame_count)
            return out.replace("pronoun: he", "") if "The clip has" in prompt else out
    _bg(job)
    _draft(job, sheet=_sheet(), qwen_cls=NoPronoun(), template=sd.BACKGROUND)
    s = sd.parse_sections(sp.load_plan(job["plan"])["chunks"][0]["prompt"])
    assert "who keeps his own appearance" in s["subject_definitions"]
    assert "whenever he is in frame" in s["retention_analysis"] and "their" not in s["retention_analysis"]


def test_drop_shot_size_takes_the_label_and_keeps_the_framing():
    assert sd.drop_shot_size("The camera remains stationary at eye level throughout, keeping her centered in medium "
                             "close-up as she gestures animatedly while seated.") == (
        "The camera remains stationary at eye level throughout, keeping her centered as she gestures animatedly while seated.")
    assert sd.drop_shot_size("framing her seated centrally on the couch throughout") == (
        "framing her seated centrally on the couch throughout")             # framing stays
    assert sd.drop_shot_size("filmed in a tight close-up, then a wide shot") == "filmed, then a wide shot"
    assert sd.drop_shot_size("She leans in to look at the close-up of the card.") == (
        "She leans in to look at the close-up of the card.")                # not a label on the shot
    assert sd.drop_shot_size("") == "" and sd.drop_shot_size(None) == ""


def test_both_background_forms_leave_the_shot_size_out_of_the_video_line():
    v = "The camera holds steady at eye level, keeping her centered in medium close-up as she gestures."
    for fill in (sd.fill_bg_ref2va, sd.fill_bg_ref2va_lean):
        sec = sd.parse_sections(fill(subject=PLACE, **dict(BKW, video_1=v)))
        line = [x for x in sec["subject_definitions"].splitlines() if x.startswith("<Video 1>")][0]
        assert "close-up" not in line and "keeping her centered as she gestures." in line


# ---------------------------------------------------------------- a prompt names <Picture 2> only when a second picture is in use

P2_STORED = "\n<Picture 2> is a closer view of the same environment: a sofa."
PLACE_1 = "<Subject 1> is the old diner environment in <Picture 1>, featuring red stools."


def test_drop_picture_2_takes_only_the_stored_line():
    assert sd.drop_picture_2(PLACE_1 + P2_STORED + "\n<Subject 2> is x.") == PLACE_1 + "\n<Subject 2> is x."
    assert sd.drop_picture_2(PLACE_1) == PLACE_1 and sd.drop_picture_2("") == ""


def test_no_second_picture_wired_means_no_picture_2_in_the_prompts_but_the_subject_keeps_its_line(job):
    for template in (sd.BACKGROUND, sd.BACKGROUND_LEAN, sd.LEAN_TEMPLATE):
        sp.update_plan(job["plan"], lambda p: [p.update(subject=PLACE_1 + P2_STORED, mode="background")] +
                       [c.update(prompt="", prompt_state="empty") for c in p["chunks"]])
        rep = _draft(job, sheet=_sheet(), qwen_cls=BgQwen2(), template=template)      # no sheet_2
        p = sp.load_plan(job["plan"])
        assert p["subject"] == PLACE_1 + P2_STORED                      # the job's own subject is as it was
        text = p["chunks"][0]["prompt"]
        assert "<Picture 2>" not in text, template
        assert any("no second picture is in use" in w and "leave it out" in w for w in rep["warnings"])
        assert rep["template"] in (sd.BACKGROUND, sd.BACKGROUND_LEAN)


def test_a_picture_wired_to_a_template_that_ignores_it_is_left_out_of_the_prompts_too(job):
    sp.update_plan(job["plan"], lambda p: p.update(subject=PLACE_1 + P2_STORED, mode="background"))
    rep = _draft(job, sheet=_sheet(), sheet_2=_sheet(), qwen_cls=BgQwen2(), template=sd.BACKGROUND)
    assert "<Picture 2>" not in sp.load_plan(job["plan"])["chunks"][0]["prompt"]
    assert any("doesn't use a second picture" in w for w in rep["warnings"])
    assert any("no second picture is in use" in w for w in rep["warnings"])


def test_a_character_job_without_the_picture_leaves_it_out_of_the_prompts(job):
    sp.update_plan(job["plan"], lambda p: p.update(
        subject=SUBJECT + "\n<Picture 2> is a second reference of <Subject 1>: her face."))
    rep = _draft(job, sheet=_sheet(), qwen_cls=CharQwen2(), template=sd.LEAN_TEMPLATE)
    p = sp.load_plan(job["plan"])
    assert "<Picture 2>" in p["subject"] and "<Picture 2>" not in p["chunks"][0]["prompt"]
    assert any("no second picture is in use" in w for w in rep["warnings"])
