"""The lean timeline template (swap_draft.LEAN_TEMPLATE): the same parts as the timeline, written short."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import conftest  # noqa: E402,F401  (ComfyUI stubs)
import swap_draft as sd  # noqa: E402

SUBJ = ("<Subject 1> is the pink-clown angel whose motion comes from <Video 1> and whose appearance comes from "
        "<Picture 1>: a young woman, red clown nose, pink wavy hair, magenta corset, long pink gloves, rainbow-striped "
        "knee-high socks, red high heels, large white feathered wings.")


def test_lean_is_the_default():
    assert sd.LEAN_TEMPLATE in sd.TEMPLATES and sd.DEFAULT_TEMPLATE == sd.LEAN_TEMPLATE


def test_lean_subject_drops_footwear_and_keeps_the_rest():
    s = sd.lean_subject(SUBJ)
    assert s == ("<Subject 1> is the pink-clown angel in <Picture 1>: a young woman, red clown nose, pink wavy hair, "
                 "magenta corset, long pink gloves, large white feathered wings.")
    assert sd.lean_subject("<Subject 1> is someone else entirely.") == "<Subject 1> is someone else entirely."


def test_lean_merge_keeps_handling_and_drops_repeats():
    m = lambda hands, look="down": {"person": "yes", "pose": "kneeling", "look": look, "hands": hands}   # noqa: E731
    moments = [(0.0, m("grip the leg")), (0.5, m("grip the ankle tightly")), (1.0, m("grip the ankle", "to the side")),
               (1.5, m("holds a small red cup up to the mouth"))]
    lean = sd.merge_moments(moments, [], 24, "she", lean=True)
    full = sd.merge_moments(moments, [], 24, "she")
    assert "ankle" not in lean and "looks to the side" not in lean and "holds a small red cup" in lean
    assert "ankle" in full


def test_lean_sections_parse_and_name_the_background():
    t = sd.fill_ref2va_lean(subject=SUBJ, video_1="A wrestler pins her opponent.", shots=["[Shot 1] She pins her."],
                            sounds="room tone.", n_shots=1, dialogue=True, audio=True, pronoun="she",
                            who="woman in a pink singlet", people=[("<Subject 2>", "a wrestler in blue", [1])],
                            placed=True, behind=["a crowd in the stands", "a scoreboard"])
    sd.parse_sections(t)
    assert "<Subject 1> (appears throughout, wherever the woman in a pink singlet is in frame): attribute_transfer - "
    assert "attribute_transfer - transfer her face, hair, costume and accessories from <Picture 1> onto " \
           "the woman in a pink singlet's position, size, framing and motion, with the mouth moving with the speech." in t
    assert "partially_preserved - preserve the background (a crowd in the stands; a scoreboard)" in t
    assert "socks" not in t and "never poses" not in t and "lips move with every word" in t


def test_lean_drops_plain_gestures():
    m = lambda hands: {"person": "yes", "pose": "walking", "look": "to the side", "hands": hands}   # noqa: E731
    out = sd.merge_moments([(0.0, m("swing loosely")), (0.5, m("gestures with left hand near body")),
                            (1.0, m("hands the cup over to the man"))], [], 24, "she", lean=True)
    assert "gestures" not in out and "hands the cup over" in out


def test_others_from_moments_finds_the_opponent_the_scene_missed():
    xs = ["An opponent in a blue singlet lying pinned on the mat", "an opponent in a blue singlet, pinned",
          "none", "blurred audience in the stands", "an opponent in a blue singlet lying face down"]
    assert sd.others_from_moments(xs) == "an opponent in a blue singlet"
    # wrestling c4's real answers: the person and the crowd in one answer
    real = ["none"] * 9 + ["opponent lying pinned, spectators watching"] * 4 + [
        "opponent pinned beneath; spectators watching", "opponent pinned; spectators watching"]
    assert sd.others_from_moments(real) == "the opponent"
    assert sd.others_from_moments(["none", "a crowd behind"]) is None
    assert sd.others_from_moments(["a referee standing on the left"]) is None        # one moment isn't enough


def test_relation_sentence():
    pr = sd.PRONOUNS["she"]
    s = sd.relation_sentence("The person lies across the other person, pinning the other person's shoulders to the "
                             "mat.", pr)
    assert s == "She lies across the other person, pinning the other person's shoulders to the mat."
    assert sd.relation_sentence("none", pr) == ""
    assert "relation" in [n for n, _, _ in sd.QUESTIONS] and "{other}" in sd.RELATION_INSTRUCTION


def test_lean_subject_drops_small_details_and_with_tails():
    s = sd.lean_subject("<Subject 1> is the pink-clown angel whose motion comes from <Video 1> and whose appearance "
                        "comes from <Picture 1>: a young woman, red clown nose, bright red lips, blue eyes, pink wavy "
                        "hair with bangs, white halo with yellow trim, rainbow choker, magenta corset with white ruffled "
                        "collar, white bow at waistline, long pink gloves with rainbow bands, red high heels, large "
                        "white feathered wings.")
    assert s == ("<Subject 1> is the pink-clown angel in <Picture 1>: a young woman, red clown nose, pink wavy hair, "
                 "white halo, magenta corset, long pink gloves, large white feathered wings.")


def test_scene_asks_for_the_background_and_the_missed_person_is_described():
    assert "Write exactly four labelled lines" in sd.SCENE_INSTRUCTION and "\nbackground: " in sd.SCENE_INSTRUCTION
    for old in sd._LEGACY_QUESTIONS["SCENE_INSTRUCTION"]:
        assert "background: what is behind" not in old
        assert sd.question_overrides(sd.questions_text().replace(sd.SCENE_INSTRUCTION, old)) == ({}, [])
    assert "other_look" in [n for n, _, _ in sd.QUESTIONS]
    assert sd.OTHER_LOOK_INSTRUCTION.format(n=3, who="a woman in pink", other="the opponent")


def test_the_scene_answers_background_line_is_read():
    raw = ("video_1: A wrestler pins another on a white mat.\nsounds: Room tone.\nbackground: a digital scoreboard; "
           "tiered seating; a caption\nothers: none")
    d = sd._labelled(raw)
    assert d["background"] == "a digital scoreboard; tiered seating; a caption" and d["sounds"] == "Room tone."
