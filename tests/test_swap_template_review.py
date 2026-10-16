"""The 2026-10-13 template review against H3's prompt guides (references/ref-en.txt, base-en.txt): later shots open
on their cut time, a still camera is said inside each shot in the guide's words, a held place holds to the end of the
shot, background drafts get the kept person's placement from the mask, and Qwen's talk about its own pictures is
taken out."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_draft as sd  # noqa: E402

STILL = "The camera holds steady at eye level, framing her seated centrally throughout."
MOVING = "A handheld camera follows her across the room."


def test_cut_times_in_the_guide_form():
    blocks = ["[Shot 1] She sits.", "[Shot 2] Jump cut. She stands.", "[Shot 3] She leaves."]
    out = sd.tidy_shots(blocks, [0.0, 3.5, 65.25])
    assert out[0] == "[Shot 1] She sits."
    assert out[1] == "[Shot 2] At 00:03.500, the shot cuts to the next part of <Video 1>. She stands."
    assert out[2] == "[Shot 3] At 01:05.250, the shot cuts to the next part of <Video 1>. She leaves."
    assert sd.tidy_shots(out, [0.0, 3.5, 65.25]) == out                 # run twice: one cut head each
    assert sd.tidy_shots(blocks)[1] == "[Shot 2] Jump cut. She stands."  # no times: as before


def test_a_still_camera_is_said_in_each_shot():
    one = sd.camera_in_shots(["[Shot 1] She sits."], STILL)
    assert one == ["[Shot 1] The camera holds a static shot throughout. She sits."]
    two = sd.camera_in_shots(sd.tidy_shots(["[Shot 1] She sits.", "[Shot 2] She stands."], [0, 2]), STILL,
                             lead="The shot establishes the room.")
    assert two[0] == "[Shot 1] The shot establishes the room. The camera holds a static shot. She sits."
    assert two[1] == ("[Shot 2] At 00:02.000, the shot cuts to the next part of <Video 1>. The camera holds a static "
                      "shot. She stands.")
    assert sd.camera_in_shots(["[Shot 1] She sits."], MOVING) == ["[Shot 1] She sits."]


def test_every_template_says_the_still_camera_and_no_negated_camera_words():
    kw = dict(video_1=STILL, shots=["[Shot 1] She sits."], sounds="hum", n_shots=1, dialogue=False, audio=True,
              pronoun="she")
    place = "<Subject 1> is the diner environment in <Picture 1>, featuring red stools, a white counter."
    char = ("<Subject 1> is the clown whose motion comes from <Video 1> and whose appearance comes from <Picture 1>: "
            "a young woman, pink hair.")
    for f, subj, extra in ((sd.fill_ref2va, char, {}), (sd.fill_ref2va_lean, char, {}),
                           (sd.fill_bg_ref2va, place, {"who": "the woman"}),
                           (sd.fill_bg_ref2va_lean, place, {"who": "the woman"})):
        dd = sd.parse_sections(f(subject=subj, **kw, **extra))["detailed_description"]
        assert "The camera holds a static shot throughout." in dd, f.__name__
        for w in ("zoom", " pan", "tilt", "never", "morph", "dissolve"):
            assert w not in dd, (f.__name__, w)


def test_the_hold_sentence_only_for_one_held_place():
    held = [(0, "At the start she is in the centre of the frame, her whole figure inside the frame.")]
    assert sd.hold_sentence(held, "She") == "She stays at this place and size in the frame to the end of the shot."
    assert sd.hold_sentence(held + [(30, "At about 2.0 seconds she has gone out of the frame.")], "She") == ""
    assert sd.hold_sentence([(0, "At the start she is in the centre of the frame, moving slowly towards the left.")],
                            "She") == ""
    assert sd.hold_sentence([], "She") == ""


def _mask(n, x0, x1, y0=0.2, y1=1.0, h=72, w=128):
    m = np.zeros((n, h, w), np.float32)
    m[:, int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 1.0
    return m


def test_background_shots_get_the_kept_persons_place():
    m = _mask(48, 0.4, 0.6)
    out = sd.bg_place_shots(["[Shot 1] <Subject 2> raises her hand."], m, [(0, 47)], 24, 24, "she")
    # no framing words for a kept person: a seated whole figure with her feet on the bottom edge was called "framed
    # close" and H3 pushed in to her face (the 60 s test job's c3-t015, 2026-10-14)
    assert out[0].startswith("[Shot 1] At the start <Subject 2> is in the centre of the frame. ")
    for w in ("close", "bottom edge", "whole figure", "filling"):
        assert w not in out[0], w
    m2 = _mask(48, 0.4, 0.6, y0=0.05)                          # head near the top, feet on the bottom edge
    assert "close" not in sd.bg_place_shots(["[Shot 1] x"], m2, [(0, 47)], 24, 24, "she")[0]
    assert out[0].endswith("<Subject 2> raises her hand. <Subject 2> stays at this place and size in the frame to the "
                           "end of the shot.")


def test_qwen_talk_about_its_pictures_is_taken_out():
    p = sd.parse_qwen("video_1: a room.\nshots:\n[Shot 1] She stays fully visible across all images without leaving. "
                      "She sits in the frame.\nsounds: hum", 1)
    assert p["shots"][0] == "[Shot 1] She stays fully visible without leaving. She sits in the frame."


def test_saved_boxes_with_the_old_questions_count_as_unedited():
    # a workflow saved before the review carries the old built-in texts: they're not edits
    for const in ("DRAFT_INSTRUCTION", "SHOT_INSTRUCTION", "BG_DRAFT_INSTRUCTION"):
        old = sd._LEGACY_QUESTIONS[const][0]
        assert old != getattr(sd, const)
        name = next(n for n, c, _ in sd.QUESTIONS if c == const)
        over, warn = sd.question_overrides(f"### {name}\n{old}\n")
        assert over == {} and warn == [], const
    assert "never moves to the centre" in sd._LEGACY_QUESTIONS["DRAFT_INSTRUCTION"][0]
    assert "never moves to the centre" not in sd.DRAFT_INSTRUCTION + sd.SHOT_INSTRUCTION


def test_qwens_centre_words_go_where_the_mask_puts_her_elsewhere():
    m = _mask(48, 0.6, 0.8)                                 # the right third (the 60 s test job's c8: x 0.70)
    out = sd.bg_place_shots(["[Shot 1] <Subject 2> sits centered in the frame, facing forward."], m, [(0, 47)], 24, 24,
                            "she")
    assert "in the right third of the frame" in out[0] and "centered" not in out[0]
    assert "<Subject 2> sits, facing forward." in out[0]
    centre = sd.bg_place_shots(["[Shot 1] <Subject 2> sits centered in the frame."], _mask(48, 0.4, 0.6), [(0, 47)],
                               24, 24, "she")
    assert "sits centered in the frame" in centre[0]          # the mask agrees: kept
    assert sd.drop_centre("capturing her seated centrally in the frame from the waist up") == \
        "capturing her seated from the waist up"


def test_no_template_cites_picture_2_inside_a_shot():
    kw = dict(video_1=STILL, shots=["[Shot 1] She sits."], sounds="hum", n_shots=1, dialogue=False, audio=True,
              pronoun="she")
    place = ("<Subject 1> is the diner environment in <Picture 1>, featuring red stools, a white counter.\n"
             "<Picture 2> is a second view of the same environment: a sofa.")
    for f in (sd.fill_bg_ref2va, sd.fill_bg_ref2va_lean):
        t = f(subject=place, who="the woman", **kw)
        s = sd.parse_sections(t)
        assert "<Picture 2>" in s["subject_definitions"] and "<Picture 2>" not in s["detailed_description"], f.__name__


def test_the_second_place_picture_is_described_as_the_view_it_is():
    # the test job's second picture was the same room from the same view; the old question asked for a close-up, Qwen wrote one,
    # and every prompt called it a close-up (2026-10-13)
    assert sd.parse_view("view: same\ndetail: the room") == "same"
    assert sd.parse_view("**view:** closer\ndetail: x") == "closer" and sd.parse_view("view: angle") == "angle"
    assert sd.parse_view("detail: only") == "same"
    place = "<Subject 1> is the lounge environment in <Picture 1>, featuring a blue wall, a brown sofa."
    same = place + "\n" + sd.picture_2_sentence("the blue wall, the brown sofa and a lamp", True, "same")
    assert "<Picture 2> is a second view of the same environment: the blue wall" in same
    folded = sd.fold_picture_2(same, True)
    assert "<Picture 2> shows the same environment, adding its detail: the blue wall" in folded
    assert "surface texture" not in folded and "close" not in folded
    assert "adds detail of the same place" in sd.picture_2_retention(same)
    angle = place + "\n" + sd.picture_2_sentence("a doorway and a bookcase", True, "angle")
    assert "from another angle, adding its detail" in sd.fold_picture_2(angle, True)
    closer = place + "\n" + sd.picture_2_sentence("the sofa's weave", True, "closer")
    assert "provides only its surface texture" in sd.fold_picture_2(closer, True)
    assert "adds only surface detail" in sd.picture_2_retention(closer)
    for f in (sd.fill_bg_ref2va, sd.fill_bg_ref2va_lean):
        t = f(subject=same, who="the woman", video_1=STILL, shots=["[Shot 1] She sits."], sounds="hum", n_shots=1,
              dialogue=False, audio=True, pronoun="she")
        assert "adds detail of the same place" in t and "surface" not in t, f.__name__
    assert "close up" not in sd.SCENE_SHEET_2_INSTRUCTION.split("view:")[0]
    over, warn = sd.question_overrides("### bg_place_2\n" + sd.OLD_SCENE_SHEET_2 + "\n")
    assert over == {} and warn == []


def test_a_wobble_across_a_zone_edge_is_not_a_move():
    # the 60 s test job's c3 (2026-10-14): seated on the centre / right-third edge, one picture just inside the centre zone
    m = _mask(120, 0.6, 0.8)                                  # x 0.70: the right third
    m[60:64] = 0
    m[60:64, :, int(0.55 * 128):int(0.75 * 128)] = 1.0        # x 0.65 for four frames: "just right of the centre"
    beats = sd.place_beats(m, list(range(0, 120, 12)), 24, "she")     # the picture at frame 60 lands in the wobble
    assert len(beats) == 1 and "right third" in beats[0][1]
    moved = _mask(120, 0.6, 0.8)
    moved[60:] = 0
    moved[60:, :, int(0.45 * 128):int(0.65 * 128)] = 1.0     # really moves to the centre and stays
    b2 = sd.place_beats(moved, list(range(0, 120, 12)), 24, "she")
    assert len(b2) == 2 and "in the centre of the frame" in b2[1][1]


def test_the_frozen_lean_forms_carry_no_framing_words():
    # 2026-10-14: words for a view, a size or a framing act as camera cues; the question asks for none, and any that
    # come back leave Qwen's camera line
    assert "how much of {who} shows (for example" not in sd.BG_DRAFT_INSTRUCTION
    assert sd.drop_shot_size("The camera remains stationary, capturing her seated in full view as she gestures.") == \
        "The camera remains stationary, capturing her seated as she gestures."
    assert sd.drop_shot_size("A fixed camera frames her from the waist up on the couch.") == \
        "A fixed camera frames her on the couch."
    p = sd.parse_qwen("video_1: a room.\nshots:\n[Shot 1] She sits, staying in frame during this entire segment.\n"
                      "sounds: hum", 1)
    assert p["shots"][0] == "[Shot 1] She sits, staying in frame."
    over, warn = sd.question_overrides("### bg_draft\n" + sd._LEGACY_QUESTIONS["BG_DRAFT_INSTRUCTION"][0] + "\n")
    assert over == {} and warn == []
