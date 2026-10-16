"""Placement beats from the source mask (swap_draft.place_beats): where the person is, how big, and when they
leave and come back, on the render's clock; and the summary / retention lock they add to fill_ref2va."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import conftest  # noqa: E402,F401  (ComfyUI stubs)
import swap_draft as sd  # noqa: E402

H, W = 90, 160


def body(x0, x1, y0=5, y1=H):
    m = np.zeros((H, W), np.float32)
    m[y0:y1, max(0, x0):min(W, x1)] = 1.0
    return m


def walk_past():
    """In at the right, across to the left, out past the left edge (a shoulder left), back in from the left."""
    frames = []
    for k in range(24):                       # 0-23: right third -> left third, width 30
        x0 = int(120 - k * 5)
        frames.append(body(x0, x0 + 30))
    for _ in range(24):                       # 24-47: a sliver at the left edge
        frames.append(body(0, 4, 30, H))
    for k in range(12):                       # 48-59: back in, left third
        frames.append(body(5 + k, 35 + k))
    return np.stack(frames)


def test_mask_place_reads_edges_and_framing():
    p = sd.mask_place(body(0, 30))
    assert p["left"] and not p["right"] and p["bottom"] == 1.0 and p["top"] < 0.1
    assert sd.mask_place(np.zeros((H, W))) is None


def test_beats_walk_past():
    m = walk_past()
    beats = sd.place_beats(m, list(range(0, 60, 6)), 24, "she")
    text = " ".join(s for _, s in beats)
    assert beats[0][1].startswith("At the start she is in the right third of the frame")
    # only what the mask measures (2026-10-14): no "framed close", no "cutting across her body"
    assert "filling most of the frame's height, her head near the top of the frame" in beats[0][1]
    assert "close" not in beats[0][1] and "moving quickly towards the left" in beats[0][1]
    assert "gone out past the left edge" in text
    assert "comes back into the frame from the left" in text
    assert "At about 1.0 seconds" in text or "At about 1.2 seconds" in text      # 24 fps clock


def test_beats_only_on_change():
    m = np.stack([body(70, 90)] * 30)
    beats = sd.place_beats(m, list(range(0, 30, 6)), 24, "he")
    assert len(beats) == 1 and "in the centre of the frame" in beats[0][1]
    assert "moving" not in beats[0][1]


def test_beats_whole_figure_and_out():
    m = np.stack([body(70, 90, 20, 70)] * 10 + [np.zeros((H, W), np.float32)] * 10)
    beats = sd.place_beats(m, [0, 6, 12, 18], 25, "they")
    assert "their whole figure inside the frame" in beats[0][1]
    assert beats[-1][1].startswith("At about 0.4 seconds they have gone out of the frame")   # frame 10, timed exactly


def test_merge_moments_puts_beats_before_their_moment():
    moments = [(0.0, {"person": "yes", "pose": "walking to the left", "look": "to the side", "hands": "swing at the sides"})]
    out = sd.merge_moments(moments, [], 24, "she", placements=[(0.0, "At the start she is in the right third of the frame.")])
    assert out.startswith("At the start she is in the right third of the frame. She is walking to the left")


def _fill(**kw):
    return sd.fill_ref2va(subject="<Subject 1> is the clown whose motion comes from <Video 1> and whose appearance "
                          "comes from <Picture 1>: a clown.", video_1="a man walks.", shots=["[Shot 1] She walks."],
                          sounds="steps.", n_shots=1, dialogue=False, audio=True, pronoun="she", **kw)


def test_fill_unchanged_without_placement():
    a = _fill()
    assert "exact place in every frame" not in a and "position across the frame, size and framing" not in a


def test_fill_locks_place_and_exits():
    a = _fill(placed=True, exits=True)
    assert "She takes the person's exact place in every frame: the same position across the frame, the same size " \
           "and the same framing, and the same moments of leaving the frame and coming back into it." in a
    assert "her position across the frame, size and framing, the moments of leaving and coming back into the " \
           "frame, her body position" in a
    b = _fill(placed=True)
    assert "the same size and the same framing." in b and "moments of leaving" not in b
    # the beats say where she is (centre included), so the old "never moves to the centre" goes
    # said positively, placed or not: a negated "never moves to the centre" names the centre (2026-10-13)
    for t in (b, _fill()):
        assert "never" not in t.split("detailed_description:")[1]
        assert "framed exactly the same way and in the same place in the frame, looking where the person looks, ""and her hands only hold what the person's hands hold." in t


def test_pace_and_already():
    m = walk_past()                                   # 5 px a frame of 160 at 24 fps: 0.75 widths a second
    first = sd.place_beats(m, list(range(0, 60, 6)), 24, "she")[0][1]
    assert "already moving quickly towards the left" in first
    slow = np.stack([body(100 - k // 4, 130 - k // 4) for k in range(48)])     # ~0.04 widths a second... 0.0375
    assert "moving" not in sd.place_beats(slow, list(range(0, 48, 6)), 24, "he")[0][1]
    mid = np.stack([body(120 - k, 150 - k) for k in range(48)])                # 1 px a frame: 0.15 widths a second
    assert ", already moving towards the left" in sd.place_beats(mid, list(range(0, 48, 6)), 24, "he")[0][1]


def test_place_away():
    ref = 0.2
    assert sd.place_away(None, ref)
    assert sd.place_away(sd.mask_place(body(0, 4, 30, H)), ref)               # a sliver at the left edge
    assert not sd.place_away(sd.mask_place(body(0, 40)), ref)                 # at the edge, but most of them in
    assert not sd.place_away(sd.mask_place(body(70, 100)), ref)


def test_away_frames_match_the_beats():
    # in, a sliver at the edge, a bigger sliver (0.6 of usual: still out, by the back-in rule), then clearly in
    ref = float((body(60, 90) > 0.5).mean())
    sliver = body(0, 4, 30, H)
    bigger = body(0, 18)                      # 18/30 = 0.6 of the usual size: over 0.5, under 0.75
    m = np.stack([body(60, 90)] * 6 + [sliver] * 6 + [bigger] * 6 + [body(10, 40)] * 6)
    frames = list(range(0, 24, 3))
    assert sd.away_frames(m, frames, ref) == {6, 9, 12, 15}
    beats = sd.place_beats(m, frames, 24, "she", ref)
    back = [f for f, s in beats if "comes back" in s]
    assert back == [18]


def test_touch_question_only_with_others():
    plain = sd.moment_instruction("a woman in a red dress")
    assert "touch:" not in plain and "exactly six short labelled lines" in plain
    q = sd.moment_instruction("a woman in a red dress", touch=True)
    assert "exactly seven short labelled lines" in q
    assert q.index("others:") < q.index("touch:") < q.index("frame:")


def test_touch_parsed_and_merged_once():
    raw = ("person: yes\npose: dancing\nlook: at the man\nhands: holds his hand\nothers: a man in a dark suit\n"
           "touch: the person's right hand holds the left hand of the man in the suit\nframe: a hall")
    m = sd.parse_moment(raw)
    assert m["touch"].startswith("the person's right hand")
    m2 = dict(m)
    m3 = dict(m, touch="none")
    out = sd.merge_moments([(0.0, m), (0.5, m2), (1.0, m3), (1.5, dict(m3))], [], 24, "she")   # two nones end it
    assert out.count("Her right hand holds the left hand of the man in the suit.") == 1      # his stays his
    assert out.endswith("She and the others no longer touch.")


def test_touch_sentence_keeps_the_partners_pronouns():
    pr = sd.PRONOUNS["she"]
    assert sd.touch_sentence("the hand of the man in the suit rests on the person's back", pr) == \
        "The hand of the man in the suit rests on her back."
    assert sd.touch_sentence("his left hand holds the person's right hand", pr) == "His left hand holds her right hand."
    assert sd.touch_sentence("none", pr) == "" and sd.touch_sentence("", pr) == ""


def test_touch_ends_only_after_two_nones():
    base = {"person": "yes", "pose": "pinning", "look": "down", "hands": "grip the leg"}
    hold = dict(base, touch="the person's arms wrap around the torso of the woman in blue")
    none = dict(base, touch="none")
    out = sd.merge_moments([(0.0, hold), (0.5, none), (1.0, hold)], [], 24, "she")
    assert "no longer touch" not in out and out.count("Her arms wrap around") == 1
    out = sd.merge_moments([(0.0, hold), (0.5, none), (1.0, none)], [], 24, "she")
    assert out.endswith("She and the others no longer touch.")


def test_scene_others_are_people_only():
    assert sd.NOT_PEOPLE.search("scoreboard visible upper-left")
    assert sd.NOT_PEOPLE.search("blurred audience members scattered across both sides of the frame")
    assert not sd.NOT_PEOPLE.search("opponent in blue singlet on the ground center")
    assert not sd.NOT_PEOPLE.search("a man holding a sign on the left")
    assert "Never name anyone" in sd.SCENE_INSTRUCTION and "Never name them" in sd.WHO_INSTRUCTION


def test_pass1_size_follows_the_source_shape():
    import swap_planner as spl
    assert spl.pass1_size(1920, 1080) == (736, 416)
    assert spl.pass1_size(720, 1280) == (416, 736)
    assert spl.pass1_size(1080, 1080) == (544, 544)
    assert spl.RETURN_NAMES[spl.OUT_RENDER_W] == "render_width" and spl.RETURN_NAMES[spl.OUT_RENDER_H] == "render_height"
    assert len(spl.RETURN_TYPES) == len(spl.RETURN_NAMES) == 21


def test_others_split_into_people_and_background():
    people, behind = sd.split_others("Opponent in blue singlet on the ground center; scoreboard visible upper-left; "
                                     "blurred audience members scattered across both sides; none")
    assert people == ["opponent in blue singlet on the ground center"]
    assert behind == ["scoreboard visible upper-left", "blurred audience members scattered across both sides"]
    v = sd.with_background("A wrestler pins her opponent on a white mat under bright lights", behind)
    assert v == ("A wrestler pins her opponent on a white mat under bright lights. Behind them, kept as they are: "
                 "scoreboard visible upper-left; blurred audience members scattered across both sides.")
    assert sd.split_others("none") == ([], [])
