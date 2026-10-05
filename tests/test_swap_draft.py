"""SeamStitch Swap Draft Prompts (design §4.6 / §5.5, B4).

The models are stand-ins here (the real Qwen / Omni / Whisper run is B4's GPU round): what these
tests hold is everything around them. A draft never overwrites a prompt, not even one written while
the run was busy; the template's output always parses into the six Ref2VA sections; a missing Omni
or Whisper degrades with a warning instead of failing; Omni's words land on Whisper's timings;
kept chunks are never drafted; the Planner routes a redraft's chunk to the node."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
import swap_draft as sd  # noqa: E402
from test_swap import job  # noqa: E402,F401
from test_swap_nodes import _plan978  # noqa: E402

SUBJECT = ("<Subject 1> (S1) is the clown angel girl whose motion comes from <Video 1> and whose appearance comes "
           "from <Picture 1>: a young woman with pink hair.")


class FakeQwen:
    """Answers the three instructions the node sends, and records them."""

    def __init__(self, on_draft=None, shots=None):
        self.calls = []
        self.loaded = 0
        self.closed = 0
        self.on_draft = on_draft
        self.shots = shots

    def load(self):
        self.loaded += 1

    def ask(self, prompt, image=None, video=None, max_tokens=2048, frame_count=16):
        self.calls.append({"prompt": prompt, "image": image is not None,
                           "frames": None if video is None else int(video.shape[0])})
        if "character sheet" in prompt:
            return "name: clown angel girl\nappearance: a young woman with pink hair, a halo and wings.\npronoun: she"
        if prompt.startswith("These are"):
            return "person: yes\nobjects: open cardboard box, white card"
        if self.on_draft:
            self.on_draft(len([c for c in self.calls if c["frames"] and "The clip has" in c["prompt"]]))
        k = int(prompt.split("The clip has ")[1].split(" shot")[0])
        says = ' She says SAYS "Hello there." to the camera.' if 'spoken: "' in prompt else ""
        n = self.shots or k
        blocks = "\n".join(f"[Shot {i + 1}] {'Jump cut. ' if i else ''}She reaches into the open cardboard box and "
                           f"lifts a white card.{says if i == n - 1 else ''}" for i in range(n))
        return (f"**video_1:** a person sitting behind an open box, filmed from a fixed high angle\n\nshots:\n{blocks}\n\n"
                f"sounds: room hum, rustling, her speech")

    def close(self):
        self.closed += 1


class FakeOmni:
    def __init__(self, text='transcript: [English] Hello there. audio_events: low room hum, a soft thud'):
        self.text = text
        self.n = 0

    def transcribe(self, wav, sr):
        self.n += 1
        return self.text

    def close(self):
        pass


class FakeWhisper:
    def __init__(self):
        self.loaded = self.closed = 0

    def load(self):
        self.loaded += 1

    def words(self, audio16k, language=None):
        return [{"word": "Hello", "start": 0.2, "end": 0.5, "p": 0.9},
                {"word": "there.", "start": 0.55, "end": 0.9, "p": 0.9}], "en"

    def close(self):
        self.closed += 1


def _sheet():
    import torch
    return torch.zeros((1, 32, 32, 3))


def _draft(job, **kw):
    kw.setdefault("qwen_cls", FakeQwen())
    kw.setdefault("omni_cls", FakeOmni())
    kw.setdefault("whisper", FakeWhisper())
    kw.setdefault("sample_vram", False)
    kw.setdefault("template", sd.TEMPLATES[0])      # the fakes answer the one-call form; per shot has its own tests
    return sd.run_draft(job["plan"], **kw)


# ---------------------------------------------------------------- the never-overwrite rule

def test_a_draft_never_overwrites_an_edited_prompt(job):
    sp.update_plan(job["plan"], lambda p: p["chunks"][0].update(prompt="the user's own prompt", prompt_state="edited"))
    rep = _draft(job, mode="all (into the draft field)", sheet=_sheet())
    p = sp.load_plan(job["plan"])
    c0, c1, c2 = p["chunks"]
    assert c0["prompt"] == "the user's own prompt" and c0["prompt_state"] == "edited"
    assert c0["draft"].startswith("subject_definitions:") and rep["chunks"][c0["id"]]["into"] == "draft"
    for c in (c1, c2):
        assert c["prompt"].startswith("subject_definitions:") and c["prompt_state"] == "draft" and not c.get("draft")
        assert rep["chunks"][c["id"]]["into"] == "prompt"
    # every stage's files, under drafts/<chunk>/
    for c in p["chunks"]:
        d = os.path.join(job["dir"], "drafts", c["id"])
        assert {"omni.txt", "whisper.json", "qwen.txt", "draft.txt", "check.json"} <= set(os.listdir(d))


def test_a_prompt_written_during_the_run_is_never_overwritten(job):
    """The emptiness check happens under the plan lock at write time, not at the run's start."""
    target = sp.load_plan(job["plan"])["chunks"][1]["id"]

    def kay_types(_n):
        sp.update_plan(job["plan"], lambda p: sp.find_chunk(p, target).update(prompt="typed meanwhile",
                                                                              prompt_state="edited"))
    _draft(job, qwen_cls=FakeQwen(on_draft=kay_types))
    c = sp.find_chunk(sp.load_plan(job["plan"]), target)
    assert c["prompt"] == "typed meanwhile" and c["draft"].startswith("subject_definitions:")


def test_empty_only_skips_chunks_with_a_prompt_and_the_subject_is_kept(job):
    def upd(p):
        p["chunks"][0]["prompt"] = "kept as is"
        p["subject"] = SUBJECT
    sp.update_plan(job["plan"], upd)
    rep = _draft(job, sheet=_sheet())
    p = sp.load_plan(job["plan"])
    assert p["chunks"][0]["id"] not in rep["chunks"] and p["chunks"][0].get("draft") is None
    assert p["subject"] == SUBJECT and not p.get("subject_draft") and "subject" not in rep
    assert SUBJECT in p["chunks"][1]["prompt"]


def test_the_subject_is_drafted_into_an_empty_subject_else_its_draft_field(job):
    rep = _draft(job, sheet=_sheet())
    p = sp.load_plan(job["plan"])
    assert rep["subject"]["into"] == "subject" and p["subject_pronoun"] == "she"
    assert p["subject"].startswith("<Subject 1> (S1) is the clown angel girl whose motion comes from <Video 1>")
    assert p["subject"] in p["chunks"][1]["prompt"]
    sp.update_plan(job["plan"], lambda q: q.update(subject=SUBJECT))
    rep = _draft(job, sheet=_sheet(), mode="all (into the draft field)")
    p = sp.load_plan(job["plan"])
    assert rep["subject"]["into"] == "subject_draft" and p["subject"] == SUBJECT and p["subject_draft"]


def test_the_subject_edit_reaches_every_prompt_and_its_draft_can_be_adopted():
    p = _plan978([(406, "anchored")])
    p["subject"] = "OLD SUBJECT"
    p["chunks"][0]["prompt"] = "subject_definitions:\nOLD SUBJECT\n..."
    p["chunks"][1]["draft"] = "subject_definitions:\nOLD SUBJECT\n..."
    r = spl.apply_op(p, {"op": "set_subject", "subject": "NEW SUBJECT"})
    assert r["subject_replaced_in"] == 2 and "NEW SUBJECT" in p["chunks"][0]["prompt"] + p["chunks"][1]["draft"]
    p["subject_draft"] = "DRAFTED"
    spl.apply_op(p, {"op": "adopt_subject_draft"})
    assert p["subject"] == "DRAFTED" and "DRAFTED" in p["chunks"][0]["prompt"] and p["subject_draft"] is None
    with pytest.raises(sp.PlanError):
        spl.apply_op(p, {"op": "adopt_subject_draft"})


# ---------------------------------------------------------------- the template

def test_the_template_parses_into_the_six_sections():
    parts = sd.parse_qwen("video_1: a person at a desk\nshots:\n[Shot 1] She looks down.\n[Shot 2] Jump cut. She "
                          'lifts a card, and SAYS "So what have we got here?"\nsounds: room hum', 2)
    text = sd.fill_ref2va(subject=SUBJECT, video_1=parts["video_1"], shots=parts["shots"], sounds=parts["sounds"],
                          n_shots=2, dialogue=True, audio=True, pronoun="she")
    s = sd.parse_sections(text)
    assert list(s) == list(sd.SECTIONS)
    assert s["subject_definitions"].startswith(SUBJECT) and "<Video 1> is the source video" in s["subject_definitions"]
    assert s["summary"].startswith("[video editing + reference generation + audio reuse]") and "the jump cut" in s["summary"]
    assert "<Audio 1>: fully_copy" in s["retention_analysis"]
    dd = s["detailed_description"]
    assert "[Shot 1] She looks down." in dd and "[Shot 2] Jump cut." in dd
    assert "<Subject 1> (S1) says <d>[English]So what have we got here?</d>" in dd
    assert "Her lips move with every word" in dd and "Every jump cut happens at exactly the same moment" in dd
    assert s["overall_soundscape"] == "The original sound of <Video 1>: room hum" and s["non_diegetic_music"] == "N/A"
    # one shot, no dialogue, no audio: no cut line, no lip line, no <Audio 1>
    t2 = sd.fill_ref2va(subject=SUBJECT, video_1="x.", shots=["[Shot 1] He sits."], sounds="", n_shots=1,
                        dialogue=False, audio=False, pronoun="they")
    s2 = sd.parse_sections(t2)
    assert "jump cut" not in t2.lower() and "<Audio 1>" not in t2 and "lips" not in t2
    assert s2["summary"].startswith("[video editing + reference generation]") and "They do exactly" in t2
    with pytest.raises(sd.DraftError):
        sd.parse_sections("summary:\nx\n")


def test_parse_qwen_tolerates_markdown_and_flags_a_wrong_shot_count():
    r = sd.parse_qwen('**video_1:** a room\n**shots:**\n[Shot 1] She waves. she says "Hi."\n**sounds:** hum', 3)
    assert r["video_1"] == "a room." and r["sounds"] == "hum"
    assert r["shots"] == ["[Shot 1] She waves. <Subject 1> (S1) says <d>[English]Hi.</d>"]
    assert any("1 shot block(s) for 3 shot(s)" in w for w in r["warnings"])
    assert sd.parse_qwen("nothing useful", 1)["warnings"]


def test_the_shots_follow_the_confirmed_cuts_inside_the_render_range():
    p = _plan978()
    assert sd.chunk_shots(p, 400, 608) == [(400, 412), (413, 428), (429, 608)]
    assert sd.chunk_shots(p, 413, 428) == [(413, 428)]          # a cut on r0 starts the range: no extra shot
    assert sd.sample_indices(209, 16)[0] == 0 and sd.sample_indices(209, 16)[-1] == 208
    assert len(sd.sample_indices(209, 16)) == 16 and sd.sample_indices(5, 16) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- dialogue

def test_omni_words_land_on_whisper_timings():
    po = sd.parse_omni("transcript: [English] So what have we got here? Link app, product manual, lovely bundled, "
                       "and a product... [voice cuts off] audio_events: room hum, a soft thud")
    assert po["language"] == "English" and po["notes"] == ["voice cuts off"] and po["events"] == "room hum, a soft thud"
    ww = [("So", 1.16, 1.3), ("what", 1.3, 1.4), ("have", 1.4, 1.5), ("we", 1.5, 1.55), ("got", 1.55, 1.7),
          ("here?", 1.7, 1.84), ("Link", 2.0, 2.2), ("app.", 2.2, 2.44), ("Product", 2.68, 3.0), ("manual.", 3.0, 3.6),
          ("Lovely", 6.5, 6.8), ("bundled", 6.8, 7.2), ("Amaran,", 7.3, 8.0)]
    words = [{"word": w, "start": s, "end": e} for w, s, e in ww]
    al = sd.align_words(po["words"], words)
    assert [a["word"] for a in al][:6] == ["So", "what", "have", "we", "got", "here?"]
    assert al[0]["start"] == 1.16 and al[5]["end"] == 1.84 and all(a["matched"] for a in al[:10])
    # "and a product..." where Whisper heard "Amaran,": spread over Whisper's span
    tail = al[-3:]
    assert [a["word"] for a in tail] == ["and", "a", "product..."]
    assert 7.3 <= tail[0]["start"] < tail[-1]["end"] <= 8.0
    lines = sd.dialogue_lines(al, 25, 209)
    assert lines[0] == {"text": "So what have we got here?", "start": 1.16, "end": 1.84, "frames": [29, 46]}
    assert lines[1]["text"].startswith("Link app,") and lines[-1]["text"].endswith("and a product...")
    assert all(0 <= ln["frames"][0] <= ln["frames"][1] <= 208 for ln in lines)


def test_no_speech_and_untimed_lines():
    assert sd.parse_omni("transcript: none audio_events: quiet hum")["speech"] is False
    assert [x["text"] for x in sd.lines_from_text("One. Two? Three")] == ["One.", "Two?", "Three"]
    assert sd.dialogue_check('<d>[English]Hello there.</d>', [{"text": "Hello there."}]) == 1.0


def test_dialogue_reaches_qwen_with_frames_and_comes_back_as_d_lines(job):
    q = FakeQwen()
    rep = _draft(job, qwen_cls=q)
    drafts = [c for c in q.calls if c["frames"] and "The clip has" in c["prompt"]]
    assert len(drafts) == 3 and all(c["frames"] == sd.DEFAULT_FRAMES for c in drafts) and q.loaded == 1 and q.closed == 1
    assert 'spoken: "Hello there." (0.2-0.9 s)' in drafts[0]["prompt"]
    assert "Objects seen in the pictures: open cardboard box, white card." in drafts[0]["prompt"]
    # the next chunk gets the previous chunk's object names, never its shot blocks
    assert "Names used for objects in the previous clip" in drafts[1]["prompt"] and "[Shot 1]" not in drafts[1]["prompt"]
    assert "low room hum, a soft thud" in drafts[0]["prompt"]          # Omni's events -> sounds
    p = sp.load_plan(job["plan"])
    assert "<Subject 1> (S1) says <d>[English]Hello there.</d>" in p["chunks"][0]["prompt"]
    w = json.load(open(os.path.join(job["dir"], "drafts", p["chunks"][0]["id"], "whisper.json"), encoding="utf-8"))
    assert w["lines"][0]["frames"] == [5, 22]
    assert rep["chunks"][p["chunks"][0]["id"]]["dialogue_coverage"] == 1.0


# ---------------------------------------------------------------- degrading

def test_a_missing_omni_and_whisper_degrade_with_a_warning(job):
    rep = _draft(job, omni_cls=False, whisper=False)
    assert any("Omni Captioner Transcribe isn't installed" in w for w in rep["warnings"])
    assert any("faster-whisper isn't available" in w for w in rep["warnings"])
    p = sp.load_plan(job["plan"])
    assert all(c["prompt"].startswith("subject_definitions:") for c in p["chunks"])
    assert "<d>" not in p["chunks"][0]["prompt"]


def test_a_missing_omni_takes_the_words_from_whisper(job):
    rep = _draft(job, omni_cls=False)
    p = sp.load_plan(job["plan"])
    assert "<Subject 1> (S1) says <d>[English]Hello there.</d>" in p["chunks"][0]["prompt"]
    assert any("Whisper alone" in w for w in rep["chunks"][p["chunks"][0]["id"]]["warnings"])


def test_a_missing_whisper_drafts_the_lines_without_timings(job):
    q = FakeQwen()
    rep = _draft(job, qwen_cls=q, whisper=False)
    d = [c for c in q.calls if c["frames"] and "The clip has" in c["prompt"]][0]
    assert "have no times: place each where it fits the action" in d["prompt"] and '"Hello there."' in d["prompt"]
    assert any("without timings" in w for w in rep["warnings"])


def test_a_failing_omni_call_degrades_per_chunk(job):
    class Broken(FakeOmni):
        def transcribe(self, wav, sr):
            raise RuntimeError("llama.cpp exited 1")
    rep = _draft(job, omni_cls=Broken())
    cid = sp.load_plan(job["plan"])["chunks"][0]["id"]
    assert any("Omni failed" in w for w in rep["chunks"][cid]["warnings"])


def test_qwen_missing_refuses_with_an_install_hint(job):
    with pytest.raises(sd.DraftError, match="ComfyUI-QwenVL"):
        _draft(job, qwen_cls=False)


# ---------------------------------------------------------------- routing and selection

def test_kept_chunks_are_never_drafted(job):
    sp.update_plan(job["plan"], lambda p: p["chunks"][0].update(keep=True))
    rep = _draft(job)
    p = sp.load_plan(job["plan"])
    assert p["chunks"][0]["id"] not in rep["chunks"] and not p["chunks"][0]["prompt"]
    assert not os.path.isdir(os.path.join(job["dir"], "drafts", p["chunks"][0]["id"]))
    rep = _draft(job, named=[p["chunks"][0]["id"]])
    assert not rep["chunks"] and any("never drafted" in w for w in rep["warnings"])


def test_a_redraft_names_its_chunk_and_selected_needs_one(job):
    p = sp.load_plan(job["plan"])
    sp.update_plan(job["plan"], lambda q: [c.update(prompt="x") for c in q["chunks"]])
    rep = _draft(job, named=[p["chunks"][1]["id"]])
    assert list(rep["chunks"]) == [p["chunks"][1]["id"]] and rep["chunks"][p["chunks"][1]["id"]]["into"] == "draft"
    with pytest.raises(sd.DraftError, match="selected"):
        sd.select_chunks(sp.load_plan(job["plan"]), "selected")
    assert sd.parse_draft_plan('{"plan": "a/plan.json", "chunks": ["c2"]}') == ("a/plan.json", ["c2"])
    assert sd.parse_draft_plan("a/plan.json") == ("a/plan.json", [])


def test_the_planner_routes_a_redraft_chunk_to_draft_plan(job, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(job["dir"].parent.parent), raising=False)
    plan = lambda r: spl.SeamStitchSwapPlanner().plan("t", "", 39, 6, 5, 10, 260, True, json.dumps(r))["result"]  # noqa: E731
    out = plan({"action": "draft"})
    assert out[spl.OUT_DRAFT] == job["plan"] and isinstance(out[0], spl.ExecutionBlocker)
    out = plan({"action": "draft", "chunk": "c2"})
    assert json.loads(out[spl.OUT_DRAFT]) == {"plan": job["plan"], "chunks": ["c2"]}


def test_the_prop_check_lists_unseen_and_carried_over_nouns():
    r = sd.prop_check(["[Shot 1] She lifts a white card and a small folding knife, then rests her hands on the box."],
                      ["open cardboard box", "white card"],
                      prev_shots=["[Shot 1] She cuts the tape with a small folding knife."])
    assert [h for h, _ in r["unseen"]] == ["knife"] and [h for h, _ in r["carried_over"]] == ["knife"]
    assert sd.prop_check(["[Shot 1] She looks at the camera with her head at the top edge of the frame."], [])["unseen"] == []


def test_hands_naming_an_unseen_prop_are_left_out_of_the_shot_block():
    moments = [(0.1, {"person": "yes", "pose": "crouching low near floor", "look": "to the side",
                      "hands": "holds a silver knife over cardboard box"}),
               (1.6, {"person": "yes", "pose": "sitting on floor", "look": "at the camera",
                      "hands": "holds a dark cigar-like object near mouth with left hand"}),
               (2.2, {"person": "yes", "pose": "sitting on floor", "look": "at the camera",
                      "hands": "holds a dark grey box lid near lap with both hands"}),
               (3.0, {"person": "yes", "pose": "sitting on floor", "look": "at the camera",
                      "hands": "holds a large brown cardboard box angled toward viewer"})]
    assert sd.drop_unseen_hands(moments, ["box", "knife"]) == ["dark cigar-like object", "dark grey box lid"]
    body = sd.merge_moments(moments, [], 25, "she")
    assert "cigar" not in body and "lid" not in body
    assert "silver knife" in body and "box angled toward viewer" in body


def test_the_node_widgets_keep_their_order():
    req = sd.SeamStitchSwapDraft.INPUT_TYPES()["required"]
    assert list(req) == ["draft_plan", "chunks", "qwen_model", "quantization", "frames_per_chunk", "transcribe",
                         "word_timings", "template", "extra_instructions", "max_tokens"]
    assert req["chunks"][1]["default"] == "empty only" and req["max_tokens"][1]["default"] == 2048
    assert req["template"][1]["default"] == "character replace (Ref2VA, timeline)" and req["frames_per_chunk"][1]["default"] == 24
    assert sd.SeamStitchSwapDraft.OUTPUT_NODE and "sheet" in sd.SeamStitchSwapDraft.INPUT_TYPES()["optional"]


# ---------------------------------------------------------------- the workshop's fixes (B4)

def test_a_looping_answer_is_cut_and_video_1_never_swallows_the_shots():
    loop = "She lifts a card. " + "She puts it down. She picks it up. " * 30
    r = sd.parse_qwen("video_1: a room [Shot 1] " + loop + 'SAYS "Hi."\nsounds: hum', 1)
    assert r["video_1"] == "a room." and r["shots"][0].count("She puts it down.") == 1
    assert r["shots"][0].endswith("<Subject 1> (S1) says <d>[English]Hi.</d>")
    assert sd.tidy_shots(["[Shot 1] Jump cut. She sits.", "[Shot 2] She stands."]) == \
        ["[Shot 1] She sits.", "[Shot 2] Jump cut. She stands."]


def test_omni_in_free_prose_gives_no_events_and_its_voice_loses_its_gender():
    po = sd.parse_omni('A single Australian male speaks: "What have we got here?" A soft rustle.')
    assert po["structured"] is False and po["events"] == "" and po["words"] == "What have we got here?"
    assert sd.parse_omni("transcript: [English] Hi. audio_events: hum")["structured"] is True
    assert sd.neutral_events("A clear male voice speaks in a quiet room.") == "a voice speaks in a quiet room."


def test_omni_in_free_prose_twice_hands_the_words_to_whisper(job):
    class Prose(FakeOmni):
        def transcribe(self, wav, sr):
            self.n += 1
            return 'A man says "Hello there" and a box rustles.'
    o = Prose()
    rep = _draft(job, omni_cls=o)
    cid = sp.load_plan(job["plan"])["chunks"][0]["id"]
    assert o.n == 6                                            # one retry per chunk
    assert any("without its transcript" in w for w in rep["chunks"][cid]["warnings"])
    assert "<Subject 1> (S1) says <d>[English]Hello there.</d>" in sp.load_plan(job["plan"])["chunks"][0]["prompt"]


def test_the_per_shot_template_describes_each_shot_from_its_own_frames():
    p = _plan978()
    shots = [(a - 400, b - 400) for a, b in sd.chunk_shots(p, 400, 608)]
    fr = sd.shot_frames(shots, 209, 16)
    assert [len(x) for x in fr] == [4, 4, 14] and all(a <= f <= b for (a, b), x in zip(shots, fr) for f in x)
    lines = [{"text": "One.", "frames": [5, 9]}, {"text": "Two.", "frames": [40, 60]}, {"text": "Three.", "frames": [205, 208]}]
    assert [x["text"] for x in sd.lines_in_shot(lines, shots[2], 3, 2)] == ["Two.", "Three."]
    assert [x["text"] for x in sd.lines_in_shot(lines, shots[0], 3, 0)] == ["One."]


def test_the_per_shot_template_end_to_end(job):
    sp.update_plan(job["plan"], lambda p: p.setdefault("cuts", []).append({"frame": 20, "confirmed": True}))

    class ShotQwen(FakeQwen):
        def ask(self, prompt, image=None, video=None, max_tokens=2048, frame_count=16):
            self.calls.append({"prompt": prompt, "frames": None if video is None else int(video.shape[0])})
            if prompt.startswith("These are"):
                return "person: yes\nobjects: white card"
            if "continuous shot" in prompt:
                return "She holds up a white card." + (' SAYS "Hello there."' if "Hello there" in prompt else "")
            return "video_1: a room\nsounds: hum"
    q = ShotQwen()
    _draft(job, qwen_cls=q, template=sd.TEMPLATES[1])
    c0 = sp.load_plan(job["plan"])["chunks"][0]
    dd = sd.parse_sections(c0["prompt"])["detailed_description"]
    assert "[Shot 1] She holds up a white card." in dd and "[Shot 2] Jump cut. She holds up a white card." in dd
    assert dd.count("<d>[English]Hello there.</d>") == 1
    shot_calls = [c for c in q.calls if "continuous shot" in c["prompt"]]
    assert len(shot_calls) >= 4 and all(c["frames"] >= 4 for c in shot_calls)


def test_each_shot_gets_exactly_its_own_lines_in_the_transcription_words():
    lines = [{"text": "Link app,"}, {"text": "product manual,"}, {"text": "got some stickies,"}]
    b = ('[Shot 3] Jump cut. She unfolds a paper, saying “Link app,” then “Product manual.” '
         '<Subject 1> (S1) says <d>[English]Lovely bundled.</d> She lifts strips.')
    out, added = sd.place_dialogue(b, lines)
    assert out.count("<d>[English]Link app,</d>") == 1 and out.count("<d>[English]product manual,</d>") == 1
    assert "Lovely bundled" not in out                       # another shot's line: moved there, not kept here
    assert "<d>[English]product manual,</d> <Subject 1> (S1) says <d>[English]got some stickies,</d>" in out and added == 1
    assert "saying" not in out and "“" not in out


def test_a_speech_verb_before_a_tag_goes_and_long_shots_split_only_when_asked():
    out, _ = sd.place_dialogue("She reads it and she mutters <Subject 1> (S1) says <d>[English]Product manual.</d> quietly.",
                               [{"text": "Product manual."}])
    assert out == "She reads it and <Subject 1> (S1) says <d>[English]Product manual.</d> quietly."
    assert sd.shot_segments([(0, 179)], 25, 0) == [(0, (0, 179), 0, 1)]
    assert sd.shot_segments([(0, 179)], 25, 4.0) == [(0, (0, 89), 0, 2), (0, (90, 179), 1, 2)]


def test_a_line_belongs_to_the_shot_its_middle_is_in_and_a_stretched_first_word_is_trimmed():
    words = [{"word": "So", "start": 0.76, "end": 1.26}, {"word": "what", "start": 1.26, "end": 1.38},
             {"word": "here?", "start": 1.7, "end": 1.86}]
    al = sd.align_words("So what here?", words)
    assert al[0]["start"] == pytest.approx(0.86)
    ln = sd.dialogue_lines(al, 25, 209)[0]
    assert ln["frames"] == [22, 46]
    shots = [(0, 12), (13, 28), (29, 208)]                    # 400-608: cuts at 413 and 429
    assert sd.lines_in_shot([ln], shots[2], 3, 2) == [ln] and sd.lines_in_shot([ln], shots[1], 3, 1) == []


def test_a_subject_answer_without_its_labels_still_parses():
    r = sd.parse_subject("pink-clown angel  \na young woman, red clown nose, pink hair with bangs, rainbow choker  \nshe")
    assert r == {"name": "pink-clown angel", "appearance": "a young woman, red clown nose, pink hair with bangs, rainbow choker",
                 "pronoun": "she"}


def test_a_dropped_line_goes_back_in_order_and_stage_notes_are_not_words():
    lines = [{"text": "One."}, {"text": "Two."}, {"text": "Three."}]
    b = "A. <Subject 1> (S1) says <d>[English]One.</d> B. <Subject 1> (S1) says <d>[English]Three.</d> C."
    out, added = sd.place_dialogue(b, lines)
    assert added == 1 and out.index("Two.") > out.index("One.") and out.index("Two.") < out.index("Three.")
    po = sd.parse_omni("transcript: [English] Lovely bundled and a... (voice trails off) audio_events: hum")
    assert po["words"] == "Lovely bundled and a..." and "voice trails off" in po["notes"]


# ---------------------------------------------------------------- the timeline template (B4 round 1's winner)

def test_timeline_moments_merge_into_a_shot_with_the_lines_at_their_times():
    assert sd.moment_frames((29, 208), 25) == sorted(set(sd.moment_frames((29, 208), 25)))
    mf = sd.moment_frames((29, 208), 25)
    assert mf[0] == 31 and mf[-1] == 206 and len(mf) == 15
    assert sd.moment_frames((0, 12), 25) == [2, 10]
    m = sd.parse_moment("pose: leaning forward over the box\nlook: at the camera\n"
                        "hands: holds a small white card up toward the camera with the right hand")
    assert m == {"person": "", "pose": "leaning forward over the box", "look": "at the camera",
                 "hands": "holds a small white card up toward the camera with the right hand", "others": "", "frame": ""}
    moments = [(0.1, {"pose": "leaning forward over the box", "look": "down into the box",
                      "hands": "reaches into the box"}),
               (0.6, {"pose": "", "look": "at the camera", "hands": "holds a small white card up toward the camera"}),
               (1.1, {"pose": "", "look": "at the camera", "hands": "holds a small white card up toward the camera"}),
               (1.6, {"pose": "", "look": "down into the box", "hands": "hold a flat cream-and-yellow card at the chest; "
                                                                   "the person's other hand rests on their knee"})]
    lines = [{"text": "What have we got here?", "frames": [20, 30]}]
    out = sd.merge_moments(moments, lines, 25, "she")
    assert out.startswith("She sits leaning forward over the box and reaches into the box, looking down into the box.")
    assert out.count("small white card") == 1                      # a repeated moment is dropped
    assert out.index("small white card") < out.index("<d>[English]What have we got here?</d>")
    assert "She holds a flat cream-and-yellow card at the chest, and she other hand" not in out
    assert "her other hand rests on her knee" in out and "holds a flat cream-and-yellow card" in out


def test_low_memory_before_qwen_is_reported(job, monkeypatch):
    monkeypatch.setattr(sd, "ram_gb", lambda: {"available": 6.0, "total": 64.0, "swap_free": 2.0})
    rep = _draft(job)
    assert rep["ram_gb_before_qwen"]["available"] == 6.0
    assert any("low memory before Qwen" in w for w in rep["warnings"])


def test_a_shot_opens_with_a_grammatical_pose():
    m = lambda pose: sd.merge_moments([(0.1, {"pose": pose, "look": "at the camera", "hands": "holds a card"})], [], 25, "she")  # noqa: E731
    assert m("sitting on the floor leaning against the wall").startswith("She is sitting on the floor leaning against the wall and holds")
    assert m("leaning forward over the box").startswith("She sits leaning forward over the box and holds")
    assert m("stands by a window").startswith("She stands by a window and holds")
    assert m("dancing on the grass").startswith("She is dancing on the grass and holds")


def test_qwen_loads_once_then_asks_through_generate(monkeypatch):
    calls = []

    class Node:
        model = None

        def load_model(self, *a):
            calls.append("load")
            self.model = object()

        def run(self, *a):
            calls.append("run")
            return ("via run",)

        def generate(self, prompt, *a, **k):
            calls.append("generate")
            return "via generate"

        def clear(self):
            self.model = None

    r = sd.QwenRunner.__new__(sd.QwenRunner)
    r.node, r.model, r.quant, r.attention = Node(), "m", "None (FP16)", "auto"
    r.load()
    assert r.ask("q1") == "via generate" and r.ask("q2") == "via generate"
    assert calls == ["load", "generate", "generate"]



# ---------------------------------------------------------------- other content (B4 content tests: phone, dance, cartoon)

def test_omni_loops_songs_speakers_and_wrappers():
    loop = sd.parse_omni("transcript: [English] " + "Oh, yeah! " * 40 + "audio_events: drums")
    assert loop["looping"] and sd.is_looping("Pika... " * 30) and not sd.is_looping("What have we got here? Link app.")
    song = sd.parse_omni('transcript: [English] "and I will let go of the world" (sung) audio_events: A male voice sings, '
                         'with a piano.')
    assert song["sung"] and song["words"] == "and I will let go of the world"
    cartoon = sd.parse_omni("**Transcript:** [English] Adventure time! **Audio Events:** - A male voice sings the phrase.")
    assert cartoon["words"] == "Adventure time!" and cartoon["sung"] and cartoon["structured"]
    talk = sd.parse_omni('transcript: [English] - [00:00-00:02] Man: "Oh." - [00:02-00:03] Man: "What?" - [00:04-00:05] '
                         'Woman: "What\'s the date today?" audio_events: - a hum')
    assert talk["words"] == "Oh. What? What's the date today?" and not talk["sung"]
    assert talk["speakers"] == ["Man", "Man", "Woman", "Woman", "Woman", "Woman"]
    ts = sd.parse_omni("transcript: [English] [00:00.000 - 00:03.120] And it's me... [00:03.120 - 00:04.520] ...again "
                       "audio_events: a squeak")
    assert ts["words"] == "And it's me... ...again"
    assert sd.interjections_only("Wow! Woo! Woo! Oh, yeah!") and not sd.interjections_only("Oh, what's that?")
    # B5b: a hum among real lines (Omni's text, no Whisper words) became a line she says
    lines = [{"text": t} for t in ["Slide that baby out.", "Mmm mmm mmm mmm mmm mmm mmm mmm mmm mmm mmm mmm mmm mmm.",
                                   "Don't need it.", "Yeah.", "Oh, wow!"]]
    kept, hum = sd.drop_hum_lines(lines)
    assert [ln["text"] for ln in kept] == ["Slide that baby out.", "Don't need it.", "Yeah.", "Oh, wow!"]
    assert hum.startswith("Mmm mmm mmm")
    assert sd.drop_hum_lines([{"text": "Oh yeah yeah!"}])[1] == "Oh yeah yeah!"
    # B7: Omni's "male speaker:" / "female speaker:" labels had stayed in the lines
    two = sd.parse_omni("transcript: [English] male speaker: Okay, of course. female speaker: You do. audio_events: a thump")
    assert two["words"] == "Okay, of course. You do."
    assert [sd.gender_of(x) for x in two["speakers"]] == ["m", "m", "m", "f", "f"]


def test_speakers_reach_the_lines_and_split_them():
    words = [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, w in enumerate(["Oh.", "What?", "What's", "the",
                                                                                     "date", "today?"])]
    al = sd.align_words("Oh. What? What's the date today?", words, ["Man", "Man", "Woman", "Woman", "Woman", "Woman"])
    lines = sd.dialogue_lines(al, 25, 100)
    assert [(ln["text"], ln.get("speaker")) for ln in lines] == [("Oh.", "Man"), ("What?", "Man"),
                                                                  ("What's the date today?", "Woman")]
    assert sd.gender_of("a woman with long brown hair") == "f" and sd.gender_of("a shirtless dancer") is None


def test_moments_without_the_main_person_and_empty_hands():
    pr = sd.PRONOUNS["she"]
    assert sd._hands_clause("not visible holding anything", pr) == ""
    assert sd._hands_clause("resting flat on the table", pr) == "is resting flat on the table"
    assert sd._hands_clause("hold a torch", pr) == "holds a torch"
    moments = [(0.1, {"person": "no", "frame": "a snowy mountain range under a blue sky"}),
               (0.6, {"person": "no", "frame": "snowy mountains under a blue sky"}),
               (1.1, {"person": "no", "frame": "a penguin on the snow"})]
    assert sd.empty_shot(moments) == ("No one is replaced in this shot: a snowy mountain range under a blue sky; then a "
                                      "penguin on the snow. It stays exactly as in <Video 1>.")
    mixed = [(0.1, {"person": "no", "hands": "holds a drum"}), (0.6, {"person": "yes", "pose": "spinning on the grass",
                                                                     "hands": "twirls a flaming torch", "look": "down"})]
    assert sd.merge_moments(mixed, [], 25, "she") == "She is spinning on the grass and twirls a flaming torch, looking down."


def test_the_main_person_is_named_by_what_lasts():
    # 100d chunk 1 (B5a): "a bald man holding a knife" -> every shot without the knife answered "no"
    assert sd.who_lasting("a bald man holding a knife") == "a bald man"
    assert sd.who_lasting("a woman, sitting at a table with a sign") == "a woman"
    assert sd.who_lasting("a man wearing a black t-shirt") == "a man wearing a black t-shirt"
    assert sd.who_lasting("a girl in a blue dress") == "a girl in a blue dress"
    assert sd.who_lasting("a shirtless dancer in a red sarong") == "a shirtless dancer in a red sarong"
    assert sd.who_lasting("dancing") == "dancing"            # nothing left to name them by: kept as it was
    assert "never by what they hold or do" in sd.WHO_INSTRUCTION


def test_a_no_that_describes_the_person_is_asked_again():
    # 100d 186-208 (B5a): "person: no" three times, each with a pose and the hands -> written as an empty shot
    asked = []

    def ask(q, pic):
        asked.append(pic)
        return "Yes."
    bent = [(t, {"person": "no", "pose": "sitting on floor leaning forward", "look": "down",
                 "hands": "holds a knife near cardboard box with both hands"}) for t in (0.1, 0.4, 0.8)]
    assert sd.recheck_absent(bent, ["p0", "p1", "p2"], "a bald man in a black t-shirt", ask) == 3
    assert asked == ["p0", "p1", "p2"] and all(sd.present(m) for _, m in bent)
    # an empty landscape describes no one: never asked, stays empty
    asked.clear()
    land = [(0.1, {"person": "no", "pose": "none visible", "hands": "none visible holding anything", "frame": "hills"})]
    assert sd.recheck_absent(land, ["p0"], "a boy", ask) == 0 and asked == [] and not sd.present(land[0][1])
    # a "no" that Qwen confirms stays a no; a shot with a "yes" is never re-asked
    other = [(0.1, {"person": "no", "pose": "riding atop a yellow creature", "hands": ""})]
    assert sd.recheck_absent(other, ["p0"], "a boy", lambda q, pic: "no") == 0 and not sd.present(other[0][1])
    mixed = [(0.1, {"person": "yes"}), (0.6, {"person": "no", "pose": "kneeling"})]
    assert sd.recheck_absent(mixed, ["p0", "p1"], "a boy", ask) == 0 and asked == []


def test_the_summary_names_the_main_person_and_keeps_everyone_else():
    t = sd.fill_ref2va(subject=SUBJECT, video_1="a garden.", shots=["[Shot 1] She dances."], sounds="drums", n_shots=1,
                       dialogue=False, audio=True, pronoun="she", who="a shirtless dancer in a red sarong", others=True)
    s = sd.parse_sections(t)["summary"]
    assert "the main person (a shirtless dancer in a red sarong) is replaced by <Subject 1>" in s
    assert "Everyone else in <Video 1> keeps exactly their own appearance and movements" in s


def test_looping_omni_and_a_song_never_become_her_lines(job):
    rep = _draft(job, omni_cls=FakeOmni("transcript: [English] " + "Pika... " * 40 + "audio_events: music"),
                 whisper=FakeWhisper())
    cid = sp.load_plan(job["plan"])["chunks"][0]["id"]
    assert any("a loop" in w for w in rep["chunks"][cid]["warnings"])
    sung = FakeOmni('transcript: [English] "Hello there." (sung) audio_events: A male voice sings over a ukulele.')
    rep = _draft(job, omni_cls=sung, mode="all (into the draft field)")
    c0 = sp.load_plan(job["plan"])["chunks"][0]
    d = c0.get("draft") or c0["prompt"]
    assert "<d>" not in d and 'a voice sings "Hello there."' in d



def test_more_poses_and_body_part_hands_read_as_sentences():
    m = lambda pose, hands: sd.merge_moments([(0.1, {"pose": pose, "look": "", "hands": hands})], [], 25, "she")  # noqa: E731
    assert m("riding atop a yellow creature", "").startswith("She is riding atop a yellow creature")
    assert m("leaning on the table", "").startswith("She sits leaning on the table")
    assert sd._hands_clause("arms outstretched upward with palms open", sd.PRONOUNS["she"]) ==         "has her arms outstretched upward with palms open"


def test_the_job_can_name_who_is_replaced():
    p = _plan978()
    spl.apply_op(p, {"op": "set_target", "target": "  the boy in the white hat  "})
    assert p["target"] == "the boy in the white hat"
    spl.apply_op(p, {"op": "set_target", "target": ""})
    assert p["target"] == ""


def test_the_strip_gets_a_playable_copy_of_a_source_the_browser_cannot_decode(tmp_path):
    import subprocess
    import av
    import timeline as tl
    src = str(tmp_path / "phone.mp4")
    subprocess.run([tl._ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=1",
                    "-c:v", "mpeg4", src], check=True)
    jd = str(tmp_path / "job")
    plan = {"source": {"path": src, "fps": 30}}
    assert spl.video_codec(src) == "mpeg4"
    view = spl.source_view(plan, jd)
    assert view.endswith("cache/source_view.mp4")
    with av.open(view) as c:
        v = c.streams.video[0]
        assert v.codec_context.name == "h264" and sum(1 for _ in c.decode(v)) == 30
    assert spl.source_view(plan, jd) == view                    # made once
    h264 = str(tmp_path / "ok.mp4")
    subprocess.run([tl._ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", h264], check=True)
    assert spl.source_view({"source": {"path": h264, "fps": 25}}, jd) == h264


# ---------------------------------------------------------------- B5b: B5a's drafter items

OMNI_SUNG = ("transcript: [English] put the tray in now. [sings] La la la la la la... Cookies. [sings] La la la "
            "la la la... Cookies. audio_events: A soft, low-pitched thud occurs at the start. A male voice sings a "
            "simple, low, rhythmic melody, \"La la la la la la...\". A brief, high-pitched \"ding\" at the end.")


def test_a_spoken_line_before_a_song_is_kept():
    po = sd.parse_omni(OMNI_SUNG)
    assert po["sung"] and po["spoken"] == "put the tray in now." and po["sung_words"].startswith("La la")
    # the old forms still read the same
    song = sd.parse_omni('transcript: [English] "and I will let go of the world" (sung) audio_events: piano')
    assert song["sung"] and song["spoken"] == "" and song["sung_words"] == "and I will let go of the world"
    both = sd.parse_omni("transcript: [English] Hi there. (singing) la la la la (speaks) Okay now. audio_events: hum")
    assert both["spoken"] == "Hi there. Okay now." and both["sung_words"] == "la la la la"


class SilentWhisper(FakeWhisper):
    def words(self, audio16k, language=None):
        return [], "en"


def test_whisper_hearing_nothing_keeps_omnis_spoken_words(job):
    # B5a: Whisper returned no words, Omni marked the stretch sung; the line was dropped
    rep = _draft(job, omni_cls=FakeOmni(OMNI_SUNG), whisper=SilentWhisper(), mode="all (into the draft field)")
    c0 = sp.load_plan(job["plan"])["chunks"][0]
    d = c0.get("draft") or c0["prompt"]
    assert "put the tray in now." in d and "<d>[English]put the tray in now.</d>" in d
    assert "La la" not in d.split("overall_soundscape")[0].split("detailed_description")[-1] or "sings" in d
    cid = c0["id"]
    assert any("sung" in w for w in rep["chunks"][cid]["warnings"])
    # six plain spoken words Whisper didn't hear are kept now, flagged; one or two Whisper words still drop them
    rep = _draft(job, omni_cls=FakeOmni("transcript: [English] We open the box and look inside. audio_events: hum"),
                 whisper=SilentWhisper(), mode="all (into the draft field)")
    c0 = sp.load_plan(job["plan"])["chunks"][0]
    assert "We open the box and look inside." in (c0.get("draft") or c0["prompt"])
    assert any("Whisper heard nothing" in w for w in rep["chunks"][cid]["warnings"])
    rep = _draft(job, omni_cls=FakeOmni("transcript: [English] We open the box and look inside. audio_events: hum"),
                 whisper=FakeWhisper(), mode="all (into the draft field)")
    assert any("likely invented" in w for w in rep["chunks"][cid]["warnings"])


def test_the_subjects_pronoun_wins_in_a_caption():
    she, he = sd.PRONOUNS["she"], sd.PRONOUNS["he"]
    assert sd._third_to(she, "holds the knife near his mouth") == "holds the knife near her mouth"
    assert sd._third_to(she, "turns the box in his hands, He looks at himself") == \
        "turns the box in her hands, She looks at herself"
    assert sd._third_to(he, "holds her phone; she smiles") == "holds his phone; he smiles"
    assert sd._third_to(she, "rests her chin on her hand") == "rests her chin on her hand"
    m = sd.merge_moments([(0.1, {"pose": "sitting on the floor", "look": "at his hands",
                                 "hands": "holds a cone near his mouth"})], [], 25, "she")
    assert " his " not in m and "near her mouth" in m and "at her hands" in m


def test_the_scene_knows_who_is_replaced_and_the_prompt_keeps_everyone_else():
    # B7, a two-person shot: the scene line was written around the man while the woman was the one replaced
    instr = sd.SCENE_INSTRUCTION.format(n=3, talk="", who="a woman in a red dress", sounds_hint="")
    assert "The main person is a woman in a red dress." in instr and "\nothers: " in instr
    ans = ("video_1: She sits on the left of a dark blue diner booth, a white cup in her hands.\nsounds: chatter.\n"
           "others: a man in a striped jacket on the right")
    assert sd._labelled(ans)["others"] == "a man in a striped jacket on the right"
    t = sd.fill_ref2va(subject=SUBJECT, video_1="a diner booth.", shots=["[Shot 1] She sits."], sounds="chatter", n_shots=1,
                       dialogue=False, audio=True, pronoun="she", who="a woman in a red dress",
                       others="a man in a striped jacket on the right")
    sec = sd.parse_sections(t)
    assert "Everyone else in <Video 1> (a man in a striped jacket on the right) keeps exactly" in sec["summary"]
    assert "The other people in <Video 1> (a man in a striped jacket on the right): fully_preserved" in sec["retention_analysis"]
    src = open(sd.__file__, encoding="utf-8").read()
    assert src.index('ci["who"] = who') < src.index("sinstr = SCENE_INSTRUCTION.format(")


def test_a_line_another_face_says_is_theirs_as_subject_2(monkeypatch):
    # B7, a two-person scene: every line went to her. Whose mouth moves decides (numbers like the measured ones).
    import swap_scores as ss
    assert ss.speaker_of([0.30, 0.31, 0.30, 0.32], [0.20, 0.38, 0.22, 0.40]) == "other"       # his mouth moves
    assert ss.speaker_of([0.20, 0.27, 0.21, 0.28], [None, None, None, None]) == "target"      # hers, alone
    assert ss.speaker_of([0.20, 0.23, 0.20, 0.24], [0.20, 0.22, 0.20, 0.23]) == "target"      # a close call stays hers
    assert ss.speaker_of([None] * 4, [None] * 4) == "target"
    lines = [{"text": "Okay, stop right there.", "frames": [0, 3]}, {"text": "What?", "frames": [4, 7]},
             {"text": "Sounds good.", "frames": [2, 2]}, {"text": "untimed"}]
    monkeypatch.setattr(ss, "face_mouths", lambda frames, masks, path, fps: (
        [0.2, 0.27, 0.21, 0.28, 0.30, 0.31, 0.30, 0.32], [None, None, None, None, 0.20, 0.38, 0.22, 0.40]))
    assert sd.attribute_lines(lines, [None] * 8, [None] * 8, "x.task", 24) == 1
    assert [ln.get("by") for ln in lines] == [None, "other", None, None]
    lines[1]["tag"] = "<Subject 2> (S2)"
    # a tag Qwen wrote as hers is put back as his where the words are his line
    block, added = sd.place_dialogue('She sits. <Subject 1> (S1) says <d>[English]What?</d> She looks up.', lines[1:2])
    assert "<Subject 2> (S2) says <d>[English]What?</d>" in block and "(S1) says" not in block
    t = sd.fill_ref2va(subject=SUBJECT, video_1="a couch.", shots=[block], sounds="room", n_shots=1, dialogue=True,
                       audio=True, pronoun="she", who="a woman in a purple top", others="a man in a scarf on the right",
                       second="a man in a scarf on the right")
    sec = sd.parse_sections(t)
    assert "<Subject 2> (S2) is a man in a scarf on the right, who keeps their own appearance and movements from <Video 1>." in sec["subject_definitions"]
    assert "<Subject 2> (appears in <Video 1>): fully_preserved" in sec["retention_analysis"]
    assert "Her lips move only with her own lines, and <Subject 2>'s with theirs" in sec["detailed_description"]
