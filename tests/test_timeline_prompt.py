"""SeamStitch Timeline Prompt: which pictures go to the 27B, in what order, with what note."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_draft as sd  # noqa: E402
import timeline_prompt as tp  # noqa: E402
from test_swap import FR, H_, W_, _clip, _write  # noqa: E402


class FakeRunner:
    last = None

    def __init__(self, mod, model):
        FakeRunner.last = self
        self.closed = False

    def load(self):
        pass

    def ask_parts(self, parts, max_tokens=1024, think=False, max_side=1280):
        self.parts, self.think = parts, think
        return "answer"

    def close(self):
        self.closed = True


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setattr(sd, "locate_qwen_gguf", lambda: object())
    monkeypatch.setattr(sd, "GgufQwenRunner", FakeRunner)
    monkeypatch.setattr(sd, "free_comfy_models", lambda: None)
    return tp.SeamStitchTimelinePrompt()


@pytest.fixture
def clip(tmp_path):
    return _write(str(tmp_path / "cut.mkv"), _clip(60, 100, slope=2.0))


def pic(v):
    return torch.full((1, H_, W_, 3), v)


def run(node, clip, **kw):
    args = dict(first_frame=pic(0.1), last_frame=pic(0.9), prompt="ASK", qwen_model="m", context_pictures=3,
                context_seconds=1.0, think=False, max_tokens=512, source_video_path=clip, start_frame=20,
                end_frame=30, frame_rate=FR)
    args.update(kw)
    return node.run(**args)


def test_context_frames_follow_the_two_pictures(node, clip):
    assert run(node, clip) == ("\n\nanswer",)   # a blank line first, so it sits apart from the alignment line
    p = FakeRunner.last.parts
    assert p[0] == "ASK" and p[1].shape == (H_, W_, 3) and float(p[1][0, 0, 0]) == pytest.approx(0.1)
    assert float(p[2][0, 0, 0]) == pytest.approx(0.9)
    assert isinstance(p[3], str) and "next 3 pictures" in p[3] and "3 pictures after those" in p[3]
    assert len(p) == 4 + 3 + 3
    assert FakeRunner.last.closed


def test_before_frames_are_earlier_than_after_frames(node, clip):
    run(node, clip)
    p = FakeRunner.last.parts
    levels = [float(f.mean()) for f in p[4:]]       # the clip brightens 2 levels a frame
    assert levels == sorted(levels) and levels[2] < levels[3]


def test_no_context_is_just_the_two_pictures(node, clip):
    run(node, clip, context_pictures=0)
    assert len(FakeRunner.last.parts) == 3
    run(node, clip, source_video_path=None)
    assert len(FakeRunner.last.parts) == 3


def test_range_at_the_clip_edges_keeps_what_exists(node, clip):
    run(node, clip, start_frame=0, end_frame=59)
    assert len(FakeRunner.last.parts) == 3
    run(node, clip, start_frame=2, end_frame=50)
    p = FakeRunner.last.parts
    assert "next 2 pictures" in p[3] and "3 pictures after those" in p[3] and len(p) == 4 + 2 + 3


def test_think_is_passed_through(node, clip):
    run(node, clip, think=True)
    assert FakeRunner.last.think is True


def test_no_model_on_disk_is_an_error(node, clip):
    with pytest.raises(sd.DraftError):
        run(node, clip, qwen_model=tp.NO_MODEL)
