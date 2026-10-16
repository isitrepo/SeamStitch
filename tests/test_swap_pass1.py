"""The Swap Planner's first-pass size and its pins. A canvas a hair off the source's shape made the guides node
centre-crop a strip off every pin (1.6% of the width per chunk at 1344 x 768 on a 1280 x 720 source), and with each
chunk pinned to the one before, the picture crept wider chunk by chunk. A single pass (first_pass_megapixels > 0)
takes a size that keeps the source's shape and gets its pins stretched to exactly that size."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import conftest  # noqa: E402,F401  (ComfyUI stubs)
import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
from test_swap import job  # noqa: E402,F401


@pytest.mark.parametrize("w, h, mp, want", [
    (1280, 720, 1.0, (1312, 736)),      # the 1344 x 768 the math expression gave is 1.6% too narrow
    (1920, 1080, 1.0, (1312, 736)),
    (720, 1280, 1.0, (736, 1312)),
    (1080, 1080, 1.0, (992, 992)),
    (1440, 1080, 1.0, (1152, 864)),
    (1280, 720, 0.6, (1024, 576)),      # exactly 16:9
    (1280, 720, 1.5, (1536, 864)),
])
def test_fit_size_keeps_the_sources_shape(w, h, mp, want):
    got = spl.fit_size(w, h, mp)
    assert got == want and got[0] % 32 == 0 and got[1] % 32 == 0
    assert abs(got[0] / got[1] / (w / h) - 1) < 0.003
    assert abs(got[0] * got[1] / (mp * 1e6) - 1) <= 0.15


def test_fit_size_beats_rounding_each_side_alone():
    # 1280 x 720 at 1 MP: rounding each side alone gives 1344 x 736, 2.7% too wide; the fit is 0.3% off
    assert spl.pass1_size(1280, 720, area=1e6) == (1344, 736)
    w, h = spl.fit_size(1280, 720, 1.0)
    assert abs(w / h / (1280 / 720) - 1) < abs(1344 / 736 / (1280 / 720) - 1) / 5


def test_the_two_pass_size_is_unchanged():
    assert spl.pass1_size(1920, 1080) == (736, 416) and spl.pass1_size(720, 1280) == (416, 736)


def test_the_widget_is_appended_last_and_off_by_default():
    it = spl.SeamStitchSwapPlanner.INPUT_TYPES()
    assert list(it["optional"]) == ["first_pass_megapixels", "forward_chain"]      # appended in that order
    assert it["optional"]["first_pass_megapixels"][1]["default"] == 0.0
    assert list(it["required"])[-2:] == ["run", "ui_state"]            # nothing before it moved
    assert spl.WIDGET_SETTINGS["first_pass_megapixels"] == "first_pass_mp"
    assert sp.DEFAULT_SETTINGS["first_pass_mp"] == 0.0 and sp._settings({})["first_pass_mp"] == 0.0


def _desc(width=1280, height=720):
    return {"chunk": "c2", "source": {"path": None, "fps": 25, "frames": 1514, "width": width, "height": height},
            "render": [100, 108], "held": 0, "length": 9, "conform": False, "render_fps": 25,
            "pins": {"start": {"source": True, "frames": [100, 100]}, "end": None}}


@pytest.fixture
def decoded(monkeypatch, tmp_path):
    """render_inputs with the video and audio stood in: records the size every decode was asked for."""
    src = tmp_path / "src.mp4"
    src.write_bytes(b"x")
    sizes = []

    def fake_decode(path, fr, first, count, size=None):
        sizes.append(size)
        w, h = size or (1280, 720)
        return torch.zeros((count, h, w, 3), dtype=torch.float32)
    monkeypatch.setattr(spl, "decode_frames", fake_decode)
    monkeypatch.setattr(spl.tl, "probe", lambda path, fr: {"frames": 1514, "width": 1280, "height": 720})
    monkeypatch.setattr(spl.tl, "cut_audio", lambda cut, fr: (np.zeros((2, 4800), dtype=np.float32), 48000))
    return str(src), sizes


def test_a_single_pass_gets_its_pins_stretched_to_the_render_size(decoded, tmp_path):
    src, sizes = decoded
    desc = _desc()
    desc["source"]["path"] = src
    plan = {"settings": {"first_pass_mp": 1.0}}
    frames, _audio, pins, (w, h, fr) = spl.render_inputs(desc, str(tmp_path), plan)
    assert desc["pass1"] == [1312, 736] and (w, h) == (1280, 720)
    assert tuple(pins["start"].shape) == (1, 736, 1312, 3)                 # the pin arrives at the render's size
    assert tuple(pins["end"].shape) == (0, 736, 1312, 3)                   # and so does an empty side
    assert (1312, 736) in sizes and sizes.count(None) == 1                  # only the render's own frames at the source's


def test_the_two_pass_flow_keeps_its_pins_at_the_sources_size(decoded, tmp_path):
    src, sizes = decoded
    desc = _desc()
    desc["source"]["path"] = src
    _frames, _audio, pins, _ = spl.render_inputs(desc, str(tmp_path), {"settings": {}})
    assert desc["pass1"] == [736, 416]
    assert tuple(pins["start"].shape) == (1, 720, 1280, 3) and tuple(pins["end"].shape) == (0, 720, 1280, 3)


def test_the_settings_op_stores_the_size_and_leaves_the_chunks_alone(job):
    import copy
    p = sp.load_plan(job["plan"])
    before = copy.deepcopy(p["chunks"])
    spl.apply_op(p, {"op": "settings", "settings": {"first_pass_mp": 1.0}})
    assert p["settings"]["first_pass_mp"] == 1.0 and p["chunks"] == before
    with pytest.raises(sp.PlanError):
        spl.apply_op(p, {"op": "settings", "settings": {"no_such_setting": 1}})
