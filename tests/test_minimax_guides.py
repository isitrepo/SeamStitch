"""SeamStitchMiniMaxGuides anchor planning. The node itself is checked against core
ComfyUI's real MiniMaxH3AddGuide outside pytest (it needs a ComfyUI install)."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def mg(monkeypatch):
    nm = types.ModuleType("comfy_extras.nodes_minimax_h3")
    nm.MiniMaxH3AddGuide = object
    ce = types.ModuleType("comfy_extras")
    ce.__path__ = []
    monkeypatch.setitem(sys.modules, "comfy_extras", ce)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3", nm)
    path = Path(__file__).resolve().parent.parent / "minimax_guides.py"
    spec = importlib.util.spec_from_file_location("seamstitch_minimax_guides", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_per_frame_pins_each_context_frame(mg):
    plan = mg.plan_anchors(4, 4, mg.ANCHOR_PER_FRAME)
    assert [(s, i) for s, i, _, _ in plan] == [("start", 0), ("start", 1), ("start", 2), ("start", 3),
                                                ("end", 0), ("end", 1), ("end", 2), ("end", 3)]
    assert [idx for *_, idx in plan] == [0, 1, 2, 3, -4, -3, -2, -1]
    assert all(count == 1 for _, _, count, _ in plan)


def test_k1_is_the_plain_first_last_pair(mg):
    assert mg.plan_anchors(1, 1, mg.ANCHOR_PER_FRAME) == [("start", 0, 1, 0), ("end", 0, 1, -1)]


def test_clip_mode_one_anchor_per_side(mg):
    assert mg.plan_anchors(5, 5, mg.ANCHOR_CLIP) == [("start", 0, 5, 0), ("end", 0, 5, -5)]
    assert mg.plan_anchors(22, 22, mg.ANCHOR_CLIP)[1] == ("end", 0, 22, -22)


@pytest.mark.parametrize("k", [2, 4, 6, 21])
def test_clip_mode_refuses_off_grid_k(mg, k):
    with pytest.raises(ValueError, match="17k\\+5"):
        mg.plan_anchors(k, k, mg.ANCHOR_CLIP)


def test_an_empty_side_is_skipped(mg):
    # the Swap Planner's pins: a side with no rendered neighbour is an empty batch
    assert mg.plan_anchors(5, 0, mg.ANCHOR_CLIP) == [("start", 0, 5, 0)]
    assert mg.plan_anchors(0, 5, mg.ANCHOR_CLIP) == [("end", 0, 5, -5)]
    assert mg.plan_anchors(0, 0, mg.ANCHOR_CLIP) == []
    assert mg.plan_anchors(0, 0, mg.ANCHOR_PER_FRAME) == []
    assert mg.plan_anchors(0, 2, mg.ANCHOR_PER_FRAME) == [("end", 0, 1, -2), ("end", 1, 1, -1)]


def test_clip_mode_still_refuses_an_off_grid_side_beside_an_empty_one(mg):
    with pytest.raises(ValueError, match="start_context has 3"):
        mg.plan_anchors(3, 0, mg.ANCHOR_CLIP)
    with pytest.raises(ValueError, match="end_context has 4"):
        mg.plan_anchors(0, 4, mg.ANCHOR_CLIP)


def test_apply_with_two_empty_sides_returns_the_conditioning_unchanged(mg):
    import torch
    empty = torch.zeros((0, 8, 8, 3))
    cond = [["c", {}]]
    assert mg.SeamStitchMiniMaxGuides().apply(cond, None, None, empty, empty, mg.ANCHOR_CLIP) == (cond,)
