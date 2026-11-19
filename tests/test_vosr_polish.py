"""SeamStitchVosrPolish: sizes, pass-through when off, and the call into the VOSR pack's nodes (faked)."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def vp(monkeypatch):
    path = Path(__file__).resolve().parent.parent / "vosr_polish.py"
    spec = importlib.util.spec_from_file_location("seamstitch_vosr_polish", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_nodes(monkeypatch, calls):
    class Out:
        def __init__(self, v):
            self.args = (v,)

    class Loader:
        @classmethod
        def execute(cls, model, dtype):
            calls.append(("load", model, dtype))
            return Out("bundle")

    class Upscale:
        @classmethod
        def execute(cls, model, image, upscale, seed, color, tile, overlap, vtile, voverlap):
            calls.append(("up", tuple(image.shape), upscale, seed, color, tile))
            return Out(image.repeat_interleave(upscale, 1).repeat_interleave(upscale, 2))

    nodes = types.ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {"VOSR2ModelLoader": Loader, "VOSR2Upscale": Upscale}
    monkeypatch.setitem(sys.modules, "nodes", nodes)


def test_plan_sizes(vp):
    assert vp.plan_sizes(720, 1280, vp.MODE_SHARPEN) == ((360, 640), 2, (720, 1280))
    assert vp.plan_sizes(721, 1281, vp.MODE_SHARPEN)[2] == (721, 1281)
    assert vp.plan_sizes(360, 640, vp.MODE_ENLARGE) == ((360, 640), 2, (720, 1280))


def test_off_passes_through_without_loading(vp, monkeypatch):
    calls = []
    _fake_nodes(monkeypatch, calls)
    x = torch.rand(2, 8, 8, 3)
    out, = vp.SeamStitchVosrPolish().polish(x, enable=False)
    assert out is x and calls == []


def test_sharpen_keeps_size_odd_dims_too(vp, monkeypatch):
    calls = []
    _fake_nodes(monkeypatch, calls)
    x = torch.rand(3, 21, 33, 3)
    out, = vp.SeamStitchVosrPolish().polish(x, mode=vp.MODE_SHARPEN, vosr_seed=5)
    assert out.shape == x.shape
    assert calls[0] == ("load", "VOSR2", "default")
    assert calls[1][:3] == ("up", (3, 10, 16, 3), 2) and calls[1][3] == 5


def test_enlarge_doubles(vp, monkeypatch):
    _fake_nodes(monkeypatch, [])
    x = torch.rand(2, 8, 12, 3)
    out, = vp.SeamStitchVosrPolish().polish(x, mode=vp.MODE_ENLARGE)
    assert out.shape == (2, 16, 24, 3)


def test_missing_pack_is_a_clear_error(vp, monkeypatch):
    nodes = types.ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {}
    monkeypatch.setitem(sys.modules, "nodes", nodes)
    with pytest.raises(RuntimeError, match="ComfyUI-VOSR2"):
        vp.SeamStitchVosrPolish().polish(torch.rand(1, 8, 8, 3))
