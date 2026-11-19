"""The zone latent's mask maths (swap_zone): no ComfyUI needed."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import swap_zone as sz  # noqa: E402


def test_usable_frames():
    assert sz.usable_frames(107) == 107
    assert sz.usable_frames(106) == 90
    assert sz.usable_frames(5) == 5
    try:
        sz.usable_frames(4)
    except ValueError:
        return
    raise AssertionError("4 frames should be refused")


def test_sample_size_detail_and_cap():
    assert sz.sample_size(512, 512, 1.0, 1.3) == (512, 512)
    assert sz.sample_size(512, 512, 2.0, 1.3) == (1024, 1024)          # 1.05 MP, under the cap
    tw, th = sz.sample_size(768, 720, 2.0, 1.3)                         # 2x would be 2.2 MP: capped near 1.3
    assert tw * th <= 1.3e6 * 1.1 and tw % 32 == 0 and th % 32 == 0
    assert sz.sample_size(768, 720, 3.0, 0) == (2304, 2176)             # no cap, rounded to 32


def test_grow_mask_is_a_square_dilation():
    m = torch.zeros(1, 9, 9)
    m[0, 4, 4] = 1
    g = sz.grow_mask(m, 2, "cpu")
    assert int(g.sum()) == 25 and g[0, 2, 2] == 1 and g[0, 6, 6] == 1 and g[0, 1, 1] == 0


def test_latent_mask_any_pixel_and_frame_pooling():
    mask = torch.zeros(5, 32, 32)
    mask[1, 17, 3] = 1                      # one pixel in frame 1
    lm = sz.latent_mask(mask, [1, 4], 2, 2)  # latent frame 0 = pixel frame 0, latent frame 1 = pixel frames 1..4
    assert lm.shape == (1, 1, 2, 2, 2)
    assert lm[0, 0, 0].sum() == 0
    assert lm[0, 0, 1, 1, 0] == 1 and lm[0, 0, 1].sum() == 1


def test_feather_ramp():
    m = torch.zeros(1, 1, 1, 1, 7)
    m[..., 3] = 1
    f = sz.feather_latent(m, 2)
    assert [round(float(v), 2) for v in f[0, 0, 0, 0]] == [0.0, 0.33, 0.67, 1.0, 0.67, 0.33, 0.0]
    assert torch.equal(sz.feather_latent(m, 0), m)


def test_guide_frames_empty_scene_then_one_anchor():
    assert sz.guide_frames(24, 76, 4) == [(4, "empty"), (8, "empty"), (12, "empty"), (16, "empty"), (20, "empty"), (76, "anchor")]
    assert sz.guide_frames(0, 0, 4) == []                              # there from the start: the anchor is the first frame
    assert sz.guide_frames(0, 30, 4) == [(30, "anchor")]
    assert sz.guide_frames(10, 8, 4) == [(4, "empty"), (8, "anchor")]  # never an empty guide on the anchor's frame
