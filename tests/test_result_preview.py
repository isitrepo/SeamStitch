"""SeamStitchResultPreview: join ratios on frame-coded clips with known answers."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_timeline import make_clip, dirs  # noqa: E402,F401  (fixture re-export)


def test_hard_cut_reads_hard_and_smooth_motion_reads_seamless(dirs):
    import timeline as tl
    import result_preview as rp
    inp, _ = dirs
    a = make_clip(str(inp / "a.mp4"), 30, 24, offset=0)
    b = make_clip(str(inp / "b.mp4"), 30, 24, offset=20)   # jumps 20 levels at the join
    cut, fr = tl.plan_cut("a.mp4\nb.mp4", 0)
    path = tl.build_cut(cut, fr, crf=0)
    r_cut, _ = rp.join_ratio(path, fr, 30)
    r_mid, _ = rp.join_ratio(path, fr, 15)
    assert rp.verdict(r_cut) == "hard cut" and r_cut > 5
    assert rp.verdict(r_mid) == "seamless"
    worst, at = rp.worst_in_range(path, fr, 20, 40)
    assert at == 30


def test_analyse_counts_inserted_frames_and_rates_inside(dirs):
    """A result that is the cut itself (identity bridge) keeps the hard cut INSIDE the
    regenerated span: both splice joins are clean, the inside row is not."""
    import timeline as tl
    import result_preview as rp
    inp, _ = dirs
    make_clip(str(inp / "a.mp4"), 30, 24, offset=0)
    make_clip(str(inp / "b.mp4"), 30, 24, offset=20)
    cut, fr = tl.plan_cut("a.mp4\nb.mp4", 0)
    path = tl.build_cut(cut, fr, crf=0)
    a = rp.analyse(path, path, fr, 24, 35)
    assert (a["inserted"], a["removed"]) == (12, 12)
    names = {j["name"]: j for j in a["joins"]}
    assert names["into the new frames"]["verdict"] == "seamless"
    assert names["back to the footage"]["verdict"] == "seamless"
    assert names["inside the new frames"]["verdict"] == "hard cut"
    assert names["inside the new frames"]["frame"] == 30
    assert a["before"]["verdict"] == "hard cut" and a["before"]["frame"] == 30


def test_result_path_prefers_audio_file():
    import result_preview as rp
    assert rp._result_path((True, ["x.png", "x.mp4", "x-audio.mp4"])) == "x-audio.mp4"
    assert rp._result_path((True, ["x.png", "x.mp4"])) == "x.mp4"
    with pytest.raises(ValueError):
        rp._result_path((True, ["x.png"]))
