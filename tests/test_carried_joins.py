"""A Swap job's joins carried through Timeline -> Result Preview -> Timeline.

Fix one join, "use as timeline", and the result still shows the job's other joins (moved by
the frames the splice added) with the bridged one marked fixed - as many rounds as it takes.
Records only ever name files the server wrote; a client path is only compared with them."""
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_timeline_swap import _clip, _js_marks, join, make_assembly, out  # noqa: E402,F401


def _cut(*pieces):
    """pieces: (path, enter, exit) laid end to end, as timeline_math.resolve_cut does."""
    out_, pos = [], 0
    for p, a, b in pieces:
        out_.append({"path": str(p), "enter": a, "exit": b, "cut_start": pos})
        pos += b - a
    return {"pieces": out_, "gaps": [], "frames": pos}


def _analysis(start, end, inserted, ratios=(1.2, 1.5)):
    import result_preview as rp
    removed = max(0, end - start + 1)
    return {"frames": 1000 + inserted - removed, "inserted": inserted, "removed": removed,
            "joins": [{"name": "into", "ratio": ratios[0], "verdict": rp.verdict(ratios[0])},
                      {"name": "back", "ratio": ratios[1], "verdict": rp.verdict(ratios[1])},
                      {"name": "inside", "ratio": 9.0, "verdict": "hard cut"}]}


def _result(outp, name):
    f = outp / name
    f.write_bytes(b"result " + name.encode())
    return f


JOINS = [join(56), join(300, "forward", "red"), join(600, "exit", "amber"), join(900, "entry", "red")]


def test_the_cut_records_the_joins_it_holds_by_cut_frame(out):
    import timeline as tl
    a = make_assembly(out, "job", "swap_a", JOINS, frames=1000)
    # trimmed to 100.. and placed after a 40-frame clip that is not an assembly
    other = out / "other.mp4"
    other.write_bytes(b"x")
    cut = _cut((other, 0, 40), (a, 100, 1000))
    rec = tl.record_cut_joins(cut, 24, out / "cut.mkv")
    assert [j["frame"] for j in rec["joins"]] == [240, 540, 840]   # 56 is trimmed off
    assert rec["jobs"] == ["job"]


def test_a_cut_without_joins_leaves_no_record(out):
    import timeline as tl
    other = out / "other.mp4"
    other.write_bytes(b"x")
    assert tl.record_cut_joins(_cut((other, 0, 40)), 24, out / "cut.mkv") is None
    assert tl.carry_joins(str(out / "cut.mkv"), str(other), "other.mp4", 5, 10, _analysis(5, 10, 6), 24) is None


def test_the_result_marks_the_bridged_join_fixed_and_moves_the_rest(out):
    import timeline as tl
    a = make_assembly(out, "job", "swap_a", JOINS, frames=1000)
    tl.record_cut_joins(_cut((a, 0, 1000)), 24, a)        # passthrough: the cut is the assembly itself
    r1 = _result(out, "seamstitch_1.mp4")
    # replace 290..309 (20 frames) with 28: the join at 300 is bridged, later ones move +8
    rec = tl.carry_joins(str(a), str(r1), "seamstitch_1.mp4", 290, 309, _analysis(290, 309, 28), 24)
    got = {j["frame"]: j.get("fixed") for j in rec["joins"]}
    assert list(got) == [56, 300, 608, 908]
    assert got[56] is None and got[608] is None and got[908] is None
    assert got[300]["verdict"] == "green" and got[300]["ratio"] == 1.5 and got[300]["result"] == "seamstitch_1.mp4"
    # the strip asks with the path "use as timeline" wrote
    j = tl.file_joins("seamstitch_1.mp4")
    assert j["carried"] and j["job"] == "job" and [x["frame"] for x in j["joins"]] == [56, 300, 608, 908]


def test_a_soft_seam_marks_the_fix_amber(out):
    import timeline as tl
    a = make_assembly(out, "job", "swap_a", JOINS, frames=1000)
    tl.record_cut_joins(_cut((a, 0, 1000)), 24, a)
    r1 = _result(out, "r.mp4")
    rec = tl.carry_joins(str(a), str(r1), "r.mp4", 290, 309, _analysis(290, 309, 20, (1.1, 2.4)), 24)
    assert [j["fixed"]["verdict"] for j in rec["joins"] if j.get("fixed")] == ["amber"]


def test_a_pure_insert_fixes_the_join_it_sits_on(out):
    import timeline as tl
    a = make_assembly(out, "job", "swap_a", JOINS, frames=1000)
    tl.record_cut_joins(_cut((a, 0, 1000)), 24, a)
    r1 = _result(out, "r.mp4")
    rec = tl.carry_joins(str(a), str(r1), "r.mp4", 600, 599, _analysis(600, 599, 12), 24)
    assert [(j["frame"], bool(j.get("fixed"))) for j in rec["joins"]] == [(56, False), (300, False), (600, True), (912, False)]


def test_two_rounds_keep_the_first_fix_and_add_the_second(out):
    import timeline as tl
    a = make_assembly(out, "job", "swap_a", JOINS, frames=1000)
    tl.record_cut_joins(_cut((a, 0, 1000)), 24, a)
    r1 = _result(out, "seamstitch_1.mp4")
    tl.carry_joins(str(a), str(r1), "seamstitch_1.mp4", 290, 309, _analysis(290, 309, 28), 24)
    # round 2: the Timeline now holds result 1 (passthrough again) and bridges the join at 908
    rec1 = tl.record_cut_joins(_cut((r1, 0, 1008)), 24, r1)
    assert [j["frame"] for j in rec1["joins"]] == [56, 300, 608, 908]
    r2 = _result(out, "seamstitch_2.mp4")
    rec = tl.carry_joins(str(r1), str(r2), "seamstitch_2.mp4", 900, 915, _analysis(900, 915, 16), 24)
    fixed = [j["frame"] for j in rec["joins"] if j.get("fixed")]
    assert fixed == [300, 908]
    assert [j["fixed"]["result"] for j in rec["joins"] if j.get("fixed")] == ["seamstitch_1.mp4", "seamstitch_2.mp4"]
    assert [j["frame"] for j in tl.file_joins(str(r2))["joins"]] == [56, 300, 608, 908]


def test_a_rewritten_result_no_longer_carries(out):
    import timeline as tl
    a = make_assembly(out, "job", "swap_a", JOINS, frames=1000)
    tl.record_cut_joins(_cut((a, 0, 1000)), 24, a)
    r1 = _result(out, "r.mp4")
    tl.carry_joins(str(a), str(r1), "r.mp4", 290, 309, _analysis(290, 309, 20), 24)
    r1.write_bytes(b"something else entirely, longer")
    assert tl.file_joins("r.mp4") is None


@pytest.mark.parametrize("bad", ["", "other.mp4", "../r.mp4", "seamstitch_swap/_carried/cut_x.json", "R.MP4/"])
def test_only_a_recorded_result_is_answered(out, bad):
    import timeline as tl
    a = make_assembly(out, "job", "swap_a", JOINS, frames=1000)
    tl.record_cut_joins(_cut((a, 0, 1000)), 24, a)
    tl.carry_joins(str(a), str(_result(out, "r.mp4")), "r.mp4", 290, 309, _analysis(290, 309, 20), 24)
    assert tl.file_joins(bad) is None


def test_the_carried_folder_is_not_listed_as_a_job(out):
    import timeline as tl
    a = make_assembly(out, "job", "swap_a", JOINS, frames=1000)
    tl.record_cut_joins(_cut((a, 0, 1000)), 24, out / "cut.mkv")
    assert [j["job"] for j in tl.swap_assemblies()] == ["job"]


def test_fixed_marks_are_skipped_when_stepping_and_take_the_fix_colour(tmp_path):
    swap = {"A": {"fps": 24, "joins": [{"frame": 56, "type": "straight", "verdict": None},
                                       {"frame": 1185, "type": "forward", "verdict": "red",
                                        "fixed": {"verdict": "green", "ratio": 1.2, "result": "r.mp4"}},
                                       {"frame": 1329, "type": "exit", "verdict": "amber"},
                                       {"frame": 1400, "type": "entry", "verdict": "green"}]}}
    r = _js_marks(tmp_path, [_clip(0, "A", 0, 1437)], swap, 24)
    assert r["colours"][1] == "#34d399"
    assert r["open"] == [56, 1329, 1400]


def test_a_real_cut_from_the_strip_line_records_its_joins(out):
    """The sequence line the strip writes, through plan_cut / build_cut as the node runs it."""
    import timeline as tl
    from test_timeline_swap import _make
    a = make_assembly(out, "job", "swap_a", [join(12, "forward", "red"), join(40, "exit", "amber")], frames=60)
    _make(a, 60)
    line = "seamstitch_swap/job/assembled/swap_a.mp4"
    cut, fr = tl.plan_cut(line, 0)
    path = tl.build_cut(cut, fr)
    assert os.path.samefile(path, a)                                  # passthrough
    assert [j["frame"] for j in tl.record_cut_joins(cut, fr, path)["joins"]] == [12, 40]
    cut, fr = tl.plan_cut(line + " @ 10..50", 0)                      # trimmed: a real cut file
    path = tl.build_cut(cut, fr)
    assert not os.path.samefile(path, a)
    assert [j["frame"] for j in tl.record_cut_joins(cut, fr, path)["joins"]] == [2, 30]
