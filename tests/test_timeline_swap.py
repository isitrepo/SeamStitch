"""SeamStitchTimeline and Swap jobs: an assembly's joins as marks on the strip.

The routes only ever serve a Swap job's own assembled/ files (by matching what the server
listed, never by opening a client path), the marks land on the same picture however the
clip sits on the strip, and nothing a saved Timeline workflow holds changes meaning."""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture()
def out(tmp_path, monkeypatch):
    import folder_paths
    inp, outp = tmp_path / "input", tmp_path / "output"
    inp.mkdir()
    outp.mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(inp), raising=False)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(outp), raising=False)
    return outp


def make_assembly(outp, job, stem, joins, frames=200, fps=24, window=None, mtime=None, ext=".mp4"):
    adir = outp / "seamstitch_swap" / job / "assembled"
    adir.mkdir(parents=True, exist_ok=True)
    video = adir / f"{stem}{ext}"
    video.write_bytes(b"not decoded here")
    report = {"file": str(video), "frames": frames, "window": window, "fps": fps, "job": job, "joins": joins}
    (adir / f"{stem}.report.json").write_text(json.dumps(report), encoding="utf-8")
    if mtime is not None:
        os.utime(video, (mtime, mtime))
    return video


def join(frame, type_="straight", verdict=None, left="c1", right="c2"):
    return {"split": "s1", "frame": frame, "type": type_, "verdict": verdict, "join_verdict": "seamless",
            "left_chunk": left, "right_chunk": right, "repair": "none", "frame_luma": {"at_splice": 1.0}}


def test_lists_jobs_with_an_assembly_newest_first(out):
    import timeline as tl
    make_assembly(out, "old_job", "swap_a", [join(10)], mtime=1000)
    make_assembly(out, "new_job", "swap_b", [join(10)], mtime=3000)
    make_assembly(out, "new_job", "swap_c", [join(10)], mtime=2000, ext=".mkv")
    # not assemblies: no report, a report with no video, a job with no assembled folder, a plan file
    (out / "seamstitch_swap" / "new_job" / "assembled" / "loose.mp4").write_bytes(b"x")
    (out / "seamstitch_swap" / "new_job" / "assembled" / "orphan.report.json").write_text("{}", encoding="utf-8")
    (out / "seamstitch_swap" / "empty_job").mkdir()
    (out / "seamstitch_swap" / "new_job" / "plan.json").write_text("{}", encoding="utf-8")
    jobs = tl.swap_assemblies()
    assert [j["job"] for j in jobs] == ["new_job", "old_job"]
    assert [a["file"] for a in jobs[0]["assemblies"]] == ["swap_b.mp4", "swap_c.mkv"]
    assert jobs[0]["assemblies"][0]["path"] == "seamstitch_swap/new_job/assembled/swap_b.mp4"


def test_no_jobs_folder_is_an_empty_list(out):
    import timeline as tl
    assert tl.swap_assemblies() == []


def test_the_listed_path_is_a_sequence_line_the_timeline_resolves(out):
    import timeline as tl
    v = make_assembly(out, "job", "swap_a", [join(10)])
    path = tl.swap_assemblies()[0]["assemblies"][0]["path"]
    assert os.path.samefile(tl.resolve_path(path), v)


def test_joins_come_from_the_report_sorted(out):
    import timeline as tl
    make_assembly(out, "job", "swap_a", [join(150, "forward", "red", "c15", "c18"), join(56)], frames=200)
    r = tl.swap_joins("seamstitch_swap/job/assembled/swap_a.mp4")
    assert r["job"] == "job" and r["frames"] == 200 and r["fps"] == 24
    assert [(j["frame"], j["type"], j["verdict"]) for j in r["joins"]] == [(56, "straight", None), (150, "forward", "red")]
    assert r["joins"][1]["left_chunk"] == "c15" and r["joins"][1]["right_chunk"] == "c18"
    assert "report" not in r and "abs" not in r          # no server paths go back to the browser


def test_a_windowed_assembly_counts_from_its_first_frame(out):
    import timeline as tl
    # window 100..199 of the source: source frame 150 is the assembly's frame 50; 90 is outside
    make_assembly(out, "job", "swap_w", [join(150), join(90)], frames=100, window=[100, 199])
    r = tl.swap_joins("seamstitch_swap/job/assembled/swap_w.mp4")
    assert [j["frame"] for j in r["joins"]] == [50]


def test_backslashes_and_the_absolute_path_name_the_same_assembly(out):
    import timeline as tl
    v = make_assembly(out, "job", "swap_a", [join(10)])
    assert tl.swap_joins("seamstitch_swap\\job\\assembled\\swap_a.mp4")["file"] == "swap_a.mp4"
    assert tl.swap_joins(str(v))["file"] == "swap_a.mp4"


@pytest.mark.parametrize("bad", [
    "",
    "seamstitch_swap/job/assembled/../assembled/swap_a.mp4",
    "seamstitch_swap/job/assembled/../plan.json",
    "../output/seamstitch_swap/job/assembled/swap_a.mp4",
    "seamstitch_swap/job/assembled/swap_a.report.json",
    "seamstitch_swap/job/takes/t.mp4",
    "seamstitch_swap/other/assembled/swap_a.mp4",
    "a.mp4",
])
def test_anything_but_a_listed_assembly_is_refused(out, bad):
    import timeline as tl
    make_assembly(out, "job", "swap_a", [join(10)])
    takes = out / "seamstitch_swap" / "job" / "takes"
    takes.mkdir()
    (takes / "t.mp4").write_bytes(b"x")
    (out / "a.mp4").write_bytes(b"x")
    with pytest.raises(tl.tm.SequenceError):
        tl.swap_joins(bad)


def test_an_absolute_path_outside_the_jobs_is_refused(out, tmp_path):
    import timeline as tl
    make_assembly(out, "job", "swap_a", [join(10)])
    elsewhere = tmp_path / "elsewhere" / "seamstitch_swap" / "job" / "assembled"
    elsewhere.mkdir(parents=True)
    (elsewhere / "swap_a.mp4").write_bytes(b"x")
    (elsewhere / "swap_a.report.json").write_text(json.dumps({"joins": [join(1)]}), encoding="utf-8")
    for bad in (str(elsewhere / "swap_a.mp4"), str(out / "seamstitch_swap" / "job" / "assembled" / ".." / "assembled" / "x.mp4")):
        with pytest.raises(tl.tm.SequenceError):
            tl.swap_joins(bad)


# ---------------------------------------------------------------- the marks (JS, run under node)

def _js_marks(tmp_path, L, swap, fr):
    node = shutil.which("node")
    if not node:
        pytest.skip("node isn't installed")
    src = open(os.path.join(ROOT, "js", "timeline.js"), encoding="utf-8").read()
    code = src[src.index("// A Swap job's joins on the strip."):src.index("function snapUp(")]
    js = tmp_path / "m.js"
    js.write_text(code + "\nconst a = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));"
                  "\nconsole.log(JSON.stringify({ marks: joinMarks(a.L, a.swap, a.fr),"
                  " colours: a.swap.A.joins.map(joinColour),"
                  " steps: [[0, 1], [56, 1], [1185, 1], [1185, -1], [57, -1], [1400, 1], [10, -1]].map(([f, d]) => (nextMark(joinMarks(a.L, a.swap, a.fr), f, d) || {}).frame ?? null),"
                  " paths: a.paths.map(maybeSwapPath) }));", encoding="utf-8")
    paths = ["seamstitch_swap/job/assembled/swap_a.mp4", "C:\\x\\output\\seamstitch_swap\\job\\assembled\\swap_a.mp4",
             "clip.mp4", "seamstitch_swap/job/takes/t.mp4"]
    (tmp_path / "a.json").write_text(json.dumps({"L": L, "swap": swap, "fr": fr, "paths": paths}), encoding="utf-8")
    return json.loads(subprocess.run([node, str(js), str(tmp_path / "a.json")], capture_output=True, text=True, check=True).stdout)


def _clip(i, path, strip, length, enter=0, exit=None):
    return {"i": i, "e": {"kind": "clip", "path": path, "enter": enter, "exit": exit}, "strip": strip, "len": length}


SWAP = {"A": {"fps": 24, "joins": [{"frame": 56, "type": "straight", "verdict": None},
                                   {"frame": 1185, "type": "forward", "verdict": "red"},
                                   {"frame": 1329, "type": "exit", "verdict": "amber"},
                                   {"frame": 1400, "type": "entry", "verdict": "green"}]}}


def test_marks_on_a_single_clip_sit_on_the_assembly_frame(tmp_path):
    r = _js_marks(tmp_path, [_clip(0, "A", 0, 1437)], SWAP, 24)
    assert [(m["frame"], m["strip"], m["i"]) for m in r["marks"]] == [(56, 56, 0), (1185, 1185, 0), (1329, 1329, 0), (1400, 1400, 0)]
    assert r["colours"] == ["#9ca3af", "#f87171", "#fbbf24", "#34d399"]
    assert r["paths"] == [True, True, False, False]
    # join > / < join from the playhead: strictly after / before it, null past the last / first
    assert r["steps"] == [56, 1185, 1329, 56, 56, None, None]


def test_marks_follow_a_trimmed_clip_placed_after_others(tmp_path):
    # another clip (100f), a gap (20f), then A trimmed to frames 1000..1349 at strip 120
    L = [_clip(0, "B", 0, 100), {"i": 1, "e": {"kind": "gap", "frames": 20}, "strip": 100, "len": 20},
         _clip(2, "A", 120, 350, enter=1000, exit=1350)]
    r = _js_marks(tmp_path, L, SWAP, 24)
    assert [(m["frame"], m["strip"], m["i"]) for m in r["marks"]] == [(1185, 305, 2), (1329, 449, 2)]


def test_a_split_clip_keeps_each_join_once(tmp_path):
    L = [_clip(0, "A", 0, 1185, exit=1185), _clip(1, "A", 1185, 252, enter=1185)]
    r = _js_marks(tmp_path, L, SWAP, 24)
    assert [(m["frame"], m["strip"], m["i"]) for m in r["marks"]] == [(56, 56, 0), (1185, 1185, 1), (1329, 1329, 1), (1400, 1400, 1)]


def test_marks_at_another_strip_rate_keep_their_time(tmp_path):
    # a 25 fps assembly on a strip forced to 50 fps: frame 100 (4.00 s) is strip frame 200
    swap = {"A": {"fps": 25, "joins": [{"frame": 100, "type": "forward", "verdict": "green"}]}}
    r = _js_marks(tmp_path, [_clip(0, "A", 0, 1000)], swap, 50)
    assert [m["strip"] for m in r["marks"]] == [200]


# ---------------------------------------------------------------- a saved workflow is unchanged

# The Timeline's widgets in the order v0.3.0 and v0.4.0 saved them (ComfyUI restores by position).
SAVED_WIDGETS = ["sequence", "target", "frame_rate", "bridge_frame_grid", "context_frames", "extend_frames",
                 "snap_to_multiple", "mismatch_fit", "assemble_crf", "cut_codec", "conform_to_24fps"]


def test_the_widgets_are_unchanged_so_a_saved_workflow_loads_the_same():
    import timeline as tl
    assert list(tl.SeamStitchTimeline.INPUT_TYPES()["required"]) == SAVED_WIDGETS
    assert not tl.SeamStitchTimeline.INPUT_TYPES().get("optional")
    src = open(os.path.join(ROOT, "js", "timeline.js"), encoding="utf-8").read()
    # the Swap marks add no widget: the one DOM panel (serialize false) is still the only one
    assert src.count("addDOMWidget(") == 1 and "addWidget(" not in src
    assert not re.search(r"properties\s*\[|\.properties\.", src)


def test_a_saved_workflow_builds_the_same_cut(out, tmp_path):
    import timeline as tl
    inp = tmp_path / "input"
    for n in ("a.mp4", "b.mp4"):
        _make(inp / n, 40)
    # widgets_values as a v0.4.0 workflow saved them, mapped by position
    saved = ["a.mp4 @ 4..\nb.mp4 @ ..25", '{"mode":"replace","start":20,"end":31}', 0, "ltx (8k+1)", 0, 0, 32,
             "crop", 12, "lossless (ffv1)", False]
    w = dict(zip(SAVED_WIDGETS, saved))
    cut, fr = tl.plan_cut(w["sequence"], w["frame_rate"])
    assert fr == 24 and cut["frames"] == 36 + 25
    assert [(os.path.basename(p["path"]), p["enter"], p["exit"]) for p in cut["pieces"]] == [("a.mp4", 4, 40), ("b.mp4", 0, 25)]
    plan = tl.tm.resolve_target(tl.tm.parse_target(w["target"]), cut)
    assert (plan["mode"], plan["start"], plan["end"]) == ("replace range", 20, 31)


def _make(path, n, fps=24):
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg isn't installed")
    subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={fps}",
                    "-frames:v", str(n), "-pix_fmt", "yuv420p", str(path)], check=True)
