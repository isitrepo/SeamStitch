"""The planner's undo: the plan goes back to how it was before the last strip edit, and renders made since stay."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
from test_swap import _add_take, _clip, job  # noqa: E402,F401


def _out(job):
    return str(job["dir"].parent.parent)


def _do(job, op, **kw):
    return spl.do_op(dict({"op": op, "job": "t"}, **kw), _out(job))


def _split(job, sid):
    return next(s for s in sp.load_plan(job["plan"])["splits"] if s["id"] == sid)


def test_undo_puts_a_moved_split_back_and_steps_back_one_edit_at_a_time(job):
    s = sp.load_plan(job["plan"])["splits"][0]
    f0 = s["frame"]
    _do(job, "move_split", split=s["id"], to=f0 + 7)
    _do(job, "move_split", split=s["id"], to=f0 + 11)
    assert _split(job, s["id"])["frame"] == f0 + 11
    r = _do(job, "undo")["result"]
    assert _split(job, s["id"])["frame"] == f0 + 7 and "move_split" in r["undid"]
    _do(job, "undo")
    assert _split(job, s["id"])["frame"] == f0
    with pytest.raises(spl.PlannerError):
        _do(job, "undo")


def test_undo_keeps_takes_rendered_after_the_edit(job):
    s = sp.load_plan(job["plan"])["splits"][0]
    f0 = s["frame"]
    _do(job, "move_split", split=s["id"], to=f0 + 5)
    p = sp.load_plan(job["plan"])
    r = p["chunks"][1]["render"]
    _add_take(job, 1, _clip(r[1] - r[0] + 1, 100, 0.2, seed=3), r)
    n = len(sp.load_plan(job["plan"])["chunks"][1]["takes"])
    _do(job, "undo")
    p = sp.load_plan(job["plan"])
    assert _split(job, s["id"])["frame"] == f0 and len(p["chunks"][1]["takes"]) == n


def test_the_plan_view_says_what_the_next_undo_would_undo(job):
    assert spl.plan_view("t", _out(job))["undo"] == {"n": 0, "last": None}
    s = sp.load_plan(job["plan"])["splits"][0]
    _do(job, "move_split", split=s["id"], to=s["frame"] + 3)
    u = spl.plan_view("t", _out(job))["undo"]
    assert u["n"] == 1 and "move_split" in u["last"] and s["id"] in u["last"]


def test_an_end_pin_uses_the_frame_after_when_the_next_take_starts_there(job):
    # 2026-10-30: c9 had been rendered with its left split a cut (take from 1391), then the split was anchored and c8's
    # end pin (frame 1390) found no picture in it: "free end" however forward chain was set
    p = sp.load_plan(job["plan"])
    r1 = p["chunks"][1]["render"]
    r2 = p["chunks"][2]["render"]
    tid = _add_take(job, 2, _clip(r2[1] - (r1[1] + 1) + 1, 100, 0.2, seed=4), [r1[1] + 1, r2[1]])
    sp.update_plan(job["plan"], lambda q: spl.apply_op(q, {"op": "split_mode", "split": q["chunks"][2]["left"], "mode": "anchored"}))
    p = sp.load_plan(job["plan"])
    K = int(sp._settings(p.get("settings"))["anchors"])
    pins, notes = spl.resolve_pins(p, 1)
    assert pins["end"] == {"take": tid, "frames": [r1[1] + 1, r1[1] + K]} and any("frame after" in n for n in notes)
