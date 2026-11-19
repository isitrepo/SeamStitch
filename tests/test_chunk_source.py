"""A chunk's take as that chunk's own source: the next change builds on it, and it flips back."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
from test_swap import _add_take, _clip, _frames, job  # noqa: E402,F401


def _takes(job):
    r = [c["render"] for c in sp.load_plan(job["plan"])["chunks"]]
    for i, rr in enumerate(r):
        _add_take(job, i, _clip(rr[1] - rr[0] + 1, 90 + 20 * i, 0.3 * i, seed=5 + i), rr)


def _out(job):
    return str(job["dir"].parent.parent)


def _flip(job, op, chunk, **kw):
    return spl.do_op(dict({"op": op, "job": "t", "chunk": chunk}, **kw), _out(job))


def test_a_chunks_take_becomes_that_chunks_own_source_and_flips_back(job):
    _takes(job)
    p0 = sp.load_plan(job["plan"])
    orig = _frames(p0["source"]["path"])
    c2 = p0["chunks"][1]
    take = c2["takes"][0]
    tf = _frames(os.path.join(str(job["dir"]), take["file"]))
    r = _flip(job, "chunk_source", "c2")["result"]
    p = sp.load_plan(job["plan"])
    assert r["overrides"] == ["c2"] and p["source"].get("patched") and p["source_orig"]["path"] == p0["source"]["path"]
    assert p["source"]["frames"] == p0["source"]["frames"] and p["source"]["audio"] == p0["source"]["audio"]
    new = _frames(p["source"]["path"])
    a, b = c2["deliver"]
    d0 = take["render"][0]
    assert np.abs(new[a:b + 1].astype(int) - tf[a - d0:b - d0 + 1].astype(int)).max() <= 1      # the take's picture in c2
    assert np.abs(new[:a].astype(int) - orig[:a].astype(int)).max() <= 1                          # the rest as filmed
    assert np.abs(new[b + 1:].astype(int) - orig[b + 1:].astype(int)).max() <= 1
    c = p["chunks"][1]
    assert [t["id"] for t in c["takes"]] == [take["id"]] and c["prompt"] == "" and len(p["chunks"][0]["takes"]) == 1   # its takes stay listed
    assert sp.next_take_id(c) == "c2-t002"
    _flip(job, "chunk_source_revert", "c2")
    q = sp.load_plan(job["plan"])
    assert q["source"]["path"] == p0["source"]["path"] and "overrides" not in q and "source_orig" not in q
    assert [t["id"] for t in q["chunks"][1]["takes"]] == [take["id"]] and q["chunks"][1].get("past_takes", []) == []
    with pytest.raises(sp.PlanError):
        _flip(job, "chunk_source_revert", "c2")


def test_one_chunk_leaves_every_other_chunk_alone_and_all_does_every_chunk(job):
    _takes(job)
    before = sp.load_plan(job["plan"])
    _flip(job, "chunk_source", "c2")
    p = sp.load_plan(job["plan"])
    for a, b in zip(before["chunks"], p["chunks"]):
        if a["id"] != "c2":                                          # the others keep takes, choice and prompt as they were
            assert (a["takes"], a["chosen"], a["prompt"], a.get("past_takes", [])) == (b["takes"], b["chosen"], b["prompt"], b.get("past_takes", []))
    r = _flip(job, "chunk_source", "all")["result"]                  # c2 is already edited: it flips again from its (new) take
    p = sp.load_plan(job["plan"])
    assert r["chunks"] == ["c1", "c2", "c3"] and r["skipped"] == {} and sorted(p["overrides"]) == ["c1", "c2", "c3"]   # c2 flips again, from the same take
    assert all(len(c["takes"]) == 1 for c in p["chunks"])                    # nothing is hidden
    r = _flip(job, "chunk_source_revert", "all")["result"]
    q = sp.load_plan(job["plan"])
    assert "overrides" not in q and q["source"]["path"] == before["source"]["path"]
    assert all(len(c["takes"]) == 1 for c in q["chunks"])


def test_back_to_the_original_restores_the_prompt_draft_and_the_drafts_record(job):
    """Q5 (2026-11-18), found on a real job copy: the flip put the prompt and draft back but not the 'drafted' record."""
    _takes(job)
    rec = {"warnings": ["check by eye"], "files": "drafts/c2/", "at": "2026-10-11T21:00:13", "into": "prompt"}
    sp.update_plan(job["plan"], lambda p: p["chunks"][1].update(prompt="my prompt", prompt_state="draft", draft="a draft", drafted=rec))
    _flip(job, "chunk_source", "c2")
    mid = sp.load_plan(job["plan"])["chunks"][1]
    assert mid["prompt"] == "" and not mid.get("drafted")
    _flip(job, "chunk_source_revert", "c2")
    c = sp.load_plan(job["plan"])["chunks"][1]
    assert (c["prompt"], c["prompt_state"], c["draft"], c["drafted"]) == ("my prompt", "draft", "a draft", rec)