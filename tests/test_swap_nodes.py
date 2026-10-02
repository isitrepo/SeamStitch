"""SeamStitch Swap Planner, Swap Option and Swap Take (B1b): the plan edits, the pins' lineage,
the render run's outputs and routing, and the take's 1:1 save, on small synthetic clips."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
from test_swap import CUTS_978, FR, H_, W_, _add_take, _clip, _frames, job  # noqa: E402,F401


def _plan978(splits=(), cuts=CUTS_978):
    p = sp.new_plan("j", {"path": "x.mp4", "frames": 978, "fps": 25, "width": 1920, "height": 1080, "audio": True})
    for f in cuts:
        spl.apply_op(p, {"op": "add_cut", "frame": f, "from": "detected"})
    for f, m in splits:
        spl.apply_op(p, {"op": "add_split", "frame": f, "mode": m})
    return p


# ---------------------------------------------------------------- Swap Option

def test_option_reads_the_chunk_value_or_the_default():
    assert spl.option_value('{"mark": false}', "mark", "true") == (False, 0, 0.0, "false")
    assert spl.option_value('{"mark": true}', "mark", "false")[0] is True
    assert spl.option_value("{}", "mark", "true")[0] is True
    assert spl.option_value("", "strength", "0.5") == (True, 0, 0.5, "0.5")
    assert spl.option_value('{"steps": 12}', "steps", "8")[1:3] == (12, 12.0)
    assert spl.option_value('{"style": "noir"}', "style", "")[3] == "noir"
    assert spl.SeamStitchSwapOption().read('{"mark": false}')[0] is False


# ---------------------------------------------------------------- plan edits

def test_ops_build_the_near_cut_plan_and_warn_on_the_split_at_256():
    p = _plan978([(47, "cut"), (256, "anchored"), (453, "anchored")])
    c = p["chunks"]
    assert [x["deliver"] for x in c] == [[0, 46], [47, 255], [256, 452], [453, 977]]
    assert c[1]["render"] == [47, 255] and c[1]["length"] == 209 and c[1]["fill"] is None
    assert c[2]["render"] == [244, 452] and c[2]["length"] == 209
    guard = [w for w in p["warnings"] if w["code"] == "guard"]
    # 256's overlap guard [238, 262] holds the cut at 250. (453 warns too: the unrendered chunk to its
    # right runs to the end of the video, so it is head-filled back to 429, onto the cut there.)
    assert [(w["frame"], w["cuts"]) for w in guard] == [(256, [250]), (453, [429])]
    assert guard[0]["guard"] == [238, 262] and "straight cut on that cut" in guard[0]["text"]
    assert c[3]["fill"] == {"kind": "head", "frames": 12}
    assert "anchored split at 256" in spl.plan_text(p) and "⚠" in spl.plan_text(p)

    q = _plan978([(47, "cut"), (250, "cut"), (453, "anchored")])
    c = q["chunks"]
    assert c[1]["deliver"] == [47, 249] and c[1]["fill"] == {"kind": "hold", "frames": 6} and c[1]["held"] == 6
    assert c[2]["render"] == [250, 458] and c[2]["fill"] == {"kind": "tail", "frames": 6}
    assert [w["frame"] for w in q["warnings"] if w["code"] == "guard"] == [453]        # only the far edge
    assert not [w for w in q["warnings"] if w.get("frame") == 250]        # a straight split on a real cut


def test_split_ops_keep_chunk_data_and_choose_take():
    p = _plan978([(203, "cut"), (412, "anchored")])
    cid = p["chunks"][2]["id"]
    spl.apply_op(p, {"op": "set_prompt", "chunk": cid, "prompt": "hello"})
    assert p["chunks"][2]["prompt_state"] == "edited"
    sid = p["splits"][1]["id"]
    spl.apply_op(p, {"op": "move_split", "split": sid, "to": 420})
    assert p["chunks"][2]["id"] == cid and p["chunks"][2]["prompt"] == "hello" and p["chunks"][2]["deliver"][0] == 420
    spl.apply_op(p, {"op": "split_mode", "split": sid, "mode": "cut"})
    assert p["chunks"][2]["render"][0] == 420
    p["chunks"][2]["takes"] = [{"id": f"{cid}-t001", "state": "ok", "render": [420, 977]}]
    spl.apply_op(p, {"op": "choose_take", "chunk": cid, "take": f"{cid}-t001"})
    assert p["chunks"][2]["chosen"] == f"{cid}-t001"
    with pytest.raises(sp.PlanError):
        spl.apply_op(p, {"op": "choose_take", "chunk": p["chunks"][0]["id"], "take": f"{cid}-t001"})
    spl.apply_op(p, {"op": "delete_split", "split": sid})
    assert len(p["chunks"]) == 2 and p["removed_chunks"][0]["id"] == cid
    with pytest.raises(sp.PlanError):
        spl.apply_op(p, {"op": "nope"})
    assert spl._chunk(p, "@500")["deliver"] == [203, 977] and spl._chunk(p, 0)["deliver"] == [0, 202]


# ---------------------------------------------------------------- pins and the descriptor

def _take(c, tid, **pins):
    t = {"id": tid, "state": "ok", "render": list(c["render"]), "pins": {"start": pins.get("start"), "end": pins.get("end")},
         "file": f"chunks/{c['id']}/{tid[-4:]}/take.mkv"}
    c.setdefault("takes", []).append(t)
    return t


def test_pins_follow_the_lineage_forward_then_two_sided():
    p = _plan978([(203, "cut"), (412, "anchored"), (609, "anchored"), (806, "cut")])
    c = p["chunks"]
    assert [x["render"] for x in c[1:4]] == [[203, 411], [400, 608], [597, 805]]
    assert spl.resolve_pins(p, 1)[0] == {"start": None, "end": None}       # straight cut on the left
    pins, notes = spl.resolve_pins(p, 2)
    assert pins == {"start": None, "end": None} and "free start" in notes[0]
    w = _take(c[1], "c2-t001")
    assert spl.resolve_pins(p, 2)[0] == {"start": {"take": w["id"], "frames": [400, 404]}, "end": None}
    x = _take(c[2], "c3-t001", start={"take": w["id"], "frames": [400, 404]})
    assert spl.resolve_pins(p, 3)[0] == {"start": {"take": x["id"], "frames": [597, 601]}, "end": None}
    _take(c[3], "c4-t001", start={"take": x["id"], "frames": [597, 601]})
    # the re-roll of the middle chunk: pinned at both ends to the neighbours' effective takes
    assert spl.resolve_pins(p, 2)[0] == {"start": {"take": "c2-t001", "frames": [400, 404]},
                                         "end": {"take": "c4-t001", "frames": [604, 608]}}
    # a chunk whose render ends in held frames can't be pinned at its end
    q = _plan978([(47, "cut"), (250, "cut"), (453, "anchored")])
    _take(q["chunks"][2], "c3-t001")
    pins, notes = spl.resolve_pins(q, 1)
    assert pins == {"start": None, "end": None}


def test_descriptor_refuses_an_empty_prompt_and_snapshots_the_run():
    p = _plan978([(203, "cut"), (412, "anchored")])
    with pytest.raises(spl.PlannerError, match="no prompt"):
        spl.render_descriptor(p, "plan.json", {"action": "render", "chunk": p["chunks"][1]["id"]})
    d = spl.render_descriptor(p, "plan.json", {"action": "render", "chunk": p["chunks"][1]["id"], "prompt": "x", "seed": 7,
                                                "options": {"mark": False}})
    assert d["prompt"] == "x" and d["seed"] == 7 and d["options"] == {"mark": False} and d["length"] == 209
    assert d["take"] == f"{p['chunks'][1]['id']}-t001" and d["render_fps"] == 24 and d["source"]["fps"] == 25
    with pytest.raises(spl.PlannerError, match="run.action"):
        spl.parse_run('{"action": "explode"}')
    assert spl.parse_run("")["action"] == "none"


# ---------------------------------------------------------------- the node, on a synthetic job

@pytest.fixture
def nodes(job, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(job["dir"].parent.parent), raising=False)
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(job["dir"].parent.parent), raising=False)
    sp.update_plan(job["plan"], lambda p: [c.__setitem__("prompt", f"prompt {c['id']}") for c in p["chunks"]])
    return job


def _run(node_job, action, **kw):
    r = dict(action=action, **kw)
    return spl.SeamStitchSwapPlanner().plan("t", "", 39, 6, 5, 10, 260, True, json.dumps(r))


def test_planner_render_run_emits_frames_audio_and_pins_from_the_takes(nodes):
    p = sp.load_plan(nodes["plan"])
    c = p["chunks"]
    r = [x["render"] for x in c]
    A = _clip(r[0][1] - r[0][0] + 1, 100, 0.5, seed=2)
    ta = _add_take(nodes, 0, A, r[0])
    res = _run(nodes, "render", chunk=c[1]["id"], seed=11)
    out = res["result"]
    desc, images, audio, prompt, seed, length, sp_, ep_, w, h, frate, sfr, opts, draft, asm = out
    assert isinstance(draft, spl.ExecutionBlocker) and isinstance(asm, spl.ExecutionBlocker)
    assert draft.message is None
    assert length == c[1]["length"] == images.shape[0] and (w, h) == (W_, H_) and (frate, sfr) == (24, 25)
    assert prompt == f"prompt {c[1]['id']}" and seed == 11 and json.loads(opts) == {"mark": True}
    held = c[1]["held"]
    n = r[1][1] - r[1][0] + 1
    src = nodes["src"]
    got = (images[:n].numpy() * 255 + 0.5).astype(np.uint8)
    assert np.array_equal(got, np.stack(src[r[1][0]:r[1][1] + 1]))
    if held:
        assert torch.equal(images[n:], images[n - 1:n].expand(held, -1, -1, -1))
    assert abs(audio["waveform"].shape[-1] / audio["sample_rate"] - length / 24.0) < 1e-3   # on H3's clock
    # start pins: the left take's own frames over the render's first 5, exactly (lossless)
    assert desc["pins"]["start"] == {"take": ta, "frames": [r[1][0], r[1][0] + 4]} and ep_.shape[0] == 0
    exp = np.stack(A[r[1][0] - r[0][0]:r[1][0] - r[0][0] + 5])
    assert np.array_equal((sp_.numpy() * 255 + 0.5).astype(np.uint8), exp)
    q = sp.load_plan(nodes["plan"])
    cc = sp.find_chunk(q, c[1]["id"])
    assert cc["state"] == "rendering" and cc["rendering"]["pins"]["start"]["take"] == ta
    assert res["ui"]["seamstitch_swap_plan"][0]["text"].startswith("job t")


def test_planner_assemble_and_empty_runs_block_the_render(nodes):
    out = _run(nodes, "assemble")["result"]
    assert out[14] == nodes["plan"] and all(isinstance(o, spl.ExecutionBlocker) for o in out[:14])
    out = spl.SeamStitchSwapPlanner().plan("t", "", 39, 6, 5, 10, 260, True, "")["result"]
    assert all(isinstance(o, spl.ExecutionBlocker) for o in out)
    sp.update_plan(nodes["plan"], lambda p: p["chunks"][2].__setitem__("prompt", " "))
    with pytest.raises(spl.PlannerError, match="no prompt"):
        _run(nodes, "render", chunk=sp.load_plan(nodes["plan"])["chunks"][2]["id"])


def test_take_saves_1to1_lossless_with_lineage_and_a_review_clip(nodes):
    import swap_take as st
    p = sp.load_plan(nodes["plan"])
    c = p["chunks"]
    r = [x["render"] for x in c]
    A = _clip(r[0][1] - r[0][0] + 1, 100, 0.5, seed=2)
    _add_take(nodes, 0, A, r[0])
    out = _run(nodes, "render", chunk=c[1]["id"], seed=11)["result"]
    desc, images = out[0], out[1]
    L = desc["length"]
    # a stand-in render: the frames shifted, so the take is not the source
    gen = torch.clamp(images * 0.9 + 0.05, 0, 1)
    with pytest.raises(st.TakeError, match="1:1"):
        st.save_take(desc, gen[:-1])
    m = (torch.rand((L, 24, 32)) > 0.5).float()
    take, tdir, rep = st.save_take(desc, gen, source_mask=m, output_mask=m, prompt={"1": {"class_type": "UNETLoader",
                                   "inputs": {"unet_name": "u"}}}, extra_pnginfo={"workflow": {"nodes": []}})
    keep = L - desc["held"]
    assert take["frames"] == keep and take["render"] == desc["render"] and take["pins"] == desc["pins"]
    got = _frames(os.path.join(tdir, "take.mkv"))
    exp = np.stack([st.to_u8(gen[i]) for i in range(keep)])
    assert got.shape[0] == keep and np.array_equal(got, exp)                 # held frames dropped, lossless
    for f in ("proxy.mp4", "review.mp4", "take.json", "prompt.txt", "api_prompt.json", "workflow.json",
              "mask_src.mkv", "mask_out.mkv"):
        assert os.path.isfile(os.path.join(tdir, f)), f
    assert take["scores"]["pose_iou"] == 1.0 and take["settings"]["model"] == "u"
    q = sp.load_plan(nodes["plan"])
    cc = sp.find_chunk(q, desc["chunk"])
    assert [t["id"] for t in cc["takes"]] == [take["id"]] and "rendering" not in cc and cc["chosen"] is None
    # the review: chunk +- 2 s, its left join typed forward (pins from the left take), its right join pending
    assert rep["review"]["window"] == [0, 79]
    assert [(j["split"], j["type"]) for j in rep["joins"]] == [(c[1]["left"], "forward"), (c[2]["left"], "pending")]
    assert len(_frames(os.path.join(tdir, "review.mp4"))) == 80
    # the next render of the same chunk gets the next id
    out2 = _run(nodes, "render", chunk=c[1]["id"], seed=12)["result"]
    assert out2[0]["take"].endswith("-t002")


def test_a_windowed_assembly_is_the_full_assembly_over_those_frames(job, tmp_path):
    p = sp.load_plan(job["plan"])
    r = [c["render"] for c in p["chunks"]]
    A = _clip(r[0][1] - r[0][0] + 1, 100, 0.5, seed=2)
    B = _clip(r[1][1] - r[1][0] + 1, 92, -0.3, seed=3)
    ta = _add_take(job, 0, A, r[0])
    tb = _add_take(job, 1, B, r[1], start={"take": ta, "frames": [r[1][0], r[1][0] + 4]})
    C2 = _clip(r[2][1] - r[2][0] + 1, 130, seed=4)
    _add_take(job, 2, C2, r[2], start={"take": tb, "frames": [r[2][0], r[2][0] + 4]})
    full = _frames(job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit")["file"])
    rep = job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit", window=(20, 60), out_file=str(tmp_path / "w.mkv"),
                             write_plan=False)
    win = _frames(rep["file"])
    assert rep["window"] == [20, 60] and rep["checks"]["frames_ok"] and rep["checks"]["audio_ok"]
    assert len(win) == 41 and np.array_equal(win, full[20:61])
    assert [j["splice"] for j in rep["joins"]] == [30, 55]
    assert len(sp.load_plan(job["plan"])["assembled"]) == 1          # a window never writes the plan
