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
    assert d["prompt"] == "x" and d["seed"] == 7 and d["options"] == {"mark": False, "mode": "character"} and d["length"] == 209
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
    desc, images, audio, prompt, seed, length, sp_, ep_, w, h, frate, sfr, opts, draft, asm = out[:15]
    assert isinstance(draft, spl.ExecutionBlocker) and isinstance(asm, spl.ExecutionBlocker)
    # no mask cached: the mark outputs are blocked, and the render group is told to track its own
    assert all(isinstance(o, spl.ExecutionBlocker) for o in out[15:17]) and out[18] is False
    assert draft.message is None
    assert length == c[1]["length"] == images.shape[0] and (w, h) == (W_, H_) and (frate, sfr) == (24, 25)
    assert prompt == f"prompt {c[1]['id']}" and seed == 11 and json.loads(opts) == {"mark": True, "mode": "character"}
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
    assert out[14] == nodes["plan"] and all(isinstance(o, spl.ExecutionBlocker) for o in out[:14] + out[15:])
    assert len(out) == len(spl.RETURN_TYPES) == len(spl.RETURN_NAMES) == 29
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


def test_take_asks_comfyui_to_free_its_memory_after_the_run(monkeypatch):
    # 2026-10-15: a second render in a row paged to disk (282 s a step against 76) with ComfyUI holding 67.7 GB; the
    # Take node sets the flags ComfyUI's /free sets, which its prompt worker applies once the run is done
    import types
    import server
    import swap_take as st
    flags = {}
    q = types.SimpleNamespace(set_flag=lambda k, v: flags.__setitem__(k, v))
    monkeypatch.setattr(server.PromptServer, "instance", types.SimpleNamespace(prompt_queue=q, routes=None))
    assert st.request_free() is True and flags == {"unload_models": True, "free_memory": True}
    monkeypatch.setattr(server.PromptServer, "instance", types.SimpleNamespace())
    assert st.request_free() is False                       # no queue (tests, an old ComfyUI): nothing breaks
    opt = st.SeamStitchSwapTake.INPUT_TYPES()["optional"]
    assert list(opt)[-5:] == ["free_memory_after", "level_match", "detail_match", "detail_anchor", "person_anchor"] and opt["free_memory_after"][1]["default"] is True


def test_take_levels_itself_to_its_pin(nodes):
    # 2026-10-15: each pinned take came out ~2.3 levels darker than its pin, and the job slid ~15 levels by c7; the
    # take is scaled per channel so its pinned frames match the pin before it is saved
    import swap_take as st
    p = sp.load_plan(nodes["plan"])
    c = p["chunks"]
    r = [x["render"] for x in c]
    A = _clip(r[0][1] - r[0][0] + 1, 100, 0.5, seed=2)
    _add_take(nodes, 0, A, r[0])
    desc, images = _run(nodes, "render", chunk=c[1]["id"], seed=11)["result"][:2]
    keep = desc["length"] - desc["held"]
    dark = torch.clamp(images * 0.9, 0, 1)                          # the render 10% darker than its source
    off, tdir0, _ = st.save_take(desc, dark, review_clip=False)
    assert off["level"]["start"]["step"] < -1 and not off["level"]["start"]["applied"]
    assert [f["code"] for f in off["flags"]] == ["level"]            # off: flagged, the frames kept as rendered
    on, tdir, _ = st.save_take(_run(nodes, "render", chunk=c[1]["id"], seed=12)["result"][0], dark, review_clip=False,
                               level_match=True)
    lv = on["level"]["start"]
    assert lv["applied"] and all(0.8 <= g <= 1.25 for g in lv["gains"]) and on["flags"] == []
    got = _frames(os.path.join(tdir, "take.mkv"))
    pin = np.stack(A[r[1][0] - r[0][0]:r[1][0] - r[0][0] + 5]).astype(np.float64)
    assert abs(got[:5].mean() - pin.mean()) < 1.0                   # the pinned frames now sit at the pin's level
    assert got.shape[0] == keep
    # gains: clamped differences are left alone; a ramp between two pins
    g, step, clamped = st.level_gains([np.full((4, 4, 3), 50, np.uint8)], [np.full((4, 4, 3), 100, np.uint8)])
    assert clamped and abs(step + 50) < 1e-6
    a, b = np.array([1.1, 1.1, 1.1]), np.array([0.9, 0.9, 0.9])
    assert np.allclose(st.gain_at(0, 11, a, b), a) and np.allclose(st.gain_at(10, 11, a, b), b)
    assert np.allclose(st.gain_at(5, 11, a, None), a) and st.gain_at(3, 11, None, None) is None


def test_take_levels_the_room_and_the_person_apart(nodes):
    # the 60 s test job's c5, c6 (2026-10-15): H3 darkened the room more than the kept person; one whole-frame gain matched
    # the room and left her ~1.5 lighter each chunk. With the person's mask, each region gets its own gains.
    import swap_take as st
    p = sp.load_plan(nodes["plan"])
    c = p["chunks"]
    r = [x["render"] for x in c]
    A = _clip(r[0][1] - r[0][0] + 1, 100, 0.5, seed=2)
    _add_take(nodes, 0, A, r[0])
    desc, images = _run(nodes, "render", chunk=c[1]["id"], seed=11)["result"][:2]
    L = desc["length"]
    h, w = int(images.shape[1]), int(images.shape[2])
    mask = torch.zeros((L, h, w))
    mask[:, :, : w // 2] = 1.0                                          # the person: the left half
    # the render: the pinned frames as the left take shows them, then the room darkened 12%, the person only 4%
    base = images.clone()
    k0 = r[1][0] - r[0][0]
    for i in range(5):
        base[i] = torch.from_numpy(A[k0 + i].astype(np.float32) / 255.0)
    m3 = mask[..., None]
    gen = torch.clamp(base * (0.88 * (1 - m3) + 0.96 * m3), 0, 1)
    take, tdir, _ = st.save_take(desc, gen, source_mask=mask, review_clip=False, level_match=True)
    lv = take["level"]["start"]
    assert lv["applied"] and "person_gains" in lv
    assert all(abs(g - 1 / 0.88) < 0.03 for g in lv["gains"]) and all(abs(g - 1 / 0.96) < 0.03 for g in lv["person_gains"])
    got = _frames(os.path.join(tdir, "take.mkv")).astype(np.float64)
    pinf = np.stack(A[k0:k0 + 5]).astype(np.float64)
    for sl in (np.s_[:, :, : w // 2 - 2], np.s_[:, :, w // 2 + 2:]):  # each half lands on the pin's level
        assert abs(got[:5][sl].mean() - pinf[sl].mean()) < 1.0


def test_take_levels_to_its_start_pin_only_and_flags_a_stale_end(nodes):
    # the 60 s test job's c7 (2026-10-16): ramping to an end pin on an older, drifted next take darkened c7's end 12-18%
    import swap_take as st
    p = sp.load_plan(nodes["plan"])
    c = p["chunks"]
    r = [x["render"] for x in c]
    A = _clip(r[0][1] - r[0][0] + 1, 100, 0.5, seed=2)
    _add_take(nodes, 0, A, r[0])
    D = [np.clip(f.astype(np.int16) - 40, 0, 255).astype(np.uint8) for f in _clip(r[2][1] - r[2][0] + 1, 100, 0.5, seed=5)]
    _add_take(nodes, 2, D, r[2])                                     # the next chunk: an old, much darker take
    desc, images = _run(nodes, "render", chunk=c[1]["id"], seed=11)["result"][:2]
    if not desc["pins"].get("end"):
        pytest.skip("the fixture's right split isn't anchored")
    take, tdir, _ = st.save_take(desc, images, review_clip=False, level_match=True)
    assert take["level"]["end"]["applied"] is False and "level_end" in [f["code"] for f in take["flags"]]
    got = _frames(os.path.join(tdir, "take.mkv"))
    keep = desc["length"] - desc["held"]
    exp_last = st.to_u8(images[keep - 1]).astype(np.float64)
    st_ = take["level"].get("start", {})
    g = np.asarray(st_["gains"]) if st_.get("applied") else np.ones(3)
    o = np.asarray(st_["offsets"]) if st_.get("applied") else np.zeros(3)
    assert abs(got[keep - 1].astype(np.float64).mean() - np.clip(exp_last * g + o, 0, 255).mean()) < 1.5   # not pulled to the dark take


def test_end_pinned_chunk_with_no_start_pin_is_levelled_to_its_end_pin(nodes):
    # 2026-10-30: a backward chain from a free chunk (each chunk end-pinned to the one after, no start pin) used to get only a
    # level_end flag; with no start pin it now follows its end pin instead
    import swap_take as st
    p = sp.load_plan(nodes["plan"])
    c = p["chunks"]
    r = [x["render"] for x in c]
    D = _clip(r[2][1] - r[2][0] + 1, 100, 0.5, seed=5)
    _add_take(nodes, 2, D, r[2])
    desc, images = _run(nodes, "render", chunk=c[1]["id"], seed=11)["result"][:2]
    if not desc["pins"].get("end") or desc["pins"].get("start"):
        pytest.skip("the fixture's splits don't give an end pin and no start pin")
    keep = desc["length"] - desc["held"]
    dark = torch.clamp(images * 0.9, 0, 1)
    k0 = desc["pins"]["end"]["frames"][0] - r[2][0]
    for j in range(len(desc["pins"]["end"]["frames"]) and (desc["pins"]["end"]["frames"][1] - desc["pins"]["end"]["frames"][0] + 1)):
        dark[keep - 1 - j] = torch.from_numpy(D[k0 + (desc["pins"]["end"]["frames"][1] - desc["pins"]["end"]["frames"][0]) - j].astype(np.float32) / 255.0 * 0.9)
    take, tdir, _ = st.save_take(desc, dark, review_clip=False, level_match=True)
    assert take["level"]["end"]["applied"] is True and "level_end" not in [f["code"] for f in take["flags"]]
    got = _frames(os.path.join(tdir, "take.mkv")).astype(np.float64)
    pin = D[desc["pins"]["end"]["frames"][1] - r[2][0]].astype(np.float64)
    assert abs(got[keep - 1].mean() - pin.mean()) < 1.5                  # the end now sits at the pin's level


def test_anchor_reference_is_a_take_with_no_pins_at_all(nodes, monkeypatch):
    # 2026-10-30: the detail / colour anchors aim at the first full chunk with no pins; an end-pinned take no longer qualifies
    import swap_take as st
    monkeypatch.setattr(st.tl, "_iter_frames", lambda path, fr, i0, i1: iter([np.full((8, 8, 3), 200, np.uint8)]))
    take = lambda tid, a, b, end: {"id": tid, "render": [a, b], "frames": b - a + 1, "file": "x", "files": {"mask_src": "m"},
                                   "pins": {"start": None, "end": ({"take": "b-t001"} if end else None)}}
    plan = {"chunks": [{"id": "a", "render": [0, 199], "chosen": "a-t001", "takes": [take("a-t001", 0, 199, True)]},
                       {"id": "b", "render": [200, 399], "chosen": "b-t001", "takes": [take("b-t001", 200, 399, False)]}],
            "source": {"fps": 24}}
    got = st.anchor_sample(plan, "job", (8, 8))
    assert got is not None and got[2] == "b-t001"                    # a-t001 starts earlier but is end-pinned


def test_level_match_matches_the_spread_too():
    # the 60 s test job's c3 -> c8 (2026-10-16): the mean held but the couch's contrast rose ~1.5-2 a chunk; the transform
    # now matches each channel's spread (standard deviation) at the pinned frame as well as its mean
    import swap_take as st
    rng = np.random.default_rng(0)
    pin = rng.normal(100, 20, (32, 32, 3)).clip(0, 255)
    out = ((pin - 100) * 1.15 + 97).clip(0, 255)                    # the take: 15% more contrast, 3 levels darker
    T, step, clamped = st.level_gains([out], [pin])
    assert not clamped and np.allclose(T[0], 1 / 1.15, atol=0.02) and step < -2
    fixed = out * T[0] + T[1]
    assert abs(fixed.mean() - pin.mean()) < 0.5 and abs(fixed.std() - pin.std()) < 0.5
    T2, _, _ = st.level_gains([out], [pin], contrast=False)       # the mean only: no offset
    assert np.allclose(T2[1], 0) and abs((out * T2[0]).mean() - pin.mean()) < 0.5


def _fine_energy(f):
    import cv2
    lum = cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.mean((lum - cv2.GaussianBlur(lum, (0, 0), 1.2)) ** 2))


def test_detail_anchor_aims_at_the_reference_not_the_pin():
    # 2026-10-29: pin-matching leaves the shortfall at every join; the anchor's energy replaces the pin's
    import cv2
    import swap_take as st
    ref = _clip(1, 100, seed=5)[0]
    soft = cv2.addWeighted(ref, 0.7, cv2.GaussianBlur(ref, (0, 0), 0.9), 0.3, 0)
    softer = cv2.GaussianBlur(ref, (0, 0), 1.0)
    room = np.ones(ref.shape[:2], bool)
    # the pin is as soft as the take (nothing to restore against it) but the reference is sharper
    assert st.detail_gain([soft], [soft])[0] == 0.0
    k, ratio, _ = st.detail_gain([soft], [soft], [room], anchor=([ref], [room]))
    assert k > 0.05 and ratio > 1
    assert st.detail_gain([ref], [soft], [room], anchor=([softer], [room]))[0] == 0.0     # never softened


def test_detail_gain_restores_the_band_the_round_trip_took():
    # 2026-10-29: each pinned take kept ~0.78 of its pin's fine detail; k boosts the take's fine band to the pin's energy
    import cv2
    import swap_take as st
    pin = _clip(1, 100, seed=5)[0]
    soft = cv2.addWeighted(pin, 0.7, cv2.GaussianBlur(pin, (0, 0), 0.9), 0.3, 0)     # ~0.8 of the band left
    k, ratio, _ = st.detail_gain([soft], [pin])
    assert 0.05 < k <= st.DETAIL_CLAMP[1] and 1 < ratio < 2
    fixed = st.boost_detail(soft, k)
    assert abs(_fine_energy(fixed) / _fine_energy(pin) - 1) < 0.1 < abs(_fine_energy(soft) / _fine_energy(pin) - 1)
    # colour untouched: the same amount on every channel, so the channel differences survive
    d = fixed.astype(np.int16) - soft.astype(np.int16)
    assert np.abs((d[..., 0] - d[..., 1])).max() <= 1 and np.abs((d[..., 1] - d[..., 2])).max() <= 1
    # already as detailed (or more): never softened
    assert st.detail_gain([pin], [soft])[0] == 0.0 and st.detail_gain([pin], [pin])[0] == 0.0
    # a region: only the masked pixels are measured
    m = np.zeros(pin.shape[:2], bool)
    m[:, : pin.shape[1] // 2] = True
    mixed = soft.copy()
    mixed[:, pin.shape[1] // 2:] = pin[:, pin.shape[1] // 2:]
    assert st.detail_gain([mixed], [pin], [m])[0] > 0.05 and st.detail_gain([mixed], [pin], [~m])[0] < 0.01


def test_take_keeps_its_pins_detail(nodes):
    import cv2
    import swap_take as st
    p = sp.load_plan(nodes["plan"])
    c = p["chunks"]
    r = [x["render"] for x in c]
    A = _clip(r[0][1] - r[0][0] + 1, 100, 0.5, seed=2)
    _add_take(nodes, 0, A, r[0])
    desc, images = _run(nodes, "render", chunk=c[1]["id"], seed=11)["result"][:2]
    soft = torch.from_numpy(np.stack([cv2.addWeighted(u, 0.7, cv2.GaussianBlur(u, (0, 0), 0.9), 0.3, 0)
                                      for u in ((x.numpy() * 255).astype(np.uint8) for x in images)])
                            .astype(np.float32) / 255)
    pin = A[r[1][0] - r[0][0]]
    off, tdir_off, _ = st.save_take(desc, soft, review_clip=False, level_match=True)
    assert off["level"]["start"]["detail"]["ratio"] > 1
    assert off["level"]["start"]["detail"]["applied"] is False
    on, tdir_on, _ = st.save_take(_run(nodes, "render", chunk=c[1]["id"], seed=12)["result"][0], soft, review_clip=False,
                                  level_match=True, detail_match=True)
    dt = on["level"]["start"]["detail"]
    assert dt["applied"] and 0.05 < dt["k"] <= st.DETAIL_CLAMP[1]
    e_pin = _fine_energy(pin)
    e_off = _fine_energy(_frames(os.path.join(tdir_off, "take.mkv"))[0])
    e_on = _fine_energy(_frames(os.path.join(tdir_on, "take.mkv"))[0])
    assert abs(e_on / e_pin - 1) < abs(e_off / e_pin - 1) and abs(e_on / e_pin - 1) < 0.2


def test_straight_cut_take_is_held_to_the_reference(nodes, monkeypatch):
    # 2026-10-29: a chunk with no start pin (a straight cut) has no pinned frame, so the anchors measure frames sampled
    # across the take against the job's first full chunk; the reference chunk itself is left alone
    import cv2
    import swap_take as st
    p = sp.load_plan(nodes["plan"])
    c = p["chunks"]
    desc, images = _run(nodes, "render", chunk=c[1]["id"], seed=11)["result"][:2]
    soft = torch.from_numpy(np.stack([cv2.addWeighted(u, 0.7, cv2.GaussianBlur(u, (0, 0), 0.9), 0.3, 0)
                                      for u in ((x.numpy() * 255).astype(np.uint8) for x in images)])
                            .astype(np.float32) / 255)
    n, h, w = int(soft.shape[0]), int(soft.shape[1]), int(soft.shape[2])
    mask = torch.zeros((n, h, w))
    mask[:, :, : w // 3] = 1.0
    sharp = _clip(1, 100, seed=5)[0]
    sharp = cv2.resize(sharp, (w, h))
    room = np.zeros((h, w), bool)
    room[:, w // 3:] = True
    ref = ([sharp] * 4, [room] * 4, "c9-t001", [~room] * 4)
    monkeypatch.setattr(st, "anchor_sample", lambda *a, **k: ref)
    take, tdir, _ = st.save_take(desc, soft, source_mask=mask, review_clip=False, level_match=True, detail_match=True,
                                 detail_anchor=True, person_anchor=True)
    fa = take["level"]["anchor"]
    assert "start" not in take["level"] and fa["ref"] == "c9-t001" and fa["detail"]["applied"] and fa["detail"]["k"] > 0.05
    monkeypatch.setattr(st, "anchor_sample", lambda *a, **k: ("c2-t001" and (ref[0], ref[1], c[1]["id"] + "-t001", ref[3])))
    own, _t, _r = st.save_take(_run(nodes, "render", chunk=c[1]["id"], seed=12)["result"][0], soft, source_mask=mask,
                               review_clip=False, detail_match=True, detail_anchor=True, person_anchor=True)
    assert "anchor" not in own["level"]


def test_detail_map_follows_where_the_detail_was_lost():
    # 2026-10-29: the wall lost less than the sofa; one gain for the room over-sharpened the wall (+25-45%). The map is
    # higher where the take lost more, and stays within 0.5-1.5 x k.
    import cv2
    import swap_take as st
    pin = _clip(1, 100, seed=5)[0]
    blur = cv2.addWeighted(pin, 0.5, cv2.GaussianBlur(pin, (0, 0), 0.9), 0.5, 0)        # lost a lot
    mild = cv2.addWeighted(pin, 0.85, cv2.GaussianBlur(pin, (0, 0), 0.9), 0.15, 0)      # lost a little
    take = mild.copy()
    half = pin.shape[1] // 2
    take[:, half:] = blur[:, half:]
    room = np.ones(pin.shape[:2], bool)
    k, ratio, kmap = st.detail_gain([take], [pin], [room], spatial=True)
    assert kmap is not None and kmap.shape == pin.shape[:2] and kmap.dtype == np.float32
    assert kmap[:, half + 20:].mean() > 1.05 * kmap[:, : half - 20].mean()
    assert kmap.min() >= 0.5 * k - 1e-6 and kmap.max() <= min(1.5 * k, st.DETAIL_CLAMP[1]) + 1e-6
    assert st.detail_gain([take], [pin], [room])[2] is None                              # not asked for
    fixed = st.boost_detail(take, kmap)
    assert abs(_fine_energy(fixed[:, half + 20:]) / _fine_energy(pin[:, half + 20:]) - 1) <         abs(_fine_energy(st.boost_detail(take, k)[:, half + 20:]) / _fine_energy(pin[:, half + 20:]) - 1)
