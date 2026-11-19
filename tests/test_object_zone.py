"""Object add by redrawing a zone (swap_plan guide "zone"): the zone, its settings, the Planner's outputs, the paste-back."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_keep as sk  # noqa: E402
import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
from test_swap import FR as FR_, H_, W_, _write, job  # noqa: F401,E402  (job is a fixture)


def test_zone_predicates_do_not_overlap():
    base = {"mode": "object", "object": {"action": "add", "cutout": {"rgb": "object/c.png", "mask": "object/m.png"}}}
    assert sp.object_composite(base) and not sp.object_zone(base)
    z = {"mode": "object", "object": {"action": "add", "guide": "zone"}}
    assert sp.object_zone(z) and not sp.object_composite(z)
    with_cut = {"mode": "object", "object": dict(base["object"], guide="zone")}
    assert sp.object_zone(with_cut) and not sp.object_composite(with_cut)       # a cut-out is no longer needed
    assert not sp.object_zone({"mode": "object", "object": {"action": "replace", "guide": "zone"}})
    assert not sp.object_zone({"mode": "object", "object": {"action": "add", "guide": "zone", "fill": "picture"}})
    assert not sp.object_zone({"mode": "character", "object": {"action": "add", "guide": "zone"}})


def test_set_object_zone_settings():
    p = {"source": {"frames": 10}, "chunks": [], "splits": []}
    r = spl.apply_op(p, {"op": "set_object", "action": "add", "guide": "zone", "zone_grow": 999, "zone_dx": -1.4,
                         "zone_dy": 9, "zone_aspect": 0.85, "zone_shape": "box", "anchor": "picture",
                         "anchor_at": 3.5, "enter_at": 2})
    o = r["object"]
    assert (o["guide"], o["zone_grow"], o["zone_dx"], o["zone_dy"], o["zone_aspect"]) == ("zone", 400, -1.4, 5.0, 0.85)
    assert (o["zone_shape"], o["anchor"], o["anchor_at"], o["enter_at"]) == ("box", "picture", 3.5, 2.0)
    s = sp.object_settings(r)
    assert s["zone_grow"] == 400 and s["anchor"] == "picture"
    for bad in ({"zone_shape": "star"}, {"anchor": "magic"}, {"guide": "nope"}):
        with pytest.raises(sp.PlanError):
            spl.apply_op(p, dict(op="set_object", **bad))
    d = sp.object_settings({})
    assert (d["anchor"], d["zone_shape"], d["enter_at"], d["anchor_at"]) == ("qwen", "ellipse", 0.0, 0.0)


def test_zone_geometry_sits_on_the_anchor_and_moves_with_dx_dy():
    o = dict(sp.OBJECT_DEFAULTS, size=1.0, sink=0.0, zone_grow=0, zone_aspect=0.5)
    cx, cy, rx, ry = sk.zone_geometry((100.0, 80.0, 40.0), 40.0, o)
    assert (cx, cy, rx, ry) == (100.0, 60.0, 10.0, 20.0)                  # 40 tall resting on the anchor, half as wide
    o2 = dict(o, zone_dx=-1.0, zone_dy=0.5, zone_grow=5)
    cx, cy, rx, ry = sk.zone_geometry((100.0, 80.0, 40.0), 40.0, o2)
    assert (cx, cy, rx, ry) == (60.0, 80.0, 15.0, 25.0)


def _masked_job(job, shape="ellipse", **obj):  # noqa: F811
    box = np.zeros((H_, W_, 3), np.uint8)
    box[30:40, 20:44] = 255                                 # the target: a shoulder
    (job["dir"] / "masks" / "m001").mkdir(parents=True)
    _write(str(job["dir"] / "masks" / "m001" / "mask.mkv"), [box] * job["N"])

    def upd(q):
        q["mode"], q["target"] = "object", "shoulder"
        q["object"] = dict({"action": "add", "guide": "zone", "size": 0.6, "zone_grow": 2, "zone_shape": shape,
                            "feather": 0, "level": "none"}, **obj)
        q["masks"] = [{"id": "m001", "range": [0, job["N"] - 1], "frames": job["N"], "file": "masks/m001/mask.mkv",
                       "target": "shoulder", "created": sp.now(), "size": [W_, H_]}]
        for c in q["chunks"]:
            c["prompt"] = "x"
    sp.update_plan(job["plan"], upd)
    return box


def test_zone_guide_follows_the_target(job):  # noqa: F811
    _masked_job(job)
    plan = sp.load_plan(job["plan"])
    g = sk.ZoneGuide(plan, str(job["dir"]), FR_, (W_, H_), 0, job["N"] - 1, "shoulder")
    assert g.first_width and abs(g.first_width - 24) < 1
    m = g.mask(5)
    assert m.shape == (H_, W_) and m.dtype == np.float32 and set(np.unique(m)) <= {0.0, 1.0}
    ys, xs = np.nonzero(m)
    assert abs(xs.mean() - 32) < 2 and ys.max() <= 41 and ys.min() < 30       # above the shoulder, resting on it
    b = g.box(5)
    assert b[0] <= xs.min() and b[2] >= xs.max() and b[1] <= ys.min() and b[3] >= ys.max()
    assert g.mask(5).sum() == g.mask(60).sum()                                  # same plan -> same zone
    far = sk.ZoneGuide(plan, str(job["dir"]), FR_, (W_, H_), 0, 5, "something else")
    assert far.mask(2) is None and far.box(2) is None


def test_zone_box_shape(job):  # noqa: F811
    _masked_job(job, shape="box")
    g = sk.ZoneGuide(sp.load_plan(job["plan"]), str(job["dir"]), FR_, (W_, H_), 0, 10, "shoulder")
    m = g.mask(3)
    ys, xs = np.nonzero(m)
    assert m.sum() == (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)      # a filled rectangle


def test_planner_zone_render_outputs(job):  # noqa: F811
    from test_swap_nodes import _run
    import folder_paths
    folder_paths.get_output_directory = lambda: str(job["dir"].parent.parent)
    folder_paths.get_temp_directory = lambda: str(job["dir"].parent.parent)
    old = sp.CROP_MIN
    sp.CROP_MIN = 16
    try:
        _masked_job(job, anchor_at=0.5, enter_at=0.25)
        c = sp.load_plan(job["plan"])["chunks"][1]
        out = _run(job, "render", chunk=c["id"], seed=3)["result"]
        desc, images, w, h = out[0], out[1], out[8], out[9]
        assert out[21] is False and out[23] is True                              # not the cut-out refine: the zone path
        assert desc["object_zone"] and "zone_mask" not in desc and desc["zone_frames"] == c["render"][1] - c["render"][0] + 1
        zm = out[22]
        assert tuple(zm.shape) == (desc["length"], h, w) and float(zm.sum()) > 0
        x0, y0, x1, y1 = desc["crop"]
        assert (x1 - x0, y1 - y0) == (w, h) and tuple(images.shape[1:3]) == (h, w)
        # the frames are the source as filmed (no cut-out pasted): a crop of it
        n = c["render"][1] - c["render"][0] + 1
        got = (images[:n].numpy() * 255 + 0.5).astype(np.uint8)
        want = np.stack(job["src"][c["render"][0]:c["render"][1] + 1])[:, y0:y1, x0:x1]
        assert np.array_equal(got, want)
        assert out[24] == 12 and out[25] == 6                                    # 0.5 s and 0.25 s at 24 fps
        assert desc["pins"] == {"start": None, "end": None}
        # not a zone job: the new outputs are off and tiny
        sp.update_plan(job["plan"], lambda q: q["object"].update(guide="composite"))
        out2 = _run(job, "render", chunk=c["id"], seed=3)["result"]
        assert out2[23] is False and tuple(out2[22].shape) == (1, 64, 64)
    finally:
        sp.CROP_MIN = old


def test_zone_without_a_cached_mask_says_so(job):  # noqa: F811
    from test_swap_nodes import _run
    import folder_paths
    folder_paths.get_output_directory = lambda: str(job["dir"].parent.parent)
    folder_paths.get_temp_directory = lambda: str(job["dir"].parent.parent)
    _masked_job(job)
    sp.update_plan(job["plan"], lambda q: q.update(masks=[]))
    c = sp.load_plan(job["plan"])["chunks"][1]
    with pytest.raises(spl.PlannerError, match="mark the chunk first"):
        _run(job, "render", chunk=c["id"], seed=3)


def test_the_assembly_pastes_the_take_only_inside_the_zone(job):  # noqa: F811
    _masked_job(job)
    plan = sp.load_plan(job["plan"])
    comp = sk.ObjectCompositor(plan, str(job["dir"]), job["dir"] / "x.mkv", 24, (W_, H_), 0, job["N"] - 1) \
        if False else None
    # ObjectCompositor reads the source from a file: use the job's own
    src_path = plan["source"]["path"]
    comp = sk.ObjectCompositor(plan, str(job["dir"]), src_path, FR_, (W_, H_), 0, job["N"] - 1)
    assert comp.zone is not None and comp.guide is None
    take = np.full((H_, W_, 3), 250, np.uint8)
    out = comp.apply(10, take)
    zone = comp.zone.mask(10) > 0.5
    src = job["src"][10]
    assert (out[zone] == 250).all()                                              # the take inside the zone
    assert np.array_equal(out[~zone], src[~zone])                                # the source everywhere else


def test_zone_spares_the_target_except_where_it_is_touched(job):  # noqa: F811
    box = _masked_job(job, shape="box", size=1.2, zone_aspect=2.0, zone_spare=True)
    g = sk.ZoneGuide(sp.load_plan(job["plan"]), str(job["dir"]), FR_, (W_, H_), 0, 10, "shoulder")
    m = g.mask(4) > 0.5
    target = box[..., 0] > 0
    assert m.any()
    near = np.zeros_like(target)
    near[30 - sk.SPARE_GROW:40 + sk.SPARE_GROW, 20 - sk.SPARE_GROW:44 + sk.SPARE_GROW] = True
    assert not (m & near & target).any()                          # none of the target's own pixels are in the zone
    sp.update_plan(job["plan"], lambda q: q["object"].update(zone_touch_r=0.3, zone_touch_dy=0.2))
    t = sk.ZoneGuide(sp.load_plan(job["plan"]), str(job["dir"]), FR_, (W_, H_), 0, 10, "shoulder").mask(4) > 0.5
    assert (t & target).any() and t.sum() > m.sum()               # the touch circle puts some of it back


def test_zone_spare_and_touch_settings():
    p = {"source": {"frames": 10}, "chunks": [], "splits": []}
    r = spl.apply_op(p, {"op": "set_object", "zone_spare": 1, "zone_touch_dx": -0.45, "zone_touch_dy": 9, "zone_touch_r": 0.25})
    o = r["object"]
    assert o["zone_spare"] is True and (o["zone_touch_dx"], o["zone_touch_dy"], o["zone_touch_r"]) == (-0.45, 5.0, 0.25)
    d = sp.object_settings({})
    assert d["zone_spare"] is False and d["zone_touch_r"] == 0.0


def test_zone_anchor_text_is_the_chunks_own_or_a_plain_sentence():
    plan = {"target": "woman in a purple top", "object": {"zone_dx": -1.2, "brief": "a man with green hair sitting next to her."}}
    auto = sp.zone_anchor_text(plan)
    assert "image 2" in auto and "as described: a man with green hair sitting next to her," in auto
    assert "left of" not in auto                                                               # the zone never says where
    assert "natural place in the room" in sp.zone_anchor_text({"target": "woman", "object": {"zone_dx": 1.0}})
    assert sp.zone_anchor_text(plan, {"anchor_text": "  Put a hat on him.  "}) == "Put a hat on him."
    assert "the person" in sp.zone_anchor_text({})


def test_set_anchor_text_and_anchor_picture_ops():
    p = {"source": {"frames": 10}, "chunks": [{"id": "c1"}], "splits": []}
    spl.apply_op(p, {"op": "set_anchor_text", "chunk": "c1", "text": " a bird on his shoulder "})
    assert p["chunks"][0]["anchor_text"] == "a bird on his shoulder"
    spl.apply_op(p, {"op": "set_anchor_text", "chunk": "c1", "text": ""})
    assert p["chunks"][0]["anchor_text"] == ""
    r = spl.apply_op(p, {"op": "set_object", "anchor": "picture", "anchor_picture": " mine.png "})
    assert r["object"]["anchor"] == "picture" and r["object"]["anchor_picture"] == "mine.png"
    assert sp.object_settings({"object": {}})["anchor_picture"] == ""


def _zone_render(job, **obj):  # noqa: F811
    from test_swap_nodes import _run
    import folder_paths
    folder_paths.get_output_directory = lambda: str(job["dir"].parent.parent)
    folder_paths.get_temp_directory = lambda: str(job["dir"].parent.parent)
    _masked_job(job, **obj)
    c = sp.load_plan(job["plan"])["chunks"][1]
    return c, lambda: _run(job, "render", chunk=c["id"], seed=3)["result"]


def test_planner_anchor_outputs_for_qwen_picture_and_composite(job, tmp_path):  # noqa: F811
    import cv2
    old = sp.CROP_MIN
    sp.CROP_MIN = 16
    try:
        c, go = _zone_render(job)
        sp.update_plan(job["plan"], lambda q: q["chunks"][1].update(anchor_text="Draw him in."))
        out = go()
        assert out[26] == "Draw him in." and out[28] is True and tuple(out[27].shape) == (1, 64, 64, 3)   # qwen: the text, no image
        sp.update_plan(job["plan"], lambda q: q["chunks"][1].update(anchor_text=""))
        assert "image 2" in go()[26]
        # your own anchor picture, fitted over the crop
        pic = tmp_path / "anchor.png"
        cv2.imwrite(str(pic), np.full((20, 30, 3), (10, 200, 30), np.uint8))
        sp.update_plan(job["plan"], lambda q: q["object"].update(anchor="picture", anchor_picture=str(pic)))
        out = go()
        h, w = out[1].shape[1:3]
        assert out[28] is False and tuple(out[27].shape) == (1, h, w, 3)
        assert abs(float(out[27][0, 0, 0, 1]) - 200 / 255) < 0.02                          # the picture's green (BGR file -> RGB)
        sp.update_plan(job["plan"], lambda q: q["object"].update(anchor_picture=""))
        with pytest.raises(spl.PlannerError, match="anchor picture is not set"):
            go()
        # the composite anchor: the cut-out placed on the anchor frame; needs the cut-out
        sp.update_plan(job["plan"], lambda q: q["object"].update(anchor="composite"))
        with pytest.raises(spl.PlannerError, match="needs Picture 1's cut-out"):
            go()
        (job["dir"] / "object").mkdir()
        cv2.imwrite(str(job["dir"] / "object" / "c.png"), np.full((20, 10, 3), 250, np.uint8))
        cv2.imwrite(str(job["dir"] / "object" / "m.png"), np.full((20, 10), 255, np.uint8))
        sp.update_plan(job["plan"], lambda q: q["object"].update(cutout={"rgb": "object/c.png", "mask": "object/m.png"}))
        out = go()
        assert out[28] is False and tuple(out[27].shape) == (1, h, w, 3)
        assert float((out[27][0] - out[1][0]).abs().max()) > 0.1                              # the cut-out is pasted on frame 0
        assert float((out[27][0] - out[1][0]).abs().mean()) < 0.1                             # and only on it
    finally:
        sp.CROP_MIN = old


def test_zone_mask_shape_is_the_targets_own_silhouette_grown(job):  # noqa: F811
    box = _masked_job(job, shape="mask", zone_grow=3)
    g = sk.ZoneGuide(sp.load_plan(job["plan"]), str(job["dir"]), FR_, (W_, H_), 0, 10, "shoulder")
    m = g.mask(4)
    ys, xs = np.nonzero(m)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (27, 42, 17, 46)         # 30:40 x 20:44, grown 3 px each side
    assert m[35, 30] == 1.0 and m[10, 5] == 0.0
    b = g.box(4)
    assert b[0] <= xs.min() and b[2] > xs.max() and b[1] <= ys.min() and b[3] > ys.max()
    assert sk.ZoneGuide(sp.load_plan(job["plan"]), str(job["dir"]), FR_, (W_, H_), 0, 5, "nothing").mask(2) is None


def test_qwen_anchor_text_helpers():
    import swap_draft as sd
    good = ("Add the man from <image2> into <image1>, standing in front of the couch to the left of the seated woman, a hand resting on "
            "her shoulder. <image1> is the canvas and <image2> is the source of his identity. The woman, the room and the lighting stay "
            "unchanged. He is lit by the same cool blue light as her.")
    assert sd.clean_anchor_text(good) == good
    assert sd.clean_anchor_text('```json\n{"rewritten_prompt": "' + good + '", "wh_ratio": "", "ratio_follow": "<image1>"}\n```') == good
    assert sd.clean_anchor_text(good.replace("<image2>", "image 2")) is None            # Qwen's own tags are required
    assert sd.clean_anchor_text("Add him.") is None and sd.clean_anchor_text("") is None
    plan = {"target": "woman in the black dress", "object": {"action": "add", "guide": "zone", "zone_dx": -1.2, "zone_touch_r": 0.25, "size": 1.9,
                                                              "brief": "a man with green hair sitting next to her"},
            "chunks": [{"id": "c1"}]}
    q = sd.qwen_anchor_question(plan, plan["chunks"][0])
    assert "a man with green hair sitting next to her" in q and "<image1>" in q and "{" not in q
    assert "left of" not in q and "near shoulder" not in q and "large in the frame" not in q       # nothing comes from the zone
    plan["object"]["brief"] = ""
    assert "natural place in the room" in sd.qwen_anchor_question(plan, plan["chunks"][0])


def test_the_drafter_writes_the_qwen_instruction_but_never_over_a_typed_one(job, monkeypatch):  # noqa: F811
    import swap_draft as sd
    _masked_job(job, zone_dx=-1.2, zone_touch_r=0.25, anchor="qwen", anchor_at=0.2, enter_at=0.0)
    good = ("Add the man from <image2> into <image1>, standing to the left of the seated shoulder. <image1> is the canvas and <image2> is "
            "the source of his identity. The room and the lighting stay unchanged. He is lit by the same cool light as the scene.")

    class FakeLarge:
        def __init__(self): self.anchor_calls = 0
        def load(self): pass
        def close(self): pass
        def ask_parts(self, parts, max_tokens=1024, think=False, max_side=1280):
            head = parts[0]
            if "image-editing model" in head:
                self.anchor_calls += 1
                return good
            if "main object" in head or "Look at" in head:
                return "Male, wild green hair, burgundy suit"
            return "video_1: A dim room with a woman.\nplacement: left of her, standing"
    sheet = np.zeros((1, 64, 64, 3), np.float32)
    import torch
    sheet = torch.from_numpy(sheet)
    pp = job["plan"]
    plan = sp.load_plan(pp)
    fq = FakeLarge()
    rep = {"stages": {}, "chunks": {}}
    sd.run_object_draft(pp, plan, [c for c in plan["chunks"]], fq, sheet, rep, [], free_models=False, jd=str(job["dir"]))
    p2 = sp.load_plan(pp)
    assert fq.anchor_calls == len(plan["chunks"]) and all(c["anchor_text"] == good for c in p2["chunks"])
    typed = "Add my own thing from <image2> into <image1>."
    sp.update_plan(pp, lambda q: q["chunks"][0].__setitem__("anchor_text", typed))
    rep2 = {"stages": {}, "chunks": {}}
    sd.run_object_draft(pp, sp.load_plan(pp), [c for c in plan["chunks"]], FakeLarge(), sheet, rep2, [], free_models=False, jd=str(job["dir"]))
    assert sp.load_plan(pp)["chunks"][0]["anchor_text"] == typed                          # a typed instruction is never replaced


def test_zone_preview_gives_the_zone_in_source_pixels(job):  # noqa: F811
    _masked_job(job, zone_dx=-1.0, zone_dy=0.5, size=0.8, zone_aspect=0.9, zone_touch_r=0.5, zone_touch_dx=-0.3, zone_touch_dy=0.4)
    out_dir = str(job["dir"].parent.parent)
    d = spl.do_zone_preview("t", 5, out_dir)
    assert d["found"] and d["w"] == W_ and d["h"] == H_ and abs(d["fw"] - 24) < 1
    ax, ay, aw = d["anchor"]
    z = d["zone"]
    fw = d["fw"]
    s = d["settings"]
    assert abs(z["cx"] - (ax - 1.0 * fw)) < 1e-6                                              # dx is in target widths from the anchor
    assert abs(z["ry"] - (0.8 * fw / 2 + s["zone_grow"])) < 1e-6 and abs(z["rx"] - (0.8 * fw * 0.9 / 2 + s["zone_grow"])) < 1e-6
    assert abs(d["touch"]["x"] - (ax - 0.3 * fw)) < 1e-6 and abs(d["touch"]["r"] - 0.5 * fw) < 1e-6
    sp.update_plan(job["plan"], lambda q: q["object"].update(zone_dx=0.0, zone_shape="mask"))     # settings are read fresh; the mask shape has no geometry
    d2 = spl.do_zone_preview("t", 5, out_dir)
    assert d2["zone"] == {"shape": "mask"}
    with pytest.raises(spl.PlannerError):
        spl.do_zone_preview("t", 9999, out_dir)


def test_zone_locks_hold_from_their_frame_until_the_next(job):  # noqa: F811
    _masked_job(job, zone_dx=-1.0, zone_dy=0.5, size=0.8, zone_aspect=0.9)
    out_dir = str(job["dir"].parent.parent)
    base = spl.do_zone_preview("t", 5, out_dir)
    assert base["key"] is None and base["keys"] == []
    plan = sp.load_plan(job["plan"])
    keys = [{"frame": 8, "zone_dx": 0.5, "size": 0.8, "zone_aspect": 0.9}, {"frame": 3, "zone_dx": -2.0, "zone_dy": 0.1}, {"frame": 99999, "zone_dx": 9}]
    res = spl.apply_op(plan, {"op": "set_object", "zone_keys": keys})
    sp.update_plan(job["plan"], lambda q: q.update(object=plan["object"]))
    z = sp.load_plan(job["plan"])["object"]["zone_keys"]
    assert [k["frame"] for k in z] == [3, 8, 99999] and z[2]["zone_dx"] == 5.0                  # sorted, clamped
    d_before, d5, d9 = (spl.do_zone_preview("t", f, out_dir) for f in (2, 5, 9))
    assert d_before["key"] is None and d_before["settings"]["zone_dx"] == -1.0                    # before the first lock: the job's numbers
    assert d5["key"] == 3 and d5["settings"]["zone_dx"] == -2.0
    assert d9["key"] == 8 and d9["settings"]["zone_dx"] == 0.5
    assert abs(d9["zone"]["cx"] - (d9["anchor"][0] + 0.5 * d9["fw"])) < 1e-6                      # the zone itself follows the lock
    assert sp.zone_at(sp.object_settings(sp.load_plan(job["plan"])), 4)["zone_dx"] == -2.0


def test_the_brief_drives_both_prompts_and_not_the_zone(job):  # noqa: F811
    import swap_draft as sd
    import torch
    _masked_job(job, zone_dx=-1.2, zone_touch_r=0.25, anchor="qwen", anchor_at=0.2, enter_at=0.0, brief="a man with green hair sitting next to her")
    seen = []

    class FakeLarge:
        def load(self): pass
        def close(self): pass
        def ask_parts(self, parts, max_tokens=1024, think=False, max_side=1280):
            head = parts[0]
            seen.append(head)
            if "image-editing model" in head:
                return ("Add the man from <image2> into <image1>, sitting on the couch beside the woman. <image1> is the canvas and <image2> is the source "
                        "of his identity. The room and the woman stay unchanged. He is lit by the same cool light as the scene, with soft shadows.")
            if "Look at" in head:
                return "Male, wild green hair, burgundy suit"
            return ("video_1: A dim room with a woman.\nplacement: she sits centre.\npeople: a woman in black on the right\n"
                    "addition: The new subject sits on the couch to the left of the woman and turns toward her.")
    pp = job["plan"]
    plan = sp.load_plan(pp)
    rep = {"stages": {}, "chunks": {}}
    sd.run_object_draft(pp, plan, list(plan["chunks"]), FakeLarge(), torch.zeros((1, 64, 64, 3)), rep, [], free_models=False, jd=str(job["dir"]))
    c = sp.load_plan(pp)["chunks"][0]
    assert "a man with green hair sitting next to her" in " ".join(seen)                    # the brief reached the sheet, scene and anchor questions
    assert "four lines" in next(h for h in seen if "frames sampled" in h)
    text = c.get("draft") or c["prompt"]                                                      # the chunk had a prompt: the draft waits beside it
    assert "<Subject 2> is added to the scene" in text and "sits on the couch to the left of the woman" in text
    assert "resting on" not in text                        # not the "bird on a shoulder" wording


def test_the_drafter_shows_the_model_the_person_alone_to_read_the_light(job):  # noqa: F811
    import swap_draft as sd
    import torch
    _masked_job(job, anchor="qwen", anchor_at=0.2, brief="a man next to her")
    calls = []

    class FakeLarge:
        def load(self): pass
        def close(self): pass
        def ask_parts(self, parts, max_tokens=1024, think=False, max_side=1280):
            calls.append(parts)
            if "image-editing model" in parts[0]:
                return ("Add the man from <image2> into <image1>, beside her. <image1> is the canvas and <image2> is the source of his identity. "
                        "The people are lit by cool blue light, so he takes a pale cool tint on skin and clothes. The room stays unchanged.")
            if "Look at" in parts[0]:
                return "Male, wild green hair, burgundy suit"
            return "video_1: A dim room.\nplacement: she sits.\npeople: none\naddition: The new subject sits to the left of her."
    pp = job["plan"]
    plan = sp.load_plan(pp)
    rep = {"stages": {}, "chunks": {}}
    sd.run_object_draft(pp, plan, list(plan["chunks"]), FakeLarge(), torch.zeros((1, 64, 64, 3)), rep, [], free_models=False, jd=str(job["dir"]))
    anchor = [p for p in calls if "image-editing model" in p[0]]
    assert anchor and "<image3>" in anchor[0][0]                                   # the question says what the third picture is
    pics = [x for x in anchor[0][1:] if not isinstance(x, str)]
    assert len(pics) == 3                                                           # the frame, the sheet, the person alone
    crop = pics[2]
    assert crop.shape[0] < H_ and crop.shape[1] <= W_ and crop.shape[2] == 3       # a close crop, not the whole frame
    assert abs(float(crop.min()) - 0.0) < 1.0 and bool((crop == 0.5).any())        # outside the mask is grey
    plan2 = {"target": "x", "object": {"brief": "b"}, "chunks": [{"id": "c1"}]}
    assert "A third picture" not in sd.qwen_anchor_question(plan2, plan2["chunks"][0]) and "A third picture" in sd.qwen_anchor_question(plan2, plan2["chunks"][0], with_crop=True)
