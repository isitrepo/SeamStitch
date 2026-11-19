"""Object jobs (swap_plan.MODE_OBJECT): the source everywhere, the take inside the target's mask."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swap_draft as sd  # noqa: E402
import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
from test_swap import H_, W_, _add_take, _clip, _frames, _write, job  # noqa: F401,E402  (job is a fixture)


def test_object_settings_defaults_and_the_add_grow():
    assert sp.object_settings({}) == sp.OBJECT_DEFAULTS
    assert sp.object_settings({"object": {"action": "add"}})["grow"] == sp.OBJECT_ADD_GROW
    assert sp.object_settings({"object": {"action": "add", "grow": 20}})["grow"] == 20
    assert sp.object_settings({"mode": "object", "object": {"junk": 1}}) == sp.OBJECT_DEFAULTS


def test_object_mode_is_a_job_mode_that_never_inverts():
    p = {"mode": "object", "guide": "backdrop"}
    assert sp.job_mode(p) == sp.MODE_OBJECT and not sp.chunk_invert(p, {})
    assert sp.chunk_invert({"mode": "character", "guide": "backdrop"}, {})


def test_set_object_op():
    p = {"source": {"frames": 10}, "chunks": [], "splits": []}
    r = spl.apply_op(p, {"op": "set_object", "action": "add", "feather": 500})
    assert r["object"]["action"] == "add" and r["object"]["grow"] == sp.OBJECT_ADD_GROW and r["object"]["feather"] == 100
    with pytest.raises(sp.PlanError):
        spl.apply_op(p, {"op": "set_object", "action": "remove"})
    spl.apply_op(p, {"op": "set_mode", "mode": "object"})
    assert sp.job_mode(p) == "object"


def test_object_assembly_is_the_source_but_the_take_inside_the_mask(job):  # noqa: F811
    p = sp.load_plan(job["plan"])
    r = [c["render"] for c in p["chunks"]]
    for i, base in enumerate((60, 70, 80)):
        _add_take(job, i, _clip(r[i][1] - r[i][0] + 1, base, seed=5 + i), r[i])
    box = np.zeros((H_, W_, 3), np.uint8)
    box[10:26, 20:36] = 255
    (job["dir"] / "masks" / "m001").mkdir(parents=True)
    _write(str(job["dir"] / "masks" / "m001" / "mask.mkv"), [box] * job["N"])

    def upd(q):
        q["mode"] = "object"
        q["target"] = "painting"
        for c in q["chunks"]:
            for t in c["takes"]:
                t.setdefault("settings", {})["options"] = {"mode": "object"}
        q["object"] = {"grow": 0, "feather": 0, "level": "none"}
        q["masks"] = [{"id": "m001", "range": [0, job["N"] - 1], "frames": job["N"], "file": "masks/m001/mask.mkv",
                       "target": "painting", "created": sp.now(), "size": [W_, H_]}]
    sp.update_plan(job["plan"], upd)
    # what the takes alone assemble to (joins and all): the same plan in character mode
    plain_plan = sp.load_plan(job["plan"])
    plain_plan["mode"] = "character"
    for c in plain_plan["chunks"]:
        for t in c["takes"]:
            t["settings"]["options"]["mode"] = "character"
    plain = _frames(job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit", plan=plain_plan, write_plan=False)["file"])
    rep = job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit")
    out = _frames(rep["file"])
    src = np.stack(job["src"])
    inside = box[..., 0] > 0
    assert np.array_equal(out[:, ~inside], src[:, ~inside])        # outside the mask: the source, every frame
    assert np.array_equal(out[:, inside], plain[:, inside])        # inside: the takes
    assert rep["object"]["frames_with_object"] == job["N"] and rep["checks"]["frames_ok"]


def test_object_draft_helpers():
    assert sd.the_target("painting on the wall") == "the painting on the wall"
    assert sd.the_target("his left shoulder") == "his left shoulder"
    v, pl = sd.parse_object_scene("<think>hm</think>\n**video_1:** A dim room.\nplacement: top centre, still")
    assert (v, pl) == ("A dim room.", "top centre, still")
    assert sd.parse_object_scene("nothing useful") == ("", "")
    t = sd.fill_object("small blue bird", "A man sits.", "his shoulder is on the left", "his left shoulder", "add")
    assert "<Subject 1> is the small blue bird shown in <Picture 1>." in t and "added on his left shoulder" in t
    assert "His shoulder is on the left." in t and "attribute_transfer" in t
    r = sd.fill_object("car photo", "", "", "painting", "replace")
    assert "replaces the painting" in r and "  " not in r
    assert [ln.rstrip(":") for ln in r.splitlines() if ln.endswith(":")] == [
        "subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape",
        "non_diegetic_music"]


def test_the_sheet_question_says_what_the_thing_stands_in_for():
    q = sd.object_details_question("black shoes", "replace")
    assert "take the place of the black shoes" in q and "Start with what it is" in q and "{" not in q
    assert sd.object_details_question("his left shoulder", "add") == sd.OBJECT_DETAILS_INSTRUCTION
    assert sd.object_details_question("", "replace") == sd.OBJECT_DETAILS_INSTRUCTION


def test_the_thing_being_replaced_is_not_declared_a_kept_person():
    people = ["a person in a red jacket with white cuffs in the centre-right",
              "a bare leg ending in a black shoe at the bottom"]
    kept, dropped = sd.object_people(people, "black shoes")
    assert kept == [people[0]] and dropped == [people[1]]
    kept, dropped = sd.object_people(["a man in a red jacket on the left", "a bare hand on his knee"],
                                     "his left shoulder", replace=False)       # an add: the person named in the target stays
    assert kept == ["a man in a red jacket on the left"] and len(dropped) == 1
    assert sd.object_people(["a man in a red jacket"], "the red jacket's pocket", replace=False)[0] == ["a man in a red jacket"]
    assert sd.object_people([], "black shoes") == ([], [])
    assert "Leave out the black shoes" in sd.OBJECT_SCENE_INSTRUCTION.format(n=4, secs=3.0, the_target="the black shoes")


def test_a_picture_target_asks_for_the_sheet_as_a_picture():
    assert sd.OBJECT_PICTURE_TARGET.search("painting on the wall") and sd.OBJECT_PICTURE_TARGET.search("the TV")
    assert not sd.OBJECT_PICTURE_TARGET.search("his left shoulder") and not sd.OBJECT_PICTURE_TARGET.search("white shoe")


def test_an_object_render_is_cropped_to_its_target_and_the_take_pastes_it_back(job):  # noqa: F811
    import torch
    import swap_take as stk
    from test_swap_nodes import _run, nodes  # noqa: F401

    import folder_paths
    folder_paths.get_output_directory = lambda: str(job["dir"].parent.parent)
    folder_paths.get_temp_directory = lambda: str(job["dir"].parent.parent)
    old = sp.CROP_MIN, sp.CROP_PAD
    sp.CROP_MIN, sp.CROP_PAD = 16, 0.25                      # the test frame is only 64 x 48
    box = np.zeros((H_, W_, 3), np.uint8)
    box[20:28, 30:38] = 255                                 # an 8 x 8 target
    (job["dir"] / "masks" / "m001").mkdir(parents=True)
    _write(str(job["dir"] / "masks" / "m001" / "mask.mkv"), [box] * job["N"])

    def upd(q):
        q["mode"] = "object"
        q["target"] = "painting"
        q["masks"] = [{"id": "m001", "range": [0, job["N"] - 1], "frames": job["N"], "file": "masks/m001/mask.mkv",
                       "target": "painting", "created": sp.now(), "size": [W_, H_]}]
        for c in q["chunks"]:
            c["prompt"] = "x"
    sp.update_plan(job["plan"], upd)
    p = sp.load_plan(job["plan"])
    c = p["chunks"][1]
    out = _run(job, "render", chunk=c["id"], seed=3)["result"]
    desc, images, w, h, smask = out[0], out[1], out[8], out[9], out[17]
    x0, y0, x1, y1 = desc["crop"]
    assert x0 <= 30 and y0 <= 20 and x1 >= 38 and y1 >= 28 and (x1 - x0, y1 - y0) == (w, h) == tuple(images.shape[2:0:-1])
    assert desc["pins"] == {"start": None, "end": None}
    assert tuple(smask.shape[1:]) == (h, w)                 # the cached mask, cropped the same way
    # the take: the render pasted into the source frame at the crop, the source everywhere else
    render = torch.full((desc["length"], h, w, 3), 1.0)
    take, tdir, _rep = stk.save_take(desc, render, score=False, mouth_sync=False, review_clip=False)
    assert take["crop"] == [x0, y0, x1, y1]
    got = _frames(os.path.join(str(job["dir"]), take["file"]))
    r0 = c["render"][0]
    for i in (0, len(got) - 1):
        src = job["src"][r0 + i]
        assert (got[i][y0:y1, x0:x1] >= 250).all()
        outside = np.ones((H_, W_), bool)
        outside[y0:y1, x0:x1] = False
        assert np.array_equal(got[i][outside], src[outside])
    assert (x1 - x0, y1 - y0) == (16, 16)
    sp.CROP_MIN, sp.CROP_PAD = old


def test_an_unlabelled_two_line_scene_answer_is_read_in_order():
    assert sd.parse_object_scene("A dim room, two people.\nThe strip along the top, still.") == (
        "A dim room, two people.", "The strip along the top, still.")
    assert sd.parse_object_scene("one line only") == ("", "")


def test_shrink_keeps_the_shape():
    import torch
    out = sd._shrink(torch.zeros((2, 100, 400, 3)), 200)
    assert tuple(out.shape) == (2, 50, 200, 3)
    assert sd._shrink(torch.zeros((1, 10, 20, 3)), 200).shape[1:3] == (10, 20)


def test_fit_cover_fills_the_box_and_keeps_the_shape():
    import swap_keep as sk
    pic = np.zeros((100, 200, 3), np.uint8)
    pic[:, :100] = 255                                      # left half white
    out = sk.fit_cover(pic, 50, 50)                         # a square box: the middle of the picture
    assert out.shape == (50, 50, 3) and out[:, :20].mean() > 200 and out[:, 30:].mean() < 50


def test_picture_fill_assembles_the_picture_inside_the_mask_in_the_targets_light(job, tmp_path):  # noqa: F811
    import cv2
    pic = np.zeros((40, 60, 3), np.uint8)
    pic[:, :, 1] = np.linspace(0, 255, 60, dtype=np.uint8)[None, :]
    cv2.imwrite(str(tmp_path / "pic.png"), pic)
    box = np.zeros((H_, W_, 3), np.uint8)
    box[8:24, 16:48] = 255
    (job["dir"] / "masks" / "m001").mkdir(parents=True)
    _write(str(job["dir"] / "masks" / "m001" / "mask.mkv"), [box] * job["N"])

    def upd(q):
        q["mode"] = "object"
        q["target"] = "painting"
        q["object"] = {"fill": "picture", "picture": str(tmp_path / "pic.png")}
        q["masks"] = [{"id": "m001", "range": [0, job["N"] - 1], "frames": job["N"], "file": "masks/m001/mask.mkv",
                       "target": "painting", "created": sp.now(), "size": [W_, H_]}]
    sp.update_plan(job["plan"], upd)
    rep = job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit")         # no takes at all: nothing is rendered
    out = _frames(rep["file"])
    src = np.stack(job["src"])
    grown = cv2.dilate((box[..., 0] > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.array_equal(out[:, ~grown], src[:, ~grown])               # the rest: the source
    inner = out[:, 10:22, 18:46].astype(np.float32)
    assert inner[..., 1][:, :, -4:].mean() - inner[..., 1][:, :, :4].mean() > 20   # the picture's ramp is there
    import swap_keep as sk
    assert abs(sk.luma(inner).mean() - sk.luma(src[:, 10:22, 18:46]).mean()) < 6   # in the target's light (luma)
    assert rep["object"]["shot_boxes"] == [(16, 8, 48, 24)] and rep["object"]["frames_with_object"] == job["N"]


def test_set_object_fill_and_picture():
    p = {"source": {"frames": 10}, "chunks": [], "splits": []}
    r = spl.apply_op(p, {"op": "set_object", "fill": "picture", "picture": " car.jpg "})
    assert (r["object"]["fill"], r["object"]["picture"]) == ("picture", "car.jpg")
    with pytest.raises(sp.PlanError):
        spl.apply_op(p, {"op": "set_object", "fill": "paint"})


def test_anchors_follow_the_targets_piece_and_hold_over_a_stray_jump():
    import swap_keep as sk
    frames = {}
    for f in range(10):
        m = np.zeros((48, 64), np.uint8)
        m[20:30, 10 + f:20 + f] = 255                       # the shoulder, moving right 1 px a frame
        if f >= 5:
            m[2:6, 50:62] = 255                             # a stray patch appears (bigger than nothing, far away)
        if f == 7:
            m[20:30, :] = 0                                 # the shoulder lost on one frame: only the stray patch
        frames[f] = m
    a = sk.anchors(frames, (64, 48))
    assert set(a) == set(range(10))
    assert all(abs(a[f][1] - 20) < 1e-6 for f in range(10))          # held on the shoulder's top, never the patch's
    assert a[9][0] > a[0][0] and all(a[f][0] < 40 for f in range(10))


def test_placement_and_paste_of_a_cutout():
    import swap_keep as sk
    cut = np.full((20, 10, 3), 200, np.uint8)
    a = np.ones((20, 10), np.float32)
    x0, y0, s = sk.placement(a.shape, (32.0, 30.0, 10.0), 10.0, {"size": 2.0, "sink": 0.0})
    assert s == 1.0 and (x0, y0) == (27.0, 10.0)                       # 20 px tall = 2 widths, feet on the anchor
    frame = np.zeros((48, 64, 3), np.uint8)
    out, al = sk.place_cutout(frame, cut, a, (x0, y0, s), tone=100.0)
    assert out[15, 30, 0] == 100 and out[5, 30, 0] == 0 and al.sum() == 200     # toned to luma 100, only its box


def test_object_composite_needs_add_render_and_a_cutout():
    p = {"mode": "object", "object": {"action": "add", "cutout": {"rgb": "object/c.png", "mask": "object/m.png"}}}
    assert sp.object_composite(p)
    assert not sp.object_composite(dict(p, object=dict(p["object"], action="replace")))
    assert not sp.object_composite(dict(p, object=dict(p["object"], fill="picture")))
    assert not sp.object_composite(dict(p, mode="character"))
    assert not sp.object_composite({"mode": "object", "object": {"action": "add"}})


def test_object_add_guides_composite_pins_and_take(job, tmp_path):  # noqa: F811
    import cv2
    import torch
    from test_swap_nodes import _run
    import folder_paths
    folder_paths.get_output_directory = lambda: str(job["dir"].parent.parent)
    folder_paths.get_temp_directory = lambda: str(job["dir"].parent.parent)
    old = sp.CROP_MIN
    sp.CROP_MIN = 16
    box = np.zeros((H_, W_, 3), np.uint8)
    box[30:40, 20:44] = 255                                 # the shoulder
    (job["dir"] / "masks" / "m001").mkdir(parents=True)
    _write(str(job["dir"] / "masks" / "m001" / "mask.mkv"), [box] * job["N"])
    (job["dir"] / "object").mkdir()
    cv2.imwrite(str(job["dir"] / "object" / "c.png"), np.full((20, 10, 3), 200, np.uint8))
    cv2.imwrite(str(job["dir"] / "object" / "m.png"), np.full((20, 10), 255, np.uint8))

    def upd(q):
        q["mode"], q["target"] = "object", "shoulder"
        q["object"] = {"action": "add", "cutout": {"rgb": "object/c.png", "mask": "object/m.png"}, "size": 1.0,
                       "shadow": 0.5, "colour": 0.5, "soften": 0.8, "grain": True}
        q["masks"] = [{"id": "m001", "range": [0, job["N"] - 1], "frames": job["N"], "file": "masks/m001/mask.mkv",
                       "target": "shoulder", "created": sp.now(), "size": [W_, H_]}]
        for c in q["chunks"]:
            c["prompt"] = "x"
    sp.update_plan(job["plan"], upd)
    try:
        c = sp.load_plan(job["plan"])["chunks"][1]
        out = _run(job, "render", chunk=c["id"], seed=3)["result"]
        assert out[21] is True and out[0]["composite_frames"] == c["render"][1] - c["render"][0] + 1
        # pins: the source is the guide, the composite pins both ends, the refine group stays off
        sp.update_plan(job["plan"], lambda q: q["object"].update(guide="pins"))
        out = _run(job, "render", chunk=c["id"], seed=3)["result"]
        desc, sp_, ep_ = out[0], out[6], out[7]
        assert out[21] is False and sp_.shape[0] >= 1 and ep_.shape[0] >= 1
        assert desc["pins"]["start"]["object_composite"] and desc["pins"]["end"]["frames"][1] == c["render"][1]
        # take: a take of this chunk refined again
        import swap_take as stk
        d2 = _run(job, "render", chunk=c["id"], seed=3)["result"][0]
        take, _d, _r = stk.save_take(d2, torch.full((d2["length"], out[9], out[8], 3), 0.5), score=False,
                                     mouth_sync=False, review_clip=False)
        sp.update_plan(job["plan"], lambda q: q["object"].update(guide="take", from_take=take["id"]))
        out = _run(job, "render", chunk=c["id"], seed=3)["result"]
        assert out[21] is True
        sp.update_plan(job["plan"], lambda q: q["object"].update(from_take="c9-t001"))
        with pytest.raises(sp.PlanError):
            _run(job, "render", chunk=c["id"], seed=3)
    finally:
        sp.CROP_MIN = old


def test_the_plain_add_prompt_declares_the_people_kept_and_the_object_next():
    raw = ("video_1: A dim room.\nplacement: his left shoulder, right of frame.\n"
           "people: a man in a red jacket in the centre; a person with a skull face on the left")
    v, p, people = sd.parse_object_scene(raw, with_people=True)
    assert people == ["a man in a red jacket in the centre", "a person with a skull face on the left"]
    assert sd.parse_object_scene("video_1: x\nplacement: y\npeople: none", with_people=True)[2] == []
    t = sd.fill_object("blue jay", v, p, "shoulder of the man in the red jacket", "add", people=people)
    assert "<Subject 1> is a man in a red jacket in the centre in <Video 1>, kept exactly as filmed." in t
    assert "<Subject 3> is the blue jay shown in <Picture 1>." in t
    assert "<Subject 1> (appears throughout): fully_preserved" in t and "<Subject 3> (appears" in t
    assert "<Subject 1>, <Subject 2>, the background, the camera and the action remain as recorded" in t
    assert "<Subject 1> and <Subject 2> stay exactly as in <Video 1>" in t
    assert [ln.rstrip(":") for ln in t.splitlines() if ln.endswith(":")] == [
        "subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape",
        "non_diegetic_music"]
    # the object from text alone: no picture, no retention line for it
    t2 = sd.fill_object("a blue jay", v, p, "shoulder of the man", "add", people=people[:1], picture=False)
    assert "<Subject 2> is a blue jay." in t2 and "<Picture 1>" not in t2 and "[video editing + audio reuse]" in t2
    assert "<Subject 2> (appears" not in t2


def test_paste_frame_takes_the_whole_take(job):  # noqa: F811
    p = sp.load_plan(job["plan"])
    r = [c["render"] for c in p["chunks"]]
    for i, base in enumerate((60, 70, 80)):
        _add_take(job, i, _clip(r[i][1] - r[i][0] + 1, base, seed=5 + i), r[i])
    plain = _frames(job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit")["file"])

    def upd(q):
        q["mode"], q["target"] = "object", "shoulder"
        q["object"] = {"action": "add", "paste": "frame"}
        for c in q["chunks"]:
            for t in c["takes"]:
                t.setdefault("settings", {})["options"] = {"mode": "object"}
    sp.update_plan(job["plan"], upd)
    out = _frames(job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit")["file"])
    assert np.array_equal(out, plain)


def test_placeholder_guide_marks_a_grey_silhouette(job):  # noqa: F811
    import cv2
    from test_swap_nodes import _run
    import folder_paths
    folder_paths.get_output_directory = lambda: str(job["dir"].parent.parent)
    folder_paths.get_temp_directory = lambda: str(job["dir"].parent.parent)
    box = np.zeros((H_, W_, 3), np.uint8)
    box[30:40, 20:44] = 255
    (job["dir"] / "masks" / "m001").mkdir(parents=True)
    _write(str(job["dir"] / "masks" / "m001" / "mask.mkv"), [box] * job["N"])
    (job["dir"] / "object").mkdir()
    cv2.imwrite(str(job["dir"] / "object" / "c.png"), np.full((20, 10, 3), 200, np.uint8))
    cv2.imwrite(str(job["dir"] / "object" / "m.png"), np.full((20, 10), 255, np.uint8))

    def upd(q):
        q["mode"], q["target"] = "object", "shoulder"
        q["object"] = {"action": "add", "cutout": {"rgb": "object/c.png", "mask": "object/m.png"}, "size": 1.0,
                       "guide": "placeholder", "crop": False, "paste": "frame"}
        q["masks"] = [{"id": "m001", "range": [0, job["N"] - 1], "frames": job["N"], "file": "masks/m001/mask.mkv",
                       "target": "shoulder", "created": sp.now(), "size": [W_, H_]}]
        for c in q["chunks"]:
            c["prompt"] = "x"
    sp.update_plan(job["plan"], upd)
    c = sp.load_plan(job["plan"])["chunks"][1]
    out = _run(job, "render", chunk=c["id"], seed=3)["result"]
    images, smask, has = out[1], out[17], out[18]
    assert has is True and out[21] is False and smask.shape[0] == images.shape[0]
    on = smask[0] > 0.5
    assert on.any() and abs(float(images[0][on].mean()) - 128 / 255) < 0.02     # the marked region is the grey placeholder


def test_a_target_that_matches_no_cached_mask_is_flagged_not_silent(job):  # noqa: F811
    """Q4 fault 2 (2026-11-17): "black shoe" against a mask marked "black shoes" gave the source back with no word."""
    p = sp.load_plan(job["plan"])
    r = [c["render"] for c in p["chunks"]]
    for i, base in enumerate((60, 70, 80)):
        _add_take(job, i, _clip(r[i][1] - r[i][0] + 1, base, seed=5 + i), r[i])
    box = np.zeros((H_, W_, 3), np.uint8)
    box[10:26, 20:36] = 255
    (job["dir"] / "masks" / "m001").mkdir(parents=True)
    _write(str(job["dir"] / "masks" / "m001" / "mask.mkv"), [box] * job["N"])

    def upd(q):
        q["mode"] = "object"
        q["target"] = "painting"
        for c in q["chunks"]:
            for t in c["takes"]:
                t.setdefault("settings", {})["options"] = {"mode": "object"}
        q["object"] = {"grow": 0, "feather": 0, "level": "none"}
        q["masks"] = [{"id": "m001", "range": [0, job["N"] - 1], "frames": job["N"], "file": "masks/m001/mask.mkv",
                       "target": "painting", "created": sp.now(), "size": [W_, H_]}]
    sp.update_plan(job["plan"], upd)
    ok = job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit", write_plan=False)
    assert not [f for f in ok["flags"] if f["code"] == "object_no_mask"]                    # an exact match: no flag
    sp.update_plan(job["plan"], lambda q: q.update(target="paintings"))                      # one letter off
    rep = job["sa"].assemble(job["plan"], fmt="test/ffv1-8bit", write_plan=False)
    hit = [f for f in rep["flags"] if f["code"] == "object_no_mask"]
    assert len(hit) == 1 and "'paintings'" in hit[0]["text"] and "'painting'" in hit[0]["text"]
    assert hit[0]["text"] in job["sa"]._summary(rep)                                         # shown in the node's report
    assert rep["object"]["frames_with_object"] == 0                                          # and it is the source, as before