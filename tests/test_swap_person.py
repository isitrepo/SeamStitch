"""SeamStitch Swap, B8: following (F) and mouth sync (M) read the replaced person when other people share
the shot. A sitcom run (several people, shot / reverse-shot) scored F on whoever SAM3 found first ("person",
max 1 object: the man in the foreground) and M on the first face mediapipe returned; here, synthetic
two-person shots with a cut, the output track on the wrong person, and her back to the camera."""
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import swap_plan as sp  # noqa: E402
import swap_planner as spl  # noqa: E402
import swap_scores as ss  # noqa: E402
import swap_take as st  # noqa: E402

H, W = 36, 64


def _box(y0, y1, x0, x1, h=H, w=W):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


LEFT, RIGHT = _box(6, 36, 4, 24), _box(4, 36, 40, 62)


def _two_shots(n=40, cut=20):
    """Shot 1: she stands left, a man right. A cut at 20: she is right, the man left. SAM3 tracked everyone:
    object 0 = the man all through (it followed him across the cut), object 1 = her in shot 1 (lost at the
    cut), object 2 = her from the cut on (picked up as a new person). Her source mask moves with her."""
    objs = np.zeros((n, 3, H, W), bool)
    src = []
    for i in range(n):
        if i < cut:
            objs[i, 0], objs[i, 1] = RIGHT, LEFT
            src.append(LEFT)
        else:
            objs[i, 0], objs[i, 2] = LEFT, RIGHT
            src.append(RIGHT)
    return objs, src


def test_the_old_output_track_on_the_wrong_person_scores_zero():
    objs, src = _two_shots()
    # what "person", max 1 gave on a sitcom scene: the first person found (the man), all through
    assert ss.pose_iou(src, list(objs[:, 0]))["pose_iou"] < 0.05


def test_pick_person_keeps_her_in_every_shot_across_a_cut():
    objs, src = _two_shots()
    hers, picks = ss.pick_person(objs, src, cuts=[20])
    assert [p["object"] for p in picks] == [1, 2]
    assert [p["shot"] for p in picks] == [[0, 20], [20, 40]]
    pi = ss.pose_iou(src, list(hers))
    assert pi["pose_iou"] == 1.0 and pi["frames"] == 40


def test_pick_person_first_frame_only_would_lose_her_at_the_cut():
    """The likeliest failure (B7, the source track): one pick for the whole take."""
    objs, src = _two_shots()
    hers, _ = ss.pick_person(objs, src, cuts=[])
    assert ss.pose_iou(src, list(hers))["pose_iou"] <= 0.5     # one object can't be her on both sides


def test_a_shot_without_her_gets_no_mask_and_no_score_not_zero():
    objs, src = _two_shots()
    src = src[:20] + [np.zeros((H, W), bool)] * 20             # she isn't in shot 2 (the reverse shot)
    hers, picks = ss.pick_person(objs, src, cuts=[20])
    assert picks[1]["object"] is None
    assert not hers[20:].any()
    pi = ss.pose_iou(src, list(hers))
    assert pi["frames"] == 20 and pi["pose_iou"] == 1.0


def test_nobody_overlapping_her_scores_low_not_skipped():
    """A take where she went somewhere else: the shot's pick is nobody, and F reads 0 for those frames."""
    objs = np.zeros((10, 1, H, W), bool)
    objs[:, 0] = RIGHT
    src = [LEFT] * 10
    hers, picks = ss.pick_person(objs, src)
    assert picks[0]["object"] is None
    assert ss.pose_iou(src, list(hers))["pose_iou"] == 0.0


def test_one_person_is_unchanged():
    """One person, one object: the pick is that object, so F is what it was."""
    objs = np.zeros((12, 1, H, W), bool)
    objs[:, 0] = _box(6, 36, 6, 26)
    src = [LEFT] * 12
    hers, _ = ss.pick_person(objs, src)
    assert ss.pose_iou(src, list(hers)) == ss.pose_iou(src, list(objs[:, 0]))


def test_unpack_bits_reads_sam3s_packed_masks():
    m = np.random.default_rng(1).random((3, 2, 8, 16)) > 0.5
    # comfy.ldm.sam3.tracker.pack_masks: bit i of each byte = pixel i of 8
    packed = (m.reshape(3, 2, 8, 2, 8) * (1 << np.arange(8))).sum(-1).astype(np.uint8)
    assert (ss.unpack_bits(packed) == m).all()


def _pack(m):
    t = torch.from_numpy(m)
    shifts = torch.arange(8)
    return (t.view(*t.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


def test_the_node_picks_her_per_shot_from_the_plans_cuts(tmp_path):
    p = sp.new_plan("j", {"path": "x.mp4", "frames": 200, "fps": 24, "width": 1280, "height": 720, "audio": True})
    for f in (120, 150):            # 150 is outside the render: ignored
        spl.apply_op(p, {"op": "add_cut", "frame": f, "from": "detected"})
    pf = sp.plan_path(str(tmp_path))
    sp.save_plan(pf, p)
    objs, src = _two_shots()        # render 100-139: the cut at 120 is frame 20
    desc = {"format": spl.CHUNK_FORMAT, "plan": pf, "chunk": "c1", "render": [100, 139]}
    track = {"packed_masks": _pack(objs), "n_frames": 40, "orig_size": (72, 128)}
    mask, rep = st.output_person(desc, track, torch.from_numpy(np.stack(src).astype(np.float32)))
    assert tuple(mask.shape) == (40, 72, 128)
    assert rep["cuts"] == [20] and [x["object"] for x in rep["picks"]] == [1, 2]
    got = [m.numpy() > 0.5 for m in mask]
    assert ss.pose_iou(src, got)["pose_iou"] > 0.95
    node_mask, node_rep = st.SeamStitchSwapOutputPerson().pick(desc, track,
                                                                torch.from_numpy(np.stack(src).astype(np.float32)))
    assert json.loads(node_rep)["objects"] == 3 and torch.equal(node_mask, mask)


def test_the_node_with_nobody_tracked_gives_an_empty_mask(tmp_path):
    p = sp.new_plan("j", {"path": "x.mp4", "frames": 200, "fps": 24, "width": 1280, "height": 720, "audio": True})
    pf = sp.plan_path(str(tmp_path))
    sp.save_plan(pf, p)
    desc = {"format": spl.CHUNK_FORMAT, "plan": pf, "chunk": "c1", "render": [0, 9]}
    mask, rep = st.output_person(desc, {"packed_masks": None, "n_frames": 10, "orig_size": (72, 128)},
                                 torch.zeros((10, 36, 64)))
    assert tuple(mask.shape) == (10, 72, 128) and float(mask.max()) == 0.0 and rep["objects"] == 0


def test_held_frames_take_the_last_source_mask(tmp_path):
    p = sp.new_plan("j", {"path": "x.mp4", "frames": 200, "fps": 24, "width": 1280, "height": 720, "audio": True})
    pf = sp.plan_path(str(tmp_path))
    sp.save_plan(pf, p)
    objs = np.zeros((12, 1, H, W), bool)
    objs[:, 0] = LEFT
    desc = {"format": spl.CHUNK_FORMAT, "plan": pf, "chunk": "c1", "render": [0, 9]}     # 10 + 2 held
    src = torch.from_numpy(np.stack([LEFT] * 10).astype(np.float32))
    mask, rep = st.output_person(desc, {"packed_masks": _pack(objs), "n_frames": 12, "orig_size": (H, W)}, src)
    assert tuple(mask.shape) == (12, H, W) and bool((mask[10:] > 0.5).any())


# ---------------------------------------------------------------- M: her face

class _P:
    def __init__(self, x, y):
        self.x, self.y = x, y


def _face(cx, opening):
    """478 landmarks around (cx, 0.4); lips 13 / 14 apart by `opening` of the 78-308 width."""
    pts = [_P(cx, 0.4) for _ in range(478)]
    wid = 0.05
    pts[78], pts[308] = _P(cx - wid / 2, 0.5), _P(cx + wid / 2, 0.5)
    gap = opening * wid * ss.AN_W / ss.AN_H
    pts[13], pts[14] = _P(cx, 0.5 - gap / 2), _P(cx, 0.5 + gap / 2)
    return pts


@pytest.fixture
def fake_mediapipe(monkeypatch):
    """A scripted FaceLandmarker: each frame's top-left pixel says which script row to return; mediapipe
    returns the man's face first, as it did on the sitcom."""
    script = {}

    class _LM:
        def __init__(self, opts):
            self.n = opts.num_faces

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def detect_for_video(self, img, ts):
            row = script[int(img.data[0, 0, 0]) + 256 * int(img.data[0, 0, 1])]
            return types.SimpleNamespace(face_landmarks=row[:self.n])

    vision = types.SimpleNamespace(
        FaceLandmarkerOptions=lambda base_options, running_mode, num_faces: types.SimpleNamespace(num_faces=num_faces),
        RunningMode=types.SimpleNamespace(VIDEO=0),
        FaceLandmarker=types.SimpleNamespace(create_from_options=lambda o: _LM(o)))
    python = types.SimpleNamespace(BaseOptions=lambda model_asset_path: None, vision=vision)
    mp = types.ModuleType("mediapipe")
    mp.Image = lambda image_format, data: types.SimpleNamespace(data=data)
    mp.ImageFormat = types.SimpleNamespace(SRGB=0)
    tasks = types.ModuleType("mediapipe.tasks")
    tasks.python = python
    pymod = types.ModuleType("mediapipe.tasks.python")
    pymod.BaseOptions, pymod.vision = python.BaseOptions, vision
    mp.tasks = tasks
    for k, v in {"mediapipe": mp, "mediapipe.tasks": tasks, "mediapipe.tasks.python": pymod}.items():
        monkeypatch.setitem(sys.modules, k, v)
    return script


def _frames(ids):
    out = []
    for i in ids:
        f = np.zeros((ss.AN_H, ss.AN_W, 3), np.uint8)
        f[0, 0, 0], f[0, 0, 1] = i % 256, i // 256
        out.append(f)
    return out


def test_mouth_reads_her_face_not_the_first_one_found(fake_mediapipe):
    n = 30
    rng = np.random.default_rng(3)
    hers = list(0.2 + 0.3 * rng.random(n))
    his = list(0.2 + 0.3 * rng.random(n))
    for t in range(n):
        fake_mediapipe[t] = [_face(0.75, his[t]), _face(0.25, hers[t])]           # the source
        fake_mediapipe[1000 + t] = [_face(0.75, his[t]), _face(0.25, hers[t] * 0.9 + 0.02)]   # the take: he's untouched
    left = [_box(0, 36, 0, 32)] * n           # her mask, left half
    s_old, _ = ss.score_frames(_frames(range(1000, 1000 + n)), _frames(range(n)), mouth=True, model_path="x")
    assert s_old["mouth"] > 0.99              # his mouth against his own: the old, meaningless green
    s_new, _ = ss.score_frames(_frames(range(1000, 1000 + n)), _frames(range(n)), left, left, mouth=True,
                               model_path="x")
    assert s_new["mouth"] > 0.99 and s_new["mouth_info"]["face_of"] == "target"
    # and it is hers: break her take's mouth and the score drops while his stays the same
    for t in range(n):
        fake_mediapipe[1000 + t] = [_face(0.75, his[t]), _face(0.25, 0.2 + 0.3 * rng.random())]
    s_bad, _ = ss.score_frames(_frames(range(1000, 1000 + n)), _frames(range(n)), left, left, mouth=True,
                               model_path="x")
    assert s_bad["mouth"] < 0.6


def test_mouth_is_not_measured_when_her_back_is_to_the_camera(fake_mediapipe):
    n = 30
    rng = np.random.default_rng(4)
    for t in range(n):
        o = 0.2 + 0.3 * rng.random()
        fake_mediapipe[t] = [_face(0.75, o)]                 # only his face: she faces away
        fake_mediapipe[1000 + t] = [_face(0.75, o)]
    left = [_box(0, 36, 0, 32)] * n
    s, _ = ss.score_frames(_frames(range(1000, 1000 + n)), _frames(range(n)), left, left, mouth=True, model_path="x")
    assert s["mouth"] is None and "not measured" in s["mouth_info"]["why"]
    assert ss.mouth_flag(s["mouth"]) is None


def test_the_mouth_cache_is_keyed_by_her_mask(tmp_path):
    a = st._mouth_cache(str(tmp_path), 10, 20, [LEFT] * 3)
    b = st._mouth_cache(str(tmp_path), 10, 20, [RIGHT] * 3)
    assert a != b and a != st._mouth_cache(str(tmp_path), 10, 20) and "mouth_her_" in a


# ---------------------------------------------------------------- the example workflow and old graphs

def test_the_example_workflow_scores_the_picked_person():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_workflows",
                     "Swap - Character Replace.json")
    with open(p, encoding="utf-8") as f:
        w = json.load(f)
    nodes = {n["id"]: n for n in w["nodes"]}
    links = {l[0]: l for l in w["links"]}
    take = next(n for n in w["nodes"] if n["type"] == "SeamStitchSwapTake")
    pick = nodes[links[next(i for i in take["inputs"] if i["name"] == "output_mask")["link"]][1]]
    assert pick["type"] == "SeamStitchSwapOutputPerson"
    up = {i["name"]: nodes[links[i["link"]][1]] for i in pick["inputs"]}
    assert up["chunk"]["type"] == "SeamStitchSwapPlanner"
    assert up["track_data"]["type"] == "SAM3_VideoTrack" and up["track_data"]["widgets_values"][1] == 0   # everyone
    # the same source mask Swap Take gets
    assert up["source_mask"]["id"] == nodes[links[next(i for i in take["inputs"] if i["name"] == "source_mask")
                                                     ["link"]][1]]["id"]
    # the output track looks for any person (the target's text drives the source tracks only)
    track_in = {i["name"]: nodes[links[i["link"]][1]] for i in up["track_data"]["inputs"] if i.get("link")}
    assert track_in["conditioning"]["widgets_values"] == ["person"]


def test_an_old_graph_without_the_picker_is_flagged():
    old = {"5": {"class_type": "SAM3_TrackToMask", "inputs": {}},
           "9": {"class_type": "SeamStitchSwapTake", "inputs": {"output_mask": ["5", 0]}}}
    new = {"5": {"class_type": "SeamStitchSwapOutputPerson", "inputs": {}},
           "9": {"class_type": "SeamStitchSwapTake", "inputs": {"output_mask": ["5", 0]}}}
    assert st.output_mask_is_picked(old) is False
    assert st.output_mask_is_picked(new) is True and st.output_mask_is_picked(None) is True
