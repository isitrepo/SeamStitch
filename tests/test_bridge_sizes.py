"""The Timeline's bridge sizes (cut and join menus): each lands the generator's length exactly on its grid.

24 + 24 frames at context 5 is 58 for MiniMax, rounded up to 73: the bridge came back 63 frames long for a 48-frame
range and played x1.3 slow (join 422, 2026-10-14). The menu now offers only sizes that need no rounding."""
import json
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _choices(tmp_path, cases):
    node = shutil.which("node")
    if not node:
        pytest.skip("node isn't installed")
    src = open(os.path.join(ROOT, "js", "timeline.js"), encoding="utf-8").read()
    a = src.index("const GRID_LTX")
    head = src[a:src.index("\n", a)]
    body = src[src.index("const BRIDGE_SECONDS"):src.index("function hideWidget(")]
    js = tmp_path / "b.js"
    js.write_text(head + "\n" + body + "\nconst c = JSON.parse(process.argv[2]);"
                  "\nconsole.log(JSON.stringify(c.map(([g, k, e, fr]) => bridgeChoices(g, k, e, fr))));", encoding="utf-8")
    out = subprocess.run([node, str(js), json.dumps(cases)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


MM, LTX, NONE = "minimax (17k+5)", "ltx (8k+1)", "none"


def test_minimax_sizes_at_context_5(tmp_path):
    (r,) = _choices(tmp_path, [[MM, 5, 0, 25]])
    assert [b["total"] for b in r] == [12, 29, 46, 63, 80]
    assert [(b["before"], b["after"]) for b in r][2] == (23, 23)          # 23 each side, as on the test join
    assert all((b["gen"] - 5) % 17 == 0 and b["gen"] == b["total"] + 10 for b in r)
    assert r[0]["before"] == 6                                           # 209's 6 + 6 still offered


def test_ltx_sizes_land_on_8k_plus_1(tmp_path):
    (r,) = _choices(tmp_path, [[LTX, 2, 0, 25]])
    assert all((b["gen"] - 1) % 8 == 0 and b["gen"] >= 9 for b in r)
    assert all(b["before"] - b["after"] in (0, 1) for b in r)          # odd totals: the extra frame before the cut


def test_extend_frames_count_too(tmp_path):
    (r,) = _choices(tmp_path, [[MM, 5, 3, 25]])
    assert all((b["total"] + 10 + 3 - 5) % 17 == 0 for b in r)


def test_no_grid_offers_even_sizes_at_the_target_times(tmp_path):
    (r,) = _choices(tmp_path, [[NONE, 5, 0, 24]])
    assert [b["total"] for b in r] == [12, 24, 36, 48, 60, 72]
    assert not any(b["exact"] for b in r)


def test_the_menus_use_the_sizes():
    src = open(os.path.join(ROOT, "js", "timeline.js"), encoding="utf-8").read()
    assert 'bridgeItems("cut"' in src and 'bridgeItems("join"' in src
    assert "each * 2" not in src
