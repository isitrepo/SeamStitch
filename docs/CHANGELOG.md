# Changelog

## 2026-09-16 — Flatten + rename for side-by-side install

Flattened the `seamstitch/` package to the repo root (ComfyUI only loads
`custom_nodes/<pack>/__init__.py`, so the nested layout could never load) and
renamed every node ID, route, JS extension name, DOM widget name, and log
prefix so this pack can be installed alongside the three packs it replaces
without route or node-ID collisions:

| old | new |
|---|---|
| `SeamweaveSimpleCombine` | `SeamStitchCombine` |
| `LoadVideoUIFirstLast` | `SeamStitchLoader` |
| `VideoSegmentRecombine` | `SeamStitchRecombine` |
| `/seamweave_combine_*` | `/seamstitch/combine/*` |
| `/load_video_ui_fl_*` | `/seamstitch/loader/*` |
| `Comfy.SeamweaveSimpleCombine` | `SeamStitch.Combine` |
| `Comfy.LoadVideoUIFirstLast` | `SeamStitch.Loader` |

Also: replaced the guarded-import `print()` with a `logging` warning, and
neutralized the `Comfyui-VideoSegmentRecombine`-specific ImportError text.

### Gate: side-by-side install (sandboxed, `--cpu`, isolated `--base-directory`/`--user-directory`)

**Variant B** — `SeamStitch` + copies of the three old live packs
(`comfy_seamweave`, `Comfyui-LoadVideoUI`, `Comfyui-VideoSegmentRecombine`) +
VHS: server started with no `RuntimeError` and no `IMPORT FAILED`.
`/object_info` contained all six node IDs — `SeamStitchLoader`,
`SeamStitchCombine`, `SeamStitchRecombine` (category `SeamStitch`) plus
`SeamweaveSimpleCombine`, `LoadVideoUIFirstLast`, `VideoSegmentRecombine`.

**Variant C** — `SeamStitch` + VHS only: same clean startup; the three new
nodes registered.

**Signature diff** (`input`/`output_name`/`output` from `/object_info`,
variant C's new nodes vs. variant B's old nodes):

```
SeamStitchCombine vs SeamweaveSimpleCombine: IDENTICAL
SeamStitchLoader vs LoadVideoUIFirstLast: IDENTICAL
SeamStitchRecombine vs VideoSegmentRecombine: IDENTICAL
```

### Incident during the gate

The first variant-B launch renamed the **live** ComfyUI install's
`user/comfyui.db` to `comfyui.db.bak` despite `--base-directory`/
`--user-directory` pointing at the sandbox: ComfyUI's legacy-database
migration (`app/database/db.py:87`, `copy_legacy_default_db`) resolves its
source path from `comfy/cli_args.py:268`
(`os.path.join(os.path.dirname(__file__), "..", "user", "comfyui.db")`) —
relative to the real install, not the CLI overrides — and only skips the
migration if a database file already exists at the *target* user directory.
The live instance wasn't running at the time, so nothing held the file lock
to refuse it (previously it had only been attempted and refused, per the
2026-09-16 review). The live `comfyui.db` was restored immediately
(renamed back from `.bak`, verified identical size/mtime to before). Fixed
for all later sandbox launches in this session by pre-creating an empty
`comfyui.db` file in each sandbox's `--user-directory` before startup, which
satisfies the migration's existing-file guard and prevents it from touching
the live path at all.
