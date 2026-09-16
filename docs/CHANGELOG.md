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

## 2026-09-16 — Close the file-path holes, fix the recombine format default

Findings 4 and 13 from the pre-release review.

- **View route** (`loader.py:24`, `/seamstitch/loader/view`): kept the
  absolute-path fast path (`js/loader.js`'s "choose file to upload" skips the
  upload entirely and points this route straight at a desktop path), but
  narrowed what it will serve — rejects an empty filename, rejects any `..`
  path segment, and requires the path to end in a known video extension,
  before checking the file actually exists. A non-existent, non-video, or
  traversal-bearing path now gets a plain 404, never a stack trace.
- **Upload routes** (`loader.py:/seamstitch/loader/upload_chunk`,
  `combine.py:/seamstitch/combine/upload_chunk`): the client-supplied
  filename is reduced to `os.path.basename` and rejected outright (HTTP 400)
  if empty, `.`/`..`, or still containing a `/` or `\` after that — so it can
  no longer be joined into a path that escapes the input directory. Both
  packs' copies of the route got the same fix independently, per the
  review's "public **and live**" scope note.
- **`SeamStitchRecombine.format`** (`recombine.py:393`): added
  `'default': 'video/h264-mp4'` to the combo widget so the frontend no
  longer silently defaults to whatever the first entry in VHS's format list
  happens to be (`video/16bit-png` on stock VHS 1.7.9) — matching what the
  node's own diagram has always shown.

### Gate (sandboxed, `--cpu`, isolated `--base-directory`/`--user-directory`, `SeamStitch` + VHS only)

1. `upload_chunk` with `filename=../evil.txt` on both the loader and combine
   routes → HTTP 400, no file appeared outside the sandbox's `input/`
   directory.
2. `/seamstitch/loader/view`: an absolute path to a real non-video file
   (`C:\Windows\win.ini`) → 404; a `..`-traversal path with a `.mp4` suffix
   tacked on → 404; a real absolute `.mp4` path → 200.
3. `/object_info` → `SeamStitchRecombine.input.required.format[1].default ==
   'video/h264-mp4'`.
4. End-to-end: queued a three-node prompt (`SeamStitchCombine` concatenating
   two clips, `SeamStitchLoader` loading a clip with audio dubbed on via
   `ffmpeg` since the sample clips are silent, `SeamStitchRecombine` splicing
   the loader's own frames back into the loader's own source at the loader's
   own frame range) via `/prompt` — completed with `status: success`, and
   `gate_s2_recombined_00001-audio.mp4` (640x360, video+audio, 2s) landed in
   the sandbox's `output/` directory.

**Second instance of the comfyui.db incident**: the first sandbox launch for
this gate hit the exact failure mode documented above again — same root
cause, same live-file rename, same immediate restore (verified identical
size/mtime). This time fixed by passing `--database-url
sqlite:///<scratch>/user/comfyui.db` explicitly, which short-circuits
`copy_legacy_default_db` at its very first check (`args.database_url is not
None`) instead of relying on the existing-file guard further down — a more
direct fix than the pre-touch approach from the prior session. **Either
guard must be applied on every future sandbox launch** — neither is the
recipe's default behavior, and omitting both reproduces the live-file rename
whenever the live ComfyUI instance isn't running to hold the lock.
