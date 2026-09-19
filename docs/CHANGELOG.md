# Changelog

## Unreleased — the Loader keeps a restored trim range instead of re-probing it

`js/loader.js` only; no Python, no node surface, no output change.

**Fix — `onConfigure` threw away a range the workflow supplied.** A graph carrying
`start_frame = 232`, `end_frame = 265`, `display_mode = "frames"` came back on the canvas as
`0` / `497` — the whole clip. The reset was real but the trigger was narrower than S5 recorded:
**it is not every saved graph, it is a graph whose range lives in the frame widgets while
`start_time`/`end_time` sit at 0.** That is exactly what the `docs/comfy_templates` API templates
carry, and exactly what `loader.py` reads back — with `display_mode == "frames"` the backend uses
`start_frame`/`end_frame` and ignores the seconds pair entirely (`loader.py:499`). `onConfigure`
called `syncFramesFromTime()` unconditionally, i.e. it treated seconds as the source of truth in
both modes, so the restored `232`/`265` were overwritten with `round(0 * 48) = 0`; the file probe
in `videoPreview.onloadedmetadata` then saw `end_time === 0`, read that as "no range set" and
filled in the whole clip. The same JSON therefore meant 232–265 to the queue and 0–497 on the
canvas.

`onConfigure` now calls a new `node.syncRestoredRange()`, which syncs **from** whichever pair
`display_mode` says the graph authored: frames mode syncs seconds from frames, seconds mode is
unchanged. An unset range is still left at 0 for the probe to fill, and the probe still supplies
the timeline's own bounds — a range that overruns the clip is still clamped to the probed length.
Insert mode's `mode` / `join_frame` / `trim_each_side` restore path is untouched.

**Fix — `syncTimeFromFrames` wrote the converted duration into the wrong widget.** Pre-existing
since `v0.1.0` (line 213 there, inherited from the upstream loader): the duration line read
`duration_frames`, divided it by the frame rate and assigned the result **back into**
`duration_frames` instead of into `duration`, so the frame count was destroyed and `duration` never
updated. Measured on a fresh node: `duration_frames = 99` at 48 fps became `2.063`. It was mostly
invisible because replace mode's `updateUI` recomputes both from `end − start` whenever a video is
loaded, and because until this session `syncTimeFromFrames` never ran on reload — but insert mode's
`updateUI` returns early, so it ate the bridge length. Assigning to `durationWidget` (which the
line's own `if (durationWidget && durationFramesWidget)` guard already named) fixes it, and is
required for the `onConfigure` fix above to be correct.

### Gate (sandbox ComfyUI, port 8293, `--cpu`, isolated `--base-directory` / `--user-directory` /
`--database-url`, only SeamStitch whitelisted, serving the dev copy; live 8188 untouched)

Driven in a real browser against the real page on `s5_ins_combined_00001.mp4` (497 frames at 48 fps,
the S5 clip), by `app.loadGraphData()` on a graph serialized from the canvas and then mutated to
each shape, reading the widget values back 3.5 s after the load:

| saved graph | before | after |
|---|---|---|
| replace, range in frames only (232/265), times 0 — the API-template shape | `0` / `497`, `duration_frames` 497 | **`232` / `265`**, `duration_frames` 33 |
| insert, `join_frame` 248, `trim_each_side` 0, bridge `duration_frames` 33 | bridge **0**, join 248 | **bridge 33**, join 248 |
| insert, `trim_each_side` 5 | bridge **0** | **bridge 33**, join 248, N 5 |
| no range set at all, frames mode | `0` / `497` | `0` / `497` (probe still fills it) |
| no range set at all, seconds mode | `0` / `497` | `0` / `497` (probe still fills it) |
| range in seconds only (4.833–5.521) | `232` / `265` | `232` / `265` (unchanged) |
| full current-version canvas save, all 26 values, frames mode | `232` / `265` | `232` / `265` (unchanged) |
| frames range overrunning the clip (400/900) | `400` / `497` | `400` / `497` (still clamped to the probe) |

Interaction, after a reload that restored 232/265: the canvas shows `start_frame` 232,
`end_frame` 265, `duration_frames` 33, the ruler runs 0–497 and the selection sits just left of the
249 tick. A real mouse drag of the start handle moved it to frame 126 with `end_frame` held at 265
and `duration_frames` recomputed to 139. The Time/Frames toggle round-trips both ways leaving
`100` / `150` / `50` intact — before the second fix, Frames→Time wrote `50 / 48 = 1.042` into
`duration_frames`.

Console: no SeamStitch error. The only 404s are core ComfyUI's for user files a fresh sandbox does
not have (`user.css`, `api/userdata/user.css`, `comfy.templates.json`, the workflow and subgraph
listings) plus one "graph accessed before initialization" from the test harness polling `window.app`
early — the same set S5 recorded. `node --check js/loader.js` passes.

**Supersedes** the "pre-existing caveat" under "S5 — insert mode on real footage", section 1, in
the Stitch 2.0 working repo's `REAL_FOOTAGE_FINDINGS.md`: the reset is fixed, and it never applied
to a plain canvas save whose two pairs agreed.

## Unreleased — insert mode proven on real footage; one off-by-one fixed

**Verdict: works with a caveat.** Insert mode does what it says — nothing removed, the frames
either side of the bridge bit-identical to their sources, sound locked to the picture within one
sample — and it is the right tool for **extending a shot**. It is the wrong tool for **joining two
clips**: on the same hard cut, same seed and same morph prompt, `insert at join` measured max
per-frame change **60.09** (std 10.36) against `replace range`'s **14.93** (std 3.19). Given two
adjacent frames across a hard cut, the model idles on clip A for ~29 frames then cuts to clip B in
two — the cut back again. Use `replace range` for a Combine seam. Full numbers, strips and method:
`REAL_FOOTAGE_FINDINGS.md`, "S5 — insert mode on real footage" (in the Stitch 2.0 working repo).

**Fix — `loader.py`, `_load_insert`: the insert anchors were off by one on a Combine output.**
`SeamStitchCombine` measures `seam_frame` in decode order and `SeamStitchRecombine` cuts on the
same convention, but the replace-mode path `_load_insert` borrows to decode the anchors maps
`start_frame` to an absolute presentation time as a bare `start_frame / fr`. Combine's own output
starts at **+31.006 ms** (AAC-priming empty edit, 1.49 frames at 48 fps), so asking for frame 248
returned decode frame 246 and *both* anchors came back as clip A's last frame: the model was asked
to morph a frame into itself, rendered a 33-frame freeze, and the cut reappeared where the frozen
bridge met clip B. Anchors are now fetched by time off the stream's real start, with the same
thousandth-of-a-frame tolerance `recombine._decode_range` uses. **Replace mode is not touched** —
the change is entirely inside `_load_insert`, and the `input_video` path needs none of it because
`_load_from_tensor` slices by index. Only a stream that does not start at t = 0 was affected, which
in practice means a Combine output.

**Fix — the four pre-insert API templates validate against `/prompt` again.** `seamstitch_combine`
was missing `resize_to` / `resize_fit` and the three `seamstitch_ltx_first_last*` templates were
missing `snap_to_multiple`, `extend_bridge`, `extend_amount`, `extend_unit`, `bridge_frame_grid`,
`mode`, `join_frame` and `trim_each_side` — widgets added since they were exported, which `/prompt`
rejects as `required_input_missing`. Each is now present at the default the node itself declares,
read from the live `object_info`; nothing else in the four files changed. All seven templates report
OK, and the addition is a proven no-op for the render, not just for validation: the fixed
`..._morph.api.json` on the same input, seed 424242 and range 232–264 measures max **14.93**, mean
6.78, std 3.19, and its decoded video stream is bit-identical to the earlier replace-mode render
(md5 `b2549d4f83c50ec2dcbb9fa2ff78c660`).

Measured 2026-09-19 on Kay's live instance (ComfyUI 0.36.0, RTX 5090) and with `python_embeded`:

- `tests/` 33 passed. New `test_insert_anchors_are_decode_order_on_an_offset_stream` remuxes
  `Test vids/4.mp4` with the same +381-tick start offset Combine produces and compares the anchors
  against a plain decode-order walk; it **fails on the pre-fix loader** (max per-pixel difference
  0.863) and passes on the fixed one. The existing anchor test could not catch this: it decoded its
  reference with the same absolute-time rule the loader used, on a clip starting at t = 0.
- **Frame math** (acceptance check 2), insert at seam 248, `trim_each_side = 0`, 33-frame bridge:
  output **527** frames = 496 input + 33 − 2 anchors − 0 dedup. Nothing removed from either clip.
- **Anchors** (check 3): on the pre-encode spliced tensor, output[247] is **bit-identical** to
  `4.mp4[247]` and output[279] to `2.mp4[0]` (max abs difference 0). In the delivered h264 mp4 the
  same frames measure 1.64 and 2.03 — encode loss, not a splice error.
- **Audio** (check 4), sync error = audio lag − picture shift, cross-correlation at corr 1.000:
  silence fallback **+0.01 / +0.02 ms**, bridge audio **+0.01 / +0.01 ms**, single-video insert
  **+0.00 / +0.01 ms**. One sample at 32 kHz is 31 µs. With `audio_mode = original` the inserted
  stretch measures **−92.37 dBFS** — genuinely silent, not source audio pulled forward.
- **`seam_frame`** (check 5): **248** on the stream-copy join (output[247]/[248] bit-identical to
  A's last and B's first frame) and **248** on a mixed-frame-rate transcode join (2.mp4 retimed to
  30 fps; differences 1.56 and 0.74 against a tolerance of 6.0). A deliberately wrong candidate
  raises "Refusing to emit an unverified seam_frame". A wrong but in-range `seam_frame` wired into
  the loader is trusted by design; out of range it fails loudly and names the clip length.
- **Backward compatibility** (check 1): `SeamStitchCombine` writes a **byte-identical** join to the
  2026-09-16 run (md5 `70d2943f735669ca611443cf59d86a68`), and the replace-mode morph render
  reproduces the recorded prompt-table numbers — max **14.93**, mean 6.78, std 3.19 against
  14.9 / 6.79 / 3.10, same seed 424242, same range. A save carrying only v0.1.0's 18 widget values
  loads as `mode = "replace range"`.
- **The single-video insert works**: 31 frames at a typed `join_frame` of `4.mp4`, continuity
  prompt, max per-frame change **4.65** / std **0.68** across the inserted span against 6–8 for the
  footage either side, both anchors bit-identical, output 279 = 248 + 31.

**Not verified.**

- The end-to-end run **through the fixed loader** on a Combine output. The live ComfyUI holds the
  pack's modules from before the fix and re-imports only on restart, which this session did not do.
  Every insert render above was made on the pre-fix build with the off-by-one cancelled at the
  graph level (`join_frame = 249`, and Recombine's `start_frame` / `end_frame` typed 248 / 247),
  after confirming both anchors were bit-identical to `4.mp4[247]` and `2.mp4[0]` — so the model
  saw the frames the fixed loader now returns. Re-run
  `seamstitch_ltx_first_last_morph_insert.api.json` unmodified after a restart to close this.
- **Whether the bridge audio sounds right.** Not listened to. Objectively it is level-matched
  (−48.79 dBFS between source at −45.44 and −49.33), mostly low rumble (71.4% of its energy below
  200 Hz, centroid 1231 Hz, peak 0.0080 against 0.0444 just before), and there is no click at
  either join. Whether it suits the shot is still a listening judgement.
- `trim_each_side > 0` in a render.

**Known, not fixed.**

- Loading any saved graph on the canvas resets the loader's `start_frame` / `end_frame` to the
  whole clip. Pre-existing — v0.1.0's `onConfigure` already called `syncFramesFromTime()` — and it
  does not affect API payloads.

## Unreleased — Loader timeline: the join marker (insert mode UI)

`js/loader.js` only. Driven from the `mode` widget callback, `onConfigure` and `onConnectionsChange`,
extending the existing `updateUI` / pointer path (no second timeline). Replace mode is untouched.
In `insert at join`: the two trim handles and the start/end widgets are hidden and replaced by one
draggable join marker (snaps to whole frames, clamped so both anchors stay inside the clip) synced
two-way with `join_frame`; `trim_each_side` and the editable bridge-length `duration` /
`duration_frames` are shown (duration is no longer overwritten by a trim and its trim-clamping hook
is bypassed); with N > 0 the removed range `[join-N, join+N-1]` is shaded; the two kept anchor
frames (`join-N-1`, `join+N`) are drawn as thumbnails in the preview corners; a prompt hint says
morph prompt for a Combine seam, continuity prompt for a single-video insert. When `seam_frame` is
wired the marker is locked (drag ignored, `not-allowed` cursor), recoloured purple and labelled
"join N - from Combine"; unwiring restores it. First switch into insert mode with `join_frame` 0
starts the marker at the clip middle.

Measured 2026-09-19 in a sandbox ComfyUI (port 8291, `--cpu`, only SeamStitch whitelisted) on
`Test vids/4.mp4` (124 frames, 24 fps), driving the real page:

- Replace mode: start/end handles and fill shown, join marker/shade/label/hint hidden; widget list
  as before plus `join_frame`/`trim_each_side` hidden.
- A real mouse drag of the marker from frame 62 to 25% of the bar set `join_frame` to 30; typing
  `join_frame` 90 put the marker at 72.5806% (90/124 = 72.58%); N = 5 shaded left 68.5484% / width
  8.06452% (expected 68.55 / 8.06) with anchors labelled frame 84 and 95.
- Wiring a `SeamStitchCombine.seam_frame` output: label "join 90 - from Combine", colour
  `rgb(167,139,250)`, cursor `not-allowed`, a synthetic pointerdown left `join_frame` unchanged;
  disconnecting restored `join 90`, `rgb(245,158,11)`, `pointer`.
- `serialize()` -> `loadGraphData()` restored mode, join 90, N 5, hidden widgets, the locked label
  and colour, the shade and both anchor labels.
- Console: no error from SeamStitch. The only errors were core ComfyUI 404s for user files that do
  not exist in a fresh sandbox (`user.css`, `comfy.templates.json`, workflow/subgraph listings) and
  one core "graph accessed before initialization" from the test harness's early import.
- Screenshots: the browser tool returns images to the session but cannot write them to disk, so
  none are saved; the numbers above are the record.
- `node --check js/loader.js js/combine.js` passes.

## Unreleased — Loader insert mode (`mode`, `join_frame`, `trim_each_side`, `seam_frame`, `insert`)

`SeamStitchLoader` gains, all appended last so workflows saved before they exist load unchanged
(`mode` defaults to `replace range`): widgets `mode`, `join_frame`, `trim_each_side`; optional
input `seam_frame` (INT, socket only); output `insert` (BOOLEAN, 15th). The frame arithmetic is
`insert_math.py` (pure Python, no ComfyUI/torch). In `insert at join`: anchors are the kept frames
`join-N-1` and `join+N` (one anchor rule, see the spec), `start_frame = join-N`,
`end_frame = join+N-1`, `images` = the two anchors, `audio` = a 1024-sample stub, `full_clip_audio`
stays real, and the existing `duration` / `duration_frames` widget is read as the wanted bridge
length, snapped to the nearest 8n+1 frames (floor 9, ties up) and emitted as `frame_count` /
`duration`. Anchors are decoded through the ordinary replace-mode path one frame each, so they get
identical crop/resize/snap. A wired `seam_frame` wins over `join_frame` (logged when they differ).
A join with no room for both anchors raises, naming the clip length. Works for `input_video` too.
The JS timeline is not touched here (S4).

Measured 2026-09-19 (`python_embeded`, torch 2.13 / av 17.0.1):

- `tests/test_loader_insert.py`: 15 passed (N=0, N>0, both boundaries, out-of-range errors, 8n+1
  snapping at 24/25/30 fps, wired-vs-typed precedence, and `Test vids/4.mp4` anchors at join 40
  against an independent `av` decode: max per-pixel difference < 3/255).
- **Replace mode regression:** the first 14 outputs, sha256 of every tensor/audio buffer plus all
  scalars, on `Test vids/4.mp4` (seconds range; frames range with `extend_bridge`; pad-resized) and
  on a seeded `input_video`/`input_audio` tensor, are byte-identical between `HEAD` before S3
  (`b3587d1`) and after. Combined comparison files hash to the same sha256 `baba03a1...53c1ea`.
- Sandbox `/object_info` (port 8297): `SeamStitchLoader` outputs are the previous 14 in order plus
  `insert` last; required widgets end `..., bridge_frame_grid, mode, join_frame, trim_each_side`;
  optional inputs `seam_frame, input_video, input_audio`.
- README diagrams regenerated with `docs/gen_diagrams.py` from that `/object_info` (the Combine and
  Recombine SVGs also picked up S1/S2's new socket/widget).

## Unreleased — Recombine can insert a bridge instead of replacing a range (`insert`)

`SeamStitchRecombine` gains an `insert` BOOLEAN input (default false, appended last so workflows
saved before it exists load unchanged). Off, nothing about the node changes. On:

- `end_frame == start_frame - 1` is accepted — an empty range, nothing removed at all. Anything
  below that still raises, naming the bound.
- The regenerated segment's **first and last frame are dropped unconditionally**, before the
  held-duplicate strip, because they are the two kept frames either side of the join that the
  generator was anchored on and that the output still carries as the source's own frames. Fewer
  than three regenerated frames raises rather than silently inserting nothing.
- The replace-mode duration warning ("the original gap was N frames") no longer fires — in insert
  mode `expected_gap` is 0 by design. It is replaced with a line reporting how many frames went
  in, how many came out, and how much longer the video got.
- Audio (`audio_splice.splice_audio(..., insert=True)`): the gap under the inserted frames is
  `bridge_audio` when wired and **silence** otherwise — never source audio read from around the
  join, which belongs to frames that are still in the output and about to play again. A bridge
  whose audio decodes short of its frames is padded with silence for the same reason (replace
  mode still borrows the source's audio there, unchanged). `audio_mode = original` in insert mode
  therefore means silence under the bridge, and says so on the console. Both joins still get the
  length-preserving equal-power crossfade.

Measured 2026-09-19, `tests/test_recombine_insert.py`, synthetic 48-frame / 24 fps / 48 kHz
fixture (lossless x264 yuv444p in mp4, per-frame fingerprints, a 200→8000 Hz chirp so a lag has
one unambiguous correlation peak):

- **Frames**, insert at join 20 with a 35-frame bridge: 48 source + 33 inserted = **81 frames**,
  **0** source frames removed; output frame 19 bit-identical to source frame 19 and output frame
  53 bit-identical to source frame 20, with neither anchor doubled. With `trim_each_side = 2`
  (range 18-21 removed): **77 frames**, and none of source frames 18-21 appears anywhere in the
  output. A 14-frame bridge with two held duplicates: 14 → 12 after the anchor drop → **11** after
  the held strip.
- **Audio**: **162000 samples = 81 frames** exactly, for both the bridge-audio and the silence
  case; drift at both joins **0 samples** (normalised cross-correlation, 250 ms windows,
  corr **1.0000** each side) — the Defect 4 method from `REAL_FOOTAGE_FINDINGS.md`. Outside the
  20 ms crossfades the output is sample-identical to the source either side of the insertion, and
  the stretch under the inserted frames is exactly zero in the silence case and exactly the
  bridge's samples in the bridge case. A bridge audio 1000 samples short is padded with silence.
- **Replace-mode regression**: the same call with `insert=false` on a fixed fixture (range 10-20,
  13-frame bridge with two held duplicates at each end, `audio_mode` `original` and `bridge`)
  produces output identical to the **v0.1.0** tag — picture sha256 `ce3fcf1080ba7b37…` for both
  modes, 48 frames, 96000 audio samples, `torch.equal` on the waveform. v0.1.0 is built from a
  detached `git worktree` at the tag inside the test, not from memory.

Every behaviour above was confirmed to be under test by mutation: reverting the anchor drop, the
insert-mode gap selection, the bridge-shortfall padding, the empty-range check, the anchor offset
carried into the bridge-audio index, the gap warning, or one frame of the replace-mode picture
each makes the suite fail. Note for the record: at `trim_each_side = 0` the pre-insert-mode gap
piece already rendered as silence by accident (its start landed past its own limit); the source
audio leak this fixes bites from `trim_each_side = 1` up, and from the bridge-shortfall fill.

## Unreleased — Combine reports where the join is (`seam_frame`)

`SeamStitchCombine` gains a fourth output, `seam_frame` (INT, appended last so existing wiring is
unaffected): the index of clip B's first frame in the combined file. It is measured, not
computed on trust - candidate = A's duration x the output's real frame rate, then output frames
`seam-1`/`seam` are compared (64x64 thumbnails, mean abs diff) against A's true last and B's true
first frame, trying +-2 neighbouring candidates; if none fit within 6.0 it raises `ValueError`
naming both clips and the measured differences. Measured 2026-09-19 on the true pair (A-last /
B-first): stream copy 4.mp4+2.mp4 0.00 / 0.00, seam 248; transcode (2.mp4 retimed to 24 fps)
0.63 / 0.21, seam 248; transcode with `resize_to=match_a`, 4.mp4+1.mp4 0.65 / 1.19, seam 248.
A pairing one frame off across the cut measured 62.5-64.0. Tests: `tests/test_combine_seam_frame.py`
(stream-copy, mixed-fps transcode, deliberately wrong candidate); `tests/conftest.py` stubs the
ComfyUI-only modules so pytest can import the pack.

## 2026-09-19 — Reconciled two dirty working copies onto `main`

Uncommitted edits had accumulated in the dev copy and the live install. Each hunk, and what it was:

- **Loader "Load Video" refresh button and `/seamstitch/loader/list_files` route removed** - an
  intentional removal (confirmed by Kay), present identically in both copies; the button had been
  in the pack since the initial flatten commit.
- **Loader `extend_bridge` / `extend_amount` / `extend_unit` / `bridge_frame_grid`** - live install
  only; described by the first "Unreleased" entry below.
- **Combine `resize_to` / `resize_fit`** - live install only; described by the second
  "Unreleased" entry below.
- The README changes for both features were also live-install only.

## Unreleased — Loader can ask a bridge generator for more frames than the picked gap

New `SeamStitchLoader` widget, `extend_bridge` (off by default), changes what the existing
`duration`/`frame_count` outputs report - no new output added. Off, they're exactly what they've
always been: the picked gap, matching `images.shape[0]`. On, they instead report the picked gap
**plus** `extend_amount` (`extend_unit`: seconds or frames), snapped to `bridge_frame_grid`.
`images`/`audio`/`first_frame`/`last_frame` are unaffected either way. Wire `frame_count` into a
downstream first/last-frame generator's own length widget (LTX-2.5:
`EmptyLTXVLatentVideo.length` / `LTXVEmptyLatentAudio.frames_number`) to give it more frames than
the original gap had to invent a transition in - the generated segment, and the recombined
video's final duration, then run longer than the picked gap by that amount.

`SeamStitchRecombine` needed no changes for this to work: it already splices in whatever frame
count the regenerated segment comes back with, regardless of how it compares to `end_frame -
start_frame` (only prints an informational warning if they differ a lot) - the actual gap was
just that nothing computed a sane, grid-snapped frame count to ask a generator for in the first
place. `bridge_frame_grid` offers `ltx (8k+1)` (LTX-2.5's temporal downsample factor), `minimax
(17k+5)` (MiniMax H3 Motion Context's own generated-length grid - see
`seamweave/comfy_bridge.py`'s `_grid_length`), or `none` for a backend with no such constraint.
MiniMax H3 separately requires its `context_length` to be one of exactly 5/22/39/56 frames - a
widget on the Motion Context node itself, which `frame_count` has no way to satisfy for you.

(Went through two earlier shapes before this: first a separate `SeamStitchBridgeLength` node
wired between Loader and the generator, then a fifth `bridge_length` output added to Loader
directly - both dropped in favor of reusing `duration`/`frame_count`, which a workflow already
had wired wherever it needed a frame count, so extending needs no new wiring at all.)

## Unreleased — Combine can resize a mismatched clip instead of refusing

`SeamStitchCombine` previously hard-errored whenever clip A and clip B weren't the same
resolution ("no fancy logic here"). New `resize_to` widget: `off` (default) keeps that
behavior; `match_a`/`match_b` resizes the other clip onto the picked one's size (aspect
ratio always kept, never stretched) before concatenating. Any resize forces the existing
transcode fallback path rather than the lossless stream-copy path, since a resized clip is
a fresh encode by definition.

New `resize_fit` widget picks how that resize reconciles the two aspect ratios: `crop`
(default) scales to fill the target frame and crops the overhang - no bars, a sliver off
two edges; `pad` scales to fit inside it and letterboxes the rest with black - every pixel
kept, a visible bar at the seam. `crop` defaults on because the common case (two different
generators' outputs that are both "16:9-ish" but not bit-identical) is a near-miss
mismatch, where a bar is far more visible than the sliver `crop` trims - confirmed on a
real pair (1472x832 vs 1376x768, about a 1.3% aspect-ratio difference): `pad` put a
visible black bar top and bottom on every frame of the resized clip, `crop` filled the
frame completely with a barely-perceptible reframe.

## v0.1.0 — 2026-09-17

First tagged release. Pushed `main` to the public repo
(`https://github.com/isitrepo/SeamStitch`), verified the clean-room install from the README's
manual steps (fresh ComfyUI checkout, fresh venv, `pip install -r` for ComfyUI/SeamStitch/VHS,
`--cpu` start): `/object_info` showed all three `SeamStitchLoader`/`SeamStitchCombine`/
`SeamStitchRecombine` IDs under category `SeamStitch`, no import failures. Tagged `v0.1.0` and
published the GitHub release: https://github.com/isitrepo/SeamStitch/releases/tag/v0.1.0.

## 2026-09-16 — Recombine keeps audio in sync after the dedup drops frames

`SeamStitchRecombine` built the picture as `before + deduped + after` but laid the source audio
down untouched from t=0, so every held frame the dedup dropped shifted the rest of the file
against its own sound — 20.8 ms per frame at 48 fps, silently.

- **Audio is spliced like the picture** (new `audio_splice.py`): cut at the same frame
  boundaries, with a length-preserving equal-power crossfade at each discontinuous join
  (`audio_crossfade_ms`, default 20).
- **`audio_mode`**: `original` (default) keeps the source audio of the surviving frames; `bridge`
  uses the new `bridge_audio` input (e.g. LTX-2.5's own generated audio), resampled, padded from
  the source when short, never stretched.
- **Start-offset fix**: frames map to audio on the file's real video-minus-audio start offset.
  `SeamStitchCombine`'s output delays its video 31 ms with an empty edit, which put every
  Combine-based splice 31 ms out even with nothing dropped.
- Measured on the live instance, same seed before/after, picture bit-identical: a repair in
  place on a 48 fps clip that dropped one frame went from +20.84 ms to 0.00 ms sync error after
  the splice; a Combine-based join went from +31.03 / +51.84 ms (before / after the splice) to
  0.00 / 0.00 ms.
- New widgets are appended after `crop_h`, so saved UI workflows keep their widget order.

## 2026-09-16 — Licence correction, attribution headers, dependency cleanup, generated diagrams

Findings 2, 3, 5-9, and 12 from the pre-release review
(`REVIEW_seamstitch_prerelease_2026-09-16.md`).

- **Licence** (finding 2): `GPL-3.0-or-later` → `GPL-3.0-only` in `README.md` and
  `pyproject.toml` — both upstream projects (WhatDreamsCost-ComfyUI, ComfyUI-VideoHelperSuite)
  ship stock GPLv3 text with no "or later" statement anywhere.
- **Modification-notice headers** (finding 3, GPL-3.0 §5(a)): added a three-line header naming
  the upstream project, file, licence, and "Modified 2026 for SeamStitch" to `loader.py`
  (derived from WhatDreamsCost-ComfyUI's Load Video UI node) and `recombine.py` (derived from
  VHS's `VHS_VideoCombine`), plus `js/loader.js`. `combine.py` and `js/combine.js` are original
  work, not derived from either upstream, so they got a one-line SeamStitch copyright/licence
  header instead.
- **Diagrams** (finding 5): the three per-node SVGs are now generated by
  [`docs/gen_diagrams.py`](gen_diagrams.py) from a captured `/object_info.json` snapshot, rather
  than hand-drawn — fixes the stale `SeamStitchRecombine.format` default
  (`recombine.py:393`, `'default': 'video/h264-mp4'`), the `SeamStitchLoader` widget order and
  hidden-widget list (matches `loader.py`'s `INPUT_TYPES` order, `js/loader.js`'s
  `toggleWidgetVisibility` hiding `display_mode`/`crop_x`/`crop_y`/`crop_w`/`crop_h` and, at the
  default `display_mode="seconds"`, the frame-mode trio), and the `SeamStitchCombine` caption
  wording (finding 6 — previews render between the file pickers and the status line, not below
  the widgets).
- **README fallback claim** (finding 7): rewrote the "Falls back to a real transcode..."
  sentence to match `combine.py`'s actual rules — frame-rate mismatch → transcode
  (`combine.py:399-401`); a stream-copy that succeeds but whose audio track fails to decode →
  transcode (`combine.py:404-408`); a resolution mismatch (`combine.py:374-377`) or an
  audio-presence mismatch (`combine.py:380-384`) raises instead.
- **README install claim** (finding 8): removed "Search for SeamStitch in ComfyUI Manager" (the
  pack isn't registered — `PublisherId` is empty in `pyproject.toml`); kept Install-via-Git-URL
  and manual clone.
- **README status** (finding 9): The author confirmed `SeamStitchRecombine` has been used in the
  author's own workflows on real footage (not the live install copy's "untested prototype"
  wording, which predates this pack's own gate testing) — updated the Status section
  accordingly for all three nodes.
- **Dependencies** (finding 12): `requirements.txt` and `pyproject.toml` now list the same five
  packages (`av`, `opencv-python`, `pillow`, `aiohttp`, `imageio-ffmpeg`); dropped `numpy` and
  `torch` (ComfyUI guarantees both — listing `torch` risks pulling a CPU wheel on a fresh
  install) and `imageio-ffmpeg`'s optional-extra duplicate; added
  `[project.urls] Repository = "https://github.com/isitrepo/SeamStitch"`.

### Gate

1. `grep -rn 'or-later|or later' README.md pyproject.toml` → no matches.
2. `python docs/gen_diagrams.py object_info.json` (captured from the S2 sandbox gate, which
   already carries the renamed `SeamStitch*` node IDs) run twice produced byte-identical SVGs
   on the second run; each node's widget rows match `object_info`'s `required`
   order minus the hidden set.
3. `pip install -r requirements.txt --dry-run` (embedded ComfyUI python) resolved all five
   packages as already satisfied — no `torch` proposal.

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
