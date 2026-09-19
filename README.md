# SeamStitch

Three ComfyUI nodes for a "trim a broken segment out of a video, regenerate it, splice it back
in seamlessly" workflow:

- **SeamStitch Loader** (`SeamStitchLoader`) — scrub/trim a video with an interactive timeline,
  and get first/last frame outputs for first-last-frame (FLF) generation pipelines, plus
  everything a downstream splice node needs to put the result back — including `bridge_length`,
  a frame count for the generator, optionally longer than the picked gap so the model gets more
  room to work with than the original footage left it.
- **SeamStitch Combine** (`SeamStitchCombine`) — losslessly concatenate two clips
  (container-level stream copy, no re-encode) so the joined file can be reopened and scrubbed for
  bridge/regeneration points.
- **SeamStitch Recombine** (`SeamStitchRecombine`) — splice a regenerated replacement segment
  back into the *original* video at the exact frame range it was cut from, then encode the
  result.

![Recommended wiring between the three nodes](docs/images/wiring_overview.svg)

> The diagrams in this README are labelled node-layout illustrations, not literal screenshots —
> built that way so no real footage/generation output ships in the repo. The three per-node
> diagrams (`load_video_ui_first_last.svg`, `combine_clips_simple.svg`,
> `video_segment_recombine.svg`) are generated straight from a running instance's own
> `/object_info` by [`docs/gen_diagrams.py`](docs/gen_diagrams.py), so their inputs, outputs,
> and widget defaults can't drift from the actual nodes. `wiring_overview.svg` is a hand-drawn
> overview of how the three connect and isn't derived from `object_info`.

## Requirements

- ComfyUI (a reasonably current version — developed against the 2025/2026-era frontend).
- Python packages: `av`, `opencv-python`, `pillow`, `aiohttp` (`numpy` and `torch` are already
  present in any ComfyUI install and aren't listed as dependencies here). See
  [requirements.txt](requirements.txt).
- **ffmpeg** on `PATH`, or the `imageio-ffmpeg` package installed (it bundles its own ffmpeg
  binary — this is the easiest route and is included in `requirements.txt`).
- **[ComfyUI-VideoHelperSuite](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite)**
  installed alongside this pack — required for `SeamStitchRecombine` only (see below). The
  other two nodes have no dependency on it.

## Install

### Option A: Install via Git URL

In ComfyUI Manager, use *Install via Git URL* with this repo's URL:
`https://github.com/isitrepo/SeamStitch.git`. Make sure **ComfyUI-VideoHelperSuite** is
installed too (Manager can install it the same way) if you want the segment-recombine node.

### Option B: manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/isitrepo/SeamStitch.git
pip install -r SeamStitch/requirements.txt
```

Make sure VHS is a sibling folder in `custom_nodes` too:

```bash
git clone https://github.com/kosinkadink/ComfyUI-VideoHelperSuite.git
pip install -r ComfyUI-VideoHelperSuite/requirements.txt
```

Restart ComfyUI. All three nodes appear under the **SeamStitch** category in the node search
(double-click the canvas and type a node's name, or right-click → Add Node → SeamStitch).

### Dependency: ComfyUI-VideoHelperSuite

`SeamStitchRecombine` is a fork of ComfyUI-VideoHelperSuite's `VHS_VideoCombine` node and
reuses VHS's own ffmpeg-format handling, encode pipeline, and audio extraction internals
directly (rather than duplicating hundreds of lines of ffmpeg-piping code). At import time it
locates VHS's installed folder automatically — it doesn't matter what that folder is named, it
scans every `custom_nodes` sibling for a `videohelpersuite` package — and adds it to `sys.path`.
**If VHS isn't found, the other two nodes still load normally**; only `SeamStitchRecombine` is
skipped, with a warning logged at startup naming the problem.

## Nodes

### SeamStitch Loader

![SeamStitch Loader node layout](docs/images/load_video_ui_first_last.svg)

Pick a video (file dropdown, upload, or drag & drop), trim it with an interactive timeline in
seconds or frames, and optionally crop/resize. Outputs include:

| Output | Notes |
| --- | --- |
| `images` / `audio` | The trimmed segment. |
| `first_frame` / `last_frame` | Single-frame batches from the trimmed range — feed these into a first-last-frame generation node. |
| `source_video_path` | The fully resolved path this node opened — wire into `SeamStitchRecombine.original_video_path`. |
| `start_frame` / `end_frame` | The frame range (at `frame_rate`) the trim covers, in the *original* video's own timeline. |
| `frame_rate` | Pass-through of the forced extraction rate, so a downstream node decodes on the identical timeline. |
| `width` / `height` | Actual resolution of `images`, read off the output tensor. |
| `full_clip_audio` | The entire source file's audio track, untouched — feed into `SeamStitchRecombine.original_audio_override` for the final combine. |
| `duration` / `frame_count` | Normally just the picked gap, exactly as decoded — see `extend_bridge` below for when they report something else. |
| `filename` | Informational. |

An optional `input_video`/`input_audio` pair lets you feed frames in directly from an upstream
node (e.g. `SeamStitchCombine`) instead of picking a file — the same trim/crop/resize controls
apply to the given frames, no disk round-trip needed.

Two boolean widgets, `save_first_frame` / `save_last_frame`, save the corresponding frame as a
PNG to ComfyUI's output directory.

**`extend_bridge`** (off by default) changes what `duration`/`frame_count` report, for feeding a
downstream first/last-frame generator's own length widget (LTX-2.5: both
`EmptyLTXVLatentVideo.length` and `LTXVEmptyLatentAudio.frames_number`) — wire `frame_count`
there. Off, they're exactly what they've always been: the picked gap, matching `images.shape[0]`.
On, they instead report the picked gap **plus** `extend_amount` (in `extend_unit` — seconds or
frames), snapped to `bridge_frame_grid`. `images`/`audio`/`first_frame`/`last_frame` are
unaffected either way — only what `duration`/`frame_count` report changes.

This matters because a first/last-frame model has to invent all of the motion between its two
pinned endpoints inside however many frames it's asked to fill — if the picked gap was short,
the model is forced to cram a plausible transition into very little time. Asking for more frames
than the gap actually had gives it more room to work with. `start_frame`/`end_frame` still mark
exactly where `SeamStitchRecombine` cuts the original footage — only the *generated* segment
gets longer, which is what makes the recombined video's total duration grow by the extra amount
(see **SeamStitch Recombine** below — it already splices in whatever length comes back, no
further changes needed there).

`bridge_frame_grid` (`ltx (8k+1)` / `minimax (17k+5)` / `none`, only used when `extend_bridge` is
on) rounds `frame_count` up to whatever grid the generator requires — LTX-2.5 needs `length % 8
== 1` (its temporal downsample factor), MiniMax H3 Motion Context needs `length % 17 == 5`; pick
`none` for a backend with no such constraint. MiniMax H3 separately requires its `context_length`
(how many real frames of motion history it's pinned on) to be one of exactly 5/22/39/56 — that's
a widget on the Motion Context node itself, not something `frame_count` can satisfy for you.

If you extend the bridge, remember to set **SeamStitch Recombine**'s `audio_mode` to `bridge` or
`combined` rather than leaving it at `original` — the source video has no audio for time beyond
the original gap, so `original` mode plays silence under the extra frames.

Based on [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI)'s
Load Video UI node, with the outputs above added on top for FLF/splice workflows.

### SeamStitch Combine

![SeamStitch Combine node layout](docs/images/combine_clips_simple.svg)

Concatenates clip A followed by clip B via ffmpeg's concat demuxer (`-c copy`) — a
container-level splice, not a decode/re-encode, so it's near-instant and introduces no
recompression or colour shift. Requires clip A and clip B to already match in resolution (unless
`resize_to` says otherwise — see below) and in whether each has an audio track (a mismatch there
always raises rather than being papered over silently); when the two frame rates differ, or when
the fast stream copy succeeds but its resulting audio track fails to decode, this falls back
automatically to a real transcode (ffmpeg's concat filter, re-encoded and forced to a single
constant frame rate) instead.

`resize_to` (`off` / `match_a` / `match_b`) controls what happens when the two resolutions don't
match: `off` (default) keeps the strict behaviour above and raises; `match_a`/`match_b` instead
resizes the other clip onto whichever one you picked (aspect ratio always kept, never stretched)
before concatenating. Either resize choice forces the transcode fallback, since a resized clip
can no longer be stream-copied.

`resize_fit` (`crop` / `pad`) picks how that resize reconciles the two aspect ratios. `crop`
(default) scales to fill the target frame and crops the overhang off two edges — no bars, the
right choice for the common case where the two clips are already close to the same aspect ratio
(e.g. two generators' outputs that are both "16:9-ish" but not identical). `pad` scales to fit
inside the frame and letterboxes the rest with black — keeps every source pixel, at the cost of a
visible bar at the seam; use it when the two clips are genuinely differently framed and cropping
would cut off something that matters.

The `seam_frame` output (INT, last) is the index of clip B's first frame in the combined file. It is measured from the written file - output frames `seam_frame-1` and `seam_frame` must reproduce clip A's true last frame and clip B's true first frame - on both the stream-copy and transcode paths, and the node raises rather than emitting a number it could not verify.

The combined file is written to ComfyUI's input directory so it can be reopened in **SeamStitch
Loader** to pick bridge start/end points — when both nodes are in the same graph, this node's
frontend auto-selects its output in any connected `SeamStitchLoader` node once it finishes
running.

### SeamStitch Recombine

![SeamStitch Recombine node layout](docs/images/video_segment_recombine.svg)

Takes a regenerated replacement clip plus the original video path and the frame range that was
cut out of it (wire these straight from **SeamStitch Loader**'s matching outputs), and:

1. Decodes the original video's frames before `start_frame` and after `end_frame`, resized to
   match the regenerated segment's resolution.
2. Drops any held/duplicate frames at the regenerated segment's own leading/trailing edge (a
   keyframe-anchored generation sometimes holds its first/last frame for an extra tick or two,
   which shows up as a stutter at the seam left in).
3. Concatenates `before + deduped_regenerated + after`.
4. Cuts the audio at the same frames as the picture, so sound and picture stay in sync on both
   sides of the splice even when step 2 dropped frames. `audio_mode` picks what plays under the
   regenerated frames:
   - `original` (default) — the source's own audio for exactly the frames that survived, so
     dropped frames take their sound with them. If nothing was dropped, the track is
     sample-identical to the source.
   - `bridge` — the `bridge_audio` input, e.g. audio generated alongside the frames by an
     audio-video model such as LTX-2.5 (`LTXVAudioVAEDecode` on the bridge's audio latent).
     Resampled and channel-matched to the source; if it runs short, the tail falls back to the
     source's audio. Never time-stretched.

   Each join that is not already continuous gets a length-preserving equal-power crossfade
   (`audio_crossfade_ms`, default 20) so a cut mid-waveform cannot click. The source track comes
   from the file itself, or from `original_audio_override` (which must be on the source file's own
   timeline, as `SeamStitchLoader`'s `full_clip_audio` is). The file's own video/audio start
   offset is honoured — `SeamStitchCombine`'s output delays its video 31 ms to cover AAC priming.
5. Encodes the result via ffmpeg. The `format` widget defaults to `video/h264-mp4`.

**Scope vs. the real VHS_VideoCombine:** video formats only (no gif/webp — pipe
`combined_images` into a stock `VHS_VideoCombine` afterward for that), no per-format extra
widgets, no meta-batch/VAE-latent support. This is a full standalone fork focused on the splice
logic, not a wrapper around the stock node.

## Recommended wiring

```
SeamStitchLoader    ──images──────────────────────► (your regeneration pipeline)
                     ──source_video_path───────────► SeamStitchRecombine.original_video_path
                     ──start_frame─────────────────► SeamStitchRecombine.start_frame
                     ──end_frame───────────────────► SeamStitchRecombine.end_frame
                     ──frame_rate──────────────────► SeamStitchRecombine.frame_rate
                     ──full_clip_audio─────────────► SeamStitchRecombine.original_audio_override (optional)

(bridge audio, optional, audio_mode = bridge) ─────► SeamStitchRecombine.bridge_audio

(regeneration pipeline output) ───────────────────► SeamStitchRecombine.regenerated_images
```

`SeamStitchCombine` feeds `SeamStitchLoader` too, for picking bridge points out of two
already-concatenated clips before regenerating the segment between them:

```
SeamStitchCombine ──images/audio──► SeamStitchLoader.input_video/input_audio
```

See [docs/images/wiring_overview.svg](docs/images/wiring_overview.svg) for the same thing as a
diagram.

## Credits / forked from

This pack builds directly on two other GPL-3.0 projects rather than writing everything from
scratch:

- **[WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI)** by
  [WhatDreamsCost](https://github.com/WhatDreamsCost) (GPL-3.0) — `SeamStitchLoader` is a
  fork of its Load Video UI node, with first/last-frame, source-path/frame-index, and
  full-clip-audio outputs added.
- **[ComfyUI-VideoHelperSuite](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite)** by
  [Kosinkadink](https://github.com/kosinkadink) (GPL-3.0) — `SeamStitchRecombine` is a fork of
  its `VHS_VideoCombine` node's encode pipeline, and is also a runtime dependency for that one
  node (see above).

If you use SeamStitch, consider starring/crediting those two projects as well.

## Status

Prototype-stage nodes. `SeamStitchLoader` and `SeamStitchCombine` have been used in the
author's own workflows on real footage; `SeamStitchRecombine` has likewise been used in the
author's own workflows on real footage, though it's the newest of the three. None are yet
published to the Comfy Registry. Issues and PRs welcome.

## License

GPL-3.0-only (see [LICENSE](LICENSE)) — required by this pack's direct reuse of code from
the two GPL-3.0 projects above.
