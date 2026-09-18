# Copyright 2026 SeamStitch contributors. Licensed GPL-3.0-only.
"""Splice an audio track the same way SeamStitchRecombine splices the picture.

The picture is `before + kept regenerated frames + after`. The dedup can drop
frames from the regenerated segment's edges, so the audio has to be cut at the
same frame boundaries or everything after the splice drifts against its own
sound - 20.8 ms per dropped frame at 48 fps, for the rest of the file.

Works on decoded PCM, so a cut lands on an exact sample rather than on a 1024
sample AAC frame; the final encode re-quantises the whole track once.

Frame -> sample positions are derived from absolute frame indices, never
accumulated, and the source track is addressed as `offset + Q(frame)` with an
integer sample offset. That keeps an untouched splice (nothing dropped, gap
audio from the source) sample-identical to the source track: both joins are
then provably continuous and get no crossfade.

Only torch - no ComfyUI imports - so it can be exercised outside ComfyUI.
"""

import math

import torch


def frame_sample(frame, frame_rate, sample_rate):
    """First sample of `frame` on a timeline whose frame 0 starts at sample 0."""
    return int(round(frame * sample_rate / float(frame_rate)))


class _Piece:
    """`length` output samples read from `src[:, start:]`. Anything outside
    `[0, limit)` reads as silence - that is how a track that starts late, ends
    early, or has no real audio for an invented frame is padded (never stretched)."""

    def __init__(self, src, start, length, limit=None):
        self.src = src
        self.start = int(start)
        self.length = max(0, int(length))
        self.limit = src.shape[-1] if limit is None else min(int(limit), src.shape[-1])

    def read(self, a, b):
        """Source samples `[a, b)`, zero outside `[0, limit)`."""
        out = torch.zeros((self.src.shape[0], max(0, b - a)), dtype=self.src.dtype)
        lo, hi = max(a, 0), min(b, self.limit)
        if hi > lo:
            out[:, lo - a:hi - a] = self.src[:, lo:hi]
        return out

    def render(self):
        return self.read(self.start, self.start + self.length)

    def continues(self, nxt):
        return (nxt.src is self.src and nxt.start == self.start + self.length
                and self.start + self.length <= self.limit)


def _crossfade(out, j, prev, nxt, n):
    """Blend across the join at output sample `j`, in place. Length-preserving:
    the overlapping material is borrowed from past the end of `prev` (post-join)
    or from before the start of `nxt` (pre-join), so neither side of the join
    moves. Equal-power, because the two sides are not the same signal."""
    after_prev = max(0, prev.limit - (prev.start + prev.length))
    before_next = max(0, min(nxt.start, nxt.limit))
    room_post = min(n, after_prev, out.shape[-1] - j)
    room_pre = min(n, before_next, j)
    if room_post >= n or (room_post >= room_pre and room_post > 0):
        m = room_post
        t = (torch.arange(m, dtype=out.dtype) + 0.5) / m
        tail = prev.read(prev.start + prev.length, prev.start + prev.length + m)
        out[:, j:j + m] = tail * torch.cos(t * math.pi / 2) + out[:, j:j + m] * torch.sin(t * math.pi / 2)
        return "crossfade" if m >= n else f"crossfade shortened to {m} samples"
    if room_pre > 0:
        m = room_pre
        t = (torch.arange(m, dtype=out.dtype) + 0.5) / m
        head = nxt.read(nxt.start - m, nxt.start)
        out[:, j - m:j] = out[:, j - m:j] * torch.cos(t * math.pi / 2) + head * torch.sin(t * math.pi / 2)
        return "crossfade" if m >= n else f"crossfade shortened to {m} samples"
    # Neither side has any material to lend: dip through silence so it cannot click.
    m_out, m_in = min(n, j), min(n, out.shape[-1] - j)
    if m_out:
        out[:, j - m_out:j] *= torch.linspace(1.0, 0.0, m_out, dtype=out.dtype)
    if m_in:
        out[:, j:j + m_in] *= torch.linspace(0.0, 1.0, m_in, dtype=out.dtype)
    return "dip through silence"


def match_format(waveform, sample_rate, channels, target_rate):
    """[C, N] waveform to `channels` channels at `target_rate`. Resampling changes
    the rate, not the duration - no time stretch."""
    waveform = waveform.detach().to("cpu", torch.float32)
    if waveform.shape[0] != channels:
        mono = waveform.mean(dim=0, keepdim=True)
        waveform = mono.expand(channels, -1).clone()
    if int(sample_rate) != int(target_rate):
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), int(target_rate))
    return waveform


def splice_audio(source, sample_rate, frame_rate, start_frame, end_frame,
                 lead_dropped, kept, av_offset_s=0.0, bridge=None, crossfade_ms=20.0,
                 combined_weight=None):
    """Cut `source` ([C, N]) to match the spliced picture.

    Picture: source frames `[0, start_frame)`, then `kept` regenerated frames (the
    first `lead_dropped` having been stripped), then source frames
    `[end_frame + 1, EOF)`. Regenerated frame `i` stands in for source frame
    `start_frame + i`.

    `av_offset_s` is the source's video start minus its audio start: source frame
    `f` plays against source sample `offset + Q(f)`.

    `bridge` ([C, N] at `sample_rate`, same channels) is audio generated alongside
    the regenerated frames; regenerated frame `i` plays against bridge sample
    `Q(i)`. What it does depends on `combined_weight`:

    - `combined_weight is None` ("bridge" mode): `bridge` *replaces* the gap audio
      outright. Without `bridge`, the gap keeps the source's own audio for exactly
      the frames that survived, so dropped frames take their sound with them.
    - `combined_weight` given ("combined" mode): the gap keeps the source's own
      audio (same as no `bridge`), and `bridge` is additively mixed on top of it
      at that weight (0-1), scaled and summed rather than swapped in, so ambience
      never drops out and the transition into/out of the gap has no sudden
      character change. The mix is clamped to [-1, 1] afterward.

    Returns `(waveform [C, M], notes)`.
    """
    q = lambda f: frame_sample(f, frame_rate, sample_rate)
    offset = int(round(av_offset_s * sample_rate))
    fade = max(0, int(round(crossfade_ms * sample_rate / 1000.0)))
    gap_frames = end_frame - start_frame + 1
    gap_len = q(start_frame + kept) - q(start_frame)
    notes = []

    before = _Piece(source, offset, q(start_frame))
    after_start = offset + q(end_frame + 1)
    after = _Piece(source, after_start, source.shape[-1] - after_start)

    is_combined = bridge is not None and combined_weight is not None
    replace_with_bridge = bridge is not None and not is_combined
    use_original_gap = not replace_with_bridge

    fill = None
    if replace_with_bridge:
        real = max(0, min(gap_len, bridge.shape[-1] - q(lead_dropped)))
        gap = _Piece(bridge, q(lead_dropped), real)
        if real < gap_len:
            # The audio VAE decodes a little short of the latent's nominal length
            # (17 latents -> 0.65 s for a 0.6875 s bridge, measured). Silence there
            # would be an audible dip in steady ambience right before the join, so
            # the shortfall is filled with the source's own audio behind those
            # frames - which also makes the join into `after` sample-continuous.
            missing = gap_len - real
            fill = _Piece(source, after_start - missing, missing)
            notes.append(f"bridge audio ends {missing} samples ({1000.0 * missing / sample_rate:.1f} ms) "
                         f"before its frames do; that tail uses the source's own audio")
    else:
        first = start_frame + lead_dropped
        if first + kept == end_frame + 1 and lead_dropped > 0:
            # Only the leading edge was cut: anchor on the end so the trailing
            # join stays sample-continuous instead of a 1-sample rounding skip.
            gap_start = after_start - gap_len
        else:
            gap_start = offset + q(first)
        gap = _Piece(source, gap_start, gap_len, limit=after_start)
        if first + kept > end_frame + 1:
            notes.append(f"{first + kept - end_frame - 1} regenerated frame(s) have no "
                         f"source audio behind them; padded with silence")

    pieces = [before, gap] + ([fill] if fill is not None else []) + [after]
    out = torch.cat([p.render() for p in pieces], dim=-1)

    # Source gap audio that runs out before its frames do (more regenerated frames
    # than the gap had) stops dead into silence; fade it so that cannot click.
    real = min(gap.length, gap.limit - gap.start)
    if use_original_gap and 0 < real < gap.length and fade > 0:
        m = min(fade, real)
        j = before.length + real
        out[:, j - m:j] *= torch.cos((torch.arange(m, dtype=out.dtype) + 0.5) / m * math.pi / 2)

    if is_combined:
        real_bridge = max(0, min(gap.length, bridge.shape[-1] - q(lead_dropped)))
        if real_bridge > 0:
            bridge_seg = _Piece(bridge, q(lead_dropped), real_bridge).render() * combined_weight
            j0 = before.length
            out[:, j0:j0 + real_bridge] += bridge_seg
            out.clamp_(-1.0, 1.0)
        notes.append(f"combined mode: bridge audio mixed over the original gap audio at weight {combined_weight:g}")
        if real_bridge < gap.length:
            notes.append(f"bridge audio covers {real_bridge} of {gap.length} gap sample(s) "
                         f"({1000.0 * (gap.length - real_bridge) / sample_rate:.1f} ms short); "
                         f"original audio alone for the remainder")

    j = 0
    live = []
    for piece in pieces:
        live.append((j, piece))
        j += piece.length
    live = [(at, piece) for at, piece in live if piece.length > 0]
    for (_, prev), (at, nxt) in zip(live, live[1:]):
        if prev.continues(nxt) or fade == 0:
            continue
        frame = round(at * frame_rate / sample_rate)
        notes.append(f"join at output frame {frame}: " + _crossfade(out, at, prev, nxt, fade))
    if gap_frames != kept and use_original_gap:
        notes.append(f"gap audio {'shortened' if kept < gap_frames else 'lengthened'} from "
                     f"{gap_frames} to {kept} frame(s) to match the picture")
    return out, notes
