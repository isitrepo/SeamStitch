#!/usr/bin/env python3
"""Regenerate the SeamStitch node-layout SVGs in docs/images/ from a captured
ComfyUI `/object_info.json` snapshot, so the README's diagrams can never
silently drift from the nodes' real inputs, outputs, and widget defaults.

Usage:
    python docs/gen_diagrams.py <object_info.json>

`object_info.json` is whatever `curl http://127.0.0.1:<port>/object_info`
returns from a running ComfyUI instance with this pack installed (a sandboxed
instance is fine - see docs/CHANGELOG.md for the recipe used to capture one).

This only regenerates the three per-node diagrams (load_video_ui_first_last.svg,
combine_clips_simple.svg, video_segment_recombine.svg). docs/images/wiring_overview.svg
is a hand-drawn overview of how the three nodes connect and is not derived from
object_info - it is left untouched.

Deliberately excluded from each diagram: widgets each node's own JS extension
hides at node-creation time (see HIDDEN_WIDGETS below) and buttons the JS adds
dynamically (e.g. "choose file to upload") - neither exists in object_info's
INPUT_TYPES, so drawing them would mean hardcoding facts this script can't
verify from the snapshot. A caption on the affected diagrams says so instead.
"""
import json
import sys
from pathlib import Path

DOCS_IMAGES = Path(__file__).parent / "images"

# ComfyUI socket types (drawn as connector circles) vs. everything else
# (drawn as a widget row with its default value).
SOCKET_TYPES = {
    "IMAGE", "AUDIO", "LATENT", "MASK", "MODEL", "VAE", "CLIP",
    "CONDITIONING", "CONTROL_NET", "STYLE_MODEL", "CLIP_VISION",
    "VHS_FILENAMES", "VHS_VIDEOINFO",
}

SOCKET_COLOR = {
    "IMAGE": "#64b5f6",
    "AUDIO": "#ff8a80",
    "VHS_FILENAMES": "#ce93d8",
}
WIDGET_TYPE_COLOR = {
    "STRING": "#b0bec5",
    "INT": "#aed581",
    "FLOAT": "#aed581",
    "BOOLEAN": "#aed581",
}
DEFAULT_SOCKET_COLOR = "#b0bec5"

# Widgets js/loader.js's toggleWidgetVisibility() hides unconditionally
# (crop_x/y/w/h, display_mode) or hides in the diagram's default state
# (display_mode defaults to "seconds", so the frame-mode trio stays hidden).
HIDDEN_WIDGETS = {
    "SeamStitchLoader": {
        "display_mode", "crop_x", "crop_y", "crop_w", "crop_h",
        "start_frame", "end_frame", "duration_frames",
    },
}

# Buttons added at runtime by each node's JS extension (onNodeCreated) -
# not part of INPUT_TYPES, so object_info can't tell us these exist. Listed
# here only to drive the caption below each diagram; never drawn as rows.
DYNAMIC_BUTTONS = {
    "SeamStitchLoader": ["choose file to upload", "Load Video"],
    "SeamStitchCombine": ["choose A file", "choose B file", "Load Video"],
    "SeamStitchRecombine": [],
}

CAPTIONS = {
    "SeamStitchLoader": "Interactive timeline/crop UI and the buttons above render below the widgets shown here.",
    "SeamStitchCombine": "A + B previews render between the file pickers and the status line.",
    "SeamStitchRecombine": None,
}

NODE_HEADER_COLOR = {
    "SeamStitchLoader": "#2f6fa5",
    "SeamStitchCombine": "#2f8f5b",
    "SeamStitchRecombine": "#a15a2f",
}

NODE_OUTPUT_FILE = {
    "SeamStitchLoader": "load_video_ui_first_last.svg",
    "SeamStitchCombine": "combine_clips_simple.svg",
    "SeamStitchRecombine": "video_segment_recombine.svg",
}

ROW_H = 24
WIDGET_ROW_H = 24


def _widget_default_text(spec):
    type_spec = spec[0]
    opts = spec[1] if len(spec) > 1 else {}
    if isinstance(type_spec, list):
        default = opts.get("default", type_spec[0] if type_spec else "")
        return str(default)
    if "default" in opts:
        return str(opts["default"])
    return ""


def _classify(node_info, hidden):
    """Split a node's required+optional inputs (in object_info's own order)
    into left-side sockets and below-the-line widget rows, dropping the
    hidden set."""
    inputs = node_info["input"]
    required = inputs.get("required", {})
    optional = inputs.get("optional", {})

    sockets = []  # (name, type, is_optional)
    widgets = []  # (name, default_text)
    for name, spec in list(required.items()) + list(optional.items()):
        is_optional = name in optional
        type_spec = spec[0]
        if isinstance(type_spec, str) and type_spec in SOCKET_TYPES:
            sockets.append((name, type_spec, is_optional))
        elif name not in hidden:
            widgets.append((name, _widget_default_text(spec)))
    return sockets, widgets


def _output_rows(node_info):
    types = node_info["output"]
    names = node_info["output_name"]
    return list(zip(names, types))


def _render_svg(title, header_color, left_sockets, right_outputs, widgets, caption):
    width = 620
    box_x, box_top = 24, 10
    header_h = 30

    n_socket_rows = max(len(left_sockets), len(right_outputs))
    socket_area_h = n_socket_rows * ROW_H
    widgets_h = len(widgets) * WIDGET_ROW_H
    caption_h = 26 if caption else 0

    content_h = header_h + 9 + socket_area_h + (12 if n_socket_rows else 3) + widgets_h + 14 + caption_h
    box_h = content_h
    height = box_top + box_h + 10

    p = []
    p.append(f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             f'xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Helvetica, Arial, sans-serif">')
    p.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#1a1a1a"/>')
    p.append(f'<rect x="{box_x}" y="{box_top}" width="{width - box_x * 2}" height="{box_h}" '
              f'rx="10" fill="#2a2a2a" stroke="#454545" stroke-width="1.5"/>')
    right_edge = width - box_x
    p.append(f'<path d="M {box_x} {box_top + header_h} Q {box_x} {box_top} {box_x + 10} {box_top} '
              f'L {right_edge - 10} {box_top} Q {right_edge} {box_top} {right_edge} {box_top + header_h} Z" '
              f'fill="{header_color}"/>')
    p.append(f'<text x="{box_x + 14}" y="{box_top + 20}" fill="#ffffff" font-size="14" font-weight="600">{title}</text>')

    socket_top = box_top + header_h - 1
    for i in range(n_socket_rows):
        row_y = socket_top + i * ROW_H
        if i % 2 == 0:
            p.append(f'<rect x="{box_x + 1}" y="{row_y}" width="{width - box_x * 2 - 2}" height="{ROW_H}" '
                      f'fill="#242424" opacity="0.4"/>')
        cy = row_y + ROW_H / 2
        if i < len(left_sockets):
            name, type_spec, is_opt = left_sockets[i]
            color = SOCKET_COLOR.get(type_spec, DEFAULT_SOCKET_COLOR)
            style = ' font-style="italic"' if is_opt else ""
            p.append(f'<circle cx="{box_x}" cy="{cy}" r="5.5" fill="{color}" stroke="#1a1a1a" stroke-width="1.5"/>')
            p.append(f'<text x="{box_x + 12}" y="{cy + 4}" fill="#e8e8e8" font-size="12"{style}>{name}</text>')
        if i < len(right_outputs):
            name, type_spec = right_outputs[i]
            color = SOCKET_COLOR.get(type_spec, DEFAULT_SOCKET_COLOR)
            p.append(f'<circle cx="{right_edge}" cy="{cy}" r="5.5" fill="{color}" stroke="#1a1a1a" stroke-width="1.5"/>')
            p.append(f'<text x="{right_edge - 12}" y="{cy + 4}" fill="#e8e8e8" font-size="12" text-anchor="end">{name}</text>')

    line_y = socket_top + socket_area_h + (9 if n_socket_rows else 0)
    p.append(f'<line x1="{box_x + 10}" y1="{line_y}" x2="{right_edge - 10}" y2="{line_y}" stroke="#454545" stroke-width="1"/>')

    wy = line_y + 3
    for i, (name, default) in enumerate(widgets):
        row_y = wy + i * WIDGET_ROW_H
        if i % 2 == 1:
            p.append(f'<rect x="{box_x + 1}" y="{row_y - 3}" width="{width - box_x * 2 - 2}" height="{WIDGET_ROW_H}" '
                      f'fill="#242424" opacity="0.4"/>')
        p.append(f'<rect x="{box_x + 10}" y="{row_y}" width="{width - box_x * 2 - 20}" height="19" '
                  f'rx="4" fill="#3a3a3a" stroke="#555" stroke-width="1"/>')
        p.append(f'<text x="{box_x + 18}" y="{row_y + 13}" fill="#9a9a9a" font-size="11">{name}</text>')
        p.append(f'<text x="{right_edge - 18}" y="{row_y + 13}" fill="#e8e8e8" font-size="11" text-anchor="end">{default}</text>')

    if caption:
        cap_y = wy + widgets_h + 18
        p.append(f'<text x="{width / 2}" y="{cap_y}" fill="#9a9a9a" font-size="10.5" '
                  f'text-anchor="middle" font-style="italic">{caption}</text>')

    p.append("</svg>")
    return "\n".join(p)


def main():
    if len(sys.argv) != 2:
        print("usage: gen_diagrams.py <object_info.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        object_info = json.load(f)

    for node_id, out_file in NODE_OUTPUT_FILE.items():
        if node_id not in object_info:
            print(f"skipping {node_id}: not present in {sys.argv[1]}", file=sys.stderr)
            continue
        node_info = object_info[node_id]
        hidden = HIDDEN_WIDGETS.get(node_id, set())
        left_sockets, widgets = _classify(node_info, hidden)
        right_outputs = _output_rows(node_info)
        svg = _render_svg(
            node_info["display_name"], NODE_HEADER_COLOR[node_id],
            left_sockets, right_outputs, widgets, CAPTIONS.get(node_id),
        )
        out_path = DOCS_IMAGES / out_file
        out_path.write_text(svg + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
