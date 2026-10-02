// SeamStitch Timeline: a one-track mini video editor for the SeamStitch splice.
//
// The UI edits two hidden widgets and nothing else:
//   sequence  one clip per line, "path @ enter..exit", "~ N" for a gap (timeline_math.py)
//   target    JSON: {"mode":"replace","start":S,"end":E} | {"mode":"gap","trim":N} | {}
// Frame numbers are at the node's frame rate. "Strip" frames include gaps (what you see);
// "cut" frames skip them (the assembled video Recombine splices).
//
// Controls modelled on chanon/comfyui-obvpm-timeline's Timeline node (GPL-3.0): a strip
// sized by played frames, trim handles on block edges, cut left/right at the playhead,
// seam pills, a next-run bar, quick (chained players) and full (server-built) preview.
// Written independently for SeamStitch's replace/bridge model.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const GRID_LTX = "ltx (8k+1)", GRID_MM = "minimax (17k+5)", GRID_NONE = "none";
const C = {
    bg: "#16181d", ruler: "#20242c", tick: "#6b7280", text: "#e5e7eb", dim: "#9ca3af",
    clip: "#2f4a7a", clipSel: "#3b62a8", clipEdge: "#8fb3ff", gap: "#3a3a3a",
    target: "rgba(168, 85, 247, 0.55)", targetEdge: "#c084fc", ctx: "rgba(168, 85, 247, 0.22)",
    play: "#f59e0b", seam: "#e5e7eb", warn: "#fbbf24", err: "#f87171", ok: "#34d399",
};
// Strip geometry lives per node (see setScale in buildTimeline): it grows with the node's
// width so the strip, its text and the buttons stay readable on a big node.


// ---------------------------------------------------------------- pure mirrors of timeline_math.py

function parseSequence(text) {
    const out = [];
    for (const raw of (text || "").split(/\r?\n/)) {
        const line = raw.trim();
        if (!line || line.startsWith("#")) continue;
        if (line.startsWith("~")) {
            const n = parseInt(line.slice(1).trim(), 10);
            if (Number.isFinite(n) && n > 0) out.push({ kind: "gap", frames: n });
            continue;
        }
        let path = line, enter = 0, exit = null;
        const at = line.lastIndexOf(" @ ");
        if (at >= 0) {
            path = line.slice(0, at).trim();
            const [a, b] = line.slice(at + 3).split("..");
            enter = a && a.trim() ? parseInt(a, 10) : 0;
            exit = b && b.trim() ? parseInt(b, 10) : null;
        }
        out.push({ kind: "clip", path, enter: enter || 0, exit });
    }
    return out;
}

function formatSequence(entries) {
    return entries.map(e => {
        if (e.kind === "gap") return `~ ${e.frames}`;
        if (e.enter || e.exit != null) return `${e.path} @ ${e.enter || 0}..${e.exit == null ? "" : e.exit}`;
        return e.path;
    }).join("\n");
}

function snapUp(n, grid) {
    n = Math.max(1, Math.round(n));
    if (grid === GRID_NONE) return n;
    if (grid === GRID_MM) { n = Math.max(5, n); const r = (n - 5) % 17; return r ? n + 17 - r : n; }
    n = Math.max(9, n); const r = (n - 1) % 8; return r ? n + 8 - r : n;
}

function snapNearest(n, grid) {
    if (grid === GRID_NONE) return Math.max(3, Math.round(n));
    const step = grid === GRID_MM ? 17 : 8, base = grid === GRID_MM ? 5 : 1, floor = grid === GRID_MM ? 5 : 9;
    n = Math.max(floor, Math.round(n));
    const lo = Math.floor((n - base) / step) * step + base, hi = lo + step;
    return (n - lo) < (hi - n) ? lo : hi;
}

function hideWidget(w) {
    if (!w) return;
    w.hidden = true;
    w.options = w.options || {};
    w.options.hidden = true;
    if (!window.LiteGraph || !window.LiteGraph.vueNodesMode) {
        w.computeSize = () => [0, -4];
        w.draw = () => { };
    }
    if (w.element) w.element.style.display = "none";
}

function el(tag, style, text) {
    const e = document.createElement(tag);
    if (style) Object.assign(e.style, style);
    if (text != null) e.textContent = text;
    return e;
}

function button(label, title, onclick) {
    const b = el("button", {
        background: "#2a2f3a", color: C.text, border: "1px solid #3b4252", borderRadius: "4px",
        padding: "0.15em 0.6em", fontSize: "1em", cursor: "pointer", whiteSpace: "nowrap", lineHeight: "1.6em",
    }, label);
    b.title = title || "";
    b.onclick = (ev) => { ev.stopPropagation(); onclick(ev); };
    b.onpointerdown = (ev) => ev.stopPropagation();
    return b;
}

// ComfyUI restores widget values by POSITION. A value saved for a slot that has since
// changed meaning (a new widget, or the blank this node's own UI panel used to save)
// lands in the wrong widget, e.g. '' in a dropdown, and the queue is refused
// ("Value not in list"). Reset any dropdown holding a value it does not offer, and any
// number that is not a number, to the widget's default.
function repairWidgetValues(node) {
    for (const w of node.widgets || []) {
        if (w.type === "combo") {
            const vals = w.options?.values;
            const list = typeof vals === "function" ? vals() : vals;
            if (Array.isArray(list) && list.length && !list.includes(w.value)) {
                const def = w.options?.default;
                w.value = list.includes(def) ? def : list[0];
            }
        } else if (w.type === "number" && !(typeof w.value === "number" && Number.isFinite(w.value))) {
            w.value = w.options?.default ?? w.options?.min ?? 0;
        }
    }
}

const viewURL = (path) => api.apiURL(`/seamstitch/loader/view?filename=${encodeURIComponent(path)}`);

async function uploadFile(file) {
    const CHUNK = 8 * 1024 * 1024, total = Math.max(1, Math.ceil(file.size / CHUNK));
    let name = file.name;
    for (let i = 0; i < total; i++) {
        const fd = new FormData();
        fd.append("file", file.slice(i * CHUNK, (i + 1) * CHUNK));
        fd.append("filename", file.name);
        fd.append("chunk_index", String(i));
        fd.append("total_chunks", String(total));
        const r = await api.fetchApi("/seamstitch/loader/upload_chunk", { method: "POST", body: fd });
        if (!r.ok) throw new Error(`upload failed (${r.status})`);
        const j = await r.json();
        if (j.name) name = j.name;
    }
    return name;
}

// ---------------------------------------------------------------- the node

app.registerExtension({
    name: "SeamStitch.Timeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SeamStitchTimeline") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            buildTimeline(this);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            repairWidgetValues(this);
            if (this.ssTimeline) setTimeout(() => this.ssTimeline.reload(), 0);
            return r;
        };
    },
});

function buildTimeline(node) {
    const W = (n) => node.widgets.find(w => w.name === n);
    const seqW = W("sequence"), targetW = W("target"), frW = W("frame_rate"), gridW = W("bridge_frame_grid");
    const ctxW = W("context_frames"), extW = W("extend_frames"), conformW = W("conform_to_24fps");
    // conform_to_24fps: the cut plays on MiniMax H3's 24 fps clock (frames unchanged).
    const h3Clock = () => (conformW && conformW.value && Math.round(S.fr) !== 24) ? 24 : S.fr;
    // Seconds <-> cut frames in the FULL preview use the built cut's own rate (24 when conformed).
    const fullFr = () => (S.full && S.full.frame_rate) || S.fr;
    hideWidget(seqW);
    hideWidget(targetW);

    // ------------------------------------------------------------ state
    const S = {
        entries: [], info: {}, fr: 24, target: {}, sel: -1, playhead: 0, pxPerFrame: 3, scroll: 0,
        mode: "quick", full: null, fullKey: "", playing: false, hover: null, drag: null,
        msg: "", msgKind: "dim", loadingProbe: 0,
    };

    // ------------------------------------------------------------ geometry (scales with the node)
    // U = node width / 800, clamped to 1..2.5. The seam pill sits at the TOP of the track, the
    // clip's name bar (drag it to reorder) under it, and the trim grips on the lower half, so
    // none of them overlap at a join between two clips.
    let U = 1, RULER_H, TRACK_Y, TRACK_H, BAND_Y, BAND_H, CANVAS_H, EDGE_PX, SEAM_Y, GRIP_Y0, LABEL_Y1;
    function setScale(u) {
        U = u;
        RULER_H = Math.round(20 * U); TRACK_Y = RULER_H + Math.round(4 * U); TRACK_H = Math.round(44 * U);
        BAND_Y = TRACK_Y + TRACK_H + Math.round(4 * U); BAND_H = Math.round(16 * U); CANVAS_H = BAND_Y + BAND_H + 2;
        EDGE_PX = Math.round(7 * U); SEAM_Y = TRACK_Y + Math.round(9 * U);
        LABEL_Y1 = TRACK_Y + Math.round(18 * U); GRIP_Y0 = LABEL_Y1 + 2;
    }
    setScale(1);
    const font = (px, bold) => `${bold ? "bold " : ""}${Math.round(px * U)}px sans-serif`;

    // ------------------------------------------------------------ DOM
    const root = el("div", {
        display: "flex", flexDirection: "column", gap: "4px", width: "100%", height: "100%",
        boxSizing: "border-box", fontFamily: "sans-serif", fontSize: "11px", color: C.text,
        userSelect: "none", outline: "none", overflow: "hidden",
    });
    root.tabIndex = 0;

    const videoBox = el("div", { position: "relative", flex: "1 1 auto", minHeight: "120px", background: "#000",
        borderRadius: "4px", overflow: "hidden" });
    const video = el("video", { width: "100%", height: "100%", objectFit: "contain", display: "block" });
    video.muted = false;
    video.playsInline = true;
    video.preload = "auto";
    const overlay = el("div", { position: "absolute", left: "6px", top: "4px", fontSize: "1em", fontWeight: "bold",
        color: C.text, textShadow: "0 0 3px #000", pointerEvents: "none" });
    const gapCover = el("div", { position: "absolute", inset: "0", background: "repeating-linear-gradient(45deg,#111 0 10px,#1b1b1b 10px 20px)",
        display: "none", alignItems: "center", justifyContent: "center", color: C.dim, fontSize: "1.2em" }, "gap - the bridge goes here");
    const dropHint = el("div", { position: "absolute", inset: "0", display: "none", alignItems: "center", justifyContent: "center",
        background: "rgba(59,98,168,0.35)", border: "2px dashed #8fb3ff", color: "#fff", fontSize: "14px" }, "drop videos to add them to the strip");
    const emptyHint = el("div", { position: "absolute", inset: "0", display: "flex", alignItems: "center", justifyContent: "center",
        color: C.dim, fontSize: "1.1em", textAlign: "center", padding: "10px" },
        "Drag videos here, or use + add.\nThen mark what to regenerate: I / O for a range, or click a cut between clips.");
    emptyHint.style.whiteSpace = "pre-line";
    videoBox.append(video, gapCover, emptyHint, overlay, dropHint);

    const bar1 = el("div", { display: "flex", gap: "4px", alignItems: "center", flexWrap: "wrap", flexShrink: "0" });
    const bPlay = button("▶", "Play / pause (space)", () => togglePlay());
    const bMode = button("quick", "quick: plays the clips one after another, instantly. full: builds the real cut (the file Recombine will splice) and plays that.", () => setMode(S.mode === "quick" ? "full" : "quick"));
    const bZoomOut = button("−", "Zoom out (wheel over the ruler)", () => zoomBy(1 / 1.4));
    const bZoomIn = button("+", "Zoom in (wheel over the ruler)", () => zoomBy(1.4));
    const bFit = button("fit", "Fit the whole strip into the node", () => fit());
    const bAdd = button("+ add", "Add a video from the input folder, or upload one", (ev) => openAddMenu(ev));
    const bText = button("✎", "Edit the strip as text", () => openTextEditor());
    const bIn = button("I", "Mark in: the replace range starts at the playhead", () => markIn());
    const bOut = button("O", "Mark out: the replace range ends at the playhead", () => markOut());
    const bClear = button("✕ mark", "Clear what is marked to regenerate", () => setTarget({}));
    const status = el("span", { marginLeft: "auto", color: C.dim, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "45%" });
    bar1.append(bPlay, bMode, el("span", { width: "6px" }), bZoomOut, bZoomIn, bFit, el("span", { width: "6px" }),
        bAdd, bText, el("span", { width: "6px" }), bIn, bOut, bClear, status);

    const canvas = el("canvas", { width: "100%", height: `${CANVAS_H}px`, display: "block", cursor: "default",
        borderRadius: "4px", touchAction: "none", flex: "0 0 auto" });

    const selBar = el("div", { display: "flex", gap: "4px", alignItems: "center", minHeight: "1.8em", flexWrap: "wrap", flexShrink: "0" });
    const nextBar = el("div", { display: "flex", gap: "6px", alignItems: "center", minHeight: "1.8em", padding: "2px 6px", flexShrink: "0",
        background: "#1e1b2e", border: "1px solid #3b2d5c", borderRadius: "4px", flexWrap: "wrap" });

    const fileInput = el("input", { display: "none" });
    fileInput.type = "file";
    fileInput.accept = "video/*";
    fileInput.multiple = true;
    fileInput.onchange = async () => { await addFiles([...fileInput.files], S.entries.length); fileInput.value = ""; };

    root.append(videoBox, bar1, canvas, selBar, nextBar, fileInput);

    const widget = node.addDOMWidget("timeline_ui", "div", root, { serialize: false, hideOnZoom: false });
    // Keep the panel out of widgets_values: the option alone did not, so a blank was saved
    // after the last real widget and shifted onto the next widget ever appended.
    widget.serialize = false;
    widget.computeSize = (width) => [Math.max(400, (width || node.size[0]) - 20), 460];
    if (node.size[0] < 760) node.size[0] = 760;
    if (node.size[1] < 820) node.size[1] = 820;

    // The widget fills whatever height the node has below its other widgets (the video
    // grows with it), and the strip/text scale with the node's width. Checked every frame
    // the node draws - the Loader does the same - because resizes arrive by several paths.
    function fitToNode() {
        const u = Math.max(1, Math.min(2.5, node.size[0] / 800));
        if (Math.abs(u - U) > 0.01) {
            setScale(u);
            root.style.fontSize = `${Math.round(11 * U)}px`;
            canvas.style.height = `${CANVAS_H}px`;
            refresh();
        }
        if (widget.last_y) {
            const h = Math.max(300, node.size[1] - widget.last_y - 15);
            if (Math.abs((parseFloat(root.style.height) || 0) - h) > 1) root.style.height = `${h}px`;
        }
    }
    const onDrawFg = node.onDrawForeground;
    node.onDrawForeground = function () { const r = onDrawFg?.apply(this, arguments); fitToNode(); return r; };
    const onResize = node.onResize;
    node.onResize = function () { const r = onResize?.apply(this, arguments); fitToNode(); return r; };

    // ------------------------------------------------------------ helpers
    function toast(msg, kind = "dim") { S.msg = msg; S.msgKind = kind; status.textContent = msg; status.style.color = C[kind] || C.dim; status.title = msg; }
    const grid = () => gridW ? gridW.value : GRID_LTX;
    const ctxK = () => ctxW ? (ctxW.value | 0) : 0;
    const extN = () => extW ? (extW.value | 0) : 0;

    function clipFrames(e) { const inf = S.info[e.path]; return inf ? inf.frames : null; }
    function played(e) {
        if (e.kind === "gap") return e.frames;
        const total = clipFrames(e);
        if (total == null) return Math.max(1, (e.exit ?? (e.enter + 24)) - e.enter);
        return Math.max(0, Math.min(e.exit ?? total, total) - e.enter);
    }
    function exitOf(e) { const t = clipFrames(e); return Math.min(e.exit ?? t, t); }

    function layout() {
        let strip = 0, cut = 0;
        return S.entries.map((e, i) => {
            const len = played(e);
            const L = { i, e, strip, len, cut: e.kind === "clip" ? cut : null };
            strip += len;
            if (e.kind === "clip") cut += len;
            return L;
        });
    }
    function totals() { const L = layout(); const last = L[L.length - 1]; return { L, strip: last ? last.strip + last.len : 0, cut: L.filter(x => x.e.kind === "clip").reduce((a, x) => a + x.len, 0) }; }
    function entryAtStrip(f) { const L = layout(); for (const x of L) if (f >= x.strip && f < x.strip + x.len) return x; return L[L.length - 1] || null; }
    // Whole frames only: callers pass pointer positions, and a fractional frame saved into
    // the target (281.83) displayed oddly and was truncated by the Python side.
    function stripToCut(f) { f = Math.floor(f); const x = entryAtStrip(f); if (!x) return 0; return x.e.kind === "clip" ? x.cut + (f - x.strip) : (x.cut ?? cutBefore(x.i)); }
    function cutBefore(i) { return layout().filter(x => x.i < i && x.e.kind === "clip").reduce((a, x) => a + x.len, 0); }
    function cutToStrip(c) { for (const x of layout()) if (x.e.kind === "clip" && c >= x.cut && c < x.cut + x.len) return x.strip + (c - x.cut); return totals().strip; }

    function save(clearTarget) {
        seqW.value = formatSequence(S.entries);
        if (clearTarget && S.target.mode === "replace") { setTarget({}, true); toast("mark cleared - the strip changed under it", "warn"); }
        if (S.target.mode === "gap" && !S.entries.some(e => e.kind === "gap")) setTarget({}, true);
        S.full = null;
        if (S.mode === "full") setMode("quick");
        app.graph?.setDirtyCanvas(true, true);
        refresh();
        // A different first clip (reordered, removed, replaced) sets the auto rate and the
        // frame_rate label: re-probe, as adding a clip does.
        const first = S.entries.find(e => e.kind === "clip");
        if ((first ? first.path : null) !== S.probedFirst) probeAll();
    }
    function setTarget(t, silent) {
        S.target = t || {};
        targetW.value = Object.keys(S.target).length ? JSON.stringify(S.target) : "";
        app.graph?.setDirtyCanvas(true, true);
        if (!silent) refresh();
    }

    async function probeAll() {
        const paths = [...new Set(S.entries.filter(e => e.kind === "clip").map(e => e.path))];
        const first = S.entries.find(e => e.kind === "clip");
        let fr = frW && frW.value > 0 ? frW.value : 0;
        S.nativeFr = 0;
        S.probedFirst = first ? first.path : null;
        if (first) {
            const inf = await probeOne(first.path, 0);
            S.nativeFr = inf ? inf.native_fps || 0 : 0;
            if (!fr) fr = inf ? Math.round(inf.native_fps) || 24 : 24;
        }
        S.fr = fr || 24;
        const token = ++S.loadingProbe;
        await Promise.all(paths.map(async p => { const inf = await probeOne(p, S.fr); if (inf && token === S.loadingProbe) S.info[p] = inf; }));
        refresh();
    }
    const probeCache = {};
    async function probeOne(path, fr) {
        const key = `${path}|${fr}`;
        if (probeCache[key]) return probeCache[key];
        try {
            const r = await api.fetchApi(`/seamstitch/timeline/probe?path=${encodeURIComponent(path)}&frame_rate=${fr}`);
            const j = await r.json();
            if (!r.ok) { toast(`${path}: ${j.error}`, "err"); return null; }
            probeCache[key] = j;
            return j;
        } catch (e) { toast(`probe failed: ${e}`, "err"); return null; }
    }

    // ------------------------------------------------------------ plan (mirror of resolve_target)
    function plan() {
        const t = totals();
        const gaps = t.L.filter(x => x.e.kind === "gap");
        const tg = S.target;
        if (!t.L.some(x => x.e.kind === "clip")) return { err: "no clips on the strip yet" };
        if (gaps.length > 1) return { err: `${gaps.length} gaps - SeamStitch does one splice at a time; close all but one` };
        if (tg.mode === "replace") {
            if (gaps.length) return { err: "there is a gap but the mark is a range - close the gap or bridge it" };
            const s = tg.start, e = tg.end;
            if (!(s >= 1 && e >= s && e <= t.cut - 2)) return { err: `range ${s}..${e} must leave a real frame either side (cut is 0..${t.cut - 1})` };
            const k = ctxK();
            if (k && (s - k < 0 || e + k > t.cut - 1)) return { err: `context_frames ${k} needs ${k} real frames either side of the range` };
            return { mode: "replace", start: s, end: e, gen: snapUp(e - s + 1 + 2 * k + extN(), grid()), k };
        }
        if (tg.mode === "gap") {
            if (!gaps.length) return { err: "the mark is a gap bridge but there is no gap" };
            const g = gaps[0], join = cutBefore(g.i), G = g.e.frames;
            if (join < 1 || join > t.cut - 1) return { err: "a bridged gap needs real footage on both sides - move it between two clips" };
            let s, e;
            if ("start" in tg || "end" in tg) { s = tg.start ?? join; e = tg.end ?? join - 1; }
            else { const tr = tg.trim | 0; s = join - tr; e = join + tr - 1; }        // older saves
            if (s === join && e === join - 1) {
                if (ctxK()) return { err: "context_frames needs frames marked around the gap (I / O) - a pure insert pins one kept frame each side" };
                return { mode: "gap", pure: true, start: s, end: e, join, G, gx: g, length: G + 2, gen: snapNearest(G + 2, grid()), k: 0 };
            }
            if (!(s >= 1 && s < join && e >= join && e <= t.cut - 2))
                return { err: `the I / O markers must straddle the gap: I before cut frame ${join}, O at or after it (now ${s}..${e}) - or press "pure insert"` };
            const k = ctxK();
            if (k && (s - k < 0 || e + k > t.cut - 1)) return { err: `context_frames ${k} needs ${k} real frames outside the markers` };
            return { mode: "gap", pure: false, start: s, end: e, join, G, gx: g, k, gen: snapUp(e - s + 1 + 2 * k + extN() + G, grid()) };
        }
        return { none: true };
    }

    // ------------------------------------------------------------ drawing
    // The frame_rate widget's label says what rate is in use: 0 is "auto" (the first clip's own
    // rate), and with conform_to_24fps on, the cut plays at H3's 24. Label only - the value
    // and what gets saved are untouched.
    function updateRateLabel() {
        if (!frW) return;
        const nat = S.nativeFr ? (Math.abs(S.nativeFr - Math.round(S.nativeFr)) < 0.01 ? `${Math.round(S.nativeFr)}` : S.nativeFr.toFixed(3)) : "";
        let label = "frame_rate";
        if (!(frW.value > 0)) label += nat ? ` (auto: ${nat} fps` : " (auto";
        else if (nat && Math.abs(S.nativeFr - frW.value) > 0.01) label += ` (clip is ${nat} fps`;
        else label += nat ? ` (${nat} fps` : "";
        if (label !== "frame_rate") {
            if (conformW && conformW.value && Math.round(S.fr) !== 24) label += " → 24 for H3";
            label += ")";
        }
        if (frW.label !== label) { frW.label = label; app.graph?.setDirtyCanvas(true, true); }
    }

    function refresh() {
        updateRateLabel();
        drawCanvas();
        drawSelBar();
        drawNextBar();
        const hasClips = S.entries.some(e => e.kind === "clip");
        emptyHint.style.display = hasClips ? "none" : "flex";
        bMode.textContent = S.mode === "full" ? "full ●" : "quick";
        bMode.style.borderColor = S.mode === "full" ? C.ok : "#3b4252";
        updateOverlay();
    }

    function f2x(f) { return f * S.pxPerFrame - S.scroll; }
    function x2f(x) { return (x + S.scroll) / S.pxPerFrame; }

    function drawCanvas() {
        const w = Math.max(50, canvas.clientWidth | 0);
        // Backing store at device pixels times the graph zoom, so text stays sharp when
        // the node is zoomed in. Drawing coordinates stay in CSS pixels.
        const k = (window.devicePixelRatio || 1) * Math.max(1, Math.min(4, app.canvas?.ds?.scale || 1));
        const bw = Math.round(w * k), bh = Math.round(CANVAS_H * k);
        if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
        const g = canvas.getContext("2d");
        g.setTransform(k, 0, 0, k, 0, 0);
        g.fillStyle = C.bg; g.fillRect(0, 0, w, CANVAS_H);
        const t = totals();
        // ruler
        g.fillStyle = C.ruler; g.fillRect(0, 0, w, RULER_H);
        const secPx = S.fr * S.pxPerFrame;
        const stepSec = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120].find(s => s * secPx >= 50 * U) || 300;
        g.fillStyle = C.tick; g.strokeStyle = C.tick; g.font = font(10); g.textBaseline = "top";
        for (let s = Math.floor(x2f(0) / S.fr / stepSec) * stepSec; s * S.fr <= x2f(w); s += stepSec) {
            const x = Math.round(f2x(s * S.fr)) + 0.5;
            if (x < 0) continue;
            g.beginPath(); g.moveTo(x, RULER_H - 6 * U); g.lineTo(x, RULER_H); g.stroke();
            g.fillText(stepSec < 1 ? `${s.toFixed(2)}s` : `${s}s`, x + 2, 3 * U);
        }
        // track
        g.textBaseline = "middle";
        const seams = [];
        for (const x of t.L) {
            const x0 = f2x(x.strip), x1 = f2x(x.strip + x.len);
            if (x1 < 0 || x0 > w) { if (x.e.kind === "clip" && x.i > 0) seams.push(x); continue; }
            if (x.e.kind === "gap") {
                g.fillStyle = C.gap; g.fillRect(x0, TRACK_Y, x1 - x0, TRACK_H);
                g.strokeStyle = "#555"; g.save(); g.beginPath(); g.rect(x0, TRACK_Y, x1 - x0, TRACK_H); g.clip();
                for (let hx = x0 - TRACK_H; hx < x1; hx += 8) { g.beginPath(); g.moveTo(hx, TRACK_Y + TRACK_H); g.lineTo(hx + TRACK_H, TRACK_Y); g.stroke(); }
                g.restore();
                g.fillStyle = C.dim; g.fillText(`gap ${x.len}f`, x0 + 4, TRACK_Y + TRACK_H / 2);
                if (S.sel === x.i) { g.strokeStyle = C.clipEdge; g.lineWidth = 2; g.strokeRect(x0 + 1, TRACK_Y + 1, x1 - x0 - 2, TRACK_H - 2); g.lineWidth = 1; }
            } else {
                const inf = S.info[x.e.path];
                g.fillStyle = S.sel === x.i ? C.clipSel : C.clip;
                g.fillRect(x0 + 1, TRACK_Y, Math.max(1, x1 - x0 - 2), TRACK_H);
                g.fillStyle = C.clipEdge;
                g.fillRect(x0 + 1, GRIP_Y0, 3 * U, TRACK_Y + TRACK_H - GRIP_Y0 - 3);
                g.fillRect(x1 - 1 - 3 * U, GRIP_Y0, 3 * U, TRACK_Y + TRACK_H - GRIP_Y0 - 3);
                g.save(); g.beginPath(); g.rect(x0 + 6, TRACK_Y, Math.max(0, x1 - x0 - 12), TRACK_H); g.clip();
                g.fillStyle = "rgba(0,0,0,0.18)"; g.fillRect(x0 + 1, TRACK_Y, Math.max(1, x1 - x0 - 2), LABEL_Y1 - TRACK_Y);
                g.fillStyle = C.dim; g.font = font(9, true); g.fillText("⠿", x0 + 7, TRACK_Y + 9 * U);
                g.fillStyle = C.text; g.font = font(11, true);
                g.fillText(x.e.path.split(/[\\/]/).pop(), x0 + 7 + 10 * U, TRACK_Y + 9 * U);
                g.font = font(10); g.fillStyle = C.dim;
                const cutNote = inf && (x.e.enter > 0 || exitOf(x.e) < inf.frames) ? ` · plays ${x.e.enter}-${exitOf(x.e) - 1} of ${inf.frames}` : "";
                g.fillText(inf ? `${x.len}f · ${(x.len / S.fr).toFixed(2)}s${cutNote}` : "probing…", x0 + 7, TRACK_Y + 30 * U);
                g.restore();
                if (x.i > 0 && t.L[x.i - 1].e.kind === "clip") seams.push(x);
            }
        }
        // target band (+ context)
        const p = plan();
        g.fillStyle = "#1c1f26"; g.fillRect(0, BAND_Y, w, BAND_H);
        if (p.mode === "replace") {
            const s0 = cutToStrip(p.start), s1 = cutToStrip(p.end) + 1;
            if (p.k) {
                g.fillStyle = C.ctx;
                g.fillRect(f2x(cutToStrip(p.start - p.k)), TRACK_Y, f2x(s0) - f2x(cutToStrip(p.start - p.k)), TRACK_H);
                g.fillRect(f2x(s1), TRACK_Y, f2x(cutToStrip(p.end + p.k) + 1) - f2x(s1), TRACK_H);
            }
            g.fillStyle = C.target; g.fillRect(f2x(s0), TRACK_Y, f2x(s1) - f2x(s0), TRACK_H);
            g.fillStyle = C.targetEdge; g.fillRect(f2x(s0), BAND_Y + 2, f2x(s1) - f2x(s0), BAND_H - 4);
            g.fillStyle = "#fff"; g.font = font(10);
            g.fillText(`regenerate ${p.end - p.start + 1}f`, f2x(s0) + 3, BAND_Y + BAND_H / 2);
        } else if (p.mode === "gap") {
            const gx = p.gx;
            const [s0, s1] = markSpan(p);
            if (!p.pure) {
                if (p.k) {
                    g.fillStyle = C.ctx;
                    g.fillRect(f2x(cutToStrip(p.start - p.k)), TRACK_Y, f2x(s0) - f2x(cutToStrip(p.start - p.k)), TRACK_H);
                    g.fillRect(f2x(s1), TRACK_Y, f2x(cutToStrip(p.end + p.k) + 1) - f2x(s1), TRACK_H);
                }
                g.fillStyle = C.target; g.fillRect(f2x(s0), TRACK_Y, f2x(s1) - f2x(s0), TRACK_H);
            }
            g.strokeStyle = C.targetEdge; g.lineWidth = 2;
            g.strokeRect(f2x(gx.strip) + 1, TRACK_Y + 1, f2x(gx.strip + gx.len) - f2x(gx.strip) - 2, TRACK_H - 2); g.lineWidth = 1;
            g.fillStyle = C.targetEdge; g.fillRect(f2x(s0), BAND_Y + 2, f2x(s1) - f2x(s0), BAND_H - 4);
            // I / O marker flags at the band's ends
            g.fillStyle = "#fff"; g.font = font(9, true);
            g.fillRect(f2x(s0), BAND_Y, 2 * U, BAND_H); g.fillRect(f2x(s1) - 2 * U, BAND_Y, 2 * U, BAND_H);
            g.fillText("I", f2x(s0) + 4 * U, BAND_Y + BAND_H / 2);
            g.textAlign = "right"; g.fillText("O", f2x(s1) - 4 * U, BAND_Y + BAND_H / 2); g.textAlign = "left";
            g.font = font(10);
            g.fillText(p.pure ? `insert ${p.gen - 2}f` : `regenerate ${p.end - p.start + 1}f + ${p.G}f new`, f2x(s0) + 14 * U, BAND_Y + BAND_H / 2);
        } else if (S.drag && S.drag.kind === "band") {
            // nothing yet
        } else {
            g.fillStyle = "#4b5563"; g.font = font(10);
            g.fillText("drag here to mark a range to regenerate", 6, BAND_Y + BAND_H / 2);
        }
        // seam pills
        for (const x of seams) {
            const sx = f2x(x.strip);
            const hot = S.hover && S.hover.kind === "seam" && S.hover.i === x.i;
            g.fillStyle = hot ? C.targetEdge : C.seam;
            g.beginPath(); g.arc(sx, SEAM_Y, (hot ? 8 : 6) * U, 0, Math.PI * 2); g.fill();
            g.fillStyle = "#111"; g.font = font(9, true); g.textAlign = "center";
            g.fillText("✂", sx, SEAM_Y + 1); g.textAlign = "left";
        }
        // playhead
        const px = Math.round(f2x(S.playhead)) + 0.5;
        g.strokeStyle = C.play; g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(px, 0); g.lineTo(px, CANVAS_H); g.stroke(); g.lineWidth = 1;
        g.fillStyle = C.play; g.beginPath(); g.moveTo(px - 5 * U, 0); g.lineTo(px + 5 * U, 0); g.lineTo(px, 7 * U); g.fill();
    }

    function drawSelBar() {
        selBar.innerHTML = "";
        const x = layout()[S.sel];
        if (!x) { selBar.append(el("span", { color: C.dim }, "drag a clip or the ruler to scrub · wheel over the picture: step frames · ⠿ name bar: reorder · lower edges: trim · ✂: bridge options")); return; }
        if (x.e.kind === "gap") {
            selBar.append(el("span", { fontWeight: "bold" }, `gap · ${x.len} frames (${(x.len / S.fr).toFixed(2)}s)`),
                button("−8", "Shorter", () => { x.e.frames = Math.max(1, x.e.frames - 8); save(false); }),
                button("+8", "Longer", () => { x.e.frames += 8; save(false); }),
                button("bridge this gap", "Regenerate the frames marked I / O either side and add the gap's frames (the Loader's extend_bridge)", () => setTarget(defaultGapMarks(gapInfo()))),
                button("remove gap", "Close the gap", () => { S.entries.splice(x.i, 1); S.sel = -1; save(true); }));
            return;
        }
        const inf = S.info[x.e.path];
        const name = el("span", { fontWeight: "bold", maxWidth: "35%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, x.e.path.split(/[\\/]/).pop());
        name.title = inf ? inf.path : x.e.path;
        const facts = inf ? `${inf.frames}f @ ${S.fr} fps · plays ${x.e.enter}-${exitOf(x.e) - 1} · ${inf.width}×${inf.height}${inf.has_audio ? "" : " · no audio"}` : "";
        selBar.append(name, el("span", { color: C.dim }, facts),
            button("cut left", "This clip starts at the playhead", () => cutAt("left")),
            button("cut right", "This clip ends at the playhead", () => cutAt("right")),
            button("split", "Split this clip in two at the playhead", () => cutAt("split")),
            button("uncut", "Play this clip in full again", () => { x.e.enter = 0; x.e.exit = null; save(true); }),
            button("remove", "Take this clip off the strip (the file stays)", () => { S.entries.splice(x.i, 1); S.sel = -1; save(true); }));
    }

    function drawNextBar() {
        nextBar.innerHTML = "";
        const p = plan();
        const lead = el("span", { color: C.targetEdge, fontWeight: "bold" }, "next run");
        if (p.err) { nextBar.append(lead, el("span", { color: C.err }, p.err)); return; }
        if (p.none) { nextBar.append(lead, el("span", { color: C.dim }, "nothing marked - press I and O at the playhead, drag on the purple row, or click ✂ between two clips")); return; }
        const sec = (n) => `${(n / S.fr).toFixed(2)}s`;
        if (p.mode === "replace") {
            const n = p.end - p.start + 1;
            const grew = p.gen - n - 2 * p.k;
            nextBar.append(lead, el("span", {}, `replaces cut frames ${p.start}-${p.end} (${n}f · ${sec(n)})`),
                el("span", { color: C.dim }, "→"),
                el("span", { fontWeight: "bold" }, `generator ${p.gen}f`),
                el("span", { color: C.dim }, `(${grid()}${p.k ? `, ${p.k} context each side` : ""}${grew > 0 ? `, +${grew}f longer` : ""})`));
        } else if (p.pure) {
            nextBar.append(lead, el("span", {}, `inserts new frames at cut frame ${p.join}, nothing replaced`),
                el("span", { color: C.dim }, "→"),
                el("span", { fontWeight: "bold" }, `generator ${p.gen}f`),
                el("span", { color: C.dim }, `(${grid()}; its first and last are the kept frames either side, so ${p.gen - 2} new frames go in for the ${p.G}f gap)`),
                button("mark around it", "Also regenerate some footage either side (I / O) - smoother over a hard cut", () => setTarget(defaultGapMarks(gapInfo()))));
        } else {
            const n = p.end - p.start + 1;
            const longer = p.gen - n - 2 * p.k;
            nextBar.append(lead, el("span", {}, `regenerates cut frames ${p.start}-${p.end} (${n}f, I/O) + ${p.G}f for the gap`),
                el("span", { color: C.dim }, "→"),
                el("span", { fontWeight: "bold" }, `generator ${p.gen}f`),
                el("span", { color: C.dim }, `(${grid()}${p.k ? `, ${p.k} context each side` : ""}; the video gets ${longer}f / ${sec(longer)} longer)`),
                button("pure insert", "Replace nothing: only new frames go into the gap", () => { const gi = gapInfo(); setTarget({ mode: "gap", start: gi.join, end: gi.join - 1 }); }));
        }
        const pic2 = el("span", { color: C.ok, marginLeft: "6px" }, `Picture 2 at ${((p.gen - 1) / h3Clock()).toFixed(2)}s${h3Clock() !== S.fr ? " (H3 24 fps clock)" : ""}`);
        pic2.title = "When the last frame (Picture 2 in a MiniMax reference prompt) appears - the node's end_seconds / picture_timing outputs carry it";
        nextBar.append(pic2, button("go to", "Move the playhead to the mark", () => seekStrip(cutToStrip(Math.max(0, p.start - 12)))));
    }

    function updateOverlay() {
        const t = totals();
        const x = entryAtStrip(S.playhead);
        const cutF = stripToCut(S.playhead);
        overlay.textContent = t.L.length ? `${(S.playhead / S.fr).toFixed(2)}s · strip ${S.playhead} · cut ${cutF}${x && x.e.kind === "clip" ? ` · ${x.e.path.split(/[\\/]/).pop()} #${x.e.enter + (S.playhead - x.strip)}` : ""}${S.mode === "full" ? " · full" : ""}` : "";
        gapCover.style.display = x && x.e.kind === "gap" && S.mode === "quick" ? "flex" : "none";
    }

    // Strip span of the marked frames: a gap target's span always includes the gap.
    function markSpan(p) {
        if (p.mode === "gap" && p.pure) return [p.gx.strip, p.gx.strip + p.gx.len];
        return [cutToStrip(p.start), cutToStrip(p.end) + 1];
    }
    function gapInfo() {
        const x = layout().find(y => y.e.kind === "gap");
        return x ? { x, join: cutBefore(x.i) } : null;
    }
    // Cut frame for a marker at strip frame f: inside the gap, I sits on the gap's far
    // edge (nothing marked before it) and O on its near edge (nothing marked after it).
    function markerCut(f, side, gi) {
        f = Math.floor(f);
        const inGap = f >= gi.x.strip && f < gi.x.strip + gi.x.len;
        if (inGap) return side === "start" ? gi.join : gi.join - 1;
        const c = stripToCut(f);
        return side === "start" ? Math.min(c, gi.join) : Math.max(c, gi.join - 1);
    }
    function gapMarks() {
        const gi = gapInfo();
        const p = plan();
        if (p.mode === "gap") return { s: p.start, e: p.end, gi };
        return { s: gi.join, e: gi.join - 1, gi };
    }
    function defaultGapMarks(gi) {
        const each = Math.max(4, Math.round(S.fr / 4));
        const cut = totals().cut;
        return { mode: "gap", start: Math.max(1, gi.join - each), end: Math.min(cut - 2, gi.join + each - 1) };
    }

    // ------------------------------------------------------------ editing
    function markIn() {
        const gi = gapInfo();
        if (gi) { const m = gapMarks(); setTarget({ mode: "gap", start: markerCut(S.playhead, "start", gi), end: m.e }); return; }
        const c = stripToCut(S.playhead);
        const t = S.target.mode === "replace" ? S.target : { mode: "replace", start: c, end: c + 11 };
        setTarget({ mode: "replace", start: c, end: Math.max(c, t.end >= c ? t.end : c + 11) });
    }
    function markOut() {
        const gi = gapInfo();
        if (gi) { const m = gapMarks(); setTarget({ mode: "gap", start: m.s, end: markerCut(S.playhead, "end", gi) }); return; }
        const c = stripToCut(S.playhead);
        const t = S.target.mode === "replace" ? S.target : { mode: "replace", start: Math.max(1, c - 11), end: c };
        setTarget({ mode: "replace", start: Math.min(t.start <= c ? t.start : c - 11, c), end: c });
    }
    function cutAt(side) {
        const x = layout()[S.sel];
        if (!x || x.e.kind !== "clip") return;
        const off = Math.round(S.playhead) - x.strip;
        if (off <= 0 || off >= x.len) { toast("move the playhead inside the selected clip first", "warn"); return; }
        const at = x.e.enter + off;
        if (side === "left") x.e.enter = at;
        else if (side === "right") x.e.exit = at;
        else S.entries.splice(x.i + 1, 0, { kind: "clip", path: x.e.path, enter: at, exit: x.e.exit }), x.e.exit = at;
        save(true);
    }
    function openGapAt(i) {
        S.entries.splice(i, 0, { kind: "gap", frames: Math.max(9, Math.round(S.fr)) });
        S.sel = i;
        save(true);
        setTarget(defaultGapMarks(gapInfo()));
        toast("gap opened - I / O mark the footage either side to regenerate with it; drag them on the purple row", "ok");
    }
    function bridgeSeam(i, each) {
        const x = layout()[i];
        const c = x.cut;
        setTarget({ mode: "replace", start: c - each, end: c + each - 1 });
        toast(`marked ${each} frames either side of the cut - drag the purple row's edges to adjust`, "ok");
    }

    async function addFiles(files, at) {
        let i = at;
        for (const f of files) {
            try {
                toast(`uploading ${f.name}…`);
                const name = await uploadFile(f);
                S.entries.splice(i++, 0, { kind: "clip", path: name, enter: 0, exit: null });
            } catch (e) { toast(`${f.name}: ${e}`, "err"); }
        }
        save(true);
        await probeAll();
        requestFit();
        toast(`added ${files.length} clip(s)`, "ok");
    }
    async function addPath(path, at) {
        S.entries.splice(at ?? S.entries.length, 0, { kind: "clip", path, enter: 0, exit: null });
        save(true);
        await probeAll();
        requestFit();
    }

    function popup(ev, items) {
        closePopup();
        const m = el("div", { position: "fixed", left: `${ev.clientX}px`, top: `${ev.clientY}px`, zIndex: 10000,
            background: "#1f2330", border: "1px solid #3b4252", borderRadius: "6px", padding: "4px",
            boxShadow: "0 6px 20px rgba(0,0,0,.5)", fontFamily: "sans-serif", fontSize: "12px", color: C.text,
            maxHeight: "50vh", overflowY: "auto", minWidth: "220px" });
        for (const it of items) {
            if (it.sep) { m.append(el("div", { borderTop: "1px solid #333", margin: "3px 0" })); continue; }
            if (it.label && !it.run) { m.append(el("div", { color: C.dim, padding: "3px 8px", fontSize: "11px" }, it.label)); continue; }
            const row = el("div", { padding: "5px 8px", cursor: "pointer", borderRadius: "4px", whiteSpace: "nowrap" }, it.text);
            row.title = it.title || "";
            row.onmouseenter = () => row.style.background = "#2e3445";
            row.onmouseleave = () => row.style.background = "";
            row.onclick = () => { closePopup(); it.run(); };
            m.append(row);
        }
        document.body.append(m);
        const r = m.getBoundingClientRect();
        if (r.right > innerWidth) m.style.left = `${innerWidth - r.width - 8}px`;
        if (r.bottom > innerHeight) m.style.top = `${innerHeight - r.height - 8}px`;
        S.popup = m;
        setTimeout(() => document.addEventListener("pointerdown", onOutside, true), 0);
    }
    function onOutside(e) { if (S.popup && !S.popup.contains(e.target)) closePopup(); }
    function closePopup() { if (S.popup) S.popup.remove(); S.popup = null; document.removeEventListener("pointerdown", onOutside, true); }

    async function openAddMenu(ev) {
        let files = [];
        try { files = (await (await api.fetchApi("/seamstitch/timeline/list")).json()).files || []; } catch { }
        popup(ev, [
            { text: "⬆ upload from disk…", run: () => fileInput.click() },
            { sep: true },
            { label: files.length ? "input folder (newest first)" : "no videos in the input folder" },
            ...files.slice(0, 60).map(f => ({ text: f, run: () => addPath(f) })),
        ]);
    }

    function seamMenu(ev, i) {
        const each = Math.max(4, Math.round(S.fr / 4));
        popup(ev, [
            { label: "the cut between these two clips" },
            { text: `bridge this cut: regenerate ${each} frames each side`, title: "Replace mode across the join - the S5-proven way to smooth a hard cut", run: () => bridgeSeam(i, each) },
            { text: `bridge this cut: regenerate ${each * 2} frames each side`, run: () => bridgeSeam(i, each * 2) },
            { text: "open a gap here (insert new frames)", title: "Insert mode: nothing is replaced, new frames go in between", run: () => openGapAt(i) },
            { sep: true },
            { text: "jump here", run: () => seekStrip(layout()[i].strip) },
        ]);
    }

    function openTextEditor() {
        const box = el("div", { position: "fixed", inset: "0", background: "rgba(0,0,0,.55)", zIndex: 10000, display: "flex", alignItems: "center", justifyContent: "center" });
        const card = el("div", { background: "#1f2330", border: "1px solid #3b4252", borderRadius: "8px", padding: "12px", width: "min(720px, 90vw)", color: C.text, fontFamily: "sans-serif", fontSize: "12px" });
        const ta = el("textarea", { width: "100%", height: "260px", background: "#111", color: C.text, border: "1px solid #444", fontFamily: "monospace", fontSize: "12px", boxSizing: "border-box" });
        ta.value = seqW.value;
        const help = el("div", { color: C.dim, margin: "6px 0" }, "One clip per line: path, or 'path @ enter..exit' (frames at the node's frame rate, exit not included). '~ N' is an N-frame gap. Lines starting with # are ignored.");
        const row = el("div", { display: "flex", gap: "6px", justifyContent: "flex-end" });
        row.append(button("cancel", "", () => box.remove()), button("apply", "", async () => {
            S.entries = parseSequence(ta.value); box.remove(); save(true); await probeAll(); requestFit();
        }));
        card.append(el("div", { fontWeight: "bold", marginBottom: "6px" }, "SeamStitch Timeline - strip as text"), ta, help, row);
        box.append(card);
        box.onpointerdown = (e) => { if (e.target === box) box.remove(); };
        document.body.append(box);
        ta.focus();
    }

    // ------------------------------------------------------------ zoom / fit
    function zoomBy(k, anchorX) {
        const ax = anchorX ?? canvas.clientWidth / 2;
        const f = x2f(ax);
        S.fitted = false;
        S.pxPerFrame = Math.min(40, Math.max(0.05, S.pxPerFrame * k));
        S.scroll = Math.max(0, f * S.pxPerFrame - ax);
        drawCanvas();
    }
    // Fit once the canvas has a real width: on workflow load the widget is laid out
    // after reload() runs, and fitting a 0-px canvas squashed the strip to the left.
    function requestFit() { S.needFit = true; if (canvas.clientWidth > 100) fit(); }
    function fit() {
        S.needFit = false;
        S.fitted = true;          // stays fitted through node resizes until the user zooms
        const t = totals();
        const w = Math.max(100, canvas.clientWidth - 10);
        if (t.strip > 0) S.pxPerFrame = Math.min(40, Math.max(0.05, w / t.strip));
        S.scroll = 0;
        drawCanvas();
    }

    // ------------------------------------------------------------ preview
    let switching = false;
    function seekStrip(f, keepPlaying) {
        const t = totals();
        S.playhead = Math.max(0, Math.min(Math.round(f), Math.max(0, t.strip - 1)));
        const x = entryAtStrip(S.playhead);
        if (S.mode === "full" && S.full) {
            const c = stripToCut(S.playhead);
            const fp = S.full.play_path || S.full.path;
            if (!video.src.includes(encodeURIComponent(fp))) video.src = viewURL(fp);
            video.currentTime = (c + 0.5) / fullFr();
        } else if (x && x.e.kind === "clip") {
            const inf = S.info[x.e.path];
            if (inf) {
                const want = viewURL(inf.path);
                const localF = x.e.enter + (S.playhead - x.strip);
                const tt = inf.base_time + (localF + 0.5) / S.fr;
                if (video.dataset.src !== want) {
                    switching = true;
                    video.dataset.src = want;
                    video.src = want;
                    video.addEventListener("loadedmetadata", () => { video.currentTime = tt; switching = false; if (keepPlaying || S.playing) video.play().catch(() => { }); }, { once: true });
                } else video.currentTime = tt;
            }
        } else if (x && x.e.kind === "gap" && !keepPlaying) video.pause();
        updateOverlay();
        drawCanvas();
    }

    async function setMode(m) {
        if (m === "full") {
            const p0 = S.playhead;
            toast("building the cut…");
            bMode.disabled = true;
            try {
                const r = await api.fetchApi("/seamstitch/timeline/build", { method: "POST", body: JSON.stringify({
                    sequence: seqW.value, frame_rate: frW ? frW.value : 0, crf: (W("assemble_crf") || {}).value ?? 12,
                    fit: (W("mismatch_fit") || {}).value || "crop", codec: (W("cut_codec") || {}).value || "lossless (ffv1)",
                    conform: !!(conformW && conformW.value) }) });
                const j = await r.json();
                if (!r.ok) throw new Error(j.error);
                S.full = j;
                S.mode = "full";
                video.dataset.src = "";
                video.src = viewURL(j.play_path || j.path);
                video.addEventListener("loadedmetadata", () => seekStrip(p0), { once: true });
                toast(j.passthrough ? "full: the source itself (one untouched clip - no re-encode)"
                    : `full: built the cut, ${j.frames} frames${j.frame_rate !== j.source_frame_rate ? ` conformed to ${j.frame_rate} fps (audio slowed)` : ""} - this is the file Recombine splices`, "ok");
            } catch (e) { toast(`build failed: ${e.message || e}`, "err"); S.mode = "quick"; }
            bMode.disabled = false;
        } else {
            S.mode = "quick";
            video.dataset.src = "";
            seekStrip(S.playhead);
        }
        refresh();
    }

    function togglePlay() {
        if (S.playing) { S.playing = false; video.pause(); clearInterval(S.gapTimer); bPlay.textContent = "▶"; return; }
        S.playing = true; bPlay.textContent = "❚❚";
        const x = entryAtStrip(S.playhead);
        if (S.mode === "quick" && x && x.e.kind === "gap") { playGap(x); return; }
        if (S.playhead >= totals().strip - 1) seekStrip(0, true);
        video.play().catch(() => { });
    }
    function playGap(x) {
        video.pause();
        clearInterval(S.gapTimer);
        const t0 = performance.now(), f0 = S.playhead;
        S.gapTimer = setInterval(() => {
            if (!S.playing) { clearInterval(S.gapTimer); return; }
            const f = f0 + Math.floor((performance.now() - t0) / 1000 * S.fr);
            if (f >= x.strip + x.len) { clearInterval(S.gapTimer); seekStrip(x.strip + x.len, true); video.play().catch(() => { }); return; }
            S.playhead = f; updateOverlay(); drawCanvas();
        }, 1000 / 30);
    }
    // Playhead follows the video. requestVideoFrameCallback is exact but only fires for
    // frames actually presented (none while the node is off screen), so timeupdate
    // drives it too.
    function tick() {
        if (!switching && S.playing && !video.paused) {
            if (S.mode === "full" && S.full) {
                const c = Math.floor(video.currentTime * fullFr());
                S.playhead = cutToStrip(Math.min(c, totals().cut - 1));
            } else {
                const x = entryAtStrip(S.playhead);
                if (x && x.e.kind === "clip") {
                    const inf = S.info[x.e.path];
                    const localF = Math.floor((video.currentTime - (inf?.base_time || 0)) * S.fr);
                    const f = x.strip + (localF - x.e.enter);
                    if (localF >= exitOf(x.e) || f >= x.strip + x.len) {
                        const next = layout()[x.i + 1];
                        if (!next) { S.playing = false; video.pause(); bPlay.textContent = "▶"; }
                        else if (next.e.kind === "gap") { S.playhead = next.strip; playGap(next); }
                        else seekStrip(next.strip, true);
                    } else if (f >= x.strip) S.playhead = f;
                }
            }
            updateOverlay();
            drawCanvas();
        }
    }
    function onFrame() { tick(); video.requestVideoFrameCallback(onFrame); }
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onFrame);
    video.addEventListener("timeupdate", tick);
    video.addEventListener("ended", () => {
        if (S.mode === "full") { S.playing = false; bPlay.textContent = "▶"; return; }
        const x = entryAtStrip(S.playhead), next = x && layout()[x.i + 1];
        if (S.playing && next) { if (next.e.kind === "gap") { S.playhead = next.strip; playGap(next); } else seekStrip(next.strip, true); }
        else { S.playing = false; bPlay.textContent = "▶"; }
    });

    // ------------------------------------------------------------ canvas input
    function local(ev) {
        const r = canvas.getBoundingClientRect();
        return { x: (ev.clientX - r.left) * (canvas.clientWidth / r.width), y: (ev.clientY - r.top) * (CANVAS_H / r.height) };
    }
    function hit(p) {
        const t = totals();
        if (p.y < RULER_H) return { kind: "ruler" };
        if (p.y >= BAND_Y) {
            const pl = plan();
            if (pl.mode === "replace" || pl.mode === "gap") {
                const [a, b] = markSpan(pl);
                const x0 = f2x(a), x1 = f2x(b);
                if (Math.abs(p.x - x0) < EDGE_PX) return { kind: "bandEdge", side: "start" };
                if (Math.abs(p.x - x1) < EDGE_PX) return { kind: "bandEdge", side: "end" };
                if (p.x > x0 && p.x < x1) return { kind: "bandBody" };
            }
            return { kind: "band" };
        }
        for (const x of t.L) {
            if (x.i > 0 && x.e.kind === "clip" && t.L[x.i - 1].e.kind === "clip" && Math.abs(p.x - f2x(x.strip)) < 8 &&
                p.y < GRIP_Y0) return { kind: "seam", i: x.i };
        }
        // Each x is inside exactly one block's [x0, x1) (the last block also owns a few
        // pixels past its end), so at a join the side of the line decides which clip trims.
        for (const x of t.L) {
            const x0 = f2x(x.strip), x1 = f2x(x.strip + x.len);
            const last = x.i === t.L.length - 1;
            if (p.x >= x0 && (p.x < x1 || (last && p.x < x1 + EDGE_PX)) && p.y >= TRACK_Y && p.y <= TRACK_Y + TRACK_H) {
                if (p.y >= GRIP_Y0 && x1 - p.x < EDGE_PX) return { kind: "edge", side: "right", i: x.i };
                if (p.y >= GRIP_Y0 && p.x - x0 < EDGE_PX && x.e.kind === "clip") return { kind: "edge", side: "left", i: x.i };
                if (p.y < LABEL_Y1) return { kind: "label", i: x.i };
                return { kind: "body", i: x.i };
            }
        }
        return { kind: "empty" };
    }
    canvas.addEventListener("pointermove", (ev) => {
        const p = local(ev);
        if (S.drag) { onDrag(p, ev); return; }
        const h = hit(p);
        S.hover = h;
        canvas.style.cursor = h.kind === "edge" || h.kind === "bandEdge" ? "ew-resize" : h.kind === "seam" ? "pointer" :
            h.kind === "label" ? "grab" : h.kind === "body" || h.kind === "ruler" ? "col-resize" : h.kind === "band" ? "crosshair" : "default";
        drawCanvas();
    });
    canvas.addEventListener("pointerleave", () => { if (!S.drag) { S.hover = null; drawCanvas(); } });
    canvas.addEventListener("pointerdown", (ev) => {
        ev.stopPropagation();
        ev.preventDefault();
        root.focus();
        const p = local(ev), h = hit(p);
        try { canvas.setPointerCapture(ev.pointerId); } catch { }   // pen/touch/synthetic pointers may refuse
        const L = layout();
        if (h.kind === "ruler") { S.drag = { kind: "scrub" }; seekStrip(x2f(p.x)); return; }
        if (h.kind === "seam") { seamMenu(ev, h.i); return; }
        if (h.kind === "edge") {
            const x = L[h.i];
            S.sel = h.i;
            S.drag = { kind: "trim", i: h.i, side: h.side, startX: p.x, enter: x.e.enter, exit: x.e.kind === "clip" ? exitOf(x.e) : null, frames: x.e.frames, moved: false };
            refresh();
            return;
        }
        if (h.kind === "label") {           // the name bar: drag to reorder
            S.sel = h.i;
            S.drag = { kind: "move", i: h.i, startX: p.x, moved: false };
            refresh();
            return;
        }
        if (h.kind === "body") {            // the clip itself: drag to scrub, like the Loader's bar
            S.sel = h.i;
            S.drag = { kind: "scrub" };
            seekStrip(x2f(p.x));
            refresh();
            return;
        }
        if (h.kind === "bandEdge") { S.drag = { kind: "bandEdge", side: h.side }; return; }
        if (h.kind === "band" || h.kind === "bandBody") {
            const gi = gapInfo();
            if (gi) {       // with a gap, a click on the row sets the nearer marker
                const f = x2f(p.x), m = gapMarks();
                if (f < gi.x.strip) setTarget({ mode: "gap", start: markerCut(f, "start", gi), end: m.e });
                else if (f >= gi.x.strip + gi.x.len) setTarget({ mode: "gap", start: m.s, end: markerCut(f, "end", gi) });
                S.drag = { kind: "bandEdge", side: f < gi.x.strip ? "start" : "end" };
                return;
            }
            const c = stripToCut(x2f(p.x));
            S.drag = { kind: "band", anchor: c };
            setTarget({ mode: "replace", start: c, end: c });
            return;
        }
        S.sel = -1;
        seekStrip(x2f(p.x));
        refresh();
    });
    function onDrag(p, ev) {
        const d = S.drag;
        const f = x2f(p.x);
        if (d.kind === "scrub") { seekStrip(f); return; }
        if (d.kind === "trim") {
            const x = layout()[d.i];
            const df = Math.round((p.x - d.startX) / S.pxPerFrame);
            if (!df && !d.moved) return;
            d.moved = true;
            if (x.e.kind === "gap") { x.e.frames = Math.max(1, d.frames + df); }
            else if (d.side === "left") x.e.enter = Math.max(0, Math.min(d.exit - 1, d.enter + df));
            else { const tot = clipFrames(x.e) ?? d.exit; const ne = Math.max(x.e.enter + 1, Math.min(tot, d.exit + df)); x.e.exit = ne >= tot ? null : ne; }
            seqW.value = formatSequence(S.entries);
            if (x.e.kind === "clip") seekStrip(d.side === "left" ? x.strip : x.strip + played(x.e) - 1);
            drawCanvas(); drawSelBar();
            return;
        }
        if (d.kind === "move") {
            if (Math.abs(p.x - d.startX) < 5 && !d.moved) return;
            d.moved = true;
            canvas.style.cursor = "grabbing";
            const L = layout();
            let to = L.findIndex(x => f < x.strip + x.len / 2);
            if (to < 0) to = L.length;
            d.to = to;
            drawCanvas();
            const g = canvas.getContext("2d");
            const ix = to < L.length ? f2x(L[to].strip) : f2x(totals().strip);
            g.fillStyle = C.play; g.fillRect(ix - 1.5 * U, TRACK_Y - 2, 3 * U, TRACK_H + 4);
            return;
        }
        if (d.kind === "bandEdge" && gapInfo()) {
            const gi = gapInfo(), m = gapMarks();
            if (d.side === "start") setTarget({ mode: "gap", start: Math.max(1, markerCut(f, "start", gi)), end: m.e });
            else setTarget({ mode: "gap", start: m.s, end: Math.min(totals().cut - 2, markerCut(f, "end", gi)) });
            seekStrip(Math.max(0, Math.min(totals().strip - 1, Math.round(f))));
            return;
        }
        if (d.kind === "band" || d.kind === "bandEdge") {
            const c = Math.max(0, Math.min(totals().cut - 1, stripToCut(f)));
            if (d.kind === "band") setTarget({ mode: "replace", start: Math.min(d.anchor, c), end: Math.max(d.anchor, c) });
            else if (d.side === "start") setTarget({ mode: "replace", start: Math.min(c, S.target.end), end: S.target.end });
            else setTarget({ mode: "replace", start: S.target.start, end: Math.max(c, S.target.start) });
            seekStrip(cutToStrip(c));
        }
    }
    canvas.addEventListener("pointerup", (ev) => {
        const d = S.drag;
        S.drag = null;
        try { canvas.releasePointerCapture(ev.pointerId); } catch { }
        if (!d) return;
        if (d.kind === "trim" && d.moved) save(true);
        if (d.kind === "move" && d.moved && d.to != null) {
            const [e] = S.entries.splice(d.i, 1);
            const to = d.to > d.i ? d.to - 1 : d.to;
            S.entries.splice(to, 0, e);
            S.sel = to;
            if (to !== d.i) save(true); else refresh();
        }
        if (d.kind === "band" || d.kind === "bandEdge") refresh();
        canvas.style.cursor = "default";
    });
    canvas.addEventListener("dblclick", (ev) => {
        const h = hit(local(ev));
        if ((h.kind === "body" || h.kind === "label") && layout()[h.i].e.kind === "clip") { S.sel = h.i; cutAt("split"); }
    });
    canvas.addEventListener("wheel", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const p = local(ev);
        if (p.y < RULER_H || ev.ctrlKey) zoomBy(ev.deltaY < 0 ? 1.2 : 1 / 1.2, p.x);
        else { S.scroll = Math.max(0, S.scroll + (ev.deltaX || ev.deltaY)); drawCanvas(); }
    }, { passive: false });

    // Wheel over the picture steps frames (shift: 10), like a jog wheel.
    videoBox.addEventListener("wheel", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        seekStrip(S.playhead + (ev.deltaY > 0 ? 1 : -1) * (ev.shiftKey ? 10 : 1));
    }, { passive: false });

    // keyboard, only while the pointer is over the widget
    let over = false;
    root.addEventListener("pointerenter", () => over = true);
    root.addEventListener("pointerleave", () => over = false);
    const onKey = (ev) => {
        if (!over || /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "")) return;
        const k = ev.key;
        const handled = () => { ev.preventDefault(); ev.stopPropagation(); };
        if (k === " ") { handled(); togglePlay(); }
        else if (k === "i" || k === "I") { handled(); markIn(); }
        else if (k === "o" || k === "O") { handled(); markOut(); }
        else if (k === "ArrowLeft") { handled(); seekStrip(S.playhead - (ev.shiftKey ? 10 : 1)); }
        else if (k === "ArrowRight") { handled(); seekStrip(S.playhead + (ev.shiftKey ? 10 : 1)); }
        else if (k === "Home") { handled(); seekStrip(0); }
        else if (k === "End") { handled(); seekStrip(totals().strip - 1); }
    };
    window.addEventListener("keydown", onKey, true);

    // drag and drop videos from the desktop
    root.addEventListener("dragover", (ev) => { if (ev.dataTransfer?.types?.includes("Files")) { ev.preventDefault(); ev.stopPropagation(); dropHint.style.display = "flex"; } });
    root.addEventListener("dragleave", () => dropHint.style.display = "none");
    root.addEventListener("drop", async (ev) => {
        dropHint.style.display = "none";
        const files = [...(ev.dataTransfer?.files || [])].filter(f => /^video\//.test(f.type) || /\.(mp4|webm|mkv|mov|m4v|avi)$/i.test(f.name));
        if (!files.length) return;
        ev.preventDefault(); ev.stopPropagation();
        const r = canvas.getBoundingClientRect();
        let at = S.entries.length;
        if (ev.clientY >= r.top && ev.clientY <= r.bottom) {
            const f = x2f((ev.clientX - r.left) * (canvas.clientWidth / r.width));
            const L = layout(); const k = L.findIndex(x => f < x.strip + x.len / 2);
            if (k >= 0) at = k;
        }
        await addFiles(files, at);
    });

    for (const w of [frW, gridW, ctxW, extW, conformW]) {
        if (!w) continue;
        const cb = w.callback;
        w.callback = function () {
            const r = cb ? cb.apply(this, arguments) : undefined;
            if (w === frW) { probeAll(); S.full = null; if (S.mode === "full") setMode("quick"); }
            else if (w === conformW) { S.full = null; if (S.mode === "full") setMode("quick"); else refresh(); }
            else refresh();
            return r;
        };
    }

    new ResizeObserver(() => {
        if ((S.needFit || S.fitted) && canvas.clientWidth > 100) fit(); else drawCanvas();
    }).observe(canvas);
    const onRemoved = node.onRemoved;
    node.onRemoved = function () { window.removeEventListener("keydown", onKey, true); closePopup(); video.pause(); video.removeAttribute("src"); return onRemoved?.apply(this, arguments); };

    // ------------------------------------------------------------ public
    node.ssTimeline = {
        async reload() {
            S.entries = parseSequence(seqW.value);
            try { S.target = targetW.value ? JSON.parse(targetW.value) : {}; } catch { S.target = {}; }
            S.full = null; S.mode = "quick";
            await probeAll();
            requestFit();
            seekStrip(0);
        },
        async useResult(path) {
            S.entries = [{ kind: "clip", path, enter: 0, exit: null }];
            S.sel = -1;
            setTarget({}, true);
            save(false);
            await probeAll();
            requestFit();
            seekStrip(0);
            toast("the result is now the strip - mark the next thing to fix", "ok");
        },
        state: S,
    };
    setTimeout(() => node.ssTimeline.reload(), 0);
}
