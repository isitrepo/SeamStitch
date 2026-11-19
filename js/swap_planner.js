// SeamStitch Swap Planner: the job's strip. Forked from js/timeline.js (the strip, ruler,
// scrubbing, wheel stepping, zoom/fit, quick/full preview, keyboard and widget repair), which
// stays untouched for the published Timeline.
//
// Three rows on one time axis (the multi-row strip, B2):
//   source   the source with its cut ticks (faint = suggested, solid = confirmed), the splits
//            (drag; pill menu: straight cut / anchored, repair, hand-back) and the chunk blocks,
//            each with its render range (overlap and fill) underneath;
//   mask     the cached source person mask (a mark run, SeamStitch Swap Mask), per range;
//   prompt   each chunk's prompt split into its [Shot n] blocks at the confirmed cuts, with the
//            dialogue lines placed in them.
// The plan lives in the job's plan.json. Every edit is a POST /seamstitch/swap/op, applied to the
// newest revision under the plan lock, and the strip reloads after every reply and every
// seamstitch_swap_plan ping (a take landing, a mask cached), so the strip and the file can't drift.
//
// Queue actions set the Planner's hidden `run` widget and queue the workflow once per item, with
// ComfyUI's partial execution aimed at the output nodes downstream of that run's Planner outputs:
// a render run reaches the render group and its Take, a mark run the mark group's Swap Mask, a
// draft / assemble run only their own nodes. ExecutionBlocker on the unused outputs stays the
// backstop for a plain Queue press. Nothing here picks a take: the user chooses.
//
// Controls follow chanon/comfyui-obvpm-timeline's Timeline (GPL-3.0) as the Timeline does: a strip
// sized by frames, a pill per seam with a menu, a next-run bar, quick (chained) and full preview.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const C = {
    bg: "#16181d", ruler: "#20242c", tick: "#6b7280", text: "#f3f4f6", dim: "#b4bccb", faint: "#4b5563", panel: "#0f1115",
    row: "#1b1e25", rowAlt: "#1e2129",
    pending: "#3a3f4b", unreviewed: "#2f4a7a", chosen: "#3b62a8", sel: "#8fb3ff",
    amber: "#fbbf24", amberFill: "#5c4512", red: "#f87171", redFill: "#5b1f24", green: "#34d399",
    render: "rgba(143,179,255,0.45)", fill: "rgba(251,191,36,0.55)", held: "rgba(248,113,113,0.55)",
    mask: "#2e7d5b", maskOld: "#1f4f3c", prompt: "#2a3142", promptDraft: "#1e3a5f", shot: "#3b4252",
    play: "#f59e0b", cut: "#e5e7eb", cutSug: "rgba(156,163,175,0.55)", draft: "#60a5fa",
    kept: "#22302a", keptHatch: "rgba(167,243,208,0.16)", keptText: "#86efac", trim: "#34d399",
};
const VERDICT = { green: C.green, amber: C.amber, red: C.red };
const TYPE_LABEL = { forward: "F", entry: "E", exit: "X" };
const MIN_PER_209 = 10;     // GPU minutes per 209-frame render through the nodes (B1b: 558-628 s)
const OUT = { render: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18], draft: [13], assemble: [14], mark: [15, 16] };
const HIDDEN = ["job", "source", "run", "ui_state"];     // job and source: picked in the panel's header
const NEW_JOB = "__new__";
const SETTING_OF = { target_render_frames: "target_render", overlap_frames: "overlap", anchor_frames: "anchors",
    floor_frames: "floor", ceiling_frames: "ceiling", conform_to_24fps: "conform_to_24fps" };

// ---------------------------------------------------------------- helpers (as timeline.js)

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

// ComfyUI restores widget values by POSITION (see timeline.js): reset a dropdown holding a value
// it doesn't offer, and a number that isn't one, to the widget's default.
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
        } else if (w.type === "toggle" && typeof w.value !== "boolean") {
            w.value = w.options?.default ?? true;
        }
    }
}

const fileURL = (path) => api.apiURL(`/seamstitch/loader/view?filename=${encodeURIComponent(path)}`);
const joinPath = (dir, rel) => `${dir.replace(/[\\/]+$/, "")}/${rel}`;
const uid = () => (crypto.randomUUID ? crypto.randomUUID().replace(/-/g, "") : `${Date.now()}${Math.random()}`.replace(".", ""));
const randSeed = () => Math.floor(Math.random() * 4294967295);

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

// Quality colours for the flag dots: the flags come from the backend (swap_scores, design §4.5, thresholds
// fitted in B3 against the reviewer's verdicts); this only maps a flag to a colour. Grey = information only (mouth).
const FLAG = { green: C.green, amber: C.amber, red: C.red, grey: "#9ca3af" };
const flagColour = (f) => FLAG[f] || C.faint;
// The tooltips: each flag's numbers, mouth with its face coverage (a hand or a prop over the mouth makes it noisy).
function flagTips(sc) {
    sc = sc || {};
    const mi = sc.mouth_info || {};
    const cuts = Object.entries(sc.cuts || {});
    const halves = (mi.halves || []).map(h => h.mouth ?? "n/a").join(" / ");
    return {
        F: `following: pose IoU ${sc.pose_iou ?? "n/a"}${sc.pose_iou_p10 != null ? `, p10 ${sc.pose_iou_p10}` : ""} (mean >= 0.60 and p10 >= 0.45 green)`,
        C: `cuts: ${cuts.length ? cuts.map(([f, s]) => `${f} ${s}`).join(", ") : "no confirmed cut inside"}`,
        M: sc.mouth != null ? `mouth sync ${sc.mouth} at lag ${mi.lag}, face on ${Math.round(100 * (mi.face || 0))}% of frames`
            + (halves ? ` (halves ${halves})` : "") + " - information only, never a gate"
            : `mouth sync n/a${mi.why ? `: ${mi.why}` : ""}`,
        R: sc.replaced ? `replaced: colour difference inside the person ${sc.replaced.person_diff} (>= 60 green; 30-60 amber: partly replaced, e.g. only the head; < 30 red: not replaced, the source came back)` : "replaced n/a",
        S: sc.scene ? `scene: background ${sc.scene.bg_psnr} dB outside the person (amber under 15: the room was rewritten)` : "scene n/a",
    };
}
const SCORE_TXT = (sc) => {
    sc = sc || {};
    const lost = Object.values(sc.cuts || {}).filter(v => v === "lost").length;
    const n = Object.keys(sc.cuts || {}).length;
    const face = sc.mouth_info?.face;
    return `F ${sc.pose_iou ?? "–"} · C ${n ? `${n - lost}/${n}` : "–"} · M ${sc.mouth ?? "–"}${face != null ? ` (face ${Math.round(100 * face)}%)` : ""}`;
};

// The prompt's [Shot n] blocks and the dialogue in each.
function parseShots(prompt) {
    const text = prompt || "";
    const re = /\[Shot\s*(\d+)\]/gi;
    const marks = [];
    let m;
    while ((m = re.exec(text))) marks.push({ n: +m[1], at: m.index, end: re.lastIndex });
    const blocks = marks.map((mk, i) => {
        const body = text.slice(mk.end, i + 1 < marks.length ? marks[i + 1].at : text.length);
        const stop = body.search(/\n\s*(overall_soundscape|non_diegetic_music|\w+_\w+:)/i);
        const b = (stop > 0 ? body.slice(0, stop) : body).trim();
        const dialogue = [...b.matchAll(/<d>\s*(?:\[[^\]]*\])?([\s\S]*?)<\/d>/gi)].map(x => x[1].trim()).filter(Boolean);
        return { n: mk.n, text: b.replace(/<\/?d>/g, "").replace(/\s+/g, " "), dialogue, at: mk.at };
    });
    return blocks;
}

// ---------------------------------------------------------------- prompt colours
// Who and what each tag stands for, at a glance (B7): Subject 1 yellow, Subject 2 green, Video blue, ...
const PROMPT_TOKENS = /(<Subject 1>|\(S1\))|(<Subject 2>|\(S2\))|(<Subject \d+>|\(S\d+\))|(<Video \d+>)|(<Picture \d+>)|(<Audio \d+>)|(<d>[\s\S]*?<\/d>)|(\[Shot \d+\])|(^[a-z_]+:[ \t]*$)/gm;
const PROMPT_STYLES = ["color:#facc15", "color:#4ade80", "color:#fb923c", "color:#60a5fa", "color:#fb7185", "color:#c084fc",
    "color:#67e8f9", "color:#f3f4f6;font-weight:bold", "color:#9ca3af;font-weight:bold"];
function escapeHtml(t) { return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function colourPrompt(text) {
    let html = "", at = 0;
    for (const m of text.matchAll(PROMPT_TOKENS)) {
        const k = m.slice(1).findIndex(g => g !== undefined);
        html += escapeHtml(text.slice(at, m.index)) + `<span style="${PROMPT_STYLES[k]}">${escapeHtml(m[0])}</span>`;
        at = m.index + m[0].length;
    }
    return html + escapeHtml(text.slice(at)) + "\n";
}
// The dialogue lines of a prompt and the speakers it can give them (the strip's per-line picker).
const LINE_RE = /(<Subject \d+>|\b[a-z][a-z' -]*? voice) \((S\d+)\) says <d>\[[^\]]*\](.*?)<\/d>/g;
function promptLines(prompt) {
    const d = ((prompt || "").split("detailed_description:")[1] || "").split("overall_soundscape:")[0];
    return [...d.matchAll(LINE_RE)].map(m => ({ who: m[1], id: m[2], text: m[3] }));
}
function promptSpeakers(prompt) {
    const defs = (prompt || "").split("summary:")[0];
    return [...[...defs.matchAll(/^(<Subject \d+>) is /gm)].map(m => m[1]), "an off-screen male voice", "an off-screen female voice"];
}

// A textarea drawn over its own coloured copy: the text is transparent, the caret and selection are the textarea's.
function colouredTextarea(style, height) {
    const font = { fontFamily: "monospace", fontSize: style.fontSize || "0.95em", lineHeight: "1.35", padding: "3px 5px",
        boxSizing: "border-box", whiteSpace: "pre-wrap", overflowWrap: "break-word", overflowY: "scroll", margin: "0" };
    const wrap = el("div", { position: "relative", flex: "1 1 auto", minWidth: "300px", minHeight: height, overflow: "hidden",
        background: "#111", border: style.border, borderRadius: "4px" });
    const back = el("div", Object.assign({ position: "absolute", inset: "0", color: "#e5e7eb", pointerEvents: "none", border: "0" }, font));
    const ta = el("textarea", Object.assign({ position: "absolute", inset: "0", width: "100%", height: "100%", background: "transparent",
        color: "transparent", caretColor: "#e5e7eb", border: "0", outline: "none", resize: "none", userSelect: "text" }, font));
    const paint = () => { back.innerHTML = colourPrompt(ta.value); back.scrollTop = ta.scrollTop; };
    ta.addEventListener("input", paint);
    ta.addEventListener("scroll", () => { back.scrollTop = ta.scrollTop; });
    wrap.append(back, ta);
    return { wrap, ta, paint };
}

// The prompt in a large editor over the page: the same text, saved back through the panel's own save.
function openLargeEditor(src, onDone) {
    const shade = el("div", { position: "fixed", inset: "0", background: "rgba(0,0,0,0.6)", zIndex: "10000",
        display: "flex", alignItems: "center", justifyContent: "center" });
    const pane = el("div", { width: "85vw", height: "85vh", display: "flex", flexDirection: "column", gap: "6px",
        background: "#0f1115", border: "1px solid #3b4252", borderRadius: "8px", padding: "10px", boxSizing: "border-box" });
    const head = el("div", { display: "flex", alignItems: "center", gap: "8px", color: "#b4bccb" },
        "The chunk's prompt (Ctrl+Enter or close saves it; Esc closes too)");
    const ta = el("textarea", { flex: "1 1 auto", width: "100%", boxSizing: "border-box", resize: "none",
        background: "#111", color: "#e5e7eb", border: "1px solid #444", borderRadius: "4px", padding: "8px",
        fontFamily: "monospace", fontSize: "14px", lineHeight: "1.45", whiteSpace: "pre-wrap" });
    ta.value = src.value;
    const close = () => {
        if (ta.value !== src.value) { src.value = ta.value; onDone(); }
        shade.remove();
    };
    const done = el("button", { marginLeft: "auto", padding: "0.2em 0.9em", cursor: "pointer" }, "close");
    done.onclick = close;
    head.append(done);
    ta.onkeydown = (e) => {
        e.stopPropagation();
        if (e.key === "Escape" || (e.key === "Enter" && (e.ctrlKey || e.metaKey))) { e.preventDefault(); close(); }
    };
    shade.onpointerdown = (e) => { if (e.target === shade) close(); };
    pane.append(head, ta);
    shade.append(pane);
    document.body.append(shade);
    ta.focus();
}

// ---------------------------------------------------------------- the node

app.registerExtension({
    name: "SeamStitch.SwapPlanner",
    setup() {
        api.addEventListener("seamstitch_swap_plan", (e) => {
            const job = e.detail?.job;
            for (const n of app.graph?._nodes || []) if (n.ssPlanner && n.ssPlanner.job() === job) n.ssPlanner.reload();
        });
        const fwd = (type) => api.addEventListener(type, (e) => {
            for (const n of app.graph?._nodes || []) n.ssPlanner?.onQueueEvent(type, e.detail);
        });
        ["execution_start", "executing", "progress", "execution_error", "execution_interrupted", "execution_success",
            "execution_cached", "status"].forEach(fwd);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SeamStitchSwapPlanner") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            buildPlanner(this);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            repairWidgetValues(this);
            if (this.ssPlanner) setTimeout(() => this.ssPlanner.restore(), 0);
            return r;
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (msg) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (msg?.seamstitch_swap_plan?.[0] && this.ssPlanner) this.ssPlanner.reload();
            return r;
        };
    },
});

function buildPlanner(node) {
    const W = (n) => node.widgets.find(w => w.name === n);
    const jobW = W("job"), srcW = W("source"), runW = W("run"), uiW = W("ui_state"), conformW = W("conform_to_24fps");
    HIDDEN.forEach(n => hideWidget(W(n)));

    // ------------------------------------------------------------ state
    const S = {
        view: null, plan: null, joins: [], status: [], jobDir: "", fr: 25, N: 0, base: 0,
        playhead: 0, pxPerFrame: 1, scroll: 0, sel: null, mode: "quick", playing: false,
        hover: null, drag: null, popup: null, queued: {}, running: null, progress: null,
        promptDirty: false, loading: 0, fitted: true, needFit: true,
        layout: { hide: {} },        // the sections' collapsed state and dragged heights, kept in ui_state
    };

    // ------------------------------------------------------------ geometry (scales with the node)
    let U = 1, RULER_H, SRC_Y, PILL_Y, BLOCK_Y, BLOCK_H, RB_Y, RB_H, MASK_Y, MASK_H, PR_Y, PR_H, CANVAS_H, EDGE_PX;
    function setScale(u) {
        U = u;
        RULER_H = Math.round(20 * U);
        SRC_Y = RULER_H + Math.round(2 * U);
        PILL_Y = SRC_Y + Math.round(8 * U);
        BLOCK_Y = SRC_Y + Math.round(17 * U); BLOCK_H = Math.round(40 * U);
        RB_Y = BLOCK_Y + BLOCK_H + Math.round(2 * U); RB_H = Math.round(4 * U);       // two render-band lanes
        MASK_Y = RB_Y + 2 * RB_H + Math.round(6 * U); MASK_H = Math.round(16 * U);
        PR_Y = MASK_Y + MASK_H + Math.round(4 * U); PR_H = Math.round(42 * U);
        CANVAS_H = PR_Y + PR_H + 2;
        EDGE_PX = Math.round(6 * U);
    }
    setScale(1);
    const font = (px, bold) => `${bold ? "bold " : ""}${Math.round(px * U)}px sans-serif`;

    // ------------------------------------------------------------ DOM
    const root = el("div", {
        display: "flex", flexDirection: "column", gap: "4px", width: "100%", height: "100%",
        boxSizing: "border-box", fontFamily: "sans-serif", fontSize: "11px", color: C.text,
        userSelect: "none", outline: "none", overflow: "hidden",
        background: C.panel, padding: "5px", borderRadius: "6px",      // the node's grey made the text hard to read
    });
    root.tabIndex = 0;

    const videoBox = el("div", { position: "relative", flex: "1 1 auto", minHeight: "140px", background: "#000",
        borderRadius: "4px", overflow: "hidden" });
    const video = el("video", { width: "100%", height: "100%", objectFit: "contain", display: "block" });
    video.playsInline = true;
    video.preload = "auto";
    const overlay = el("div", { position: "absolute", left: "6px", top: "4px", fontSize: "1em", fontWeight: "bold",
        color: C.text, textShadow: "0 0 3px #000", pointerEvents: "none" });
    const cover = el("div", { position: "absolute", inset: "0", background: "repeating-linear-gradient(45deg,#111 0 10px,#1b1b1b 10px 20px)",
        display: "none", alignItems: "center", justifyContent: "center", color: C.dim, fontSize: "1.2em" }, "");
    const dropHint = el("div", { position: "absolute", inset: "0", display: "none", alignItems: "center", justifyContent: "center",
        background: "rgba(59,98,168,0.35)", border: "2px dashed #8fb3ff", color: "#fff", fontSize: "14px" }, "drop the source video");
    const emptyHint = el("div", { position: "absolute", inset: "0", display: "flex", flexDirection: "column", gap: "10px",
        alignItems: "center", justifyContent: "center", color: C.dim, fontSize: "1.1em", textAlign: "center", padding: "10px" });
    const emptyText = el("div", { whiteSpace: "pre-line" });
    emptyHint.append(emptyText, button("load video ▾", "From the input folder, or upload one (or drop a video here)", (ev) => openSourceMenu(ev)));
    const zoneCv = el("canvas", { position: "absolute", left: "0", top: "0", width: "100%", height: "100%", pointerEvents: "none" });
    videoBox.append(video, cover, emptyHint, zoneCv, overlay, dropHint);

    const bar1 = el("div", { display: "flex", gap: "4px", alignItems: "center", flexWrap: "wrap", flexShrink: "0" });
    const bPlay = button("▶", "Play / pause (space)", () => togglePlay());
    const bMode = button("quick", "", (ev) => openViewMenu(ev));
    const bZoomOut = button("−", "Zoom out (wheel over the ruler)", () => zoomBy(1 / 1.4));
    const bZoomIn = button("+", "Zoom in (wheel over the ruler)", () => zoomBy(1.4));
    const bFit = button("fit", "Fit the whole source into the node", () => fit());
    const bDetect = button("detect + plan", "Find the cuts (ffmpeg scene > 0.15) and place the splits from them in one step: the cuts come in confirmed, the auto splits (209-frame renders, every 197, nudged off the cuts) follow. Then review: delete a wrong cut, drag or re-mode a split. Runs by itself when a video is loaded.", () => detectPlan());
    const bCuts = button("cuts ▾", "Cut detection only, confirm suggestions, re-place the auto splits", (ev) => openCutsMenu(ev));
    const bClear = button("clear all", "Start fresh: remove every cut, split, kept stretch and prompt (one chunk over the whole video). Takes stay on disk (listed under removed chunks) and the cached masks stay. Then 'detect + plan' starts over.", () => clearAll());
    const bUndo = button("↶ undo", "Undo the last edit made in this panel (a split or cut moved, a mode or take changed, a prompt edited...). Renders made since stay. Keeps the last 50 edits.", () => undoLast());
    const bDraft = button("draft prompts", "Queue a draft run: SeamStitch Swap Draft Prompts fills every empty prompt (others into their draft field)", () => queueDraft());
    const bMark = button("mark ▾", "Track and cache the source person mask (SAM3) up front, so it can be checked on the mask row before any render", (ev) => openMarkMenu(ev));
    const bPending = button("render pending", "Queue every chunk without a usable take, left to right (pins chain at execution)", () => renderPending());
    const bAssemble = button("assemble ▾", "Join the effective takes over the original audio", (ev) => openAssembleMenu(ev));
    const status = el("span", { marginLeft: "auto", color: C.dim, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "40%" });
    const gap = () => el("span", { width: "6px" });
    bar1.append(bPlay, bMode, gap(), bZoomOut, bZoomIn, bFit, gap(), bDetect, bCuts, bClear, bUndo, gap(),
        bDraft, bMark, bPending, bAssemble, status);

    // The job and its video, picked rather than typed (the job / source widgets stay, hidden, so a
    // saved workflow keeps its values by position).
    const head = el("div", { display: "flex", gap: "6px", alignItems: "center", flexShrink: "0", flexWrap: "wrap",
        padding: "3px 6px", background: "#1b2230", border: "1px solid #2b3446", borderRadius: "4px" });
    const jobSel = el("select", { background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px",
        padding: "0.15em 0.3em", fontSize: "1em", maxWidth: "22em" });
    jobSel.onpointerdown = (e) => e.stopPropagation();
    jobSel.onchange = () => pickJob(jobSel.value);
    const videoName = el("span", { fontWeight: "bold", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "30em" });
    const videoFacts = el("span", { color: C.dim, whiteSpace: "nowrap" });
    const bVideo = button("load video ▾", "Choose the job's video: from the input folder or upload one (or drop it on the node). A new job is named after it, and its cuts and splits are found straight away.", (ev) => openSourceMenu(ev));
    head.append(el("span", { color: C.dim }, "job"), jobSel, el("span", { width: "8px" }), el("span", { color: C.dim }, "video"), videoName, videoFacts, bVideo);

    // What the job swaps (one two-way switch: character replace, or background replace = mark everything but the
    // person) and who: SAM3's source text (the mark and the render's own track; the output score keeps "person") and
    // the drafter's main person. A chunk can name its own.
    const targetRow = el("div", { display: "flex", gap: "6px", alignItems: "center", flexShrink: "0", padding: "2px 6px" });
    const targetIn = el("input", { flex: "0 1 22em", background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px", padding: "0.15em 0.3em" });
    targetIn.placeholder = "person (the drafter picks who)";
    targetIn.title = "Who to replace, as a short noun phrase (\"woman in a purple top\"): what SAM3 marks in the source, and who the drafter writes about. Empty: SAM3 marks \"person\" and the drafter picks. Changing it needs a re-mark and a redraft.";
    targetIn.onpointerdown = (e) => e.stopPropagation();
    targetIn.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter") targetIn.blur(); };
    targetIn.onchange = () => op({ op: "set_target", target: targetIn.value });
    const MODE_TIPS = {
        character: "Character replace: the person is replaced by the character on the sheet (Load Character Sheet).",
        background: "Background replace: the person is kept and everything around them becomes the place on the sheet "
            + "(Load Character Sheet: a picture of the place). A chunk with nobody in it still needs its prompt written by hand.",
        object: "Object replace / add: the target is a thing (\"the painting on the wall\") or where a new thing goes (\"his left shoulder\"), "
            + "and the sheet is the new object. Usually on a finished swap's assembly. Assemble keeps the source everywhere and pastes the "
            + "render only inside the target's mask, grown and feathered (the settings beside this switch).",
    };
    const modeBox = el("div", { display: "flex", border: "1px solid #3b4252", borderRadius: "4px", overflow: "hidden", flexShrink: "0" });
    const modeBtns = {};
    for (const [m, t] of [["character", "Character replace"], ["background", "Background replace"], ["object", "Object"]]) {
        const b = el("button", { border: "none", padding: "0.2em 0.7em", cursor: "pointer", font: "inherit" }, t);
        b.title = MODE_TIPS[m] + " Switching keeps everything: each mode has its own subject, prompts and takes, and flipping back brings them back.";
        b.onpointerdown = (e) => e.stopPropagation();
        b.onclick = () => { if (P() && jobMode() !== m) op({ op: "set_mode", mode: m }); };
        modeBtns[m] = b;
        modeBox.append(b);
    }
    // character replace only: what the guide marks. The backdrop (everything but the person) is EXPERIMENTAL:
    // made for a plain green screen, but over ~50 test renders it made no reliable difference (2026-10-08)
    const guideBox = el("div", { display: "flex", alignItems: "center", gap: "4px", flexShrink: "0" });
    const guideInner = el("div", { display: "flex", border: "1px solid #3b4252", borderRadius: "4px", overflow: "hidden" });
    const guideBtns = {};
    for (const [g, t, tip] of [["person", "person", "Mark the person in the guide (the usual way)."],
        ["backdrop", "backdrop (exp.)", "Experimental: mark everything but the person in the guide, and still replace the person. Made for a plain green or blue screen; in testing it made no reliable difference to the character following the person."]]) {
        const b = el("button", { border: "none", padding: "0.2em 0.6em", cursor: "pointer", font: "inherit" }, t);
        b.title = tip;
        b.onpointerdown = (e) => e.stopPropagation();
        b.onclick = () => { if (P() && jobGuide() !== g) op({ op: "set_guide", guide: g }); };
        guideBtns[g] = b;
        guideInner.append(b);
    }
    guideBox.append(el("span", { color: C.dim }, "mark"), guideInner);
    // object jobs only: replace the target or add onto it, and how the render is pasted back (swap_plan.object_settings)
    const objBox = el("div", { display: "none", alignItems: "center", gap: "4px", flexShrink: "0" });
    const objSel = el("select", { background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px" });
    [["replace", "replace it"], ["add", "add onto it"]].forEach(([v, t]) => { const o = el("option", null, t); o.value = v; objSel.append(o); });
    objSel.title = "replace it: the sheet's object takes the target's place. add onto it: the sheet's object is added on the target (a bird on a shoulder); the paste mask is grown wider (60 px).";
    const objNum = (key, tip) => { const i = el("input", { width: "3.5em", background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px" }); i.type = "number"; i.min = 0; i.title = tip;
        i.onpointerdown = (e) => e.stopPropagation(); i.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter") i.blur(); };
        i.onchange = () => op({ op: "set_object", [key]: parseInt(i.value, 10) || 0 }); return i; };
    const objGrow = objNum("grow", "px the target's mask is grown by before the render is pasted in");
    const objFeather = objNum("feather", "px of soft edge on the paste");
    objSel.onpointerdown = (e) => e.stopPropagation();
    objSel.onchange = () => op(objSel.value === "add" ? { op: "set_object", action: "add", guide: "zone" } : { op: "set_object", action: objSel.value });   // an add redraws a zone
    // what fills the target: the H3 render (a 3D thing: a bird, a prop), or a picture (a flat thing: a painting,
    // a poster, a screen), fitted over the target and toned to its light, with no render at all
    const objFill = el("select", { background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px" });
    [["render", "fill: H3 render"], ["picture", "fill: picture (flat)"]].forEach(([v, t]) => { const o = el("option", null, t); o.value = v; objFill.append(o); });
    objFill.title = "H3 render: the chunk's take, pasted in through the target's mask. picture: for a flat target (a painting, poster or screen): the picture below fitted over it and matched to its light, no render needed.";
    objFill.onpointerdown = (e) => e.stopPropagation();
    objFill.onchange = () => op({ op: "set_object", fill: objFill.value });
    const objPicIn = el("input"); objPicIn.type = "file"; objPicIn.accept = "image/*"; objPicIn.style.display = "none";
    objPicIn.onchange = async () => {
        const f = objPicIn.files?.[0];
        if (!f) return;
        try { const name = await uploadFile(f); await op({ op: "set_object", picture: name, fill: "picture" }); toast(`picture: ${name}`, "green"); }
        catch (e) { toast(`picture upload: ${e}`, "red"); }
        objPicIn.value = "";
    };
    const objPic = button("picture…", "Pick the picture for fill: picture (it is copied into ComfyUI's input folder)", () => objPicIn.click());
    objBox.append(objSel, el("span", { color: C.dim }, "grow"), objGrow, el("span", { color: C.dim }, "feather"), objFeather, objFill, objPic, objPicIn);
    // an add that REDRAWS A ZONE (guide "zone", swap_plan.object_zone) instead of pasting a cut-out: its own row, shown for an
    // add with fill H3 render. The numbers are in target widths (the target's own width on the first frame), like size.
    const zoneRow = el("div", { display: "none", alignItems: "center", gap: "4px", flexWrap: "wrap", flexShrink: "0", padding: "0 6px 2px" });
    const zStyle = { background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px" };
    const zLab = (t, tip) => { const l = el("span", { color: C.dim }, t); if (tip) l.title = tip; return l; };
    const zNum = (key, tip, step, int) => { const i = el("input", Object.assign({ width: "4em" }, zStyle)); i.type = "number"; i.step = step; i.title = tip;
        i.onpointerdown = (e) => e.stopPropagation(); i.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter") i.blur(); };
        i.onchange = () => op({ op: "set_object", [key]: int ? (parseInt(i.value, 10) || 0) : (parseFloat(i.value) || 0) }); return i; };
    const zSel = (key, opts, tip) => { const sl = el("select", zStyle); opts.forEach(([v, t]) => { const o = el("option", null, t); o.value = v; sl.append(o); });
        sl.title = tip; sl.onpointerdown = (e) => e.stopPropagation(); sl.onchange = () => op({ op: "set_object", [key]: sl.value }); return sl; };
    const zGuide = zSel("guide", [["composite", "guide: pasted cut-out (its render group was removed)"], ["zone", "guide: redraw a zone"]],
        "pasted cut-out: the old way (the cut-out of Picture 1 pasted on the target, refined img2img; it looked stuck on). Its render group is no longer in the workflow: use redraw a zone. redraw a zone: no cut-out; Qwen draws the thing into one frame (the anchor) and H3 redraws only a zone around where it goes, in the scene's own light, the rest of the crop and the audio held.");
    const zShape = zSel("zone_shape", [["ellipse", "ellipse"], ["box", "box"], ["mask", "the target's own mask"]], "The zone's shape. An ellipse leaves the corners as filmed; a box covers smoke or a body that fills it. A box with a soft feather hides its edge best. The target's own mask is for taking something OUT: H3 redraws exactly where it was (grown by the zone grow).");
    const zSize = zNum("size", "The zone's height, in target widths (the target's width on its first frame).", 0.1);
    const zAspect = zNum("zone_aspect", "The zone's width over its height.", 0.05);
    const zDx = zNum("zone_dx", "Move the zone sideways from the target's top-centre, in target widths (negative = to the left). 0 = on the target, as a bird on a shoulder.", 0.05);
    const zDy = zNum("zone_dy", "Move the zone down from where a thing standing on the target's top would sit, in target widths. A person beside the target needs it positive.", 0.05);
    const zGrow = zNum("zone_grow", "px the zone is padded by.", 5, true);
    const zSpare = el("input"); zSpare.type = "checkbox"; zSpare.title = "Keep the target's own pixels out of the zone (a person added beside someone: she stays exactly as filmed).";
    zSpare.onpointerdown = (e) => e.stopPropagation(); zSpare.onchange = () => op({ op: "set_object", zone_spare: zSpare.checked });
    const zTx = zNum("zone_touch_dx", "Where the new thing touches the target: the circle's centre sideways from the target's top-centre, in target widths.", 0.05);
    const zTy = zNum("zone_touch_dy", "Where the new thing touches the target: the circle's centre down from the target's top-centre, in target widths.", 0.05);
    const zTr = zNum("zone_touch_r", "The touch circle's radius, in target widths (0 = no touch). It puts a part of the spared target back in the zone: a hand on her shoulder.", 0.05);
    const zAnchor = zNum("anchor_at", "Seconds from the chunk's start: the frame Qwen draws the thing into (the anchor). Pick a moment it is fully in the scene and the target is not hidden.", 0.1);
    const zEnter = zNum("enter_at", "Seconds from the chunk's start: when the thing enters. The zone is held empty before it (0 = it is there from the start).", 0.1);
    const zAnch = zSel("anchor", [["qwen", "anchor: Qwen draws it"], ["composite", "anchor: the cut-out placed (no model)"], ["picture", "anchor: my picture"]],
        "Who makes the anchor frame (the thing in place, in one frame, that H3 follows). Qwen draws it from the chunk's instruction below and Picture 1. The cut-out composite places Picture 1's cut-out on the target (no model; it can look stuck on, H3 repaints it). My picture: a finished frame of your own, the crop's size.");
    const zPicIn = el("input"); zPicIn.type = "file"; zPicIn.accept = "image/*"; zPicIn.style.display = "none";
    zPicIn.onchange = async () => {
        const f = zPicIn.files?.[0];
        if (!f) return;
        try { const name = await uploadFile(f); await op({ op: "set_object", anchor_picture: name, anchor: "picture" }); toast(`anchor picture: ${name}`, "green"); }
        catch (e) { toast(`anchor picture upload: ${e}`, "red"); }
        zPicIn.value = "";
    };
    const zPic = button("anchor picture…", "Pick your own anchor frame (copied into ComfyUI's input folder): the thing already in place, cropped like the render", () => zPicIn.click());
    const zLevel = zSel("level", [["ring", "level: match the scene"], ["none", "level: as drawn"]], "match the scene: the zone is scaled to the grade of the picture just outside it, so a thing the model drew a little bright blends in. as drawn: no correction.");
    // the zone locked to a spot from a frame on (plan.object.zone_keys): it holds there until the next lock; before the first lock the numbers here hold
    const ZONE_KEY_FIELDS = ["size", "zone_aspect", "zone_dx", "zone_dy", "zone_touch_dx", "zone_touch_dy", "zone_touch_r"];
    const zKeys = () => (((P() || {}).object || {}).zone_keys || []);
    const zKeyLbl = el("span", { color: C.dim, fontSize: "0.9em" }, "");
    const zLock = button("🔒 lock here", "Lock the zone where it is now, from this frame on: it stays in that spot until the next lock. Move it first (drag, or the numbers), then lock; lock again later in the clip to move it again.", async () => {
        if (!zoneGeo?.found) { toast("the zone isn't shown on this frame (mark the target first)", "red"); return; }
        const fr = Math.round(S.playhead), key = { frame: fr };
        for (const f of ZONE_KEY_FIELDS) key[f] = zoneGeo.settings[f];
        await op({ op: "set_object", zone_keys: zKeys().filter(k => k.frame !== fr).concat(key) }, true);
        loadZone();
    });
    const zUnlock = button("unlock", "Remove the lock that holds on this frame: the zone goes back to the one before it (or to the numbers here).", async () => {
        if (zoneGeo?.key == null) { toast("no lock holds on this frame", "red"); return; }
        await op({ op: "set_object", zone_keys: zKeys().filter(k => k.frame !== zoneGeo.key) }, true);
        loadZone();
    });
    // what to add, in your own words: the drafter's vision model reads it with the frames and the sheet and writes both the H3 prompt and the Qwen instruction from it (the zone below is only where H3 redraws)
    const zBrief = el("input", Object.assign({ width: "26em", maxWidth: "100%" }, zStyle)); zBrief.type = "text";
    zBrief.placeholder = "what to add, e.g. a man with green hair sitting next to her";
    zBrief.title = "What to add and where, in plain words. 'draft prompts' / 'redraft' use it with what the model sees in the frames and the sheet to write the H3 prompt and the Qwen instruction. The circles below only mark where H3 redraws: they are not told to the drafter.";
    zBrief.onpointerdown = (e) => e.stopPropagation(); zBrief.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter") zBrief.blur(); };
    zBrief.onchange = () => op({ op: "set_object", brief: zBrief.value });
    const zoneBits = [zLab("add", "what to add, in your words"), zBrief, zLock, zUnlock, zKeyLbl, zLab("shape"), zShape, zLab("size", "height, in target widths"), zSize, zLab("w/h"), zAspect, zLab("dx"), zDx, zLab("dy"), zDy, zLab("pad px"), zGrow,
        zLab("spare target", "keep the target as filmed"), zSpare, zLab("touch dx"), zTx, zLab("dy"), zTy, zLab("r"), zTr, zLab("anchor s"), zAnchor, zLab("enters s"), zEnter, zAnch, zPic, zPicIn, zLevel];
    zoneRow.append(zGuide, ...zoneBits);
    const targetLbl = el("span", { color: C.dim }, "replace");
    targetRow.append(modeBox, guideBox, objBox, targetLbl, targetIn);

    const canvas = el("canvas", { width: "100%", height: `${CANVAS_H}px`, display: "block", cursor: "default",
        borderRadius: "4px", touchAction: "none", flex: "0 0 auto" });

    // Fixed heights below the strip: a panel that grew with its content shrank the player and moved
    // the strip under the pointer whenever the selection changed.
    const selBox = el("div", { display: "flex", flexDirection: "column", gap: "4px", flex: "0 0 auto", height: "19em",
        overflowY: "auto", borderTop: "1px solid #2b3040", paddingTop: "3px" });
    const nextBar = el("div", { display: "flex", gap: "6px", alignItems: "center", height: "1.9em", overflow: "hidden", padding: "2px 6px", flexShrink: "0",
        background: "#1e1b2e", border: "1px solid #3b2d5c", borderRadius: "4px", flexWrap: "wrap" });
    // The warnings: a titled panel, one readable row each (wrapped, not cut off), click one to jump to it.
    const warnBox = el("div", { display: "flex", flexDirection: "column", gap: "2px", flex: "0 0 auto", height: "8.5em", overflowY: "auto",
        background: "#17140d", border: "1px solid #4a3b12", borderRadius: "4px", padding: "3px 6px", boxSizing: "border-box" });
    // The three sections under the player collapse from the toolbar; the panel and the warnings drag taller or shorter
    // (their corner), and the player takes what's left.
    // A drag bar under a section. CSS resize corners were drawn but never started a drag inside the node (the canvas
    // takes the pointer), so the bar captures the pointer itself; the canvas zoom scales the DOM widget.
    function grip(box, key) {
        const g = el("div", { height: "10px", flex: "0 0 auto", cursor: "ns-resize", borderRadius: "3px", margin: "1px 10%",
            background: "repeating-linear-gradient(0deg, #4b5568 0 2px, #2a303d 2px 4px)", border: "1px solid #4b5568" });
        g.title = "Drag up or down to make this section shorter or taller (a taller node gives everything more room)";
        g.onpointerdown = (e) => {
            e.stopPropagation(); e.preventDefault(); g.setPointerCapture(e.pointerId);
            const y0 = e.clientY, h0 = box.offsetHeight, k = app.canvas?.ds?.scale || 1;
            g.onpointermove = (m) => { box.style.height = `${Math.round(Math.max(48, h0 + (m.clientY - y0) / k))}px`; };
            g.onpointerup = () => { g.onpointermove = null; S.layout[key] = box.style.height; saveUiSoon(); };
        };
        return g;
    }
    const selGrip = grip(selBox, "selH"), warnGrip = grip(warnBox, "warnH");
    const SECTIONS = [["strip", "the strip (cuts, splits, chunks, mask and prompt rows)", () => [canvas, nextBar]],
        ["panel", "the selection panel (a chunk's prompt and takes)", () => [selBox, selGrip]], ["warnings", "the warnings", () => [warnBox, warnGrip]]];
    const secButtons = SECTIONS.map(([k, what]) => button(k, `Show or hide ${what}`, () => {
        S.layout.hide[k] = !S.layout.hide[k]; applyLayout(); saveUiSoon(); }));
    function applyLayout() {
        SECTIONS.forEach(([k, , els], i) => {
            for (const e of els()) e.style.display = S.layout.hide[k] ? "none" : (e === canvas || e === selGrip || e === warnGrip ? "block" : "flex");
            secButtons[i].textContent = `${S.layout.hide[k] ? "▸" : "▾"} ${k}`;
        });
        if (S.layout.selH) selBox.style.height = S.layout.selH;
        if (S.layout.warnH) warnBox.style.height = S.layout.warnH;
    }
    status.before(gap(), ...secButtons);
    applyLayout();


    const fileInput = el("input", { display: "none" });
    fileInput.type = "file";
    fileInput.accept = "video/*";
    fileInput.onchange = async () => { if (fileInput.files[0]) await setSourceFile(fileInput.files[0]); fileInput.value = ""; };

    root.append(head, targetRow, zoneRow, videoBox, bar1, canvas, nextBar, selBox, selGrip, warnBox, warnGrip, fileInput);

    const widget = node.addDOMWidget("swap_planner_ui", "div", root, { serialize: false, hideOnZoom: false });
    // Keep the panel out of widgets_values (a saved blank shifted onto later widgets: timeline.js).
    widget.serialize = false;
    const fcw = W("forward_chain");
    if (fcw) { const prev = fcw.callback; fcw.callback = function () { const r = prev ? prev.apply(this, arguments) : undefined; try { refresh(); } catch { } return r; }; }
    widget.computeSize = (width) => [Math.max(500, (width || node.size[0]) - 20), 640];
    if (node.size[0] < 980) node.size[0] = 980;
    if (node.size[1] < 1080) node.size[1] = 1080;

    function fitToNode() {
        const u = Math.max(1, Math.min(2.5, node.size[0] / 1000));
        if (Math.abs(u - U) > 0.01) {
            setScale(u);
            root.style.fontSize = `${Math.round(11 * U)}px`;
            canvas.style.height = `${CANVAS_H}px`;
            refresh();
        }
        if (widget.last_y) {
            const h = Math.max(420, node.size[1] - widget.last_y - 15);
            if (Math.abs((parseFloat(root.style.height) || 0) - h) > 1) root.style.height = `${h}px`;
        }
    }
    const onDrawFg = node.onDrawForeground;
    node.onDrawForeground = function () { const r = onDrawFg?.apply(this, arguments); fitToNode(); return r; };
    const onResize = node.onResize;
    node.onResize = function () { const r = onResize?.apply(this, arguments); fitToNode(); return r; };

    // ------------------------------------------------------------ plan access
    const job = () => (jobW?.value || "").trim();
    const P = () => S.plan;
    // the job's mode: "character" or "background" (a plan from before the switch carries invert instead)
    // the mask that renders: the backdrop marked for a background replace, or a character replace guided by the backdrop
    const maskView = () => jobMode() === "object" ? "mask_obj" : (jobMode() === "background" || jobGuide() === "backdrop") ? "mask_bg" : "mask";
    const jobGuide = () => (P()?.guide === "backdrop" ? "backdrop" : "person");
    const jobMode = () => { const p = P(); return !p ? "character" : (p.mode === "background" || p.mode === "character" || p.mode === "object")
        ? p.mode : (p.invert ? "background" : "character"); };
    const chunks = () => P()?.chunks || [];
    const splits = () => P()?.splits || [];
    const cuts = () => P()?.cuts || [];
    const confirmed = () => cuts().filter(c => c.confirmed !== false).map(c => c.frame);
    const settings = () => Object.assign({ overlap: 12, anchors: 5, target_render: 209, floor: 124, ceiling: 209, trained_max: 362, guard: 6, hand_back: 12 }, P()?.settings || {});
    const statusOf = (cid) => S.status.find(s => s.chunk === cid) || {};
    const chunkIndex = (cid) => chunks().findIndex(c => c.id === cid);
    const chunkAt = (f) => chunks().find(c => f >= c.deliver[0] && f <= c.deliver[1]);
    const splitById = (sid) => splits().find(s => s.id === sid);
    const joinAt = (sid) => S.joins.find(j => j.split === sid);
    const warnsFor = (key, id) => (S.view?.warnings || []).filter(w => w[key] === id);
    const takeOf = (cid, tid) => (chunks().find(c => c.id === cid)?.takes || []).find(t => t.id === tid);
    const kept = (c) => !!(c && c.keep);          // keep original (§4.10): never rendered, marked or drafted
    // the trim handles: the start trim sits on the kept start's right split, else at frame 0; the end trim mirrors it
    function trims() {
        const cs = chunks(), N = S.N;
        const s0 = cs.length > 1 && kept(cs[0]) ? splitById(cs[1].left)?.frame : null;
        const s1 = cs.length > 1 && kept(cs.at(-1)) ? splitById(cs.at(-1).left)?.frame : null;
        return { start: s0 ?? 0, end: s1 ?? N, hasStart: s0 != null, hasEnd: s1 != null };
    }
    const label = (cid) => { const i = chunkIndex(cid); return i >= 0 ? `${i + 1}` : cid; };

    function toast(msg, kind = "dim") { status.textContent = msg; status.style.color = C[kind] || C.dim; status.title = msg; }

    async function reload() {
        const j = job();
        const token = ++S.loading;
        if (!j) { S.view = null; S.plan = null; S.joins = []; S.status = []; refresh(); return; }
        try {
            const r = await api.fetchApi(`/seamstitch/swap/plan?job=${encodeURIComponent(j)}`);
            const d = await r.json();
            if (token !== S.loading) return;
            if (d.error) { S.view = null; S.plan = null; S.joins = []; S.status = []; S.err = d.error; refresh(); return; }
            S.err = null;
            const firstLoad = !S.plan || S.plan.job !== d.plan.job || S.plan.source?.path !== d.plan.source?.path;
            S.view = d; S.plan = d.plan; S.joins = d.joins || []; S.status = d.status || []; S.jobDir = d.job_dir;
            S.fr = Math.round(+d.plan.source.fps) || 25; S.N = +d.plan.source.frames || 0;
            if (S.sel && !selValid()) S.sel = null;
            if (firstLoad) {
                refreshJobs();
                await probeSource();
                if (!S.restored) { restoreUi(); S.restored = true; }
                requestFit();
                seek(S.playhead);
            }
            refresh();
        } catch (e) { if (token === S.loading) toast(`plan: ${e}`, "red"); }
    }
    function selValid() {
        const s = S.sel;
        if (!s) return false;
        if (s.kind === "chunk") return chunkIndex(s.id) >= 0;
        if (s.kind === "split") return !!splitById(s.id);
        if (s.kind === "cut") return cuts().some(c => c.frame === s.frame);
        return false;
    }
    async function probeSource() {
        try {
            const r = await api.fetchApi(`/seamstitch/timeline/probe?path=${encodeURIComponent(P().source.path)}&frame_rate=${S.fr}`);
            const j = await r.json();
            S.base = r.ok ? (j.base_time || 0) : 0;
        } catch { S.base = 0; }
    }

    // Every write: one op, applied to the newest revision under the plan lock, then a reload.
    async function op(body, quiet) {
        if (!job()) { toast("set a job name first", "amber"); return null; }
        try {
            const r = await api.fetchApi("/seamstitch/swap/op", { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify(Object.assign({ job: job() }, body)) });
            const j = await r.json();
            if (j.error) { toast(`${body.op}: ${j.error}`, "red"); await reload(); return null; }
            if (!quiet) toast(`${body.op} ✓ (rev ${j.rev})`, "green");
            await reload();
            return j;
        } catch (e) { toast(`${body.op}: ${e}`, "red"); return null; }
    }

    // ------------------------------------------------------------ source and job
    // Loading a video starts a job named after it (taken names get _2, _3...), and its cuts and
    // splits are found straight away (detect + plan). The dropdown switches between jobs.
    async function refreshJobs() {
        let jobs = [];
        try { jobs = (await (await api.fetchApi("/seamstitch/swap/jobs")).json()).jobs || []; } catch { }
        S.jobs = jobs;
        jobSel.innerHTML = "";
        const opt = (v, t) => { const o = el("option", null, t); o.value = v; jobSel.append(o); return o; };
        if (!job() || !jobs.some(j => j.job === job())) opt("", job() ? `${job()} (no plan yet)` : "choose a job…");
        for (const j of jobs) {
            const v = (j.source || "").split(/[\\/]/).pop();
            opt(j.job, `${j.job} · ${v} · ${j.chunks} chunk${j.chunks === 1 ? "" : "s"}${j.takes ? `, ${j.takes} take${j.takes === 1 ? "" : "s"}` : ""}`);
        }
        opt(NEW_JOB, "＋ new job: load a video…");
        jobSel.value = jobs.some(j => j.job === job()) ? job() : "";
    }
    function pickJob(v) {
        if (v === NEW_JOB) {
            jobSel.value = job();
            const r = jobSel.getBoundingClientRect();
            openSourceMenu({ clientX: r.left, clientY: r.bottom });
            return;
        }
        if (!v || v === job()) return;
        if (S.promptDirty) savePrompt();
        jobW.value = v;
        srcW.value = (S.jobs || []).find(j => j.job === v)?.source || "";
        S.restored = false; S.sel = null; S.plan = null;
        app.graph?.setDirtyCanvas(true, true);
        reload();
    }
    async function createJob(source) {
        const stem = source.split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
        let name = stem;
        try { name = (await (await api.fetchApi(`/seamstitch/swap/jobs?name_for=${encodeURIComponent(stem)}`)).json()).free_name || stem; } catch { }
        jobW.value = name;
        srcW.value = source;
        app.graph?.setDirtyCanvas(true, true);
        S.restored = false; S.sel = null; S.plan = null;
        const r = await op({ op: "create", source }, true);
        if (!r) return;
        toast(`job ${name} created on ${stem}: finding its cuts and splits…`, "green");
        await reload();
        await detectPlan();
    }
    async function setSourceFile(file) {
        try { toast(`uploading ${file.name}…`); const name = await uploadFile(file); await createJob(name); }
        catch (e) { toast(`${file.name}: ${e}`, "red"); }
    }
    async function openSourceMenu(ev) {
        let files = [];
        try { files = (await (await api.fetchApi("/seamstitch/timeline/list")).json()).files || []; } catch { }
        popup(ev, [
            { label: "load a video: a new job, named after it" },
            { text: "⬆ upload from disk…", run: () => fileInput.click() },
            { sep: true },
            { label: files.length ? "input folder (newest first)" : "no videos in the input folder" },
            ...files.slice(0, 60).map(f => ({ text: f, run: () => createJob(f) })),
        ]);
    }

    // ------------------------------------------------------------ cuts and splits
    async function detectPlan() {
        if (!P()) return;
        toast("finding the cuts and placing the splits…");
        try {
            const r = await api.fetchApi("/seamstitch/swap/detect_cuts", { method: "POST", body: JSON.stringify({ job: job(), plan: true }) });
            const j = await r.json();
            if (j.error) throw new Error(j.error);
            const weak = (j.weak || []).length ? ` · ${j.weak.length} faint suggested cuts (scene 0.10-0.15, same-room jumps?) at ${j.weak.join(", ")}: check them by eye` : "";
            toast((j.kept_splits ? `${j.found.length} cuts found and confirmed; splits kept (chunks have prompts or takes): cuts ▾ → auto splits to re-place them`
                : `${j.found.length} cuts, splits at ${(j.splits || []).join(" / ") || "none (one chunk)"}: review them (delete a wrong cut, drag or re-mode a split)`) + weak, "green");
        } catch (e) { toast(`detect + plan: ${e.message || e}`, "red"); }
        await reload();
    }
    async function detectCuts() {
        if (!P()) return;
        toast("detecting cuts…");
        try {
            const r = await api.fetchApi("/seamstitch/swap/detect_cuts", { method: "POST", body: JSON.stringify({ job: job() }) });
            const j = await r.json();
            if (j.error) throw new Error(j.error);
            toast(`detected ${j.found.length} cuts (${j.added.length} new suggestions)${(j.weak || []).length ? ` and ${j.weak.length} faint ones (scene 0.10-0.15)` : ""}: confirm, move or delete them`, "green");
        } catch (e) { toast(`detect cuts: ${e.message || e}`, "red"); }
        await reload();
    }
    function openCutsMenu(ev) {
        if (!P()) return;
        const sug = cuts().filter(c => c.confirmed === false).length;
        popup(ev, [
            { text: "detect cuts only (faint suggestions, splits untouched)", run: () => detectCuts() },
            ...(sug ? [{ text: `confirm all ${sug} suggested cuts`, run: () => op({ op: "confirm_cuts" }) }] : []),
            { text: "re-place the auto splits from the confirmed cuts", run: () => autoSplits() },
        ]);
    }
    async function clearAll() {
        if (!P()) return;
        const takes = chunks().reduce((a, c) => a + (c.takes || []).length, 0);
        const prompts = chunks().filter(c => (c.prompt || "").trim()).length;
        if (!confirm(`Clear everything on job ${job()}: ${cuts().length} cuts, ${splits().length} splits, ` +
            `${chunks().filter(kept).length} kept stretches and ${prompts} prompts?\n\n` +
            `Nothing is erased: ${takes ? `${takes} takes stay on disk (listed under removed chunks), ` : ""}` +
            `the cached masks stay. Then press 'detect + plan' to start over.`)) return;
        const r = await op({ op: "clear_all" }, true);
        if (r) { S.sel = null; toast(`cleared: ${r.result.cleared.cuts} cuts, ${r.result.cleared.splits} splits, ${r.result.cleared.kept} kept, ${r.result.cleared.prompts} prompts; masks kept (${r.result.cleared.masks_kept})`, "green"); refresh(); }
    }
    async function undoLast() {
        if (S.promptDirty) savePrompt();
        const r = await op({ op: "undo" }, true);
        if (r) toast(`undid: ${r.result?.undid || "the last edit"}`, "green");
    }
    async function autoSplits() {
        const has = chunks().some(c => (c.takes || []).length || (c.prompt || "").trim());
        if (has && !confirm("Auto splits replace every split; chunks with prompts or takes are dropped (takes stay on disk, listed under removed_chunks). Go ahead?")) return;
        const unconf = cuts().filter(c => c.confirmed === false).length;
        const r = await op({ op: "auto_splits", force: has });
        if (r) toast(`auto splits at ${(r.result?.splits || []).join(" / ")}${unconf ? ` (${unconf} suggested cuts ignored: confirm them first)` : ""}`, "green");
    }
    function nearestConfirmed(f, within) {
        let best = null;
        for (const c of confirmed()) if (Math.abs(c - f) <= within && (best == null || Math.abs(c - f) < Math.abs(best - f))) best = c;
        return best;
    }
    async function splitAtPlayhead() {
        if (!P()) return;
        const f = Math.round(S.playhead);
        if (f <= 0 || f >= S.N) return;
        const c = nearestConfirmed(f, 1);
        if (splits().some(s => s.frame === (c ?? f))) { toast(`there is a split at ${c ?? f} already`, "amber"); return; }
        // on a confirmed cut (within a frame): a straight cut at the cut, the first frame of the new shot
        const r = await op({ op: "add_split", frame: c ?? f, mode: c != null ? "cut" : "anchored" });
        if (r) { S.sel = { kind: "split", id: r.result?.split }; refresh(); }
    }
    async function toggleCutAtPlayhead() {
        const f = Math.round(S.playhead);
        const hit = cuts().find(c => c.frame === f);
        if (hit) await op({ op: hit.confirmed === false ? "confirm_cut" : "delete_cut", frame: f });
        else await op({ op: "add_cut", frame: f, from: "manual", confirmed: true });
    }

    function splitMenu(ev, s) {
        const j = joinAt(s.id), rep = s.repair || {};
        const near = nearestConfirmed(s.frame, 40);
        const L = chunks()[chunkIndex(chunks().find(c => c.left === s.id)?.id) - 1];
        const R = chunks().find(c => c.left === s.id);
        popup(ev, [
            { label: `split ${s.id} at ${s.frame} · ${s.mode === "cut" ? "straight cut" : "anchored"}${j ? ` · ${j.type}` : ""}` },
            { text: `${s.mode === "cut" ? "●" : "○"} straight cut`, title: "The right chunk renders from J on its own: for a split on a source cut", run: () => op({ op: "split_mode", split: s.id, mode: "cut" }) },
            { text: `${s.mode !== "cut" ? "●" : "○"} anchored`, title: "The right chunk renders from J - overlap, pinned on the left take's frames, spliced at J", run: () => op({ op: "split_mode", split: s.id, mode: "anchored" }) },
            { sep: true },
            { label: "repair" },
            ...["auto", "lock", "fade", "cut"].map(m => ({ text: `${(rep.mode || "auto") === m ? "●" : "○"} ${m}${m === "auto" && j ? ` (${j.repair})` : ""}`,
                run: () => op({ op: "split_repair", split: s.id, repair: Object.assign({}, rep, { mode: m }) }) })),
            { label: "hand-back" },
            ...[12, 25].map(h => ({ text: `${(rep.hand_back || settings().hand_back) === h ? "●" : "○"} ${h} frames${h === 25 ? " (a smaller dip)" : ""}`,
                run: () => op({ op: "split_repair", split: s.id, repair: Object.assign({}, rep, { hand_back: h === settings().hand_back ? null : h }) }) })),
            { text: "custom…", run: () => { const v = parseInt(prompt("hand-back frames", rep.hand_back || settings().hand_back), 10); if (v > 1) op({ op: "split_repair", split: s.id, repair: Object.assign({}, rep, { hand_back: v }) }); } },
            { sep: true },
            ...(near != null && near !== s.frame ? [{ text: `snap to the cut at ${near}`, run: () => op({ op: "move_split", split: s.id, to: near }) }] : []),
            ...(j?.stale && L ? [{ text: `re-roll ${label(L.id)} to fit (two-sided)`, run: () => reroll(L.id, 1) }] : []),
            ...(j?.stale && R ? [{ text: `re-roll ${label(R.id)} to fit (two-sided)`, run: () => reroll(R.id, 1) }] : []),
            { text: "jump here", run: () => seek(s.frame) },
            { text: "delete split", run: () => op({ op: "delete_split", split: s.id }).then(() => { S.sel = null; refresh(); }) },
        ]);
    }

    // ------------------------------------------------------------ queue
    function graphPlannerId() { return String(node.id); }
    function outputNodeType(ct) {
        const t = window.LiteGraph?.registered_node_types?.[ct];
        return !!(t?.nodeData?.output_node);
    }
    // Output nodes downstream of the given Planner outputs, in the API prompt.
    function targetsFor(output, slots) {
        const me = graphPlannerId();
        const kids = {};
        for (const [id, n] of Object.entries(output)) {
            for (const v of Object.values(n.inputs || {})) {
                if (Array.isArray(v) && v.length === 2 && typeof v[1] === "number") (kids[String(v[0])] ||= []).push({ id, slot: v[1] });
            }
        }
        const seen = new Set(), q = (kids[me] || []).filter(k => slots.includes(k.slot)).map(k => k.id);
        while (q.length) {
            const id = q.shift();
            if (seen.has(id)) continue;
            seen.add(id);
            for (const k of kids[id] || []) q.push(k.id);
        }
        return [...seen].filter(id => outputNodeType(output[id].class_type));
    }
    // A failed render in a left-to-right batch takes the rest of that batch off ComfyUI's queue:
    // the chunks after it would otherwise render free where they were meant to pin to it.
    async function cancelRestOfBatch(q) {
        if (!q.batch) return [];
        const rest = Object.entries(S.queued).filter(([, x]) => x.batch === q.batch && x.seq > q.seq);
        if (!rest.length) return [];
        try {
            await api.fetchApi("/queue", { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ delete: rest.map(([pid]) => pid) }) });
        } catch { }
        for (const [pid] of rest) delete S.queued[pid];
        return rest.map(([, x]) => x.run.chunk);
    }
    // Renders and drafts start on a clear machine: ComfyUI drops its models and its cache (which holds them in
    // RAM), as its own /free does. Only ComfyUI can drop its cache, between queue items: so before queueing,
    // and again when a render starts with another waiting behind it (B4: Qwen ran out of the paging file right
    // after a render; B6: a strip render bottomed at 0.7 GB free RAM). Each render reloads its models.
    async function freeMemory(wait) {
        try {
            await api.fetchApi("/free", { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ unload_models: true, free_memory: true }) });
            if (wait) await new Promise(r => setTimeout(r, 1500));     // an idle worker applies it at once
        } catch { }
    }
    async function queueRun(run, kind, quiet, batch) {
        if (kind === "render" || kind === "draft") await freeMemory(true);
        run = Object.assign({}, run, { nonce: uid() });
        runW.value = JSON.stringify(run);
        try {
            const p = await app.graphToPrompt();
            if (!p.output[graphPlannerId()]) throw new Error("this Planner isn't in the queued graph (bypassed or muted?)");
            const targets = targetsFor(p.output, OUT[kind]);
            // a target or invert the graph can't pass on would be marked as "person" and cached under the target's name (B7)
            const named = P() && (P().target || jobMode() === "background" || chunks().some(c => (c.options || {}).target));
            if ((kind === "render" || kind === "mark") && named) {
                const keys = new Set(Object.values(p.output).filter(n => n.class_type === "SeamStitchSwapOption").map(n => n.inputs?.key));
                const lack = ["target", "invert"].filter(k => !keys.has(k));
                if (lack.length) throw new Error(`this workflow has no Swap Option for ${lack.join(" / ")}, so SAM3 would mark "person": open the example workflow again (or add the option to your render and mark groups)`);
            }
            if (!targets.length) {
                const need = { render: "a render group ending in SeamStitch Swap Take", draft: "SeamStitch Swap Draft Prompts on draft_plan",
                    assemble: "SeamStitch Swap Assemble on assemble_plan", mark: "a mark group ending in SeamStitch Swap Mask on mark_chunk / mark_images" }[kind];
                throw new Error(`nothing to run: wire ${need}`);
            }
            // The frontend's own queuePrompt, not whatever other packs wrapped around it: a wrapper
            // written as (number, data) drops the options, and with them the partial-execution
            // targets, so every output node runs (found live: BKWILDCARDS' preview hook).
            const queue = Object.getPrototypeOf(api).queuePrompt || api.queuePrompt;
            const res = await queue.call(api, 0, p, { partialExecutionTargets: targets });
            if (res?.node_errors && Object.keys(res.node_errors).length) throw new Error(`node errors: ${JSON.stringify(res.node_errors).slice(0, 300)}`);
            S.queued[res.prompt_id] = { run, kind, at: Date.now(), targets, batch: batch?.id, seq: batch ? batch.seq++ : 0 };
            if (!quiet) toast(`queued ${kind}${run.chunk ? ` ${label(run.chunk)}` : ""}`, "green");
            return res.prompt_id;
        } catch (e) {
            const msg = e?.response?.error?.message || e?.message || String(e);
            toast(`${kind}: ${msg}`, "red");
            return null;
        } finally {
            runW.value = "";   // a plain Queue press afterwards runs the panel only
            app.graph?.setDirtyCanvas(true, true);
        }
    }
    function emptyPrompts() { return chunks().filter(c => !kept(c) && !(c.prompt || "").trim()); }
    function needsRender(c) {
        if (kept(c)) return false;
        const st = statusOf(c.id);
        return ["pending", "range changed", "missing file"].includes(st.state) && !queuedFor(c.id) && !st.rendering;
    }
    function queuedFor(cid) { return Object.values(S.queued).find(q => q.run.chunk === cid && q.kind === "render"); }
    async function renderPending() {
        if (!P()) return;
        const empty = emptyPrompts();
        if (empty.length) { toast(`render pending refused: no prompt on chunk ${empty.map(c => label(c.id)).join(", ")} (every chunk needs one first)`, "red"); S.sel = { kind: "chunk", id: empty[0].id }; refresh(); return; }
        const todo = chunks().filter(needsRender);
        if (!todo.length) { toast("nothing pending: every chunk has a take (re-roll one to make another)", "amber"); return; }
        let n = 0;
        const batch = { id: uid(), seq: 0 };
        for (const c of todo) {                    // left to right: each one's pins resolve at execution
            if (await queueRun({ action: "render", chunk: c.id, seed: seedFor(c), prompt: c.prompt, options: c.options || {} }, "render", true, batch)) n++;
            else break;
        }
        toast(`queued ${n} render${n === 1 ? "" : "s"}: chunks ${todo.slice(0, n).map(c => label(c.id)).join(", ")}`, n ? "green" : "red");
        refresh();
    }
    function seedFor(c, fresh) {
        const m = c.seed_mode;
        if (m && typeof m === "object" && Number.isInteger(m.fixed) && !fresh) return m.fixed;
        return randSeed();
    }
    async function reroll(cid, n, fresh) {
        const c = chunks().find(x => x.id === cid);
        if (!c) return;
        if (kept(c)) { toast(`chunk ${label(cid)} is kept as the original: switch 'keep original' off to render it`, "amber"); return; }
        if (!(c.prompt || "").trim()) { toast(`chunk ${label(cid)} has no prompt`, "red"); return; }
        if (S.promptDirty && S.sel?.id === cid) await savePrompt();
        const fixed = c.seed_mode && typeof c.seed_mode === "object";
        let k = 0;
        for (let i = 0; i < n; i++) {
            // re-roll x N: N new seeds; a single re-roll keeps a fixed seed (a prompt-only A/B)
            const seed = n > 1 || fresh ? randSeed() : seedFor(c);
            if (await queueRun({ action: "render", chunk: c.id, seed, prompt: c.prompt, options: c.options || {} }, "render", true)) k++;
        }
        if (k) toast(`queued ${k} render${k === 1 ? "" : "s"} of chunk ${label(cid)}${n === 1 && fixed && !fresh ? ` (fixed seed ${c.seed_mode.fixed})` : ""}`, "green");
        refresh();
    }
    async function queueDraft(chunk) {
        if (!P()) return;
        await queueRun(chunk ? { action: "draft", chunk } : { action: "draft" }, "draft");
    }
    // A chunk is marked shot by shot: SAM3 keeps one track and doesn't find the target again after a cut (B7).
    // A long shot goes in pieces, its first two frames on their own (test clip, 2026-10-17: three dark shots of 126,
    // 442 and 346 frames lost the track after their first frame or two and came back empty; the same shots marked in
    // 50-frame pieces held, except each shot's opening piece, which held once its first two frames went alone).
    const MARK_PIECE = 50, MARK_HEAD = 2;
    function markPieces(a, b) {
        if (b - a + 1 <= MARK_PIECE + 10) return [[a, b]];
        const out = [[a, a + MARK_HEAD - 1]];
        for (let s = a + MARK_HEAD; s <= b;) {
            let e = Math.min(b, s + MARK_PIECE - 1);
            if (b - e < MARK_PIECE / 3) e = b;          // no short last piece
            out.push([s, e]);
            s = e + 1;
        }
        return out;
    }
    async function markChunk(c, quiet) {
        const pieces = shotsOf(c).flatMap(s => markPieces(s.a, s.b));
        let n = 0;
        for (const [a, b] of pieces) if (await queueRun({ action: "mark", chunk: c.id, range: [a, b] }, "mark", true)) n++; else break;
        if (!quiet && n) toast(`queued ${n} mark run${n === 1 ? "" : "s"} for chunk ${label(c.id)} (shot by shot, long shots in pieces)`, "green");
        return n === pieces.length;
    }
    // A shot the target isn't in gets an empty mark (nobody to replace): SAM3 run there marks something else anyway
    // (a dark background behind a held object, a shadow on a wall) and the render repaints it as the character.
    async function markNobody(c, s) {
        if (await op({ op: "empty_mask", chunk: c.id, range: [s.a, s.b] }, true))
            toast(`shot ${s.a}-${s.b}: marked as nobody (left as filmed)`, "green");
    }
    // Object keep (swap_keep): something the swap must leave as filmed (test clip c10, 2026-10-20: H3 drew his white
    // shoes black). Its SAM3 mask is marked like the person's, under its own target, and Swap Assemble pastes the source
    // through it, grown and feathered. Its masks stay off the person's mask row.
    const objectKeeps = () => P()?.object_keeps || [];
    const personMasks = () => (P()?.masks || []).filter(m => m.kind !== "keep");
    // an Object job's masks: the cached masks of its target (all of them while the target is empty)
    const objectMasks = () => { const t = (P()?.target || "").trim(); return personMasks().filter(m => !t || m.target === t); };
    const keepMasksOf = (k) => (P()?.masks || []).filter(m => m.target === k.target);
    function keepCoverage(k, a, b) {
        const lo = Math.max(a, k.range[0]), hi = Math.min(b, k.range[1]);
        if (hi < lo) return null;
        const hit = new Set();
        for (const m of keepMasksOf(k)) for (let f = Math.max(lo, m.range[0]); f <= Math.min(hi, m.range[1]); f++) hit.add(f);
        return hit.size / (hi - lo + 1);
    }
    async function markKeep(k, a, b) {
        const lo = Math.max(a, k.range[0]), hi = Math.min(b, k.range[1]);
        if (hi < lo) return;
        const edges = [lo, ...confirmed().filter(f => f > lo && f <= hi), hi + 1];
        const pieces = edges.slice(0, -1).flatMap((s, i) => markPieces(s, edges[i + 1] - 1));
        let n = 0;
        for (const [x, y] of pieces) if (await queueRun({ action: "mark", keep: k.id, range: [x, y] }, "mark", true)) n++; else break;
        if (n) toast(`queued ${n} mark run${n === 1 ? "" : "s"} for "${k.target}" (${lo}-${hi}, shot by shot)`, "green");
    }
    function keepSection(c) {
        const row = () => el("div", { display: "flex", gap: "4px", alignItems: "center", flexWrap: "wrap" });
        const box = el("div", { display: "flex", flexDirection: "column", gap: "2px", padding: "2px 0" });
        const stop = (e) => e.stopPropagation();
        const inp = (style) => { const i = el("input", Object.assign({ background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px" }, style)); i.onpointerdown = stop; i.onkeydown = (e) => { stop(e); if (e.key === "Enter") i.blur(); }; return i; };
        const here = objectKeeps().filter(k => k.range[1] >= c.deliver[0] && k.range[0] <= c.deliver[1]);
        box.append(el("span", { color: C.dim }, "keep from source (pasted over the take at assembly through its own SAM3 mask; mark it, then assemble)"));
        for (const k of here) {
            const r = row();
            const on = el("input"); on.type = "checkbox"; on.checked = k.on !== false; on.onpointerdown = stop;
            on.onchange = () => op({ op: "set_object_keep", id: k.id, on: on.checked });
            const cov = keepCoverage(k, c.render[0], c.render[1]);
            const pct = cov == null ? 0 : Math.round(cov * 100);
            const num = (key, def, tip) => { const i = inp({ width: "3.5em" }); i.type = "number"; i.min = 0; i.value = k[key] ?? def; i.title = tip; i.onchange = () => op({ op: "set_object_keep", id: k.id, [key]: parseInt(i.value, 10) || 0 }); return i; };
            const lvl = el("select", { background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px" });
            [["none", "level: as filmed"], ["ring", "level: match the take"]].forEach(([v, t]) => { const o = el("option", null, t); o.value = v; lvl.append(o); });
            lvl.value = k.level || "none"; lvl.onpointerdown = stop;
            lvl.title = "match the take: the source pixels are scaled to the take's grade, measured on a ring just outside the mask";
            lvl.onchange = () => op({ op: "set_object_keep", id: k.id, level: lvl.value });
            r.append(on, el("span", { fontWeight: "bold" }, k.target), el("span", { color: C.dim }, `${k.range[0]}-${k.range[1]}`),
                el("span", { color: pct >= 100 ? "#d1fae5" : pct ? C.amber : C.red }, pct >= 100 ? "mask ✓" : pct ? `mask ${pct}%` : "not marked"),
                button("mark", `Queue the mark runs of "${k.target}" over this chunk's part of the keep, shot by shot`, () => markKeep(k, c.render[0], c.render[1])),
                button("view", "Play this keep's mask previews (the object inverted) in the player", () => setMode(`keep:${k.id}`)),
                el("span", { color: C.dim }, "grow"), num("grow", 6, "px the mask is grown by (covers where the take drew the object a little off)"),
                el("span", { color: C.dim }, "feather"), num("feather", 4, "px of soft edge"), lvl,
                button("remove", "Remove this keep (its masks stay cached)", () => op({ op: "remove_object_keep", id: k.id })));
            box.append(r);
        }
        const add = row();
        const t = inp({ flex: "0 1 16em" });
        t.placeholder = "what to keep, e.g. white shoe";
        const known = [...new Set(objectKeeps().map(k => k.target))];
        if (known.length) { const dl = el("datalist"); dl.id = `ss-keep-${node.id}`; known.forEach(v => { const o = el("option"); o.value = v; dl.append(o); }); add.append(dl); t.setAttribute("list", dl.id); }
        const addFor = async (range, chunk) => {
            const v = t.value.trim();
            if (!v) { toast("type what to keep first (SAM3's text, e.g. white shoe)", "amber"); return; }
            const res = await op(Object.assign({ op: "add_object_keep", target: v }, chunk ? { chunk } : { range }), true);
            if (res) toast(`keep "${v}" added: mark it, then assemble`, "green");
        };
        add.append(t, button("add for this chunk", "Keep it over this chunk's frames", () => addFor(null, c.id)),
            button("add for the whole video", "Keep it over every frame (mark the chunks it's in)", () => addFor([0, S.N - 1])));
        box.append(add);
        return box;
    }
    function openMarkMenu(ev) {
        if (!P()) return;
        const sel0 = S.sel?.kind === "chunk" ? chunks().find(c => c.id === S.sel.id) : null;
        const sel = kept(sel0) ? null : sel0;
        const unmarked = chunks().filter(c => !kept(c) && (statusOf(c.id).mask || 0) < 1);
        popup(ev, [
            { label: "the source person mask (SAM3), cached per chunk's render range" },
            ...(sel ? [{ text: `mark chunk ${label(sel.id)} (${sel.render[0]}-${sel.render[1]}, ${shotsOf(sel).length} shot${shotsOf(sel).length === 1 ? "" : "s"})`, run: () => markChunk(sel) }] : []),
            ...(sel ? shotsOf(sel).map(s => ({ text: `nobody in shot ${s.a}-${s.b} (the target isn't in it: an empty mark, left as filmed)`, run: () => markNobody(sel, s) })) : []),
            { text: `mark every unmarked chunk (${unmarked.length})`, run: async () => { let k = 0; for (const c of unmarked) { if (!await markChunk(c, true)) break; k++; } toast(`queued the mark runs of ${k} chunk${k === 1 ? "" : "s"}, one per shot`, "green"); } },
            { text: "view the mask row in the player", run: () => setMode(maskView()) },
        ]);
    }
    function openAssembleMenu(ev) {
        if (!P()) return;
        popup(ev, [
            { text: "assemble now (CPU, off the queue)", title: "POST /seamstitch/swap/assemble: the Assemble node's code, with its defaults", run: () => assembleNow() },
            { text: "queue an assemble run (Swap Assemble node, its widgets)", run: () => queueRun({ action: "assemble" }, "assemble") },
            ...(P().assembled?.length ? [{ sep: true }, { text: `play the latest assembly (${P().assembled.at(-1).file.split("/").pop()})`, run: () => setMode("full") }] : []),
            { sep: true },
            { text: "edit every chunk from its own take (chunk by chunk, no assembly)", title: "Each chunk with a take renders from that take next time. Same as 'edit this' on each one; chunks without a take are skipped.", run: () => chunkSource("all", {}) },
            ...(Object.keys(P().overrides || {}).length ? [{ text: `every chunk back to the original (${Object.keys(P().overrides).length} edited)`, run: () => chunkSource("all", null) }] : []),
        ]);
    }
    async function chunkSource(c, t) {              // t = null: back to the original; c = "all": every chunk
        const all = c === "all", nm = all ? "every chunk" : `chunk ${label(c.id)}`;
        toast(t || (all && t !== null) ? `making ${all ? "each chunk's take" : t.id} the source of ${nm} (builds the patched source, under a minute)…` : `${nm} back to the original…`);
        const body = all ? { op: t === null ? "chunk_source_revert" : "chunk_source", chunk: "all" }
            : t ? { op: "chunk_source", chunk: c.id, take: t.id } : { op: "chunk_source_revert", chunk: c.id };
        const r = await op(body, true);
        if (!r) return;
        video.dataset.src = "";                      // the same view, re-pointed at the rebuilt source (a view button is never changed)
        const sk = Object.keys(r.result.skipped || {}).length;
        toast(body.op === "chunk_source" ? `${nm} now renders from ${all ? "its take" : t.id}${sk ? ` (${sk} skipped: ${Object.entries(r.result.skipped).map(([k, v]) => `${label(k)} ${v}`).join("; ")})` : ""}: the other chunks keep their takes` : `${nm} renders from the original again`, "green");
        seek(S.playhead);
    }
    async function assembleNow() {
        toast("assembling (CPU)…");
        bAssemble.disabled = true;
        try {
            const r = await api.fetchApi("/seamstitch/swap/assemble", { method: "POST", body: JSON.stringify({ plan: S.view.path }) });
            const j = await r.json();
            if (j.error) throw new Error(j.error);
            toast(`assembled ${j.frames} frames in ${j.seconds}s${j.flags.length ? ` · ${j.flags.length} flag(s): ${j.flags.map(f => f.code).join(", ")}` : ""}`, j.flags.length ? "amber" : "green");
            await reload();
            setMode("full");
        } catch (e) { toast(`assemble: ${e.message || e}`, "red"); }
        bAssemble.disabled = false;
    }

    function onQueueEvent(type, d) {
        const pid = d?.prompt_id;
        const q = pid && S.queued[pid];
        if (type === "execution_start" && q) {
            S.running = pid; S.progress = null;
            if (q.kind === "render" && Object.entries(S.queued).some(([p, x]) => p !== pid && x.kind === "render")) freeMemory();
        }
        else if (type === "progress" && q) { S.progress = { value: d.value, max: d.max, node: d.node }; drawCanvas(); }
        else if (type === "execution_error" && q) {
            const msg = `${d.node_type || ""}: ${d.exception_message || "error"}`.trim();
            if (q.kind === "render") op({ op: "render_failed", chunk: q.run.chunk, nonce: q.run.nonce, error: msg }, true);
            delete S.queued[pid]; S.running = null;
            cancelRestOfBatch(q).then((gone) => {
                toast(`${q.kind}${q.run.chunk ? ` ${label(q.run.chunk)}` : ""} failed: ${msg}${gone.length ? ` · took chunk ${gone.map(label).join(", ")} off the queue (they pin to it)` : ""}`, "red");
                refresh();
            });
        } else if (type === "execution_interrupted" && q) {
            if (q.kind === "render") op({ op: "render_failed", chunk: q.run.chunk, nonce: q.run.nonce, error: "interrupted" }, true);
            delete S.queued[pid]; S.running = null;
            cancelRestOfBatch(q).then(() => refresh());
        } else if (type === "execution_success" && q) {
            delete S.queued[pid]; S.running = null; S.progress = null; reload();
        }
    }

    // ------------------------------------------------------------ prompts and takes
    async function savePrompt() {
        const c = S.sel?.kind === "chunk" ? chunks().find(x => x.id === S.sel.id) : null;
        if (!c || !S.promptBox) return;
        const v = S.promptBox.value;
        S.promptDirty = false;
        if (v !== (c.prompt || "")) await op({ op: "set_prompt", chunk: c.id, prompt: v }, true);
        toast(`prompt saved for chunk ${label(c.id)}`, "green");
    }

    // ------------------------------------------------------------ geometry helpers
    function f2x(f) { return f * S.pxPerFrame - S.scroll; }
    function x2f(x) { return (x + S.scroll) / S.pxPerFrame; }
    function shotsOf(c) {
        if (kept(c) || !c.render) return [];
        const [r0, r1] = c.render;
        const inner = confirmed().filter(f => f > r0 && f <= r1);
        const edges = [r0, ...inner, r1 + 1];
        return edges.slice(0, -1).map((a, i) => ({ a, b: edges[i + 1] - 1 }));
    }

    // ------------------------------------------------------------ drawing
    function refresh() {
        drawCanvas();
        drawSel();
        drawNext();
        drawWarnings();
        const has = !!P();
        emptyText.textContent = !job() ? "Load a video to start a job (its cuts and splits are found straight away),\nor pick a job from the list above."
            : `job ${job()} has no plan here${S.err ? ` (${S.err.split(":")[0]})` : ""}.\nLoad a video to start a job, or pick another from the list above.`;
        emptyHint.style.display = has ? "none" : "flex";
        for (const b of [bDetect, bCuts, bClear, bDraft, bMark, bPending, bAssemble]) b.disabled = !has;
        const u = S.view?.undo;
        bUndo.disabled = !has || !u?.n;
        bUndo.title = u?.n ? `Undo: ${u.last} (${u.n} edit${u.n === 1 ? "" : "s"} kept). Renders made since stay.` : "Nothing to undo yet: it undoes edits made from this panel (the last 50 are kept).";
        if (!S.jobs || S.jobsFor !== job()) { S.jobsFor = job(); refreshJobs(); }
        else jobSel.value = S.jobs.some(j => j.job === job()) ? job() : "";
        const src = P()?.source;
        videoName.textContent = src ? src.path.split(/[\\/]/).pop() : "none";
        videoName.title = src ? src.path : "";
        videoFacts.textContent = src ? `${src.frames} frames · ${src.fps} fps · ${src.width}×${src.height} · ${(src.frames / src.fps).toFixed(1)} s` : "";
        if (document.activeElement !== targetIn) targetIn.value = P()?.target || "";
        const jm = jobMode();
        for (const [m, b] of Object.entries(modeBtns)) {
            const on = has && m === jm;
            Object.assign(b.style, { background: on ? (m === "background" ? "#2f6f8f" : m === "object" ? "#7a5a2f" : "#3f7f4f") : "#1b1f27",
                color: on ? "#fff" : C.dim, fontWeight: on ? "600" : "400" });
            b.disabled = !has;
        }
        targetLbl.textContent = jm === "background" ? "keep" : jm === "object" ? "target" : "replace";
        objBox.style.display = has && jm === "object" ? "flex" : "none";
        if (jm === "object") {
            const o = Object.assign({ action: "replace", feather: 6 }, P().object || {});
            const grow = (P().object || {}).grow ?? (o.action === "add" ? 60 : 12);
            if (document.activeElement !== objSel) objSel.value = o.action;
            if (document.activeElement !== objGrow) objGrow.value = grow;
            if (document.activeElement !== objFeather) objFeather.value = o.feather;
            if (document.activeElement !== objFill) objFill.value = o.fill || "render";
            objPic.textContent = o.picture ? `picture: ${String(o.picture).split(/[\\/]/).pop()}` : "picture…";
            objPic.style.display = (o.fill || "render") === "picture" ? "" : "none";
            // the zone row: an add by H3 render; the zone's own controls only when its guide is the zone
            const isAdd = o.action === "add" && (o.fill || "render") === "render";
            zoneRow.style.display = has && isAdd ? "flex" : "none";
            const zv = Object.assign({ guide: "composite", size: 1.3, zone_shape: "ellipse", zone_aspect: 1, zone_dx: 0, zone_dy: 0, zone_grow: 40, zone_spare: false,
                zone_touch_dx: 0, zone_touch_dy: 0, zone_touch_r: 0, anchor_at: 0, enter_at: 0, level: "ring" }, P().object || {});
            const zoneOn = zv.guide === "zone";
            for (const c of zoneBits) c.style.display = zoneOn ? "" : "none";
            if (document.activeElement !== zBrief) zBrief.value = zv.brief || "";
            for (const [c, v] of [[zGuide, zv.guide === "zone" ? "zone" : "composite"], [zShape, zv.zone_shape], [zLevel, zv.level], [zAnch, zv.anchor || "qwen"]]) if (document.activeElement !== c) c.value = v;
            zPic.style.display = zoneOn && (zv.anchor || "qwen") === "picture" ? "" : "none";
            zPic.textContent = zv.anchor_picture ? `anchor: ${String(zv.anchor_picture).split(/[\/]/).pop()}` : "anchor picture…";
            for (const [c, v] of [[zSize, zv.size], [zAspect, zv.zone_aspect], [zDx, zv.zone_dx], [zDy, zv.zone_dy], [zGrow, zv.zone_grow], [zTx, zv.zone_touch_dx],
                [zTy, zv.zone_touch_dy], [zTr, zv.zone_touch_r], [zAnchor, zv.anchor_at], [zEnter, zv.enter_at]]) if (document.activeElement !== c) c.value = v;
            zSpare.checked = !!zv.zone_spare;
        }
        guideBox.style.display = has && jm === "character" ? "flex" : "none";
        for (const [g, b] of Object.entries(guideBtns)) {
            const on = g === jobGuide();
            Object.assign(b.style, { background: on ? "#4a5468" : "#1b1f27", color: on ? "#fff" : C.dim, fontWeight: on ? "600" : "400" });
        }
        targetIn.title = jm === "background"
            ? "Who to keep, as a short noun phrase (\"man in a black t-shirt\"): what SAM3 marks in the source (everything else is replaced), and who the drafter writes about. Empty: SAM3 marks \"person\" and the drafter picks. Changing it needs a re-mark and a redraft."
            : "Who to replace, as a short noun phrase (\"woman in a purple top\"): what SAM3 marks in the source, and who the drafter writes about. Empty: SAM3 marks \"person\" and the drafter picks. Changing it needs a re-mark and a redraft.";
        targetIn.disabled = !has;
        bMode.textContent = { source: "source", quick: "quick", full: "full ●", mask: "mask: character", mask_bg: "mask: background", mask_obj: "mask: object" }[S.mode]
            || `keep: ${objectKeeps().find(k => `keep:${k.id}` === S.mode)?.target || "?"}`;
        // the background mask is the character one with every pixel inverted (the person back to the original,
        // everything else marked): the same preview video, inverted in the player
        video.style.filter = S.mode === "mask_bg" ? "invert(1)" : "";
        bMode.title = "View: source (the original) · quick (each chunk's effective take, chained) · full (the latest assembly) · mask (the cached person mask)";
        updateOverlay();
        zoneSoon();
        saveUiSoon();
    }

    function drawCanvas() {
        const w = Math.max(50, canvas.clientWidth | 0);
        const k = (window.devicePixelRatio || 1) * Math.max(1, Math.min(4, app.canvas?.ds?.scale || 1));
        const bw = Math.round(w * k), bh = Math.round(CANVAS_H * k);
        if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
        const g = canvas.getContext("2d");
        g.setTransform(k, 0, 0, k, 0, 0);
        g.fillStyle = C.bg; g.fillRect(0, 0, w, CANVAS_H);
        // row backgrounds
        g.fillStyle = C.row; g.fillRect(0, SRC_Y, w, RB_Y + 2 * RB_H - SRC_Y);
        g.fillStyle = C.rowAlt; g.fillRect(0, MASK_Y, w, MASK_H);
        g.fillStyle = C.row; g.fillRect(0, PR_Y, w, PR_H);
        if (!P()) return;
        const N = S.N, fr = S.fr, sel = S.sel;
        // ruler: seconds labels, frame ticks when zoomed in
        g.fillStyle = C.ruler; g.fillRect(0, 0, w, RULER_H);
        const secPx = fr * S.pxPerFrame;
        const stepSec = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60].find(s => s * secPx >= 50 * U) || 120;
        g.strokeStyle = C.tick; g.fillStyle = C.tick; g.font = font(10); g.textBaseline = "top"; g.lineWidth = 1;
        if (S.pxPerFrame >= 4) for (let f = Math.max(0, Math.floor(x2f(0))); f <= Math.min(N, x2f(w)); f++) {
            const x = Math.round(f2x(f)) + 0.5; g.beginPath(); g.moveTo(x, RULER_H - 3 * U); g.lineTo(x, RULER_H); g.stroke();
        }
        for (let s = Math.max(0, Math.floor(x2f(0) / fr / stepSec) * stepSec); s * fr <= Math.min(N, x2f(w)); s += stepSec) {
            const x = Math.round(f2x(s * fr)) + 0.5;
            g.beginPath(); g.moveTo(x, RULER_H - 7 * U); g.lineTo(x, RULER_H); g.stroke();
            g.fillText(stepSec < 1 ? `${s.toFixed(2)}s` : `${s}s`, x + 2, 2 * U);
        }
        // the source's end
        g.fillStyle = "#000"; g.fillRect(f2x(N), SRC_Y, Math.max(0, w - f2x(N)), CANVAS_H - SRC_Y);

        // chunk blocks
        g.textBaseline = "middle";
        const cs = chunks();
        cs.forEach((c, i) => {
            const st = statusOf(c.id);
            const x0 = f2x(c.deliver[0]), x1 = f2x(c.deliver[1] + 1);
            if (x1 < 0 || x0 > w) return;
            const bw_ = Math.max(1, x1 - x0 - 2);
            if (kept(c)) {                       // kept as the original: hatched, no render, no dots
                hatch(g, x0 + 1, BLOCK_Y, bw_, BLOCK_H);
                const selected = sel?.kind === "chunk" && sel.id === c.id;
                g.lineWidth = selected ? 2.5 : 1; g.strokeStyle = selected ? C.sel : "#2f4f40";
                g.strokeRect(x0 + 1.5, BLOCK_Y + 0.5, bw_ - 1, BLOCK_H - 1); g.lineWidth = 1;
                g.save(); g.beginPath(); g.rect(x0 + 3, BLOCK_Y, Math.max(0, bw_ - 4), BLOCK_H); g.clip();
                const n = c.deliver[1] - c.deliver[0] + 1;
                g.fillStyle = C.keptText; g.font = font(11, true); g.fillText(`${i + 1} original`, x0 + 6, BLOCK_Y + 9 * U);
                g.font = font(10); g.fillStyle = C.dim;
                g.fillText(`${n}f · ${(n / S.fr).toFixed(2)}s · kept: never rendered`, x0 + 6, BLOCK_Y + 22 * U);
                g.restore();
                return;
            }
            const q = queuedFor(c.id);
            let fill = C.pending, mark = "";
            if (st.state === "unreviewed") { fill = C.unreviewed; mark = "?"; }
            else if (st.state === "chosen") { fill = C.chosen; mark = "✓"; }
            else if (st.state === "range changed") fill = C.amberFill;
            else if (st.state === "missing file") fill = C.redFill;
            g.fillStyle = fill; g.fillRect(x0 + 1, BLOCK_Y, bw_, BLOCK_H);
            if (st.rendering || q) {          // queued / rendering: striped, with progress
                g.save(); g.beginPath(); g.rect(x0 + 1, BLOCK_Y, bw_, BLOCK_H); g.clip();
                g.strokeStyle = "rgba(255,255,255,0.12)"; g.lineWidth = 6 * U;
                for (let hx = x0 - BLOCK_H; hx < x1; hx += 14 * U) { g.beginPath(); g.moveTo(hx, BLOCK_Y + BLOCK_H); g.lineTo(hx + BLOCK_H, BLOCK_Y); g.stroke(); }
                g.restore(); g.lineWidth = 1;
                const running = S.running && S.queued[S.running]?.run.chunk === c.id;
                if (running && S.progress?.max) {
                    g.fillStyle = C.play; g.fillRect(x0 + 1, BLOCK_Y + BLOCK_H - 3 * U, bw_ * S.progress.value / S.progress.max, 3 * U);
                }
            }
            const empty = !(c.prompt || "").trim();
            const selected = sel?.kind === "chunk" && sel.id === c.id;
            g.lineWidth = selected ? 2.5 : empty ? 2 : 1;
            g.strokeStyle = selected ? C.sel : empty ? C.red : st.state === "range changed" ? C.amber : st.state === "missing file" ? C.red : "#4b5563";
            g.strokeRect(x0 + 1.5, BLOCK_Y + 0.5, bw_ - 1, BLOCK_H - 1); g.lineWidth = 1;
            // text
            g.save(); g.beginPath(); g.rect(x0 + 3, BLOCK_Y, Math.max(0, bw_ - 4), BLOCK_H); g.clip();
            const n = c.deliver[1] - c.deliver[0] + 1;
            g.fillStyle = C.text; g.font = font(11, true);
            g.fillText(`${i + 1}${mark ? " " + mark : ""}`, x0 + 6, BLOCK_Y + 9 * U);
            g.font = font(10); g.fillStyle = "#d1d5db";
            const fill_ = c.fill ? ` +${c.fill.kind} ${c.fill.frames}` : "";
            g.fillText(`${n}f · ${(n / S.fr).toFixed(2)}s`, x0 + 6 + 22 * U, BLOCK_Y + 9 * U);
            g.fillStyle = C.dim;
            g.fillText(`render ${c.length}f${fill_} · ${st.takes || 0} take${st.takes === 1 ? "" : "s"}${st.rendering ? " · rendering" : q ? " · queued" : ""}`, x0 + 6, BLOCK_Y + 22 * U);
            // flag dots: F following, R replaced, C cuts, M mouth (grey / amber / green: information only), S the scene alarm
            const fl = st.flags || {};
            [["F", flagColour(fl.following)], ["R", flagColour(fl.replaced)], ["C", flagColour(fl.cuts)], ["M", flagColour(fl.mouth)],
                ...(fl.scene === "amber" ? [["S", C.amber]] : [])].forEach(([t, col], k) => {
                const dx = x0 + 8 * U + k * 22 * U, dy = BLOCK_Y + 33 * U;
                g.fillStyle = col; g.beginPath(); g.arc(dx, dy, 3.5 * U, 0, Math.PI * 2); g.fill();
                g.fillStyle = C.dim; g.font = font(8, true); g.fillText(t, dx + 5 * U, dy);
            });
            g.restore();
            if (st.failed) {                    // failed: red badge (the error on hover)
                g.fillStyle = C.red; g.beginPath(); g.arc(x1 - 9 * U, BLOCK_Y + 9 * U, 6 * U, 0, Math.PI * 2); g.fill();
                g.fillStyle = "#111"; g.font = font(9, true); g.textAlign = "center"; g.fillText("!", x1 - 9 * U, BLOCK_Y + 9.5 * U); g.textAlign = "left";
            }
            // render range underneath: overlap and fill (two lanes so neighbours' overlaps don't hide each other)
            const lane = RB_Y + (i % 2) * RB_H;
            const [r0, r1] = c.render;
            g.fillStyle = C.render; g.fillRect(f2x(r0), lane, f2x(r1 + 1) - f2x(r0), RB_H - 1);
            if (c.fill) {
                g.fillStyle = c.fill.kind === "hold" ? C.held : C.fill;
                if (c.fill.kind === "tail") g.fillRect(f2x(r1 + 1 - c.fill.frames), lane, c.fill.frames * S.pxPerFrame, RB_H - 1);
                else if (c.fill.kind === "head") g.fillRect(f2x(r0), lane, c.fill.frames * S.pxPerFrame, RB_H - 1);
                else g.fillRect(f2x(r1 + 1), lane, c.fill.frames * S.pxPerFrame, RB_H - 1);
            }
        });

        // mask row: cached segments (newest drawn last), then the uncovered part of each chunk hatched
        const masks = personMasks().slice().sort((a, b) => String(a.created).localeCompare(String(b.created)));
        masks.forEach((m, i) => {
            g.fillStyle = i === masks.length - 1 ? C.mask : C.maskOld;
            g.fillRect(f2x(m.range[0]), MASK_Y + 2, f2x(m.range[1] + 1) - f2x(m.range[0]), MASK_H - 4);
            g.strokeStyle = "#0f2b20"; g.strokeRect(f2x(m.range[0]) + 0.5, MASK_Y + 2.5, f2x(m.range[1] + 1) - f2x(m.range[0]) - 1, MASK_H - 5);
        });
        g.font = font(9); g.fillStyle = C.text;
        cs.forEach((c, i) => {
            const st = statusOf(c.id), x0 = f2x(c.deliver[0]), x1 = f2x(c.deliver[1] + 1);
            if (x1 < 0 || x0 > w) return;
            if (kept(c)) {
                hatch(g, x0 + 1, MASK_Y + 1, x1 - x0 - 2, MASK_H - 2);
                g.save(); g.beginPath(); g.rect(x0 + 2, MASK_Y, Math.max(0, x1 - x0 - 4), MASK_H); g.clip();
                g.fillStyle = C.keptText; g.fillText("original: no mask needed", x0 + 5, MASK_Y + MASK_H / 2); g.restore();
                return;
            }
            const pct = Math.round((st.mask || 0) * 100);
            const inD = (f) => f >= c.deliver[0] && f <= c.deliver[1];
            const lost = (st.mask_empty || []).filter(inD), filled = (st.mask_filled || []).filter(inD);
            const span = (a) => `${a[0]}${a.length > 1 ? `-${a.at(-1)}` : ""}`;
            g.fillStyle = lost.length ? C.red : pct >= 100 ? "#d1fae5" : pct ? C.amber : C.faint;
            g.save(); g.beginPath(); g.rect(x0 + 2, MASK_Y, Math.max(0, x1 - x0 - 4), MASK_H); g.clip();
            g.fillText((pct >= 100 ? `mask ✓${(c.options || {}).mark === false ? " (marking off)" : ""}` : pct ? `mask ${pct}%` : "no mask: the render tracks its own")
                + (lost.length ? ` · no person on ${lost.length} frame${lost.length === 1 ? "" : "s"} (${span(lost)})` : "")
                + (filled.length ? ` · ${filled.length} filled (${span(filled)})` : ""), x0 + 5, MASK_Y + MASK_H / 2);
            g.restore();
            // frames where SAM3 found nobody: red where still empty, amber where a short hole was filled
            // from the frames either side, so a lost track shows before any render
            for (const [fs, col] of [[filled, C.amber], [lost, C.red]]) {
                g.fillStyle = col;
                for (const f of fs) g.fillRect(f2x(f), MASK_Y + 1, Math.max(1.5, S.pxPerFrame), MASK_H - 2);
            }
        });

        // prompt row: [Shot n] blocks at the confirmed cuts, dialogue in them
        cs.forEach((c) => {
            const st = statusOf(c.id), x0 = f2x(c.deliver[0]), x1 = f2x(c.deliver[1] + 1);
            if (x1 < 0 || x0 > w) return;
            if (kept(c)) {
                hatch(g, x0 + 1, PR_Y + 1, x1 - x0 - 2, PR_H - 2);
                g.save(); g.beginPath(); g.rect(x0 + 2, PR_Y, Math.max(0, x1 - x0 - 4), PR_H); g.clip();
                g.font = font(10); g.fillStyle = C.keptText; g.fillText("original: no prompt needed", x0 + 6, PR_Y + 9 * U);
                if (sel?.kind === "chunk" && sel.id === c.id) { g.strokeStyle = C.sel; g.lineWidth = 2; g.strokeRect(x0 + 2, PR_Y + 1, x1 - x0 - 4, PR_H - 2); g.lineWidth = 1; }
                g.restore();
                return;
            }
            const empty = !(c.prompt || "").trim();
            const shots = shotsOf(c), blocks = parseShots(c.prompt);
            g.save(); g.beginPath(); g.rect(x0 + 1, PR_Y, Math.max(0, x1 - x0 - 2), PR_H); g.clip();
            g.fillStyle = empty ? "#2a1719" : st.prompt === "draft" ? C.promptDraft : C.prompt;
            g.fillRect(x0 + 1, PR_Y + 1, x1 - x0 - 2, PR_H - 2);
            if (empty) {
                g.fillStyle = C.red; g.beginPath(); g.arc(x0 + 9 * U, PR_Y + 9 * U, 3.5 * U, 0, Math.PI * 2); g.fill();
                g.font = font(10); g.fillText(st.draft ? "no prompt (a draft is waiting: adopt it)" : "no prompt: render blocked", x0 + 16 * U, PR_Y + 9 * U);
            } else {
                const mismatch = blocks.length && blocks.length !== shots.length;
                shots.forEach((s, k) => {
                    const sx0 = Math.max(f2x(s.a), x0 + 1), sx1 = Math.min(f2x(s.b + 1), x1 - 1);
                    if (k > 0) { g.fillStyle = C.cut; g.fillRect(f2x(s.a), PR_Y + 2, 1, PR_H - 4); }
                    if (sx1 - sx0 < 6) return;
                    const blk = blocks[k];
                    g.save(); g.beginPath(); g.rect(sx0 + 2, PR_Y, sx1 - sx0 - 4, PR_H); g.clip();
                    g.font = font(9, true); g.fillStyle = blk ? (st.prompt === "draft" ? C.draft : C.text) : C.amber;
                    g.fillText(blk ? `Shot ${blk.n}` : `shot ${k + 1}: no block`, sx0 + 4, PR_Y + 8 * U);
                    g.font = font(9); g.fillStyle = C.dim;
                    if (blk) {
                        g.fillText(blk.text.slice(0, 160), sx0 + 4, PR_Y + 20 * U);
                        if (blk.dialogue.length) { g.fillStyle = "#c4b5fd"; g.fillText(`“${blk.dialogue.join(" … ")}”`, sx0 + 4, PR_Y + 32 * U); }
                    }
                    g.restore();
                });
                if (!blocks.length) { g.font = font(9); g.fillStyle = C.dim; g.fillText((c.prompt || "").replace(/\s+/g, " ").slice(0, 200), x0 + 5, PR_Y + PR_H / 2); }
                if (mismatch) { g.fillStyle = C.amber; g.font = font(9, true); g.textAlign = "right"; g.fillText(`${blocks.length} shot blocks / ${shots.length} shots`, x1 - 4, PR_Y + 8 * U); g.textAlign = "left"; }
                if (st.prompt === "draft") { g.fillStyle = C.draft; g.font = font(9, true); g.textAlign = "right"; g.fillText("draft", x1 - 4, PR_Y + PR_H - 7 * U); g.textAlign = "left"; }
            }
            if (sel?.kind === "chunk" && sel.id === c.id) { g.strokeStyle = C.sel; g.lineWidth = 2; g.strokeRect(x0 + 2, PR_Y + 1, x1 - x0 - 4, PR_H - 2); g.lineWidth = 1; }
            g.restore();
        });

        // cut ticks: faint = suggested, solid = confirmed (a triangle on the ruler to grab)
        for (const c of cuts()) {
            const x = Math.round(f2x(c.frame)) + 0.5;
            if (x < -10 || x > w + 10) continue;
            const conf = c.confirmed !== false, hot = (S.hover?.kind === "cut" && S.hover.frame === c.frame) || (sel?.kind === "cut" && sel.frame === c.frame);
            g.strokeStyle = conf ? C.cut : C.cutSug; g.lineWidth = conf ? 1 : 1;
            if (!conf) g.setLineDash([3 * U, 3 * U]);
            g.beginPath(); g.moveTo(x, SRC_Y); g.lineTo(x, BLOCK_Y + BLOCK_H); g.stroke();
            g.setLineDash([]);
            g.fillStyle = hot ? C.play : conf ? C.cut : C.cutSug;
            g.beginPath(); g.moveTo(x - 4 * U, RULER_H - 8 * U); g.lineTo(x + 4 * U, RULER_H - 8 * U); g.lineTo(x, RULER_H); g.closePath();
            if (conf || hot) g.fill(); else { g.strokeStyle = C.cutSug; g.stroke(); }
        }

        // splits: a bar through every row, a pill on top (mode, join verdict, F / E / X, stale, ⚠)
        for (const s of splits()) {
            const dragging = S.drag?.kind === "split" && S.drag.id === s.id;
            const f = dragging ? S.drag.to : s.frame;
            const x = Math.round(f2x(f)) + 0.5;
            if (x < -20 || x > w + 20) continue;
            const j = joinAt(s.id), warn = warnsFor("split", s.id).length > 0;
            const selected = sel?.kind === "split" && sel.id === s.id;
            g.strokeStyle = selected || dragging ? C.sel : s.mode === "cut" ? "#f3f4f6" : "#a5b4fc";
            g.lineWidth = selected || dragging ? 2.5 : 1.5;
            g.beginPath(); g.moveTo(x, PILL_Y); g.lineTo(x, CANVAS_H); g.stroke(); g.lineWidth = 1;
            let fill = "#6b7280", txt = "·", tcol = "#111";
            if (s.mode === "cut") { fill = "#f3f4f6"; txt = "✂"; }
            else if (j?.stale) { fill = C.amber; txt = "stale"; }
            else if (j && j.linked) { fill = VERDICT[j.verdict] || "#93c5fd"; txt = TYPE_LABEL[j.type] || "A"; }
            else if (j && j.type === "pending") { fill = "#6b7280"; txt = "A"; tcol = "#e5e7eb"; }
            else if (j && j.type === "original") { fill = C.kept; txt = "="; tcol = C.keptText; }
            const pw = (txt.length > 1 ? 30 : 16) * U, ph = 13 * U;
            g.fillStyle = fill;
            roundRect(g, x - pw / 2, PILL_Y - ph / 2, pw, ph, ph / 2); g.fill();
            g.fillStyle = tcol; g.font = font(9, true); g.textAlign = "center"; g.fillText(txt, x, PILL_Y + 0.5); g.textAlign = "left";
            if (warn) { g.fillStyle = C.amber; g.font = font(11, true); g.fillText("⚠", x + pw / 2 + 2, PILL_Y + 0.5); }
            if (dragging) { g.fillStyle = C.sel; g.font = font(10, true); g.fillText(`${f}${S.drag.snap ? " (cut)" : ""}`, x + 6, BLOCK_Y + BLOCK_H - 6 * U); }
        }
        // trim handles (keep original at the start / end): a grip on each side of the strip
        const tr = trims();
        for (const [edge, f, has] of [["start", tr.start, tr.hasStart], ["end", tr.end, tr.hasEnd]]) {
            const dragging = S.drag?.kind === "trim" && S.drag.edge === edge;
            const ff = dragging ? S.drag.to : f;
            const x = f2x(ff), gw = 9 * U;
            const gx = edge === "start" ? (ff > 0 ? x - gw : x) : (ff < S.N ? x : x - gw);
            const hot = dragging || (S.hover?.kind === "trim" && S.hover.edge === edge);
            if (dragging) {                     // the stretch that would be kept
                const a = edge === "start" ? 0 : ff, b = edge === "start" ? ff : S.N;
                hatch(g, f2x(a), BLOCK_Y, f2x(b) - f2x(a), BLOCK_H);
                g.fillStyle = C.sel; g.font = font(10, true);
                g.fillText(`${edge === "start" ? `keep 0-${ff - 1}` : `keep ${ff}-${S.N - 1}`}${S.drag.snap ? " (cut)" : ""}`, Math.max(4, x + (edge === "start" ? 6 : -110 * U)), BLOCK_Y + BLOCK_H + 9 * U);
            }
            g.fillStyle = hot ? C.sel : has ? C.trim : "rgba(52,211,153,0.55)";
            roundRect(g, gx, BLOCK_Y + 4 * U, gw, BLOCK_H - 8 * U, 3 * U); g.fill();
            g.fillStyle = "#0b1f17"; g.font = font(9, true); g.textAlign = "center";
            g.fillText(edge === "start" ? "⟦" : "⟧", gx + gw / 2, BLOCK_Y + BLOCK_H / 2); g.textAlign = "left";
        }
        // a split being dragged: its overlap guard, so a cut inside it shows before the drop
        if (S.drag?.kind === "split") {
            const s = splitById(S.drag.id);
            if (s && s.mode !== "cut") {
                const st = settings(), lo = S.drag.to - st.overlap - st.guard, hi = S.drag.to + st.guard;
                g.fillStyle = "rgba(251,191,36,0.12)"; g.fillRect(f2x(lo), BLOCK_Y, f2x(hi + 1) - f2x(lo), BLOCK_H);
            }
        }
        // playhead
        const px = Math.round(f2x(S.playhead)) + 0.5;
        g.strokeStyle = C.play; g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(px, 0); g.lineTo(px, CANVAS_H); g.stroke(); g.lineWidth = 1;
        g.fillStyle = C.play; g.beginPath(); g.moveTo(px - 5 * U, 0); g.lineTo(px + 5 * U, 0); g.lineTo(px, 7 * U); g.fill();
        // row tags
        g.font = font(8, true); g.fillStyle = "rgba(156,163,175,0.6)";
        g.fillText("MASK", 3, MASK_Y - 3 * U); g.fillText("PROMPT", 3, PR_Y - 2 * U);
    }
    function hatch(g, x, y, w, h) {
        if (w <= 0) return;
        g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
        g.fillStyle = C.kept; g.fillRect(x, y, w, h);
        g.strokeStyle = C.keptHatch; g.lineWidth = 3 * U;
        for (let hx = x - h; hx < x + w; hx += 9 * U) { g.beginPath(); g.moveTo(hx, y + h); g.lineTo(hx + h, y); g.stroke(); }
        g.restore(); g.lineWidth = 1;
    }
    function roundRect(g, x, y, w, h, r) {
        g.beginPath(); g.moveTo(x + r, y); g.lineTo(x + w - r, y); g.arcTo(x + w, y, x + w, y + r, r); g.lineTo(x + w, y + h - r);
        g.arcTo(x + w, y + h, x + w - r, y + h, r); g.lineTo(x + r, y + h); g.arcTo(x, y + h, x, y + h - r, r); g.lineTo(x, y + r); g.arcTo(x, y, x + r, y, r); g.closePath();
    }

    // ------------------------------------------------------------ the selection panel
    function drawSel() {
        selBox.innerHTML = "";
        S.promptBox = null;
        if (!P()) return;
        const s = S.sel;
        const row = () => el("div", { display: "flex", gap: "4px", alignItems: "center", flexWrap: "wrap" });
        if (!s) {
            const r = row();
            r.append(el("span", { color: C.dim }, "click a chunk (its prompt and takes), a split's pill (its menu) or a cut's triangle · drag a split to move it (snaps to frames and nearby confirmed cuts) · S split at the playhead · C cut at the playhead · space play · ←/→ step (shift 10) · wheel over the picture: step frames"));
            selBox.append(r);
            return;
        }
        if (s.kind === "cut") {
            const c = cuts().find(x => x.frame === s.frame);
            if (!c) return;
            const r = row();
            r.append(el("span", { fontWeight: "bold" }, `cut at ${c.frame} (${(c.frame / S.fr).toFixed(2)}s)`),
                el("span", { color: c.confirmed === false ? C.dim : C.green }, c.confirmed === false ? `suggested (${c.from}${c.weak ? `, faint: scene ${c.score}` : ""})` : `confirmed (${c.from})`),
                button(c.confirmed === false ? "confirm" : "unconfirm", "Only confirmed cuts drive the warnings, the fill and the lost-cut flag", () => op({ op: "confirm_cut", frame: c.frame, confirmed: c.confirmed === false })),
                button("−1", "Move the cut a frame earlier", () => moveCut(c, c.frame - 1)),
                button("+1", "Move the cut a frame later", () => moveCut(c, c.frame + 1)),
                button("split here", "A straight-cut split on this cut", () => op({ op: "add_split", frame: c.frame, mode: "cut" })),
                button("jump", "", () => seek(c.frame)),
                button("delete", "", () => op({ op: "delete_cut", frame: c.frame }).then(() => { S.sel = null; refresh(); })));
            selBox.append(r);
            return;
        }
        if (s.kind === "split") {
            const sp_ = splitById(s.id);
            if (!sp_) return;
            const j = joinAt(sp_.id);
            const r = row();
            const m = j?.measure;
            const jf = j?.flags || {};
            const meas = m ? ` · jump ${m.frame_luma?.at_splice ?? "?"} / char ${m.char_luma?.at_splice ?? "?"}${m.join_ratio != null ? `, ratio ${m.join_ratio}` : ""}${m.follow?.pose_iou != null ? `, follow ${m.follow.pose_iou}` : ""} (${m.source})`
                + (jf.verdict ? ` · colour ${jf.colour ?? "–"}, motion ${jf.motion ?? "–"}, following ${jf.following ?? "–"}, lineage ${jf.lineage ?? "–"} → ${jf.verdict}` : "") : "";
            r.append(el("span", { fontWeight: "bold" }, `split ${sp_.id} at ${sp_.frame}`),
                el("span", { color: C.dim }, `${sp_.mode === "cut" ? "straight cut" : "anchored"}${j ? ` · ${j.type}${j.left_take ? ` ${j.left_take} | ${j.right_take}` : ""} · splice ${j.splice} · ${j.override || j.repair}${j.hand_back !== settings().hand_back ? ` · hand-back ${j.hand_back}` : ""}` : ""}${meas}`),
                button(sp_.mode === "cut" ? "make anchored" : "make straight cut", "", () => op({ op: "split_mode", split: sp_.id, mode: sp_.mode === "cut" ? "anchored" : "cut" })),
                button("menu ▾", "Repair, hand-back, snap, re-roll to fit, delete", (ev) => splitMenu(ev, sp_)),
                button("−1", "", () => op({ op: "move_split", split: sp_.id, to: sp_.frame - 1 })),
                button("+1", "", () => op({ op: "move_split", split: sp_.id, to: sp_.frame + 1 })),
                button("jump", "", () => seek(sp_.frame)));
            selBox.append(r);
            for (const w of warnsFor("split", sp_.id)) {
                const wr = row();
                wr.append(el("span", { color: C.amber }, `⚠ ${w.text}`));
                for (const rem of w.remedies || []) {
                    if (rem.action === "straight_cut") wr.append(button(`straight cut at ${rem.frame}`, "", async () => { await op({ op: "move_split", split: sp_.id, to: rem.frame }); await op({ op: "split_mode", split: sp_.id, mode: "cut" }); }));
                    if (rem.action === "move") for (const [k, v] of Object.entries(rem.clear || {})) wr.append(button(`move ${k} to ${v}`, "The nearest frame whose overlap guard holds no cut", () => op({ op: "move_split", split: sp_.id, to: v })));
                    if (rem.action === "anchor_onto_original") wr.append(button("anchor onto the original", "The render pins its first (or last) frames to the source and fades in (or out) over the overlap (§4.10; untested on the GPU: T-KEEP)", () => op({ op: "split_mode", split: sp_.id, mode: "anchored" })));
                    if (rem.action === "move_to_cut") wr.append(button(`move to the cut at ${rem.frame}`, "A straight cut on a real source cut is clean", () => op({ op: "move_split", split: sp_.id, to: rem.frame })));
                }
                selBox.append(wr);
            }
            return;
        }
        // a chunk: prompt editor, options, seed mode, takes, buttons
        const c = chunks().find(x => x.id === s.id);
        if (!c) return;
        const st = statusOf(c.id), k = chunkIndex(c.id);
        const head = row();
        const n = c.deliver[1] - c.deliver[0] + 1;
        const keepBox = el("input"); keepBox.type = "checkbox"; keepBox.checked = kept(c);
        keepBox.onpointerdown = (e) => e.stopPropagation();
        keepBox.onchange = () => op({ op: "keep", chunk: c.id, keep: keepBox.checked });
        const keepL = el("label", { display: "flex", alignItems: "center", gap: "4px", marginLeft: "8px", color: C.keptText,
            fontWeight: "bold", padding: "1px 6px", border: `1px solid ${kept(c) ? C.trim : "#2f4f40"}`, borderRadius: "4px",
            background: kept(c) ? "#14281e" : "transparent", cursor: "pointer" });
        keepL.title = "Keep the original: never rendered, masked, drafted or prompted; the assembly shows the source here untouched";
        keepL.append(keepBox, el("span", null, "keep original"));
        if (kept(c)) {
            const tr_ = trims();
            const edge = k === 0 && tr_.hasStart ? "start" : k === chunks().length - 1 && tr_.hasEnd ? "end" : null;
            head.append(el("span", { fontWeight: "bold" }, `chunk ${k + 1} (${c.id})`),
                el("span", { color: C.keptText }, `original: delivers ${c.deliver[0]}-${c.deliver[1]} (${n}f · ${(n / S.fr).toFixed(2)}s) from the source, untouched; never rendered, masked or drafted`),
                keepL);
            if (edge) head.append(button("remove the trim", "Give these frames back to the next chunk (the split goes)", () => op({ op: "keep", edge, frame: edge === "start" ? 0 : S.N })));
            selBox.append(head);
            const note = row();
            note.append(el("span", { color: C.dim }, "Its joins: a straight cut (clean on a source cut; mid-shot it warns), or anchored onto the original (the render pins to the source and fades in or out over the overlap)."));
            selBox.append(note);
            return;
        }
        head.append(el("span", { fontWeight: "bold" }, `chunk ${k + 1} (${c.id})`),
            el("span", { color: C.dim }, `delivers ${c.deliver[0]}-${c.deliver[1]} (${n}f · ${(n / S.fr).toFixed(2)}s) · renders ${c.render[0]}-${c.render[1]}${c.fill ? ` +${c.fill.kind} ${c.fill.frames}` : ""} = ${c.length}f · ${st.state}${st.take ? ` (${st.take})` : ""}`));
        if (st.failed) { const f = el("span", { color: C.red }, `failed: ${st.failed.slice(0, 120)}`); f.title = st.failed; head.append(f, button("clear", "Clear the failed state", () => op({ op: "clear_state", chunk: c.id }))); }
        if (st.rendering && !queuedFor(c.id)) head.append(el("span", { color: C.amber }, "marked rendering (not in this page's queue)"), button("clear", "The render is gone (ComfyUI restarted?): clear its state", () => op({ op: "clear_state", chunk: c.id })));
        selBox.append(head);

        const acts = row();
        const pins = forecastPins(k, new Set([k]));
        const sideTxt = pins.start || pins.end ? `${pins.start ? `pins ${pins.start}` : "free start"} | ${pins.end ? `pins ${pins.end}` : "free end"}` : "free (no pins)";
        const hasTake = (c.takes || []).length > 0;
        acts.append(button(hasTake ? "re-roll" : "render", `Queue a render of this chunk (${sideTxt}). Seed: ${c.seed_mode && typeof c.seed_mode === "object" && Number.isInteger(c.seed_mode.fixed) ? `fixed ${c.seed_mode.fixed}` : "new"}`, () => reroll(c.id, 1)));
        const nSel = el("select", { background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px" });
        [2, 3, 4].forEach(v => { const o = el("option", null, `× ${v}`); o.value = v; nSel.append(o); });
        nSel.onpointerdown = (e) => e.stopPropagation();
        acts.append(button("re-roll × N", "N renders with N new seeds, all against the same neighbours (two-sided when both have takes)", () => reroll(c.id, +nSel.value, true)), nSel);
        const staleSides = [joinAt(c.left), S.joins.find(j => j.left_chunk === c.id)].filter(j => j?.stale);
        if (staleSides.length) acts.append(button("re-roll to fit", `A stale join at ${staleSides.map(j => j.frame).join(", ")}: re-render this chunk pinned to both neighbours' current takes`, () => reroll(c.id, 1, true)));
        acts.append(button("mark", "Queue the mark runs: track and cache this chunk's source mask of its target, one run per shot", () => markChunk(c)),
            el("span", { color: C.dim }, sideTxt));
        // options and seed mode
        const mk = el("input"); mk.type = "checkbox"; mk.checked = (c.options || {}).mark !== false;
        mk.onchange = () => op({ op: "set_options", chunk: c.id, options: { mark: mk.checked } });
        mk.onpointerdown = (e) => e.stopPropagation();
        const mkL = el("label", { display: "flex", alignItems: "center", gap: "3px", marginLeft: "8px" }); mkL.append(mk, el("span", null, "mark (SAM3 marked guide)"));
        const seedSel = el("select", { background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px" });
        [["new", "seed: new each take"], ["fixed", "seed: fixed"]].forEach(([v, t]) => { const o = el("option", null, t); o.value = v; seedSel.append(o); });
        const fixed = c.seed_mode && typeof c.seed_mode === "object";
        seedSel.value = fixed ? "fixed" : "new";
        const seedIn = el("input", { width: "9em", background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px", display: fixed ? "" : "none" });
        seedIn.type = "number"; seedIn.value = fixed ? c.seed_mode.fixed : "";
        seedSel.onpointerdown = seedIn.onpointerdown = (e) => e.stopPropagation();
        seedSel.onchange = () => { if (seedSel.value === "new") op({ op: "seed_mode", chunk: c.id, seed_mode: "new" }); else { seedIn.style.display = ""; seedIn.focus(); } };
        seedIn.onchange = () => { const v = parseInt(seedIn.value, 10); if (Number.isInteger(v) && v >= 0) op({ op: "seed_mode", chunk: c.id, seed_mode: { fixed: v } }); };
        acts.append(mkL, seedSel, seedIn, keepL);
        selBox.append(acts);
        // this chunk's own target (empty: the job's, above the player)
        const tgt = row();
        const ownT = el("input", { flex: "0 1 22em", background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px" });
        ownT.value = (c.options || {}).target || "";
        ownT.placeholder = P().target ? `the job's: ${P().target}` : "the job's (person)";
        ownT.title = "Who this chunk replaces, when it isn't the job's target (another scene, another person). Re-mark and redraft after a change.";
        ownT.onpointerdown = (e) => e.stopPropagation();
        ownT.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter") ownT.blur(); };
        ownT.onchange = () => op({ op: "set_options", chunk: c.id, options: { target: ownT.value.trim() } });
        tgt.append(el("span", { color: C.dim }, jobMode() === "background" ? "keep" : "replace"), ownT);
        selBox.append(tgt);
        selBox.append(keepSection(c));

        // the prompt editor (adopt draft, redraft)
        const pr = row();
        const box = colouredTextarea({ border: `1px solid ${(c.prompt || "").trim() ? "#444" : C.red}` }, "7.5em");
        pr.style.flex = "1 0 auto";
        pr.style.alignItems = "stretch";
        const ta = box.ta;
        ta.value = c.prompt || "";
        box.paint();
        ta.placeholder = "No prompt: a render won't start. Draft prompts, or write the six-section Ref2VA prompt with a [Shot n] block per shot.";
        ta.oninput = () => { S.promptDirty = true; pSave.style.borderColor = C.amber; };
        ta.onblur = () => { if (S.promptDirty) savePrompt(); };
        ta.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); savePrompt(); } };
        ta.onpointerdown = (e) => e.stopPropagation();
        ta.onwheel = (e) => e.stopPropagation();
        S.promptBox = ta;
        if (S.focusShot != null) {
            const blk = parseShots(ta.value)[S.focusShot];
            setTimeout(() => { ta.focus(); if (blk) { ta.setSelectionRange(blk.at, blk.at); ta.scrollTop = Math.max(0, ta.value.slice(0, blk.at).split("\n").length - 2) * 14; } }, 0);
            S.focusShot = null;
        }
        const side = el("div", { display: "flex", flexDirection: "column", gap: "3px" });
        const pSave = button("save", "Save the prompt (ctrl+enter, or leave the box)", () => savePrompt());
        const pBig = button("⤢ large", "Open the prompt in a large editor over the page (Esc or close puts it back)",
            () => openLargeEditor(ta, () => { box.paint(); S.promptDirty = true; savePrompt(); }));
        side.append(el("span", { color: st.prompt === "empty" ? C.red : st.prompt === "draft" ? C.draft : C.dim, fontWeight: "bold" }, st.prompt === "empty" ? "● no prompt" : st.prompt === "draft" ? "draft" : "edited"), pSave, pBig);
        if (c.draft) {
            const dv = button("adopt draft", "Replace the prompt with the waiting draft (compare in the tooltip)", () => op({ op: "adopt_draft", chunk: c.id }));
            dv.title = `The draft:\n\n${c.draft.slice(0, 1500)}`;
            side.append(dv);
        }
        side.append(button("redraft", "Queue a draft run of this chunk: SeamStitch Swap Draft Prompts drafts it into the prompt if that's empty, otherwise into the draft field (never over your prompt)", () => queueDraft(c.id)));
        pr.append(box.wrap, side);
        selBox.append(pr);
        // what the last draft said about this chunk (the draft report, per chunk): when it ran, where it went, and every note or warning
        if (c.drafted) {
            const nts = c.drafted.warnings || [];
            const rep = el("div", { fontSize: "0.9em", color: nts.some(w => /could not|unusable|absent|no brief|wrote no|lacked/i.test(w)) ? C.red : C.dim, whiteSpace: "pre-wrap", padding: "0 6px 2px" },
                `last draft ${String(c.drafted.at || "").replace("T", " ")} → ${c.drafted.into || "?"}` + (nts.length ? "\n" + nts.map(w => "• " + w).join("\n") : " · no warnings"));
            selBox.append(rep);
        }

        // a zone add whose anchor Qwen draws: this chunk's instruction (empty: a plain sentence is written from the target and the zone)
        const ob = P().object || {};
        if (jobMode() === "object" && ob.action === "add" && (ob.fill || "render") === "render" && ob.guide === "zone" && (ob.anchor || "qwen") === "qwen") {
            const qr = el("div", { display: "flex", flexDirection: "column", gap: "2px", padding: "2px 0" });
            qr.append(el("span", { color: C.dim }, "Qwen anchor instruction (this chunk): what Qwen draws into the anchor frame, from Picture 1 (image 2)"));
            const qa = el("textarea", { width: "100%", minHeight: "4.5em", background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px", font: "inherit", boxSizing: "border-box" });
            qa.value = c.anchor_text || "";
            qa.placeholder = "Empty: a plain sentence is used (\"Add the person or thing from image 2 into image 1, to the left of the target ...\"). Say what goes where: \"Add the man from image 2 standing left of the woman, his hand on her shoulder. Keep her, the room and the light unchanged.\"";
            qa.onpointerdown = (e) => e.stopPropagation(); qa.onwheel = (e) => e.stopPropagation();
            qa.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) qa.blur(); };
            qa.onchange = () => op({ op: "set_anchor_text", chunk: c.id, text: qa.value });
            qr.append(qa);
            selBox.append(qr);
        }

        // who says each line: the drafter reads whose mouth moves; where it couldn't see a face, pick here
        const lines = promptLines(c.prompt);
        if (lines.length) {
            const lr = el("div", { display: "flex", flexDirection: "column", gap: "2px", padding: "2px 0" });
            lr.append(el("span", { color: C.dim }, "who says each line (the prompt's speaker; H3 moves that person's lips)"));
            const speakers = promptSpeakers(c.prompt);
            lines.forEach((ln, i) => {
                const r = row();
                const sel = el("select", { background: "#111", color: C.text, border: "1px solid #3b4252", borderRadius: "4px", maxWidth: "16em" });
                for (const s of speakers.includes(ln.who) ? speakers : [ln.who, ...speakers]) { const o = el("option", null, s); o.value = s; sel.append(o); }
                sel.value = ln.who;
                sel.onpointerdown = (e) => e.stopPropagation();
                sel.onchange = async () => { if (S.promptDirty) await savePrompt(); op({ op: "set_line_speaker", chunk: c.id, line: i, speaker: sel.value }); };
                r.append(sel, el("span", { color: C.dim }, `(${ln.id})`), el("span", { color: "#67e8f9" }, `"${ln.text.trim()}"`));
                lr.append(r);
            });
            selBox.append(lr);
        }

        // the subject: one per job, shared by every chunk's subject_definitions (an edit replaces it in each prompt)
        const sr = row();
        const sbox = colouredTextarea({ border: "1px solid #444", fontSize: "0.9em" }, "3.2em");
        const subj = sbox.ta;
        subj.value = P().subject || "";
        sbox.paint();
        subj.placeholder = "Subject (one per job): drafted from the sheet by Draft Prompts, or write '<Subject 1> (S1) is the ... whose motion comes from <Video 1> and whose appearance comes from <Picture 1>: ...'";
        subj.title = "The job's subject, shared by every chunk: saving an edit replaces the old subject text in every chunk's prompt and draft";
        const saveSubj = () => { if (subj.value.trim() !== (P().subject || "").trim()) op({ op: "set_subject", subject: subj.value }, true).then(r => { if (r) toast(`subject saved${r.result?.subject_replaced_in ? `, replaced in ${r.result.subject_replaced_in} prompt(s)` : ""}`, "green"); }); };
        subj.onblur = saveSubj;
        subj.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveSubj(); } };
        subj.onpointerdown = (e) => e.stopPropagation();
        subj.onwheel = (e) => e.stopPropagation();
        const sSide = el("div", { display: "flex", flexDirection: "column", gap: "3px" });
        sSide.append(el("span", { color: (P().subject || "").trim() ? C.dim : C.amber, fontWeight: "bold" }, "subject"));
        if ((P().subject_draft || "").trim()) {
            const ad = button("adopt subject draft", "Replace the subject (and its text in every chunk's prompt) with the waiting draft", () => op({ op: "adopt_subject_draft" }));
            ad.title = `The subject draft:

${P().subject_draft}`;
            sSide.append(ad);
        }
        sr.append(sbox.wrap, sSide);
        selBox.append(sr);

        // the takes list: chosen radio, ▶ in context, use this prompt and seed, delete to trash
        // only the job's mode's takes: the other mode's are kept, and come back with the switch
        // the same rule as swap_plan.take_mode: the take's own render options carry its mode; an older take is told by invert
        const takeMode = (t) => {
            const o = t.settings?.options || {};
            if (["character", "background", "object"].includes(o.mode)) return o.mode;
            return (o.invert === true || String(o.invert).toLowerCase() === "true") ? "background" : "character";
        };
        const takes = (c.takes || []).filter(t => takeMode(t) === jobMode());
        const otherTakes = (c.takes || []).length - takes.length;
        const otherModes = [...new Set((c.takes || []).map(takeMode).filter(m => m !== jobMode()))].join(" / ");
        if (otherTakes) selBox.append(el("div", { color: C.dim, fontSize: "0.9em", padding: "0 6px" },
            `${otherTakes} ${otherModes} take${otherTakes === 1 ? "" : "s"} kept (switch the job's mode to see ${otherTakes === 1 ? "it" : "them"})`));
        if (takes.length) {
            // display-only rank (the backend's swap_scores.rank_key): following first, then lost cuts; mouth shown,
            // never ranked. Nothing picks a take.
            const ranked = (st.rank || []).map(id => takes.find(t => t.id === id)).filter(Boolean);
            const tbl = el("div", { display: "flex", flexDirection: "column", gap: "2px", border: "1px solid #2b3040", borderRadius: "4px", padding: "3px" });
            for (const t of takes) {
                const tr = row();
                const radio = el("input"); radio.type = "radio"; radio.name = `chosen_${node.id}_${c.id}`; radio.checked = c.chosen === t.id;
                radio.title = "Choose this take: its joins are recomputed on the CPU";
                radio.onpointerdown = (e) => e.stopPropagation();
                radio.onchange = () => op({ op: "choose_take", chunk: c.id, take: t.id });
                const sc = t.scores || {};
                const rank = ranked.indexOf(t) + 1;
                const eff = st.take === t.id;
                const pinName = (p) => p.source ? "source" : p.take;       // a pin on the original has no take
                const lineage = [t.pins?.start ? `← ${pinName(t.pins.start)}` : "", t.pins?.end ? `→ ${pinName(t.pins.end)}` : ""].filter(Boolean).join(" ") || "free";
                const tf = (st.take_flags || {})[t.id] || {};
                const tips = flagTips(sc);
                const dots = el("span", { display: "inline-flex", gap: "2px" });
                for (const k of ["F", "R", "C", "M", ...(tf.scene === "amber" ? ["S"] : [])]) {
                    const fk = { F: "following", R: "replaced", C: "cuts", M: "mouth", S: "scene" }[k];
                    const d = el("span", { color: flagColour(tf[fk]), fontWeight: "bold", fontFamily: "monospace" }, `●${k}`);
                    d.title = tips[k];
                    dots.append(d);
                }
                const info = el("span", { color: eff ? C.text : C.dim, fontFamily: "monospace" },
                    `${t.id.split("-").pop()} · seed ${t.seed} · ${(t.created || "").replace("T", " ").slice(5, 16)} · ${SCORE_TXT(sc)} · #${rank || "–"} · ${lineage}${eff && c.chosen !== t.id ? " · in use (unreviewed)" : ""}`);
                info.title = Object.values(tips).join("\n");
                tr.append(radio, dots, info,
                    button("▶", "Play its review clip (the chunk ± 2 s with both joins as if chosen) in the player", () => playReview(c, t)),
                    button("use prompt + seed", "Load this take's prompt into the chunk and fix the seed to its seed", () => op({ op: "use_take", take: t.id })),
                    button("edit this", "Make this take the picture THIS chunk renders from, so the next change builds on it (no assembly: the job's source gets this chunk's frames swapped in). The chunk's takes, prompt and masks are put aside, 'back to the original' brings them back.", () => chunkSource(c, t)),
                    button("🗑", "Move this take into the job's trash/ (nothing is erased)", () => { if (confirm(`Move ${t.id} to the trash?`)) op({ op: "delete_take", take: t.id }); }));
                (t.flags || []).forEach(f => tr.append(el("span", { color: C.amber }, `⚠ ${f.code}`)));
                tbl.append(tr);
            }
            if (c.chosen) { const tr = row(); tr.append(button("unchoose", "Back to unreviewed: the first take is used until one is chosen", () => op({ op: "choose_take", chunk: c.id, take: null }))); tbl.append(tr); }
            selBox.append(tbl);
        }
        const ov = (P().overrides || {})[c.id];
        if (ov) {
            const r = row();
            r.append(el("span", { color: C.amber }, `renders from take ${ov.take.split("-").pop()} (an edit of it), not the original`),
                button("back to the original", "This chunk renders from the original footage again; its earlier takes, prompt and masks come back, takes made on the edited picture stay listed", () => chunkSource(c, null)));
            selBox.append(r);
        }
        const tr = (P().trash || []).filter(e => e.kind === "take" && e.chunk === c.id);
        if (tr.length) {
            const r = row();
            r.append(el("span", { color: C.dim }, `trash: ${tr.length}`), ...tr.map(e => button(`restore ${e.take.id.split("-").pop()}`, "", () => op({ op: "restore_take", take: e.take.id }))));
            selBox.append(r);
        }
    }
    async function moveCut(c, to) {
        if (to <= 0 || to >= S.N) return;
        await op({ op: "delete_cut", frame: c.frame }, true);
        await op({ op: "add_cut", frame: to, from: c.from, confirmed: c.confirmed !== false });
        S.sel = { kind: "cut", frame: to }; refresh();
    }

    // ------------------------------------------------------------ the next-run bar
    // Mirrors swap_planner.resolve_pins: a start pin from the left neighbour's effective take (or one
    // rendered earlier in the same batch), an end pin from the right neighbour's effective take.
    // The node's own toggle decides (it writes the plan's setting when a render is queued), so the label follows it at once.
    const forwardChainOn = () => { const w = W("forward_chain"); return w ? !!w.value : !!settings().forward_chain; };
    function forecastPins(k, batch, earlier = new Set()) {
        const cs = chunks(), c = cs[k], K = settings().anchors, out = { start: null, end: null };
        if (!c || K <= 0) return out;
        const L = c.left && splitById(c.left);
        const covers = (cid, a, b) => { const st = statusOf(cid); return st.take && st.render && st.render[0] <= a && b <= st.render[1] && ["unreviewed", "chosen"].includes(st.state); };
        if (kept(c)) return out;
        if (k > 0 && L && L.mode === "anchored") {
            const prev = cs[k - 1];
            if (kept(prev)) out.start = `${k + 1}←source`;          // anchored onto the original
            else if (earlier.has(k - 1) || covers(prev.id, c.render[0], c.render[0] + K - 1)) out.start = `${k + 1}←${k}`;
        }
        if (k + 1 < cs.length) {
            const R = cs[k + 1].left && splitById(cs[k + 1].left);
            if (R && R.mode === "anchored" && !c.held && kept(cs[k + 1])) out.end = `${k + 1}→source`;
            else if (R && R.mode === "anchored" && !c.held && !batch.has(k + 1) && !forwardChainOn()
                && (covers(cs[k + 1].id, c.render[1] - K + 1, c.render[1]) || covers(cs[k + 1].id, c.render[1] + 1, c.render[1] + K))) out.end = `${k + 1}→${k + 2}`;
        }
        return out;
    }
    function drawNext() {
        nextBar.innerHTML = "";
        const lead = el("span", { color: "#c084fc", fontWeight: "bold" }, "next run");
        if (!P()) { nextBar.append(lead, el("span", { color: C.dim }, "no job yet")); return; }
        const busy = Object.values(S.queued);
        const q = busy.length ? el("span", { color: C.amber, marginLeft: "auto" }, `${busy.length} queued from this Planner`) : null;
        const cs = chunks();
        const sel = S.sel?.kind === "chunk" ? cs.find(c => c.id === S.sel.id) : null;
        if (sel && kept(sel)) {
            nextBar.append(lead, el("span", { color: C.keptText }, `chunk ${chunkIndex(sel.id) + 1} is kept as the original: nothing to render`));
            if (q) nextBar.append(q);
            return;
        }
        if (sel && (sel.takes || []).length) {
            const k = chunkIndex(sel.id), p = forecastPins(k, new Set([k]));
            const two = p.start && p.end;
            nextBar.append(lead, el("span", {}, `re-roll chunk ${k + 1}, ${two ? `two-sided (${k} | ${k + 2})` : p.start ? `pinned at its start (${k})` : p.end ? `pinned at its end (${k + 2})` : "free"}`),
                el("span", { color: C.dim }, `~${(MIN_PER_209 * sel.length / 209).toFixed(0)} GPU-min, ${sel.length} frames`));
            if (q) nextBar.append(q);
            return;
        }
        const empty = emptyPrompts();
        const todo = cs.map((c, k) => [c, k]).filter(([c]) => needsRender(c));
        if (!todo.length) {
            nextBar.append(lead, el("span", { color: C.dim }, cs.every(c => kept(c) || (c.takes || []).length) ? "every chunk has a take (or is kept as the original): review, re-roll or assemble" : "everything pending is queued"));
            if (q) nextBar.append(q);
            return;
        }
        const batch = new Set(todo.map(([, k]) => k)), earlier = new Set();
        const pins = [];
        for (const [, k] of todo) { const p = forecastPins(k, batch, earlier); if (p.start) pins.push(p.start); if (p.end) pins.push(p.end); earlier.add(k); }
        const frames = todo.reduce((a, [c]) => a + c.length, 0);
        const ks = todo.map(([, k]) => k + 1);
        const range = ks.length > 2 && ks.at(-1) - ks[0] === ks.length - 1 ? `${ks[0]}-${ks.at(-1)}` : ks.join(", ");
        nextBar.append(lead, el("span", {}, `render pending: chunk${ks.length > 1 ? "s" : ""} ${range}`),
            el("span", { color: C.dim }, `~${Math.round(MIN_PER_209 * frames / 209)} GPU-min (${frames} frames)${pins.length ? `, pins ${pins.join(", ")}` : ", no pins"}`));
        if (empty.length) nextBar.append(el("span", { color: C.red }, `blocked: no prompt on ${empty.map(c => chunkIndex(c.id) + 1).join(", ")}`));
        if (q) nextBar.append(q);
    }
    function drawWarnings() {
        warnBox.innerHTML = "";
        const ws = S.view?.warnings || [];
        const extra = chunks().filter(c => ["missing file", "range changed"].includes(statusOf(c.id).state)).length + S.joins.filter(j => j.stale).length;
        const n = ws.length + extra;
        warnBox.append(el("div", { color: n ? C.amber : C.dim, fontWeight: "bold", position: "sticky", top: "0", background: "#17140d", paddingBottom: "1px" },
            n ? `⚠ ${n} warning${n === 1 ? "" : "s"}: none of them blocks a render · click one to jump to it` : "no warnings"));
        for (const w of ws) {
            const r = el("div", { color: "#fcd34d", cursor: "pointer", whiteSpace: "normal", lineHeight: "1.3", padding: "1px 0",
                borderTop: "1px solid #2a2310" }, `⚠ ${w.chunk != null && w.deliver ? `chunk ${w.chunk + 1} (${w.deliver[0]}-${w.deliver[1]}): ` : ""}${w.text}`);
            r.title = "click to select it on the strip";
            r.onclick = () => { if (w.split) S.sel = { kind: "split", id: w.split }; else if (w.chunk != null && chunks()[w.chunk]) S.sel = { kind: "chunk", id: chunks()[w.chunk].id }; seek(w.frame ?? chunks()[w.chunk]?.deliver[0] ?? S.playhead); refresh(); };
            warnBox.append(r);
        }
        for (const c of chunks()) {
            const st = statusOf(c.id);
            if (st.state === "missing file") warnBox.append(el("div", { color: C.red }, `✕ chunk ${label(c.id)}: the effective take's file is gone (${st.take})`));
            if (st.state === "range changed") warnBox.append(el("div", { color: C.amber }, `⚠ chunk ${label(c.id)}: a split moved and no take covers it now: re-render it`));
        }
        for (const j of S.joins) if (j.stale) warnBox.append(el("div", { color: C.amber }, `⚠ split at ${j.frame}: stale join (${j.left_take} | ${j.right_take}): re-roll to fit`));
    }

    // ------------------------------------------------------------ preview: source / quick / full / mask
    // A view is a list of segments over source frames, each played from one file:
    // {a, b, url, t0} where t0 is the file time of frame a's start; url null = nothing to show.
    function segments() {
        if (!P()) return [];
        const N = S.N, fr = S.fr, srcURL = fileURL(S.view?.source_view || P().source.path);   // a browser-playable copy when the codec isn't
        if (S.mode === "source") return [{ a: 0, b: N - 1, url: srcURL, t0: S.base, what: "source" }];
        if (S.mode === "full") {
            const a = P().assembled?.at(-1);
            if (!a) return [{ a: 0, b: N - 1, url: null, what: "no assembly yet: assemble first" }];
            return [{ a: 0, b: N - 1, url: fileURL(joinPath(S.jobDir, a.file)), t0: 0, what: `assembly ${a.file.split("/").pop()}` }];
        }
        const keepView = String(S.mode).startsWith("keep:") ? objectKeeps().find(k => `keep:${k.id}` === S.mode) : null;
        if (String(S.mode).startsWith("keep:") && !keepView) return [{ a: 0, b: N - 1, url: null, what: "that object keep is gone" }];
        if (S.mode === "mask" || S.mode === "mask_bg" || S.mode === "mask_obj" || keepView) {
            const out = [];
            let f = 0;
            const segs = (keepView ? keepMasksOf(keepView) : S.mode === "mask_obj" ? objectMasks() : personMasks()).slice().sort((x, y) => String(y.created).localeCompare(String(x.created)));
            while (f < N) {
                const m = segs.find(s => s.range[0] <= f && f <= s.range[1]);
                if (m) {
                    let b = m.range[1];
                    for (const s of segs) { if (s === m) break; if (s.range[0] > f && s.range[0] <= b) b = s.range[0] - 1; }
                    out.push({ a: f, b, url: m.preview ? fileURL(joinPath(S.jobDir, m.preview)) : null, t0: (f - m.range[0]) / fr, what: `mask ${m.id}` });
                    f = b + 1;
                } else {
                    const nxt = segs.filter(s => s.range[0] > f).map(s => s.range[0]);
                    const b = nxt.length ? Math.min(...nxt) - 1 : N - 1;
                    out.push({ a: f, b, url: null, what: "no mask cached here" });
                    f = b + 1;
                }
            }
            return out;
        }
        // quick: each chunk's effective take (its proxy) over its delivered frames, the source where none
        return chunks().map(c => {
            const st = statusOf(c.id);
            if (st.proxy && ["unreviewed", "chosen"].includes(st.state))
                return { a: c.deliver[0], b: c.deliver[1], url: fileURL(joinPath(S.jobDir, st.proxy)), t0: (c.deliver[0] - st.render[0]) / fr, what: `${st.take}${st.state === "unreviewed" ? " (unreviewed)" : ""}` };
            const ov = (P().overrides || {})[c.id];
            return { a: c.deliver[0], b: c.deliver[1], url: srcURL, t0: S.base + c.deliver[0] / fr, what: ov ? `edit of ${ov.take.split("-").pop()} (no new take yet)` : "source (no take)" };
        });
    }
    const segAt = (f) => segments().find(s => f >= s.a && f <= s.b);
    let switching = false;
    function seek(f, keepPlaying) {
        if (!P()) return;
        S.playhead = Math.max(0, Math.min(Math.round(f), S.N - 1));
        const sg = segAt(S.playhead);
        if (sg && sg.url) {
            cover.style.display = "none";
            const t = sg.t0 + (S.playhead - sg.a + 0.5) / S.fr;
            if (video.dataset.src !== sg.url) {
                switching = true;
                video.dataset.src = sg.url;
                video.src = sg.url;
                video.addEventListener("loadedmetadata", () => { video.currentTime = t; switching = false; if (keepPlaying || S.playing) video.play().catch(() => { }); }, { once: true });
            } else video.currentTime = t;
        } else {
            cover.textContent = sg ? sg.what : "";
            cover.style.display = "flex";
            if (!keepPlaying) video.pause();
        }
        updateOverlay();
        drawCanvas();
    }
    function setMode(m) {
        S.mode = m;
        video.dataset.src = "";
        seek(S.playhead, S.playing);
        refresh();
    }
    function openViewMenu(ev) {
        popup(ev || { clientX: bMode.getBoundingClientRect().left, clientY: bMode.getBoundingClientRect().bottom }, [
            { text: `${S.mode === "source" ? "●" : "○"} source: the original`, run: () => setMode("source") },
            { text: `${S.mode === "quick" ? "●" : "○"} quick: each chunk's effective take, chained (proxies)`, run: () => setMode("quick") },
            { text: `${S.mode === "full" ? "●" : "○"} full: the latest assembly`, run: () => setMode("full") },
            { text: `${S.mode === "mask" ? "●" : "○"} mask: character (the person marked, as a character replace renders it)`, run: () => setMode("mask") },
            { text: `${S.mode === "mask_bg" ? "●" : "○"} mask: background (everything but the person marked, as a background replace renders it)`, run: () => setMode("mask_bg") },
            { text: `${S.mode === "mask_obj" ? "●" : "○"} mask: object (the target marked, as an Object job pastes it)`, run: () => setMode("mask_obj") },
            ...objectKeeps().map(k => ({ text: `${S.mode === `keep:${k.id}` ? "●" : "○"} keep: ${k.target} (its mask, the object inverted)`, run: () => setMode(`keep:${k.id}`) })),
        ]);
    }
    function playReview(c, t) {
        if (!t.review?.file) { toast(`${t.id} has no review clip`, "amber"); return; }
        S.playing = false; bPlay.textContent = "▶";
        S.reviewing = { take: t.id, window: t.review.window };
        video.dataset.src = "";
        video.src = fileURL(joinPath(S.jobDir, t.review.file));
        video.addEventListener("loadedmetadata", () => { video.currentTime = 0; video.play().catch(() => { }); }, { once: true });
        cover.style.display = "none";
        toast(`playing ${t.id}'s review clip: ${t.review.window[0]}-${t.review.window[1]} (press a view button to go back)`, "dim");
        updateOverlay();
    }
    function updateOverlay() {
        zoneSoon();
        if (!P()) { overlay.textContent = ""; return; }
        if (S.reviewing && video.dataset.src === "") {
            const f = S.reviewing.window[0] + Math.floor(video.currentTime * S.fr);
            overlay.textContent = `review ${S.reviewing.take} · frame ${f} · ${video.currentTime.toFixed(2)}s`;
            return;
        }
        const c = chunkAt(S.playhead), sg = segAt(S.playhead);
        overlay.textContent = `${(S.playhead / S.fr).toFixed(2)}s · frame ${S.playhead}${c ? ` · chunk ${label(c.id)}` : ""} · ${S.mode}${sg ? `: ${sg.what}` : ""}`;
    }
    function togglePlay() {
        if (S.reviewing) { S.reviewing = null; video.dataset.src = ""; seek(S.playhead); }
        if (S.playing) { S.playing = false; video.pause(); clearInterval(S.gapTimer); bPlay.textContent = "▶"; return; }
        S.playing = true; bPlay.textContent = "❚❚";
        if (S.playhead >= S.N - 1) seek(0, true);
        const sg = segAt(S.playhead);
        if (sg && !sg.url) { playGap(sg); return; }
        video.play().catch(() => { });
    }
    function playGap(sg) {
        video.pause();
        clearInterval(S.gapTimer);
        const t0 = performance.now(), f0 = S.playhead;
        S.gapTimer = setInterval(() => {
            if (!S.playing) { clearInterval(S.gapTimer); return; }
            const f = f0 + Math.floor((performance.now() - t0) / 1000 * S.fr);
            if (f > sg.b) { clearInterval(S.gapTimer); if (sg.b + 1 >= S.N) { S.playing = false; bPlay.textContent = "▶"; return; } nextSeg(sg.b + 1); return; }
            S.playhead = f; updateOverlay(); drawCanvas();
        }, 1000 / 30);
    }
    function nextSeg(f) {
        const sg = segAt(f);
        if (!sg) { S.playing = false; bPlay.textContent = "▶"; return; }
        if (!sg.url) { S.playhead = f; playGap(sg); return; }
        seek(f, true);
    }
    function tick() {
        if (switching || !S.playing || video.paused || S.reviewing) { if (S.reviewing) updateOverlay(); return; }
        const sg = segAt(S.playhead);
        if (!sg || !sg.url) return;
        const f = sg.a + Math.floor((video.currentTime - sg.t0) * S.fr + 1e-3);
        if (f > sg.b) { if (sg.b + 1 >= S.N) { S.playing = false; video.pause(); bPlay.textContent = "▶"; } else nextSeg(sg.b + 1); }
        else if (f >= sg.a) S.playhead = f;
        updateOverlay();
        drawCanvas();
    }
    function onFrame() { tick(); video.requestVideoFrameCallback(onFrame); }
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onFrame);
    video.addEventListener("timeupdate", tick);
    video.addEventListener("ended", () => {
        if (S.reviewing) return;
        const sg = segAt(S.playhead);
        if (S.playing && sg && sg.b + 1 < S.N) nextSeg(sg.b + 1);
        else { S.playing = false; bPlay.textContent = "▶"; }
    });

    // ------------------------------------------------------------ zoom / fit / ui_state
    function zoomBy(k, anchorX) {
        const ax = anchorX ?? canvas.clientWidth / 2;
        const f = x2f(ax);
        S.fitted = false;
        S.pxPerFrame = Math.min(40, Math.max(0.05, S.pxPerFrame * k));
        S.scroll = Math.max(0, f * S.pxPerFrame - ax);
        drawCanvas(); saveUiSoon();
    }
    function requestFit() { S.needFit = true; if (canvas.clientWidth > 100) fit(); }
    function fit() {
        S.needFit = false; S.fitted = true;
        const w = Math.max(100, canvas.clientWidth - 10);
        if (S.N > 0) S.pxPerFrame = Math.min(40, Math.max(0.05, w / S.N));
        S.scroll = 0;
        drawCanvas(); saveUiSoon();
    }
    let uiTimer = null;
    function saveUiSoon() {
        clearTimeout(uiTimer);
        uiTimer = setTimeout(() => {
            if (!uiW || !P()) return;
            const v = JSON.stringify({ job: job(), zoom: S.fitted ? "fit" : +S.pxPerFrame.toFixed(4), scroll: Math.round(S.scroll),
                sel: S.sel, playhead: S.playhead, view: S.mode, layout: S.layout });
            if (uiW.value !== v) uiW.value = v;
        }, 400);
    }
    function restoreUi() {
        try {
            const u = uiW?.value ? JSON.parse(uiW.value) : null;
            if (!u || u.job !== job()) return;
            if (u.zoom !== "fit" && u.zoom > 0) { S.pxPerFrame = u.zoom; S.scroll = u.scroll || 0; S.fitted = false; S.needFit = false; }
            S.sel = u.sel || null; S.playhead = u.playhead || 0; S.mode = u.view || "quick";
            if (u.layout) { S.layout = Object.assign({ hide: {} }, u.layout); applyLayout(); }
            if (S.sel && !selValid()) S.sel = null;
        } catch { }
    }

    // ------------------------------------------------------------ canvas input
    function local(ev) {
        const r = canvas.getBoundingClientRect();
        return { x: (ev.clientX - r.left) * (canvas.clientWidth / r.width), y: (ev.clientY - r.top) * (CANVAS_H / r.height) };
    }
    function hit(p) {
        if (!P()) return { kind: "empty" };
        if (p.y < RULER_H) {
            for (const c of cuts()) if (Math.abs(p.x - f2x(c.frame)) < 6 * U && p.y > RULER_H - 11 * U) return { kind: "cut", frame: c.frame };
            return { kind: "ruler" };
        }
        if (p.y >= BLOCK_Y && p.y <= BLOCK_Y + BLOCK_H) {
            const tr = trims(), gw = 9 * U;
            for (const [edge, f] of [["start", tr.start], ["end", tr.end]]) {
                const x = f2x(f);
                const gx = edge === "start" ? (f > 0 ? x - gw : x) : (f < S.N ? x : x - gw);
                if (p.x >= gx - 1 && p.x <= gx + gw + 1) return { kind: "trim", edge, frame: f };
            }
        }
        for (const s of splits()) {
            const x = f2x(s.frame);
            if (Math.abs(p.y - PILL_Y) < 8 * U && Math.abs(p.x - x) < 16 * U) return { kind: "pill", id: s.id };
        }
        for (const s of splits()) if (Math.abs(p.x - f2x(s.frame)) < EDGE_PX && p.y > PILL_Y + 6 * U) return { kind: "split", id: s.id };
        const f = Math.floor(x2f(p.x));
        const c = chunkAt(f);
        if (!c || f >= S.N) return { kind: "empty" };
        if (p.y >= PR_Y) {
            const k = shotsOf(c).findIndex(s => f >= s.a && f <= s.b);
            return { kind: "prompt", id: c.id, shot: k };
        }
        if (p.y >= MASK_Y && p.y < MASK_Y + MASK_H) return { kind: "mask", id: c.id };
        return { kind: "chunk", id: c.id };
    }
    canvas.addEventListener("pointermove", (ev) => {
        const p = local(ev);
        if (S.drag) { onDrag(p); return; }
        const h = hit(p);
        S.hover = h;
        canvas.style.cursor = h.kind === "split" || h.kind === "trim" ? "ew-resize" : h.kind === "pill" || h.kind === "cut" ? "pointer" : h.kind === "chunk" || h.kind === "ruler" ? "col-resize" : h.kind === "prompt" ? "text" : "default";
        // hover text: the failed render's error, a split's warnings
        if (h.kind === "chunk" && statusOf(h.id).failed) canvas.title = `failed: ${statusOf(h.id).failed}`;
        else if (h.kind === "pill" || h.kind === "split") canvas.title = warnsFor("split", h.id).map(w => w.text).join("\n");
        else if (h.kind === "trim") canvas.title = `drag to keep the ${h.edge} of the video as the original (never rendered or masked); drag back to the ${h.edge === "start" ? "start" : "end"} to undo`;
        else if (h.kind === "cut") { const c = cuts().find(x => x.frame === h.frame); canvas.title = `cut at ${h.frame}: ${c?.confirmed === false ? "suggested (double-click to confirm)" : "confirmed"} · drag to move`; }
        else canvas.title = "";
        drawCanvas();
    });
    canvas.addEventListener("pointerleave", () => { if (!S.drag) { S.hover = null; drawCanvas(); } });
    canvas.addEventListener("pointerdown", (ev) => {
        ev.stopPropagation(); ev.preventDefault();
        root.focus();
        if (!P()) return;
        const p = local(ev), h = hit(p);
        try { canvas.setPointerCapture(ev.pointerId); } catch { }
        if (S.reviewing) { S.reviewing = null; video.dataset.src = ""; }
        if (h.kind === "ruler") { S.drag = { kind: "scrub" }; seek(x2f(p.x)); return; }
        if (h.kind === "cut") { S.sel = { kind: "cut", frame: h.frame }; S.drag = { kind: "cut", frame: h.frame, to: h.frame, startX: p.x, moved: false }; refresh(); return; }
        if (h.kind === "trim") { S.drag = { kind: "trim", edge: h.edge, from: h.frame, to: h.frame, startX: p.x, moved: false }; return; }
        if (h.kind === "pill") { S.sel = { kind: "split", id: h.id }; refresh(); splitMenu(ev, splitById(h.id)); return; }
        if (h.kind === "split") { const s = splitById(h.id); S.sel = { kind: "split", id: h.id }; S.drag = { kind: "split", id: h.id, to: s.frame, startX: p.x, moved: false }; refresh(); return; }
        if (h.kind === "chunk" || h.kind === "mask") {
            if (!(S.sel?.kind === "chunk" && S.sel.id === h.id)) { if (S.promptDirty) savePrompt(); S.sel = { kind: "chunk", id: h.id }; }
            S.drag = { kind: "scrub" }; seek(x2f(p.x)); refresh(); return;
        }
        if (h.kind === "prompt") {
            if (S.promptDirty) savePrompt();
            S.sel = { kind: "chunk", id: h.id }; S.focusShot = h.shot; seek(x2f(p.x)); refresh(); return;
        }
        if (S.promptDirty) savePrompt();
        S.sel = null; seek(x2f(p.x)); refresh();
    });
    function onDrag(p) {
        const d = S.drag;
        const f = x2f(p.x);
        if (d.kind === "scrub") { seek(f); return; }
        if (d.kind === "split" || d.kind === "cut" || d.kind === "trim") {
            if (!d.moved && Math.abs(p.x - d.startX) < 3) return;
            d.moved = true;
            let to = d.kind === "trim" ? Math.max(0, Math.min(S.N, Math.round(f))) : Math.max(1, Math.min(S.N - 1, Math.round(f)));
            d.snap = false;
            if (d.kind === "split" || (d.kind === "trim" && to > 0 && to < S.N)) {   // snap to a confirmed cut within ~8 px (at least 2 frames)
                const near = nearestConfirmed(to, Math.max(2, Math.round(8 * U / S.pxPerFrame)));
                if (near != null) { to = near; d.snap = true; }
            }
            d.to = to;
            S.playhead = to; updateOverlay();
            drawCanvas();
            if (d.kind === "cut") { const g = canvas.getContext("2d"); g.fillStyle = C.play; g.fillRect(f2x(to) - 1, SRC_Y, 2, BLOCK_H + 17 * U); }
        }
    }
    canvas.addEventListener("pointerup", async (ev) => {
        const d = S.drag;
        S.drag = null;
        try { canvas.releasePointerCapture(ev.pointerId); } catch { }
        if (!d) return;
        if (d.kind === "split" && d.moved) {
            const s = splitById(d.id);
            if (s && d.to !== s.frame) {
                if (splits().some(x => x.id !== s.id && x.frame === d.to)) toast(`there is a split at ${d.to} already`, "amber");
                else await op({ op: "move_split", split: s.id, to: d.to });
            }
            seek(d.to);
        } else if (d.kind === "trim" && d.moved && d.to !== d.from) {
            // one write: the split at the frame, the outer piece kept (or the trim removed at the very end)
            const r = await op({ op: "keep", edge: d.edge, frame: d.to }, true);
            if (r) toast(d.to <= 0 || d.to >= S.N ? `${d.edge} trim removed` : `kept the ${d.edge} as the original: ${d.edge === "start" ? `0-${d.to - 1}` : `${d.to}-${S.N - 1}`}`, "green");
        } else if (d.kind === "cut" && d.moved && d.to !== d.frame) {
            const c = cuts().find(x => x.frame === d.frame);
            if (c) await moveCut(c, d.to);
        }
        refresh();
    });
    canvas.addEventListener("dblclick", (ev) => {
        const h = hit(local(ev));
        if (h.kind === "cut") { const c = cuts().find(x => x.frame === h.frame); if (c) op({ op: "confirm_cut", frame: c.frame, confirmed: c.confirmed === false }); }
        else if (h.kind === "mask") setMode(maskView());
        else if (h.kind === "chunk") { seek(x2f(local(ev).x)); splitAtPlayhead(); }
    });
    canvas.addEventListener("wheel", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const p = local(ev);
        if (p.y < RULER_H || ev.ctrlKey) zoomBy(ev.deltaY < 0 ? 1.2 : 1 / 1.2, p.x);
        else { S.fitted = false; S.scroll = Math.max(0, S.scroll + (ev.deltaX || ev.deltaY)); drawCanvas(); saveUiSoon(); }
    }, { passive: false });
    videoBox.addEventListener("wheel", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        if (S.reviewing) { video.pause(); video.currentTime = Math.max(0, video.currentTime + (ev.deltaY > 0 ? 1 : -1) * (ev.shiftKey ? 10 : 1) / S.fr); updateOverlay(); return; }
        seek(S.playhead + (ev.deltaY > 0 ? 1 : -1) * (ev.shiftKey ? 10 : 1));
    }, { passive: false });

    // keyboard, only while the pointer is over the widget (the Timeline's keys, plus S and C)
    let over = false;
    root.addEventListener("pointerenter", () => over = true);
    root.addEventListener("pointerleave", () => over = false);
    const onKey = (ev) => {
        if (!over || !P() || /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "")) return;
        const k = ev.key;
        const handled = () => { ev.preventDefault(); ev.stopPropagation(); };
        if (k === " ") { handled(); togglePlay(); }
        else if (k === "ArrowLeft") { handled(); seek(S.playhead - (ev.shiftKey ? 10 : 1)); }
        else if (k === "ArrowRight") { handled(); seek(S.playhead + (ev.shiftKey ? 10 : 1)); }
        else if (k === "Home") { handled(); seek(0); }
        else if (k === "End") { handled(); seek(S.N - 1); }
        else if (k === "s" || k === "S") { handled(); splitAtPlayhead(); }
        else if (k === "c" || k === "C") { handled(); toggleCutAtPlayhead(); }
        else if (k === "PageDown" || k === "PageUp") {
            handled();
            const marks = [...new Set([...splits().map(s => s.frame), ...confirmed()])].sort((a, b) => a - b);
            const to = k === "PageDown" ? marks.find(f => f > S.playhead) : marks.filter(f => f < S.playhead).pop();
            if (to != null) seek(to);
        } else if ((k === "Delete" || k === "Backspace") && S.sel) {
            handled();
            if (S.sel.kind === "split") op({ op: "delete_split", split: S.sel.id }).then(() => { S.sel = null; refresh(); });
            else if (S.sel.kind === "cut") op({ op: "delete_cut", frame: S.sel.frame }).then(() => { S.sel = null; refresh(); });
        } else if (k === "Escape") { handled(); S.sel = null; refresh(); }
    };
    window.addEventListener("keydown", onKey, true);

    // drop the source video on the widget
    root.addEventListener("dragover", (ev) => { if (ev.dataTransfer?.types?.includes("Files")) { ev.preventDefault(); ev.stopPropagation(); dropHint.style.display = "flex"; } });
    root.addEventListener("dragleave", () => dropHint.style.display = "none");
    root.addEventListener("drop", async (ev) => {
        dropHint.style.display = "none";
        const f = [...(ev.dataTransfer?.files || [])].find(x => /^video\//.test(x.type) || /\.(mp4|webm|mkv|mov|m4v|avi)$/i.test(x.name));
        if (!f) return;
        ev.preventDefault(); ev.stopPropagation();
        await setSourceFile(f);
    });

    // ------------------------------------------------------------ popups (as timeline.js)
    function popup(ev, items) {
        closePopup();
        const m = el("div", { position: "fixed", left: `${ev.clientX}px`, top: `${ev.clientY}px`, zIndex: 10000,
            background: "#1f2330", border: "1px solid #3b4252", borderRadius: "6px", padding: "4px",
            boxShadow: "0 6px 20px rgba(0,0,0,.5)", fontFamily: "sans-serif", fontSize: "12px", color: C.text,
            maxHeight: "60vh", overflowY: "auto", minWidth: "220px" });
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

    // ------------------------------------------------------------ widgets
    if (jobW) {
        const cb = jobW.callback;
        jobW.callback = function () { const r = cb?.apply(this, arguments); S.restored = false; S.sel = null; reload(); return r; };
    }
    for (const [name, key] of Object.entries(SETTING_OF)) {
        const w = W(name);
        if (!w) continue;
        const cb = w.callback;
        w.callback = function () {
            const r = cb?.apply(this, arguments);
            const s = settings();
            const v = typeof s[key] === "boolean" ? !!w.value : +w.value;
            if (P() && s[key] !== v) op({ op: "settings", settings: { [key]: v } });
            return r;
        };
    }

    new ResizeObserver(() => { if ((S.needFit || S.fitted) && canvas.clientWidth > 100 && P()) fit(); else drawCanvas(); }).observe(canvas);
    const onRemoved = node.onRemoved;
    node.onRemoved = function () { window.removeEventListener("keydown", onKey, true); closePopup(); video.pause(); video.removeAttribute("src"); return onRemoved?.apply(this, arguments); };

    // ------------------------------------------------------------ zone editor
    // An object add that redraws a zone: the zone, the touch circle and the target's anchor drawn over the video on the playhead's frame
    // (the same geometry the render uses: GET /seamstitch/swap/zone_preview), and dragged to set the numbers. Every number in the zone
    // row is in target widths (`fw` px), so a drag is converted back with the same formulas as swap_keep.zone_geometry.
    let zoneGeo = null, zoneDrag = null, zoneNote = "", zoneTimer = 0, zoneSeq = 0, zoneHandles = [];
    function zoneWanted() {
        const o = P()?.object || {};
        return !!P() && jobMode() === "object" && o.action === "add" && (o.fill || "render") === "render" && o.guide === "zone";
    }
    function zoneSoon() {
        clearTimeout(zoneTimer);
        if (!zoneWanted()) { zoneGeo = null; zoneNote = ""; drawZone(); return; }
        zoneTimer = setTimeout(loadZone, 140);
    }
    async function loadZone() {
        const seq = ++zoneSeq;
        try {
            const r = await api.fetchApi(`/seamstitch/swap/zone_preview?job=${encodeURIComponent(job())}&frame=${Math.round(S.playhead)}`);
            const d = await r.json();
            if (seq !== zoneSeq) return;
            if (d.error) { zoneGeo = null; zoneNote = d.error; } else { zoneGeo = d; zoneNote = d.found ? "" : (d.why || "the target is not found on this frame"); }
        } catch (e) { if (seq !== zoneSeq) return; zoneGeo = null; zoneNote = String(e); }
        drawZone();
    }
    const zs = () => Object.assign({}, zoneGeo?.settings || {}, zoneDrag?.over || {});
    function zoneView() {
        const W = videoBox.clientWidth, H = videoBox.clientHeight, vw = zoneGeo?.w || P()?.source?.width || 1280, vh = zoneGeo?.h || P()?.source?.height || 720;
        const k = Math.min(W / vw, H / vh);
        return { W, H, k, x0: (W - vw * k) / 2, y0: (H - vh * k) / 2 };
    }
    function zoneShapeOf(s) {            // swap_keep.zone_geometry
        const fw = zoneGeo.fw, [ax, ay] = zoneGeo.anchor, hgt = s.size * fw, wid = hgt * s.zone_aspect, gr = s.zone_grow || 0;
        return { cx: ax + s.zone_dx * fw, cy: ay - hgt * (0.5 - (s.sink ?? 0.12)) + s.zone_dy * fw, rx: wid / 2 + gr, ry: hgt / 2 + gr, hgt, wid, gr };
    }
    function drawZone() {
        zKeyLbl.textContent = zoneGeo?.found ? (zoneGeo.key != null ? `🔒 locked from frame ${zoneGeo.key} (${zoneGeo.keys.length} lock${zoneGeo.keys.length === 1 ? "" : "s"})`
            : zoneGeo.keys?.length ? `before the first lock (frame ${zoneGeo.keys[0]})` : "no locks: one spot for the whole clip") : "";
        const b = zoneView();
        zoneCv.width = Math.max(1, b.W); zoneCv.height = Math.max(1, b.H);
        const ctx = zoneCv.getContext("2d");
        ctx.clearRect(0, 0, b.W, b.H);
        zoneHandles = [];
        if (!zoneWanted()) return;
        const note = (t) => { ctx.font = "12px sans-serif"; ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(6, b.H - 24, ctx.measureText(t).width + 12, 20); ctx.fillStyle = "#ffd27a"; ctx.fillText(t, 12, b.H - 9); };
        if (!zoneGeo || !zoneGeo.found) { if (zoneNote) note("zone: " + zoneNote); return; }
        const s = zs(), px = (x) => b.x0 + x * b.k, py = (y) => b.y0 + y * b.k, fw = zoneGeo.fw, [ax, ay] = zoneGeo.anchor;
        // the target's anchor (the zone's numbers are measured from here, in target widths)
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5; ctx.beginPath();
        ctx.moveTo(px(ax) - 8, py(ay)); ctx.lineTo(px(ax) + 8, py(ay)); ctx.moveTo(px(ax), py(ay) - 8); ctx.lineTo(px(ax), py(ay) + 8); ctx.stroke();
        ctx.font = "11px sans-serif"; ctx.fillStyle = "#fff"; ctx.fillText("target (1 = its width)", px(ax) + 10, py(ay) - 6);
        ctx.strokeStyle = "rgba(255,255,255,0.45)"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(px(ax - fw / 2), py(ay) + 14); ctx.lineTo(px(ax + fw / 2), py(ay) + 14); ctx.stroke();   // one target-width: the unit of every zone number
        if (s.zone_shape === "mask") { note("zone = the target's own mask (nothing to drag; the zone grow is the only setting)"); return; }
        const z = zoneShapeOf(s);
        ctx.fillStyle = "rgba(0,200,255,0.16)"; ctx.strokeStyle = "#00c8ff"; ctx.lineWidth = 2;
        if (s.zone_shape === "box") { ctx.fillRect(px(z.cx - z.rx), py(z.cy - z.ry), 2 * z.rx * b.k, 2 * z.ry * b.k); ctx.strokeRect(px(z.cx - z.rx), py(z.cy - z.ry), 2 * z.rx * b.k, 2 * z.ry * b.k); }
        else { ctx.beginPath(); ctx.ellipse(px(z.cx), py(z.cy), z.rx * b.k, z.ry * b.k, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
        const hs = [{ k: "move", x: px(z.cx), y: py(z.cy), c: "#00c8ff", r: 6 }, { k: "w", x: px(z.cx + z.rx), y: py(z.cy), c: "#fff", r: 5 }, { k: "h", x: px(z.cx), y: py(z.cy + z.ry), c: "#fff", r: 5 }];
        if ((s.zone_touch_r || 0) > 0) {
            const tx = ax + (s.zone_touch_dx || 0) * fw, ty = ay + (s.zone_touch_dy || 0) * fw, tr = s.zone_touch_r * fw;
            ctx.strokeStyle = "#ffb454"; ctx.fillStyle = "rgba(255,180,84,0.2)"; ctx.beginPath(); ctx.arc(px(tx), py(ty), tr * b.k, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
            hs.push({ k: "touch", x: px(tx), y: py(ty), c: "#ffb454", r: 6 }, { k: "touchr", x: px(tx + tr), y: py(ty), c: "#fff", r: 5 });
        }
        for (const h of hs) { ctx.fillStyle = h.c; ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.fillRect(h.x - h.r, h.y - h.r, 2 * h.r, 2 * h.r); ctx.strokeRect(h.x - h.r, h.y - h.r, 2 * h.r, 2 * h.r); }
        zoneHandles = hs;
        note((zoneDrag ? "" : "zone: drag the cyan square to move it, the white squares to resize it") + ((s.zone_touch_r || 0) > 0 ? " · amber = where he touches her (drag it)" : " · double-click to place where he touches her"));
    }
    const zonePoint = (e) => { const r = videoBox.getBoundingClientRect(); return { x: (e.clientX - r.left) * (videoBox.clientWidth / (r.width || 1)), y: (e.clientY - r.top) * (videoBox.clientHeight / (r.height || 1)) }; };   // the graph canvas may be zoomed: the box is drawn scaled
    const zoneHit = (p) => zoneHandles.find(h => Math.abs(h.x - p.x) <= h.r + 5 && Math.abs(h.y - p.y) <= h.r + 5);
    videoBox.addEventListener("pointerdown", (e) => {
        if (!zoneWanted() || !zoneGeo?.found || e.button !== 0) return;
        const h = zoneHit(zonePoint(e));
        if (!h) return;
        e.preventDefault(); e.stopPropagation();
        try { videoBox.setPointerCapture(e.pointerId); } catch { }
        const s0 = zs();
        zoneDrag = { h: h.k, over: {}, s0, z0: s0.zone_shape === "mask" ? null : zoneShapeOf(s0) };
    }, true);
    videoBox.addEventListener("pointermove", (e) => {
        if (!zoneWanted() || !zoneGeo?.found) { videoBox.style.cursor = ""; return; }
        const p = zonePoint(e);
        if (!zoneDrag) { videoBox.style.cursor = zoneHit(p) ? "grab" : ""; return; }
        const b = zoneView(), sx = (p.x - b.x0) / b.k, sy = (p.y - b.y0) / b.k, fw = zoneGeo.fw, [ax, ay] = zoneGeo.anchor, d = zoneDrag, s0 = d.s0, z0 = d.z0;
        const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v)), o = {}, sink = s0.sink ?? 0.12;
        if (d.h === "move") { o.zone_dx = clamp((sx - ax) / fw, -5, 5); o.zone_dy = clamp((sy - ay + z0.hgt * (0.5 - sink)) / fw, -5, 5); }
        else if (d.h === "w") { const wid = Math.max(0.1 * z0.hgt, 2 * (Math.abs(sx - z0.cx) - z0.gr)); o.zone_aspect = clamp(wid / z0.hgt, 0.1, 10); }
        else if (d.h === "h") {
            const hgt = Math.max(0.1 * fw, 2 * (Math.abs(sy - z0.cy) - z0.gr));
            o.size = clamp(hgt / fw, 0.1, 5); o.zone_aspect = clamp(z0.wid / (o.size * fw), 0.1, 10); o.zone_dy = clamp((z0.cy - ay + o.size * fw * (0.5 - sink)) / fw, -5, 5);   // the centre stays put
        }
        else if (d.h === "touch") { o.zone_touch_dx = clamp((sx - ax) / fw, -5, 5); o.zone_touch_dy = clamp((sy - ay) / fw, -5, 5); }
        else if (d.h === "touchr") { const tx = ax + (s0.zone_touch_dx || 0) * fw, ty = ay + (s0.zone_touch_dy || 0) * fw; o.zone_touch_r = clamp(Math.hypot(sx - tx, sy - ty) / fw, 0.02, 3); }
        d.over = o;
        drawZone();
    });
    async function zoneSave(vals) {       // into the lock that holds on this frame, or the job's own numbers when there is none
        if (zoneGeo) Object.assign(zoneGeo.settings, vals);
        if (zoneGeo?.key != null) return op({ op: "set_object", zone_keys: zKeys().map(k => k.frame === zoneGeo.key ? Object.assign({}, k, vals) : k) }, true);
        return op(Object.assign({ op: "set_object" }, vals), true);
    }
    async function zoneRelease(e) {
        if (!zoneDrag) return;
        const d = zoneDrag; zoneDrag = null;
        try { videoBox.releasePointerCapture(e.pointerId); } catch { }
        const vals = {};
        for (const [k, v] of Object.entries(d.over)) vals[k] = Math.round(v * 1000) / 1000;
        if (Object.keys(vals).length) await zoneSave(vals);
        loadZone();
    }
    videoBox.addEventListener("pointerup", zoneRelease);
    videoBox.addEventListener("pointercancel", zoneRelease);
    videoBox.addEventListener("dblclick", (e) => {                      // place the touch circle where he touches her
        if (!zoneWanted() || !zoneGeo?.found) return;
        const p = zonePoint(e), b = zoneView(), fw = zoneGeo.fw, [ax, ay] = zoneGeo.anchor;
        e.preventDefault(); e.stopPropagation();
        zoneSave({ zone_touch_dx: Math.round((((p.x - b.x0) / b.k - ax) / fw) * 1000) / 1000, zone_touch_dy: Math.round((((p.y - b.y0) / b.k - ay) / fw) * 1000) / 1000,
            zone_touch_r: (zoneGeo.settings?.zone_touch_r || 0) > 0 ? zoneGeo.settings.zone_touch_r : 0.25 }).then(loadZone);
    }, true);
    new ResizeObserver(() => drawZone()).observe(videoBox);

    // ------------------------------------------------------------ public
    node.ssPlanner = {
        job, reload, onQueueEvent, reroll,
        restore() { S.restored = false; S.plan = null; reload(); },
        select(cid) { S.sel = { kind: "chunk", id: cid }; const c = chunks().find(x => x.id === cid); if (c) seek(c.deliver[0]); refresh(); },
        state: S,
        zone: () => ({ wanted: zoneWanted(), geo: zoneGeo, note: zoneNote, handles: zoneHandles, drag: zoneDrag, view: zoneView() }),     // for tests and debugging
    };
    setTimeout(() => reload(), 0);
}
