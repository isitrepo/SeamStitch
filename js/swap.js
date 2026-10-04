// SeamStitch Swap: the Take node's review panel and the Assemble node's player (the Planner's strip
// is js/swap_planner.js). Both reuse Result Preview's player and bar (js/result_preview.js): a
// video, a seconds ruler over a track with the joins marked, drag to scrub, wheel to step frames.
//  - Swap Take: the take's review clip (the chunk ± 2 s with both joins as if this take were
//    chosen), chips for following, cuts, mouth (grey) and the two joins, and "choose this take",
//    "keep current", "re-roll (new seed)", "show in Planner". Nothing here picks a take.
//  - Swap Assemble: the assembly with every join marked and a chip per join.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const COL = { green: "#34d399", amber: "#fbbf24", red: "#f87171", grey: "#9ca3af", none: "#6b7280" };

function el(tag, style, text) {
    const e = document.createElement(tag);
    if (style) Object.assign(e.style, style);
    if (text != null) e.textContent = text;
    return e;
}
function button(label, title, onclick) {
    const b = el("button", { background: "#2a2f3a", color: "#e5e7eb", border: "1px solid #3b4252", borderRadius: "4px",
        padding: "0.2em 0.7em", fontSize: "1em", cursor: "pointer", lineHeight: "1.5em" }, label);
    b.title = title || "";
    b.onclick = (e) => { e.stopPropagation(); onclick(); };
    b.onpointerdown = (e) => e.stopPropagation();
    return b;
}
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
const viewURL = (v, path) => v
    ? api.apiURL(`/view?filename=${encodeURIComponent(v.filename)}&subfolder=${encodeURIComponent(v.subfolder)}&type=${v.type}&t=${Date.now()}`)
    : api.apiURL(`/seamstitch/loader/view?filename=${encodeURIComponent(path)}`);

// The split pill's colour: the backend's verdict (swap_scores.join_flags, fitted in B3); the rule below is only
// the fallback for a take saved before B3, with the same thresholds.
function joinVerdict(j) {
    if (!j || j.type === "straight" || j.type === "pending") return null;
    if (j.verdict !== undefined) return j.verdict;
    const rank = { green: 0, amber: 1, red: 2 };
    let worst = j.type === "stale" ? "amber" : null;
    const up = (c) => { if (c && (worst == null || rank[c] > rank[worst])) worst = c; };
    const v = j.frame_luma?.at_splice;                 // the character's jump is shown, not judged (B3)
    if (v != null) up(Math.abs(v) <= 1 ? "green" : Math.abs(v) <= 2 ? "amber" : "red");
    if (j.join_ratio != null) up(j.join_ratio < 3 ? "green" : j.join_ratio < 5 ? "amber" : "red");
    return worst;
}
const followCol = (v) => v == null ? "none" : v >= 0.60 ? "green" : v >= 0.5 ? "amber" : "red";
const flagCol = (f) => f || "none";

// ---------------------------------------------------------------- the shared player and bar
function player(node, minW, minH) {
    const root = el("div", { display: "flex", flexDirection: "column", gap: "5px", width: "100%", height: "100%",
        boxSizing: "border-box", fontFamily: "sans-serif", fontSize: "11px", color: "#e5e7eb", overflow: "hidden" });
    const videoBox = el("div", { position: "relative", flex: "1 1 auto", minHeight: "120px", background: "#000", borderRadius: "4px", overflow: "hidden" });
    const video = el("video", { width: "100%", height: "100%", objectFit: "contain", display: "block" });
    video.controls = true;
    const counter = el("div", { position: "absolute", left: "6px", top: "4px", fontWeight: "bold", color: "#e5e7eb",
        textShadow: "0 0 3px #000", pointerEvents: "none" });
    videoBox.append(video, counter);
    const bar = el("canvas", { width: "100%", display: "block", cursor: "col-resize", borderRadius: "3px", flex: "0 0 auto", touchAction: "none" });
    const chips = el("div", { display: "flex", gap: "6px", flexWrap: "wrap", alignItems: "center", flexShrink: "0" });
    const actions = el("div", { display: "flex", gap: "5px", flexWrap: "wrap", alignItems: "center", flexShrink: "0" });
    const note = el("div", { color: "#9ca3af", flexShrink: "0", overflowWrap: "anywhere", whiteSpace: "pre-wrap" });
    root.append(videoBox, bar, chips, actions, note);
    const w = node.addDOMWidget("swap_panel", "div", root, { serialize: false });
    w.serialize = false;     // keep the panel out of widgets_values
    w.computeSize = (width) => [Math.max(300, (width || node.size[0]) - 20), 380];
    if (node.size[0] < minW) node.size[0] = minW;
    if (node.size[1] < minH) node.size[1] = minH;

    let U = 1, BAR_H = 36, RULER_H = 14;
    function fitToNode() {
        const u = Math.max(1, Math.min(2.5, node.size[0] / 600));
        if (Math.abs(u - U) > 0.01 || !root.style.fontSize) {
            U = u; RULER_H = Math.round(14 * U); BAR_H = RULER_H + Math.round(22 * U);
            root.style.fontSize = `${Math.round(11 * U)}px`;
            bar.style.height = `${BAR_H}px`;
            draw();
        }
        if (w.last_y) {
            const h = Math.max(260, node.size[1] - w.last_y - 15);
            if (Math.abs((parseFloat(root.style.height) || 0) - h) > 1) root.style.height = `${h}px`;
        }
    }
    const onDrawFg = node.onDrawForeground;
    node.onDrawForeground = function () { const r = onDrawFg?.apply(this, arguments); fitToNode(); return r; };

    // D: {frames, fps, first (source frame of the clip's frame 0), span: [a, b] | null, marks: [{frame, colour, label}]}
    let D = null, loopAt = null;
    const curFrame = () => D ? Math.max(0, Math.min(D.frames - 1, Math.floor(video.currentTime * D.fps + 1e-3))) : 0;
    const seekFrame = (f) => { if (!D) return; f = Math.max(0, Math.min(D.frames - 1, Math.round(f))); video.currentTime = (f + 0.5) / D.fps; };
    function draw() {
        const Wd = Math.max(50, bar.clientWidth | 0);
        const k = (window.devicePixelRatio || 1) * Math.max(1, Math.min(4, app.canvas?.ds?.scale || 1));
        const bw = Math.round(Wd * k), bh = Math.round(BAR_H * k);
        if (bar.width !== bw || bar.height !== bh) { bar.width = bw; bar.height = bh; }
        const g = bar.getContext("2d");
        g.setTransform(k, 0, 0, k, 0, 0);
        g.fillStyle = "#20242c"; g.fillRect(0, 0, Wd, BAR_H);
        g.fillStyle = "#171a20"; g.fillRect(0, 0, Wd, RULER_H);
        if (!D) return;
        const x = (f) => f / Math.max(1, D.frames) * Wd;
        const secPx = D.fps * Wd / Math.max(1, D.frames);
        const step = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120].find(t => t * secPx >= 50 * U) || 300;
        g.strokeStyle = "#6b7280"; g.fillStyle = "#9ca3af"; g.font = `${Math.round(10 * U)}px sans-serif`; g.textBaseline = "top"; g.lineWidth = 1;
        for (let t = 0; t * D.fps <= D.frames; t += step) {
            const tx = Math.round(x(t * D.fps)) + 0.5;
            g.beginPath(); g.moveTo(tx, RULER_H - 5 * U); g.lineTo(tx, RULER_H); g.stroke();
            g.fillText(step < 1 ? `${t.toFixed(2)}s` : `${t}s`, tx + 2, 1.5 * U);
        }
        const T0 = RULER_H, TH = BAR_H - RULER_H;
        if (D.span) { g.fillStyle = "rgba(59,98,168,0.7)"; g.fillRect(x(D.span[0] - D.first), T0 + 3 * U, Math.max(2, x(D.span[1] + 1 - D.first) - x(D.span[0] - D.first)), TH - 6 * U); }
        for (const m of D.marks || []) {
            g.fillStyle = COL[m.colour] || COL.none; g.fillRect(x(m.frame - D.first) - 1.5 * U, T0, 3 * U, TH);
            if (m.label) { g.fillStyle = "#e5e7eb"; g.font = `bold ${Math.round(9 * U)}px sans-serif`; g.fillText(m.label, x(m.frame - D.first) + 3 * U, T0 + 2 * U); }
        }
        const px = x(video.currentTime * D.fps);
        g.fillStyle = "#f59e0b"; g.fillRect(px - U, 0, 2 * U, BAR_H);
        g.beginPath(); g.moveTo(px - 5 * U, 0); g.lineTo(px + 5 * U, 0); g.lineTo(px, 6 * U); g.fill();
        counter.textContent = `frame ${D.first + curFrame()} · ${video.currentTime.toFixed(2)}s`;
    }
    let scrubbing = false;
    const scrubTo = (ev) => { const r = bar.getBoundingClientRect(); seekFrame((ev.clientX - r.left) / r.width * D.frames); };
    bar.addEventListener("pointerdown", (ev) => { ev.stopPropagation(); ev.preventDefault(); if (!D) return; scrubbing = true; loopAt = null; video.pause(); bar.setPointerCapture(ev.pointerId); scrubTo(ev); });
    bar.addEventListener("pointermove", (ev) => { if (scrubbing) scrubTo(ev); });
    bar.addEventListener("pointerup", (ev) => { scrubbing = false; try { bar.releasePointerCapture(ev.pointerId); } catch { } });
    videoBox.addEventListener("wheel", (ev) => {
        if (!D) return;
        ev.preventDefault(); ev.stopPropagation();
        video.pause(); loopAt = null;
        seekFrame(curFrame() + (ev.deltaY > 0 ? 1 : -1) * (ev.shiftKey ? 10 : 1));
    }, { passive: false });
    video.addEventListener("timeupdate", () => {
        draw();
        if (loopAt != null && D && video.currentTime * D.fps > loopAt + D.fps) video.currentTime = Math.max(0, (loopAt - D.fps) / D.fps);
    });
    video.addEventListener("seeked", draw);
    if (video.requestVideoFrameCallback) { const onF = () => { draw(); video.requestVideoFrameCallback(onF); }; video.requestVideoFrameCallback(onF); }
    new ResizeObserver(draw).observe(bar);
    return {
        root, chips, actions, note, video,
        load(src, d, startFrame) {
            D = d; loopAt = null;
            video.src = src;
            video.addEventListener("loadedmetadata", () => { seekFrame(startFrame ?? 0); draw(); }, { once: true });
            draw();
        },
        loop(frame) { if (!D) return; loopAt = frame - D.first; video.currentTime = Math.max(0, (loopAt - D.fps) / D.fps); video.play().catch(() => { }); },
        stop() { loopAt = null; video.pause(); },
    };
}

function chip(text, colour, title) {
    const c = el("span", { border: `1px solid ${COL[colour] || COL.none}`, color: COL[colour] || COL.none, borderRadius: "10px", padding: "0.1em 0.7em" }, text);
    c.title = title || "";
    return c;
}
function joinChip(j, name) {
    const v = joinVerdict(j);
    const fl = j.frame_luma?.at_splice, cl = j.char_luma?.at_splice;
    const t = j.type === "pending" ? `${name}: pending (no take next door)` : j.type === "straight" ? `${name}: straight cut` :
        `${name}: ${j.type}${j.override ? ` (${j.override})` : ""} · ${fl ?? "?"} / ${cl ?? "?"}${j.join_ratio != null ? ` · ${j.join_ratio}x` : ""}`;
    const f = j.flags || {};
    return chip(t, v || (j.type === "straight" ? "grey" : "none"), `split ${j.split} at ${j.frame}, splice ${j.splice}, repair ${j.override || j.repair}. ` +
        (f.verdict ? `colour ${f.colour ?? "–"}, motion ${f.motion ?? "–"}, following ${f.following ?? "–"}, lineage ${f.lineage ?? "–"}: the worst decides. ` : "") +
        "Colour = the whole-frame luma jump at the splice (<= 1.0 / 2.0; the character's is shown, not judged: two SAM3 tracks disagree); " +
        "motion = Result Preview's join ratio (< 3.0 / 5.0); following = pose IoU mean +-25 frames; " +
        "lineage = linked or stale (design §4.5).");
}
const planners = (job) => (app.graph?._nodes || []).filter(n => n.type === "SeamStitchSwapPlanner" && n.ssPlanner && n.ssPlanner.job() === job);

// ---------------------------------------------------------------- Take
function setupTake(node) {
    const P = player(node, 560, 640);
    P.note.textContent = "Run a render: the take's review clip (the chunk ± 2 s, both joins as if chosen) plays here.";
    let D = null;
    async function op(body, done) {
        const r = await api.fetchApi("/seamstitch/swap/op", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(Object.assign({ job: D.job }, body)) });
        const j = await r.json();
        P.note.textContent = j.error ? `error: ${j.error}` : done(j);
        planners(D.job).forEach(n => n.ssPlanner.reload());
    }
    node.ssTake = {
        show(d) {
            D = d;
            const win = d.window || [0, 0];
            const marks = (d.joins || []).filter(j => j.splice >= win[0] && j.splice <= win[1])
                .map(j => ({ frame: j.splice, colour: joinVerdict(j) || (j.type === "straight" ? "grey" : "none"), label: { forward: "F", entry: "E", exit: "X", stale: "stale", straight: "✂" }[j.type] || "" }));
            const v = d.review || d.proxy;
            if (v) P.load(viewURL(v), { frames: win[1] - win[0] + 1, fps: d.fps, first: win[0], span: d.deliver, marks }, Math.max(0, (d.deliver?.[0] ?? win[0]) - win[0] - d.fps));
            P.chips.innerHTML = "";
            const sc = d.scores || {};
            const q = d.quality || {};
            const cuts = sc.cuts ? Object.entries(sc.cuts) : [];
            const mi = sc.mouth_info || {};
            P.chips.append(
                chip(`following ${sc.pose_iou ?? "n/a"}${sc.pose_iou_p10 != null ? ` (p10 ${sc.pose_iou_p10})` : ""}`, q.following ? flagCol(q.following) : followCol(sc.pose_iou),
                    "Pose IoU, source vs output person masks (r10): mean >= 0.60 and p10 >= 0.45 green; < 0.50 / 0.30 red."),
                ...(sc.replaced ? [chip(`replaced ${sc.replaced.person_diff}`, flagCol(q.replaced),
                    "Colour difference, source vs output, inside the source person: >= 60 green; 30-60 amber (partly replaced, e.g. only the head); < 30 red (not replaced: the source came back). Following measures the outline, not who is inside it.")] : []),
                chip(`cuts ${cuts.length ? cuts.map(([f, s]) => `${f} ${s}`).join(", ") : "none inside"}`, flagCol(q.cuts),
                    "Per confirmed cut inside the chunk (scorer.cut_stats): copied >= 3.0 with spread <= 2, lost < 2.0, else unsure."),
                chip(`mouth ${sc.mouth ?? "n/a"}${mi.face != null ? ` · face ${Math.round(100 * mi.face)}%` : ""}`, flagCol(q.mouth) === "none" ? "grey" : flagCol(q.mouth),
                    `Mouth sync at the best lag within +-2 (lag ${mi.lag ?? "?"}), over the frames where both faces are found. Information only, never a gate.${mi.why ? " " + mi.why : ""}`),
                ...(sc.scene ? [chip(`scene ${sc.scene.bg_psnr} dB`, flagCol(q.scene), "Background PSNR outside the person: amber under 15 dB (the room was rewritten). An alarm, never ranked.")] : []),
                ...(d.joins || []).map((j) => joinChip(j, j.right_take === d.take ? "join in" : j.left_take === d.take ? "join out" : `${j.split} @${j.frame}`)));
            P.actions.innerHTML = "";
            P.actions.append(
                ...(d.joins || []).filter(j => j.type !== "pending").map(j => button(`▶ ${j.frame}`, "Loop a second either side of this join", () => P.loop(j.splice))),
                button("■", "", () => P.stop()),
                button("choose this take", "Make this take the chunk's chosen take (its joins are recomputed on the CPU)", () => op({ op: "choose_take", chunk: D.chunk, take: D.take }, (j) => `${D.take} chosen (plan rev ${j.rev})`)),
                button("keep current", "Leave the chunk's choice as it is", () => { P.note.textContent = `kept the current choice for ${D.chunk}`; }),
                button("re-roll (new seed)", "Queue another render of this chunk from its Planner, with a new seed", () => {
                    const ps = planners(D.job);
                    if (!ps.length) { P.note.textContent = `no Swap Planner on job ${D.job} in this graph`; return; }
                    ps[0].ssPlanner.reroll(D.chunk, 1, true);
                }),
                button("show in Planner", "Select this chunk on the Planner and move the canvas to it", () => {
                    const ps = planners(D.job);
                    if (!ps.length) { P.note.textContent = `no Swap Planner on job ${D.job} in this graph`; return; }
                    ps[0].ssPlanner.select(D.chunk);
                    try { app.canvas.centerOnNode(ps[0]); } catch { }
                }));
            P.note.textContent = (d.text || d.take) + (d.flags?.length ? "\n" + d.flags.map(f => "! " + f.text).join("\n") : "");
        },
    };
}

// ---------------------------------------------------------------- Assemble
function setupAssemble(node) {
    const P = player(node, 560, 660);
    P.note.textContent = "Run an assemble: the whole video plays here with every join marked.";
    node.ssAssemble = {
        show(d) {
            const marks = (d.joins || []).map(j => ({ frame: j.splice, colour: joinVerdict(j) || (j.type === "straight" ? "grey" : "none"),
                label: { forward: "F", entry: "E", exit: "X", stale: "stale", straight: "✂", pending: "?" }[j.type] || "" }));
            P.load(viewURL(d.view, d.file), { frames: d.frames, fps: d.fps, first: 0, span: null, marks }, 0);
            P.chips.innerHTML = "";
            P.chips.append(...(d.joins || []).map(j => joinChip(j, `${j.split} @${j.splice}`)));
            P.actions.innerHTML = "";
            P.actions.append(...(d.joins || []).map(j => button(`▶ ${j.splice}`, "Loop a second either side of this join", () => P.loop(j.splice))), button("■", "", () => P.stop()));
            P.note.textContent = `${d.frames} frames at ${d.fps} fps · ${String(d.file).split(/[\\/]/).pop()}` + (d.flags?.length ? "\n" + d.flags.map(f => "! " + f.text).join("\n") : "");
        },
    };
}

app.registerExtension({
    name: "SeamStitch.Swap",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const hook = (setup, key, show) => {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () { const r = onNodeCreated?.apply(this, arguments); setup(this); return r; };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () { const r = onConfigure?.apply(this, arguments); repairWidgetValues(this); return r; };
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (msg) {
                const r = onExecuted?.apply(this, arguments);
                const d = msg?.[key]?.[0];
                if (d) show(this, d);
                return r;
            };
        };
        if (nodeData.name === "SeamStitchSwapTake") hook(setupTake, "seamstitch_swap_take", (n, d) => n.ssTake?.show(d));
        else if (nodeData.name === "SeamStitchSwapAssemble") hook(setupAssemble, "seamstitch_swap_assemble", (n, d) => n.ssAssemble?.show(d));
    },
});
