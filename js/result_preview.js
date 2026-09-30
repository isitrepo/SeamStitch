// SeamStitch Result Preview: saves the spliced video (VHS's encode path), plays it with the
// regenerated span marked, rates its joins, saves frames as PNGs, and hands the result back
// to the Timeline for the next splice. Idea from chanon/comfyui-obvpm-timeline's Result
// Preview (GPL-3.0); written independently for SeamStitch's replace/bridge splice.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const COL = { "seamless": "#34d399", "soft bump": "#fbbf24", "hard cut": "#f87171", "n/a": "#9ca3af" };

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

const fmt = (r) => r == null ? "n/a" : `${r.toFixed(2)}x`;
const viewURL = (v, path) => v
    ? api.apiURL(`/view?filename=${encodeURIComponent(v.filename)}&subfolder=${encodeURIComponent(v.subfolder)}&type=${v.type}&t=${Date.now()}`)
    : api.apiURL(`/seamstitch/loader/view?filename=${encodeURIComponent(path)}`);

app.registerExtension({
    name: "SeamStitch.ResultPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SeamStitchResultPreview") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            build(this);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            repairWidgetValues(this);
            return r;
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (msg) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            const d = msg?.seamstitch_result?.[0];
            if (d && this.ssResult) this.ssResult.show(d);
            return r;
        };
    },
});

function build(node) {
    const root = el("div", { display: "flex", flexDirection: "column", gap: "5px", width: "100%", height: "100%",
        boxSizing: "border-box", fontFamily: "sans-serif", fontSize: "11px", color: "#e5e7eb", overflow: "hidden" });
    const videoBox = el("div", { position: "relative", flex: "1 1 auto", minHeight: "120px", background: "#000", borderRadius: "4px", overflow: "hidden" });
    const video = el("video", { width: "100%", height: "100%", objectFit: "contain", display: "block" });
    video.controls = true;
    video.loop = false;
    const counter = el("div", { position: "absolute", left: "6px", top: "4px", fontWeight: "bold", color: "#e5e7eb",
        textShadow: "0 0 3px #000", pointerEvents: "none" });
    videoBox.append(video, counter);
    const bar = el("canvas", { width: "100%", display: "block", cursor: "col-resize", borderRadius: "3px", flex: "0 0 auto", touchAction: "none" });
    const verdicts = el("div", { display: "flex", gap: "6px", flexWrap: "wrap", alignItems: "center", flexShrink: "0" });
    const actions = el("div", { display: "flex", gap: "5px", flexWrap: "wrap", alignItems: "center", flexShrink: "0" });
    const note = el("div", { color: "#9ca3af", flexShrink: "0", overflowWrap: "anywhere" },
        "Run the graph: the spliced result is saved and shows here with its joins rated.");
    root.append(videoBox, bar, verdicts, actions, note);
    const w = node.addDOMWidget("result_ui", "div", root, { serialize: false });
    w.serialize = false;   // keep the panel out of widgets_values (see repairWidgetValues)
    w.computeSize = (width) => [Math.max(300, (width || node.size[0]) - 20), 380];
    if (node.size[0] < 560) node.size[0] = 560;
    if (node.size[1] < 700) node.size[1] = 700;

    // Same sizing as the Timeline: fill the node's height below its widgets (the player
    // grows with it) and scale text/bar with its width (x1 at 600 px, up to x2.5).
    let U = 1, BAR_H = 22;
    function fitToNode() {
        const u = Math.max(1, Math.min(2.5, node.size[0] / 600));
        if (Math.abs(u - U) > 0.01 || !root.style.fontSize) {
            U = u; BAR_H = Math.round(22 * U);
            root.style.fontSize = `${Math.round(11 * U)}px`;
            bar.style.height = `${BAR_H}px`;
            drawBar();
        }
        if (w.last_y) {
            const h = Math.max(260, node.size[1] - w.last_y - 15);
            if (Math.abs((parseFloat(root.style.height) || 0) - h) > 1) root.style.height = `${h}px`;
        }
    }
    const onDrawFg = node.onDrawForeground;
    node.onDrawForeground = function () { const r = onDrawFg?.apply(this, arguments); fitToNode(); return r; };
    const onResize = node.onResize;
    node.onResize = function () { const r = onResize?.apply(this, arguments); fitToNode(); return r; };

    let D = null, loopSeam = null;
    const curFrame = () => D ? Math.max(0, Math.min(D.frames - 1, Math.floor(video.currentTime * D.frame_rate + 1e-3))) : 0;
    const seekFrame = (f) => { if (!D) return; f = Math.max(0, Math.min(D.frames - 1, Math.round(f))); video.currentTime = (f + 0.5) / D.frame_rate; };

    function drawBar() {
        const W = Math.max(50, bar.clientWidth | 0);
        const k = (window.devicePixelRatio || 1) * Math.max(1, Math.min(4, app.canvas?.ds?.scale || 1));
        const bw = Math.round(W * k), bh = Math.round(BAR_H * k);
        if (bar.width !== bw || bar.height !== bh) { bar.width = bw; bar.height = bh; }
        const g = bar.getContext("2d");
        g.setTransform(k, 0, 0, k, 0, 0);
        g.fillStyle = "#20242c"; g.fillRect(0, 0, W, BAR_H);
        if (!D) return;
        const x = (f) => f / Math.max(1, D.frames) * W;
        const j0 = D.joins[0].frame, j1 = D.joins[1].frame;
        g.fillStyle = "rgba(168,85,247,0.6)"; g.fillRect(x(j0), 3 * U, Math.max(2, x(j1) - x(j0)), BAR_H - 6 * U);
        for (const j of D.joins.slice(0, 2)) { g.fillStyle = COL[j.verdict]; g.fillRect(x(j.frame) - 1.5 * U, 0, 3 * U, BAR_H); }
        const inner = D.joins[2];
        if (inner && inner.ratio != null) { g.fillStyle = COL[inner.verdict]; g.fillRect(x(inner.frame) - U, 6 * U, 2 * U, BAR_H - 12 * U); }
        g.fillStyle = "#f59e0b"; g.fillRect(x(video.currentTime * D.frame_rate) - U, 0, 2 * U, BAR_H);
        counter.textContent = `frame ${curFrame()} / ${D.frames - 1} · ${video.currentTime.toFixed(2)}s`;
    }

    // Drag along the bar to scrub; wheel over the picture steps frames (shift: 10).
    let scrubbing = false;
    const scrubTo = (ev) => { const r = bar.getBoundingClientRect(); seekFrame((ev.clientX - r.left) / r.width * D.frames); };
    bar.addEventListener("pointerdown", (ev) => {
        ev.stopPropagation(); ev.preventDefault();
        if (!D) return;
        scrubbing = true; loopSeam = null; video.pause();
        bar.setPointerCapture(ev.pointerId);
        scrubTo(ev);
    });
    bar.addEventListener("pointermove", (ev) => { if (scrubbing) scrubTo(ev); });
    bar.addEventListener("pointerup", (ev) => { scrubbing = false; try { bar.releasePointerCapture(ev.pointerId); } catch { } });
    videoBox.addEventListener("wheel", (ev) => {
        if (!D) return;
        ev.preventDefault(); ev.stopPropagation();
        video.pause(); loopSeam = null;
        seekFrame(curFrame() + (ev.deltaY > 0 ? 1 : -1) * (ev.shiftKey ? 10 : 1));
    }, { passive: false });
    video.addEventListener("timeupdate", () => {
        drawBar();
        if (loopSeam != null && D && video.currentTime * D.frame_rate > loopSeam + D.frame_rate * 1.0) {
            video.currentTime = Math.max(0, (loopSeam - D.frame_rate) / D.frame_rate);
        }
    });
    video.addEventListener("seeked", drawBar);
    if (video.requestVideoFrameCallback) {
        const onFrame = () => { drawBar(); video.requestVideoFrameCallback(onFrame); };
        video.requestVideoFrameCallback(onFrame);
    }

    function seamLoop(f) {
        loopSeam = f;
        video.currentTime = Math.max(0, (f - D.frame_rate) / D.frame_rate);
        video.play().catch(() => { });
    }

    async function saveFrame() {
        if (!D) return;
        const f = curFrame();
        video.pause();
        note.textContent = `saving frame ${f}…`;
        try {
            const r = await api.fetchApi("/seamstitch/result/grab", { method: "POST",
                body: JSON.stringify({ path: D.path, frame: f, frame_rate: D.frame_rate }) });
            const j = await r.json();
            if (!r.ok) throw new Error(j.error);
            note.textContent = `saved frame ${f} as ${j.path.split(/[\\/]/).pop()} (output/seamstitch_frames)`;
            note.title = j.path;
        } catch (e) { note.textContent = `save frame failed: ${e.message || e}`; }
    }

    function show(d) {
        D = d;
        loopSeam = null;
        video.src = viewURL(d.view, d.play_path || d.path);
        video.addEventListener("loadedmetadata", () => { seekFrame(Math.max(0, d.joins[0].frame - d.frame_rate)); drawBar(); }, { once: true });

        verdicts.innerHTML = "";
        const chip = (label, v, r, title) => {
            const c = el("span", { border: `1px solid ${COL[v]}`, color: COL[v], borderRadius: "10px", padding: "0.1em 0.7em" }, `${label}: ${fmt(r)} ${v}`);
            c.title = title || "";
            return c;
        };
        verdicts.append(
            chip("before", d.before.verdict, d.before.ratio, `The worst join inside the replaced range on the original, at frame ${d.before.frame}`),
            el("span", { color: "#9ca3af" }, "→"),
            ...d.joins.map(j => chip(j.name, j.verdict, j.ratio, `Picture change into frame ${j.frame} / the typical change around it. ~1 moves like the footage; a hard cut reads many times higher.`)));

        actions.innerHTML = "";
        actions.append(
            button("▶ seam 1", "Loop a second either side of the join into the new frames", () => seamLoop(d.joins[0].frame)),
            button("▶ seam 2", "Loop a second either side of the join back to the footage", () => seamLoop(d.joins[1].frame)),
            button("▶ worst inside", "Loop around the biggest picture change inside the new frames", () => seamLoop(d.joins[2].frame)),
            button("■ stop", "", () => { loopSeam = null; video.pause(); }),
            button("📷 save frame", "Save the frame under the playhead as a PNG (output/seamstitch_frames), colour-converted from the saved video - use it as a first/last-frame reference instead of a screen grab", saveFrame),
            button("use as timeline", "Put this result on the SeamStitch Timeline node as its only clip, so the next splice starts from here", () => useAsTimeline(d.timeline_path || d.path)));
        const delta = d.inserted - d.removed;
        const where = d.saved ? `saved: ${d.path.split(/[\\/]/).pop()}` : d.path.split(/[\\/]/).pop();
        note.textContent = `${d.frames} frames at ${d.frame_rate} fps · ${d.removed} replaced by ${d.inserted} (${delta >= 0 ? "+" : ""}${delta} = ${(delta / d.frame_rate).toFixed(2)}s) · ${where}${d.play_path && d.play_path !== d.path ? " (playing a proxy)" : ""}`;
        note.title = d.path;
        drawBar();
    }

    function useAsTimeline(path) {
        const tls = app.graph._nodes.filter(n => n.type === "SeamStitchTimeline" && n.ssTimeline);
        if (!tls.length) { note.textContent = "no SeamStitch Timeline node in this graph"; return; }
        for (const n of tls) n.ssTimeline.useResult(path);
        note.textContent = `now on the timeline: ${path.split(/[\\/]/).pop()}`;
    }

    new ResizeObserver(drawBar).observe(bar);
    node.ssResult = { show };
}
