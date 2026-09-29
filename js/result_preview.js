// SeamStitch Result Preview: the spliced result, its two joins rated, and a way back onto
// the Timeline for the next splice. Idea from chanon/comfyui-obvpm-timeline's Result
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
        padding: "3px 9px", fontSize: "11px", cursor: "pointer" }, label);
    b.title = title || "";
    b.onclick = (e) => { e.stopPropagation(); onclick(); };
    b.onpointerdown = (e) => e.stopPropagation();
    return b;
}
const fmt = (r) => r == null ? "n/a" : `${r.toFixed(2)}x`;

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
        boxSizing: "border-box", fontFamily: "sans-serif", fontSize: "11px", color: "#e5e7eb" });
    const video = el("video", { width: "100%", flex: "1 1 auto", minHeight: "120px", background: "#000", borderRadius: "4px", objectFit: "contain" });
    video.controls = true;
    video.loop = false;
    const bar = el("canvas", { width: "100%", height: "22px", display: "block", cursor: "pointer", borderRadius: "3px" });
    bar.height = 22;
    const verdicts = el("div", { display: "flex", gap: "6px", flexWrap: "wrap", alignItems: "center" });
    const actions = el("div", { display: "flex", gap: "5px", flexWrap: "wrap", alignItems: "center" });
    const note = el("div", { color: "#9ca3af" }, "Run the graph: the spliced result shows here with both joins rated.");
    root.append(video, bar, verdicts, actions, note);
    const w = node.addDOMWidget("result_ui", "div", root, { serialize: false });
    w.computeSize = (width) => [Math.max(300, (width || node.size[0]) - 20), 380];
    if (node.size[0] < 520) node.size[0] = 520;
    if (node.size[1] < 520) node.size[1] = 520;
    // Fill the node's height below its inputs, so resizing the node grows the player.
    const onDrawFg = node.onDrawForeground;
    node.onDrawForeground = function () {
        const r = onDrawFg?.apply(this, arguments);
        if (w.last_y) {
            const h = Math.max(260, node.size[1] - w.last_y - 15);
            if (Math.abs((parseFloat(root.style.height) || 0) - h) > 1) root.style.height = `${h}px`;
        }
        return r;
    };

    let D = null, loopSeam = null;

    function drawBar() {
        const W = Math.max(50, bar.clientWidth | 0);
        if (bar.width !== W) bar.width = W;
        const g = bar.getContext("2d");
        g.fillStyle = "#20242c"; g.fillRect(0, 0, W, 22);
        if (!D) return;
        const x = (f) => f / Math.max(1, D.frames) * W;
        const j0 = D.joins[0].frame, j1 = D.joins[1].frame;
        g.fillStyle = "rgba(168,85,247,0.6)"; g.fillRect(x(j0), 3, Math.max(2, x(j1) - x(j0)), 16);
        for (const j of D.joins.slice(0, 2)) { g.fillStyle = COL[j.verdict]; g.fillRect(x(j.frame) - 1.5, 0, 3, 22); }
        const inner = D.joins[2];
        if (inner && inner.ratio != null) { g.fillStyle = COL[inner.verdict]; g.fillRect(x(inner.frame) - 1, 6, 2, 10); }
        const cur = video.currentTime * D.frame_rate;
        g.fillStyle = "#f59e0b"; g.fillRect(x(cur) - 1, 0, 2, 22);
    }
    bar.addEventListener("pointerdown", (ev) => {
        ev.stopPropagation();
        if (!D) return;
        const r = bar.getBoundingClientRect();
        const f = (ev.clientX - r.left) / r.width * D.frames;
        video.currentTime = (Math.floor(f) + 0.5) / D.frame_rate;
    });
    video.addEventListener("timeupdate", () => {
        drawBar();
        if (loopSeam != null && D && video.currentTime * D.frame_rate > loopSeam + D.frame_rate * 1.0) {
            video.currentTime = Math.max(0, (loopSeam - D.frame_rate) / D.frame_rate);
        }
    });

    function seamLoop(f) {
        loopSeam = f;
        video.currentTime = Math.max(0, (f - D.frame_rate) / D.frame_rate);
        video.play().catch(() => { });
    }

    function show(d) {
        D = d;
        loopSeam = null;
        const src = d.view ? api.apiURL(`/view?filename=${encodeURIComponent(d.view.filename)}&subfolder=${encodeURIComponent(d.view.subfolder)}&type=${d.view.type}&t=${Date.now()}`)
            : api.apiURL(`/seamstitch/loader/view?filename=${encodeURIComponent(d.path)}`);
        video.src = src;
        video.addEventListener("loadedmetadata", () => { video.currentTime = Math.max(0, (d.joins[0].frame - d.frame_rate) / d.frame_rate); drawBar(); }, { once: true });

        verdicts.innerHTML = "";
        const chip = (label, v, r, title) => {
            const c = el("span", { border: `1px solid ${COL[v]}`, color: COL[v], borderRadius: "10px", padding: "1px 8px" }, `${label}: ${fmt(r)} ${v}`);
            c.title = title || "";
            return c;
        };
        verdicts.append(
            chip("before", d.before.verdict, d.before.ratio, `The worst join inside the replaced range on the original, at frame ${d.before.frame}`),
            el("span", { color: "#9ca3af" }, "→"),
            ...d.joins.map(j => chip(j.name, j.verdict, j.ratio, `Picture change into frame ${j.frame} / the median change around it. ~1 moves like the footage; a hard cut reads many times higher.`)));

        actions.innerHTML = "";
        actions.append(
            button("▶ seam 1", "Loop a second either side of the join into the new frames", () => seamLoop(d.joins[0].frame)),
            button("▶ seam 2", "Loop a second either side of the join back to the footage", () => seamLoop(d.joins[1].frame)),
            button("▶ worst inside", "Loop around the biggest picture change inside the new frames", () => seamLoop(d.joins[2].frame)),
            button("■ stop loop", "", () => { loopSeam = null; video.pause(); }),
            button("use as timeline", "Put this result on the SeamStitch Timeline node as its only clip, so the next splice starts from here", () => useAsTimeline(d.view && d.view.type === "output" ? (d.view.subfolder ? `${d.view.subfolder}/${d.view.filename}` : d.view.filename) : d.path)));
        const delta = d.inserted - d.removed;
        note.textContent = `${d.frames} frames at ${d.frame_rate} fps · ${d.removed} replaced by ${d.inserted} (${delta >= 0 ? "+" : ""}${delta} = ${(delta / d.frame_rate).toFixed(2)}s) · ${d.path.split(/[\\/]/).pop()}`;
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
