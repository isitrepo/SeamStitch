// SeamStitch Swap: the minimal panels until the strip UI (B2).
//  - Swap Planner: hides its run / ui_state widgets and draws a read-only summary of the job
//    (chunks, takes, joins, warnings) from GET /seamstitch/swap/plan.
//  - Swap Take: plays the take's review clip (the chunk +- 2 s with both joins as if chosen), with
//    "choose this take" and "keep current". Nothing here picks a take on its own.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HIDDEN = ["run", "ui_state"];

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
const widget = (node, name) => node.widgets?.find((w) => w.name === name);

async function fetchPlan(job) {
    if (!job) return null;
    const r = await api.fetchApi(`/seamstitch/swap/plan?job=${encodeURIComponent(job)}`);
    return r.json();
}

// ---------------------------------------------------------------- Planner
function setupPlanner(node) {
    for (const name of HIDDEN) {
        const w = widget(node, name);
        if (!w) continue;
        w.hidden = true;                       // value still saved (widgets_values keep their positions)
        w.computeSize = () => [0, -4];
    }
    node.ssText = "Swap Planner: set job (and source) to start.";
    node.ssRefresh = async () => {
        const job = widget(node, "job")?.value;
        try {
            const d = await fetchPlan(job);
            node.ssText = d?.error ? `job ${job}: ${d.error}` : (d?.text || node.ssText);
        } catch (e) {
            node.ssText = `job ${job}: ${e}`;
        }
        node.setDirtyCanvas(true, true);
    };
    const jw = widget(node, "job");
    if (jw) {
        const cb = jw.callback;
        jw.callback = function () { const r = cb?.apply(this, arguments); node.ssRefresh(); return r; };
    }
    const onDrawFg = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        const r = onDrawFg?.apply(this, arguments);
        if (this.flags?.collapsed) return r;
        const lines = String(this.ssText || "").split("\n");
        const lh = 13;
        let y = (this.widgets_start_y || 0) + 10;
        for (const w of this.widgets || []) if (!w.hidden && w.last_y != null) y = Math.max(y, w.last_y + 28);
        ctx.save();
        ctx.font = "11px monospace";
        ctx.textBaseline = "top";
        for (const line of lines) {
            ctx.fillStyle = line.includes("⚠") ? "#fbbf24" : line.startsWith(" c") ? "#e5e7eb" : "#9ca3af";
            ctx.fillText(line, 10, y, this.size[0] - 20);
            y += lh;
        }
        ctx.restore();
        const need = y + 10;
        if (this.size[1] < need) this.size[1] = need;
        return r;
    };
    if (node.size[0] < 620) node.size[0] = 620;
}

// ---------------------------------------------------------------- Take
function setupTake(node) {
    const root = el("div", { display: "flex", flexDirection: "column", gap: "5px", width: "100%", height: "100%",
        boxSizing: "border-box", fontFamily: "sans-serif", fontSize: "11px", color: "#e5e7eb", overflow: "hidden" });
    const video = el("video", { width: "100%", flex: "1 1 auto", minHeight: "120px", background: "#000", borderRadius: "4px",
        objectFit: "contain" });
    video.controls = true;
    const info = el("div", { whiteSpace: "pre-wrap", fontFamily: "monospace", color: "#cbd5e1", flexShrink: "0" },
        "Run a render: the take's review clip (the chunk ± 2 s, both joins as if chosen) plays here.");
    const actions = el("div", { display: "flex", gap: "5px", alignItems: "center", flexShrink: "0" });
    const status = el("span", { color: "#9ca3af" });
    let D = null;
    const choose = button("choose this take", "Make this take the chunk's chosen take (its joins are recomputed on the CPU).", async () => {
        if (!D) return;
        const r = await api.fetchApi("/seamstitch/swap/op", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ job: D.job, op: "choose_take", chunk: D.chunk, take: D.take }) });
        const j = await r.json();
        status.textContent = j.error ? `error: ${j.error}` : `${D.take} chosen (plan rev ${j.rev})`;
    });
    const keep = button("keep current", "Leave the chunk's current choice as it is.", () => {
        if (D) status.textContent = `kept the current choice for ${D.chunk}`;
    });
    actions.append(choose, keep, status);
    root.append(video, info, actions);
    const w = node.addDOMWidget("swap_take_ui", "div", root, { serialize: false });
    w.serialize = false;
    w.computeSize = (width) => [Math.max(300, (width || node.size[0]) - 20), 360];
    if (node.size[0] < 520) node.size[0] = 520;
    if (node.size[1] < 560) node.size[1] = 560;
    node.ssTake = {
        show(d) {
            D = d;
            const v = d.review || d.proxy;
            if (v) video.src = api.apiURL(`/view?filename=${encodeURIComponent(v.filename)}&subfolder=${encodeURIComponent(v.subfolder)}&type=${v.type}&t=${Date.now()}`);
            const win = d.window ? ` · review ${d.window[0]}-${d.window[1]}` : "";
            info.textContent = (d.text || `${d.take}`) + win + (d.flags?.length ? "\n" + d.flags.map((f) => "! " + f.text).join("\n") : "");
            status.textContent = "";
        },
    };
}

app.registerExtension({
    name: "SeamStitch.Swap",
    setup() {
        api.addEventListener("seamstitch_swap_plan", (e) => {
            const job = e.detail?.job;
            for (const n of app.graph?._nodes || []) {
                if (n.type === "SeamStitchSwapPlanner" && n.ssRefresh && widget(n, "job")?.value === job) n.ssRefresh();
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SeamStitchSwapPlanner") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                setupPlanner(this);
                return r;
            };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure?.apply(this, arguments);
                setTimeout(() => this.ssRefresh?.(), 0);
                return r;
            };
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (msg) {
                const r = onExecuted?.apply(this, arguments);
                const d = msg?.seamstitch_swap_plan?.[0];
                if (d?.text) { this.ssText = d.text; this.setDirtyCanvas(true, true); }
                return r;
            };
        } else if (nodeData.name === "SeamStitchSwapTake") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                setupTake(this);
                return r;
            };
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (msg) {
                const r = onExecuted?.apply(this, arguments);
                const d = msg?.seamstitch_swap_take?.[0];
                if (d && this.ssTake) this.ssTake.show(d);
                return r;
            };
        }
    },
});
