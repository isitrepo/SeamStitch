// SeamStitch Frame Grab: a player with a timeline, to pick the exact first and last frame to take
// out of a video. Drag the green (first) and red (last) markers, or scrub and press "first = here" /
// "last = here"; the node's first_frame / last_frame numbers follow. Frame numbers count like the
// Swap Planner and the Timeline (the file's own frame rate, rounded). Same player maths as the
// Result Preview: frame f is shown at (f + 0.5) / frame_rate.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const viewURL = (path) => api.apiURL(`/seamstitch/loader/view?filename=${encodeURIComponent(path)}`);

function el(tag, style, text) {
    const e = document.createElement(tag);
    if (style) Object.assign(e.style, style);
    if (text != null) e.textContent = text;
    return e;
}
function button(label, title, onclick, colour) {
    const b = el("button", { background: colour || "#2a2f3a", color: "#e5e7eb", border: "1px solid #3b4252", borderRadius: "4px",
        padding: "0.2em 0.7em", fontSize: "1em", cursor: "pointer", lineHeight: "1.5em" }, label);
    b.title = title || "";
    b.onclick = (e) => { e.stopPropagation(); onclick(); };
    b.onpointerdown = (e) => e.stopPropagation();
    return b;
}

app.registerExtension({
    name: "SeamStitch.FrameGrab",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SeamStitchFrameGrab") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            build(this);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (this.ssGrab) setTimeout(() => this.ssGrab.reload(), 0);
            return r;
        };
    },
});

function build(node) {
    const W = (n) => node.widgets.find(w => w.name === n);
    const videoW = W("video"), firstW = W("first_frame"), lastW = W("last_frame"), frW = W("frame_rate");

    const root = el("div", { display: "flex", flexDirection: "column", gap: "5px", width: "100%", height: "100%",
        boxSizing: "border-box", fontFamily: "sans-serif", fontSize: "11px", color: "#e5e7eb", overflow: "hidden", outline: "none" });
    root.tabIndex = 0;
    const videoBox = el("div", { position: "relative", flex: "1 1 auto", minHeight: "120px", background: "#000", borderRadius: "4px", overflow: "hidden" });
    const video = el("video", { width: "100%", height: "100%", objectFit: "contain", display: "block" });
    video.controls = false; video.loop = false; video.muted = true; video.preload = "auto";
    const counter = el("div", { position: "absolute", left: "6px", top: "4px", fontWeight: "bold", color: "#e5e7eb",
        textShadow: "0 0 3px #000", pointerEvents: "none" });
    videoBox.append(video, counter);
    const bar = el("canvas", { width: "100%", display: "block", cursor: "col-resize", borderRadius: "3px", flex: "0 0 auto", touchAction: "none" });
    const steps = el("div", { display: "flex", gap: "5px", flexWrap: "wrap", alignItems: "center", flexShrink: "0" });
    const marks = el("div", { display: "flex", gap: "5px", flexWrap: "wrap", alignItems: "center", flexShrink: "0" });
    const note = el("div", { color: "#9ca3af", flexShrink: "0", overflowWrap: "anywhere" }, "Pick a video.");
    root.append(videoBox, bar, steps, marks, note);
    const w = node.addDOMWidget("framegrab_ui", "div", root, { serialize: false, hideOnZoom: false });
    w.serialize = false;   // keep the panel out of widgets_values
    w.computeSize = (width) => [Math.max(300, (width || node.size[0]) - 20), 380];
    if (node.size[0] < 560) node.size[0] = 560;
    if (node.size[1] < 760) node.size[1] = 760;

    let U = 1, BAR_H = 44, RULER_H = 14;
    function fitToNode() {
        const u = Math.max(1, Math.min(2.5, node.size[0] / 600));
        if (Math.abs(u - U) > 0.01 || !root.style.fontSize) {
            U = u; RULER_H = Math.round(14 * U); BAR_H = RULER_H + Math.round(30 * U);
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

    let D = null;   // {path, frames, frame_rate, width, height}
    const curFrame = () => D ? Math.max(0, Math.min(D.frames - 1, Math.floor(video.currentTime * D.frame_rate + 1e-3))) : 0;
    const seekFrame = (f) => { if (!D) return; f = Math.max(0, Math.min(D.frames - 1, Math.round(f))); video.currentTime = (f + 0.5) / D.frame_rate; };
    const firstN = () => D ? Math.max(0, Math.min(D.frames - 1, firstW.value | 0)) : 0;
    const lastN = () => D ? ((lastW.value | 0) < 0 ? D.frames - 1 : Math.min(D.frames - 1, lastW.value | 0)) : 0;
    function setWidget(wg, v) {
        wg.value = v;
        if (wg.callback) { try { wg.callback(v); } catch { } }
        app.graph.setDirtyCanvas(true, true);
    }
    const setFirst = (f) => { setWidget(firstW, Math.max(0, Math.min(D.frames - 1, Math.round(f)))); drawBar(); };
    const setLast = (f) => { setWidget(lastW, Math.max(0, Math.min(D.frames - 1, Math.round(f)))); drawBar(); };

    function drawBar() {
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
        const secPx = D.frame_rate * Wd / Math.max(1, D.frames);
        const step = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120].find(t => t * secPx >= 50 * U) || 300;
        g.strokeStyle = "#6b7280"; g.fillStyle = "#9ca3af"; g.font = `${Math.round(10 * U)}px sans-serif`;
        g.textBaseline = "top"; g.lineWidth = 1;
        for (let t = 0; t * D.frame_rate <= D.frames; t += step) {
            const tx = Math.round(x(t * D.frame_rate)) + 0.5;
            g.beginPath(); g.moveTo(tx, RULER_H - 5 * U); g.lineTo(tx, RULER_H); g.stroke();
            g.fillText(step < 1 ? `${t.toFixed(2)}s` : `${t}s`, tx + 2, 1.5 * U);
        }
        const a = firstN(), b = lastN();
        // the span between the two markers
        g.fillStyle = "rgba(96,165,250,0.25)";
        g.fillRect(x(Math.min(a, b)), RULER_H, Math.max(2, x(Math.abs(b - a) + 1)), BAR_H - RULER_H);
        const marker = (f, col, label) => {
            const mx = x(f + 0.5);
            g.fillStyle = col; g.fillRect(mx - 1.5 * U, RULER_H, 3 * U, BAR_H - RULER_H);
            g.beginPath(); g.moveTo(mx - 6 * U, BAR_H); g.lineTo(mx + 6 * U, BAR_H); g.lineTo(mx, BAR_H - 9 * U); g.fill();
            g.font = `bold ${Math.round(10 * U)}px sans-serif`; g.textBaseline = "top";
            const tw = g.measureText(label).width;
            g.fillText(label, Math.max(2, Math.min(Wd - tw - 2, mx + 4 * U)), RULER_H + 2 * U);
        };
        marker(a, "#34d399", `first ${a}`);
        marker(b, "#f87171", `last ${b}`);
        // playhead through ruler and track, with the Timeline's little flag
        const px = x(video.currentTime * D.frame_rate);
        g.fillStyle = "#f59e0b"; g.fillRect(px - U, 0, 2 * U, BAR_H);
        g.beginPath(); g.moveTo(px - 5 * U, 0); g.lineTo(px + 5 * U, 0); g.lineTo(px, 6 * U); g.fill();
        counter.textContent = `frame ${curFrame()} / ${D.frames - 1} · ${video.currentTime.toFixed(2)}s · first ${a} · last ${b}`;
    }

    // Drag a marker (grab within ~8 px), or anywhere else on the bar to scrub the playhead.
    let drag = null;   // "first" | "last" | "scrub"
    const frameAt = (ev) => { const r = bar.getBoundingClientRect(); return Math.floor((ev.clientX - r.left) / r.width * D.frames); };
    bar.addEventListener("pointerdown", (ev) => {
        ev.stopPropagation(); ev.preventDefault();
        if (!D) return;
        video.pause();
        const r = bar.getBoundingClientRect(), fpx = r.width / D.frames;
        const mx = (f) => r.left + (f + 0.5) * fpx;
        const dA = Math.abs(ev.clientX - mx(firstN())), dB = Math.abs(ev.clientX - mx(lastN()));
        drag = Math.min(dA, dB) <= 8 ? (dA <= dB ? "first" : "last") : "scrub";
        bar.setPointerCapture(ev.pointerId);
        move(ev);
    });
    function move(ev) {
        const f = Math.max(0, Math.min(D.frames - 1, frameAt(ev)));
        if (drag === "first") setFirst(f);
        else if (drag === "last") setLast(f);
        seekFrame(f);
    }
    bar.addEventListener("pointermove", (ev) => { if (drag && D) move(ev); });
    bar.addEventListener("pointerup", (ev) => { drag = null; try { bar.releasePointerCapture(ev.pointerId); } catch { } });

    const step = (n) => { if (!D) return; video.pause(); seekFrame(curFrame() + n); };
    videoBox.addEventListener("wheel", (ev) => {
        if (!D) return;
        ev.preventDefault(); ev.stopPropagation();
        step((ev.deltaY > 0 ? 1 : -1) * (ev.shiftKey ? 10 : 1));
    }, { passive: false });
    root.addEventListener("keydown", (ev) => {
        if (!D) return;
        if (ev.key === "ArrowLeft") { step(ev.shiftKey ? -10 : -1); ev.preventDefault(); ev.stopPropagation(); }
        else if (ev.key === "ArrowRight") { step(ev.shiftKey ? 10 : 1); ev.preventDefault(); ev.stopPropagation(); }
    });
    videoBox.addEventListener("click", () => { root.focus(); if (video.paused) video.play().catch(() => { }); else video.pause(); });
    video.addEventListener("timeupdate", drawBar);
    video.addEventListener("seeked", drawBar);
    if (video.requestVideoFrameCallback) {
        const onFrame = () => { drawBar(); video.requestVideoFrameCallback(onFrame); };
        video.requestVideoFrameCallback(onFrame);
    }

    steps.append(
        button("−10", "Back 10 frames (shift + left arrow)", () => step(-10)),
        button("◀ −1", "Back one frame (left arrow, or the wheel over the picture)", () => step(-1)),
        button("▶ play / pause", "Play or pause (or click the picture)", () => { if (!D) return; if (video.paused) video.play().catch(() => { }); else video.pause(); }),
        button("+1 ▶", "Forward one frame (right arrow)", () => step(1)),
        button("+10", "Forward 10 frames (shift + right arrow)", () => step(10)));
    marks.append(
        button("first = here", "Take the frame under the playhead as the FIRST frame", () => { if (D) setFirst(curFrame()); }, "#14532d"),
        button("last = here", "Take the frame under the playhead as the LAST frame", () => { if (D) setLast(curFrame()); }, "#7f1d1d"),
        button("last = end of clip", "Set the last frame to the clip's final frame (-1)", () => { if (D) { setWidget(lastW, -1); drawBar(); } }),
        button("go to first", "Show the first frame", () => { if (D) seekFrame(firstN()); }),
        button("go to last", "Show the last frame", () => { if (D) seekFrame(lastN()); }));

    // a typed number (or a workflow load) moves the markers and shows that frame
    for (const [wg, show] of [[firstW, firstN], [lastW, lastN]]) {
        const orig = wg.callback;
        wg.callback = function () { const r = orig?.apply(this, arguments); drawBar(); if (D) seekFrame(show()); return r; };
    }
    for (const wg of [videoW, frW]) {
        if (!wg) continue;
        const orig = wg.callback;
        wg.callback = function () { const r = orig?.apply(this, arguments); reload(); return r; };
    }

    let loadId = 0;
    async function reload() {
        const id = ++loadId;
        if (!videoW || !videoW.value || videoW.value === "none") { D = null; note.textContent = "Pick a video."; drawBar(); return; }
        note.textContent = "reading the clip…";
        try {
            const q = `video=${encodeURIComponent(videoW.value)}&frame_rate=${frW ? (frW.value | 0) : 0}`;
            const r = await api.fetchApi(`/seamstitch/framegrab/info?${q}`);
            const j = await r.json();
            if (id !== loadId) return;
            if (!r.ok) throw new Error(j.error);
            D = j;
            video.src = viewURL(j.path);
            video.addEventListener("loadedmetadata", () => { seekFrame(firstN()); drawBar(); }, { once: true });
            note.textContent = `${j.frames} frames at ${j.frame_rate} fps · ${j.width}x${j.height} · ${videoW.value}. Drag the green / red markers, or scrub and press "first = here" / "last = here". Queue to save both frames. If the "images" input is connected, the numbers index that batch instead.`;
        } catch (e) { D = null; note.textContent = `could not read the clip: ${e.message || e}`; }
        drawBar();
    }

    // "choose video file": copy a video from anywhere on the computer into ComfyUI's input folder (the
    // Loader's upload route, in chunks so big files go through), then pick it. A file that is already
    // there (same name and size) is reused, not copied again.
    if (videoW) {
        const fileInput = document.createElement("input");
        fileInput.type = "file"; fileInput.accept = "video/*"; fileInput.style.display = "none";
        document.body.appendChild(fileInput);
        const btn = node.addWidget("button", "choose video file", null, () => fileInput.click());
        btn.serialize = false; btn.options = Object.assign(btn.options || {}, { serialize: false }); // a button holds no value: keep it out of widgets_values so saved workflows still line up
        const wi = node.widgets.indexOf(btn), vi = node.widgets.indexOf(videoW);
        if (wi > -1 && vi > -1) { node.widgets.splice(wi, 1); node.widgets.splice(vi + 1, 0, btn); }
        const pick = (name) => {
            if (videoW.options && Array.isArray(videoW.options.values) && !videoW.options.values.includes(name)) videoW.options.values.push(name);
            videoW.value = name;
            reload();
        };
        const upload = async (file) => {
            try {
                const safe = file.name.replace(/[^a-zA-Z0-9.\-_]/g, "_");
                try {
                    const c = await api.fetchApi(`/seamstitch/loader/check_file?filename=${encodeURIComponent(safe)}&size=${file.size}`);
                    if (c.status === 200) { const cj = await c.json(); if (cj.exists) { pick(cj.name); return; } }
                } catch (e) { /* no dedupe check: just upload */ }
                const CHUNK = 10 * 1024 * 1024, total = Math.max(1, Math.ceil(file.size / CHUNK));
                const name = Date.now() + "_" + safe;
                for (let i = 0; i < total; i++) {
                    btn.name = `uploading… ${Math.round((i / total) * 100)}%`; node.setDirtyCanvas(true, false);
                    const fd = new FormData();
                    fd.append("file", file.slice(i * CHUNK, (i + 1) * CHUNK));
                    fd.append("filename", name); fd.append("chunk_index", i); fd.append("total_chunks", total);
                    const r = await api.fetchApi("/seamstitch/loader/upload_chunk", { method: "POST", body: fd });
                    if (r.status !== 200) throw new Error("upload failed (" + r.status + ")");
                    if (i === total - 1) pick((await r.json()).name);
                }
            } catch (e) { note.textContent = `could not load the video: ${e.message || e}`; }
            finally { btn.name = "choose video file"; node.setDirtyCanvas(true, false); fileInput.value = ""; }
        };
        fileInput.addEventListener("change", (e) => { if (e.target.files.length) upload(e.target.files[0]); });
    }

    new ResizeObserver(drawBar).observe(bar);
    node.ssGrab = { reload };
    setTimeout(reload, 0);
}
