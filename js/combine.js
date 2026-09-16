import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Wires a "choose file to upload" button to a dropdown widget: chunked
// upload (mirrors Load Video UI's own pattern, namespaced to this node's own
// routes so the two packages don't have to depend on each other), plus a
// preview <video> element the caller owns and passes in - this function
// never creates its own LiteGraph widget for the preview, see below for why.
function attachFilePicker(node, dropdownWidget, label, fileInput, preview) {
    const updatePreview = (filename) => {
        if (!filename || filename === "none") {
            preview.style.display = "none";
            return;
        }
        preview.src = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input`);
        preview.style.display = "block";
    };

    const btnWidget = node.addWidget("button", `choose ${label} file`, null, () => {
        fileInput.click();
    });

    const CHUNK_SIZE = 10 * 1024 * 1024;

    const uploadFile = async (file) => {
        try {
            const safeName = file.name.replace(/[^a-zA-Z0-9.\-_]/g, "_");
            try {
                const checkResp = await api.fetchApi(
                    `/seamstitch/combine/check_file?filename=${encodeURIComponent(safeName)}&size=${file.size}`
                );
                if (checkResp.status === 200) {
                    const result = await checkResp.json();
                    if (result.exists) {
                        setSelected(result.name);
                        return;
                    }
                }
            } catch (e) {
                console.warn("[SeamStitch] check_file failed, uploading anyway", e);
            }

            btnWidget.name = "uploading...";
            node.setDirtyCanvas(true, false);

            if (file.size > CHUNK_SIZE) {
                const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
                const uploadName = Date.now() + "_" + safeName;
                for (let i = 0; i < totalChunks; i++) {
                    btnWidget.name = `uploading... ${Math.round((i / totalChunks) * 100)}%`;
                    node.setDirtyCanvas(true, false);
                    const chunk = file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);
                    const formData = new FormData();
                    formData.append("file", chunk);
                    formData.append("filename", uploadName);
                    formData.append("chunk_index", i);
                    formData.append("total_chunks", totalChunks);
                    const resp = await api.fetchApi("/seamstitch/combine/upload_chunk", {
                        method: "POST",
                        body: formData,
                    });
                    if (resp.status !== 200) throw new Error("Chunk upload failed");
                    if (i === totalChunks - 1) {
                        const data = await resp.json();
                        setSelected(data.name);
                    }
                }
            } else {
                const body = new FormData();
                body.append("image", file);
                const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                if (resp.status !== 200) throw new Error(`Upload failed: ${resp.statusText}`);
                const data = await resp.json();
                setSelected(data.name);
            }
        } catch (error) {
            console.error("[SeamStitch] Upload failed", error);
        } finally {
            btnWidget.name = `choose ${label} file`;
            node.setDirtyCanvas(true, false);
            fileInput.value = "";
        }
    };

    const setSelected = (filename) => {
        if (dropdownWidget.options?.values && !dropdownWidget.options.values.includes(filename)) {
            dropdownWidget.options.values.push(filename);
        }
        dropdownWidget.value = filename;
        updatePreview(filename);
        node.setDirtyCanvas(true, false);
    };

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) uploadFile(e.target.files[0]);
    });

    const originalCallback = dropdownWidget.callback;
    dropdownWidget.callback = function () {
        if (originalCallback) originalCallback.apply(this, arguments);
        updatePreview(this.value);
    };
    updatePreview(dropdownWidget.value);
}

// Both previews live inside ONE LiteGraph DOM widget (a flex column with the
// two <video> elements as children) instead of one widget each. That's not
// cosmetic - it's what makes resizing work at all:
//
// LiteGraph stacks DOM widgets top to bottom using each widget's *declared*
// computeSize height, not its actual rendered CSS height. The first attempt
// here gave each preview its own widget and grew their CSS height directly;
// that desynced widget B's declared top-of-widget position (still based on
// widget A's original, smaller computeSize) from widget A's real, now-taller
// bottom edge, so B's top visibly slid under A's bottom as A grew.
//
// A single widget sidesteps that: there is no second widget position to
// desync from, and ordinary flexbox (flex: 1 1 0 on each video) keeps the
// two children's shared edge in sync for free. computeSize for this widget
// stays a FIXED number forever (never derived from the node's current
// size) - deriving it from node.size[1] was an earlier, separate bug: it
// closes a feedback loop through LiteGraph's own node-minimum-size math
// (bigger preview -> bigger computed minimum -> node auto-grows -> resize
// fires again with the bigger size -> preview grows again -> ...). Instead,
// only the *visual* CSS height is nudged, from onDrawForeground, using
// domWidget.last_y (where LiteGraph actually drew this widget once
// everything above it - the two choose-file buttons - is laid out) exactly
// the way Load Video UI's own single DOM widget already does.
function buildPreviewPanel(node, previewA, previewB) {
    const container = document.createElement("div");
    Object.assign(container.style, {
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        width: "100%",
        boxSizing: "border-box",
    });
    container.appendChild(previewA);
    container.appendChild(previewB);

    const domWidget = node.addDOMWidget("seamstitch_previews", "div", container, { serialize: false });
    domWidget.computeSize = () => [0, 260];

    const origOnDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        if (origOnDrawForeground) origOnDrawForeground.apply(this, arguments);

        if (domWidget.last_y != null) {
            const remaining = this.size[1] - domWidget.last_y - 18;
            const target = Math.max(160, remaining);
            const current = parseFloat(container.style.height);
            if (isNaN(current) || Math.abs(current - target) > 1) {
                container.style.height = `${target}px`;
            }
        }
    };
}

function makePreviewVideo() {
    const preview = document.createElement("video");
    Object.assign(preview.style, {
        flex: "1 1 0",
        minHeight: "80px",
        width: "100%",
        boxSizing: "border-box",
        background: "#000",
        borderRadius: "4px",
        objectFit: "contain",
        display: "none",
    });
    preview.controls = true;
    preview.muted = true;
    return preview;
}

// After a run, auto-select this node's freshly written file in whatever
// Load Video UI node(s) its `images` output feeds (the same link that feeds
// input_video, so no separate video_path wiring is needed) and refresh their
// preview - so "bypass everything downstream, run Combine+Load Video UI,
// scrub, un-bypass, run again" never requires manually hunting through the
// video dropdown in between.
//
// This is real cross-package coupling (reaching into a SeamStitchLoader
// node by class name and calling helpers it exposes on itself) rather than a
// generic mechanism, because the two are explicitly meant to be used
// together for exactly this workflow - it does nothing if no such node is
// downstream. Driven by the "executed" websocket event ComfyUI sends after a
// node runs (the same channel VHS_VideoCombine uses to refresh its own
// preview widget with no button click needed), reading the "ui" data
// SeamStitchCombine.combine() returns alongside its normal outputs.
function wireAutoSelect(api, app) {
    api.addEventListener("executed", (event) => {
        const detail = event.detail;
        const videoPath = detail?.output?.video_path?.[0];
        if (!videoPath) return;

        const sourceNode = app.graph.getNodeById(detail.node);
        if (!sourceNode || sourceNode.type !== "SeamStitchCombine") return;

        const imagesOutput = sourceNode.outputs?.find((o) => o.name === "images");
        const targetIds = new Set();
        for (const linkId of imagesOutput?.links || []) {
            const link = app.graph.links[linkId];
            if (link) targetIds.add(link.target_id);
        }

        for (const targetId of targetIds) {
            const targetNode = app.graph.getNodeById(targetId);
            if (!targetNode || targetNode.type !== "SeamStitchLoader") continue;

            const videoWidget = targetNode.widgets?.find((w) => w.name === "video");
            if (!videoWidget) continue;

            if (videoWidget.options?.values && !videoWidget.options.values.includes(videoPath)) {
                videoWidget.options.values.unshift(videoPath);
            }
            videoWidget.value = videoPath;
            targetNode._should_reset_trim = true;
            if (targetNode.updatePreview) targetNode.updatePreview(videoPath);
            if (targetNode.syncFramesFromTime) targetNode.syncFramesFromTime();
            targetNode.setDirtyCanvas(true, false);
        }
    });
}

app.registerExtension({
    name: "SeamStitch.Combine",
    async setup() {
        wireAutoSelect(api, app);
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SeamStitchCombine") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            const videoAWidget = node.widgets.find((w) => w.name === "video_a");
            const videoBWidget = node.widgets.find((w) => w.name === "video_b");

            const previewA = makePreviewVideo();
            const previewB = makePreviewVideo();
            const fileInputA = document.createElement("input");
            const fileInputB = document.createElement("input");
            for (const fi of [fileInputA, fileInputB]) {
                fi.type = "file";
                fi.accept = "video/*";
                fi.style.display = "none";
                document.body.appendChild(fi);
            }

            if (videoAWidget) attachFilePicker(node, videoAWidget, "A", fileInputA, previewA);
            if (videoBWidget) attachFilePicker(node, videoBWidget, "B", fileInputB, previewB);
            buildPreviewPanel(node, previewA, previewB);

            const originalOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                for (const fi of [fileInputA, fileInputB]) {
                    if (fi.parentNode) fi.parentNode.removeChild(fi);
                }
                if (originalOnRemoved) originalOnRemoved.apply(this, arguments);
            };

            const status = document.createElement("div");
            Object.assign(status.style, {
                whiteSpace: "pre-wrap",
                fontSize: "11px",
                color: "#aaa",
                padding: "4px",
                fontFamily: "monospace",
            });
            status.textContent = "Not loaded yet.";
            const statusWidget = node.addDOMWidget("seamstitch_status", "div", status, { serialize: false });
            statusWidget.computeSize = () => [0, 40];

            // Pulls in both chosen files right now (a header probe, not a full
            // decode) so a bad pick is caught before queuing the whole workflow.
            node.addWidget("button", "Load Video", null, async () => {
                status.textContent = "Loading...";
                node.setDirtyCanvas(true, false);

                const probeOne = async (label, filename) => {
                    if (!filename || filename === "none") return `${label}: no file chosen`;
                    try {
                        const resp = await api.fetchApi(
                            `/seamstitch/combine/probe?filename=${encodeURIComponent(filename)}`
                        );
                        const data = await resp.json();
                        if (!data.ok) return `${label}: ERROR - ${data.error}`;
                        const dur = data.duration ? data.duration.toFixed(2) : "?";
                        return `${label}: ${data.width}x${data.height}, ${dur}s, ` +
                            `${data.frame_count} frames @ ${data.fps.toFixed(2)}fps, ` +
                            `audio: ${data.has_audio ? "yes" : "no"}`;
                    } catch (e) {
                        return `${label}: ERROR - ${e}`;
                    }
                };

                const [lineA, lineB] = await Promise.all([
                    probeOne("A", videoAWidget?.value),
                    probeOne("B", videoBWidget?.value),
                ]);
                status.textContent = `${lineA}\n${lineB}`;
                node.setDirtyCanvas(true, false);
            });

            return r;
        };
    },
});
