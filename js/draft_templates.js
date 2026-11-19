// SeamStitch Swap Draft Prompts: your own templates, from the node.
// A template is a LAYOUT for the draft (any job: character, background or object): the drafter writes the prompt as it always does, and your
// layout puts its six sections where you want them and adds your own words. In the template box, write {summary}, {subject_definitions},
// {retention_analysis}, {detailed_description}, {overall_soundscape}, {non_diegetic_music} (or {prompt} for all of it) where each piece goes.
//   "preview template"  shows the layout with stand-in text, before it is saved
//   "save template"     stores it under template_name, built on the built-in template in template_base, as "my: <name>" in the dropdown
//   "delete template"   removes the "my: ..." template picked in the dropdown
// Picking a saved template loads its text, name and base back into the boxes so it can be edited and saved again. Picking a built-in template makes it
// the base. "template_example" (read-only) always shows an example of what the layout in the box gives for the base picked, and updates by itself.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PREFIX = "my: ";
const AUTO = "(automatic: the job's own template)";

function widget(node, name) { return node.widgets?.find((w) => w.name === name); }

async function call(path, body) {
    const r = await api.fetchApi(path, body ? { method: "POST", body: JSON.stringify(body) } : undefined);
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    return j;
}

function setChoices(node, choices, pick) {
    const w = widget(node, "template");
    if (!w) return;
    w.options.values = choices;
    if (pick && choices.includes(pick)) w.value = pick;
    else if (!choices.includes(w.value)) w.value = choices[0];
    node.setDirtyCanvas?.(true, true);
}

async function loadSaved(node) {
    const w = widget(node, "template");
    if (!w || !(w.value || "").startsWith(PREFIX)) return;
    try {
        const j = await call("/seamstitch/draft_template/list");
        const n = w.value.slice(PREFIX.length), t = j.templates[n];
        if (!t) return;
        const form = widget(node, "template_form"), name = widget(node, "template_name"), base = widget(node, "template_base");
        if (form) form.value = t.text;
        if (name) name.value = n;
        if (base) base.value = t.base || AUTO;
        node.setDirtyCanvas?.(true, true);
        refreshExample(node);
    } catch (e) { console.warn("[SeamStitch] template load:", e); }
}

// the read-only box: the layout in template_form applied to an example prompt of the base picked in template_base
const timers = new WeakMap();
function refreshExample(node) {
    clearTimeout(timers.get(node));
    timers.set(node, setTimeout(async () => {
        const ex = widget(node, "template_example");
        if (!ex) return;
        const base = widget(node, "template_base")?.value;
        let text;
        try {
            const j = await call("/seamstitch/draft_template/preview", { text: widget(node, "template_form")?.value || "", base: base === AUTO ? "" : base });
            text = (j.note ? "(" + j.note.trim() + ")\n\n" : "") + j.text;
        } catch (e) { text = "(can't show an example yet: " + (e.message || e) + ")"; }
        ex.value = text;
        node.setDirtyCanvas?.(true, true);
    }, 250));
}

// a plain overlay with the text in it (alert() cuts long text)
function showText(title, text) {
    const shade = document.createElement("div");
    shade.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10000;display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "background:#1b2230;color:#e5e7eb;border:1px solid #3b4252;border-radius:6px;max-width:900px;width:90vw;max-height:85vh;display:flex;flex-direction:column;padding:12px;font:13px sans-serif;";
    const h = document.createElement("div"); h.textContent = title; h.style.cssText = "font-weight:bold;margin-bottom:8px;";
    const pre = document.createElement("pre"); pre.textContent = text; pre.style.cssText = "white-space:pre-wrap;overflow:auto;margin:0 0 10px;background:#111;padding:8px;border-radius:4px;";
    const b = document.createElement("button"); b.textContent = "close"; b.style.cssText = "align-self:flex-end;padding:4px 14px;";
    b.onclick = () => shade.remove();
    shade.onclick = (e) => { if (e.target === shade) shade.remove(); };
    box.append(h, pre, b); shade.append(box); document.body.append(shade);
}

app.registerExtension({
    name: "SeamStitch.DraftTemplates",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SeamStitchSwapDraft") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;
            const tw = widget(node, "template");
            if (tw) {
                const cb = tw.callback;
                tw.callback = function () {
                    const x = cb?.apply(this, arguments);
                    const v = tw.value || "";
                    if (v.startsWith(PREFIX)) loadSaved(node);
                    else { const bw = widget(node, "template_base"); if (bw && (bw.options?.values || []).includes(v)) bw.value = v; refreshExample(node); }
                    return x;
                };
            }
            for (const n of ["template_base", "template_form"]) {                      // changing either updates the example
                const w = widget(node, n);
                if (!w) continue;
                const cb = w.callback;
                w.callback = function () { const x = cb?.apply(this, arguments); refreshExample(node); return x; };
            }
            // plain names for the boxes, and the example is read-only
            const lab = { template_form: "YOUR TEMPLATE LAYOUT (edit this)", template_name: "name to save it as", template_base: "built on (which built-in template)",
                          template_example: "example result (read-only)", qwen_questions: "Qwen's questions (advanced: not part of templates)" };
            for (const [k, v] of Object.entries(lab)) { const w = widget(node, k); if (w) w.label = v; }
            const exw = widget(node, "template_example");
            if (exw) { setTimeout(() => { if (exw.inputEl) exw.inputEl.readOnly = true; }, 0); }
            const button = (label, fn) => {
                const b = node.addWidget("button", label, null, fn);
                b.serialize = false; (b.options ||= {}).serialize = false;
                return b;
            };
            button("preview template", async () => {
                try {
                    const bs = widget(node, "template_base")?.value;
                    const j = await call("/seamstitch/draft_template/preview", { text: widget(node, "template_form")?.value || "", base: bs === AUTO ? "" : bs });
                    showText("1. An example of what the base drafts (stand-in answers)", [j.example, "=========================================", "2. Your layout applied to it", "=========================================", j.text].join("\n\n"));
                } catch (e) { alert("Can't preview: " + (e.message || e)); }
            });
            button("save template", async () => {
                const name = (widget(node, "template_name")?.value || "").trim();
                const text = widget(node, "template_form")?.value || "";
                const base = widget(node, "template_base")?.value;
                if (!name) { alert("Type a name for the template in 'template_name' first."); return; }
                try {
                    const j = await call("/seamstitch/draft_template/save", { name, text, base: base === AUTO ? "" : base });
                    setChoices(node, j.choices, j.entry);
                    alert(`Saved as "${j.entry}". It is now in the template list.`);
                } catch (e) { alert("Not saved: " + (e.message || e)); }
            });
            button("delete template", async () => {
                const cur = widget(node, "template")?.value || "";
                if (!cur.startsWith(PREFIX)) { alert("Pick one of your own templates ('my: …') in the template list first; the built-in ones can't be deleted."); return; }
                if (!confirm(`Delete the saved template "${cur}"? This can't be undone.`)) return;
                try {
                    const j = await call("/seamstitch/draft_template/delete", { entry: cur });
                    setChoices(node, j.choices, null);
                } catch (e) { alert("Not deleted: " + (e.message || e)); }
            });
            // the list as saved on disk (a workflow opened later may hold an entry that was deleted, or miss a new one)
            call("/seamstitch/draft_template/list").then((j) => { setChoices(node, j.choices, widget(node, "template")?.value); refreshExample(node); }).catch(() => { });
            return r;
        };
    },
});
