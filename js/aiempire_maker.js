// AI Empire · Dataset Maker box:
//  - preview strip: 3 photos of the chosen body type (starting at test_photo), no generating needed
//  - 🔑 RunPod key button: saves the key on the pod (/workspace/.runpod_key), never inside the workflow file
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "AIEmpireDatasetMaker";
const SHOW = 3;

function el(tag, style, text) {
  const e = document.createElement(tag);
  if (style) e.style.cssText = style;
  if (text != null) e.textContent = text;
  return e;
}

app.registerExtension({
  name: "aiempire.datasetMaker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;
      const w = (name) => node.widgets?.find((x) => x.name === name);

      // ---- 🔑 RunPod key
      const keyBtn = node.addWidget("button", "🔑 RunPod key (only for Nano Banana / Seedream)", null, async () => {
        const key = prompt("Paste your RunPod API key (runpod.io → Settings → API Keys).\nIt's saved on this pod only, not in the workflow. Empty = remove it.", "");
        if (key === null) return;
        try {
          const res = await api.fetchApi("/aiempire/runpod_key", { method: "POST", body: JSON.stringify({ key: key.trim() }) });
          setKeyLabel((await res.json()).saved);
        } catch (e) { alert("Could not save the key: " + e); }
      });
      keyBtn.serialize = false;
      keyBtn.options = { ...(keyBtn.options || {}), serialize: false };
      const setKeyLabel = (saved) => {
        const t = saved ? "🔑 RunPod key: saved ✅ (click to change)" : "🔑 RunPod key: not set (only needed for Nano Banana / Seedream)";
        keyBtn.name = t; keyBtn.label = t;
        node.setDirtyCanvas(true, true);
      };
      api.fetchApi("/aiempire/runpod_key").then((r) => r.json()).then((j) => setKeyLabel(j.saved)).catch(() => {});

      // ---- preview strip
      const box = el("div", "display:flex;flex-direction:column;gap:4px;width:100%;height:100%;box-sizing:border-box;padding:2px 4px;");
      const head = el("div", "font:12px sans-serif;color:#bbb;", "Preview");
      const row = el("div", "display:flex;gap:6px;flex:1;min-height:0;");
      const nav = el("div", "display:flex;gap:6px;align-items:center;font:12px sans-serif;color:#bbb;");
      const prev = el("button", "cursor:pointer;padding:2px 10px;", "◀");
      const next = el("button", "cursor:pointer;padding:2px 10px;", "▶");
      const info = el("span", "flex:1;text-align:center;", "");
      nav.append(prev, info, next);
      box.append(head, row, nav);
      const cells = [];
      for (let i = 0; i < SHOW; i++) {
        const cell = el("div", "flex:1;display:flex;flex-direction:column;align-items:center;min-width:0;cursor:pointer;");
        const img = el("img", "width:100%;height:100%;object-fit:contain;border-radius:4px;background:#111;min-height:0;flex:1;");
        const cap = el("div", "font:11px sans-serif;color:#aaa;", "");
        cell.append(img, cap);
        cell.title = "Click to use this photo for the test";
        row.append(cell);
        cells.push({ cell, img, cap });
      }
      const dom = node.addDOMWidget("preview", "aiempire_preview", box, { serialize: false, hideOnZoom: false, getMinHeight: () => 260 });
      dom.serialize = false;

      let files = [], loadedSet = null;
      const refresh = async () => {
        const set = w("body_type")?.value;
        if (!set) return;
        if (set !== loadedSet) {
          try {
            files = (await (await api.fetchApi(`/aiempire/preset_files?set=${encodeURIComponent(set)}`)).json()).files || [];
          } catch { files = []; }
          loadedSet = set;
        }
        const t = w("test_photo");
        const start = Math.max(1, Math.min(t?.value || 1, Math.max(files.length - SHOW + 1, 1)));
        head.textContent = files.length ? `Preview · ${set} · ${files.length} photos (nothing is generated)` : `Preview · ${set}: no photos`;
        info.textContent = files.length ? `photos ${start}–${Math.min(start + SHOW - 1, files.length)} of ${files.length}` : "";
        cells.forEach((c, i) => {
          const f = files[start - 1 + i];
          c.cell.style.visibility = f ? "visible" : "hidden";
          if (!f) return;
          const n = start + i;
          c.img.src = api.apiURL(`/aiempire/preset_thumb?set=${encodeURIComponent(set)}&file=${encodeURIComponent(f)}`);
          const chosen = n === (t?.value || 1);
          c.cap.textContent = chosen ? `#${n} ← test photo` : `#${n}`;
          c.img.style.outline = chosen ? "2px solid #e8a33d" : "none";
          c.cell.onclick = () => { if (t) { t.value = n; refresh(); node.setDirtyCanvas(true, true); } };
        });
      };
      const step = (d) => {
        const t = w("test_photo");
        if (!t || !files.length) return;
        t.value = Math.max(1, Math.min(files.length, (t.value || 1) + d));
        refresh();
        node.setDirtyCanvas(true, true);
      };
      prev.onclick = () => step(-SHOW);
      next.onclick = () => step(SHOW);

      for (const name of ["body_type", "test_photo"]) {
        const wid = w(name);
        if (!wid) continue;
        const cb = wid.callback;
        wid.callback = function () { const rr = cb?.apply(this, arguments); refresh(); return rr; };
      }
      node.aiempireRefresh = refresh;
      setTimeout(refresh, 50);
      if (node.size[1] < 620) node.setSize([Math.max(node.size[0], 460), 620]);
      return r;
    };

    // values arrive after creation when a workflow is loaded
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => this.aiempireRefresh?.(), 50);
      return r;
    };
  },
});
