// AI Empire · Template Presets: "Upload template photos" button.
// Picks many photos (or one .zip) and uploads them to input/templates/<set>/.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// node type -> name of its template-set dropdown
const NODES = { AIEmpireTemplatePresets: "template_set", AIEmpireBodyPresetMaker: "source_set" };
const PLACEHOLDER = "upload templates first";
const OK_EXT = /\.(png|jpe?g|webp|bmp|txt|zip)$/i;

async function uploadOne(file, subfolder) {
  const body = new FormData();
  body.append("image", file, file.name);
  body.append("type", "input");
  body.append("subfolder", subfolder);
  body.append("overwrite", "true");
  const res = await api.fetchApi("/upload/image", { method: "POST", body });
  if (res.status !== 200) throw new Error(`${file.name}: ${res.status} ${res.statusText}`);
}

app.registerExtension({
  name: "aiempire.templatePresets",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const SET_WIDGET = NODES[nodeData.name];
    if (!SET_WIDGET) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const node = this;
      const setWidget = () => node.widgets.find((w) => w.name === SET_WIDGET);

      const input = document.createElement("input");
      input.type = "file";
      input.multiple = true;
      input.accept = "image/png,image/jpeg,image/webp,image/bmp,.txt,.zip";
      input.style.display = "none";
      document.body.appendChild(input);

      // open the file picker straight from the click (browsers block it after a delay), ask for the set name after
      const btn = node.addWidget("button", "📁 Upload template photos", null, () => {
        input.value = "";
        input.click();
      });
      const setLabel = (t) => { btn.name = t; btn.label = t; };
      btn.serialize = false;
      btn.options = { ...(btn.options || {}), serialize: false };

      input.addEventListener("change", async () => {
        const files = [...input.files].filter((f) => OK_EXT.test(f.name));
        if (!files.length) return;
        const current = setWidget()?.value;
        const suggested = current && current !== PLACEHOLDER ? current : "my_templates";
        const name = (prompt(`Put these ${files.length} files in which template set? (folder in input/templates)`, suggested) || "").trim();
        if (!name) return;
        const set = name.replace(/[^A-Za-z0-9_\-]+/g, "_");
        let done = 0;
        const failed = [];
        for (const f of files) {
          setLabel(`⏳ Uploading ${done + 1}/${files.length}…`);
          node.setDirtyCanvas(true, true);
          try {
            await uploadOne(f, `templates/${set}`);
          } catch (e) {
            failed.push(e.message);
          }
          done++;
        }
        const w = setWidget();
        if (w) {
          const vals = (w.options.values || []).filter((v) => v !== PLACEHOLDER);
          if (!vals.includes(set)) vals.push(set);
          w.options.values = vals.sort();
          w.value = set;
        }
        setLabel(failed.length
          ? `⚠️ ${done - failed.length}/${files.length} uploaded (${failed.length} failed)`
          : `✅ ${files.length} uploaded to "${set}" · upload more`);
        if (failed.length) console.warn("[AI Empire] upload failed:", failed);
        node.setDirtyCanvas(true, true);
      });

      const onRemoved = node.onRemoved;
      node.onRemoved = function () {
        input.remove();
        return onRemoved?.apply(this, arguments);
      };
      return r;
    };
  },
});

// ---- live progress on "Save Dataset": title + bar + browser tab title
function showProgress(progress, left, dataset) {
  const m = /^(\d+)\/(\d+)$/.exec(progress || "");
  if (!m) return;
  const done = +m[1], total = +m[2];
  for (const node of app.graph?._nodes || []) {
    if (node.type !== "AIEmpireSaveDataset") continue;
    node.aiempireProgress = { done, total };
    if (!node.aiempireBaseTitle) node.aiempireBaseTitle = node.title;
    node.title = `${node.aiempireBaseTitle}  ·  ${done >= total ? "✅" : "⏳"} ${done} / ${total}`;
    node.setDirtyCanvas(true, true);
  }
  document.title = done >= total ? `✅ ${dataset} done (${total})` : `⏳ ${done}/${total} · ${dataset}`;
}

api.addEventListener("aiempire.progress", (e) => showProgress(e.detail.progress, e.detail.left, e.detail.dataset));

app.registerExtension({
  name: "aiempire.saveProgress",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AIEmpireSaveDataset") return;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      const r = onExecuted?.apply(this, arguments);
      if (msg?.progress?.[0]) showProgress(msg.progress[0], 0, "dataset");
      return r;
    };
    const onDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = onDraw?.apply(this, arguments);
      const p = this.aiempireProgress;
      if (p && p.total && !this.flags?.collapsed) {
        const w = this.size[0] - 20, y = 4;
        ctx.fillStyle = "#333"; ctx.fillRect(10, y, w, 6);
        ctx.fillStyle = p.done >= p.total ? "#5cb85c" : "#e8a33d";
        ctx.fillRect(10, y, (w * Math.min(p.done, p.total)) / p.total, 6);
      }
      return r;
    };
  },
});
