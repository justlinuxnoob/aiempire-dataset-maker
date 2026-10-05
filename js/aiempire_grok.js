// AI Empire · Grok Prompt box:
//  - 🔑 xAI key button: saves the key on the pod (/workspace/.xai_key), never inside the workflow file
//  - shows the prompt Grok wrote after every run (read-only, copy it from there)
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "AIEmpireGrokPrompt";

app.registerExtension({
  name: "aiempire.grokPrompt",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;

      const keyBtn = node.addWidget("button", "🔑 xAI key", null, async () => {
        const key = prompt("Paste your xAI API key (console.x.ai → API Keys).\nIt's saved on this pod only, not in the workflow. Empty = remove it.", "");
        if (key === null) return;
        try {
          const res = await api.fetchApi("/aiempire/xai_key", { method: "POST", body: JSON.stringify({ key: key.trim() }) });
          setKeyLabel((await res.json()).saved);
        } catch (e) { alert("Could not save the key: " + e); }
      });
      keyBtn.serialize = false;
      keyBtn.options = { ...(keyBtn.options || {}), serialize: false };
      const setKeyLabel = (saved) => {
        const t = saved ? "🔑 xAI key: saved ✅ (click to change)" : "🔑 xAI key: not set (needed for 📷 / 💡 modes)";
        keyBtn.name = t; keyBtn.label = t;
        node.setDirtyCanvas(true, true);
      };
      api.fetchApi("/aiempire/xai_key").then((r) => r.json()).then((j) => setKeyLabel(j.saved)).catch(() => {});

      // read-only box that shows the last prompt
      const ta = document.createElement("textarea");
      ta.readOnly = true;
      ta.placeholder = "The prompt Grok writes shows up here after you press Run.";
      ta.style.cssText = "width:100%;height:100%;box-sizing:border-box;resize:none;font:12px sans-serif;" +
        "background:#151515;color:#ddd;border:1px solid #333;border-radius:4px;padding:6px;";
      const out = node.addDOMWidget("last_prompt", "aiempire_last_prompt", ta, { serialize: false, hideOnZoom: false });
      out.computeSize = (w) => [w, 120];
      node._aiempirePromptBox = ta;
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const t = message?.text;
      if (this._aiempirePromptBox && t && t.length) this._aiempirePromptBox.value = Array.isArray(t) ? t.join("\n") : String(t);
    };
  },
});
