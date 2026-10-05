// AI Empire · Prompt Writer (local) box: shows the prompt it wrote after every run (read-only, copy it from there)
import { app } from "../../scripts/app.js";

const NODE = "AIEmpirePromptWriter";

app.registerExtension({
  name: "aiempire.promptWriter",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const ta = document.createElement("textarea");
      ta.readOnly = true;
      ta.placeholder = "The prompt shows up here after you press Run.";
      ta.style.cssText = "width:100%;height:100%;box-sizing:border-box;resize:none;font:12px sans-serif;" +
        "background:#151515;color:#ddd;border:1px solid #333;border-radius:4px;padding:6px;";
      const out = this.addDOMWidget("last_prompt", "aiempire_last_prompt", ta, { serialize: false, hideOnZoom: false });
      out.computeSize = (w) => [w, 120];
      this._aiempirePromptBox = ta;
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
