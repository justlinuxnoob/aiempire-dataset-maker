"""Builds the AI Empire Krea 2 Turbo + your LoRA workflow (UI + API format).

Same graph as ComfyUI's official "Text to Image (Krea-2 Turbo)" template (8 steps, CFG 1, euler/simple,
fp8 model + text encoder, Qwen-Image VAE) + a LoRA loader for the student's character LoRA.
The pod downloads the models with KREA_TURBO=1.

Run:  python tools/build_krea_turbo.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K2 = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main"

MODELS = {
    "unet": ("krea2_turbo_fp8_scaled.safetensors", f"{K2}/diffusion_models/krea2_turbo_fp8_scaled.safetensors", "diffusion_models"),
    "clip": ("qwen3vl_4b_fp8_scaled.safetensors", f"{K2}/text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "text_encoders"),
    "vae": ("qwen_image_vae.safetensors", f"{K2}/vae/qwen_image_vae.safetensors", "vae"),
}

PROMPT = ("zvx woman, close-up selfie, looking at the camera, soft smile, cream knit sweater, cozy cafe, "
          "soft window light, candid smartphone photo, natural skin texture")

WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
    "CLIPTextEncode": ["text"],
    "ConditioningZeroOut": [],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "VAEDecode": [],
    "AIEmpireSaveClean": ["filename_prefix", "format", "quality"],
}

NOTE = "\n".join([
    "## AI Empire · Krea 2 Turbo + your LoRA",
    "",
    "**1 · Upload your LoRA** (once)",
    "JupyterLab (port 8888) → `models/loras/` → upload your `.safetensors`.",
    "Then press **R** in ComfyUI (refresh) so it shows up in the list.",
    "",
    "**2 · Your LoRA** box: pick your file. Strength **0.9** (0.8 = looser, 1.0 = strongest likeness).",
    "",
    "**3 · Prompt** box: start with your **trigger word** (the one from the Dataset Maker, e.g. `zvx woman`).",
    "Then: shot, outfit, place, light. Never describe her face, hair, eyes or body: the LoRA knows them.",
    "",
    "**4 · Run.** Images are saved to `output/` as JPEG with no workflow inside.",
    "",
    "**Testing which LoRA file is best?** Keep the prompt and set the KSampler seed to *fixed*,",
    "then swap the file in *Your LoRA* (1500 / 2000 / 2500 / final) and compare.",
    "",
    "**Size:** 1024 × 1536 (portrait, Instagram-ready). Square: 1280 × 1280.",
    "**GPU:** any 24 GB+ card (4090, 3090, 5090, A5000 and up).",
    "",
    "**Models** (the pod downloads them with `KREA_TURBO=1`)",
    *[f"- {m[2]}: [{m[0]}]({m[1]})" for m in MODELS.values()],
])


def build():
    nodes, links, lid = [], [], [0]

    def mp(key):
        m = MODELS[key]
        return {"models": [{"name": m[0], "url": m[1], "directory": m[2]}]}

    def add(nid, ntype, pos, size, widgets, outputs, title=None, props=None, color=None):
        n = {"id": nid, "type": ntype, "pos": pos, "size": size, "flags": {}, "order": 0, "mode": 0, "inputs": [],
             "outputs": [{"name": o[0], "type": o[1], "links": []} for o in outputs],
             "properties": {"Node name for S&R": ntype, **(props or {})}, "widgets_values": widgets}
        if title:
            n["title"] = title
        if color:
            n["color"], n["bgcolor"] = color
        nodes.append(n)

    def node(nid):
        return next(n for n in nodes if n["id"] == nid)

    def link(src, slot, dst, name, typ):
        lid[0] += 1
        node(src)["outputs"][slot]["links"].append(lid[0])
        node(dst)["inputs"].append({"name": name, "type": typ, "link": lid[0]})
        links.append([lid[0], src, slot, dst, len(node(dst)["inputs"]) - 1, typ])

    GREEN, ORANGE = ("#1f3320", "#162416"), ("#3d2a14", "#2a1d0e")

    add(1, "MarkdownNote", [-480, 0], [440, 640], [NOTE], [], title="READ ME")
    # your settings
    add(5, "LoraLoaderModelOnly", [0, 0], [400, 82], ["my_lora.safetensors", 0.9], [("MODEL", "MODEL")],
        title="Your LoRA", color=ORANGE)
    add(6, "CLIPTextEncode", [0, 120], [400, 220], [PROMPT], [("CONDITIONING", "CONDITIONING")], title="Prompt", color=GREEN)
    add(8, "EmptyLatentImage", [0, 380], [400, 106], [1024, 1536, 1], [("LATENT", "LATENT")], title="Size")
    # result
    add(11, "AIEmpireSaveClean", [440, 0], [520, 760], ["aiempire_krea", "JPEG", 95], [], title="Your image", color=GREEN)
    # machine room
    add(2, "UNETLoader", [1000, 0], [360, 82], [MODELS["unet"][0], "default"], [("MODEL", "MODEL")],
        title="Krea 2 Turbo", props=mp("unet"))
    add(3, "CLIPLoader", [1000, 120], [360, 106], [MODELS["clip"][0], "krea2", "default"], [("CLIP", "CLIP")], props=mp("clip"))
    add(4, "VAELoader", [1000, 260], [360, 58], [MODELS["vae"][0]], [("VAE", "VAE")], props=mp("vae"))
    add(7, "ConditioningZeroOut", [1000, 360], [240, 26], [], [("CONDITIONING", "CONDITIONING")])
    add(9, "KSampler", [1000, 430], [320, 262], [42, "randomize", 8, 1.0, "euler", "simple", 1.0], [("LATENT", "LATENT")])
    add(10, "VAEDecode", [1000, 730], [220, 46], [], [("IMAGE", "IMAGE")])

    link(2, 0, 5, "model", "MODEL")
    link(3, 0, 6, "clip", "CLIP")
    link(6, 0, 7, "conditioning", "CONDITIONING")
    link(5, 0, 9, "model", "MODEL")
    link(6, 0, 9, "positive", "CONDITIONING")
    link(7, 0, 9, "negative", "CONDITIONING")
    link(8, 0, 9, "latent_image", "LATENT")
    link(9, 0, 10, "samples", "LATENT")
    link(4, 0, 10, "vae", "VAE")
    link(10, 0, 11, "images", "IMAGE")

    groups = [
        {"id": 1, "title": "Your settings", "bounding": [-20, -60, 440, 580], "color": "#a1612c", "flags": {}},
        {"id": 2, "title": "Your image", "bounding": [420, -60, 560, 860], "color": "#3f8f4e", "flags": {}},
        {"id": 3, "title": "⚙️ Machine room · no need to touch", "bounding": [980, -60, 420, 880], "color": "#555555", "flags": {}},
    ]

    for i, n in enumerate(sorted(nodes, key=lambda n: (n["pos"][0], n["pos"][1]))):
        n["order"] = i
    for l in links:
        assert l[0] in node(l[1])["outputs"][l[2]]["links"], l
        assert node(l[3])["inputs"][l[4]]["link"] == l[0], l

    workflow = {"id": "5e0c2f64-7a1b-4c8e-9d3f-4b2ea2000002", "revision": 0,
                "last_node_id": max(n["id"] for n in nodes), "last_link_id": lid[0],
                "nodes": nodes, "links": links, "groups": groups, "config": {},
                "extra": {"ds": {"scale": 0.7, "offset": [520, 120]}}, "version": 0.4}

    api = {}
    for n in nodes:
        if n["type"] == "MarkdownNote":
            continue
        inputs = {w: v for w, v in zip(WIDGETS[n["type"]], n["widgets_values"]) if w}
        for inp in n["inputs"]:
            l = next(x for x in links if x[0] == inp["link"])
            inputs[inp["name"]] = [str(l[1]), l[2]]
        api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs, "_meta": {"title": n.get("title", n["type"])}}

    os.makedirs(os.path.join(ROOT, "workflows", "api"), exist_ok=True)
    with open(os.path.join(ROOT, "workflows", "AI_Empire_Krea2_Turbo_LoRA.json"), "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ROOT, "workflows", "api", "AI_Empire_Krea2_Turbo_LoRA_api.json"), "w", encoding="utf-8") as f:
        json.dump(api, f, indent=2, ensure_ascii=False)
    print("ok Krea2_Turbo_LoRA", len(nodes), "nodes", len(links), "links")


if __name__ == "__main__":
    build()
