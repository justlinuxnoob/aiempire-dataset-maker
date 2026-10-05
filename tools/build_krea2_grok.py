"""Builds the AI Empire Krea 2 Turbo + Grok prompt workflow (UI + API format).

Same models and settings as the Krea 2 Turbo realism setup we studied, rebuilt with open nodes only:
  - Krea 2 Turbo fp8 + Qwen3-VL 4B fp8 + Wan 2.1 VAE
  - LoRAs: skin detail 0.5, RawGirl V2 0.75, your character LoRA
  - ModelSamplingAuraFlow shift 6, 8 steps, CFG 1 (BasicGuider), euler_ancestral,
    beta57 = core BetaSamplingScheduler alpha 0.5 / beta 0.7 (identical to RES4LYF's beta57, no extra node pack)
  - AI Empire Grok Prompt (photo → prompt / idea → prompt / your own prompt)
  - AI Empire Photo Finish (phone look) + Save Image (no workflow inside)
The pod downloads the models with KREA_TURBO=1.

Run:  python tools/build_krea2_grok.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = "https://huggingface.co"
K2 = f"{HF}/Comfy-Org/Krea-2/resolve/main"
KL = f"{HF}/dci05049/krea2/resolve/main"

MODELS = {
    "unet": ("krea2_turbo_fp8_scaled.safetensors", f"{K2}/diffusion_models/krea2_turbo_fp8_scaled.safetensors", "diffusion_models"),
    "clip": ("qwen3vl_4b_fp8_scaled.safetensors", f"{K2}/text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "text_encoders"),
    "vae": ("wan_2.1_vae.safetensors", f"{HF}/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors", "vae"),
    "skin": ("skindetails_krea2_loraholic.safetensors", f"{KL}/skindetails_krea2_loraholic.safetensors", "loras"),
    "raw": ("RawGirlV2_epoch_10.safetensors", f"{KL}/RawGirlV2_epoch_10.safetensors", "loras"),
}

IDEA = "mirror selfie in a bright bathroom, holding a gold iPhone, oversized grey hoodie, morning light"

WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
    "ModelSamplingAuraFlow": ["shift"],
    "LoadImage": ["image", None],
    "AIEmpireGrokPrompt": ["mode", "trigger_word", "hair_and_eyes", "idea", "model", "temperature", "variation", None],
    "CLIPTextEncode": ["text"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "BetaSamplingScheduler": ["steps", "alpha", "beta"],
    "KSamplerSelect": ["sampler_name"],
    "RandomNoise": ["noise_seed", None],
    "BasicGuider": [],
    "SamplerCustomAdvanced": [],
    "VAEDecode": [],
    "PreviewImage": [],
    "AIEmpirePhotoFinish": ["exposure", "gamma", "contrast", "saturation", "vibrance", "glow", "glow_radius",
                            "glow_threshold", "lens_softness", "hand_shake", "iso", "grain_size", "final_contrast",
                            "jpeg_quality", "seed", None],
    "AIEmpireSaveClean": ["filename_prefix", "format", "quality"],
}

NOTE = "\n".join([
    "## AI Empire · Krea 2 Turbo + Grok prompts",
    "",
    "**1 · Your LoRA** (once): JupyterLab (port 8888) → `models/loras/` → upload your `.safetensors`, press **R** in ComfyUI.",
    "Pick it in **Your LoRA**. Strength **0.9** (1.0 = strongest likeness).",
    "",
    "**2 · 🔑 xAI key** (once): click the button on the **Grok Prompt** box, paste your key from console.x.ai.",
    "It's saved on the pod, never inside this workflow.",
    "",
    "**3 · Grok Prompt** box → *mode*:",
    "- 📷 **photo → prompt**: upload a photo in **Reference photo**. Grok copies its pose, outfit, place and light.",
    "  Optional *idea* = changes (\"make the outfit black\").",
    "- 💡 **idea → prompt**: type a short idea, Grok writes the full prompt.",
    "- ✍️ **my prompt**: no Grok, your *idea* text is the prompt.",
    "*trigger_word* always goes first. *hair_and_eyes* (optional) goes right after it.",
    "Grok never describes her face, hair, eyes or body: your LoRA knows them.",
    "",
    "**4 · Run.** The prompt Grok wrote shows in the Grok box. *variation* on **fixed** = Grok is asked once",
    "and every Run makes a new image from the same prompt. Change *variation* (or set randomize) for a new prompt.",
    "",
    "**Size:** 1088 × 1920 (9:16). 2:3 = 1024 × 1536. Square = 1280 × 1280.",
    "**GPU:** 24 GB+ (4090, 3090, 5090, A5000 and up). Grok costs well under 1 cent per prompt.",
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

    def link(src, slot, dst, name, typ, widget=False):
        lid[0] += 1
        node(src)["outputs"][slot]["links"].append(lid[0])
        inp = {"name": name, "type": typ, "link": lid[0]}
        if widget:  # a widget turned into an input (e.g. the prompt text coming from Grok)
            inp["widget"] = {"name": name}
        node(dst)["inputs"].append(inp)
        links.append([lid[0], src, slot, dst, len(node(dst)["inputs"]) - 1, typ])

    GREEN, ORANGE, BLUE = ("#1f3320", "#162416"), ("#3d2a14", "#2a1d0e"), ("#1d2a3d", "#141d2a")

    add(1, "MarkdownNote", [-500, 0], [460, 760], [NOTE], [], title="READ ME")
    # your settings
    add(5, "LoraLoaderModelOnly", [0, 0], [420, 82], ["my_lora.safetensors", 0.9], [("MODEL", "MODEL")],
        title="Your LoRA", color=ORANGE)
    add(20, "LoadImage", [0, 120], [420, 360], ["your_reference_photo.png", "image"],
        [("IMAGE", "IMAGE"), ("MASK", "MASK")], title="Reference photo (📷 mode)")
    add(21, "AIEmpireGrokPrompt", [0, 520], [420, 520],
        ["💡 idea → prompt", "zvx woman", "", IDEA, "grok-4.20-0309-non-reasoning", 0.7, 0, "fixed"],
        [("prompt", "STRING")], title="Grok Prompt", color=BLUE)
    add(8, "EmptyLatentImage", [0, 1080], [420, 106], [1088, 1920, 1], [("LATENT", "LATENT")], title="Size")
    # result
    add(11, "AIEmpireSaveClean", [460, 0], [560, 900], ["aiempire_krea", "JPEG", 95], [], title="Your image", color=GREEN)
    add(23, "PreviewImage", [460, 940], [560, 420], [], [], title="Before the phone look")
    # machine room
    add(2, "UNETLoader", [1060, 0], [380, 82], [MODELS["unet"][0], "default"], [("MODEL", "MODEL")],
        title="Krea 2 Turbo", props=mp("unet"))
    add(3, "CLIPLoader", [1060, 120], [380, 106], [MODELS["clip"][0], "krea2", "default"], [("CLIP", "CLIP")], props=mp("clip"))
    add(4, "VAELoader", [1060, 260], [380, 58], [MODELS["vae"][0]], [("VAE", "VAE")], props=mp("vae"))
    add(12, "LoraLoaderModelOnly", [1060, 360], [380, 82], [MODELS["skin"][0], 0.5], [("MODEL", "MODEL")],
        title="Skin detail LoRA", props=mp("skin"))
    add(13, "LoraLoaderModelOnly", [1060, 480], [380, 82], [MODELS["raw"][0], 0.75], [("MODEL", "MODEL")],
        title="RawGirl V2 LoRA (phone-photo realism)", props=mp("raw"))
    add(14, "ModelSamplingAuraFlow", [1060, 600], [380, 58], [6], [("MODEL", "MODEL")])
    add(6, "CLIPTextEncode", [1060, 700], [380, 120], [""], [("CONDITIONING", "CONDITIONING")], title="Prompt (from Grok)")
    add(15, "BasicGuider", [1060, 860], [240, 46], [], [("GUIDER", "GUIDER")])
    add(16, "BetaSamplingScheduler", [1060, 940], [300, 106], [8, 0.5, 0.7], [("SIGMAS", "SIGMAS")], title="beta57 scheduler")
    add(17, "KSamplerSelect", [1060, 1080], [300, 58], ["euler_ancestral"], [("SAMPLER", "SAMPLER")])
    add(18, "RandomNoise", [1060, 1180], [300, 82], [42, "randomize"], [("NOISE", "NOISE")], title="Seed")
    add(9, "SamplerCustomAdvanced", [1480, 0], [300, 106], [], [("output", "LATENT"), ("denoised_output", "LATENT")])
    add(10, "VAEDecode", [1480, 160], [220, 46], [], [("IMAGE", "IMAGE")])
    add(22, "AIEmpirePhotoFinish", [1480, 260], [340, 440],
        [-0.05, 1.05, 0.98, 1.09, -0.04, 0.075, 50.0, 0.3, 0.35, 1.9, "ISO 800", 0.5, 1.1, 98, 42, "randomize"],
        [("IMAGE", "IMAGE")], title="Photo Finish (phone look)")

    # model chain: Krea 2 → skin detail → RawGirl → your LoRA → shift 6
    link(2, 0, 12, "model", "MODEL")
    link(12, 0, 13, "model", "MODEL")
    link(13, 0, 5, "model", "MODEL")
    link(5, 0, 14, "model", "MODEL")
    # prompt: reference photo → Grok → text encoder
    link(20, 0, 21, "image", "IMAGE")
    link(3, 0, 6, "clip", "CLIP")
    link(21, 0, 6, "text", "STRING", widget=True)
    # sampling
    link(14, 0, 15, "model", "MODEL")
    link(6, 0, 15, "conditioning", "CONDITIONING")
    link(14, 0, 16, "model", "MODEL")
    link(18, 0, 9, "noise", "NOISE")
    link(15, 0, 9, "guider", "GUIDER")
    link(17, 0, 9, "sampler", "SAMPLER")
    link(16, 0, 9, "sigmas", "SIGMAS")
    link(8, 0, 9, "latent_image", "LATENT")
    link(9, 0, 10, "samples", "LATENT")
    link(4, 0, 10, "vae", "VAE")
    # finish + save
    link(10, 0, 23, "images", "IMAGE")
    link(10, 0, 22, "image", "IMAGE")
    link(22, 0, 11, "images", "IMAGE")

    groups = [
        {"id": 1, "title": "Your settings", "bounding": [-20, -60, 460, 1270], "color": "#a1612c", "flags": {}},
        {"id": 2, "title": "Your image", "bounding": [440, -60, 600, 1440], "color": "#3f8f4e", "flags": {}},
        {"id": 3, "title": "⚙️ Machine room · no need to touch", "bounding": [1040, -60, 800, 1340], "color": "#555555", "flags": {}},
    ]

    for i, n in enumerate(sorted(nodes, key=lambda n: (n["pos"][0], n["pos"][1]))):
        n["order"] = i
    for l in links:
        assert l[0] in node(l[1])["outputs"][l[2]]["links"], l
        assert node(l[3])["inputs"][l[4]]["link"] == l[0], l

    workflow = {"id": "5e0c2f64-7a1b-4c8e-9d3f-4b2ea2000003", "revision": 0,
                "last_node_id": max(n["id"] for n in nodes), "last_link_id": lid[0],
                "nodes": nodes, "links": links, "groups": groups, "config": {},
                "extra": {"ds": {"scale": 0.6, "offset": [540, 120]}}, "version": 0.4}

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
    with open(os.path.join(ROOT, "workflows", "AI_Empire_Krea2_Turbo_Grok.json"), "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ROOT, "workflows", "api", "AI_Empire_Krea2_Turbo_Grok_api.json"), "w", encoding="utf-8") as f:
        json.dump(api, f, indent=2, ensure_ascii=False)
    print("ok Krea2_Turbo_Grok", len(nodes), "nodes", len(links), "links")


if __name__ == "__main__":
    build()
