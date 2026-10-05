"""Builds the AI Empire Krea 2 Turbo realism workflow (UI + API format).

A node-for-node rebuild of the Krea 2 Turbo realism setup we studied:
  - Krea 2 Turbo fp8 + Qwen3-VL 4B fp8 (type krea2) + Wan 2.1 VAE
  - LoRAs on model + clip: skin detail 0.5, RawGirl V2 0.75, your character LoRA 1.0
  - Resolution Selector 9:16 at 2 MP (multiple of 32) -> empty latent
  - ModelSamplingAuraFlow shift 6, 8 steps, CFG 1, euler_ancestral,
    beta57 = core BetaSamplingScheduler alpha 0.5 / beta 0.7 (identical to RES4LYF's beta57, no extra node pack)
  - Photo Finish (phone look) in place of the closed Camera Look + Renoise + post-process nodes
  - Save Image with no workflow inside + Image Comparer (raw vs finished)
The pod downloads the base models with KREA_TURBO=1; the two style LoRAs and your LoRA go in models/loras.

Run:  python tools/build_krea2_realism.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = "https://huggingface.co"
K2 = f"{HF}/Comfy-Org/Krea-2/resolve/main"

MODELS = {
    "unet": ("krea2_turbo_fp8_scaled.safetensors", f"{K2}/diffusion_models/krea2_turbo_fp8_scaled.safetensors", "diffusion_models"),
    "clip": ("qwen3vl_4b_fp8_scaled.safetensors", f"{K2}/text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "text_encoders"),
    "vae": ("wan_2.1_vae.safetensors", f"{HF}/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors", "vae"),
}
SKIN_LORA = "skindetails_krea2_loraholic.safetensors"
RAW_LORA = "RawGirlV2_epoch_10.safetensors"

# 3 blocks: who (trigger + hair/eyes) / the shot (pose, outfit, place, light) / the photo look
PROMPT = ("zvx woman, long straight red hair, green eyes,\n\n"
          "she sits cross-legged on a white bed with rumpled sheets, wearing an oversized grey hoodie, "
          "holding a gold iPhone in one hand, head turned towards the camera, soft morning window light.\n\n"
          "casual amateur shot iphone, ")

WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "ModelSamplingAuraFlow": ["shift"],
    "ResolutionSelector": ["aspect_ratio", "megapixels", "multiple"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "CLIPTextEncode": ["text"],
    "BetaSamplingScheduler": ["steps", "alpha", "beta"],
    "KSamplerSelect": ["sampler_name"],
    "RandomNoise": ["noise_seed", None],
    "BasicGuider": [],
    "SamplerCustomAdvanced": [],
    "VAEDecode": [],
    "AIEmpirePhotoFinish": ["exposure", "gamma", "contrast", "saturation", "vibrance", "glow", "glow_radius",
                            "glow_threshold", "lens_softness", "hand_shake", "iso", "grain_size", "final_contrast",
                            "jpeg_quality", "seed", None],
    "AIEmpireSaveClean": ["filename_prefix", "format", "quality"],
    "Image Comparer (rgthree)": [None],
}

NOTE = "\n".join([
    "## AI Empire · Krea 2 Turbo realism",
    "",
    "**1 · LoRAs** (once): JupyterLab (port 8888) → `models/loras/` → upload:",
    f"- your character LoRA, plus `{SKIN_LORA}` and `{RAW_LORA}`.",
    "Press **R** in ComfyUI, then pick your file in **Your LoRA** (strength **1.0**).",
    "",
    "**2 · Prompt**, 3 blocks:",
    "- **who:** trigger word + hair and eyes (`zvx woman, long straight red hair, green eyes`)",
    "- **the shot:** pose, outfit, props, place, light, where she looks. Plain sentences, like describing a photo.",
    "- **the look:** `casual amateur shot iphone`",
    "",
    "**3 · Run.** Saved to `output/` as JPEG with no workflow inside. The slider shows raw vs finished.",
    "",
    "**Size:** 9:16 at 2 MP. Change the ratio in **Resolution**.",
    "**GPU:** 24 GB+ (4090, 3090, 5090, A5000 and up).",
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
        if widget:  # a widget turned into an input (width/height from the Resolution box)
            inp["widget"] = {"name": name}
        node(dst)["inputs"].append(inp)
        links.append([lid[0], src, slot, dst, len(node(dst)["inputs"]) - 1, typ])

    GREEN, ORANGE = ("#1f3320", "#162416"), ("#3d2a14", "#2a1d0e")

    add(1, "MarkdownNote", [-500, 0], [460, 700], [NOTE], [], title="READ ME")
    # your settings
    add(5, "LoraLoader", [0, 0], [420, 126], ["my_lora.safetensors", 1.0, 1.0],
        [("MODEL", "MODEL"), ("CLIP", "CLIP")], title="Your LoRA", color=ORANGE)
    add(6, "CLIPTextEncode", [0, 160], [420, 380], [PROMPT], [("CONDITIONING", "CONDITIONING")], title="Prompt", color=GREEN)
    add(7, "ResolutionSelector", [0, 580], [420, 126], ["9:16 (Portrait Widescreen)", 2, 32],
        [("width", "INT"), ("height", "INT")], title="Resolution")
    # result
    add(11, "AIEmpireSaveClean", [460, 0], [560, 900], ["clean", "JPEG", 100], [], title="Your image", color=GREEN)
    add(23, "Image Comparer (rgthree)", [460, 940], [560, 900], [[]], [("images", "IMAGE")],
        title="Raw vs finished", props={"comparer_mode": "Slide"})
    # machine room
    add(2, "UNETLoader", [1060, 0], [380, 82], [MODELS["unet"][0], "default"], [("MODEL", "MODEL")],
        title="Krea 2 Turbo", props=mp("unet"))
    add(3, "CLIPLoader", [1060, 120], [380, 106], [MODELS["clip"][0], "krea2", "default"], [("CLIP", "CLIP")], props=mp("clip"))
    add(4, "VAELoader", [1060, 260], [380, 58], [MODELS["vae"][0]], [("VAE", "VAE")], props=mp("vae"))
    add(12, "LoraLoader", [1060, 360], [380, 126], [SKIN_LORA, 0.5, 0.5], [("MODEL", "MODEL"), ("CLIP", "CLIP")],
        title="Skin detail LoRA")
    add(13, "LoraLoader", [1060, 520], [380, 126], [RAW_LORA, 0.75, 0.75], [("MODEL", "MODEL"), ("CLIP", "CLIP")],
        title="RawGirl V2 LoRA")
    add(14, "ModelSamplingAuraFlow", [1060, 680], [380, 58], [6], [("MODEL", "MODEL")])
    add(8, "EmptyLatentImage", [1060, 780], [300, 106], [1088, 1920, 1], [("LATENT", "LATENT")])
    add(15, "BasicGuider", [1060, 920], [240, 46], [], [("GUIDER", "GUIDER")], title="CFG 1 guider")
    add(16, "BetaSamplingScheduler", [1060, 1000], [300, 106], [8, 0.5, 0.7], [("SIGMAS", "SIGMAS")], title="beta57 scheduler")
    add(17, "KSamplerSelect", [1060, 1140], [300, 58], ["euler_ancestral"], [("SAMPLER", "SAMPLER")])
    add(18, "RandomNoise", [1060, 1240], [300, 82], [42, "randomize"], [("NOISE", "NOISE")], title="Seed")
    add(9, "SamplerCustomAdvanced", [1480, 0], [300, 106], [], [("output", "LATENT"), ("denoised_output", "LATENT")])
    add(10, "VAEDecode", [1480, 160], [220, 46], [], [("IMAGE", "IMAGE")])
    add(22, "AIEmpirePhotoFinish", [1480, 260], [340, 440],
        [-0.05, 1.05, 0.98, 1.09, -0.04, 0.075, 50.0, 0.3, 0.35, 1.9, "ISO 400", 0.5, 1.0, 98, 42, "randomize"],
        [("IMAGE", "IMAGE")], title="Photo Finish (camera look + grain + grade)")

    # model + clip chain: Krea 2 → skin detail → RawGirl → your LoRA
    link(2, 0, 12, "model", "MODEL")
    link(3, 0, 12, "clip", "CLIP")
    link(12, 0, 13, "model", "MODEL")
    link(12, 1, 13, "clip", "CLIP")
    link(13, 0, 5, "model", "MODEL")
    link(13, 1, 5, "clip", "CLIP")
    link(5, 0, 14, "model", "MODEL")
    link(5, 1, 6, "clip", "CLIP")
    # size
    link(7, 0, 8, "width", "INT", widget=True)
    link(7, 1, 8, "height", "INT", widget=True)
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
    # finish, save, compare
    link(10, 0, 22, "image", "IMAGE")
    link(22, 0, 11, "images", "IMAGE")
    link(10, 0, 23, "image_a", "IMAGE")
    link(22, 0, 23, "image_b", "IMAGE")

    groups = [
        {"id": 1, "title": "Your settings", "bounding": [-20, -60, 460, 780], "color": "#a1612c", "flags": {}},
        {"id": 2, "title": "Your image", "bounding": [440, -60, 600, 1920], "color": "#3f8f4e", "flags": {}},
        {"id": 3, "title": "⚙️ Machine room · no need to touch", "bounding": [1040, -60, 800, 1400], "color": "#555555", "flags": {}},
    ]

    for i, n in enumerate(sorted(nodes, key=lambda n: (n["pos"][0], n["pos"][1]))):
        n["order"] = i
    for l in links:
        assert l[0] in node(l[1])["outputs"][l[2]]["links"], l
        assert node(l[3])["inputs"][l[4]]["link"] == l[0], l

    workflow = {"id": "5e0c2f64-7a1b-4c8e-9d3f-4b2ea2000004", "revision": 0,
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
    with open(os.path.join(ROOT, "workflows", "AI_Empire_Krea2_Turbo_Realism.json"), "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ROOT, "workflows", "api", "AI_Empire_Krea2_Turbo_Realism_api.json"), "w", encoding="utf-8") as f:
        json.dump(api, f, indent=2, ensure_ascii=False)
    print("ok Krea2_Turbo_Realism", len(nodes), "nodes", len(links), "links")


if __name__ == "__main__":
    build()
