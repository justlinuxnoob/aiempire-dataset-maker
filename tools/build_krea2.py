"""Builds the AI Empire Krea2 RAW workflow (UI + API format).

  1 · Krea2 (RawGirl finetune) + realism LoRA stack, 8 steps
  2 · Flux 2 Klein 9B realism pass (2 steps, reference = the Krea2 image)
  3 · Skin detailer -> colour -> skin oil -> AI Empire Photo Finish -> save (no workflow inside)

Run:  python tools/build_krea2.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = "https://huggingface.co"

MODELS = {
    "krea_unet": ("RawGirlKrea2_v10_int8_convrot.safetensors", f"{HF}/dci05049/krea2/resolve/main/RawGirlKrea2_v10_int8_convrot.safetensors", "diffusion_models"),
    "krea_clip": ("qwen3vl_4b_bf16.safetensors", f"{HF}/Comfy-Org/Qwen3-VL/resolve/b58e627c376915e49cb6bba978416085aa31767f/text_encoders/qwen3vl_4b_bf16.safetensors", "text_encoders"),
    "krea_vae": ("wan_2.1_vae.safetensors", f"{HF}/dci05049/wan-animate/resolve/main/wan_2.1_vae.safetensors", "vae"),
    "klein_unet": ("flux-2-klein-9b.safetensors", f"{HF}/dci05049/flux2-klein-9b/resolve/main/flux-2-klein-9b.safetensors", "diffusion_models"),
    "klein_clip": ("qwen_3_8b_fp8mixed.safetensors", f"{HF}/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors", "text_encoders"),
    "klein_vae": ("flux2-vae.safetensors", f"{HF}/dci05049/flux2-klein-9b/resolve/main/flux2-vae.safetensors", "vae"),
}

KREA_LORAS = [  # (file, strength) - same stack as the reference pipeline (candid was loaded twice at 1.0 = 2.0)
    ("candid_krea2_loraholic.safetensors", 2.0),
    ("skindetails_krea2_loraholic.safetensors", 0.6),
    ("real_3d_krea2_loraholic.safetensors", 0.8),
    ("ass_v2_krea2_loraholic.safetensors", 1.4),
    ("breast_size_v2_krea2_loraholic.safetensors", 0.8),
]
KLEIN_LORAS = [
    ("f2k_9B_lcs_consist_20260415.safetensors", 1.0),
    ("Samsung_fluxklein9b.safetensors", 0.6),
    ("Klein_realistic_I2I.safetensors", 0.3),
    ("HighResolution9B.safetensors", 0.3),
]

PROMPT = ("candid smartphone photo of a young woman sitting by a cafe window, beige knit sweater, holding a coffee cup, "
          "natural daylight, soft smile, slightly messy hair, real skin texture, casual everyday moment")
KLEIN_PROMPT = ("improve quality and photorealism, realistic skin texture, delete imperfections, clean skin. "
                "low dynamic range lightning, flat lightning, remove tatoos")
SKIN_PROMPT = "photorealistic skin with natural pore-level texture, smooth even skin tone, soft subsurface scattering, clean sharp detail"

WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "ModelSamplingAuraFlow": ["shift"],
    "CLIPTextEncode": ["text"],
    "ConditioningZeroOut": [],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "EmptyFlux2LatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "VAEDecode": [], "VAEEncode": [], "ReferenceLatent": [], "GetImageSize": [], "BasicGuider": [],
    "SamplerCustomAdvanced": [], "PreviewImage": [],
    "RandomNoise": ["noise_seed", None],
    "KSamplerSelect": ["sampler_name"],
    "Flux2Scheduler": ["steps", "width", "height"],
    "SkinDetailer": ["mode", "texture_model", "texture_strength", "face_identity_protect", "sam3_model",
                     "segmentation_threshold", "mask_expand", "mask_feather", "seed", None, "steps", "cfg", "denoise",
                     "preserve_tone", "skin_prompt", "sampler_name", "scheduler", "processing_mode"],
    "FameGridColorFinish": ["color_strength", "sharpen_strength"],
    "RealismInjector": ["enable_micro_shadows", "shadow_strength", "shadow_radius", "shadow_detail_threshold",
                        "enable_eye_moisture", "moisture_strength", "catchlight_intensity", "tear_meniscus",
                        "enable_skin_oil", "oil_strength", "t_zone_gloss", "side_matte", "global_mix"],
    "AIEmpirePhotoFinish": ["exposure", "gamma", "contrast", "saturation", "vibrance", "glow", "glow_radius",
                            "glow_threshold", "lens_softness", "hand_shake", "iso", "grain_size", "final_contrast",
                            "jpeg_quality", "seed", None],
    "AIEmpireSaveClean": ["filename_prefix", "format", "quality"],
}

NOTE = "\n".join([
    "## AI Empire · Krea2 RAW",
    "",
    "**How to use**",
    "1. Write your prompt in **Prompt** (green box). Add your LoRA's trigger word at the start.",
    "2. Your own character LoRA: click **+ Add Lora** in *Krea2 LoRAs*, pick it, strength 0.8-1.0.",
    "3. Press **Run**. Final image is saved to `output/` as JPEG with **no workflow inside**.",
    "",
    "**What happens**",
    "- **1 · Krea2**: RawGirl Krea2 finetune + realism LoRA stack, 8 steps, CFG 1, 4.4 MP (9:16).",
    "- **2 · Klein realism pass**: Flux 2 Klein 9B re-renders the image in 2 steps for real skin and flat phone light.",
    "- **3 · Finish**: skin detailer (SAM3 skin mask + skin texture) → colour → skin oil → phone look (glow, softness, ISO 200 grain).",
    "",
    "**Tips**",
    "- Faster tests: set Krea2 latent to **1088 x 1920**.",
    "- Skip a stage: right-click its group → *Bypass Group Nodes*.",
    "- Turn a LoRA off: click its toggle in the LoRA box.",
    "- GPU: 48 GB+ (A6000 / A40 / L40S / 5090 / A100). Both models stay loaded.",
    "",
    "**Models** (the pod downloads them with `KREA2=1`)",
    *[f"- {m[2]}: [{m[0]}]({m[1]})" for m in MODELS.values()],
])


def lora_widgets(loras):
    return [{}, {"type": "PowerLoraLoaderHeaderWidget"},
            *[{"on": True, "lora": f, "strength": s, "strengthTwo": None} for f, s in loras], {}, ""]


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
        e = {"name": name, "type": typ, "link": lid[0]}
        if widget:
            e["widget"] = {"name": name}
        node(dst)["inputs"].append(e)
        links.append([lid[0], src, slot, dst, len(node(dst)["inputs"]) - 1, typ])

    GREEN, ORANGE = ("#1f3320", "#162416"), ("#3d2a14", "#2a1d0e")
    IMG = [("IMAGE", "IMAGE")]

    add(1, "MarkdownNote", [-460, 0], [420, 760], [NOTE], [], title="READ ME")

    # ---- 1 · Krea2
    add(2, "UNETLoader", [0, 0], [360, 82], [MODELS["krea_unet"][0], "default"], [("MODEL", "MODEL")], title="Krea2 model (RawGirl)", props=mp("krea_unet"))
    add(3, "CLIPLoader", [0, 120], [360, 106], [MODELS["krea_clip"][0], "krea2", "default"], [("CLIP", "CLIP")], props=mp("krea_clip"))
    add(4, "VAELoader", [0, 260], [360, 58], [MODELS["krea_vae"][0]], [("VAE", "VAE")], props=mp("krea_vae"))
    add(5, "Power Lora Loader (rgthree)", [0, 360], [360, 290], lora_widgets(KREA_LORAS), [("MODEL", "MODEL"), ("CLIP", "CLIP")], title="Krea2 LoRAs", color=ORANGE)
    add(6, "ModelSamplingAuraFlow", [0, 690], [360, 58], [6.0], [("MODEL", "MODEL")])
    add(7, "CLIPTextEncode", [400, 0], [420, 200], [PROMPT], [("CONDITIONING", "CONDITIONING")], title="Prompt", color=GREEN)
    add(8, "ConditioningZeroOut", [400, 240], [240, 26], [], [("CONDITIONING", "CONDITIONING")])
    add(9, "EmptyLatentImage", [400, 310], [300, 106], [1568, 2816, 1], [("LATENT", "LATENT")], title="Size (9:16, 4.4 MP)")
    add(10, "KSampler", [860, 0], [320, 262], [42, "randomize", 8, 1.0, "euler_ancestral", "simple", 1.0], [("LATENT", "LATENT")], title="KSampler (Krea2)")
    add(11, "VAEDecode", [860, 300], [220, 46], [], IMG)
    add(12, "PreviewImage", [860, 390], [320, 400], [], [], title="1 · Krea2 raw")

    # ---- 2 · Klein realism pass
    add(20, "UNETLoader", [1260, 0], [360, 82], [MODELS["klein_unet"][0], "default"], [("MODEL", "MODEL")], title="Flux 2 Klein 9B", props=mp("klein_unet"))
    add(21, "CLIPLoader", [1260, 120], [360, 106], [MODELS["klein_clip"][0], "flux2", "default"], [("CLIP", "CLIP")], props=mp("klein_clip"))
    add(22, "VAELoader", [1260, 260], [360, 58], [MODELS["klein_vae"][0]], [("VAE", "VAE")], props=mp("klein_vae"))
    add(23, "Power Lora Loader (rgthree)", [1260, 360], [360, 230], lora_widgets(KLEIN_LORAS), [("MODEL", "MODEL"), ("CLIP", "CLIP")], title="Klein LoRAs", color=ORANGE)
    add(24, "CLIPTextEncode", [1660, 0], [380, 160], [KLEIN_PROMPT], [("CONDITIONING", "CONDITIONING")], title="Realism prompt")
    add(25, "VAEEncode", [1660, 200], [220, 46], [], [("LATENT", "LATENT")])
    add(26, "ReferenceLatent", [1660, 280], [220, 46], [], [("CONDITIONING", "CONDITIONING")])
    add(27, "GetImageSize", [1660, 360], [220, 66], [], [("width", "INT"), ("height", "INT"), ("batch_size", "INT")])
    add(28, "EmptyFlux2LatentImage", [1660, 460], [260, 106], [1024, 1024, 1], [("LATENT", "LATENT")])
    add(29, "Flux2Scheduler", [1660, 600], [260, 106], [2, 1024, 1024], [("SIGMAS", "SIGMAS")])
    add(30, "KSamplerSelect", [2080, 0], [260, 58], ["exp_heun_2_x0"], [("SAMPLER", "SAMPLER")])
    add(31, "RandomNoise", [2080, 100], [260, 82], [42, "randomize"], [("NOISE", "NOISE")])
    add(32, "BasicGuider", [2080, 220], [220, 46], [], [("GUIDER", "GUIDER")])
    add(33, "SamplerCustomAdvanced", [2080, 300], [260, 106], [], [("output", "LATENT"), ("denoised_output", "LATENT")])
    add(34, "VAEDecode", [2080, 450], [220, 46], [], IMG)
    add(35, "PreviewImage", [2080, 540], [320, 400], [], [], title="2 · After Klein pass")

    # ---- 3 · Finish
    add(40, "SkinDetailer", [2480, 0], [380, 520],
        ["skin texture (stable)", "auto (1xSkinContrast-High-SuperUltraCompact)", 1.0, 0.0, "auto (sam3.1_multiplex_fp16)",
         0.75, 4, 12.0, 42, "randomize", 8, 1.0, 0.22, 0.85, SKIN_PROMPT, "euler", "simple", "auto"],
        [("image", "IMAGE"), ("skin_mask", "MASK")], title="Skin Detailer")
    add(41, "FameGridColorFinish", [2900, 0], [300, 82], [0.05, 0.0], IMG, title="Colour finish")
    add(42, "RealismInjector", [2900, 120], [320, 380],
        [False, 0.3, 2.0, 0.015, False, 0.7, 0.8, 0.6, True, 0.8, 0.7, 0.5, 1.0], IMG, title="Skin oil")
    add(43, "AIEmpirePhotoFinish", [3260, 0], [340, 440],
        [-0.05, 1.05, 0.98, 1.09, -0.04, 0.075, 50.0, 0.3, 0.25, 1.9, "ISO 200", 0.5, 1.1, 100, 42, "randomize"],
        IMG, title="Photo Finish (phone look)", color=ORANGE)
    add(44, "AIEmpireSaveClean", [3640, 0], [420, 560], ["aiempire_krea2", "JPEG", 100], [], title="Save (no workflow inside)", color=GREEN)

    # links · stage 1
    link(2, 0, 5, "model", "MODEL")
    link(3, 0, 5, "clip", "CLIP")
    link(5, 0, 6, "model", "MODEL")
    link(5, 1, 7, "clip", "CLIP")
    link(7, 0, 8, "conditioning", "CONDITIONING")
    link(6, 0, 10, "model", "MODEL")
    link(7, 0, 10, "positive", "CONDITIONING")
    link(8, 0, 10, "negative", "CONDITIONING")
    link(9, 0, 10, "latent_image", "LATENT")
    link(10, 0, 11, "samples", "LATENT")
    link(4, 0, 11, "vae", "VAE")
    link(11, 0, 12, "images", "IMAGE")
    # stage 2
    link(20, 0, 23, "model", "MODEL")
    link(21, 0, 23, "clip", "CLIP")
    link(23, 1, 24, "clip", "CLIP")
    link(11, 0, 25, "pixels", "IMAGE")
    link(22, 0, 25, "vae", "VAE")
    link(24, 0, 26, "conditioning", "CONDITIONING")
    link(25, 0, 26, "latent", "LATENT")
    link(11, 0, 27, "image", "IMAGE")
    link(27, 0, 28, "width", "INT", widget=True)
    link(27, 1, 28, "height", "INT", widget=True)
    link(27, 0, 29, "width", "INT", widget=True)
    link(27, 1, 29, "height", "INT", widget=True)
    link(23, 0, 32, "model", "MODEL")
    link(26, 0, 32, "conditioning", "CONDITIONING")
    link(31, 0, 33, "noise", "NOISE")
    link(32, 0, 33, "guider", "GUIDER")
    link(30, 0, 33, "sampler", "SAMPLER")
    link(29, 0, 33, "sigmas", "SIGMAS")
    link(28, 0, 33, "latent_image", "LATENT")
    link(33, 0, 34, "samples", "LATENT")
    link(22, 0, 34, "vae", "VAE")
    link(34, 0, 35, "images", "IMAGE")
    # stage 3
    link(34, 0, 40, "image", "IMAGE")
    link(40, 0, 41, "image", "IMAGE")
    link(41, 0, 42, "image", "IMAGE")
    link(42, 0, 43, "image", "IMAGE")
    link(43, 0, 44, "images", "IMAGE")

    groups = [
        {"id": 1, "title": "1 · Krea2", "bounding": [-20, -60, 1220, 880], "color": "#3f789e", "flags": {}},
        {"id": 2, "title": "2 · Klein realism pass", "bounding": [1240, -60, 1180, 1020], "color": "#8a5a9e", "flags": {}},
        {"id": 3, "title": "3 · Finish + save", "bounding": [2460, -60, 1620, 640], "color": "#6a8f4e", "flags": {}},
    ]

    for i, n in enumerate(sorted(nodes, key=lambda n: (n["pos"][0], n["pos"][1]))):
        n["order"] = i
    for l in links:
        assert l[0] in node(l[1])["outputs"][l[2]]["links"], l
        assert node(l[3])["inputs"][l[4]]["link"] == l[0], l

    workflow = {"id": "5e0c2f64-7a1b-4c8e-9d3f-4b2ea2000001", "revision": 0,
                "last_node_id": max(n["id"] for n in nodes), "last_link_id": lid[0],
                "nodes": nodes, "links": links, "groups": groups, "config": {},
                "extra": {"ds": {"scale": 0.45, "offset": [520, 120]}}, "version": 0.4}

    # API format
    api = {}
    for n in nodes:
        if n["type"] == "MarkdownNote":
            continue
        if n["type"] == "Power Lora Loader (rgthree)":
            inputs = {"PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"}}
            k = 0
            for w in n["widgets_values"]:
                if isinstance(w, dict) and "lora" in w:
                    k += 1
                    inputs[f"lora_{k}"] = {"on": w["on"], "lora": w["lora"], "strength": w["strength"]}
            inputs["➕ Add Lora"] = ""
        else:
            inputs = {w: v for w, v in zip(WIDGETS[n["type"]], n["widgets_values"]) if w}
        for inp in n["inputs"]:
            l = next(x for x in links if x[0] == inp["link"])
            inputs[inp["name"]] = [str(l[1]), l[2]]
        api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs, "_meta": {"title": n.get("title", n["type"])}}

    os.makedirs(os.path.join(ROOT, "workflows", "api"), exist_ok=True)
    with open(os.path.join(ROOT, "workflows", "AI_Empire_Krea2_RAW.json"), "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ROOT, "workflows", "api", "AI_Empire_Krea2_RAW_api.json"), "w", encoding="utf-8") as f:
        json.dump(api, f, indent=2, ensure_ascii=False)
    print("ok Krea2_RAW", len(nodes), "nodes", len(links), "links")


if __name__ == "__main__":
    build()
