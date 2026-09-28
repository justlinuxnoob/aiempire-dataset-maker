"""Builds the dataset-maker workflows (drag-into-ComfyUI format + API format).

  workflows/AI_Empire_Dataset_Maker_FireRed11.json    <- FireRed Image Edit 1.1 (open-source, default)
  workflows/AI_Empire_Dataset_Maker_Qwen2511.json     <- Qwen Image Edit 2511 (open-source)
  workflows/AI_Empire_Dataset_Maker_NanoBanana.json   <- Nano Banana Pro with your own Google key
  workflows/api/*_api.json                            <- API format (serverless later)

Edit-model workflows: face (+ optional body photo) -> edit model -> saved as <name>_raw
                      -> Z-Image realism pass -> saved as <name>  (the one to train on)
Run:  python tools/build_workflow.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = "https://huggingface.co"

M = {
    "clip": ("qwen_2.5_vl_7b_fp8_scaled.safetensors", f"{HF}/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders"),
    "vae": ("qwen_image_vae.safetensors", f"{HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors", "vae"),
    "z_unet": ("z_image_turbo_bf16.safetensors", f"{HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors", "diffusion_models"),
    "z_clip": ("qwen_3_4b.safetensors", f"{HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders"),
    "z_vae": ("ae.safetensors", f"{HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors", "vae"),
}

CONFIGS = {
    "FireRed11": {
        "kind": "edit",
        "title": "FireRed Image Edit 1.1",
        "unet": ("FireRed-Image-Edit-1.1-transformer.safetensors", f"{HF}/FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI/resolve/main/FireRed-Image-Edit-1.1-transformer.safetensors", "diffusion_models"),
        "lora": ("FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors", f"{HF}/FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI/resolve/main/FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors", "loras"),
        "steps": 8,
        "ref_method": False,
        "notes": [
            "Edit model: **FireRed Image Edit 1.1** (Apache 2.0, by Xiaohongshu), built for identity consistency. 41 GB file.",
            "80 GB GPU (A100/H100): leave UNet weight_dtype = default. 48 GB GPU (A6000/A40/L40S): set weight_dtype = **fp8_e4m3fn**.",
        ],
    },
    "Qwen2511": {
        "kind": "edit",
        "title": "Qwen Image Edit 2511",
        "unet": ("qwen_image_edit_2511_fp8mixed.safetensors", f"{HF}/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors", "diffusion_models"),
        "lora": ("Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", f"{HF}/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "loras"),
        "steps": 8,
        "ref_method": True,
        "notes": ["Edit model: **Qwen Image Edit 2511** (Apache 2.0). fp8mixed file, 20 GB, fits 24-48 GB GPUs."],
    },
    "NanoBanana": {"kind": "nanobanana", "title": "Nano Banana Pro"},
}

WIDGETS = {
    "LoadImage": ["image", None], "UNETLoader": ["unet_name", "weight_dtype"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"], "ModelSamplingAuraFlow": ["shift"],
    "CFGNorm": ["strength"], "CLIPLoader": ["clip_name", "type", "device"], "VAELoader": ["vae_name"],
    "AIEmpireDatasetPresets": ["trigger_word", "preset_set", "extra_description", "how_many", "start_at", "seed", None,
                               "custom_presets", "use_body_reference", "dataset_name"],
    "AIEmpireNanoBanana": ["auth", "model", "resolution", "api_key", "vertex_project", "vertex_location"],
    "FluxKontextImageScale": [], "TextEncodeQwenImageEditPlus": ["prompt"],
    "FluxKontextMultiReferenceLatentMethod": ["reference_latents_method"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "VAEDecode": [], "VAEEncode": [], "CLIPTextEncode": ["text"], "ConditioningZeroOut": [],
    "AIEmpireSaveDataset": ["dataset_name", "make_zip", "suffix"],
}

HOW_TO = [
    "**How to use**",
    "1. Load her face in **Your face** (clear, front-facing, even light).",
    "2. Optional: load a full-body photo of her in **Body reference**, select that box and press **Ctrl+B** to switch it on (it starts purple = off), then tick **use_body_reference** in Dataset Presets.",
    "3. **Dataset Presets**: trigger word (e.g. `zvx woman`), *extra_description* (e.g. `with long straight dark brown hair, brown eyes`), *dataset_name*.",
    "4. First run: **how_many = 3**. If they look like her, set 40 and run again.",
]


def build(key, cfg):
    nodes, links, lid = [], [], [0]

    def mp(m):
        return {"models": [{"name": m[0], "url": m[1], "directory": m[2]}]}

    def add(nid, ntype, pos, size, widgets, outputs, title=None, props=None, color=None, mode=0):
        n = {"id": nid, "type": ntype, "pos": pos, "size": size, "flags": {}, "order": 0, "mode": mode, "inputs": [],
             "outputs": [{"name": o[0], "type": o[1], "links": [], **({"shape": 6} if len(o) > 2 and o[2] else {})} for o in outputs],
             "properties": {"Node name for S&R": ntype, **(props or {})}, "widgets_values": widgets}
        if title:
            n["title"] = title
        if color:
            n["color"], n["bgcolor"] = color
        nodes.append(n)

    def node(nid):
        return next(n for n in nodes if n["id"] == nid)

    def link(src, slot, dst, name, typ, widget=False, optional=False):
        lid[0] += 1
        node(src)["outputs"][slot]["links"].append(lid[0])
        e = {"name": name, "type": typ, "link": lid[0]}
        if widget:
            e["widget"] = {"name": name}
        if optional:
            e["shape"] = 7
        node(dst)["inputs"].append(e)
        links.append([lid[0], src, slot, dst, len(node(dst)["inputs"]) - 1, typ])

    def socket(nid, name, typ):
        node(nid)["inputs"].append({"name": name, "type": typ, "link": None, "shape": 7})

    ORANGE, GREEN, PURPLE = ("#3d2a14", "#2a1d0e"), ("#1f3320", "#162416"), ("#2e2240", "#1f172b")
    PRESET_W = ["zvx woman", "core40", "", 3, 1, 42, "fixed", "", False, "my_influencer"]
    PRESET_OUT = [("prompts", "STRING", True), ("captions", "STRING", True), ("widths", "INT", True), ("heights", "INT", True),
                  ("seeds", "INT", True), ("count", "INT"), ("realism_prompts", "STRING", True), ("dataset_name", "STRING")]

    # ---- shared inputs
    add(1, "LoadImage", [380, 0], [320, 360], ["your_face.png", "image"], [("IMAGE", "IMAGE"), ("MASK", "MASK")], title="Your face", color=GREEN)
    add(19, "LoadImage", [380, 420], [320, 360], ["your_body.png", "image"], [("IMAGE", "IMAGE"), ("MASK", "MASK")],
        title="Body reference (optional, Ctrl+B to turn on)", color=PURPLE, mode=4)
    add(9, "AIEmpireDatasetPresets", [740, 0], [400, 400], PRESET_W, PRESET_OUT, title="Dataset Presets", color=ORANGE)

    if cfg["kind"] == "nanobanana":
        note = "\n".join([f"## AI Empire · Dataset Maker ({cfg['title']})", "", *HOW_TO,
                          "5. **Nano Banana box**: choose *Vertex AI* (Google $300 free credits) or *AI Studio API key*.",
                          "",
                          "**Your Google key (set it on the RunPod template, not in the workflow)**",
                          "- Vertex AI: env `VERTEX_SA_JSON` = the whole service-account JSON (use a RunPod *secret*). Project is read from it.",
                          "- AI Studio: env `GEMINI_API_KEY` = your key. Or paste it in *api_key* (then don't share this workflow!).",
                          "- Model: `gemini-3-pro-image` (Nano Banana Pro) or `gemini-3.1-flash-image` (Nano Banana 2).",
                          "- Refused shots (Google's filter) are skipped automatically. Saved as `<name>_nanobanana`.",
                          "- No GPU needed: this runs on Google. Use the cheapest pod."])
        add(18, "MarkdownNote", [0, 0], [360, 620], [note], [], title="READ ME")
        add(30, "AIEmpireNanoBanana", [1180, 0], [400, 330], ["Vertex AI", "gemini-3-pro-image", "2K", "", "", "global"],
            [("images", "IMAGE", True), ("captions", "STRING", True)], title="Nano Banana (your Google key)", color=ORANGE)
        add(17, "AIEmpireSaveDataset", [1620, 0], [560, 640], ["my_influencer", True, "_nanobanana"], [], title="Save Dataset (Nano Banana)", color=GREEN)
        link(1, 0, 30, "face", "IMAGE")
        link(9, 0, 30, "prompts", "STRING")
        link(9, 1, 30, "captions", "STRING")
        link(9, 2, 30, "widths", "INT")
        link(9, 3, 30, "heights", "INT")
        link(19, 0, 30, "body", "IMAGE", optional=True)
        link(30, 0, 17, "images", "IMAGE")
        link(30, 1, 17, "captions", "STRING")
        link(9, 7, 17, "dataset_name", "STRING", widget=True)
        groups = [
            {"id": 1, "title": "1 · Your face + presets", "bounding": [370, -60, 780, 860], "color": "#b06634", "flags": {}},
            {"id": 2, "title": "2 · Nano Banana", "bounding": [1170, -60, 420, 420], "color": "#3f789e", "flags": {}},
            {"id": 3, "title": "3 · Save dataset", "bounding": [1610, -60, 580, 720], "color": "#6a8f4e", "flags": {}},
        ]
    else:
        note = "\n".join([f"## AI Empire · Dataset Maker ({cfg['title']})", "", *HOW_TO,
                          "", "**What gets saved** (in `output/datasets/`)",
                          "- `<name>_raw` = straight from the edit model",
                          "- `<name>` = after the **Z-Image realism pass** (real skin + light, same face). Train on this one.",
                          "- Don't want the realism pass? Right-click the *5 · Realism pass* group → Bypass group nodes.",
                          "", "**Settings**", *[f"- {x}" for x in cfg["notes"]],
                          f"- Edit: {cfg['steps']} steps, CFG 1, euler / simple (Lightning LoRA). Max quality: Ctrl+B the LoRA and use 40 steps, CFG 4.",
                          "- Realism pass: denoise 0.30 (higher = more realistic but can change the face; 0.25-0.35 is the sweet spot).",
                          "", "**Models**",
                          *[f"- {m[2]}: [{m[0]}]({m[1]})" for m in (cfg["unet"], cfg["lora"], M["clip"], M["vae"], M["z_unet"], M["z_clip"], M["z_vae"])]])
        add(18, "MarkdownNote", [740, 460], [400, 700], [note], [], title="READ ME")
        # models
        add(2, "UNETLoader", [0, 0], [340, 82], [cfg["unet"][0], "default"], [("MODEL", "MODEL")], title="Edit model", props=mp(cfg["unet"]))
        add(3, "LoraLoaderModelOnly", [0, 120], [340, 82], [cfg["lora"][0], 1.0], [("MODEL", "MODEL")], title="Lightning 8-step LoRA", props=mp(cfg["lora"]))
        add(4, "ModelSamplingAuraFlow", [0, 240], [340, 58], [3.1], [("MODEL", "MODEL")])
        add(5, "CFGNorm", [0, 340], [340, 58], [1], [("MODEL", "MODEL")])
        add(6, "CLIPLoader", [0, 440], [340, 106], [M["clip"][0], "qwen_image", "default"], [("CLIP", "CLIP")], props=mp(M["clip"]))
        add(7, "VAELoader", [0, 580], [340, 58], [M["vae"][0]], [("VAE", "VAE")], props=mp(M["vae"]))
        add(8, "FluxKontextImageScale", [380, 820], [260, 30], [], [("IMAGE", "IMAGE")])
        # generate
        add(10, "TextEncodeQwenImageEditPlus", [1180, 0], [360, 160], [""], [("CONDITIONING", "CONDITIONING")], title="Prompt (from presets)")
        add(11, "TextEncodeQwenImageEditPlus", [1180, 220], [360, 160], [""], [("CONDITIONING", "CONDITIONING")], title="Negative (leave empty)")
        if cfg["ref_method"]:
            add(12, "FluxKontextMultiReferenceLatentMethod", [1580, 0], [300, 58], ["index_timestep_zero"], [("CONDITIONING", "CONDITIONING")])
            add(13, "FluxKontextMultiReferenceLatentMethod", [1580, 220], [300, 58], ["index_timestep_zero"], [("CONDITIONING", "CONDITIONING")])
        add(14, "EmptySD3LatentImage", [1580, 420], [300, 106], [1024, 1024, 1], [("LATENT", "LATENT")])
        add(15, "KSampler", [1920, 0], [320, 262], [42, "fixed", cfg["steps"], 1, "euler", "simple", 1], [("LATENT", "LATENT")], title="KSampler (edit)")
        add(16, "VAEDecode", [1920, 320], [220, 46], [], [("IMAGE", "IMAGE")])
        # realism pass
        add(20, "UNETLoader", [1180, 760], [340, 82], [M["z_unet"][0], "default"], [("MODEL", "MODEL")], title="Z-Image Turbo", props=mp(M["z_unet"]))
        add(21, "ModelSamplingAuraFlow", [1180, 880], [340, 58], [3], [("MODEL", "MODEL")])
        add(22, "CLIPLoader", [1180, 980], [340, 106], [M["z_clip"][0], "lumina2", "default"], [("CLIP", "CLIP")], props=mp(M["z_clip"]))
        add(23, "VAELoader", [1180, 1120], [340, 58], [M["z_vae"][0]], [("VAE", "VAE")], props=mp(M["z_vae"]))
        add(24, "CLIPTextEncode", [1580, 760], [320, 120], [""], [("CONDITIONING", "CONDITIONING")], title="Realism prompt (from presets)")
        add(25, "ConditioningZeroOut", [1580, 920], [240, 26], [], [("CONDITIONING", "CONDITIONING")])
        add(26, "VAEEncode", [1580, 1000], [240, 46], [], [("LATENT", "LATENT")])
        add(27, "KSampler", [1920, 760], [320, 262], [42, "fixed", 8, 1, "dpmpp_2m_sde", "beta", 0.3], [("LATENT", "LATENT")], title="KSampler (realism, denoise 0.30)")
        add(28, "VAEDecode", [1920, 1080], [220, 46], [], [("IMAGE", "IMAGE")])
        # save
        add(17, "AIEmpireSaveDataset", [2300, 0], [520, 620], ["my_influencer", False, "_raw"], [], title="Save RAW (edit model only)", color=GREEN)
        add(29, "AIEmpireSaveDataset", [2300, 760], [520, 620], ["my_influencer", True, ""], [], title="Save FINAL (after realism pass) ← train on this", color=GREEN)

        link(2, 0, 3, "model", "MODEL")
        link(3, 0, 4, "model", "MODEL")
        link(4, 0, 5, "model", "MODEL")
        link(1, 0, 8, "image", "IMAGE")
        for enc in (10, 11):
            link(6, 0, enc, "clip", "CLIP")
            link(7, 0, enc, "vae", "VAE")
            link(8, 0, enc, "image1", "IMAGE")
            link(19, 0, enc, "image2", "IMAGE", optional=True)
            socket(enc, "image3", "IMAGE")
        link(9, 0, 10, "prompt", "STRING", widget=True)
        link(9, 2, 14, "width", "INT", widget=True)
        link(9, 3, 14, "height", "INT", widget=True)
        link(5, 0, 15, "model", "MODEL")
        if cfg["ref_method"]:
            link(10, 0, 12, "conditioning", "CONDITIONING")
            link(11, 0, 13, "conditioning", "CONDITIONING")
            link(12, 0, 15, "positive", "CONDITIONING")
            link(13, 0, 15, "negative", "CONDITIONING")
        else:
            link(10, 0, 15, "positive", "CONDITIONING")
            link(11, 0, 15, "negative", "CONDITIONING")
        link(14, 0, 15, "latent_image", "LATENT")
        link(9, 4, 15, "seed", "INT", widget=True)
        link(15, 0, 16, "samples", "LATENT")
        link(7, 0, 16, "vae", "VAE")
        # realism
        link(20, 0, 21, "model", "MODEL")
        link(22, 0, 24, "clip", "CLIP")
        link(9, 6, 24, "text", "STRING", widget=True)
        link(24, 0, 25, "conditioning", "CONDITIONING")
        link(16, 0, 26, "pixels", "IMAGE")
        link(23, 0, 26, "vae", "VAE")
        link(21, 0, 27, "model", "MODEL")
        link(24, 0, 27, "positive", "CONDITIONING")
        link(25, 0, 27, "negative", "CONDITIONING")
        link(26, 0, 27, "latent_image", "LATENT")
        link(9, 4, 27, "seed", "INT", widget=True)
        link(27, 0, 28, "samples", "LATENT")
        link(23, 0, 28, "vae", "VAE")
        # save
        link(16, 0, 17, "images", "IMAGE")
        link(9, 1, 17, "captions", "STRING")
        link(9, 7, 17, "dataset_name", "STRING", widget=True)
        link(28, 0, 29, "images", "IMAGE")
        link(9, 1, 29, "captions", "STRING")
        link(9, 7, 29, "dataset_name", "STRING", widget=True)
        groups = [
            {"id": 1, "title": "1 · Models", "bounding": [-20, -60, 380, 720], "color": "#444", "flags": {}},
            {"id": 2, "title": "2 · Your face + presets", "bounding": [370, -60, 780, 1240], "color": "#b06634", "flags": {}},
            {"id": 3, "title": "3 · Edit model", "bounding": [1170, -60, 1090, 620], "color": "#3f789e", "flags": {}},
            {"id": 5, "title": "5 · Realism pass (Z-Image)", "bounding": [1170, 700, 1090, 500], "color": "#8a5a9e", "flags": {}},
            {"id": 4, "title": "4 · Save dataset", "bounding": [2290, -60, 540, 1460], "color": "#6a8f4e", "flags": {}},
        ]

    # execution-order hint
    for i, n in enumerate(sorted(nodes, key=lambda n: (n["pos"][0], n["pos"][1]))):
        n["order"] = i

    for l in links:
        assert l[0] in node(l[1])["outputs"][l[2]]["links"], l
        assert node(l[3])["inputs"][l[4]]["link"] == l[0], l

    ids = {"FireRed11": "5e0c2f64-7a1b-4c8e-9d3f-f1e3d11a0001", "Qwen2511": "5e0c2f64-7a1b-4c8e-9d3f-a1e3e4e5e6e7",
           "NanoBanana": "5e0c2f64-7a1b-4c8e-9d3f-0a0ba0a0a0b1"}
    workflow = {"id": ids[key], "revision": 0, "last_node_id": max(n["id"] for n in nodes), "last_link_id": lid[0],
                "nodes": nodes, "links": links, "groups": groups, "config": {},
                "extra": {"ds": {"scale": 0.55, "offset": [60, 120]}}, "version": 0.4}

    # API format (bypassed nodes and their links dropped)
    bypassed = {n["id"] for n in nodes if n["mode"] == 4}
    api = {}
    for n in nodes:
        if n["type"] == "MarkdownNote" or n["id"] in bypassed:
            continue
        inputs = {w: v for w, v in zip(WIDGETS[n["type"]], n["widgets_values"]) if w}
        for inp in n["inputs"]:
            if inp["link"] is None:
                continue
            l = next(x for x in links if x[0] == inp["link"])
            if l[1] in bypassed:
                inputs.pop(inp["name"], None)
                continue
            inputs[inp["name"]] = [str(l[1]), l[2]]
        api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs, "_meta": {"title": n.get("title", n["type"])}}

    os.makedirs(os.path.join(ROOT, "workflows", "api"), exist_ok=True)
    with open(os.path.join(ROOT, "workflows", f"AI_Empire_Dataset_Maker_{key}.json"), "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ROOT, "workflows", "api", f"AI_Empire_Dataset_Maker_{key}_api.json"), "w", encoding="utf-8") as f:
        json.dump(api, f, indent=2, ensure_ascii=False)
    print("ok", key, len(nodes), "nodes", len(links), "links")


for k, c in CONFIGS.items():
    build(k, c)
