"""Builds the dataset-maker workflows (drag-into-ComfyUI format + API format).

  workflows/AI_Empire_Dataset_Maker_FireRed11.json    <- FireRed Image Edit 1.1 (open-source, default)
  workflows/AI_Empire_Dataset_Maker_Qwen2511.json     <- Qwen Image Edit 2511 (open-source)
  workflows/AI_Empire_Dataset_Maker_NanoBanana.json   <- Nano Banana Pro with your own Google key
  workflows/AI_Empire_Dataset_Maker_Qwen21_Templates.json <- Qwen-Image 2.1 over a folder of your own template photos
  workflows/AI_Empire_Dataset_Maker_Qwen21_BodyPresets.json <- turns base photos into a body-type preset (athletic, curvy ...)
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
    "Qwen21": {
        "kind": "edit",
        "q21": True,
        "title": "Qwen-Image 2.1",
        "unet": ("qwen_image_2.1_int8_convrot.safetensors", f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors", "diffusion_models"),
        "clip": ("qwen3vl_8b_int8_convrot.safetensors", f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors", "text_encoders"),
        "vae": ("qwen_image_2.1_vae_bf16.safetensors", f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors", "vae"),
        "steps": 25,
        "size_scale": 1.5,
        "notes": [
            "Edit model: **Qwen-Image 2.1** (best open edit model right now, native 2K). int8 file, runs on 24-48 GB GPUs.",
            "Images are made at **1.5x size** (~2.3 MP): *size_scale* in Dataset Presets. 1.0 = faster, 2.0 = max detail.",
            "*resolution* in the Qwen 2.1 encoder = how big her face photo is fed in (1536 default).",
        ],
    },
    "Qwen21_Templates": {
        "kind": "templates",
        "title": "Qwen-Image 2.1 · your template photos",
        "unet": ("qwen_image_2.1_int8_convrot.safetensors", f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors", "diffusion_models"),
        "clip": ("qwen3vl_8b_int8_convrot.safetensors", f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors", "text_encoders"),
        "vae": ("qwen_image_2.1_vae_bf16.safetensors", f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors", "vae"),
        "steps": 25,
    },
    "Qwen21_BodyPresets": {
        "kind": "body",
        "title": "Qwen-Image 2.1 · body preset maker",
        "unet": ("qwen_image_2.1_int8_convrot.safetensors", f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors", "diffusion_models"),
        "clip": ("qwen3vl_8b_int8_convrot.safetensors", f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors", "text_encoders"),
        "vae": ("qwen_image_2.1_vae_bf16.safetensors", f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors", "vae"),
        "steps": 25,
    },
}

WIDGETS = {
    "LoadImage": ["image", None], "UNETLoader": ["unet_name", "weight_dtype"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"], "ModelSamplingAuraFlow": ["shift"],
    "CFGNorm": ["strength"], "CLIPLoader": ["clip_name", "type", "device"], "VAELoader": ["vae_name"],
    "AIEmpireDatasetPresets": ["trigger_word", "preset_set", "extra_description", "how_many", "start_at", "seed", None,
                               "custom_presets", "use_body_reference", "dataset_name", "size_scale"],
    "AIEmpireTemplatePresets": ["template_set", "trigger_word", "extra_description", "how_many", "start_at", "seed", None,
                                "use_body_reference", "dataset_name", "skip_done"],
    "AIEmpireBodyPresetMaker": ["source_set", "output_set", "body_target", "instructions", "how_many", "start_at", "seed", None, "skip_done", "use_body_example"],
    "AIEmpireSaveTemplateSet": ["output_set", "source_set"],
    "AIEmpireNanoBanana": ["auth", "model", "resolution", "api_key", "vertex_project", "vertex_location"],
    "FluxKontextImageScale": [], "TextEncodeQwenImageEditPlus": ["prompt"],
    "FluxKontextMultiReferenceLatentMethod": ["reference_latents_method"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "VAEDecode": [], "VAEEncode": [], "CLIPTextEncode": ["text"], "ConditioningZeroOut": [],
    "AIEmpireSaveDataset": ["dataset_name", "make_zip", "suffix"],
    "QwenImage21Cache": ["device", "dtype"],
    "TextEncodeQwenImage21": ["prompt", "negative_prompt", "resolution"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
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
    q21 = cfg.get("q21", False)
    PRESET_W = ["zvx woman", "core40", "", 3, 1, 42, "fixed", "", False, "my_influencer", cfg.get("size_scale", 1.0)]
    PRESET_OUT = [("prompts", "STRING", True), ("captions", "STRING", True), ("widths", "INT", True), ("heights", "INT", True),
                  ("seeds", "INT", True), ("count", "INT"), ("realism_prompts", "STRING", True), ("dataset_name", "STRING")]

    # ---- shared inputs
    if cfg["kind"] == "body":
        return build_body(key, cfg, nodes, links, lid, add, node, link, mp, ORANGE, GREEN)
    add(1, "LoadImage", [380, 0], [320, 360], ["your_face.png", "image"], [("IMAGE", "IMAGE"), ("MASK", "MASK")], title="Your face", color=GREEN)
    add(19, "LoadImage", [380, 420], [320, 360], ["your_body.png", "image"], [("IMAGE", "IMAGE"), ("MASK", "MASK")],
        title="Body reference (optional, Ctrl+B to turn on)", color=PURPLE, mode=4)
    if cfg["kind"] != "templates":
        add(9, "AIEmpireDatasetPresets", [740, 0], [400, 400], PRESET_W, PRESET_OUT, title="Dataset Presets", color=ORANGE)

    if cfg["kind"] == "templates":
        note = "\n".join([
            f"## AI Empire · Dataset Maker ({cfg['title']})", "",
            "Every photo you upload = 1 dataset image: **same pose, outfit, place and light as the photo, her face from *Your face*.** 50 photos in = 50 dataset images out.", "",
            "**How to use**",
            "1. Load her face in **Your face** (clear, front-facing, even light).",
            "2. On **Template Presets** click **📁 Upload template photos**, select all your photos at once (or one `.zip`), then type a set name (e.g. `athletic`). They go to `input/templates/<set>/`.",
            "3. Pick the set in *template_set*. Fill *trigger_word* and *extra_description* (e.g. `with long straight blonde hair, blue eyes`): hair and eye colour here, or she keeps the template's hair.",
            "4. First run: **how_many = 3**. If they look like her, set **how_many = 0** (= all photos) and run again. Finished ones are skipped (*skip_done*), so a crash just means press Run again.",
            "5. Optional: Ctrl+B the **Body reference** box, load a full-body photo of her and tick *use_body_reference*. Off = her body comes from each template.", "",
            "**Template tips**",
            "- One woman per photo, face visible. A covered or turned-away face gives a bad image.",
            "- Captions: put `swap_01.txt` next to `swap_01.jpg` (e.g. `mirror selfie, black top, bathroom`) and it's added after the trigger word. No .txt = trigger word only.",
            "- One folder per body type (`athletic`, `curvy` …) = your body presets.", "",
            "**What gets saved**: `output/datasets/<name>/<name>_<template>.png` + `.txt`, and `<name>.zip`.",
            "Download: `http://<pod-url>/view?filename=<name>.zip&subfolder=datasets&type=output`", "",
            "**Settings**",
            "- Qwen-Image 2.1, 25 steps, CFG 1, euler / simple.",
            "- Output = same shape and composition as each template. Size = *resolution* in the encoder: 1536 ≈ 2.3 MP (native 2K). 1024 = faster, 2048 = max detail.", "",
            "**Models**",
            *[f"- {m[2]}: [{m[0]}]({m[1]})" for m in (cfg["unet"], cfg["clip"], cfg["vae"])]])
        add(18, "MarkdownNote", [740, 520], [420, 760], [note], [], title="READ ME")
        add(40, "AIEmpireTemplatePresets", [740, 0], [420, 470],
            ["my_templates", "zvx woman", "", 3, 1, 42, "fixed", False, "my_influencer", True],
            [("templates", "IMAGE", True), ("prompts", "STRING", True), ("captions", "STRING", True),
             ("seeds", "INT", True), ("count", "INT"), ("dataset_name", "STRING"), ("file_names", "STRING", True)],
            title="Template Presets (upload your photos here)", color=ORANGE)
        add(2, "UNETLoader", [0, 0], [340, 82], [cfg["unet"][0], "default"], [("MODEL", "MODEL")], title="Edit model", props=mp(cfg["unet"]))
        add(5, "QwenImage21Cache", [0, 120], [340, 82], ["auto", "default"], [("MODEL", "MODEL")])
        add(6, "CLIPLoader", [0, 440], [340, 106], [cfg["clip"][0], "qwen_image", "default"], [("CLIP", "CLIP")], props=mp(cfg["clip"]))
        add(7, "VAELoader", [0, 580], [340, 58], [cfg["vae"][0]], [("VAE", "VAE")], props=mp(cfg["vae"]))
        add(10, "TextEncodeQwenImage21", [1200, 0], [360, 220], ["", "", 1536],
            [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")],
            title="Qwen 2.1 encoder (1 = template, 2 = face, 3 = body)")
        add(15, "KSampler", [1940, 0], [320, 262], [42, "fixed", cfg["steps"], 1, "euler", "simple", 1], [("LATENT", "LATENT")], title="KSampler (edit)")
        add(16, "VAEDecode", [1940, 320], [220, 46], [], [("IMAGE", "IMAGE")])
        add(17, "AIEmpireSaveDataset", [2300, 0], [520, 620], ["my_influencer", True, ""], [], title="Save Dataset ← train on this", color=GREEN)

        link(2, 0, 5, "model", "MODEL")
        link(6, 0, 10, "clip", "CLIP")
        link(7, 0, 10, "vae", "VAE", optional=True)
        link(40, 0, 10, "images.image_1", "IMAGE", optional=True)
        link(1, 0, 10, "images.image_2", "IMAGE", optional=True)
        link(19, 0, 10, "images.image_3", "IMAGE", optional=True)
        link(40, 1, 10, "prompt", "STRING", widget=True)
        link(5, 0, 15, "model", "MODEL")
        link(10, 0, 15, "positive", "CONDITIONING")
        link(10, 1, 15, "negative", "CONDITIONING")
        link(10, 2, 15, "latent_image", "LATENT")
        link(40, 3, 15, "seed", "INT", widget=True)
        link(15, 0, 16, "samples", "LATENT")
        link(7, 0, 16, "vae", "VAE")
        link(16, 0, 17, "images", "IMAGE")
        link(40, 2, 17, "captions", "STRING")
        link(40, 5, 17, "dataset_name", "STRING", widget=True)
        link(40, 6, 17, "file_names", "STRING", optional=True)
        groups = [
            {"id": 1, "title": "1 · Models", "bounding": [-20, -60, 380, 720], "color": "#444", "flags": {}},
            {"id": 2, "title": "2 · Your face + template photos", "bounding": [370, -60, 810, 1360], "color": "#b06634", "flags": {}},
            {"id": 3, "title": "3 · Edit model", "bounding": [1190, -60, 1090, 620], "color": "#3f789e", "flags": {}},
            {"id": 4, "title": "4 · Save dataset", "bounding": [2290, -60, 540, 700], "color": "#6a8f4e", "flags": {}},
        ]
    elif cfg["kind"] == "nanobanana":
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
                          *(["- `<name>` = straight from Qwen-Image 2.1 + captions + zip. Train on this one.",
                             "- Realism pass (Z-Image) is **off** here: Qwen 2.1 already looks real. To try it: select the *5 · Realism pass* group + *Save FINAL* and press Ctrl+B."] if q21 else
                            ["- `<name>_raw` = straight from the edit model",
                             "- `<name>` = after the **Z-Image realism pass** (real skin + light, same face). Train on this one.",
                             "- Don't want the realism pass? Right-click the *5 · Realism pass* group → Bypass group nodes."]),
                          "", "**Settings**", *[f"- {x}" for x in cfg["notes"]],
                          (f"- Edit: {cfg['steps']} steps, CFG 1, euler / simple (official Qwen-Image 2.1 settings)." if q21 else
                           f"- Edit: {cfg['steps']} steps, CFG 1, euler / simple (Lightning LoRA). Max quality: Ctrl+B the LoRA and use 40 steps, CFG 4."),
                          *([] if q21 else ["- Realism pass: denoise 0.30 (higher = more realistic but can change the face; 0.25-0.35 is the sweet spot)."]),
                          "", "**Models**",
                          *[f"- {m[2]}: [{m[0]}]({m[1]})" for m in ((cfg["unet"], cfg["clip"], cfg["vae"]) if q21 else (cfg["unet"], cfg["lora"], M["clip"], M["vae"])) + (M["z_unet"], M["z_clip"], M["z_vae"])]])
        add(18, "MarkdownNote", [740, 460], [400, 700], [note], [], title="READ ME")
        # models
        add(2, "UNETLoader", [0, 0], [340, 82], [cfg["unet"][0], "default"], [("MODEL", "MODEL")], title="Edit model", props=mp(cfg["unet"]))
        if q21:
            add(5, "QwenImage21Cache", [0, 120], [340, 82], ["auto", "default"], [("MODEL", "MODEL")])
            add(6, "CLIPLoader", [0, 440], [340, 106], [cfg["clip"][0], "qwen_image", "default"], [("CLIP", "CLIP")], props=mp(cfg["clip"]))
            add(7, "VAELoader", [0, 580], [340, 58], [cfg["vae"][0]], [("VAE", "VAE")], props=mp(cfg["vae"]))
            add(10, "TextEncodeQwenImage21", [1180, 0], [360, 220], ["", "", 1536],
                [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")], title="Qwen 2.1 encoder (prompt from presets)")
        else:
            add(3, "LoraLoaderModelOnly", [0, 120], [340, 82], [cfg["lora"][0], 1.0], [("MODEL", "MODEL")], title="Lightning 8-step LoRA", props=mp(cfg["lora"]))
            add(4, "ModelSamplingAuraFlow", [0, 240], [340, 58], [3.1], [("MODEL", "MODEL")])
            add(5, "CFGNorm", [0, 340], [340, 58], [1], [("MODEL", "MODEL")])
            add(6, "CLIPLoader", [0, 440], [340, 106], [M["clip"][0], "qwen_image", "default"], [("CLIP", "CLIP")], props=mp(M["clip"]))
            add(7, "VAELoader", [0, 580], [340, 58], [M["vae"][0]], [("VAE", "VAE")], props=mp(M["vae"]))
            add(8, "FluxKontextImageScale", [380, 820], [260, 30], [], [("IMAGE", "IMAGE")])
            # generate
            add(10, "TextEncodeQwenImageEditPlus", [1180, 0], [360, 160], [""], [("CONDITIONING", "CONDITIONING")], title="Prompt (from presets)")
            add(11, "TextEncodeQwenImageEditPlus", [1180, 220], [360, 160], [""], [("CONDITIONING", "CONDITIONING")], title="Negative (leave empty)")
        if cfg.get("ref_method"):
            add(12, "FluxKontextMultiReferenceLatentMethod", [1580, 0], [300, 58], ["index_timestep_zero"], [("CONDITIONING", "CONDITIONING")])
            add(13, "FluxKontextMultiReferenceLatentMethod", [1580, 220], [300, 58], ["index_timestep_zero"], [("CONDITIONING", "CONDITIONING")])
        add(14, "EmptyLatentImage" if q21 else "EmptySD3LatentImage", [1580, 420], [300, 106], [1024, 1024, 1], [("LATENT", "LATENT")])
        add(15, "KSampler", [1920, 0], [320, 262], [42, "fixed", cfg["steps"], 1, "euler", "simple", 1], [("LATENT", "LATENT")], title="KSampler (edit)")
        add(16, "VAEDecode", [1920, 320], [220, 46], [], [("IMAGE", "IMAGE")])
        RM = 4 if q21 else 0  # Qwen-Image 2.1 looks real on its own: realism pass off (Ctrl+B the group to turn it on)
        # realism pass
        add(20, "UNETLoader", [1180, 760], [340, 82], [M["z_unet"][0], "default"], [("MODEL", "MODEL")], title="Z-Image Turbo", props=mp(M["z_unet"]), mode=RM)
        add(21, "ModelSamplingAuraFlow", [1180, 880], [340, 58], [3], [("MODEL", "MODEL")], mode=RM)
        add(22, "CLIPLoader", [1180, 980], [340, 106], [M["z_clip"][0], "lumina2", "default"], [("CLIP", "CLIP")], props=mp(M["z_clip"]), mode=RM)
        add(23, "VAELoader", [1180, 1120], [340, 58], [M["z_vae"][0]], [("VAE", "VAE")], props=mp(M["z_vae"]), mode=RM)
        add(24, "CLIPTextEncode", [1580, 760], [320, 120], [""], [("CONDITIONING", "CONDITIONING")], title="Realism prompt (from presets)", mode=RM)
        add(25, "ConditioningZeroOut", [1580, 920], [240, 26], [], [("CONDITIONING", "CONDITIONING")], mode=RM)
        add(26, "VAEEncode", [1580, 1000], [240, 46], [], [("LATENT", "LATENT")], mode=RM)
        add(27, "KSampler", [1920, 760], [320, 262], [42, "fixed", 8, 1, "dpmpp_2m_sde", "beta", 0.3], [("LATENT", "LATENT")], title="KSampler (realism, denoise 0.30)", mode=RM)
        add(28, "VAEDecode", [1920, 1080], [220, 46], [], [("IMAGE", "IMAGE")], mode=RM)
        # save
        if q21:
            add(17, "AIEmpireSaveDataset", [2300, 0], [520, 620], ["my_influencer", True, ""], [], title="Save Dataset ← train on this", color=GREEN)
        else:
            add(17, "AIEmpireSaveDataset", [2300, 0], [520, 620], ["my_influencer", False, "_raw"], [], title="Save RAW (edit model only)", color=GREEN)
        add(29, "AIEmpireSaveDataset", [2300, 760], [520, 620], ["my_influencer", True, ""], [], title="Save FINAL (after realism pass) ← train on this", color=GREEN, mode=RM)

        if q21:
            link(2, 0, 5, "model", "MODEL")
            link(6, 0, 10, "clip", "CLIP")
            link(7, 0, 10, "vae", "VAE", optional=True)
            link(1, 0, 10, "images.image_1", "IMAGE", optional=True)
            link(19, 0, 10, "images.image_2", "IMAGE", optional=True)
            socket(10, "images.image_3", "IMAGE")
        else:
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
        if q21:
            link(10, 0, 15, "positive", "CONDITIONING")
            link(10, 1, 15, "negative", "CONDITIONING")
        elif cfg["ref_method"]:
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

    write(key, nodes, links, lid, groups, node)


def write(key, nodes, links, lid, groups, node):
    # execution-order hint
    for i, n in enumerate(sorted(nodes, key=lambda n: (n["pos"][0], n["pos"][1]))):
        n["order"] = i

    for l in links:
        assert l[0] in node(l[1])["outputs"][l[2]]["links"], l
        assert node(l[3])["inputs"][l[4]]["link"] == l[0], l

    ids = {"FireRed11": "5e0c2f64-7a1b-4c8e-9d3f-f1e3d11a0001", "Qwen2511": "5e0c2f64-7a1b-4c8e-9d3f-a1e3e4e5e6e7", "Qwen21": "5e0c2f64-7a1b-4c8e-9d3f-21e21e210001",
           "NanoBanana": "5e0c2f64-7a1b-4c8e-9d3f-0a0ba0a0a0b1", "Qwen21_Templates": "5e0c2f64-7a1b-4c8e-9d3f-7e3a1a7e0002",
           "Qwen21_BodyPresets": "5e0c2f64-7a1b-4c8e-9d3f-b0d1b0d10003"}
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


def build_body(key, cfg, nodes, links, lid, add, node, link, mp, ORANGE, GREEN):
    note = "\n".join([
        f"## AI Empire · Body Preset Maker ({cfg['title'].split(' · ')[0]})", "",
        "Turns your base photos into a **body-type preset**: every photo gets the same body, everything else (face, hair, pose, outfit, place, light) stays.",
        "The result is a new template set, ready for the Template Presets dataset workflow.", "",
        "**How to use**",
        "1. On **Body Preset Maker** click **📁 Upload template photos**, select your base photos, name the set (e.g. `base`).",
        "2. *output_set* = the new preset name (`athletic`). *body_target* = the body every photo gets.",
        "3. *instructions* (optional), one line per photo:",
        "   - `1-35 | keep` → copied unchanged (already the right body)",
        "   - `swap_40 | her hips are very wide, make them much narrower` → extra help for one photo",
        "   - numbers = position in the set (sorted by name), or use the file name",
        "4. **how_many = 3** first. Check them, then **0** = all. Finished photos are skipped on re-runs; delete a bad one from `input/templates/<preset>/` and run again to redo it.",
        "5. **Body example (recommended):** select the purple box, Ctrl+B, load ONE photo that already has the target body (best: one of your own `keep` photos, so the edited ones match them), tick *use_body_example*. Qwen copies the figure from it, nothing else.",
        "6. Next preset (curvy, skinny …): same *source_set*, new *output_set* + *body_target* (+ a new body example), clear *instructions*.", "",
        "Captions: a `.txt` next to a base photo is copied into the preset too.", "",
        "**Settings**: Qwen-Image 2.1, 25 steps, CFG 1, euler / simple. Output keeps each photo's size and shape (*resolution* 1536 ≈ 2.3 MP).", "",
        "**Models**",
        *[f"- {m[2]}: [{m[0]}]({m[1]})" for m in (cfg["unet"], cfg["clip"], cfg["vae"])]])
    add(18, "MarkdownNote", [380, 560], [440, 640], [note], [], title="READ ME")
    add(50, "AIEmpireBodyPresetMaker", [380, 0], [440, 500],
        ["my_templates", "athletic",
         "a slim athletic body: flat toned stomach with visible abs, slim waist, narrow hips, slim toned arms and shoulders, slim toned legs, small to medium bust, lean natural proportions",
         "# one line per photo (optional)\n# 1-35 | keep\n# swap_40 | her hips are wide, make them narrower\n", 3, 1, 42, "fixed", True, False],
        [("images", "IMAGE", True), ("prompts", "STRING", True), ("seeds", "INT", True), ("file_names", "STRING", True),
         ("output_set", "STRING"), ("source_set", "STRING")],
        title="Body Preset Maker (upload base photos here)", color=ORANGE)
    add(19, "LoadImage", [860, 300], [320, 400], ["body_example.png", "image"], [("IMAGE", "IMAGE"), ("MASK", "MASK")],
        title="Body example (optional, Ctrl+B to turn on)", color=("#2e2240", "#1f172b"), mode=4)
    add(2, "UNETLoader", [0, 0], [340, 82], [cfg["unet"][0], "default"], [("MODEL", "MODEL")], title="Edit model", props=mp(cfg["unet"]))
    add(5, "QwenImage21Cache", [0, 120], [340, 82], ["auto", "default"], [("MODEL", "MODEL")])
    add(6, "CLIPLoader", [0, 440], [340, 106], [cfg["clip"][0], "qwen_image", "default"], [("CLIP", "CLIP")], props=mp(cfg["clip"]))
    add(7, "VAELoader", [0, 580], [340, 58], [cfg["vae"][0]], [("VAE", "VAE")], props=mp(cfg["vae"]))
    add(10, "TextEncodeQwenImage21", [860, 0], [360, 220], ["", "", 1536],
        [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")], title="Qwen 2.1 encoder (1 = base photo, 2 = body example)")
    add(15, "KSampler", [1260, 0], [320, 262], [42, "fixed", cfg["steps"], 1, "euler", "simple", 1], [("LATENT", "LATENT")], title="KSampler (edit)")
    add(16, "VAEDecode", [1260, 320], [220, 46], [], [("IMAGE", "IMAGE")])
    add(51, "AIEmpireSaveTemplateSet", [1620, 0], [520, 620], ["athletic", "my_templates"], [], title="Save preset → input/templates/<output_set>", color=GREEN)
    link(2, 0, 5, "model", "MODEL")
    link(6, 0, 10, "clip", "CLIP")
    link(7, 0, 10, "vae", "VAE", optional=True)
    link(50, 0, 10, "images.image_1", "IMAGE", optional=True)
    link(19, 0, 10, "images.image_2", "IMAGE", optional=True)
    link(50, 1, 10, "prompt", "STRING", widget=True)
    link(5, 0, 15, "model", "MODEL")
    link(10, 0, 15, "positive", "CONDITIONING")
    link(10, 1, 15, "negative", "CONDITIONING")
    link(10, 2, 15, "latent_image", "LATENT")
    link(50, 2, 15, "seed", "INT", widget=True)
    link(15, 0, 16, "samples", "LATENT")
    link(7, 0, 16, "vae", "VAE")
    link(16, 0, 51, "images", "IMAGE")
    link(50, 3, 51, "file_names", "STRING")
    link(50, 4, 51, "output_set", "STRING", widget=True)
    link(50, 5, 51, "source_set", "STRING", widget=True)
    groups = [
        {"id": 1, "title": "1 · Models", "bounding": [-20, -60, 380, 720], "color": "#444", "flags": {}},
        {"id": 2, "title": "2 · Base photos + body", "bounding": [370, -60, 470, 1280], "color": "#b06634", "flags": {}},
        {"id": 3, "title": "3 · Edit model", "bounding": [850, -60, 750, 780], "color": "#3f789e", "flags": {}},
        {"id": 4, "title": "4 · Save preset", "bounding": [1610, -60, 540, 700], "color": "#6a8f4e", "flags": {}},
    ]
    write(key, nodes, links, lid, groups, node)


for k, c in CONFIGS.items():
    build(k, c)
