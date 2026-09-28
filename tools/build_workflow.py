"""Builds workflows/AI_Empire_Dataset_Maker_Qwen2511.json (drag-into-ComfyUI format)
and workflows/api/AI_Empire_Dataset_Maker_Qwen2511_api.json (API format, for serverless later).
Run:  python tools/build_workflow.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "unet": ("qwen_image_edit_2511_fp8mixed.safetensors", "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors", "diffusion_models"),
    "lora": ("Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "loras"),
    "clip": ("qwen_2.5_vl_7b_fp8_scaled.safetensors", "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders"),
    "vae": ("qwen_image_vae.safetensors", "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors", "vae"),
}

NOTE = """## AI Empire · Dataset Maker (Qwen Image Edit 2511)

**How to use**
1. Load your influencer's face in **Your face** (clear, front-facing, even light).
2. In **Dataset Presets**: set the trigger word, and write what must stay the same in *extra_description* (e.g. `with long blonde hair and green eyes, curvy body`).
3. First run: set **how_many = 3** to check the face. If it looks like her, set it back to 40.
4. Press **Run**. Images + matching .txt captions are saved to `output/datasets/<name>/`, plus a `.zip` next to it.

**Settings that matter**
- 4 steps · CFG 1 · euler / simple (Lightning LoRA). Without the LoRA: 20-40 steps, CFG 4.
- Default model file is **fp8mixed** (20 GB, fits RTX 4090/5090). On a 48 GB+ GPU you can use the **bf16** file (41 GB) for best quality.
- Using the bf16 file on a 24-32 GB GPU? Set UNet weight_dtype = fp8_e4m3fn.

**Models (download into ComfyUI/models/...)**
- diffusion_models: [qwen_image_edit_2511_fp8mixed](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors) (or the bf16 file from the same folder)
- loras: [Qwen-Image-Edit-2511-Lightning-4steps bf16](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors) (fp32 version works too)
- text_encoders: [qwen_2.5_vl_7b_fp8_scaled](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors)
- vae: [qwen_image_vae](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors)
"""

nodes = []
links = []
_link_id = [0]


def model_prop(key):
    name, url, d = MODELS[key]
    return [{"name": name, "url": url, "directory": d}]


def add(nid, ntype, pos, size, widgets, outputs, title=None, props=None, color=None):
    n = {
        "id": nid, "type": ntype, "pos": pos, "size": size, "flags": {}, "order": 0, "mode": 0,
        "inputs": [], "outputs": [{"name": o[0], "type": o[1], "links": [], **({"shape": 6} if len(o) > 2 and o[2] else {})} for o in outputs],
        "properties": {"Node name for S&R": ntype, **(props or {})},
        "widgets_values": widgets,
    }
    if title:
        n["title"] = title
    if color:
        n["color"], n["bgcolor"] = color
    nodes.append(n)
    return n


def node(nid):
    return next(n for n in nodes if n["id"] == nid)


def link(src, src_slot, dst, input_name, typ, widget=False):
    _link_id[0] += 1
    lid = _link_id[0]
    s, d = node(src), node(dst)
    s["outputs"][src_slot]["links"].append(lid)
    entry = {"name": input_name, "type": typ, "link": lid}
    if widget:
        entry["widget"] = {"name": input_name}
    d["inputs"].append(entry)
    links.append([lid, src, src_slot, dst, len(d["inputs"]) - 1, typ])
    return lid


def slot(nid, name, typ, optional=False):
    """declare an unconnected socket input (so it shows as a dot)"""
    e = {"name": name, "type": typ, "link": None}
    if optional:
        e["shape"] = 7
    node(nid)["inputs"].append(e)


ORANGE = ("#3d2a14", "#2a1d0e")
GREEN = ("#1f3320", "#162416")

# ---- loaders (left column)
add(1, "LoadImage", [0, 0], [320, 360], ["your_face.png", "image"], [("IMAGE", "IMAGE"), ("MASK", "MASK")], title="Your face", color=GREEN)
add(2, "UNETLoader", [0, 420], [320, 82], [MODELS["unet"][0], "default"], [("MODEL", "MODEL")], props={"models": model_prop("unet")})
add(3, "LoraLoaderModelOnly", [0, 540], [320, 82], [MODELS["lora"][0], 1.0], [("MODEL", "MODEL")], title="Lightning 4-step LoRA", props={"models": model_prop("lora")})
add(4, "ModelSamplingAuraFlow", [0, 660], [320, 58], [3.1], [("MODEL", "MODEL")])
add(5, "CFGNorm", [0, 760], [320, 58], [1], [("MODEL", "MODEL")])
add(6, "CLIPLoader", [0, 860], [320, 106], [MODELS["clip"][0], "qwen_image", "default"], [("CLIP", "CLIP")], props={"models": model_prop("clip")})
add(7, "VAELoader", [0, 1000], [320, 58], [MODELS["vae"][0]], [("VAE", "VAE")], props={"models": model_prop("vae")})

# ---- presets (middle)
add(9, "AIEmpireDatasetPresets", [380, 0], [400, 330],
    ["ohwx woman", "core40", "", 40, 1, 42, "fixed", ""],
    [("prompts", "STRING", True), ("captions", "STRING", True), ("widths", "INT", True), ("heights", "INT", True), ("seeds", "INT", True), ("count", "INT")],
    title="Dataset Presets", color=ORANGE)
add(8, "FluxKontextImageScale", [380, 380], [260, 30], [], [("IMAGE", "IMAGE")])

# ---- encoding
add(10, "TextEncodeQwenImageEditPlus", [820, 0], [360, 160], [""], [("CONDITIONING", "CONDITIONING")], title="Prompt (from presets)")
add(11, "TextEncodeQwenImageEditPlus", [820, 220], [360, 160], [""], [("CONDITIONING", "CONDITIONING")], title="Negative (leave empty)")
add(12, "FluxKontextMultiReferenceLatentMethod", [1220, 0], [300, 58], ["index_timestep_zero"], [("CONDITIONING", "CONDITIONING")])
add(13, "FluxKontextMultiReferenceLatentMethod", [1220, 220], [300, 58], ["index_timestep_zero"], [("CONDITIONING", "CONDITIONING")])
add(14, "EmptySD3LatentImage", [1220, 420], [300, 106], [1024, 1024, 1], [("LATENT", "LATENT")])

# ---- sampling + save
add(15, "KSampler", [1560, 0], [320, 262], [42, "fixed", 4, 1, "euler", "simple", 1], [("LATENT", "LATENT")])
add(16, "VAEDecode", [1560, 320], [220, 46], [], [("IMAGE", "IMAGE")])
add(17, "AIEmpireSaveDataset", [1920, 0], [520, 620], ["my_influencer", True], [], title="Save Dataset (images + captions)", color=GREEN)
add(18, "MarkdownNote", [380, 480], [400, 560], [NOTE], [], title="READ ME")

# ---- wiring
link(2, 0, 3, "model", "MODEL")
link(3, 0, 4, "model", "MODEL")
link(4, 0, 5, "model", "MODEL")
link(1, 0, 8, "image", "IMAGE")

link(6, 0, 10, "clip", "CLIP")
link(7, 0, 10, "vae", "VAE")
link(8, 0, 10, "image1", "IMAGE")
slot(10, "image2", "IMAGE", True)
slot(10, "image3", "IMAGE", True)
link(9, 0, 10, "prompt", "STRING", widget=True)

link(6, 0, 11, "clip", "CLIP")
link(7, 0, 11, "vae", "VAE")
link(8, 0, 11, "image1", "IMAGE")
slot(11, "image2", "IMAGE", True)
slot(11, "image3", "IMAGE", True)

link(10, 0, 12, "conditioning", "CONDITIONING")
link(11, 0, 13, "conditioning", "CONDITIONING")

link(9, 2, 14, "width", "INT", widget=True)
link(9, 3, 14, "height", "INT", widget=True)

link(5, 0, 15, "model", "MODEL")
link(12, 0, 15, "positive", "CONDITIONING")
link(13, 0, 15, "negative", "CONDITIONING")
link(14, 0, 15, "latent_image", "LATENT")
link(9, 4, 15, "seed", "INT", widget=True)

link(15, 0, 16, "samples", "LATENT")
link(7, 0, 16, "vae", "VAE")

link(16, 0, 17, "images", "IMAGE")
link(9, 1, 17, "captions", "STRING")

# execution order hint for the UI
order = [1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
for i, nid in enumerate(order):
    node(nid)["order"] = i

workflow = {
    "id": "5e0c2f64-7a1b-4c8e-9d3f-a1e3e4e5e6e7",
    "revision": 0,
    "last_node_id": max(n["id"] for n in nodes),
    "last_link_id": _link_id[0],
    "nodes": nodes,
    "links": links,
    "groups": [
        {"id": 1, "title": "1 · Models", "bounding": [-20, -60, 360, 1140], "color": "#444", "flags": {}},
        {"id": 2, "title": "2 · Your face + presets", "bounding": [360, -60, 440, 1120], "color": "#b06634", "flags": {}},
        {"id": 3, "title": "3 · Generate", "bounding": [800, -60, 1100, 620], "color": "#3f789e", "flags": {}},
        {"id": 4, "title": "4 · Save dataset", "bounding": [1900, -60, 560, 700], "color": "#6a8f4e", "flags": {}},
    ],
    "config": {},
    "extra": {"ds": {"scale": 0.6, "offset": [80, 120]}},
    "version": 0.4,
}

# ---- API format (for serverless / validation)
api = {}
by_id = {n["id"]: n for n in nodes}
WIDGET_NAMES = {
    "LoadImage": ["image", "upload"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
    "ModelSamplingAuraFlow": ["shift"],
    "CFGNorm": ["strength"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "AIEmpireDatasetPresets": ["trigger_word", "preset_set", "extra_description", "how_many", "start_at", "seed", None, "custom_presets"],
    "FluxKontextImageScale": [],
    "TextEncodeQwenImageEditPlus": ["prompt"],
    "FluxKontextMultiReferenceLatentMethod": ["reference_latents_method"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "VAEDecode": [],
    "AIEmpireSaveDataset": ["dataset_name", "make_zip"],
}
for n in nodes:
    if n["type"] == "MarkdownNote":
        continue
    inputs = {}
    for wname, val in zip(WIDGET_NAMES[n["type"]], n["widgets_values"]):
        if wname and wname != "upload":
            inputs[wname] = val
    for inp in n["inputs"]:
        if inp["link"] is None:
            continue
        l = next(x for x in links if x[0] == inp["link"])
        inputs[inp["name"]] = [str(l[1]), l[2]]
    api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs, "_meta": {"title": n.get("title", n["type"])}}

os.makedirs(os.path.join(ROOT, "workflows", "api"), exist_ok=True)
with open(os.path.join(ROOT, "workflows", "AI_Empire_Dataset_Maker_Qwen2511.json"), "w", encoding="utf-8") as f:
    json.dump(workflow, f, indent=2, ensure_ascii=False)
with open(os.path.join(ROOT, "workflows", "api", "AI_Empire_Dataset_Maker_Qwen2511_api.json"), "w", encoding="utf-8") as f:
    json.dump(api, f, indent=2, ensure_ascii=False)
print("ok", len(nodes), "nodes", len(links), "links")
