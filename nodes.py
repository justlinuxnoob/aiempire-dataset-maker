"""AI Empire Dataset Maker nodes.

Two nodes:
  * AIEmpireDatasetPresets - turns a preset list (or your own lines) into
    lists of prompts, captions, sizes and seeds. Every node after it runs
    once per preset automatically.
  * AIEmpireSaveDataset - saves every image with a matching .txt caption
    (name_001.png + name_001.txt ...) and zips the folder for download.
"""

import json
import os
import re
import zipfile

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths

PRESET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "presets")

SIZES = {
    "square": (1024, 1024),   # close-ups
    "portrait": (896, 1152),  # half body
    "tall": (832, 1216),      # full body
}

PROMPT_TEMPLATE = (
    "Create a new photo of the exact same woman from picture 1{extra}. "
    "Keep her identity identical: same face shape, same eyes and eye colour, "
    "same eyebrows, same nose, same lips, same skin tone, same hair colour and hairstyle. "
    "{shot}. "
    "Photorealistic smartphone photo, natural skin texture, realistic lighting, sharp focus on her face."
)


def _preset_files():
    if not os.path.isdir(PRESET_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith(".json"))


def _load_preset(name):
    with open(os.path.join(PRESET_DIR, name + ".json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["presets"]


def _parse_custom(text):
    """One preset per line:  shot description | caption | square/portrait/tall
    Caption and size are optional. Lines starting with # are ignored."""
    presets = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        shot = parts[0]
        caption = parts[1] if len(parts) > 1 and parts[1] else shot
        size = parts[2].lower() if len(parts) > 2 and parts[2].lower() in SIZES else "portrait"
        presets.append({"shot": shot, "caption": caption, "size": size})
    return presets


class AIEmpireDatasetPresets:
    @classmethod
    def INPUT_TYPES(cls):
        files = _preset_files() or ["core40"]
        return {
            "required": {
                "trigger_word": ("STRING", {"default": "ohwx woman", "tooltip": "Unique word that goes at the start of every caption. Use the same one when training."}),
                "preset_set": (files, {"default": "core40" if "core40" in files else files[0]}),
                "extra_description": ("STRING", {"default": "", "multiline": False, "tooltip": "Things the model must keep, e.g. 'with long blonde hair and green eyes, curvy body'. Not written into captions."}),
                "how_many": ("INT", {"default": 40, "min": 1, "max": 500, "tooltip": "Run only the first N presets (use 3 for a quick test)."}),
                "start_at": ("INT", {"default": 1, "min": 1, "max": 500}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            },
            "optional": {
                "custom_presets": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional. One per line: shot | caption | square/portrait/tall. If filled, these are used instead of the preset set."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("prompts", "captions", "widths", "heights", "seeds", "count")
    OUTPUT_IS_LIST = (True, True, True, True, True, False)
    FUNCTION = "build"
    CATEGORY = "AI Empire"

    def build(self, trigger_word, preset_set, extra_description, how_many, start_at, seed, custom_presets=""):
        presets = _parse_custom(custom_presets) if custom_presets and custom_presets.strip() else _load_preset(preset_set)
        chosen = presets[start_at - 1:start_at - 1 + how_many]
        if not chosen:
            raise ValueError("No presets selected - check 'start_at' and 'how_many'.")

        extra = extra_description.strip().rstrip(".")
        extra = (" " + extra) if extra and not extra.startswith(",") else (extra if extra else "")
        trigger = trigger_word.strip()

        prompts, captions, widths, heights, seeds = [], [], [], [], []
        for i, p in enumerate(chosen):
            shot = p["shot"].strip().rstrip(".")
            prompts.append(PROMPT_TEMPLATE.format(extra=extra, shot=shot[0].upper() + shot[1:]))
            cap = p.get("caption", shot).strip().rstrip(".")
            captions.append(f"{trigger}, {cap}" if trigger else cap)
            w, h = SIZES.get(p.get("size", "portrait"), SIZES["portrait"])
            widths.append(w)
            heights.append(h)
            seeds.append((seed + i) % 0xFFFFFFFFFFFFFFFF)
        return (prompts, captions, widths, heights, seeds, len(chosen))


def _safe_name(name):
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()) or "dataset"
    return name[:64]


class AIEmpireSaveDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "captions": ("STRING", {"forceInput": True}),
                "dataset_name": ("STRING", {"default": "my_influencer"}),
                "make_zip": ("BOOLEAN", {"default": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "AI Empire"

    def save(self, images, captions, dataset_name, make_zip):
        name = _safe_name(dataset_name[0])
        do_zip = bool(make_zip[0])
        out_root = folder_paths.get_output_directory()
        subfolder = f"datasets/{name}"
        folder = os.path.join(out_root, "datasets", name)
        os.makedirs(folder, exist_ok=True)

        # continue numbering if the folder already has images
        existing = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
        nums = [int(m.group(1)) for f in existing if (m := re.search(r"_(\d{3,})\.png$", f))]
        counter = max(nums) + 1 if nums else 1

        ui_images = []
        n = 0
        for idx, batch in enumerate(images):
            caption = captions[idx] if idx < len(captions) else captions[-1]
            for img in batch:
                arr = np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8)
                base = f"{name}_{counter:03d}"
                meta = PngInfo()
                meta.add_text("ai_generated", "true")
                meta.add_text("caption", caption)
                Image.fromarray(arr).save(os.path.join(folder, base + ".png"), pnginfo=meta, compress_level=4)
                with open(os.path.join(folder, base + ".txt"), "w", encoding="utf-8") as f:
                    f.write(caption)
                ui_images.append({"filename": base + ".png", "subfolder": subfolder, "type": "output"})
                counter += 1
                n += 1

        if do_zip:
            zip_path = os.path.join(out_root, "datasets", name + ".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(os.listdir(folder)):
                    if f.lower().endswith((".png", ".txt")):
                        z.write(os.path.join(folder, f), arcname=os.path.join(name, f))

        print(f"[AI Empire] Saved {n} images + captions to {folder}")
        return {"ui": {"images": ui_images}}


NODE_CLASS_MAPPINGS = {
    "AIEmpireDatasetPresets": AIEmpireDatasetPresets,
    "AIEmpireSaveDataset": AIEmpireSaveDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIEmpireDatasetPresets": "AI Empire · Dataset Presets",
    "AIEmpireSaveDataset": "AI Empire · Save Dataset (images + captions)",
}
