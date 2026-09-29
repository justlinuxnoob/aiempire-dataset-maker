"""AI Empire Dataset Maker nodes.

  * AIEmpireDatasetPresets - turns a preset list (or your own lines) into lists
    of prompts, captions, sizes and seeds. Nodes after it run once per preset.
  * AIEmpireNanoBanana     - makes the dataset with Google Nano Banana Pro /
    Nano Banana 2, using YOUR OWN Google key (AI Studio key or Vertex AI).
  * AIEmpireSaveDataset    - saves every image with a matching .txt caption
    (name_001.png + name_001.txt ...) and zips the folder for download.
"""

import io
import json
import os
import re
import time
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

EDIT_PROMPT = (
    "Create a new photo of the exact same woman from picture 1{extra}. "
    "Keep her identity identical: same face shape, same eyes and eye colour, "
    "same eyebrows, same nose, same lips, same skin tone, same hair colour and hairstyle. "
    "{body}"
    "{shot}. "
    "Photorealistic smartphone photo, natural skin texture, realistic lighting, sharp focus on her face."
)
BODY_SENTENCE = "Her body shape, figure and proportions are exactly the same as the woman in picture 2. "

REALISM_PROMPT = (
    "Candid photorealistic smartphone photo of a young woman{extra}, {caption}. "
    "Natural skin texture with visible pores, realistic lighting, true-to-life colours, sharp focus."
)


def _preset_files():
    if not os.path.isdir(PRESET_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith(".json"))


def _load_preset(name):
    with open(os.path.join(PRESET_DIR, name + ".json"), "r", encoding="utf-8") as f:
        return json.load(f)["presets"]


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


def _first(v):
    return v[0] if isinstance(v, list) else v


class AIEmpireDatasetPresets:
    @classmethod
    def INPUT_TYPES(cls):
        files = _preset_files() or ["core40"]
        return {
            "required": {
                "trigger_word": ("STRING", {"default": "zvx woman", "tooltip": "Unique word that starts every caption. Use the same one when training."}),
                "preset_set": (files, {"default": "core40" if "core40" in files else files[0]}),
                "extra_description": ("STRING", {"default": "", "multiline": False, "tooltip": "What must stay the same, e.g. 'with long straight dark brown hair, brown eyes'. Not written into captions."}),
                "how_many": ("INT", {"default": 3, "min": 1, "max": 500, "tooltip": "Run only the first N presets (3 for a quick test, 40 for the full set)."}),
                "start_at": ("INT", {"default": 1, "min": 1, "max": 500}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            },
            "optional": {
                "custom_presets": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional. One per line: shot | caption | square/portrait/tall. If filled, used instead of the preset set."}),
                "use_body_reference": ("BOOLEAN", {"default": False, "tooltip": "Turn on when you also load a full-body photo of her (Body reference box)."}),
                "dataset_name": ("STRING", {"default": "my_influencer", "tooltip": "Folder / zip name for the dataset."}),
                "size_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": "Multiplies every image size (1.0 = ~1 MP, 1.5 = ~2.3 MP for 2K models like Qwen-Image 2.1)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("prompts", "captions", "widths", "heights", "seeds", "count", "realism_prompts", "dataset_name")
    OUTPUT_IS_LIST = (True, True, True, True, True, False, True, False)
    FUNCTION = "build"
    CATEGORY = "AI Empire"

    def build(self, trigger_word, preset_set, extra_description, how_many, start_at, seed,
              custom_presets="", use_body_reference=False, dataset_name="my_influencer", size_scale=1.0):
        presets = _parse_custom(custom_presets) if custom_presets and custom_presets.strip() else _load_preset(preset_set)
        chosen = presets[start_at - 1:start_at - 1 + how_many]
        if not chosen:
            raise ValueError("No presets selected - check 'start_at' and 'how_many'.")

        extra = extra_description.strip().rstrip(".")
        if extra and not extra.startswith(","):
            extra = " " + extra
        trigger = trigger_word.strip()
        body = BODY_SENTENCE if use_body_reference else ""

        prompts, captions, widths, heights, seeds, realism = [], [], [], [], [], []
        for i, p in enumerate(chosen):
            shot = p["shot"].strip().rstrip(".")
            prompts.append(EDIT_PROMPT.format(extra=extra, body=body, shot=shot[0].upper() + shot[1:]))
            cap = p.get("caption", shot).strip().rstrip(".")
            captions.append(f"{trigger}, {cap}" if trigger else cap)
            realism.append(REALISM_PROMPT.format(extra=extra, caption=cap))
            w, h = SIZES.get(p.get("size", "portrait"), SIZES["portrait"])
            if size_scale != 1.0:
                w, h = int(round(w * size_scale / 32)) * 32, int(round(h * size_scale / 32)) * 32
            widths.append(w)
            heights.append(h)
            seeds.append((seed + i) % 0xFFFFFFFFFFFFFFFF)
        return (prompts, captions, widths, heights, seeds, len(chosen), realism, dataset_name.strip() or "my_influencer")


# ----------------------------------------------------------------- Nano Banana

ASPECTS = {"1:1": 1.0, "4:5": 0.8, "3:4": 0.75, "2:3": 2 / 3, "9:16": 9 / 16, "5:4": 1.25, "4:3": 4 / 3, "3:2": 1.5, "16:9": 16 / 9}


def _aspect(w, h):
    r = w / h
    return min(ASPECTS, key=lambda k: abs(ASPECTS[k] - r))


def _tensor_to_pil(t):
    arr = t[0] if len(t.shape) == 4 else t
    arr = np.clip(255.0 * arr.cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr[..., :3])


def _pil_to_tensor(img):
    import torch
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _make_client(auth, api_key, project, location):
    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError("google-genai is not installed. Run: pip install google-genai") from e

    if auth == "Vertex AI":
        sa_json = os.environ.get("VERTEX_SA_JSON", "").strip()
        if sa_json:
            path = "/tmp/aiempire_vertex_sa.json"
            with open(path, "w", encoding="utf-8") as f:
                f.write(sa_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
            if not project:
                try:
                    project = json.loads(sa_json).get("project_id", "")
                except Exception:
                    pass
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        if not project:
            raise RuntimeError("Vertex AI needs a project ID: fill 'vertex_project' or set GOOGLE_CLOUD_PROJECT / VERTEX_SA_JSON on the pod.")
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise RuntimeError("Vertex AI needs your service-account key: set VERTEX_SA_JSON (paste the whole JSON) as a RunPod env/secret.")
        return genai.Client(vertexai=True, project=project, location=location)

    key = api_key.strip() or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("No Google API key: paste it into 'api_key' or set GEMINI_API_KEY on the pod.")
    return genai.Client(api_key=key)


class AIEmpireNanoBanana:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face": ("IMAGE",),
                "prompts": ("STRING", {"forceInput": True}),
                "captions": ("STRING", {"forceInput": True}),
                "widths": ("INT", {"forceInput": True}),
                "heights": ("INT", {"forceInput": True}),
                "auth": (["AI Studio API key", "Vertex AI"], {"default": "Vertex AI"}),
                "model": ("STRING", {"default": "gemini-3-pro-image", "tooltip": "Nano Banana Pro = gemini-3-pro-image · Nano Banana 2 = gemini-3.1-flash-image"}),
                "resolution": (["1K", "2K", "4K"], {"default": "2K"}),
                "api_key": ("STRING", {"default": "", "tooltip": "AI Studio key. Leave empty to use GEMINI_API_KEY from the pod settings (safer: the key isn't saved in the workflow)."}),
                "vertex_project": ("STRING", {"default": "", "tooltip": "Leave empty to read it from VERTEX_SA_JSON / GOOGLE_CLOUD_PROJECT."}),
                "vertex_location": ("STRING", {"default": "global"}),
            },
            "optional": {
                "body": ("IMAGE",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "captions")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "run"
    CATEGORY = "AI Empire"

    def run(self, face, prompts, captions, widths, heights, auth, model, resolution,
            api_key, vertex_project, vertex_location, body=None):
        from google.genai import types
        import comfy.model_management as mm
        import comfy.utils

        auth, model, resolution = _first(auth), _first(model).strip(), _first(resolution)
        client = _make_client(auth, _first(api_key), _first(vertex_project).strip(), _first(vertex_location).strip())
        face_img = _tensor_to_pil(face[0])
        body_img = _tensor_to_pil(body[0]) if body else None

        out_imgs, out_caps, errors = [], [], []
        pbar = comfy.utils.ProgressBar(len(prompts))
        models_to_try = [model] + ([model + "-preview"] if not model.endswith("-preview") else [])

        for i, prompt in enumerate(prompts):
            mm.throw_exception_if_processing_interrupted()
            w = widths[i] if i < len(widths) else widths[-1]
            h = heights[i] if i < len(heights) else heights[-1]
            contents = ["Picture 1 (the woman to keep):", face_img]
            if body_img is not None:
                contents += ["Picture 2 (her body reference):", body_img]
            contents.append(prompt)
            cfg = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=_aspect(w, h), image_size=resolution),
            )
            img, last_err = None, ""
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(model=models_to_try[0], contents=contents, config=cfg)
                    for cand in (resp.candidates or []):
                        for part in (cand.content.parts if cand.content and cand.content.parts else []):
                            if getattr(part, "inline_data", None) and part.inline_data.data:
                                img = Image.open(io.BytesIO(part.inline_data.data))
                                break
                        if img is not None:
                            break
                    if img is not None:
                        break
                    reason = ""
                    if resp.candidates:
                        reason = str(getattr(resp.candidates[0], "finish_reason", ""))
                    last_err = f"no image returned (blocked/refused? {reason})"
                    break  # refusals don't get better by retrying
                except Exception as e:  # network / quota / wrong model name
                    last_err = str(e)
                    if ("NOT_FOUND" in last_err or "404" in last_err) and len(models_to_try) > 1:
                        models_to_try.pop(0)
                        continue
                    time.sleep(3 * (attempt + 1))
            if img is not None:
                out_imgs.append(_pil_to_tensor(img))
                out_caps.append(captions[i] if i < len(captions) else captions[-1])
                print(f"[AI Empire] Nano Banana {i + 1}/{len(prompts)} ✔")
            else:
                errors.append(f"#{i + 1}: {last_err[:200]}")
                print(f"[AI Empire] Nano Banana {i + 1}/{len(prompts)} skipped - {last_err[:200]}")
            pbar.update(1)

        if not out_imgs:
            raise RuntimeError("Nano Banana made no images. First errors:\n" + "\n".join(errors[:3]))
        if errors:
            print(f"[AI Empire] {len(errors)} shot(s) skipped:\n" + "\n".join(errors))
        return (out_imgs, out_caps)


# ----------------------------------------------------------------- saving

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
            },
            "optional": {
                "suffix": ("STRING", {"default": "", "tooltip": "Added to the folder name, e.g. _raw"}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "AI Empire"

    def save(self, images, captions, dataset_name, make_zip, suffix=None):
        sfx = (_first(suffix) if suffix else "") or ""
        name = _safe_name(_first(dataset_name) + sfx)
        do_zip = bool(_first(make_zip))
        out_root = folder_paths.get_output_directory()
        subfolder = f"datasets/{name}"
        folder = os.path.join(out_root, "datasets", name)
        os.makedirs(folder, exist_ok=True)

        existing = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
        nums = [int(m.group(1)) for f in existing if (m := re.search(r"_(\d{3,})\.png$", f))]
        counter = max(nums) + 1 if nums else 1

        ui_images, n = [], 0
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
    "AIEmpireNanoBanana": AIEmpireNanoBanana,
    "AIEmpireSaveDataset": AIEmpireSaveDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIEmpireDatasetPresets": "AI Empire · Dataset Presets",
    "AIEmpireNanoBanana": "AI Empire · Nano Banana (your Google key)",
    "AIEmpireSaveDataset": "AI Empire · Save Dataset (images + captions)",
}
