"""AI Empire Dataset Maker nodes.

  * AIEmpireDatasetPresets - turns a preset list (or your own lines) into lists
    of prompts, captions, sizes and seeds. Nodes after it run once per preset.
  * AIEmpireTemplatePresets - same idea, but every photo in input/templates/<set>/
    is one shot: her face goes into that photo's pose, outfit, place and light.
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


# ----------------------------------------------------------------- template photos

TEMPLATE_ROOT = "templates"  # inside ComfyUI's input folder: input/templates/<set>/
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Qwen-Image 2.1 labels its reference images <image1>, <image2>, ... in order.
# image 1 = the template (the output keeps its size and composition), image 2 = her face, image 3 = body (optional)
# {extra} = the node's extra_description. Editable in the node ('prompt').
TEMPLATE_PROMPT = (
    "Image 1 is the photo to edit. Image 2 only shows who the woman is.\n"
    "Turn the woman in image 1 into the woman from image 2{extra}: same face shape, eyes and eye colour, eyebrows, nose, lips, "
    "skin tone, hair colour and hairstyle.\n"
    "Do not paste or copy the face photo from image 2. Redraw her face naturally inside image 1: her head angle, face direction, "
    "gaze, facial expression and the light and shadows on her face all follow image 1.\n"
    "Keep everything else from image 1 exactly the same: the pose, the hands, the outfit, the background, the camera angle, "
    "the framing and the lighting.\n"
    "Photorealistic smartphone photo, natural skin texture with visible pores."
)
TEMPLATE_KEEP_BODY = " Keep her body shape from image 1."
TEMPLATE_BODY_REF = " Her body shape, figure and proportions are exactly the same as the woman in image 3."


def _templates_dir():
    return os.path.join(folder_paths.get_input_directory(), TEMPLATE_ROOT)


def _template_sets():
    root = _templates_dir()
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith("."))


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _unzip_uploads(folder):
    """A .zip uploaded into the set folder is unpacked once (images only, no sub-folders)."""
    for z in [f for f in os.listdir(folder) if f.lower().endswith(".zip")]:
        zpath = os.path.join(folder, z)
        try:
            with zipfile.ZipFile(zpath) as zf:
                for member in zf.namelist():
                    base = os.path.basename(member)
                    if not base or base.startswith(".") or "__MACOSX" in member:
                        continue
                    if not base.lower().endswith(IMAGE_EXTS + (".txt",)):
                        continue
                    with zf.open(member) as src, open(os.path.join(folder, base), "wb") as dst:
                        dst.write(src.read())
            os.remove(zpath)
            print(f"[AI Empire] unpacked {z}")
        except zipfile.BadZipFile:
            print(f"[AI Empire] {z} is not a valid zip - skipped")


def _load_template(path, max_side=2048):
    from PIL import ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        k = max_side / max(w, h)
        img = img.resize((int(w * k), int(h * k)), Image.LANCZOS)
    return img


class AIEmpireTemplatePresets:
    """Every photo in input/templates/<set>/ becomes one dataset image:
    same pose, outfit, place and light as the photo, her face from 'Your face'."""

    @classmethod
    def INPUT_TYPES(cls):
        sets = _template_sets() or ["upload templates first"]
        return {
            "required": {
                "template_set": (sets, {"tooltip": "Folder in input/templates/. Use the Upload button on this box to add photos."}),
                "trigger_word": ("STRING", {"default": "zvx woman", "tooltip": "Starts every caption. Use the same one when training."}),
                "extra_description": ("STRING", {"default": "", "tooltip": "What must stay the same, e.g. 'with long straight blonde hair, blue eyes'. Not written into captions."}),
                "how_many": ("INT", {"default": 3, "min": 0, "max": 1000, "tooltip": "3 for a quick test. 0 = every photo in the set."}),
                "start_at": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            },
            "optional": {
                "use_body_reference": ("BOOLEAN", {"default": False, "tooltip": "Turn on when you also load a full-body photo of her (Body reference box). Off = body comes from each template."}),
                "dataset_name": ("STRING", {"default": "my_influencer"}),
                "skip_done": ("BOOLEAN", {"default": True, "tooltip": "Skip templates that already have a saved image in this dataset (resume after a crash)."}),
                "one_per_run": ("BOOLEAN", {"default": True, "tooltip": "ON = each Run makes ONE image (the next one not done yet) and saves it right away. Set the Run count next to the Run button to how many you want. OFF = all at once, saved only at the very end."}),
                "prompt": ("STRING", {"default": TEMPLATE_PROMPT, "multiline": True, "tooltip": "The instruction Qwen gets for every photo. {extra} = extra_description."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("templates", "prompts", "captions", "seeds", "count", "dataset_name", "file_names")
    OUTPUT_IS_LIST = (True, True, True, True, False, False, True)
    FUNCTION = "build"
    CATEGORY = "AI Empire"

    @classmethod
    def IS_CHANGED(cls, template_set, dataset_name="my_influencer", **kw):
        # re-read when templates are added/changed, or when finished images appear (so skip_done works on re-runs)
        state = []
        for folder in (os.path.join(_templates_dir(), template_set),
                       os.path.join(folder_paths.get_output_directory(), "datasets", _safe_name(dataset_name))):
            if os.path.isdir(folder):
                state += [f"{f}:{os.path.getmtime(os.path.join(folder, f))}" for f in sorted(os.listdir(folder))]
        return "|".join(state)

    def build(self, template_set, trigger_word, extra_description, how_many, start_at, seed,
              use_body_reference=False, dataset_name="my_influencer", skip_done=True, one_per_run=True, prompt=TEMPLATE_PROMPT):
        folder = os.path.join(_templates_dir(), template_set)
        if not os.path.isdir(folder):
            raise ValueError("No template photos yet: click 'Upload template photos' on the Template Presets box.")
        _unzip_uploads(folder)
        files = sorted((f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS) and not f.startswith(".")), key=_natural_key)
        if not files:
            raise ValueError(f"input/templates/{template_set} has no images (.png .jpg .jpeg .webp).")

        chosen = files[start_at - 1:] if how_many == 0 else files[start_at - 1:start_at - 1 + how_many]
        if not chosen:
            raise ValueError(f"'start_at' is past the end: this set has {len(files)} photos.")

        name = _safe_name(dataset_name)
        out_folder = os.path.join(folder_paths.get_output_directory(), "datasets", name)
        extra = extra_description.strip().rstrip(".")
        if extra and not extra.startswith(","):
            extra = " " + extra
        base = (prompt or TEMPLATE_PROMPT).strip()
        base = base.replace("{extra}", extra) if "{extra}" in base else base + (f" She is{extra}." if extra else "")
        prompt = base + (TEMPLATE_BODY_REF if use_body_reference else TEMPLATE_KEEP_BODY)
        trigger = trigger_word.strip()

        imgs, prompts, captions, seeds, stems, skipped = [], [], [], [], [], 0
        for f in chosen:
            stem = _safe_name(os.path.splitext(f)[0]).strip("_") or "img"
            if skip_done and os.path.exists(os.path.join(out_folder, f"{name}_{stem}.png")):
                skipped += 1
                continue
            img = _load_template(os.path.join(folder, f))
            imgs.append(_pil_to_tensor(img))
            prompts.append(prompt)
            cap_file = os.path.join(folder, os.path.splitext(f)[0] + ".txt")
            cap = ""
            if os.path.exists(cap_file):
                with open(cap_file, "r", encoding="utf-8") as cf:
                    cap = cf.read().strip()
            captions.append(", ".join(x for x in (trigger, cap) if x) or trigger)
            seeds.append((seed + files.index(f)) % 0xFFFFFFFFFFFFFFFF)
            stems.append(stem)
            if one_per_run:
                break

        left = sum(1 for f in chosen if not os.path.exists(os.path.join(out_folder, f"{name}_{_safe_name(os.path.splitext(f)[0]).strip('_') or 'img'}.png")))
        if one_per_run and imgs:
            print(f"[AI Empire] Templates '{template_set}': making {stems[0]} ({left} of {len(chosen)} still to do)")
        else:
            print(f"[AI Empire] Templates '{template_set}': {len(imgs)} to make, {skipped} already done")
        if not imgs:
            raise ValueError(f"✅ All done: every template in this range is already in datasets/{name}. "
                             "To redo one, delete its image there. To redo all, change 'dataset_name'.")
        return (imgs, prompts, captions, seeds, len(imgs), name, stems)


# ----------------------------------------------------------------- body presets

ATHLETIC = ("a slim athletic body: flat toned stomach with visible abs, slim waist, narrow hips, "
            "slim toned arms and shoulders, slim toned legs, small to medium bust, lean natural proportions")

BODY_PROMPT = (
    "Change only her body shape in image 1 to {target}.{extra} "
    "Keep her face, hair, skin tone, pose, hands, outfit, background, camera angle, framing and lighting exactly the same. "
    "The clothes fit her new body naturally. Photorealistic smartphone photo, natural skin texture."
)
BODY_PROMPT_REF = (
    "Change only her body shape in image 1 so her figure and proportions match the woman in image 2: {target}.{extra} "
    "Keep her face, hair, skin tone, pose, hands, outfit, background, camera angle, framing and lighting from image 1 exactly the same. "
    "Take nothing else from image 2. The clothes fit her new body naturally. Photorealistic smartphone photo, natural skin texture."
)


def _parse_instructions(text, files):
    """Lines like  'swap_01 | keep',  '1-35 | keep',  '7 | her hips are wide, make them narrower'.
    Left side: file name (with or without extension), a number, or a range (numbers = position in the set)."""
    by_file = {}
    stems = {os.path.splitext(f)[0].lower(): f for f in files}
    names = {f.lower(): f for f in files}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        key, what = (x.strip() for x in line.split("|", 1))
        k = key.lower()
        targets = []
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", k)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            targets = files[max(a, 1) - 1:b]
        elif k.isdigit():
            targets = files[int(k) - 1:int(k)] if int(k) >= 1 else []
        elif k in names:
            targets = [names[k]]
        elif k in stems:
            targets = [stems[k]]
        else:
            print(f"[AI Empire] body instructions: no file matches '{key}' - line ignored")
        for f in targets:
            by_file[f] = what
    return by_file


class AIEmpireBodyPresetMaker:
    """Makes a body-type preset: every photo in a template set gets the same body type,
    everything else stays. 'keep' photos are copied unchanged. Output = a new template set."""

    @classmethod
    def INPUT_TYPES(cls):
        sets = _template_sets() or ["upload templates first"]
        return {
            "required": {
                "source_set": (sets, {"tooltip": "Your base photos (input/templates/<set>). Use the Upload button to add them."}),
                "output_set": ("STRING", {"default": "athletic", "tooltip": "Name of the new preset folder, e.g. athletic, curvy, skinny."}),
                "body_target": ("STRING", {"default": ATHLETIC, "multiline": True, "tooltip": "The body type every photo gets."}),
                "instructions": ("STRING", {"default": "# one line per photo (optional)\n# 1-35 | keep\n# swap_40 | her hips are wide, make them narrower\n", "multiline": True,
                                            "tooltip": "keep = copy unchanged. Anything else is added to the edit for that photo. Photos not listed get the normal edit."}),
                "how_many": ("INT", {"default": 3, "min": 0, "max": 1000, "tooltip": "3 for a quick test. 0 = every photo."}),
                "start_at": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            },
            "optional": {
                "skip_done": ("BOOLEAN", {"default": True, "tooltip": "Skip photos already in the output set (resume after a crash). Turn off to redo them."}),
                "use_body_example": ("BOOLEAN", {"default": False, "tooltip": "Turn on when you load a photo with the target body in the Body example box (best: one of your own photos that already has it)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "prompts", "seeds", "file_names", "output_set", "source_set")
    OUTPUT_IS_LIST = (True, True, True, True, False, False)
    FUNCTION = "build"
    CATEGORY = "AI Empire"

    @classmethod
    def IS_CHANGED(cls, source_set, output_set="athletic", **kw):
        state = []
        for folder in (os.path.join(_templates_dir(), source_set), os.path.join(_templates_dir(), _safe_name(output_set))):
            if os.path.isdir(folder):
                state += [f"{f}:{os.path.getmtime(os.path.join(folder, f))}" for f in sorted(os.listdir(folder))]
        return "|".join(state)

    def build(self, source_set, output_set, body_target, instructions, how_many, start_at, seed, skip_done=True, use_body_example=False):
        import shutil
        src = os.path.join(_templates_dir(), source_set)
        if not os.path.isdir(src):
            raise ValueError("No base photos yet: click 'Upload template photos' on this box.")
        out_name = _safe_name(output_set)
        if out_name == source_set:
            raise ValueError("output_set must be a different folder than source_set.")
        dst = os.path.join(_templates_dir(), out_name)
        os.makedirs(dst, exist_ok=True)
        _unzip_uploads(src)
        files = sorted((f for f in os.listdir(src) if f.lower().endswith(IMAGE_EXTS) and not f.startswith(".")), key=_natural_key)
        if not files:
            raise ValueError(f"input/templates/{source_set} has no images.")
        chosen = files[start_at - 1:] if how_many == 0 else files[start_at - 1:start_at - 1 + how_many]
        if not chosen:
            raise ValueError(f"'start_at' is past the end: this set has {len(files)} photos.")

        per_file = _parse_instructions(instructions, files)
        done_stems = {os.path.splitext(f)[0] for f in os.listdir(dst) if f.lower().endswith(IMAGE_EXTS)}
        target = body_target.strip().rstrip(".")

        imgs, prompts, seeds, stems, kept, skipped = [], [], [], [], 0, 0
        for f in chosen:
            stem = os.path.splitext(f)[0]
            if skip_done and stem in done_stems:
                skipped += 1
                continue
            what = per_file.get(f, "").strip()
            cap = os.path.join(src, stem + ".txt")
            if what.lower() == "keep":
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                if os.path.exists(cap):
                    shutil.copy2(cap, os.path.join(dst, stem + ".txt"))
                kept += 1
                continue
            extra = f" {what[0].upper()}{what[1:].rstrip('.')}." if what else ""
            imgs.append(_pil_to_tensor(_load_template(os.path.join(src, f))))
            prompts.append((BODY_PROMPT_REF if use_body_example else BODY_PROMPT).format(target=target, extra=extra))
            seeds.append((seed + files.index(f)) % 0xFFFFFFFFFFFFFFFF)
            stems.append(stem)

        print(f"[AI Empire] Body preset '{out_name}': {len(imgs)} to edit, {kept} kept as they are, {skipped} already done")
        if not imgs:
            raise ValueError(f"Nothing left to edit: {kept} photos copied unchanged, {skipped} already done. "
                             f"Preset is in input/templates/{out_name}.")
        return (imgs, prompts, seeds, stems, out_name, source_set)


class AIEmpireSaveTemplateSet:
    """Saves edited photos into input/templates/<output_set>/ with their original names
    (+ copies the caption .txt), so the result shows up as a new preset in Template Presets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "file_names": ("STRING", {"forceInput": True}),
                "output_set": ("STRING", {"default": "athletic"}),
                "source_set": ("STRING", {"default": ""}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "AI Empire"

    def save(self, images, file_names, output_set, source_set):
        import shutil
        out_name = _safe_name(_first(output_set))
        src = os.path.join(_templates_dir(), os.path.basename(_first(source_set) or ""))
        dst = os.path.join(_templates_dir(), out_name)
        os.makedirs(dst, exist_ok=True)
        ui, n = [], 0
        for idx, batch in enumerate(images):
            stem = os.path.basename(file_names[idx] if idx < len(file_names) else f"img_{idx + 1:03d}")
            for b, img in enumerate(batch):
                base = stem + (f"_{b + 1}" if b else "")
                for old in os.listdir(dst):  # replace an older version with another extension
                    if os.path.splitext(old)[0] == base and old.lower().endswith(IMAGE_EXTS):
                        os.remove(os.path.join(dst, old))
                arr = np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(dst, base + ".png"), compress_level=4)
                ui.append({"filename": base + ".png", "subfolder": f"{TEMPLATE_ROOT}/{out_name}", "type": "input"})
                n += 1
            cap = os.path.join(src, stem + ".txt")
            if os.path.exists(cap):
                shutil.copy2(cap, os.path.join(dst, stem + ".txt"))
        print(f"[AI Empire] Saved {n} photos to input/templates/{out_name}")
        return {"ui": {"images": ui}}


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
                "file_names": ("STRING", {"forceInput": True, "tooltip": "From Template Presets: each image is named after its template, so re-runs skip finished ones."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "AI Empire"

    def save(self, images, captions, dataset_name, make_zip, suffix=None, file_names=None):
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
            stem = _safe_name(file_names[idx]).strip("_") if file_names and idx < len(file_names) else None
            for b, img in enumerate(batch):
                arr = np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8)
                if stem:
                    base = f"{name}_{stem}" + (f"_{b + 1}" if b else "")
                else:
                    base = f"{name}_{counter:03d}"
                    counter += 1
                meta = PngInfo()
                meta.add_text("ai_generated", "true")
                meta.add_text("caption", caption)
                Image.fromarray(arr).save(os.path.join(folder, base + ".png"), pnginfo=meta, compress_level=4)
                with open(os.path.join(folder, base + ".txt"), "w", encoding="utf-8") as f:
                    f.write(caption)
                ui_images.append({"filename": base + ".png", "subfolder": subfolder, "type": "output"})
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
    "AIEmpireTemplatePresets": AIEmpireTemplatePresets,
    "AIEmpireBodyPresetMaker": AIEmpireBodyPresetMaker,
    "AIEmpireSaveTemplateSet": AIEmpireSaveTemplateSet,
    "AIEmpireNanoBanana": AIEmpireNanoBanana,
    "AIEmpireSaveDataset": AIEmpireSaveDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIEmpireDatasetPresets": "AI Empire · Dataset Presets",
    "AIEmpireTemplatePresets": "AI Empire · Template Presets (your photos)",
    "AIEmpireBodyPresetMaker": "AI Empire · Body Preset Maker",
    "AIEmpireSaveTemplateSet": "AI Empire · Save Template Set (new preset)",
    "AIEmpireNanoBanana": "AI Empire · Nano Banana (your Google key)",
    "AIEmpireSaveDataset": "AI Empire · Save Dataset (images + captions)",
}
