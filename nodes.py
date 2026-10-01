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


# body-type presets that ship inside the pod (presets/templates/<type>/image_XX.jpg + .txt), in this order
BUILTIN_TEMPLATES = os.path.join(PRESET_DIR, "templates")
PRESET_ORDER = ["athletic", "curvy", "petite", "busty", "thick", "plus"]


def _templates_dir():
    return os.path.join(folder_paths.get_input_directory(), TEMPLATE_ROOT)


def _subdirs(root):
    if not os.path.isdir(root):
        return []
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")]


def _template_sets():
    """Built-in body presets first (in PRESET_ORDER), then any sets uploaded to input/templates/."""
    names = set(_subdirs(BUILTIN_TEMPLATES)) | set(_subdirs(_templates_dir()))
    first = [n for n in PRESET_ORDER if n in names]
    return first + sorted(names - set(first))


def _set_folder(name):
    """input/templates/<name> if the user uploaded one with that name, else the built-in preset."""
    if name not in _template_sets():
        raise ValueError(f"Unknown body type / template set: {name}")
    user = os.path.join(_templates_dir(), name)
    return user if os.path.isdir(user) else os.path.join(BUILTIN_TEMPLATES, name)


def _set_files(name):
    folder = _set_folder(name)
    return sorted((f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS) and not f.startswith(".")), key=_natural_key)


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
                "skip_done": ("BOOLEAN", {"default": True, "tooltip": "Skip templates that already have a saved image in this dataset (resume after a crash). OFF = redo the first photo of the range, one per Run (no auto-continue). To redo one image, better: delete it and keep this ON."}),
                "one_per_run": ("BOOLEAN", {"default": True, "tooltip": "ON = each Run makes ONE image (the next one not done yet) and saves it right away. Set the Run count next to the Run button to how many you want. OFF = all at once, saved only at the very end."}),
                "prompt": ("STRING", {"default": TEMPLATE_PROMPT, "multiline": True, "tooltip": "The instruction Qwen gets for every photo. {extra} = extra_description."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("templates", "prompts", "captions", "seeds", "count", "dataset_name", "file_names", "remaining", "progress")
    OUTPUT_IS_LIST = (True, True, True, True, False, False, True, False, False)
    FUNCTION = "build"
    CATEGORY = "AI Empire"

    @classmethod
    def IS_CHANGED(cls, template_set, dataset_name="my_influencer", **kw):
        # re-read when templates are added/changed, or when finished images appear (so skip_done works on re-runs)
        state = []
        try:
            set_folder = _set_folder(template_set)
        except ValueError:
            set_folder = ""
        for folder in (set_folder,
                       os.path.join(folder_paths.get_output_directory(), "datasets", _safe_name(dataset_name))):
            if os.path.isdir(folder):
                state += [f"{f}:{os.path.getmtime(os.path.join(folder, f))}" for f in sorted(os.listdir(folder))]
        return "|".join(state)

    def build(self, template_set, trigger_word, extra_description, how_many, start_at, seed,
              use_body_reference=False, dataset_name="my_influencer", skip_done=True, one_per_run=True, prompt=TEMPLATE_PROMPT,
              skip_stems=None):
        try:
            folder = _set_folder(template_set)
        except ValueError:
            raise ValueError("No template photos yet: click 'Upload template photos' on the Template Presets box.")
        if folder.startswith(_templates_dir()):
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
            if skip_stems and stem in skip_stems:
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
            raise ValueError(f"✅ All done: every photo is already in output/datasets/{name} (zip: datasets/{name}.zip). "
                             "To redo one, delete it there and Run. For a new dataset, change 'dataset_name'.")
        # still to do after this run. With skip_done OFF the next run would pick this same photo again,
        # so auto-continue would loop forever: then it's one image per Run, no auto-continue.
        remaining = max(left - len(imgs), 0) if (one_per_run and skip_done) else 0
        progress = f"{len(chosen) - left + len(imgs)}/{len(chosen)}"  # done after this run / total
        return (imgs, prompts, captions, seeds, len(imgs), name, stems, remaining, progress)


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


def _queue_again(prompt, extra_pnginfo):
    """Puts the same workflow back in the queue (like pressing Run again)."""
    import threading
    import urllib.request

    def go():
        try:
            import server
            inst = server.PromptServer.instance
            port = getattr(inst, "port", None) or 8188
            # ComfyUI stamps "is_changed" into the running prompt; re-sending that stamp would make
            # the next run a 100% cache hit (nothing happens), so send a clean copy
            clean = {nid: {k: v for k, v in node.items() if k != "is_changed"} for nid, node in prompt.items()}
            body = {"prompt": clean, "extra_data": {"extra_pnginfo": extra_pnginfo} if extra_pnginfo else {}}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/prompt", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:
            print(f"[AI Empire] could not queue the next one automatically ({e}). Press Run again.")

    threading.Timer(0.5, go).start()


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
                "remaining": ("INT", {"forceInput": True, "tooltip": "From Template Presets: how many are still to do."}),
                "progress": ("STRING", {"forceInput": True, "tooltip": "From Template Presets: done / total, shown on this box."}),
                "auto_continue": ("BOOLEAN", {"default": True, "tooltip": "ON = after saving, it queues the next one by itself until all are done. Press Run once. Cancel (X) stops it."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "AI Empire"

    def save(self, images, captions, dataset_name, make_zip, suffix=None, file_names=None,
             remaining=None, auto_continue=None, prompt=None, extra_pnginfo=None, progress=None):
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
        # show the WHOLE dataset in the box (newest first), not just this run's image
        pngs = sorted((f for f in os.listdir(folder) if f.lower().endswith(".png")),
                      key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
        gallery = [{"filename": f, "subfolder": subfolder, "type": "output"} for f in pngs[:100]]
        left = _first(remaining) if remaining else 0
        prog = _first(progress) if progress else ""
        if prog:
            try:  # live progress on the Save box + browser tab (js/aiempire_templates.js)
                import server
                server.PromptServer.instance.send_sync("aiempire.progress", {"progress": prog, "left": int(left or 0), "dataset": name})
            except Exception:
                pass
        if left and _first(auto_continue) is not False and prompt and _first(prompt):
            _queue_again(_first(prompt), _first(extra_pnginfo) if extra_pnginfo else None)
            print(f"[AI Empire] {left} still to do - queued the next one (press X / Cancel to stop)")
        elif remaining:
            print("[AI Empire] ✅ All templates in this range are done.")
        return {"ui": {"images": gallery or ui_images, "progress": [prog] if prog else []}}


# ----------------------------------------------------------------- the simple student workflow
#
#   Your face ─┐
#   Dataset Maker (body type, test / whole dataset, engine) ─► [Qwen machine room] ─► Make image ─► Save Dataset
#
# "Make image" takes the Qwen result, or (engine = Nano Banana Pro / Seedream) calls the RunPod API instead
# and Qwen never runs (lazy input), so those runs don't touch the GPU.

# the prompt BFS Head Swap v1.1 was trained with (verbatim) + {extra} = hair / eyes
QWEN_SWAP_PROMPT = ("head_swap: start with <image1> as the base image, keeping its lighting, environment, and background. "
                    "remove the head from <image1> completely and replace it with the head from <image2>{extra}, strictly preserving "
                    "the hair, eye color, nose structure from <image2>. copy the direction of the eye, head rotation, micro expressions "
                    "from <image1>, high quality, sharp details, 4k")

# Nano Banana Pro / Seedream: image 1 = the preset photo (the one that gets edited), image 2 = her face
API_SWAP_PROMPT = (
    "Output image 1, edited. Image 1 is the photo to edit. Image 2 only shows who the woman is.\n"
    "Replace the woman's face and hair in image 1 with the woman from image 2{extra}: same face shape, eyes and eye colour, "
    "eyebrows, nose, lips, skin tone, hair colour and hairstyle.\n"
    "Do not paste the face photo from image 2. Redraw her face naturally inside image 1: her head angle, gaze, facial "
    "expression and the light and shadows on her face follow image 1.\n"
    "Keep everything else in image 1 exactly the same: her body shape, pose, hands, outfit, background, camera angle, "
    "framing and lighting. Never output the woman from image 2's photo.\n"
    "Photorealistic smartphone photo, natural skin texture."
)

MODE_TEST = "🧪 Test 1 photo"
MODE_ALL = "🚀 Whole dataset (all photos)"
ENGINE_QWEN = "Qwen · free, runs on this pod"
ENGINE_NANO = "Nano Banana Pro · RunPod API · ~$0.14/photo"
ENGINE_SEED = "Seedream · RunPod API · ~$0.03/photo"
RUNPOD_MODELS = {
    ENGINE_NANO: ("nano", "https://api.runpod.ai/v2/nano-banana-pro-edit"),
    ENGINE_SEED: ("seedream", "https://api.runpod.ai/v2/seedream-v4-edit"),
}
NANO_RATIOS = ["1:1", "3:2", "2:3", "4:3", "3:4", "4:5", "5:4", "9:16", "16:9", "21:9"]


def _key_file():
    # /workspace/.runpod_key on the pod (next to input/ and output/, survives restarts on a network volume)
    return os.path.join(os.path.dirname(os.path.abspath(folder_paths.get_input_directory())), ".runpod_key")


def _runpod_key():
    """The key the user saved with the 🔑 button, else env AIEMPIRE_RUNPOD_KEY.
    NOT env RUNPOD_API_KEY: RunPod puts its own pod-scoped key there in every pod, and that one can't call the public endpoints."""
    try:
        with open(_key_file(), encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    except OSError:
        pass
    return os.environ.get("AIEMPIRE_RUNPOD_KEY", "").strip()


def _failed_path(folder, stem):
    return os.path.join(folder, f".{stem}.failed")


def _failed_count(folder, stem, engine):
    """How often RunPod failed this photo with this engine (switching engine starts fresh)."""
    try:
        with open(_failed_path(folder, stem), encoding="utf-8") as f:
            d = json.load(f)
        return int(d.get("count", 0)) if d.get("engine") == engine else 0
    except (OSError, ValueError):
        return 0


def _hair_eyes(text):
    t = (text or "").strip().rstrip(".")
    if not t:
        return ""
    return t if t.lower().startswith(("with ", ",")) else "with " + t


def _jpeg_data_url(img, max_side=2048):
    import base64
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        k = max_side / max(w, h)
        img = img.resize((round(w * k), round(h * k)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _runpod_http(url, key, body=None, timeout=60):
    import urllib.error
    import urllib.request
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:300]
        if e.code in (401, 403):
            raise RuntimeError("RunPod says the API key is wrong. Click '🔑 RunPod key' on the Dataset Maker box and paste it again.")
        raise RuntimeError(f"RunPod HTTP {e.code}: {msg}")
    except (urllib.error.URLError, OSError) as e:  # network hiccup / timeout: retried like any other failure
        raise RuntimeError(f"can't reach RunPod ({getattr(e, 'reason', e)})")


def _runpod_edit(engine, template, face, prompt, job_timeout=240, tries=3):
    """One edit through a RunPod public endpoint. Returns a PIL image. A job stuck > job_timeout is cancelled and retried."""
    import urllib.request
    import comfy.model_management as mm
    kind, base = RUNPOD_MODELS[engine]
    base = os.environ.get("AIEMPIRE_RUNPOD_BASE", "") or base  # tests only
    key = _runpod_key()
    if not key:
        raise RuntimeError("No RunPod API key yet: click '🔑 RunPod key' on the Dataset Maker box and paste your key "
                           "(runpod.io → Settings → API Keys). Or switch engine back to Qwen.")
    w, h = template.size
    images = [_jpeg_data_url(template), _jpeg_data_url(face, 1536)]
    if kind == "seedream":
        k = 2048 / max(w, h)
        payload = {"prompt": prompt, "images": images,
                   "size": f"{max(16, round(w * k / 16) * 16)}*{max(16, round(h * k / 16) * 16)}"}
    else:
        import math
        ratio = min(NANO_RATIOS, key=lambda r: abs(math.log(int(r.split(':')[0]) / int(r.split(':')[1])) - math.log(w / h)))
        payload = {"prompt": prompt, "images": images, "resolution": "2k", "aspect_ratio": ratio, "output_format": "png"}

    last = ""
    for attempt in range(1, tries + 1):
        job = None
        try:
            res = _runpod_http(f"{base}/run", key, {"input": payload})
            job, start = res.get("id"), time.time()
            while res.get("status") in ("IN_QUEUE", "IN_PROGRESS") or (job and not res.get("status")):
                try:
                    mm.throw_exception_if_processing_interrupted()
                except BaseException:
                    _runpod_http(f"{base}/cancel/{job}", key, {}, timeout=20)
                    raise
                if time.time() - start > job_timeout:
                    _runpod_http(f"{base}/cancel/{job}", key, {}, timeout=20)
                    raise RuntimeError(f"stuck in RunPod's queue for {job_timeout // 60} min")
                time.sleep(3)
                res = _runpod_http(f"{base}/status/{job}", key, timeout=30)
            if res.get("status") != "COMPLETED":
                raise RuntimeError(f"{res.get('status')}: {res.get('error') or str(res.get('output'))[:200]}")
            out = res.get("output") or {}
            url = out.get("image_url") or out.get("result")
            if not url:
                raise RuntimeError(f"no image in RunPod's answer: {str(out)[:200]}")
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    img = Image.open(io.BytesIO(r.read())).convert("RGB")
            except OSError as e:
                raise RuntimeError(f"couldn't download the result ({e})")
            print(f"[AI Empire] {kind} ✔ (${float(out.get('cost') or 0):.2f})")
            return img
        except RuntimeError as e:
            last = str(e)
            if "API key is wrong" in last:
                raise
            print(f"[AI Empire] {kind} try {attempt}/{tries} failed: {last}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"{kind} failed 3 times ({last})")


class AIEmpireDatasetMaker:
    """The one box students fill in: body type, test or whole dataset, engine."""

    @classmethod
    def INPUT_TYPES(cls):
        sets = _template_sets() or ["no presets found"]
        return {
            "required": {
                "body_type": (sets, {"default": sets[0], "tooltip": "Which preset to use. The photos are shown below this box."}),
                "mode": ([MODE_TEST, MODE_ALL], {"default": MODE_TEST,
                         "tooltip": "Test = makes 1 photo (the one picked in test_photo) so you can check quality. "
                                    "Whole dataset = makes every photo of the body type, one after another, and zips them."}),
                "test_photo": ("INT", {"default": 1, "min": 1, "max": 999, "tooltip": "Which photo to test with (see the preview below)."}),
                "trigger_word": ("STRING", {"default": "zvx woman", "tooltip": "Starts every caption. Use the same word when you train the LoRA."}),
                "hair_and_eyes": ("STRING", {"default": "", "tooltip": "Her hair and eye colour, e.g. 'long straight blonde hair, blue eyes'. Empty = taken from your face photo."}),
                "engine": ([ENGINE_QWEN, ENGINE_NANO, ENGINE_SEED], {"default": ENGINE_QWEN,
                           "tooltip": "Qwen = free, runs on this pod's GPU. Nano Banana Pro / Seedream = paid per photo with your RunPod API key (click 🔑 RunPod key)."}),
                "dataset_name": ("STRING", {"default": "my_influencer", "tooltip": "Folder + zip name in output/datasets/. New name = new dataset."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "STRING", "STRING", "INT", "STRING", "AIEMPIRE_JOB")
    RETURN_NAMES = ("templates", "prompts", "captions", "seeds", "dataset_name", "file_names", "remaining", "progress", "job")
    OUTPUT_IS_LIST = (True, True, True, True, False, True, False, False, False)
    FUNCTION = "build"
    CATEGORY = "AI Empire"

    @classmethod
    def IS_CHANGED(cls, body_type, mode, dataset_name="my_influencer", **kw):
        if mode == MODE_TEST:
            return float("nan")  # every test press makes a fresh photo
        return AIEmpireTemplatePresets.IS_CHANGED(body_type, dataset_name=dataset_name)

    def build(self, body_type, mode, test_photo, trigger_word, hair_and_eyes, engine, dataset_name):
        files = _set_files(body_type)
        if not files:
            raise ValueError(f"The '{body_type}' preset has no photos.")
        test = mode == MODE_TEST
        name = _safe_name(dataset_name) + ("_test" if test else "")
        if test and test_photo > len(files):
            raise ValueError(f"'{body_type}' has {len(files)} photos: set test_photo between 1 and {len(files)}.")
        seed = int(time.time() * 1000) % 1_000_000 if test else 42
        skip = set()
        retry_note = ""
        if not test:
            # photos RunPod refused earlier with this engine: skipped on the first pass, retried once at the end
            out_folder = os.path.join(folder_paths.get_output_directory(), "datasets", name)
            stems_all = [_safe_name(os.path.splitext(f)[0]).strip("_") or "img" for f in files]
            pending = [st for st in stems_all if not os.path.exists(os.path.join(out_folder, f"{name}_{st}.png"))]
            failed = {st: _failed_count(out_folder, st, engine) for st in pending}
            given_up = [st for st in pending if failed[st] >= 2]
            fresh = [st for st in pending if failed[st] == 0]
            if not fresh:
                retry = [st for st in pending if failed[st] == 1]
                if not retry:
                    if given_up:
                        raise ValueError(f"✅ Done, except {len(given_up)} photo(s) RunPod refused twice: {', '.join(given_up)}. "
                                         "Switch engine (e.g. Qwen) and Run to make those, or train without them.")
                else:
                    retry_note = f" (retrying {len(retry)} photo(s) that failed earlier)"
                skip = set(given_up)
            else:
                skip = {st for st in pending if failed[st] >= 1}
        out = AIEmpireTemplatePresets().build(
            body_type, trigger_word, _hair_eyes(hair_and_eyes),
            how_many=1 if test else 0, start_at=test_photo if test else 1, seed=seed,
            use_body_reference=False, dataset_name=name, skip_done=not test, one_per_run=True,
            prompt=QWEN_SWAP_PROMPT, skip_stems=skip)
        imgs, prompts, captions, seeds, _count, name, stems, remaining, progress = out
        if test:
            tag = {ENGINE_QWEN: "qwen", ENGINE_NANO: "nanobanana", ENGINE_SEED: "seedream"}.get(engine, "test")
            stems = [f"{st}_{tag}" for st in stems]
        if not test:
            # keep going after this one while anything is left that may still work (incl. one retry of failed ones)
            remaining = sum(1 for st in pending if st not in stems and _failed_count(out_folder, st, engine) < 2)
        job = {"engine": engine, "extra": _hair_eyes(hair_and_eyes), "test": test,
               "dataset": name, "stem": stems[0] if stems else "", "more": remaining}
        what = f"test photo {test_photo}" if test else f"{progress} of '{body_type}'{retry_note}"
        print(f"[AI Empire] Dataset Maker: {what} · engine: {engine.split(' ·')[0]}")
        return (imgs, prompts, captions, seeds, name, stems, remaining, progress, job)


class AIEmpireMakeImage:
    """Picks the result: Qwen's image, or (engine = Nano Banana Pro / Seedream) one made through the RunPod API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job": ("AIEMPIRE_JOB",),
                "template": ("IMAGE",),
                "face": ("IMAGE",),
                "qwen_image": ("IMAGE", {"lazy": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "make"
    CATEGORY = "AI Empire"

    def check_lazy_status(self, job, template=None, face=None, qwen_image=None, prompt=None, extra_pnginfo=None):
        # only run the Qwen part of the graph when Qwen is the engine
        return ["qwen_image"] if job["engine"] not in RUNPOD_MODELS and qwen_image is None else []

    def make(self, job, template, face, qwen_image=None, prompt=None, extra_pnginfo=None):
        if job["engine"] not in RUNPOD_MODELS:
            return (qwen_image,)
        extra = (" " + job["extra"]) if job["extra"] else ""
        try:
            img = _runpod_edit(job["engine"], _tensor_to_pil(template), _tensor_to_pil(face), API_SWAP_PROMPT.format(extra=extra))
        except RuntimeError as e:
            if "API key" in str(e) or not job.get("stem"):
                raise
            if job.get("test"):
                raise RuntimeError(f"{e}. Press Run to try again, pick another test photo, or switch engine.")
            # whole dataset: don't let one refused photo stop the run
            folder = os.path.join(folder_paths.get_output_directory(), "datasets", job["dataset"])
            os.makedirs(folder, exist_ok=True)
            n = _failed_count(folder, job["stem"], job["engine"]) + 1
            with open(_failed_path(folder, job["stem"]), "w", encoding="utf-8") as f:
                json.dump({"engine": job["engine"], "count": n, "error": str(e)[:300]}, f)
            if prompt:
                _queue_again(prompt, extra_pnginfo)  # next photo (failed ones get one more try at the end)
            raise RuntimeError(f"Photo {job['stem']} skipped ({e}). The run continues with the next photo; "
                               "skipped photos get one more try at the end.")
        return (_pil_to_tensor(img),)


def _register_routes():
    """Preview thumbnails of the presets + saving the RunPod key on the pod (never inside the workflow file)."""
    try:
        import server
        from aiohttp import web
    except Exception:
        return
    routes = server.PromptServer.instance.routes
    thumbs = {}

    @routes.get("/aiempire/preset_files")
    async def preset_files(request):
        name = request.query.get("set", "")
        try:
            files = _set_files(name)
        except ValueError:
            return web.json_response({"files": [], "count": 0})
        return web.json_response({"files": files, "count": len(files)})

    @routes.get("/aiempire/preset_thumb")
    async def preset_thumb(request):
        name, fname = request.query.get("set", ""), os.path.basename(request.query.get("file", ""))
        try:
            files = _set_files(name)
        except ValueError:
            return web.Response(status=404)
        if fname not in files:
            return web.Response(status=404)
        path = os.path.join(_set_folder(name), fname)
        ck = (path, os.path.getmtime(path))
        if ck not in thumbs:
            img = _load_template(path, max_side=512)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            thumbs[ck] = buf.getvalue()
        return web.Response(body=thumbs[ck], content_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @routes.get("/aiempire/runpod_key")
    async def key_status(request):
        return web.json_response({"saved": bool(_runpod_key()), "from_env": bool(os.environ.get("AIEMPIRE_RUNPOD_KEY", "").strip())})

    @routes.post("/aiempire/runpod_key")
    async def key_save(request):
        data = await request.json()
        key = str(data.get("key", "")).strip()
        path = _key_file()
        if key:
            with open(path, "w", encoding="utf-8") as f:
                f.write(key)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        elif os.path.exists(path):
            os.remove(path)
        return web.json_response({"saved": bool(_runpod_key())})


_register_routes()


NODE_CLASS_MAPPINGS = {
    "AIEmpireDatasetPresets": AIEmpireDatasetPresets,
    "AIEmpireTemplatePresets": AIEmpireTemplatePresets,
    "AIEmpireBodyPresetMaker": AIEmpireBodyPresetMaker,
    "AIEmpireSaveTemplateSet": AIEmpireSaveTemplateSet,
    "AIEmpireNanoBanana": AIEmpireNanoBanana,
    "AIEmpireSaveDataset": AIEmpireSaveDataset,
    "AIEmpireDatasetMaker": AIEmpireDatasetMaker,
    "AIEmpireMakeImage": AIEmpireMakeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIEmpireDatasetPresets": "AI Empire · Dataset Presets",
    "AIEmpireTemplatePresets": "AI Empire · Template Presets (your photos)",
    "AIEmpireBodyPresetMaker": "AI Empire · Body Preset Maker",
    "AIEmpireSaveTemplateSet": "AI Empire · Save Template Set (new preset)",
    "AIEmpireNanoBanana": "AI Empire · Nano Banana (your Google key)",
    "AIEmpireSaveDataset": "AI Empire · Save Dataset (images + captions)",
    "AIEmpireDatasetMaker": "AI Empire · Dataset Maker",
    "AIEmpireMakeImage": "AI Empire · Make image (Qwen or RunPod API)",
}
