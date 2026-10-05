"""AI Empire local prompt writer.

  * AIEmpirePromptWriter - writes the Krea 2 prompt for you, 100% local, no API key:
    it uses the Krea 2 text encoder (Qwen3-VL 4B, the model that is already loaded) as a small
    vision chat model.
      📷 photo → prompt : copies the shot of a reference photo (pose, outfit, place, light, camera)
      💡 idea → prompt  : turns a short idea ("gym mirror selfie, black set") into a full prompt
      ✍️ my prompt      : no writing, your text goes straight through
    The prompt always starts with your trigger word (+ optional hair/eyes line). The writer never
    describes her face, hair, eyes or body: the character LoRA knows those.
"""

import re

MODE_PHOTO = "📷 photo → prompt"
MODE_IDEA = "💡 idea → prompt"
MODE_MANUAL = "✍️ my prompt"
MODES = [MODE_PHOTO, MODE_IDEA, MODE_MANUAL]

PLACEHOLDER_MAX_SIDE = 64  # the pod puts a tiny grey placeholder in input/ so the Load Image box never errors

SYSTEM = """You write image prompts for Krea 2, a photorealistic text-to-image model, for an AI influencer character.
A character LoRA already knows her face, hair, eyes, skin and body. Your prompt describes the SHOT, never the person.

Write ONE paragraph of 40-90 words, plain natural English, in this order:
1. shot type and camera angle (close-up selfie, mirror selfie, half-body, full-body, from above, low angle...)
2. pose, action and expression (what she does, where she looks, hands)
3. outfit (garments, colours, materials), plainly
4. place and background (specific, real-world details)
5. light (time of day, light source, direction, colour)
6. the photo look, e.g. "casual amateur iPhone photo, natural colours, slight grain"

Rules:
- Refer to her only as "she"/"her" or "the woman". Her name is added in front of your text.
- She is an adult woman. Never use the words girl, teen, teenage, schoolgirl, child or kid.
- Never describe her face, facial features, hair, eye colour, skin tone, ethnicity, age, weight or body shape.
- Never name a real person or celebrity.
- No lists, no quotes, no markdown, no "Prompt:" label, no commentary. Output the paragraph only.
- If a photo shows a person who could be under 18, output exactly: REFUSE"""

PHOTO_TASK = ("Write the prompt for this photo. Copy its shot as faithfully as possible: same framing, camera angle, "
              "pose, expression, outfit, place and light. Describe what is actually there.")
IDEA_TASK = "Write the prompt for this idea: {idea}"
CHANGES = " Apply these changes on top: {idea}"

_BANNED = re.compile(r"\b(teen|teens|teenage\w*|teenager\w*|school\s*girl\w*|underage|under[- ]age|child|children|kids?|"
                     r"loli\w*|preteen\w*|young[- ]looking|jailbait)\b", re.I)


def _clean(text):
    t = (text or "").strip()
    t = t.partition("</think>")[2] if "</think>" in t else t
    t = t.strip().strip('"').strip()
    t = re.sub(r"^\s*(prompt|here is the prompt)\s*:\s*", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\bgirls\b", "women", t, flags=re.I)
    t = re.sub(r"\bgirl\b", "woman", t, flags=re.I)
    return t.strip()


def _check_adult(text, where):
    m = _BANNED.search(text or "")
    if m:
        raise RuntimeError(f"Blocked: the {where} contains '{m.group(0)}'. Prompts must describe an adult woman only.")


def _assemble(trigger, hair_and_eyes, body):
    head = ", ".join(p.strip().strip(",") for p in (trigger, hair_and_eyes) if p and p.strip())
    body = body.strip()
    if head and body and body[:1].isupper() and not body[:2].isupper():
        body = body[0].lower() + body[1:]  # "zvx woman, mirror selfie ..." not "zvx woman, Mirror selfie ..."
    return f"{head}, {body}" if head and body else (head or body)


class AIEmpirePromptWriter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The Krea 2 text encoder (Qwen3-VL 4B). It writes the prompt, nothing extra to download."}),
                "mode": (MODES, {"default": MODE_IDEA}),
                "trigger_word": ("STRING", {"default": "zvx woman", "tooltip": "Your LoRA trigger word. Always first in the prompt."}),
                "hair_and_eyes": ("STRING", {"default": "", "tooltip": "Optional, e.g. 'long straight red hair, green eyes'. Added right after the trigger word."}),
                "idea": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "📷 photo mode: optional changes (e.g. 'make the outfit black'). "
                                               "💡 idea mode: your idea. ✍️ my prompt: the whole prompt (without the trigger word)."}),
                "creativity": ("FLOAT", {"default": 0.7, "min": 0.05, "max": 1.5, "step": 0.05, "tooltip": "Higher = more varied wording."}),
                "variation": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True,
                                      "tooltip": "fixed = the prompt is written once and reused. Change it / randomize = a new prompt."}),
            },
            "optional": {
                "image": ("IMAGE", {"lazy": True, "tooltip": "Reference photo (only used in 📷 photo mode)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "AI Empire"

    def check_lazy_status(self, mode, image=None, **kwargs):
        # the reference photo is only loaded in photo mode
        return ["image"] if mode == MODE_PHOTO and image is None else []

    def run(self, clip, mode, trigger_word, hair_and_eyes, idea, creativity, variation, image=None):
        idea = (idea or "").strip()
        _check_adult(trigger_word + " " + hair_and_eyes, "trigger word / hair line")

        if mode == MODE_MANUAL:
            if not idea:
                raise RuntimeError("✍️ my prompt mode: type your prompt in the 'idea' box.")
            _check_adult(idea, "prompt")
            p = _assemble(trigger_word, hair_and_eyes, _clean(idea))
            return {"ui": {"text": [p]}, "result": (p,)}

        if idea:
            _check_adult(idea, "idea")
        if mode == MODE_PHOTO:
            if image is None:
                raise RuntimeError("📷 photo mode needs a reference photo: upload one in the 'Reference photo' box.")
            if max(image.shape[1], image.shape[2]) <= PLACEHOLDER_MAX_SIDE:
                raise RuntimeError("Upload your reference photo in the 'Reference photo' box first "
                                   "(or switch the Prompt Writer to 💡 idea mode).")
            task = PHOTO_TASK + (CHANGES.format(idea=idea) if idea else "")
            image = image[:1, :, :, :3]
        else:
            if not idea:
                raise RuntimeError("💡 idea mode: type your idea in the 'idea' box (e.g. 'gym mirror selfie, black set').")
            task = IDEA_TASK.format(idea=idea)
            image = None

        # same calls as ComfyUI's built-in "Generate Text" node
        tokens = clip.tokenize(task, image=image, min_length=1, thinking=False, system_prompt=SYSTEM)
        ids = clip.generate(tokens, do_sample=True, max_length=320, temperature=float(creativity), top_k=64,
                            top_p=0.95, min_p=0.05, repetition_penalty=1.05, seed=int(variation))
        out = _clean(clip.decode(ids))
        if not out or out.upper().startswith("REFUSE"):
            raise RuntimeError("The prompt writer refused this reference (it may show someone under 18). Use another photo.")
        _check_adult(out, "written prompt")
        p = _assemble(trigger_word, hair_and_eyes, out)
        print(f"[AI Empire] prompt: {p}")
        return {"ui": {"text": [p]}, "result": (p,)}


NODE_CLASS_MAPPINGS = {"AIEmpirePromptWriter": AIEmpirePromptWriter}
NODE_DISPLAY_NAME_MAPPINGS = {"AIEmpirePromptWriter": "AI Empire · Prompt Writer (local)"}
