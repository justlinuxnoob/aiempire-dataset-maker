"""AI Empire Grok prompt node.

  * AIEmpireGrokPrompt - writes the Krea 2 prompt for you with Grok (xAI API, YOUR OWN xAI key):
      📷 photo → prompt : copies the shot of a reference photo (pose, outfit, place, light, camera)
      💡 idea → prompt  : turns a short idea ("gym mirror selfie, black set") into a full prompt
      ✍️ my prompt      : no Grok, your text goes straight through
    The prompt always starts with your trigger word (+ optional hair/eyes line). Grok never describes
    her face, hair, eyes or body: the character LoRA knows those.

The key is saved on the pod with the 🔑 button (/workspace/.xai_key), never inside the workflow file.
Env XAI_API_KEY works too.
"""

import base64
import io
import json
import os
import re
import time

import numpy as np
from PIL import Image

import folder_paths

XAI_URL = "https://api.x.ai/v1/chat/completions"
MODELS = ["grok-4.20-0309-non-reasoning", "grok-4.20-0309-reasoning"]

MODE_PHOTO = "📷 photo → prompt"
MODE_IDEA = "💡 idea → prompt"
MODE_MANUAL = "✍️ my prompt (no Grok)"
MODES = [MODE_PHOTO, MODE_IDEA, MODE_MANUAL]

PLACEHOLDER_MAX_SIDE = 64  # the pod puts a tiny grey placeholder in input/ so the Load Image box never errors

SYSTEM = """You write image prompts for Krea 2, a photorealistic text-to-image model, for an AI influencer character.
A character LoRA already knows her face, hair, eyes, skin and body. Your prompt describes the SHOT, never the person.

Write ONE paragraph of 40-90 words, plain natural English, in this order:
1. shot type and camera angle (close-up selfie, mirror selfie, half-body, full-body, from above, low angle...)
2. pose, action and expression (what she does, where she looks, hands)
3. outfit (garments, colours, materials) exactly as given, plainly
4. place and background (specific, real-world details)
5. light (time of day, light source, direction, colour)
6. the photo look, e.g. "casual amateur iPhone photo, natural colours, slight grain"

Hard rules:
- Refer to her only as "she"/"her" or "the woman". The caller adds her name/trigger word in front of your text.
- She is an adult woman. Never use the words girl, teen, teenage, schoolgirl, young-looking, child, kid, minor, petite girl.
  Never mention school uniforms or anything that makes her look under 18.
- Never describe her face, facial features, hair, eye colour, skin tone, ethnicity, age, weight or body shape.
- Never name or describe a real, identifiable person, brand mascot or celebrity.
- No lists, no quotes, no markdown, no "Prompt:" label, no commentary. Output the paragraph only.
- If a reference photo shows a person who could be under 18, output exactly: REFUSE"""

PHOTO_TASK = ("Copy the shot of this reference photo as faithfully as possible: same framing, camera angle, pose, "
              "expression, outfit, place and light. Describe what is actually there, plainly. "
              "Ignore who the person in the photo is.")
IDEA_TASK = "Turn this idea into the prompt: {idea}"
CHANGES = "Apply these changes on top: {idea}"

_BANNED = re.compile(r"\b(teen|teens|teenage\w*|teenager\w*|school\s*girl\w*|underage|under[- ]age|child|children|kids?|"
                     r"loli\w*|preteen\w*|young[- ]looking|jailbait)\b", re.I)


# ---------------------------------------------------------------- key
def _key_file():
    # /workspace/.xai_key on the pod (next to input/ and output/, survives restarts on a network volume)
    return os.path.join(os.path.dirname(os.path.abspath(folder_paths.get_input_directory())), ".xai_key")


def _xai_key():
    try:
        with open(_key_file(), encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    except OSError:
        pass
    return os.environ.get("XAI_API_KEY", "").strip()


# ---------------------------------------------------------------- helpers
def _data_url(t, max_side=1024):
    arr = (t[..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    img = Image.fromarray(arr)
    w, h = img.size
    if max(w, h) > max_side:
        k = max_side / max(w, h)
        img = img.resize((round(w * k), round(h * k)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _clean(text):
    t = (text or "").strip().strip('"').strip()
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
    return f"{head}, {body}" if head and body else (head or body)


def _call_grok(key, model, messages, temperature, tries=3):
    import urllib.error
    import urllib.request
    body = json.dumps({"model": model, "messages": messages, "temperature": float(temperature),
                       "max_tokens": 600, "stream": False}).encode()
    last = None
    for attempt in range(tries):
        req = urllib.request.Request(os.environ.get("AIEMPIRE_XAI_URL", "") or XAI_URL, data=body, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode())
            return data["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (401, 403):
                raise RuntimeError("xAI says the API key is wrong. Click '🔑 xAI key' on the Grok box and paste it again.")
            if e.code in (429,) or e.code >= 500:
                last = f"xAI HTTP {e.code}: {msg}"
            else:
                raise RuntimeError(f"xAI HTTP {e.code}: {msg}")
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            last = f"can't reach xAI ({getattr(e, 'reason', e)})"
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(last or "xAI request failed")


# ---------------------------------------------------------------- node
class AIEmpireGrokPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (MODES, {"default": MODE_PHOTO}),
                "trigger_word": ("STRING", {"default": "zvx woman", "tooltip": "Your LoRA trigger word. Always first in the prompt."}),
                "hair_and_eyes": ("STRING", {"default": "", "tooltip": "Optional, e.g. 'long straight red hair, green eyes'. Added right after the trigger word, never written by Grok."}),
                "idea": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "📷 photo mode: optional changes (e.g. 'make the outfit black'). "
                                               "💡 idea mode: your idea. ✍️ my prompt: the whole prompt (without the trigger word)."}),
                "model": (MODELS, {"default": MODELS[0]}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
                "variation": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True,
                                      "tooltip": "fixed = Grok is asked once and the prompt is reused (saves money). Change it / randomize = new prompt."}),
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
        # the reference photo is only loaded (and paid for) in photo mode
        return ["image"] if mode == MODE_PHOTO and image is None else []

    def run(self, mode, trigger_word, hair_and_eyes, idea, model, temperature, variation, image=None):
        idea = (idea or "").strip()
        _check_adult(trigger_word + " " + hair_and_eyes, "trigger word / hair line")

        if mode == MODE_MANUAL:
            if not idea:
                raise RuntimeError("✍️ my prompt mode: type your prompt in the 'idea' box.")
            _check_adult(idea, "prompt")
            p = _assemble(trigger_word, hair_and_eyes, _clean(idea))
            return {"ui": {"text": [p]}, "result": (p,)}

        key = _xai_key()
        if not key:
            raise RuntimeError("No xAI key yet: click '🔑 xAI key' on the Grok box (console.x.ai → API Keys).")

        content = []
        if mode == MODE_PHOTO:
            if image is None:
                raise RuntimeError("📷 photo mode needs a reference photo: upload one in the 'Reference photo' box.")
            if max(image.shape[1], image.shape[2]) <= PLACEHOLDER_MAX_SIDE:
                raise RuntimeError("Upload your reference photo in the 'Reference photo' box first "
                                   "(or switch the Grok box to 💡 idea mode).")
            text = PHOTO_TASK + (" " + CHANGES.format(idea=idea) if idea else "")
            if idea:
                _check_adult(idea, "idea")
            content.append({"type": "text", "text": text})
            for t in image[:4]:  # up to 4 photos of a batch
                content.append({"type": "image_url", "image_url": {"url": _data_url(t), "detail": "high"}})
        else:
            if not idea:
                raise RuntimeError("💡 idea mode: type your idea in the 'idea' box (e.g. 'gym mirror selfie, black set').")
            _check_adult(idea, "idea")
            content.append({"type": "text", "text": IDEA_TASK.format(idea=idea)})

        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        out = _clean(_call_grok(key, model, messages, temperature))
        if not out or out.upper().startswith("REFUSE"):
            raise RuntimeError("Grok refused this reference (it may show someone under 18). Use another photo.")
        _check_adult(out, "Grok prompt")
        p = _assemble(trigger_word, hair_and_eyes, out)
        print(f"[AI Empire] Grok prompt: {p}")
        return {"ui": {"text": [p]}, "result": (p,)}


# ---------------------------------------------------------------- routes (🔑 button)
def _register_routes():
    try:
        import server
        from aiohttp import web
    except Exception:
        return
    routes = server.PromptServer.instance.routes

    @routes.get("/aiempire/xai_key")
    async def key_status(request):
        return web.json_response({"saved": bool(_xai_key()), "from_env": bool(os.environ.get("XAI_API_KEY", "").strip())})

    @routes.post("/aiempire/xai_key")
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
        return web.json_response({"saved": bool(_xai_key())})


_register_routes()


NODE_CLASS_MAPPINGS = {"AIEmpireGrokPrompt": AIEmpireGrokPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"AIEmpireGrokPrompt": "AI Empire · Grok Prompt"}
