#!/usr/bin/env python3
"""
AI Empire - make body-type presets from the athletic set with Nano Banana Pro
(RunPod public endpoint). Runs on your own PC, no pod needed.

Folder layout (next to this script):
    body_presets.py
    photos.json          per-photo pose / outfit / visible body parts / caption (made by Claude)
    athletic/            the 50 athletic photos (any names with a number work, they get renamed image_01...)
    body_refs/           curvy.jpg, thick.jpg, ...  (one body example per type, optional)

Commands:
    python3 body_presets.py curvy --test           -> 3 test photos (13, 25, 47) into _test/curvy/
    python3 body_presets.py curvy --test 5,22,41   -> pick your own test photos
    python3 body_presets.py curvy                  -> the whole set into curvy/ (skips ones already done)
    python3 body_presets.py all                    -> every body type, one after another

Photos that show none of the body parts a type changes (close-ups, headshots) are copied unchanged
for free. Every output gets its caption .txt from photos.json.

API key: paste it the first time you run it. It's saved in .runpod_key next to this script (only readable by you),
so it never asks again. Wrong key? The file is deleted automatically and it asks again next run.
Don't share this folder with the .runpod_key file inside it.
"""
import argparse
import base64
import concurrent.futures as cf
import getpass
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ─────────────────────────── settings you can change ───────────────────────────

SOURCE = "athletic"          # folder with the finished athletic 50
REFS = "body_refs"           # folder with one body example photo per type (name = type)
RESOLUTION = "2k"            # 1k / 2k ($0.14)  or 4k ($0.24). Pipeline works at 2048, so 2k.
AT_ONCE = 3                  # photos running at the same time
TRIES = 3                    # attempts per photo before giving up

TEST_PICKS = [13, 25, 47]    # default --test photos: full body standing, kneeling, bent forward

BODY = {
    "curvy":     "make her softer and a little curvier instead of athletic: no visible abs or muscle definition, a soft smooth flat stomach, slightly fuller hips and thighs, a slightly fuller bust. Not muscular, not heavy, still slim.",
    "thick":     "make her clearly thick: much wider hips, a big round butt and thick full thighs, with a defined waist; her bust stays as it is.",
    "petite":    "make her chest clearly smaller, a small A-cup bust, and her butt clearly smaller and flatter, with narrower hips and slimmer thighs. A small, slender, petite figure, the opposite of curvy.",
    "petiteb":   "flat-chested, small flat butt, narrow straight hips",
    "busty":     "make her bust clearly much bigger and fuller, at least two cup sizes larger; keep her waist slim and her hips and legs as they are.",
    "plus":      "a plus-size build, US clothing size 20: heavy soft arms, a big soft belly, very wide hips, very thick thighs, a large bust and a softer, fuller face",
}

# body parts each type changes. A photo is only sent if it shows at least one of them.
PARTS = {
    "curvy":     ["chest", "waist", "hips", "butt", "legs"],
    "thick":     ["hips", "butt", "legs"],
    "petite":    ["chest", "hips", "butt", "legs"],
    "petiteb":   ["chest", "hips", "butt", "legs"],
    "busty":     ["chest"],
    "plus":      ["chest", "waist", "hips", "butt", "legs", "arms"],
}
# how big the change should look. Curvy is a gentle step from athletic; the others are clearly different bodies.
STRENGTH = {
    "curvy": "The change is gentle but visible: same girl, just softer and less toned.",
    "slim": "The change must be obvious at first glance: she is clearly thinner than in the original. "
            "Her clothes stay the same size, so they sit a little looser on her.",
    "plus": "The change must be obvious at first glance: she is clearly a plus-size woman now, not just curvy. "
            "Her neck, jaw and cheeks can be slightly fuller to match her body, but she is still recognizably "
            "the same person. Her clothes stretch tighter over her fuller body.",
}
STRENGTH_DEFAULT = "The body change must be obvious at first glance compared to the original."
PART_WORDS = {"chest": "chest", "waist": "waist", "hips": "hips", "butt": "butt", "legs": "thighs", "arms": "arms"}

# image 1 = body example, image 2 = the athletic photo (same order that worked for 36-50)
PROMPT_WITH_REF = (
    "Output image 2, edited. Image 2 is a photo of a woman {pose}, wearing {outfit}. Keep the woman from image 2 "
    "with her own face, hair, pose, background, camera angle, framing and lighting, all exactly the same. Keep exactly "
    "the same clothes: {outfit}, same colors, same cut, same coverage, not more revealing. The only change: give the woman "
    "in image 2 the body proportions of the woman in image 1 — match the size and shape of her {parts} closely: {body} "
    "Copy nothing else from image 1: not her face, skin tone, clothes, pose or background. Natural, realistic skin texture like a real phone photo, not "
    "airbrushed; no scars, marks or blemishes on her arms, legs or body. {strength} Realistic anatomy. Her clothes fit her new shape naturally. "
    "Photorealistic smartphone photo."
)
# weight changes (slim / plus): the "edit, keep everything" prompts barely moved her, so these recreate the
# photo with a different body instead. Used when there is no body example for the type.
REIMAGINE = {"slim", "plus"}
PROMPT_REIMAGINE = (
    "Recreate this exact photo: the same scene, background, lighting, camera angle, framing and pose, the same hair, "
    "and the same outfit ({outfit}, same colors and style, same coverage). But the woman has a completely different "
    "body: {body}. The difference from the original photo must be obvious at a glance. Her face stays recognizable as "
    "the same woman. Her clothes fit this new body naturally. Realistic anatomy, natural realistic skin texture (not airbrushed), no scars or marks. "
    "Photorealistic smartphone photo."
)
# short prompts for making her SMALLER: the long "keep everything" prompt made the model do almost nothing.
SHORT = {
    "petite": (
        "Make her breasts much smaller, nearly flat-chested, a small A-cup, and make her butt and hips much smaller "
        "and flatter, like a skinny petite girl. Keep her face, hair, outfit ({outfit}), pose, background and "
        "lighting the same. Photorealistic phone photo."
    ),
    "petiteb": (
        "This same girl, but flat-chested, with a small flat butt and narrow boyish hips: a skinny, petite, straight "
        "figure with almost no curves. Same face, hair, outfit ({outfit}), pose, background and lighting. "
        "Photorealistic phone photo."
    ),
}
# used when body_refs/<type>.* is missing: only the athletic photo is sent
PROMPT_NO_REF = (
    "Edit this photo of a woman {pose}, wearing {outfit}. Keep her own face, hair, pose, background, camera angle, "
    "framing and lighting, all exactly the same. Keep exactly the same clothes: {outfit}, same colors, same cut, same "
    "coverage, not more revealing. The only change: {body} In this photo you can see her {parts}, so reshape only those. "
    "Natural, realistic skin texture like a real phone photo, not airbrushed; no scars, marks or blemishes on her arms, legs or body. "
    "{strength} Realistic anatomy. Her clothes fit her new shape naturally. Photorealistic smartphone photo."
)

# ────────────────────────────────────────────────────────────────────────────────

MODELS = {
    "nano": "https://api.runpod.ai/v2/nano-banana-pro-edit",   # Nano Banana Pro Edit, $0.14 at 2K
    "nano2": "https://api.runpod.ai/v2/google-nano-banana-2-edit",  # Nano Banana 2 Edit (newer, cheaper than Pro)
    "seedream": "https://api.runpod.ai/v2/seedream-v4-edit",   # Seedream 4.0 Edit, ~$0.03
}
TYPE_MODEL = {}   # which model a type uses by default, e.g. {"petite": "seedream"}; everything else = nano


def model_url(model):
    return os.environ.get("NB_BASE_URL") or MODELS[model]


def seedream_size(w, h, long_side=2048):
    s = long_side / max(w, h)
    return f"{max(16, round(w * s / 16) * 16)}*{max(16, round(h * s / 16) * 16)}"
RATIOS = ["1:1", "3:2", "2:3", "4:3", "3:4", "4:5", "5:4", "9:16", "16:9", "21:9"]
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PRINT = threading.Lock()


def say(*a):
    with PRINT:
        print(*a, flush=True)


KEY_FILE = HERE / ".runpod_key"


def get_key():
    """env RUNPOD_API_KEY > saved .runpod_key > ask once and save it."""
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
        if key:
            return key
    key = getpass.getpass("RunPod API key (hidden while you paste, saved for next time): ").strip()
    if key:
        KEY_FILE.write_text(key)
        try:
            KEY_FILE.chmod(0o600)
        except OSError:
            pass
        say(f"🔑 key saved in {KEY_FILE.name} — you won't be asked again")
    return key


def ensure_pillow():
    """Pillow shrinks photos before upload. If it's missing, make a private venv next to the script once."""
    try:
        import PIL  # noqa: F401
        return
    except ImportError:
        pass
    venv = HERE / ".venv"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if Path(sys.executable).resolve() == py.resolve():
        sys.exit("Pillow still missing inside .venv - delete the .venv folder and run again.")
    if not py.exists():
        print("First run: setting up a small Python environment (Pillow)... ~20 seconds")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
        subprocess.check_call([str(py), "-m", "pip", "install", "-q", "pillow"])
    os.execv(str(py), [str(py), *sys.argv])


def natural(p):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]


def number_of(p):
    import re
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else None


def load_info():
    f = HERE / "photos.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items() if k.isdigit()}


def tidy_names(folder):
    """36.jpeg / my_influencer_image7.png ... -> image_36.jpeg / image_07.png (numbers kept, nothing else changes)."""
    moved = 0
    for p in list(folder.iterdir()):
        n = number_of(p) if p.suffix.lower() in EXTS | {".txt"} else None
        if n is None:
            continue
        new = p.with_name(f"image_{n:02d}{p.suffix.lower()}")
        if new != p and not new.exists():
            p.rename(new)
            moved += 1
    if moved:
        say(f"📝 renamed {moved} files in {folder.name}/ to image_01 ... style")


def join_words(words):
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def build_prompt(kind, info, parts, with_ref):
    if kind in SHORT:
        return SHORT[kind].format(outfit=info.get("outfit", "her own clothes"))
    if kind in REIMAGINE and not with_ref:
        return PROMPT_REIMAGINE.format(outfit=info.get("outfit", "her own clothes"), body=BODY[kind].rstrip("."))
    tpl = PROMPT_WITH_REF if with_ref else PROMPT_NO_REF
    return tpl.format(
        pose=info.get("pose", "in this photo"),
        outfit=info.get("outfit", "her own clothes"),
        body=BODY[kind],
        parts=join_words([PART_WORDS[x] for x in parts]),
        strength=STRENGTH.get(kind, STRENGTH_DEFAULT),
    )


def photos(folder):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in EXTS], key=natural)


def as_data_url(path, max_side):
    """JPEG q95, long side <= max_side: keeps each upload ~1 MB instead of ~8 MB of PNG."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), (w, h)


def nearest_ratio(w, h):
    def val(r):
        a, b = r.split(":")
        return int(a) / int(b)
    import math
    return min(RATIOS, key=lambda r: abs(math.log(val(r)) - math.log(w / h)))


def http(url, key, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:400]
        if e.code in (401, 403):
            KEY_FILE.unlink(missing_ok=True)
            raise SystemExit(f"🛑 RunPod says the API key is wrong or has no access ({e.code}): {msg}")
        raise RuntimeError(f"HTTP {e.code}: {msg}")


JOB_TIMEOUT = int(os.environ.get("NB_JOB_TIMEOUT", "240"))  # seconds per photo before cancelling and retrying


def run_job(key, payload, label="", base=None):
    """/run returns a job id at once; poll /status. A job stuck longer than JOB_TIMEOUT is cancelled
    (so it isn't billed) and the caller retries it."""
    base = base or model_url("nano")
    res = http(f"{base}/run", key, {"input": payload}, timeout=60)
    job = res.get("id")
    start = time.time()
    warned = False
    while res.get("status") in ("IN_QUEUE", "IN_PROGRESS") or (job and not res.get("status")):
        waited = time.time() - start
        if waited > JOB_TIMEOUT:
            try:
                http(f"{base}/cancel/{job}", key, {}, timeout=30)
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"stuck for {JOB_TIMEOUT // 60} min in RunPod's queue — cancelled, trying again")
        if waited > 90 and not warned:
            say(f"  ⏳ {label} is slow ({res.get('status', 'waiting').lower().replace('_', ' ')}) — still waiting, max {JOB_TIMEOUT // 60} min")
            warned = True
        time.sleep(3)
        res = http(f"{base}/status/{job}", key, timeout=30)
    if res.get("status") != "COMPLETED":
        raise RuntimeError(f"{res.get('status')}: {res.get('error') or json.dumps(res.get('output'))[:300]}")
    out = res.get("output") or {}
    url = out.get("image_url") or out.get("result")
    if not url:
        raise RuntimeError(f"no image in answer: {json.dumps(out)[:300]}")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read(), float(out.get("cost") or 0)


def find_ref(kind):
    folder = HERE / REFS
    if not folder.is_dir():
        return None
    for p in folder.iterdir():
        if p.stem.lower() == kind.lower() and p.suffix.lower() in EXTS:
            return p
    return None


def make_type(kind, key, info, only=None, test=False, model=None):
    src = HERE / SOURCE
    default_model = TYPE_MODEL.get(kind, "nano")
    model = model or default_model
    base = model_url(model)
    folder = kind if (model == default_model or not test) else f"{kind}_{model}"
    out = HERE / ("_test" if test else "") / folder
    out.mkdir(parents=True, exist_ok=True)

    todo = photos(src)
    if only:
        todo = [p for p in todo if number_of(p) in only]
        if not todo:
            sys.exit(f"🛑 none of {sorted(only)} found in {src}")

    ref = find_ref(kind)
    ref_url = as_data_url(ref, 1536)[0] if ref else None
    say(f"\n━━ {kind} ━━ {len(todo)} photos → {out.relative_to(HERE)}/  "
        f"({'body example: ' + ref.name if ref else 'no body example, text only'}) · model: {model}")

    cost = 0.0
    failed = []
    done = 0

    def one(p):
        n = number_of(p)
        stem = f"image_{n:02d}" if n is not None else p.stem
        target = out / (stem + ".png")
        meta = info.get(n, {})
        cap = meta.get("caption")
        if cap:
            (out / (stem + ".txt")).write_text(cap, encoding="utf-8")
        elif p.with_suffix(".txt").exists():
            shutil.copy2(p.with_suffix(".txt"), out / (stem + ".txt"))
        if target.exists():
            if not test:
                return p, "skip", 0.0, None
            target.unlink()  # test = always fresh, so an old result can't be mistaken for a new one
        if meta:
            parts = [x for x in PARTS[kind] if x in meta.get("shows", [])]
        else:
            parts = list(PARTS[kind])  # no info for this photo: edit it anyway
        if not parts:
            shutil.copy2(p, target) if p.suffix.lower() == ".png" else _to_png(p, target)
            return p, "kept", 0.0, None
        photo_url, (w, h) = as_data_url(p, 2048)
        images = ([ref_url] if ref_url else []) + [photo_url]
        prompt = build_prompt(kind, meta, parts, bool(ref_url))
        if model == "seedream":
            payload = {"prompt": prompt, "images": images, "size": seedream_size(w, h)}
        else:
            payload = {
                "prompt": prompt,
                "images": images,
                "resolution": RESOLUTION,
                "aspect_ratio": nearest_ratio(w, h),
                "output_format": "png",
            }
        err = None
        for attempt in range(1, TRIES + 1):
            try:
                data, c = run_job(key, payload, f"image_{n:02d}" if n is not None else p.stem, base)
                tmp = target.with_suffix(".part")
                tmp.write_bytes(data)
                _normalise(tmp, target)
                return p, "ok", c, None
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                err = str(e)
                if attempt < TRIES:
                    time.sleep(3 * attempt)
        return p, "fail", 0.0, err

    with cf.ThreadPoolExecutor(AT_ONCE) as pool:
        for p, state, c, err in pool.map(one, todo):
            done += 1
            cost += c
            icon = {"ok": "✅", "skip": "⏭ ", "kept": "📋", "fail": "❌"}[state]
            extra = {"skip": "already done", "kept": "no body part to change, copied", "fail": err, "ok": f"${c:.2f}"}[state]
            n = number_of(p)
            say(f"  {icon} {done:>2}/{len(todo)}  image_{n:02d}  {extra}" if n is not None else f"  {icon} {p.stem}  {extra}")
            if state == "fail":
                failed.append(p.stem)

    say(f"━━ {kind} done · spent ${cost:.2f}" + (f" · ❌ failed: {', '.join(failed)} (run again to retry)" if failed else ""))
    return cost, failed


def make_review(kind):
    """Small copy of a finished set for checking in chat: 1024 px JPEGs, well under 30 MB."""
    import zipfile
    from PIL import Image
    folder = HERE / kind
    files = photos(folder) if folder.is_dir() else []
    if not files:
        sys.exit(f"🛑 {kind}/ has no photos yet — run: python3 body_presets.py {kind}")
    out = HERE / f"{kind}_review.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in files:
            im = Image.open(p).convert("RGB")
            im.thumbnail((1024, 1024))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
            z.writestr(p.stem + ".jpg", buf.getvalue())
    say(f"📦 {out.name}: {len(files)} photos, {out.stat().st_size / 1e6:.1f} MB — send this one")


def _to_png(src, target):
    from PIL import Image
    Image.open(src).save(target, "PNG")


def _normalise(tmp, target):
    """Save as a real PNG whatever RunPod sent back."""
    from PIL import Image
    try:
        with Image.open(tmp) as im:
            fmt = im.format
            if fmt != "PNG":
                im.convert("RGB").save(target, "PNG")
        if fmt == "PNG":
            tmp.replace(target)
        else:
            tmp.unlink()
    except Exception:
        tmp.replace(target)


def main():
    ap = argparse.ArgumentParser(description="Make body-type presets from the athletic set with Nano Banana Pro.")
    ap.add_argument("type", help="one of: " + ", ".join(BODY) + ", or all")
    ap.add_argument("--model", choices=list(MODELS), help="override the model for this run (nano or seedream)")
    ap.add_argument("--review", action="store_true",
                    help="make <type>_review.zip (small JPEGs, fits in a chat upload) instead of generating")
    ap.add_argument("--test", metavar="NUMS", nargs="?", const="default",
                    help="test run into _test/<type>/: photos 13,25,47, or your own numbers e.g. --test 5,22,41")
    a = ap.parse_args()

    kinds = list(BODY) if a.type == "all" else [a.type]
    for k in kinds:
        if k not in BODY:
            sys.exit(f"🛑 unknown type '{k}'. Pick from: {', '.join(BODY)}, all")
    if not (HERE / SOURCE).is_dir() or not photos(HERE / SOURCE):
        sys.exit(f"🛑 put the athletic photos in {HERE / SOURCE}/ first (image_01.png ... image_50.png)")

    ensure_pillow()
    if a.review:
        for k in kinds:
            make_review(k)
        return
    tidy_names(HERE / SOURCE)
    info = load_info()
    if not info:
        say("⚠ photos.json not found next to the script: every photo gets edited with the plain prompt")
    wrote = 0
    for p in photos(HERE / SOURCE):  # the athletic preset gets its captions too
        cap = info.get(number_of(p), {}).get("caption")
        if cap and not p.with_suffix(".txt").exists():
            p.with_suffix(".txt").write_text(cap, encoding="utf-8")
            wrote += 1
    if wrote:
        say(f"📝 wrote {wrote} captions into {SOURCE}/")
    key = get_key()
    if not key:
        sys.exit("🛑 no API key")

    if a.test == "default":
        only = set(TEST_PICKS)
    else:
        only = {int(x) for x in a.test.split(",") if x.strip()} if a.test else None
    total, all_failed = 0.0, {}
    for k in kinds:
        c, f = make_type(k, key, info, only=only, test=bool(only), model=a.model)
        total += c
        if f:
            all_failed[k] = f
    if len(kinds) > 1:
        say(f"\n🎉 all done · total ${total:.2f}" + (f" · failed: {all_failed}" if all_failed else ""))


if __name__ == "__main__":
    main()
