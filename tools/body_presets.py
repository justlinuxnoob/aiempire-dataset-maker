#!/usr/bin/env python3
"""
AI Empire - make body-type presets from the athletic set with Nano Banana Pro
(RunPod public endpoint). Runs on your own PC, no pod needed.

Folder layout (next to this script):
    athletic/            image_01.png ... image_50.png   (+ optional image_01.txt captions)
    body_refs/           curvy.jpg, thick.jpg, ...        (one body example per type, optional)

Commands:
    python3 body_presets.py curvy --test 5,22,41   -> 3 test photos into _test/curvy/
    python3 body_presets.py curvy                  -> all 50 into curvy/ (skips ones already done)
    python3 body_presets.py all                    -> every body type below, one after another

API key: set RUNPOD_API_KEY, or just run it and paste the key when asked (it is never saved).
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

# photo numbers to copy unchanged (close-ups / headshots with no body to change)
KEEP_AS_IS = []              # e.g. [3, 7, 12]

BODY = {
    "curvy":     "give her a curvy hourglass figure: fuller bust and hips, a clearly narrower waist, soft fuller thighs.",
    "thick":     "give her wider hips, a fuller rounder butt and thicker thighs, with a defined waist; her bust stays as it is.",
    "busty":     "make her bust noticeably fuller and larger; keep her waist slim and her hips and legs as they are.",
    "slim":      "make her slimmer and more petite overall: slender arms, narrow hips, slim legs, a smaller bust.",
    "slimthick": "keep a slim toned waist and flat stomach, but give her fuller hips, a rounder butt and fuller thighs.",
    "plus":      "make her plus-size: a softer, fuller body overall, with fuller arms, stomach, hips and thighs.",
}

# image 1 = body example, image 2 = the athletic photo (same order that worked for 36-50)
PROMPT_WITH_REF = (
    "Output image 2, edited. Keep the woman from image 2 with her own face, hair, clothes, pose, background, "
    "camera angle, framing and lighting, all exactly the same. Image 1 only shows the body shape to aim for; "
    "do not copy anything else from it. The only change: {body} Her skin is smooth and clean everywhere: no scars, "
    "marks, lines, creases or blemishes on her arms, legs or body. Realistic, natural proportions. Her own clothes "
    "fit her new shape naturally. Photorealistic smartphone photo."
)
# used when body_refs/<type>.* is missing: only the athletic photo is sent
PROMPT_NO_REF = (
    "Edit this photo. Keep the woman with her own face, hair, clothes, pose, background, camera angle, framing and "
    "lighting, all exactly the same. The only change: {body} Her skin is smooth and clean everywhere: no scars, "
    "marks, lines, creases or blemishes on her arms, legs or body. Realistic, natural proportions. Her own clothes "
    "fit her new shape naturally. Photorealistic smartphone photo."
)

# ────────────────────────────────────────────────────────────────────────────────

BASE_URL = os.environ.get("NB_BASE_URL", "https://api.runpod.ai/v2/nano-banana-pro-edit")
RATIOS = ["1:1", "3:2", "2:3", "4:3", "3:4", "4:5", "5:4", "9:16", "16:9", "21:9"]
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PRINT = threading.Lock()


def say(*a):
    with PRINT:
        print(*a, flush=True)


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
            raise SystemExit(f"🛑 RunPod says the API key is wrong or has no access ({e.code}): {msg}")
        raise RuntimeError(f"HTTP {e.code}: {msg}")


def run_job(key, payload):
    """runsync, and if RunPod hands back a job id instead, poll /status until it finishes."""
    res = http(f"{BASE_URL}/runsync", key, {"input": payload}, timeout=300)
    deadline = time.time() + 600
    while res.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        if time.time() > deadline:
            raise RuntimeError("timed out after 10 minutes")
        time.sleep(4)
        res = http(f"{BASE_URL}/status/{res['id']}", key)
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


def make_type(kind, key, only=None, test=False):
    src = HERE / SOURCE
    out = HERE / ("_test" if test else "") / kind
    out.mkdir(parents=True, exist_ok=True)

    todo = photos(src)
    if only:
        todo = [p for p in todo if number_of(p) in only]
        if not todo:
            sys.exit(f"🛑 none of {sorted(only)} found in {src}")

    ref = find_ref(kind)
    ref_url = as_data_url(ref, 1536)[0] if ref else None
    prompt = (PROMPT_WITH_REF if ref else PROMPT_NO_REF).format(body=BODY[kind])
    say(f"\n━━ {kind} ━━ {len(todo)} photos → {out.relative_to(HERE)}/  "
        f"({'body example: ' + ref.name if ref else 'no body example, text only'})")

    cost = 0.0
    failed = []
    done = 0

    def one(p):
        target = out / (p.stem + ".png")
        cap = p.with_suffix(".txt")
        if cap.exists():
            shutil.copy2(cap, out / cap.name)
        if target.exists():
            return p, "skip", 0.0, None
        if not test and number_of(p) in KEEP_AS_IS:
            shutil.copy2(p, target) if p.suffix.lower() == ".png" else _to_png(p, target)
            return p, "kept", 0.0, None
        photo_url, (w, h) = as_data_url(p, 2048)
        payload = {
            "prompt": prompt,
            "images": ([ref_url] if ref_url else []) + [photo_url],
            "resolution": RESOLUTION,
            "aspect_ratio": nearest_ratio(w, h),
            "output_format": "png",
        }
        err = None
        for attempt in range(1, TRIES + 1):
            try:
                data, c = run_job(key, payload)
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
            extra = {"skip": "already done", "kept": "copied unchanged", "fail": err, "ok": f"${c:.2f}"}[state]
            say(f"  {icon} {done:>2}/{len(todo)}  {p.stem}  {extra}")
            if state == "fail":
                failed.append(p.stem)

    say(f"━━ {kind} done · spent ${cost:.2f}" + (f" · ❌ failed: {', '.join(failed)} (run again to retry)" if failed else ""))
    return cost, failed


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
    ap.add_argument("--test", metavar="NUMS", help="only these photo numbers, into _test/<type>/  e.g. 5,22,41")
    a = ap.parse_args()

    kinds = list(BODY) if a.type == "all" else [a.type]
    for k in kinds:
        if k not in BODY:
            sys.exit(f"🛑 unknown type '{k}'. Pick from: {', '.join(BODY)}, all")
    if not (HERE / SOURCE).is_dir() or not photos(HERE / SOURCE):
        sys.exit(f"🛑 put the athletic photos in {HERE / SOURCE}/ first (image_01.png ... image_50.png)")

    ensure_pillow()
    key = os.environ.get("RUNPOD_API_KEY") or getpass.getpass("RunPod API key (hidden, not saved): ").strip()
    if not key:
        sys.exit("🛑 no API key")

    only = {int(x) for x in a.test.split(",") if x.strip()} if a.test else None
    total, all_failed = 0.0, {}
    for k in kinds:
        c, f = make_type(k, key, only=only, test=bool(only))
        total += c
        if f:
            all_failed[k] = f
    if len(kinds) > 1:
        say(f"\n🎉 all done · total ${total:.2f}" + (f" · failed: {all_failed}" if all_failed else ""))


if __name__ == "__main__":
    main()
