"""AI Empire · LoRA Trainer helper (runs next to the AI Toolkit UI).

1. Points AI Toolkit at /workspace/aitk (datasets, LoRAs) and copies HF_TOKEN into its settings.
2. A dataset .zip dropped in datasets/ (e.g. straight from the Dataset Maker) is unzipped into its own folder.
3. Every dataset gets a ready Krea 2 job with our settings. The trigger word is read from the captions
   ("zvx woman, close-up selfie, ..." -> "zvx woman"). The student only presses Start.
4. Once a Hugging Face token is known, Krea 2 Raw is downloaded in the background so training starts sooner.
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile

WS = os.environ.get("AIEMPIRE_WS", "/workspace/aitk")
API = os.environ.get("AIEMPIRE_API", "http://127.0.0.1:8675")
DATASETS = os.path.join(WS, "datasets")
OUTPUT = os.path.join(WS, "output")
MARK = ".aiempire_job"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

# ---- our Krea 2 character-LoRA settings (Kajus's tested lineup) ----
MODEL = "krea/Krea-2-Raw"
STEPS = int(os.environ.get("AIEMPIRE_STEPS", "3000"))
RANK = int(os.environ.get("AIEMPIRE_RANK", "64"))
SAMPLE_PROMPTS = [
    "[trigger], close-up selfie, looking at the camera, soft smile, cream knit sweater, cozy cafe, window daylight",
    "[trigger], full body, walking towards the camera, black dress, city street at night, neon lights",
    "[trigger], half body, three-quarter angle, relaxed expression, white linen shirt, beach at sunset, golden hour",
    "[trigger], medium shot, sitting on a sofa, laughing, grey hoodie, living room, warm lamp light",
]


def log(*a):
    print("[AI Empire]", *a, flush=True)


def api(path, body=None):
    req = urllib.request.Request(API + path, method="POST" if body is not None else "GET")
    req.add_header("Content-Type", "application/json")
    auth = os.environ.get("AI_TOOLKIT_AUTH", "").strip()
    if auth:
        req.add_header("Authorization", "Bearer " + auth)
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(req, data=data, timeout=30) as r:
        return json.loads(r.read() or b"{}")


def wait_for_ui():
    for _ in range(600):
        try:
            api("/api/settings")
            return True
        except Exception:
            time.sleep(2)
    log("AI Toolkit UI never came up")
    return False


def setup_settings():
    s = api("/api/settings")
    token = os.environ.get("HF_TOKEN", "").strip() or s.get("HF_TOKEN", "")
    want = {
        "HF_TOKEN": token,
        "TRAINING_FOLDER": OUTPUT,
        "DATASETS_FOLDER": DATASETS,
        "MODELS_PATH": s.get("MODELS_PATH", "") or "",
    }
    if any(s.get(k) != v for k, v in want.items()):
        api("/api/settings", want)
        log("settings: datasets ->", DATASETS, "| LoRAs ->", OUTPUT, "| HF token", "set" if token else "NOT set")


def current_token():
    try:
        return (api("/api/settings").get("HF_TOKEN") or "").strip()
    except Exception:
        return os.environ.get("HF_TOKEN", "").strip()


# ---------------------------------------------------------------- zips
def safe_name(name):
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return name or "dataset"


def unpack_zips(sizes):
    """Unzip finished uploads. A zip is only touched once its size stopped changing."""
    for f in sorted(os.listdir(DATASETS)):
        if not f.lower().endswith(".zip"):
            continue
        p = os.path.join(DATASETS, f)
        size = os.path.getsize(p)
        if sizes.get(p) != size:  # still uploading (or first look)
            sizes[p] = size
            continue
        dest = os.path.join(DATASETS, safe_name(os.path.splitext(f)[0]))
        n = 0
        try:
            with zipfile.ZipFile(p) as z:
                os.makedirs(dest, exist_ok=True)
                for m in z.infolist():
                    base = os.path.basename(m.filename)
                    if m.is_dir() or not base or base.startswith(".") or "__MACOSX" in m.filename:
                        continue
                    if not base.lower().endswith(IMG_EXT + (".txt",)):
                        continue
                    with z.open(m) as src, open(os.path.join(dest, base), "wb") as out:
                        out.write(src.read())
                    n += 1
        except zipfile.BadZipFile:
            log(f"{f}: not a finished zip yet, will retry")
            continue
        os.remove(p)
        sizes.pop(p, None)
        log(f"📦 {f} -> datasets/{os.path.basename(dest)} ({n} files)")


# ---------------------------------------------------------------- jobs
def read_trigger(folder):
    for f in sorted(os.listdir(folder)):
        if f.endswith(".txt"):
            try:
                text = open(os.path.join(folder, f), encoding="utf-8", errors="ignore").read().strip()
            except OSError:
                continue
            trig = text.split(",")[0].strip()
            if 0 < len(trig) <= 40:
                return trig
    return None


def job_config(name, folder, trigger):
    return {
        "job": "extension",
        "config": {
            "name": name,
            "process": [{
                "type": "diffusion_trainer",
                "training_folder": OUTPUT,
                "sqlite_db_path": "./aitk_db.db",
                "device": "cuda",
                "trigger_word": trigger,
                "performance_log_every": 10,
                "network": {
                    "type": "lora", "linear": RANK, "linear_alpha": RANK,
                    "lokr_full_rank": True, "lokr_factor": -1,
                    "network_kwargs": {"ignore_if_contains": []},
                },
                "save": {
                    "dtype": "bf16", "save_every": 500, "max_step_saves_to_keep": 6,
                    "save_format": "diffusers", "push_to_hub": False,
                },
                "datasets": [{
                    "folder_path": folder, "mask_path": None, "mask_min_value": 0.1,
                    "default_caption": "", "caption_ext": "txt", "caption_dropout_rate": 0.05,
                    "cache_latents_to_disk": True, "is_reg": False, "network_weight": 1,
                    "resolution": [1024], "controls": [], "shrink_video_to_frames": True,
                    "num_frames": 1, "flip_x": False, "flip_y": False, "num_repeats": 1,
                }],
                "train": {
                    "batch_size": 1, "bypass_guidance_embedding": True, "steps": STEPS,
                    "gradient_accumulation": 1, "train_unet": True, "train_text_encoder": False,
                    "gradient_checkpointing": True, "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit", "timestep_type": "sigmoid", "content_or_style": "balanced",
                    "optimizer_params": {"weight_decay": 1e-4}, "unload_text_encoder": False,
                    "cache_text_embeddings": False, "lr": 1e-4,
                    "ema_config": {"use_ema": False, "ema_decay": 0.99},
                    "skip_first_sample": False, "force_first_sample": False, "disable_sampling": False,
                    "dtype": "bf16", "diff_output_preservation": False,
                    "diff_output_preservation_multiplier": 1.0, "diff_output_preservation_class": "person",
                    "switch_boundary_every": 1, "loss_type": "mse",
                },
                "logging": {"log_every": 1, "use_ui_logger": True},
                "model": {
                    "name_or_path": MODEL, "arch": "krea2",
                    "quantize": False, "qtype": "qfloat8", "quantize_te": False, "qtype_te": "qfloat8",
                    "low_vram": False, "layer_offloading": False,
                    "model_kwargs": {}, "compile": False,
                },
                "sample": {
                    "sampler": "flowmatch", "sample_every": 500, "sample_start_step": 0,
                    "width": 1024, "height": 1280,
                    "samples": [{"prompt": p} for p in SAMPLE_PROMPTS],
                    "neg": "", "seed": 42, "walk_seed": True,
                    "guidance_scale": 4, "sample_steps": 30, "num_frames": 1, "fps": 1,
                },
            }],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }


def make_jobs():
    try:
        existing = {j["name"] for j in api("/api/jobs").get("jobs", [])}
    except Exception as e:
        log("can't read jobs:", e)
        return
    for d in sorted(os.listdir(DATASETS)):
        folder = os.path.join(DATASETS, d)
        if d.startswith(".") or not os.path.isdir(folder) or os.path.exists(os.path.join(folder, MARK)):
            continue
        files = os.listdir(folder)
        if not any(f.lower().endswith(IMG_EXT) for f in files) or not any(f.endswith(".txt") for f in files):
            continue
        trigger = read_trigger(folder)
        base = safe_name(d) + "_krea2"
        name, i = base, 2
        while name in existing:
            name, i = f"{base}_{i}", i + 1
        try:
            api("/api/jobs", {"name": name, "gpu_ids": "0", "job_config": job_config(name, folder, trigger)})
        except Exception as e:
            log(f"could not create the job for {d}:", e)
            continue
        existing.add(name)
        with open(os.path.join(folder, MARK), "w") as f:
            f.write(name)
        log(f"🏋️ job '{name}' ready (trigger word: {trigger or 'none found, captions used as they are'}) -> press Start")


# ---------------------------------------------------------------- model pre-download
def predownload():
    started = False
    while not started:
        token = current_token()
        if not token:
            time.sleep(20)
            continue
        started = True
        try:
            import huggingface_hub as hh
            log("⬇️ downloading Krea 2 Raw + Qwen3-VL in the background (first time only, ~35 GB)")
            hh.hf_hub_download(MODEL, "raw.safetensors", token=token)
            hh.snapshot_download("Qwen/Qwen3-VL-4B-Instruct", token=token)
            hh.snapshot_download("Qwen/Qwen-Image", allow_patterns=["vae/*"], token=token)
            log("✅ models downloaded")
        except Exception as e:
            msg = str(e)
            if "401" in msg or "403" in msg or "gated" in msg.lower() or "GatedRepo" in type(e).__name__:
                log("🛑 Hugging Face refused the download. Open huggingface.co/krea/Krea-2-Raw, click Agree,"
                    " and use a token with Read access (AI Toolkit -> Settings).")
                started = False
                time.sleep(60)
            else:
                log("download will happen when training starts instead:", msg[:300])


def main():
    os.makedirs(DATASETS, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)
    if not wait_for_ui():
        return
    try:
        setup_settings()
    except Exception as e:
        log("settings not saved:", e)
    if os.environ.get("AIEMPIRE_NO_DOWNLOAD") != "1":
        threading.Thread(target=predownload, daemon=True).start()
    sizes = {}
    log("watching datasets/ (drop a dataset .zip there)")
    while True:
        try:
            unpack_zips(sizes)
            make_jobs()
        except Exception as e:
            log("helper error:", e)
        time.sleep(float(os.environ.get("AIEMPIRE_POLL", "10")))


if __name__ == "__main__":
    main()
