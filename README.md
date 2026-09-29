# AI Empire · Dataset Maker

ComfyUI nodes + a workflow that turn **one face photo** into a full **LoRA training dataset**: 40 varied shots (close-ups, half-body, full-body, different angles, outfits, places and light), each saved with a matching `.txt` caption.

Runs 100% open-source on your own GPU. No API keys, no paid services.

- **Models:** FireRed Image Edit 1.1 (default, best identity consistency) or Qwen Image Edit 2511, each with its Lightning 8-step LoRA. Both Apache 2.0, commercial use OK
- **Qwen-Image 2.1 version** (`AI_Empire_Dataset_Maker_Qwen21.json`, pod env `QWEN21=1`): best open edit model, native 2K, 25 steps, images at 1.5x size
- **Output:** `ComfyUI/output/datasets/<name>/<name>_001.png` + `<name>_001.txt` … and `<name>.zip`
- **Captions:** `trigger word, framing, angle, expression, outfit, place, light`. They never describe the face or hair, so the LoRA ties the face to the trigger word.

## Install (RunPod / any ComfyUI)
1. ComfyUI → **Manager** → **Install via Git URL** → paste this repo's URL → restart ComfyUI.
2. Drag `workflows/AI_Empire_Dataset_Maker_FireRed11.json` (or `_Qwen2511.json`) into ComfyUI.
3. Download any missing models it lists (links are in the READ ME note inside the workflow).

## Your own template photos (Qwen-Image 2.1)
`workflows/AI_Empire_Dataset_Maker_Qwen21_Templates.json`: upload ~50 photos, get ~50 dataset images. Each one keeps the photo's pose, outfit, place and light, with her face from **Your face**.
1. **Template Presets** box → **📁 Upload template photos** → select all photos (or one `.zip`) → name the set (e.g. `athletic`). Saved to `input/templates/<set>/`, so one folder per body type = your body presets.
2. Optional caption per photo: `swap_01.txt` next to `swap_01.jpg`, added after the trigger word.
3. `how_many = 3` to test, `0` = all. Finished images are skipped on re-runs (`skip_done`), so after a crash just press Run again.
4. Output: `datasets/<name>/<name>_<template>.png` + `.txt` + `<name>.zip`, same size/shape as each template (encoder *resolution* 1536 ≈ 2.3 MP).

## GPU
- FireRed (41 GB model): 80 GB GPU (A100 / H100) as is; on 48 GB (A6000, A40, L40S) set UNet `weight_dtype = fp8_e4m3fn`.
- Qwen fp8mixed (20 GB model): runs on 24–48 GB GPUs as is.

## Your own presets
Fill the **custom_presets** box, one shot per line:
```
standing on a bridge at night, black coat | full body, bridge, night, black coat | tall
```
Format: `shot description | caption | square / portrait / tall`. Caption and size are optional.

## Nodes
- **AI Empire · Nano Banana:** makes the dataset with Nano Banana Pro / Nano Banana 2 using your own Google key (Vertex AI or AI Studio). Refused shots are skipped.
- **AI Empire · Template Presets:** loops over a folder of your own template photos (upload button on the node), with resume.
- **AI Empire · Dataset Presets:** trigger word, preset set, extra description (what must stay the same), how many, start at, seed.
- **AI Empire · Save Dataset:** saves images + captions, numbers them, zips the folder.
- **AI Empire · Photo Finish:** phone-photo look in one node (levels, soft glow, lens softness, hand shake, ISO grain, JPEG).
- **AI Empire · Save Image (no workflow inside):** saves JPEG/PNG without the workflow embedded, so images can't be dragged into ComfyUI to copy it.

## Our RunPod template (one click, nothing to install)
The image is built automatically by GitHub Actions on every push: `ghcr.io/justlinuxnoob/aiempire-dataset-maker:latest`

RunPod → **My Templates → New Template**:
- **Container image:** `ghcr.io/justlinuxnoob/aiempire-dataset-maker:latest`
- **Container disk:** 30 GB
- **Volume disk:** 100 GB, mounted at `/workspace` (first boot downloads ~30 GB: Qwen-Image 2.1)
- **Expose HTTP ports:** `8188`
- **Env: none needed.** With no env vars the pod sets up the Qwen-Image 2.1 dataset maker. Optional extras:
  - `EDIT_MODEL` = `qwen21` (default) / `firered` / `qwen` (2511) / `both` / `none` (Nano Banana only)
  - `REALISM` = `1` downloads Z-Image Turbo for the realism pass (default on for FireRed / Qwen 2511, off for Qwen 2.1)
  - `QWEN_PRECISION=bf16` for the full Qwen model
  - `QWEN21=1` adds the Qwen-Image 2.1 edit test workflow (~30 GB)
  - `KREA2=1` adds the Krea2 RAW workflow: RawGirl Krea2 + Flux 2 Klein 9B realism pass + skin detailer + phone-look finish (~60 GB, 48 GB+ GPU). Set `EDIT_MODEL=none` + `REALISM=0` for a Krea2-only pod (100 GB volume is enough)
  - `VIDEO=minimax` adds MiniMax H3 video (image-to-video + reference-to-video, ~75 GB more: use a 200 GB volume, 48 GB+ GPU). Set `EDIT_MODEL=none` + `REALISM=0` for a video-only pod
  - Nano Banana: `VERTEX_SA_JSON` (whole service-account JSON, as a RunPod **secret**) or `GEMINI_API_KEY`

On boot the pod downloads the models (first time only), then starts ComfyUI. Open **Connect → HTTP 8188** and drag in the workflow `.json` (workflows are NOT inside the image; they're delivered separately).
Download the finished zip at `http://<pod-url>/view?filename=<name>.zip&subfolder=datasets&type=output`.
