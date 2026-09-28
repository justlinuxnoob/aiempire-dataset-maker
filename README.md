# AI Empire · Dataset Maker

ComfyUI nodes + a workflow that turn **one face photo** into a full **LoRA training dataset**: 40 varied shots (close-ups, half-body, full-body, different angles, outfits, places and light), each saved with a matching `.txt` caption.

Runs 100% open-source on your own GPU. No API keys, no paid services.

- **Models:** FireRed Image Edit 1.1 (default, best identity consistency) or Qwen Image Edit 2511, each with its Lightning 8-step LoRA. Both Apache 2.0, commercial use OK
- **Output:** `ComfyUI/output/datasets/<name>/<name>_001.png` + `<name>_001.txt` … and `<name>.zip`
- **Captions:** `trigger word, framing, angle, expression, outfit, place, light`. They never describe the face or hair, so the LoRA ties the face to the trigger word.

## Install (RunPod / any ComfyUI)
1. ComfyUI → **Manager** → **Install via Git URL** → paste this repo's URL → restart ComfyUI.
2. Drag `workflows/AI_Empire_Dataset_Maker_FireRed11.json` (or `_Qwen2511.json`) into ComfyUI.
3. Download any missing models it lists (links are in the READ ME note inside the workflow).

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
- **AI Empire · Dataset Presets:** trigger word, preset set, extra description (what must stay the same), how many, start at, seed.
- **AI Empire · Save Dataset:** saves images + captions, numbers them, zips the folder.

## Our RunPod template (one click, nothing to install)
The image is built automatically by GitHub Actions on every push: `ghcr.io/justlinuxnoob/aiempire-dataset-maker:latest`

RunPod → **My Templates → New Template**:
- **Container image:** `ghcr.io/justlinuxnoob/aiempire-dataset-maker:latest`
- **Container disk:** 30 GB
- **Volume disk:** 100 GB, mounted at `/workspace` (models download here on first boot: ~52 GB for FireRed, ~30 GB for Qwen)
- **Expose HTTP ports:** `8188`
- **Env (optional):** `EDIT_MODEL` = `firered` (default) / `qwen` / `both`; `QWEN_PRECISION=bf16` for the full Qwen model

On boot the pod downloads the models (first time only), then starts ComfyUI. Open **Connect → HTTP 8188**, then **Workflows → AI_Empire_Dataset_Maker_FireRed11**.
Download the finished zip at `http://<pod-url>/view?filename=<name>.zip&subfolder=datasets&type=output`.
