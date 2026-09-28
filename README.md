# AI Empire · Dataset Maker

ComfyUI nodes + a workflow that turn **one face photo** into a full **LoRA training dataset**: 40 varied shots (close-ups, half-body, full-body, different angles, outfits, places and light), each saved with a matching `.txt` caption.

Runs 100% open-source on your own GPU. No API keys, no paid services.

- **Model:** Qwen Image Edit 2511 + Lightning 4-step LoRA (Apache 2.0, commercial use OK)
- **Output:** `ComfyUI/output/datasets/<name>/<name>_001.png` + `<name>_001.txt` … and `<name>.zip`
- **Captions:** `trigger word, framing, angle, expression, outfit, place, light`. They never describe the face or hair, so the LoRA ties the face to the trigger word.

## Install (RunPod / any ComfyUI)
1. ComfyUI → **Manager** → **Install via Git URL** → paste this repo's URL → restart ComfyUI.
2. Drag `workflows/AI_Empire_Dataset_Maker_Qwen2511.json` into ComfyUI.
3. Download any missing models it lists (links are in the READ ME note inside the workflow).

## GPU
- RTX 4090 / 5090 (24–32 GB): keep UNet `weight_dtype = fp8_e4m3fn`.
- 48 GB+ (L40S, A6000, RTX 6000): set it to `default` for best quality.

## Your own presets
Fill the **custom_presets** box, one shot per line:
```
standing on a bridge at night, black coat | full body, bridge, night, black coat | tall
```
Format: `shot description | caption | square / portrait / tall`. Caption and size are optional.

## Nodes
- **AI Empire · Dataset Presets:** trigger word, preset set, extra description (what must stay the same), how many, start at, seed.
- **AI Empire · Save Dataset:** saves images + captions, numbers them, zips the folder.
