#!/bin/bash
# AI Empire · Dataset Maker pod start-up
#  1. downloads the models into /workspace/models (skipped if already there)
#  2. starts ComfyUI on port 8188
set -e

WS=/workspace
M=$WS/models
mkdir -p "$M/diffusion_models" "$M/loras" "$M/text_encoders" "$M/vae" "$M/checkpoints" "$WS/output" "$WS/input"

HF=https://huggingface.co

fetch() {
  local dir="$1" name="$2" url="$3"
  if [ -s "$dir/$name" ] && [ ! -f "$dir/$name.aria2" ]; then
    echo "[AI Empire] ✔ $name already downloaded"
    return
  fi
  echo "[AI Empire] ⬇ downloading $name ..."
  aria2c -x 16 -s 16 -k 1M -c --console-log-level=warn --summary-interval=20 \
    -d "$dir" -o "$name" "$url"
}

# QWEN_PRECISION=fp8 (default, 20 GB, fits 24-32 GB GPUs) or bf16 (41 GB, for 48 GB+ GPUs)
if [ "${QWEN_PRECISION:-fp8}" = "bf16" ]; then
  UNET=qwen_image_edit_2511_bf16.safetensors
else
  UNET=qwen_image_edit_2511_fp8mixed.safetensors
fi

echo "[AI Empire] Checking models (first boot takes a few minutes)..."
fetch "$M/diffusion_models" "$UNET" \
  "$HF/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/$UNET"
fetch "$M/loras" "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" \
  "$HF/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
fetch "$M/text_encoders" "qwen_2.5_vl_7b_fp8_scaled.safetensors" \
  "$HF/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
fetch "$M/vae" "qwen_image_vae.safetensors" \
  "$HF/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"

echo "[AI Empire] ✅ Models ready. Starting ComfyUI on port 8188..."
cd /opt/ComfyUI
exec python main.py --listen 0.0.0.0 --port 8188 \
  --output-directory "$WS/output" --input-directory "$WS/input" ${COMFY_ARGS}
