#!/bin/bash
# AI Empire · Dataset Maker pod start-up
#  1. downloads the models into /workspace/models (skipped if already there)
#  2. starts ComfyUI on port 8188
set -e

WS=/workspace
M=$WS/models
mkdir -p "$M/diffusion_models" "$M/loras" "$M/text_encoders" "$M/vae" "$M/checkpoints" "$M/upscale_models" "$WS/output" "$WS/input"

HF=https://huggingface.co

# JupyterLab on port 8888 (starts first, so you can upload photos while the models download)
# Set JUPYTER_PASSWORD on the template to protect it (RunPod convention); empty = no password.
mkdir -p "$WS/input/templates"
nohup jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=8888 \
  --ServerApp.token="${JUPYTER_PASSWORD:-}" --ServerApp.password="" \
  --ServerApp.allow_origin='*' --ServerApp.root_dir="$WS" \
  --FileContentsManager.delete_to_trash=False \
  > "$WS/jupyter.log" 2>&1 &
echo "[AI Empire] 📁 JupyterLab on port 8888 (templates go in input/templates/<set>/)"

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

# No env vars needed: by default the pod sets up the Qwen-Image 2.1 dataset maker + Z-Image realism pass.
# EDIT_MODEL = qwen21 (default) | firered | qwen (2511) | both (firered + qwen 2511) | none (Nano Banana only)
# REALISM = 1 (default: download Z-Image Turbo for the realism pass) | 0
# QWEN_PRECISION = fp8 (default, 20 GB) | bf16 (41 GB)   -- only used for the Qwen model
# PORTRAIT = 1 -> portrait generator only (Z-Image Turbo + instagram LoRA, ~20 GB): nothing else downloads
if [ "${PORTRAIT:-0}" = "1" ]; then EDIT_MODEL="${EDIT_MODEL:-none}"; REALISM="${REALISM:-0}"; fi
EDIT_MODEL="${EDIT_MODEL:-qwen21}"
echo "[AI Empire] Checking models for EDIT_MODEL=$EDIT_MODEL (first boot takes a few minutes)..."

if [ "$EDIT_MODEL" != "none" ] && [ "$EDIT_MODEL" != "qwen21" ]; then
  # shared by FireRed and Qwen
  fetch "$M/text_encoders" "qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "$HF/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
  fetch "$M/vae" "qwen_image_vae.safetensors" \
    "$HF/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"
fi

# realism pass models: on by default for FireRed / Qwen 2511, off for Qwen-Image 2.1 (looks real on its own)
if [ "$EDIT_MODEL" = "qwen21" ]; then REALISM="${REALISM:-0}"; else REALISM="${REALISM:-1}"; fi
if [ "$REALISM" = "1" ]; then
  # Z-Image Turbo for the realism pass
  fetch "$M/diffusion_models" "z_image_turbo_bf16.safetensors" \
    "$HF/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
  fetch "$M/text_encoders" "qwen_3_4b.safetensors" \
    "$HF/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
  fetch "$M/vae" "ae.safetensors" \
    "$HF/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"
fi

if [ "$EDIT_MODEL" = "firered" ] || [ "$EDIT_MODEL" = "both" ]; then
  fetch "$M/diffusion_models" "FireRed-Image-Edit-1.1-transformer.safetensors" \
    "$HF/FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI/resolve/main/FireRed-Image-Edit-1.1-transformer.safetensors"
  fetch "$M/loras" "FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors" \
    "$HF/FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI/resolve/main/FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors"
fi

if [ "$EDIT_MODEL" = "qwen" ] || [ "$EDIT_MODEL" = "both" ]; then
  if [ "${QWEN_PRECISION:-fp8}" = "bf16" ]; then
    UNET=qwen_image_edit_2511_bf16.safetensors
  else
    UNET=qwen_image_edit_2511_fp8mixed.safetensors
  fi
  fetch "$M/diffusion_models" "$UNET" \
    "$HF/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/$UNET"
  fetch "$M/loras" "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors" \
    "$HF/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
fi

# QWEN21 = 1 -> Qwen-Image 2.1 edit (test workflow), ~30 GB
if [ "${QWEN21:-0}" = "1" ] || [ "$EDIT_MODEL" = "qwen21" ]; then
  Q21="$HF/Comfy-Org/Qwen-Image-2.1/resolve/main"
  fetch "$M/diffusion_models" "qwen_image_2.1_int8_convrot.safetensors" "$Q21/diffusion_models/qwen_image_2.1_int8_convrot.safetensors"
  fetch "$M/text_encoders" "qwen3vl_8b_int8_convrot.safetensors" "$Q21/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
  fetch "$M/vae" "qwen_image_2.1_vae_bf16.safetensors" "$Q21/vae/qwen_image_2.1_vae_bf16.safetensors"
  # BFS Head Swap v1.1 LoRA (MIT) - used by the Qwen 2.1 template workflow
  fetch "$M/loras" "bfs_head_v1.1_qwen_2.1.safetensors" "$HF/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1.1_qwen_2.1.safetensors"
  # prompt-enhancer encoder: only the Qwen 2.1 TEST workflow uses it, the dataset workflows don't (saves ~10 GB)
  if [ "${QWEN21:-0}" = "1" ]; then
    fetch "$M/text_encoders" "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors" "$Q21/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"
  fi
fi

# KREA2 = 1 -> Krea2 RAW workflow: RawGirl Krea2 + Flux 2 Klein 9B realism pass + skin detailer (~60 GB)
if [ "${KREA2:-0}" = "1" ]; then
  D="$HF/dci05049"
  fetch "$M/diffusion_models" "RawGirlKrea2_v10_int8_convrot.safetensors" "$D/krea2/resolve/main/RawGirlKrea2_v10_int8_convrot.safetensors"
  fetch "$M/text_encoders" "qwen3vl_4b_bf16.safetensors" "$HF/Comfy-Org/Qwen3-VL/resolve/b58e627c376915e49cb6bba978416085aa31767f/text_encoders/qwen3vl_4b_bf16.safetensors"
  fetch "$M/vae" "wan_2.1_vae.safetensors" "$D/wan-animate/resolve/main/wan_2.1_vae.safetensors"
  fetch "$M/diffusion_models" "flux-2-klein-9b.safetensors" "$D/flux2-klein-9b/resolve/main/flux-2-klein-9b.safetensors"
  fetch "$M/text_encoders" "qwen_3_8b_fp8mixed.safetensors" "$HF/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"
  fetch "$M/vae" "flux2-vae.safetensors" "$D/flux2-klein-9b/resolve/main/flux2-vae.safetensors"
  fetch "$M/checkpoints" "sam3.1_multiplex_fp16.safetensors" "$HF/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"
  fetch "$M/upscale_models" "1xSkinContrast-High-SuperUltraCompact.pth" "$D/spicy-sdxl/resolve/main/1xSkinContrast-High-SuperUltraCompact.pth"
  fetch "$M/upscale_models" "1x-ITF-SkinDiffDetail-Lite-v1.pth" "$D/krea2/resolve/main/1x-ITF-SkinDiffDetail-Lite-v1.pth"
  # Krea2 LoRAs
  # (no refusal-reduction / NSFW LoRAs: they break the Krea 2 licence, and this image is a public template)
  for f in candid_krea2_loraholic skindetails_krea2_loraholic real_3d_krea2_loraholic \
           ass_v2_krea2_loraholic breast_size_v2_krea2_loraholic Krea2-realism-V1 RealisticSnapshotKrea2 \
           bloomgirls-ultrarealism-krea2_4k lenovo_krea2 SummerVibesHM_krea2_epoch8 RawGirlV2_epoch_10; do
    fetch "$M/loras" "$f.safetensors" "$D/krea2/resolve/main/$f.safetensors"
  done
  fetch "$M/loras" "michelle_krea_2_000003000.safetensors" "$D/krea2/resolve/main/michelle%20krea%202_000003000.safetensors"
  # Klein LoRAs
  for f in f2k_9B_lcs_consist_20260415 Samsung_fluxklein9b Klein_realistic_I2I HighResolution9B; do
    fetch "$M/loras" "$f.safetensors" "$D/flux2-klein-9b/resolve/main/$f.safetensors"
  done
fi

# PORTRAIT = 1 -> portrait-gen.json: Z-Image Turbo bf16 + qwen_3_4b + ae + instagram_zimageturbo LoRA (file names = the workflow's)
if [ "${PORTRAIT:-0}" = "1" ]; then
  ZI="$HF/Comfy-Org/z_image_turbo/resolve/main/split_files"
  if [ -s "$M/diffusion_models/z_image_turbo_bf16.safetensors" ] && [ ! -e "$M/diffusion_models/z_image_turbo.safetensors" ]; then
    ln -s z_image_turbo_bf16.safetensors "$M/diffusion_models/z_image_turbo.safetensors"
  fi
  # all four at once (faster first boot); a failed download stops the pod with a clear message
  PIDS=""
  fetch "$M/diffusion_models" "z_image_turbo.safetensors" "$ZI/diffusion_models/z_image_turbo_bf16.safetensors" & PIDS="$PIDS $!"
  fetch "$M/text_encoders" "qwen_3_4b.safetensors" "$ZI/text_encoders/qwen_3_4b.safetensors" & PIDS="$PIDS $!"
  fetch "$M/vae" "ae.safetensors" "$ZI/vae/ae.safetensors" & PIDS="$PIDS $!"
  fetch "$M/loras" "instagram_zimageturbo.safetensors" "$HF/datasets/pofkeb/instagramification/resolve/main/instagram_zimageturbo.safetensors" & PIDS="$PIDS $!"
  for p in $PIDS; do wait "$p" || { echo "[AI Empire] 🛑 a portrait model failed to download, restart the pod to retry"; exit 1; }; done
  echo "[AI Empire] 📸 Portrait generator ready: drag portrait-gen.json into ComfyUI"
fi

# VIDEO = minimax  -> MiniMax H3 image-to-video + reference-to-video (~75 GB extra, use a 200 GB volume)
if [ "${VIDEO:-}" = "minimax" ]; then
  H3="$HF/Comfy-Org/MiniMax-H3/resolve/main"
  fetch "$M/diffusion_models" "minimax_h3_fl2va_pruned_int8_convrot.safetensors" "$H3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  fetch "$M/diffusion_models" "minimax_h3_ref2va_pruned_int8_convrot.safetensors" "$H3/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
  fetch "$M/text_encoders" "qwen3vl_32b_minimax_h3_int8_convrot.safetensors" "$H3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
  fetch "$M/vae" "minimax_h3_video_vae_fp16.safetensors" "$H3/vae/minimax_h3_video_vae_fp16.safetensors"
  fetch "$M/vae" "minimax_h3_audio_vae_fp32.safetensors" "$H3/vae/minimax_h3_audio_vae_fp32.safetensors"
fi

echo "[AI Empire] ✅ Models ready. Starting ComfyUI on port 8188..."
cd /opt/ComfyUI
exec python main.py --listen 0.0.0.0 --port 8188 \
  --output-directory "$WS/output" --input-directory "$WS/input" ${COMFY_ARGS}
