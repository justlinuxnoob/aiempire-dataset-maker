#!/bin/bash
# AI Empire · LoRA Trainer pod start
# Everything a student makes lives in /workspace/aitk so it survives a pod restart.
WS=/workspace/aitk
mkdir -p "$WS/datasets" "$WS/output" "$WS/hf"
AITK=/app/ai-toolkit

echo "[AI Empire] 🏋️ LoRA Trainer (Krea 2) starting"

# job list / settings database -> on the volume
if [ ! -f "$WS/aitk_db.db" ] && [ -f "$AITK/aitk_db.db" ] && [ ! -L "$AITK/aitk_db.db" ]; then
  cp "$AITK/aitk_db.db" "$WS/aitk_db.db"
fi
ln -sfn "$WS/aitk_db.db" "$AITK/aitk_db.db"
(cd "$AITK/ui" && npx prisma db push --skip-generate >/dev/null 2>&1) || echo "[AI Empire] ⚠ database check skipped"

# JupyterLab on 8888: drag your dataset .zip into datasets/, download LoRAs from output/
# Set JUPYTER_PASSWORD on the template to protect it; empty = no password.
nohup jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=8888 \
  --ServerApp.token="${JUPYTER_PASSWORD:-}" --ServerApp.password="" \
  --ServerApp.allow_origin='*' --ServerApp.root_dir="$WS" \
  --FileContentsManager.delete_to_trash=False \
  > "$WS/jupyter.log" 2>&1 &

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[AI Empire] ⚠ No HF_TOKEN set. Krea 2 Raw needs one: huggingface.co/krea/Krea-2-Raw -> Agree,"
  echo "             then huggingface.co/settings/tokens -> Read token -> paste it in AI Toolkit -> Settings."
fi

# our helper: settings, zip unpacking, one Krea 2 job per dataset, model pre-download
nohup python3 /opt/aiempire/aiempire_helper.py > "$WS/aiempire.log" 2>&1 &

echo "[AI Empire] ✅ AI Toolkit on port 8675 · JupyterLab on port 8888"
# the official AI Toolkit start (SSH + UI in the foreground)
exec /start.sh
