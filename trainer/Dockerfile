# AI Empire · LoRA Trainer pod (Krea 2)
# The official AI Toolkit image (ostris/aitoolkit) + our helper:
#   - everything lives on /workspace (datasets, LoRAs, job list, model downloads)
#   - drop a dataset .zip from the Dataset Maker -> it unzips itself
#   - a Krea 2 job with our settings is created for every dataset (trigger word read from the captions)
#   - JupyterLab on 8888 for easy upload/download
# Built on the official image on purpose: RunPod hosts usually have its layers cached, so pods start fast.
FROM ostris/aitoolkit:latest

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/aitk/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Krea 2 support must be inside (it landed in AI Toolkit in 2026). If an older base
# ever comes back, update the AI Toolkit source to a commit that has it.
ARG AITK_COMMIT=ecee894
RUN if [ ! -d /app/ai-toolkit/extensions_built_in/diffusion_models/krea2 ]; then \
      echo "Base image has no Krea 2 -> updating AI Toolkit to ${AITK_COMMIT}" && \
      git clone https://github.com/ostris/ai-toolkit.git /tmp/aitk && cd /tmp/aitk && git checkout ${AITK_COMMIT} && \
      rsync -a --delete --exclude 'ui/node_modules' /tmp/aitk/ /app/ai-toolkit/ && rm -rf /tmp/aitk && \
      cd /app/ai-toolkit && pip install --no-cache-dir --break-system-packages -r requirements.txt && \
      cd ui && npm ci && npm run update_db && npm run build ; \
    fi && test -d /app/ai-toolkit/extensions_built_in/diffusion_models/krea2

RUN pip install --no-cache-dir --break-system-packages jupyterlab

COPY aiempire_helper.py /opt/aiempire/aiempire_helper.py
COPY start.sh /aiempire_start.sh
RUN chmod +x /aiempire_start.sh

EXPOSE 8675 8888
WORKDIR /
CMD ["/aiempire_start.sh"]
