# AI Empire · Dataset Maker pod
# ComfyUI + our nodes + workflow. Models download on first boot into /workspace
# (attach a network volume so they only download once).
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential \
        git aria2 wget curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && pip install --upgrade pip wheel setuptools

# PyTorch with CUDA 12.8 (works on RTX 4090, 5090, RTX 6000 Blackwell, L40S, A100, H100)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

ARG COMFY_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
    && cd /opt/ComfyUI && git checkout ${COMFY_REF} \
    && pip install -r requirements.txt

# Google SDK for the Nano Banana node
RUN pip install google-genai

# our nodes only - NO workflows in the image (members get the .json files from the course)
COPY __init__.py nodes.py /opt/ComfyUI/custom_nodes/aiempire-dataset-maker/
COPY presets /opt/ComfyUI/custom_nodes/aiempire-dataset-maker/presets

COPY docker/extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8188
CMD ["/start.sh"]
