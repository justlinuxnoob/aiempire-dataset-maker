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

# Google SDK for the Nano Banana node + OpenCV for the skin-oil node
RUN pip install google-genai opencv-python-headless

# JupyterLab on port 8888: file browser (drag photos in, download datasets) + terminal
RUN pip install jupyterlab

# public custom nodes used by the Krea2 workflow (pinned so updates can't break it)
RUN cd /opt/ComfyUI/custom_nodes \
    && git clone https://github.com/rgthree/rgthree-comfy.git && git -C rgthree-comfy checkout 449c58fcdd612f7733e54c51f6758ead63fa180b \
    && git clone https://github.com/aiorbust-jy/comfyui-skin-detailer.git && git -C comfyui-skin-detailer checkout 6942c8dd3a1e3e939149b5c2e2193607e7a39df2 \
    && git clone https://github.com/aiorbust-jy/comfyui-realism-injector.git && git -C comfyui-realism-injector checkout 84aeb2486919a4355c1687486735da36d0204312 \
    && git clone https://github.com/Elevenheights/ComfyUI-FameGridColorFinish.git && git -C ComfyUI-FameGridColorFinish checkout 944c14e53fffee3e9e6dbedad36ab95b0e1629f7

# our nodes only - NO workflows in the image (members get the .json files from the course)
COPY __init__.py nodes.py photo_nodes.py /opt/ComfyUI/custom_nodes/aiempire-dataset-maker/
COPY presets /opt/ComfyUI/custom_nodes/aiempire-dataset-maker/presets
COPY js /opt/ComfyUI/custom_nodes/aiempire-dataset-maker/js

COPY docker/extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8188
CMD ["/start.sh"]
