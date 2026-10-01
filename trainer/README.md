# AI Empire · LoRA Trainer (Krea 2)

Image: `ghcr.io/justlinuxnoob/aiempire-dataset-maker:lora-trainer` (built by GitHub Actions on every push to `trainer/`).

It's the official AI Toolkit (Ostris) with our extras:
- 📦 Drop the Dataset Maker `.zip` in `datasets/` → it unzips itself.
- 🏋️ Every dataset gets a ready **Krea 2** job with our settings. The trigger word is read from the captions, so you just press **Start**.
- 💾 Everything is on `/workspace/aitk` (datasets, LoRAs, the job list, model downloads), so a restart loses nothing.
- ⬇️ Krea 2 Raw downloads in the background as soon as a Hugging Face token is set.

## Job settings (made automatically)
Krea 2 Raw · LoRA rank 64 / alpha 64 · 3000 steps · lr 1e-4 · timestep sigmoid · resolution 1024 · quantization off · Low VRAM off · layer offloading off · saves every 500 steps (all 6 kept, so you can compare 1500 vs 2000 vs 3000) · 4 sample images every 500 steps.
Train on **Raw**, generate on **Turbo** (or our Krea2 workflow).

Optional template env vars: `AIEMPIRE_STEPS` (default 3000), `AIEMPIRE_RANK` (default 64). Every setting can also be changed in the job before pressing Start.

## RunPod template (My Templates → New Template)
| Field | Value |
|---|---|
| Name | AI Empire · LoRA Trainer (Krea 2) |
| Type | Pod |
| Container image | `ghcr.io/justlinuxnoob/aiempire-dataset-maker:lora-trainer` |
| Container disk | 30 GB |
| Volume disk | 100 GB, mounted at `/workspace` |
| Expose HTTP ports | `8675,8888` (8675 = AI Toolkit, 8888 = JupyterLab) |
| Env `HF_TOKEN` | leave empty in the template; each student pastes their own when deploying |
| Env `AI_TOOLKIT_AUTH` | leave empty; students set a password (without one, anyone with the pod URL can open it) |
| Env `JUPYTER_PASSWORD` | same idea, for JupyterLab |
| Visibility | **Public** (needed for the creator 1%) |

Share it as `https://console.runpod.io/deploy?template=<TEMPLATE_ID>&ref=9s65jq8z`.

GPU: **RTX PRO 6000 (96 GB)** or **H100 / A100 80 GB**. On 48 GB cards, turn quantization on in the job first.
The image uses CUDA 13 wheels, so the host needs a recent driver. If the pod log says the CUDA driver is too old, deploy again with the CUDA version filter set to 13.0.

## Student steps
1. Hugging Face: open huggingface.co/krea/Krea-2-Raw → **Agree**. Then Settings → Access Tokens → new **Read** token.
2. Deploy the template → pick the GPU → paste the token into `HF_TOKEN` (+ the two passwords) → Deploy.
3. Port **8888** (JupyterLab) → drag your dataset `.zip` into `datasets/`. Or use AI Toolkit → Datasets → upload.
4. Port **8675** (AI Toolkit) → Jobs → `<your dataset>_krea2` → ▶ **Start**.
5. Check the sample images every 500 steps. When it's done, download the LoRA from `output/<job>/` (JupyterLab or the job page).
6. Stop **and terminate** the pod.

Logs: `/workspace/aitk/aiempire.log` (our helper), and the job page in AI Toolkit.
