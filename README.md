# AI Empire · Dataset Maker

**▶ New here? Start with the free video series: [joinaiempire.com/video](https://joinaiempire.com/video)** (video 1: make your first AI face).
Guides and the full course: https://joinaiempire.com

ComfyUI nodes and workflows that turn **one face photo** into a **character LoRA training dataset**: 50 photos (close-ups, half-body and full-body shots in different poses, outfits, places and light), each saved with a matching `.txt` caption. It runs in ComfyUI on a RunPod pod.

- **Engines:** you pick one in the Dataset Maker box.
  - **Qwen** (free): Qwen-Image 2.1 + BFS head swap, running on your pod's GPU. No API key.
  - **Nano Banana Pro** (paid, ~$0.14 per photo at 2K) or **Seedream** (paid, ~$0.03 per photo): run through RunPod's public API and are charged to your RunPod account. They need a RunPod API key, entered once with the **🔑 RunPod key** button. The prices are the ones in the engine list; RunPod sets them and they can change.
- **Template photos:** 6 body types × 50 photos, built into the pod. All of them are fully AI-generated.
- **Output:** `output/datasets/<name>/` (images + `.txt` captions) and `output/datasets/<name>.zip`.
- **Captions:** the trigger word, then the template photo's own caption: pose, outfit, place, light and framing. Example: `zvx woman, close-up portrait looking over her bare shoulder, head turned, plain grey wall, soft even light`. Captions never describe her face or hair, so the LoRA ties her face to the trigger word.

## The main workflow: `workflows/AI_Empire_Dataset_Maker.json`
One face photo + one settings box → a 50-photo dataset with captions. ComfyUI is on port 8188, JupyterLab on port 8888.

1. **1 · Your face**: upload a clear, front-facing photo of her.
2. **2 · Dataset Maker** box:
   - *body_type*: `athletic`, `curvy`, `petite`, `busty`, `thick` or `plus`. 50 AI-generated template photos each, with captions (`presets/templates/`). The box previews 3 photos at a time (◀ ▶ to browse). Click one to make it the test photo. Sets you upload to `input/templates/<set>/` show up in this list too.
   - *trigger_word*: default `zvx woman`. Every caption starts with it. Use the same word when you train.
   - *hair_and_eyes*: her hair and eye colour, e.g. `long straight blonde hair, blue eyes`. Empty = taken from your face photo. It only goes into the edit prompt, never into the captions.
   - *engine*: Qwen, Nano Banana Pro or Seedream (see above).
   - *dataset_name*: folder and zip name. New name = new dataset.
3. *mode* = **🧪 Test 1 photo** → **Run**. It makes one photo (the *test_photo*) with a new seed every time and shows it in the Save box. Tests are saved in `output/datasets/<name>_test/`, so they never mix into the dataset. Not happy? Run again, or click another photo in the preview.
4. *mode* = **🚀 Whole dataset (all photos)** → **Run** once. It makes one photo, saves it, then queues the next by itself until every photo is done. Progress shows on the Save box and in the browser tab title. You can close the tab: it keeps going on the pod. **Stop** = X in the queue. **Run** again continues where it stopped (finished photos are skipped).
5. Download `output/datasets/<name>.zip` with JupyterLab (port 8888). The zip is rewritten after every photo, so it always holds the whole dataset so far.

### Qwen engine (free)
Qwen-Image 2.1 + BFS head swap, 25 steps, CFG 1, euler / simple. Encoder *resolution* 1536 (about 2.3 MP). Each photo keeps its template's shape.

### Paid engines (Nano Banana Pro, Seedream)
- **🔑 RunPod key**: click it once and paste your key (runpod.io → Settings → API Keys). It's saved on the pod in `/workspace/.runpod_key`, never in the workflow file. Paste nothing to remove it. Env `AIEMPIRE_RUNPOD_KEY` works too. RunPod's own `RUNPOD_API_KEY` is ignored on purpose: that's the pod-scoped key RunPod puts in every pod, and it can't call the public endpoints.
- Size: Nano Banana Pro is asked for 2K at the aspect ratio closest to the template. Seedream is asked for 2048 px on the long side, same shape as the template.
- With a paid engine the Qwen part of the workflow doesn't run, so those photos don't use the pod's GPU.
- Each photo gets 3 tries. A job stuck in RunPod's queue for more than 4 minutes is cancelled and tried again.
- **Refused or failed photos** (whole dataset): the photo is skipped and the run moves on, so it never stops halfway. Skipped photos get one more try at the end. A photo that fails twice is left out, and the run lists them: switch engine (e.g. Qwen) and Run to make those, or train without them. Switching engine gives every photo a fresh start.
- In test mode a failure just shows the error: Run again, pick another test photo, or switch engine.
- A wrong key stops the run and asks you to paste it again.

## Krea 2 Turbo + local prompt writer: `workflows/AI_Empire_Krea2_Turbo_PromptWriter.json`
Your character LoRA on Krea 2 Turbo, with the prompt written on the pod itself. No API, no key. Pod: `KREA_TURBO=1`.
- **Models:** Krea 2 Turbo fp8, Qwen3-VL 4B fp8, Wan 2.1 VAE + your LoRA 0.9. No extra LoRAs.
- **Sampling:** shift 6, 8 steps, CFG 1, euler_ancestral, beta57 (core *BetaSamplingScheduler* at alpha 0.5 / beta 0.7, the same curve as RES4LYF's beta57, so no extra node pack).
- **Prompt Writer box:** uses the Krea 2 text encoder (Qwen3-VL 4B, a vision chat model that is already loaded) to write the prompt, through ComfyUI's own text generation. Nothing extra to download.
  - 📷 *photo → prompt*: copies a reference photo's shot (pose, outfit, place, light, camera).
  - 💡 *idea → prompt*: a short idea becomes a full prompt.
  - ✍️ *my prompt*: your text as is.
  - The trigger word always goes first, then the optional *hair_and_eyes* line. The writer never describes her face, hair, eyes or body.
  - Prompts always describe an adult woman: "girl" becomes "woman", and youth words or a reference photo of someone who could be under 18 stop the run.
  - *variation* on **fixed** = the prompt is written once, every Run reuses it with a new image seed.
- **Finish:** Photo Finish (phone look) + Save Image with no workflow inside. *Before the phone look* shows the raw render.
- Rebuild the file with `python tools/build_krea2_local.py`.

## Next step: train the LoRA
Use the LoRA Trainer template: [justlinuxnoob/lora-training](https://github.com/justlinuxnoob/lora-training), image `ghcr.io/justlinuxnoob/lora-training:latest`. Drop the dataset `.zip` into `datasets/` in JupyterLab and it unzips itself.

You set up the training job yourself in AI Toolkit (port 8675). The template no longer makes the job for you. Type the same trigger word you used here (e.g. `zvx woman`) into the job's **Trigger Word** field: it's empty by default.

Making new body presets: `tools/body_presets.py` (Nano Banana Pro, Nano Banana 2 or Seedream through the RunPod API, run on your own PC).

## Install (RunPod / any ComfyUI)
1. ComfyUI → **Manager** → **Install via Git URL** → paste this repo's URL → restart ComfyUI.
2. Drag `workflows/AI_Empire_Dataset_Maker.json` into ComfyUI.
3. Models: the main workflow uses the Qwen-Image 2.1 models and the BFS head swap LoRA. Our pod downloads them on first boot (file names and links are in `docker/start.sh`). The other workflows list their models in the READ ME note inside the workflow.

## Older workflows (40 presets)
`AI_Empire_Dataset_Maker_FireRed11.json`, `_Qwen2511.json`, `_Qwen21.json` and `_NanoBanana.json` use the **Dataset Presets** box: 40 written shot descriptions (`presets/core40.json`) instead of template photos.
- **Models:** FireRed Image Edit 1.1 or Qwen Image Edit 2511, each with its Lightning 8-step LoRA. Both Apache 2.0, commercial use OK
- **Qwen-Image 2.1 version** (`AI_Empire_Dataset_Maker_Qwen21.json`): uses the models the pod downloads by default. Native 2K, 25 steps, images at 1.5x size.
- **Output:** `output/datasets/<name>/<name>_001.png` + `<name>_001.txt` … and `<name>.zip`. FireRed and Qwen 2511 also save the edit model's raw images in `<name>_raw`.
- **Captions:** `trigger word, framing, angle, expression, outfit, place, light`. They never describe the face or hair.

## Your own template photos (Qwen-Image 2.1)
`workflows/AI_Empire_Dataset_Maker_Qwen21_Templates.json`: upload ~50 photos, get ~50 dataset images. Each one keeps the photo's pose, outfit, place and light, with her face from **Your face**.
1. **Template Presets** box → **📁 Upload template photos** → select all photos (or one `.zip`) → name the set (e.g. `athletic`). Saved to `input/templates/<set>/`, so one folder per body type = your body presets. A set with the same name as a built-in one replaces it.
2. Optional caption per photo: `swap_01.txt` next to `swap_01.jpg`, added after the trigger word.
3. `how_many = 3` to test, `0` = all. Finished images are skipped on re-runs (`skip_done`), so after a crash just press Run again.
4. Output: `datasets/<name>/<name>_<template>.png` + `.txt` + `<name>.zip`, same shape as each template. Encoder *resolution* is 2048 here (max detail); 1536 is about 2x faster with less face detail.

## Body-type presets (athletic, curvy …)
`workflows/AI_Empire_Dataset_Maker_Qwen21_BodyPresets.json`: turns your base photos into a preset where every photo has the same body. Face, hair, pose, outfit, place and light stay.
1. **Body Preset Maker** box → upload your base photos (e.g. set `base`), set *output_set* (`athletic`) and *body_target*.
2. *instructions*, one line per photo: `1-35 | keep` (copied unchanged), `swap_40 | her hips are wide, make them narrower` (extra help for one photo).
3. The result lands in `input/templates/athletic/` (captions copied too) and shows up in the Template Presets and Dataset Maker lists.

## GPU
- Qwen-Image 2.1 (int8, main workflow): runs on 24–48 GB GPUs.
- Nano Banana Pro / Seedream: run on RunPod's API, not on your pod's GPU.
- FireRed (41 GB model): 80 GB GPU (A100 / H100) as is; on 48 GB (A6000, A40, L40S) set UNet `weight_dtype = fp8_e4m3fn`.
- Qwen 2511 fp8mixed (20 GB model): runs on 24–48 GB GPUs as is.

## Your own presets
For the older workflows: fill the **custom_presets** box, one shot per line:
```
standing on a bridge at night, black coat | full body, bridge, night, black coat | tall
```
Format: `shot description | caption | square / portrait / tall`. Caption and size are optional.

## Nodes
- **AI Empire · Dataset Maker:** the one settings box of the main workflow: body type, test or whole dataset, trigger word, hair and eyes, engine, dataset name. Shows a preview of the preset photos and has the 🔑 RunPod key button.
- **AI Empire · Make image (Qwen or RunPod API):** passes on Qwen's image, or makes the photo with Nano Banana Pro / Seedream through RunPod's API.
- **AI Empire · Nano Banana:** makes the dataset with Nano Banana Pro / Nano Banana 2 using your own Google key (Vertex AI or AI Studio). Refused shots are skipped.
- **AI Empire · Template Presets:** loops over a folder of your own template photos (upload button on the node), with resume.
- **AI Empire · Body Preset Maker / Save Template Set:** makes a body-type preset from your base photos.
- **AI Empire · Dataset Presets:** trigger word, preset set, extra description (what must stay the same), how many, start at, seed.
- **AI Empire · Save Dataset:** saves images + captions, numbers or names them, zips the folder, and queues the next photo by itself (*auto_continue*).
- **AI Empire · Photo Finish:** phone-photo look in one node (levels, soft glow, lens softness, hand shake, ISO grain, JPEG).
- **AI Empire · Save Image (no workflow inside):** saves JPEG/PNG without the workflow embedded, so images can't be dragged into ComfyUI to copy it.

## Our RunPod template (nothing to install)
GitHub Actions builds the image on every push to `main` that changes it (README, `tools/` and `workflows/` changes don't trigger a build): `ghcr.io/justlinuxnoob/aiempire-dataset-maker:latest`

RunPod → **My Templates → New Template**:
- **Container image:** `ghcr.io/justlinuxnoob/aiempire-dataset-maker:latest`
- **Container disk:** 30 GB
- **Volume disk:** 100 GB, mounted at `/workspace` (the first boot downloads the Qwen-Image 2.1 models once)
- **Expose HTTP ports:** `8188,8888` (8188 = ComfyUI, 8888 = JupyterLab file browser: drag photos into `input/templates/<set>/`, download from `output/datasets/`)
- **Env: none needed.** Recommended: `JUPYTER_PASSWORD` = a password for JupyterLab (without it, anyone with the pod URL can open it). With no env vars the pod sets up the Qwen-Image 2.1 dataset maker. Optional extras:
  - `AIEMPIRE_RUNPOD_KEY` = your RunPod API key for the paid engines (as a RunPod **secret**). The 🔑 RunPod key button does the same without env vars.
  - `EDIT_MODEL` = `qwen21` (default) / `firered` / `qwen` (2511) / `both` / `none` (Nano Banana only)
  - `REALISM` = `1` downloads Z-Image Turbo for the realism pass (default on for FireRed / Qwen 2511, off for Qwen 2.1)
  - `QWEN_PRECISION=bf16` for the full Qwen Image Edit 2511 model
  - `KREA_TURBO=1` = Krea 2 Turbo + your LoRA generator only (`AI_Empire_Krea2_Turbo_LoRA.json` and `AI_Empire_Krea2_Turbo_PromptWriter.json`, ~21 GB, nothing else downloads; any 24 GB+ GPU)
  - `PORTRAIT=1` = portrait generator only (Z-Image Turbo + instagram LoRA for `portrait-gen.json`, ~20 GB, nothing else downloads; any 24 GB+ GPU)
  - `QWEN21=1` also downloads the prompt-enhancer text encoder for the Qwen-Image 2.1 edit test workflow (`AI_Empire_Qwen_Image_2.1_Edit_TEST.json`)
  - `KREA2=1` adds the Krea2 RAW workflow: RawGirl Krea2 + Flux 2 Klein 9B realism pass + skin detailer + phone-look finish (~60 GB, 48 GB+ GPU). Set `EDIT_MODEL=none` + `REALISM=0` for a Krea2-only pod (100 GB volume is enough)
  - `VIDEO=minimax` adds MiniMax H3 video (image-to-video + reference-to-video, ~75 GB more: use a 200 GB volume, 48 GB+ GPU). Set `EDIT_MODEL=none` + `REALISM=0` for a video-only pod
  - Nano Banana node: `VERTEX_SA_JSON` (whole service-account JSON, as a RunPod **secret**) or `GEMINI_API_KEY`

On boot the pod starts JupyterLab, downloads the models (first time only), then starts ComfyUI. Open **Connect → HTTP 8188** and drag in the workflow `.json` (workflows are NOT inside the image; they're delivered separately).
Download the finished zip with JupyterLab (`output/datasets/<name>.zip`), or at `http://<pod-url>/view?filename=<name>.zip&subfolder=datasets&type=output`.
