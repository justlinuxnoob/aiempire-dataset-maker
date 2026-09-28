"""AI Empire photo finishing nodes.

  * AIEmpirePhotoFinish - one node for the whole "real phone photo" finish:
    levels (exposure / gamma / contrast / saturation / vibrance), soft glow,
    lens + sensor softness, ISO grain, final punch and optional JPEG look.
  * AIEmpireSaveClean   - saves JPEG/PNG WITHOUT the workflow inside the file,
    so nobody can drag your images into ComfyUI and get your workflow.
"""

import io
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths

ISO_NOISE = {  # std of luminance noise (0..1 image range)
    "off": 0.0,
    "ISO 50": 0.004,
    "ISO 100": 0.007,
    "ISO 200": 0.011,
    "ISO 400": 0.017,
    "ISO 800": 0.026,
    "ISO 1600": 0.038,
}


def _luma(x):
    return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]).unsqueeze(-1)


def _gauss_kernel(sigma, device):
    radius = max(1, int(round(sigma * 3)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _blur(x, sigma):
    """Separable gaussian blur on BHWC."""
    if sigma <= 0.05:
        return x
    k = _gauss_kernel(sigma, x.device)
    c = x.shape[-1]
    y = x.permute(0, 3, 1, 2)
    pad = k.numel() // 2
    y = F.pad(y, (pad, pad, 0, 0), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    y = F.pad(y, (0, 0, pad, pad), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return y.permute(0, 2, 3, 1)


def _big_blur(x, sigma):
    """Large blur done at low resolution (fast for glow)."""
    if sigma <= 0.05:
        return x
    h, w = x.shape[1:3]
    scale = max(1, int(sigma // 6))
    y = x.permute(0, 3, 1, 2)
    if scale > 1:
        y = F.interpolate(y, size=(max(1, h // scale), max(1, w // scale)), mode="area")
    y = _blur(y.permute(0, 2, 3, 1), sigma / scale).permute(0, 3, 1, 2)
    if scale > 1:
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
    return y.permute(0, 2, 3, 1)


def _motion_blur(x, length):
    """Tiny horizontal hand-shake blur."""
    if length <= 0.1:
        return x
    n = max(2, int(np.ceil(length)) | 1)
    k = torch.ones(n, device=x.device, dtype=torch.float32)
    frac = length - np.floor(length)
    if frac > 0 and n > 2:
        k[0] = k[-1] = frac
    k = k / k.sum()
    c = x.shape[-1]
    y = x.permute(0, 3, 1, 2)
    pad = n // 2
    y = F.pad(y, (pad, pad, 0, 0), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return y.permute(0, 2, 3, 1)


def _jpeg(x, quality):
    out = []
    for img in x:
        arr = (img.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality), subsampling=0)
        buf.seek(0)
        out.append(torch.from_numpy(np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0))
    return torch.stack(out).to(x.device)


class AIEmpirePhotoFinish:
    """Turns a clean AI render into something that looks shot on a phone."""

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo, hi, st=0.01: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": st})
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": f(-0.05, -2.0, 2.0),
                "gamma": f(1.05, 0.5, 2.0),
                "contrast": f(0.98, 0.5, 1.5),
                "saturation": f(1.09, 0.0, 2.0),
                "vibrance": f(-0.04, -1.0, 1.0),
                "glow": f(0.075, 0.0, 1.0, 0.005),
                "glow_radius": f(50.0, 1.0, 200.0, 1.0),
                "glow_threshold": f(0.3, 0.0, 1.0),
                "lens_softness": f(0.25, 0.0, 3.0),
                "hand_shake": f(1.9, 0.0, 10.0, 0.1),
                "iso": (list(ISO_NOISE.keys()), {"default": "ISO 200"}),
                "grain_size": f(0.5, 0.0, 3.0),
                "final_contrast": f(1.10, 0.5, 1.5),
                "jpeg_quality": ("INT", {"default": 100, "min": 30, "max": 100, "tooltip": "100 = off"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "AI Empire"

    def run(self, image, exposure, gamma, contrast, saturation, vibrance, glow, glow_radius,
            glow_threshold, lens_softness, hand_shake, iso, grain_size, final_contrast,
            jpeg_quality, seed):
        x = image[..., :3].float().clone()

        # levels
        x = x * (2.0 ** exposure)
        x = x.clamp(0, 1) ** (1.0 / gamma)
        x = (x - 0.5) * contrast + 0.5
        l = _luma(x)
        x = l + (x - l) * saturation
        if vibrance != 0:
            mx = x.max(-1, keepdim=True).values
            mn = x.min(-1, keepdim=True).values
            sat = (mx - mn).clamp(0, 1)
            l = _luma(x)
            x = l + (x - l) * (1 + vibrance * (1 - sat))
        x = x.clamp(0, 1)

        # soft glow from the bright parts (screen blend)
        if glow > 0:
            bright = ((_luma(x) - glow_threshold) / max(1e-4, 1 - glow_threshold)).clamp(0, 1)
            halo = _big_blur(x * bright, glow_radius)
            x = 1 - (1 - x) * (1 - halo * glow)

        # lens / sensor softness + hand shake
        x = _blur(x, lens_softness)
        x = _motion_blur(x, hand_shake)

        # ISO grain (luminance, stronger in the shadows like a real sensor)
        std = ISO_NOISE.get(iso, 0.0)
        if std > 0:
            g = torch.Generator(device="cpu").manual_seed(int(seed) % (2 ** 63))
            noise = torch.randn((x.shape[0], x.shape[1], x.shape[2], 1), generator=g).to(x.device)
            if grain_size > 0.05:
                noise = _blur(noise, grain_size)
                noise = noise / noise.std().clamp(min=1e-6)
            shadow = 1.3 - 0.6 * _luma(x)
            x = x + noise * std * shadow

        # final punch
        x = ((x - 0.5) * final_contrast + 0.5).clamp(0, 1)

        if jpeg_quality < 100:
            x = _jpeg(x, jpeg_quality)
        return (x,)


class AIEmpireSaveClean:
    """Save images with NO workflow / prompt inside the file."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "aiempire"}),
                "format": (["JPEG", "PNG"], {"default": "JPEG"}),
                "quality": ("INT", {"default": 100, "min": 50, "max": 100}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "AI Empire"

    def save(self, images, filename_prefix, format, quality):
        full, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        ext = "jpg" if format == "JPEG" else "png"
        results = []
        for img in images:
            arr = (img[..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            fname = f"{name}_{counter:05}_.{ext}"
            pil = Image.fromarray(arr)
            if format == "JPEG":
                pil.save(os.path.join(full, fname), format="JPEG", quality=quality, subsampling=0)
            else:
                pil.save(os.path.join(full, fname), format="PNG", compress_level=4)
            results.append({"filename": fname, "subfolder": subfolder, "type": "output"})
            counter += 1
        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "AIEmpirePhotoFinish": AIEmpirePhotoFinish,
    "AIEmpireSaveClean": AIEmpireSaveClean,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIEmpirePhotoFinish": "AI Empire · Photo Finish (phone look)",
    "AIEmpireSaveClean": "AI Empire · Save Image (no workflow inside)",
}
