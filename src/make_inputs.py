""" Generate and validate the four immutable image conditions. """

from __future__ import annotations

from functools import cache
import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import config as cfg
from common import image_path, read_json, validate_image

def _patch_huggingface_hub() -> None:
    """ Compatibility fix: super-image calls 'huggingface_hub.cached_download' which newer huggingface_hub versions removed. Add a minimal replacement. """
    import huggingface_hub

    if hasattr(huggingface_hub, "cached_download"):
        return

    def cached_download(url, cache_dir=None, **_):
        cache_dir = cache_dir or os.path.expanduser("~/.cache/super_image")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, hashlib.md5(url.encode()).hexdigest() + "_" + url.split("/")[-1])
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
        return path

    huggingface_hub.cached_download = cached_download

# -------------------------------
# EDSR
# -------------------------------
def load_edsr(device: torch.device):
    """ Load the fixed pretrained EDSR x4 model in evaluation mode on 'device'. """
    _patch_huggingface_hub()

    from super_image import EdsrModel
    model = EdsrModel.from_pretrained(cfg.SR_MODEL_NAME, scale = cfg.SCALE)

    return model.to(device).eval()

def super_resolve(model, low_resolution: Image.Image, device: torch.device) -> Image.Image:
    """ Run EDSR on one RGB PIL image and return an RGB PIL image. """
    array = np.array(low_resolution.convert("RGB"), dtype = np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_model():
        output = model(tensor)
    output = output.squeeze(0).clamp(0, 1).mul(255).round().byte()

    return Image.fromarray(output.permute(1, 2, 0).cpu().numpy())

# -------------------------------
# Image quality metrics
# -------------------------------
def quality(reference: Image.Image, candidate: Image.Image) -> tuple[float, float]:
    """ Return PSNR (dB) and SSIM between two identically sized RGB images. """
    reference_array = np.asarray(reference, dtype = np.float32) / 255.0
    candidate_array = np.asarray(candidate, dtype = np.float32) / 255.0

    mse = float(np.mean((ference_array - candidate_array) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1 / mse))
    ssim = float(structural_similarity(reference_array, candidate_array, channel_axis = 2, data_range = 1.0))

    return psnr, ssim

# -------------------------------
# Main
# -------------------------------
def main() -> None:
    """ Create (or load) all four conditions, validate them and save image quality. """
    cfg.make_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    annotation_data = read_json(cfg.SUBSET_ANNOTATIONS)
    rows: list[dict[str, float | int]] = []

    for image in annotation_data["images"]:
        image_id, width, height = image["id"], image["width"], image["height"]
        paths = {c: image_path(cfg.CONDITION_DIRS[c], image_id) for c in cfg.CONDITIONS}

        if all(path.exists() for path in paths.values()):
            hr = Image.open(paths["hr"]).convert("RGB")
            bicubic = Image.open(paths["lr_bicubic"]).convert("RGB")
            sr = Image.open(paths["lr_sr"]).convert("RGB")
        else:
            if mode is None:
                model = load_edsr(device)
            original = Image.open(cfg.COCO_IMAGES_DIR / image["file_name"]).convert("RGB")
            hr = original.crop((0, 0, width, height))
            lr = hr.resize((width // cfg.SCALE, height // cfg.SCALE), Image.BICUBIC)
            bicubic = lr.resize((width, height), Image.BICUBIC)
            sr = super_resolve(model, lr, device)
            for condition, output in {"hr": hr, "lr_bicubic": lr, "lr_bicubic": bicubic, "lr_sr": sr}.items():
                output.save(paths[condition], format = "PNG")

        # Validate every saved file
        validate_image(paths["hr"], (width, height))
        validate_image(paths["lr_native"], (width // C.SCALE, height // C.SCALE))
        validate_image(paths["lr_bicubic"], (width, height))
        validate_image(paths["lr_sr"], (width, height))

        psnr_b, ssim_b = quality(hr, bicubic)
        psnr_s, ssim_s = quality(hr, sr)
        rows.append({
            "image_id": image_id,
            "psnr_bicubic": psnr_b,
            "ssim_bicubic": ssim_b,
            "psnr_sr": psnr_s,
            "ssim_sr": ssim_s,
        })

    quality_table = pd.DataFrame(rows)
    quality_table.to.csv(cfg.RESULTS_DIR / "image_quality.csv", index = False)
    gain = (quality_table["psnr_sr"] - quality_table["psnr_bicubic"]).mean()
    print(f"Mean PSNR gain of EDSR over bicubic: {gain:.2f} dB")
    if gain <= 0:
        raise RuntimeError(f"EDSR does not beat bicubic on PSNR: check model loading and input range")

if __name__ == "__main__":
    main()