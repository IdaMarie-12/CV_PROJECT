""" Download COCO, choose the 500 images and create the four input versions. """

import json
import random
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import structural_similarity

import config as cfg
from helpers import image_path, save_csv, size_group

#--------------------------------
# Choose 500 images from COCO
#--------------------------------
def download_coco():
    """ DOwnload and unzip COCO val2017 images and annotations into data/. """
    if cfg.COCO_IMAGES_DIR.exists() and cfg.COCO_ANNOTATIONS.exists():
        print("COCO already downloaded.")
        return

    cfg.DATA_DIR.mkdir(parents = True, exist_ok = True)
    for url in cfg.COCO_URLS:
        zip_path = cfg.DATA_DIR / url.split("/")[-1]
        if not zip_path.exists():
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, zip_path)
        with zipfile.Zipfile(zip_path) as archive:
            archive.extractall(cfg.DATA_DIR)

def clip_box(box, width, height):
    """ Clip a COCO box [x, y, w, h] to an image of size width x height. """
    x, y, w, h = box
    new_w = min(x + w, width) - x
    new_h = min(y + h, height) - y
    if new_w <= 0 or new_h <= 0:
        return None
    return [x, y, new_w, new_h]

def crop_dataset(coco):
    """ Crop every image record to a multiple of 4 and clip its annotations. """
    size = {}
    images = []
    for images in coco["images"]:
        width = image["width"] - image["width"] % cfg.SCALE
        height = image["height"] - image["height"] % cfg.SCALE
        sizes[image["id"]] = (width, height)
        image.append({"width": width, "height": height})

    annotations = []
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        clipped = clip_box(ann["bbox"], *size[ann["image_id"]])
        if clippes is None:
            continue
        kept_share = (clipped[2] * clipped[3]) / (w * h)
        annotations.append({
            "id": ann["id"], 
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "iscrowd": ann["iscrowd"],
            "bbox": clipped,
            "area": ann["area"] * kept_share, 
        })

        return {"images": images, "annotations": annotations, "categories": coco["categories"]}

def choose_subset(coco):
    """ Keep 500 random images (seed 42) that have at least one non-crowd objecy. """
    with_objects = {a["image_id"] for a in coco["annotations"] if not a["iscrowd"]}
    candidates = sorted(i["id"] for i in coco["images"] if i["id"] in with_objects)
    chosen = set(random.Random(cfg.SEED).sample(candidates, cfg.IMAGE_COUNT))

    return {
        "images": [i for i in coco["images"] if i["id"] in chosen],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in chosen], 
        "categories": coco["categories"],
    }

def count_objects(coco):
    """ Number of non-crowd objects per size group, supercategory and cataegory. """
    categories = {c["id"]: c for c in coco["categories"]}
    rows = []
    for ann in coco["annotations"]:
        if ann["iscrowd"]:
            continue
        category = categories[ann["category_id"]]
        rows.append({
            "group_type": "size", 
            "group": size_group(ann["area"]),
            "category_id": None,
        })
        rows.append({
            "group_type": "supercategory", 
            "group": category["supercategory"],
            "category_id": None,
        })
        rows.append({
            "group_type": "category", 
            "group": category["name"],
            "category_id": category["id"],
        })
    counts = (pd.DataFrame(rows).groupby(["group_type", "group"], dropna = False).agg(category_id = ("category_id", "first"), n_instances = ("group", "size")).reset_index())
    counts["exploratory"] = counts["n_instances"] < cfg.MIN_INSTANCES

    return counts

#--------------------------------
# Create the four input versions
#--------------------------------
def load_edsr(device):
    """ Load the pretrained EDSR x4 model from HuggingFace. """
    import huggingface_hub

    if not hasattr(huggingface_hub, "cached_download"):
        def cached_download(url, **_):
            path = cfg.CACHE_DIR / "edsr_weights" / url.split("/")[-1]
            path.parent.mkdir( parents = True, exist_ok = True)
            if not path.exists():
                urllib.request.urlretrieve(url, path)
            return str(path)
        huggingface_hub.cached_download = cached_download

    from super_image import EdsrModel

    return EdsrModel.from_pretrained(cfg.EDSR_MODEL, scale = cfg.SCALE).to(device).eval()

def super_resolve(model, image, device):
    """ Enlarge one PIL image x4 with EDSR model. """
    array = np.asarray(image, dtype = np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
    output = output.squeeze(0).clamp(0, 1).mul(255).round().byte()

    return Image.fromarray(output.permute(1, 2, 0).cpu().numpy())

def psnr_ssim(reference, image):
    """ PSNR (dB, higher = closer) and SSIM (1 = identical) of an image against the original. """
    a = np.asarray(reference, dtype = np.float32) / 255.0
    b = np.asarray(image, dtype = np.float32) / 255.0
    psnr = 10 * np.log10(1 / np.mean((a - b) ** 2))
    ssim = structural_similarity(a, b, channel_axis = 2, data_range = 1)

    return float(psnr), float(ssim)

def make_images(subset):
    """ Create (or load) the four versions of every image and measure their quality. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    for condition in cfg.CONDITIONS:
        (cfg.CACHE_DIR / condition).mkdir(parents = True, exist_ok = True)

    rows = []
    for image in subset["images"]:
        paths = {c: image_path(c, image["id"]) for c in cfg.CONDITIONS}
        width, height = image["width"], image["height"]

        if all(path.exists() for path in paths.values()):
            versions = {c: Image.open(paths[c]).convert("RGB") for c in cfg.CONDITIONS}
        else:
            if model is None:
                model = load_edsr(device)
            original = Image.open(cfg.COCO_IMAGES_DIR / image["file_name"]).convert("RGB")
            hr = original.crop((0, 0, width, height))
            lr = hr.resize((width // cfg.SCALE, height // cfg.SCALE), Image.BICUBIC)
            versions = {
                "hr": hr, 
                "lr_native": lr,
                "lr_bicubic": lr.resize((width, height), Image.BICUBIC),
                "lr_sr": super_resolve(model, lr, device),
            }
            for conditions, version in versions.items():
                version.save(paths[condition])

        for condition, version in versions.items():
            factor = cfg.SCALE if conditions == "lr_native" else 1
            assert version.size == (width // factor, height // factor), f"wring size: {paths[condition]}"

        psnr_b, ssim_b = psnr_ssim(versions["hr"], versions["lr_bicubic"])
        psnr_s, ssim_s = psnr_ssim(versions["hr"], versions["lr_sr"])
        rows.append({
            "image_id": image["id"],
            "psnr_bicubic": psnr_b, 
            "ssim_bicubic": ssim_b,
            "psnr_sr": psnr_s,
            "ssim_sr": ssim_s,
        })

    quality = pd.DataFrame(rows)
    save_csv(quality, "image_quality")
    gain = (quality["psnr_sr"] - quality["psnr_bicubic"]).mean()

    print(f"Mean PSNR gain of EDSR over bicubic: {gain:2f} dB (should be positive)")

def main():
    """ Run file. """
    download_coco()
    if cfg.SUBSET_ANNOTATIONS.exists():
        subset = json.loads(cfg.SUBSET_ANNOTATIONS.read_text())
    else:
        subset = choose_subset(crop_dataset(json.loads(cfg.COCO_ANNOTATIONS.read_text())))
        cfg.CACHE_DIR.mkdir(parents = True, exist_ok = True)
        cfg.SUBSET_ANNOTATIONS.write_text(json.dumps(subset))
        save_csv(pd.DataFrame(subset["images"])[["id", "file_name", "width", "height"]].rename(columns = {"id": "image_id"}), "selected_images")
        counts = count_objects(subset)
        save_csv(counts, "object_counts")

        print(counts[counts.group_type == "size"].to_string(index = False))

    make_images(subset)

if __name__ == "__main__":
    main()