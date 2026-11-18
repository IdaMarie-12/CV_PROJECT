""" Download COCO, create the fixed subset and crop-aware annotations. """

from __future__ import annotations

import random
import urllib.request
import zipfile
from collections import Counter
from typing import Any
import config as cfg
from common import read_json, write_json

# -------------------------------
# Download COCO
# -------------------------------
def download_coco() -> None:
    """ Download COCO val2017 into 'cfg.COCO_DIR' unless it is already there. """
    if cfg.COCO_IMAGES_DIR.exists() and cfg.COCO_ANNOTATIONS.exists():
        return

    cfg.COCO_DIR.mkdir(parents = True, exist_ok = True)
    for name, url in cfg.COCO_URLS.items():
        zip_path = cfg.COCO_DIR / name
        if not zip_path.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(cfg.COCO_DIR)
    if not cfg.COCO_ANNOTATIONS.exists():
        raise FileNotFoundError(f"Missing COCO annotations at {cfg.COCO_ANNOTATIONS}")

# -------------------------------
# Crop
# -------------------------------
def crop_annotations(annotations: dict[str, Any], crop_width: int, crop_height: int) -> dict[str, Any] | None:
    """ Clip one COCO annotation to the top-left crop. """
    x, y, width, height = annotation["bbox"]

    if width <= 0 or height <= 0:
        return None

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + width, crop_width), min(y + height, crop_height)

    if clipped_width <= 0 or clipped_height <= 0:
        return None

    kept_share = (clipped_width * clipped_height) / (width * height)
    
    updated = {key: value for key, value in annotation.items() if key != "segmentation"}
    updated["bbox"] = [x1, y1, clipped_width, clipped_height]
    updated["area"] = annotation["area"] * kept_share

    return updated

def crop_annotations_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """ Return a new COCO dataset whose inage sizes and boxes are cropped. """
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in dataset["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    cropped_images, cropped_annotations = [], []
    for image in dataset["inages"]:
        crop_width = image["width"] - image["width"] % cfg.SCALE
        crop_height = image["height"] - image["height"] % cfg.SCALE
        cropped_images.append({**image, "width": crop_width, "height": crop_height})

        for annotation in annotations_by_image.get(image["id"], []):
            clipped = crop_annotation(annotation, crop_width, crop_height)
            if clipped is not None:
                cropped_annotations.append(clipped)

    return {**dataset, "images": cropped_images, "annotations": cropped_annotations}

# -------------------------------
# Choose images
# -------------------------------
def eligible_image_ids(dataset: dict[str, Any]) -> list[int]:
    """ Images with at least one non-crowd object after cropping, sorted for determinism. """
    with_objects = {a["image_id"] for a in dataset["annotations"] if not a.get("iscrowd", 0)}

    return sorted(image["id"] for image in dataset["images"] if image["id"] in with_objects)

def choose_image_ids(candidates: list[int], image_count: int) -> list[int]:
    """ Select a deterministic random subset of image IDs with 'cfg.SEED'. """
    if image_count > len(candidates):
        raise ValueError("Requested more images than are eligible in COCO val2017")

    generator = random.Random(cfg.SEED)

    return sorted(generator.sample(candidates, image_count))

def filter_images(dataset: dict[str, Any], image_ids: list[int]) -> dict[str, Any]:
    """ Keep only the selected iamges and their annotations. """
    selected = set(image_ids)

    return {
        **dataset, 
        "images": [image for image in dataset["images"] if image["id"] in selected],
        "annotations": [a for a in dataset["annotations"] if a["image_id"] in selected],
    }

# -------------------------------
# Get object counts
# -------------------------------
def object_counts(dataset: dict[str, Any]) -> tuple[Counter, Counter, Counter]:
    """ Count non-crowd objects by size group, supercategory and category ID. """
    supercategory_of = {c["id"]: c["supercategory"] for c in dataset["categories"]}
    sizes, supercategories, categories = Counter(), Counter(), Counter()

    for annotation in dataset["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        sizes[cfg.size_group(annotation["area"])] += 1
        supercategories[supercategory_of[annotation["category_id"]]] += 1
        categories[annotation["category_id"]] += 1

    return sizes, supercategories, categories

# -------------------------------
# Validation
# -------------------------------
def validate_subset(subset: dict[str, Any]) -> None:
    """ Fail clearly if image sizes, boxes or categories IDs are invalid. """
    category_ids = {c["id"] for c in subset["categories"]}
    images = {image["id"]: image for image in subset["images"]}

    for image in images.values():
        if image["width"] % cfg.SCALE or image["height"] % cfg.SCALE:
            raise ValueError(f"Image {image['id']} has size {image['width']}x{image['height']} not divisible by {cfg.SCALE}")

    for annotation in subset["annotations"]:
        image = images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        if annotation["category_id"] not in category_ids:
            raise ValueError(f"Annotation {annotation['id']} has invalid category ID {annotation['category_id']}")
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            raise ValueError(f"Invalid box in annotation {annotation['id']}")
        if x + width > image["width"] + 1e-6 or y + height > image["height"] + 1e-6:
            raise ValueError(f"Box outside crop in annotation {annotation['id']}")

# -------------------------------
# Main
# -------------------------------
def main() -> None:
    """ Build the final subset and save counts before inference begins. """
    cfg.make_directories()
    download_coco()
    cropped = crop_annotation_dataset(read_json(cfg.COCO_ANNOTATIONS))
    candidates = eligible_image_ids(cropped)

    subset = filter_images(cropped, choose_image_ids(candidates, cfg.IMAGE_COUNT))
    sizes, supercategories, categories = object_counts(subset)

    validate_subset(subset)
    qualifying = sorted(cid for cid, n in cateogires.items() if n >= cfg.MIN_INSTANCES_FOR_REPORT)
    selected_ids = sorted(image["id"] for image in subset["images"])

    write_json(cfg.SUBSET_ANNOTATIONS, subset, indent = None)
    write_json(cfg.SELECTED_IMAGES, selected_ids)
    write_json(cfg.QUALIFYING_CATEGORIES, qualifying)
    write_json(cfg.OBJECT_COUNTS, {
        "image_count": cfg.IMAGE_COUNT, 
        "size_counts": dict(sizes),
        "supercategory_counts": dict(supercategories),
        "category_counts": {str(k): v for k, v in sorted(categories.items())},
        "qualifying_categories_ids": qualifying, 
        "exploratory_size_groups": {sorted(group for group in cfg.SIZE_GROUPS if sizes[group] < cfg.MIN_INSTANCES_FOR_REPORT)},
        "exploratory_supercategories": {sorted(name for name in supercategories.items() if n < cfg.MIN_INSTANCES_FOR_REPORT)},
    })

    print(f"Selected {cfg.IMAGE_COUNT} images; size counts: {dict(sizes)}")
    print(f"Qualifying categories (>= {cfg.MIN_INSTANCES_FOR_REPORT} instances): {qualifying}")

if __name__ == "__main__":
    main()