""" Run pretrained YOLO11m on all four cached image conditions. """

from __future__ import annotations

from typing import Any

from PIL import Image
from ultralytics import YOLO

import config as cfg
from common import image_path, read_json, write_json

# -------------------------------
# Sanity checks
# -------------------------------
def coco_category_ids(annotation_data: dict[str, Any]) -> list[int]:
    """ Return COCO category IDs in the standard detector class order (sorted IDs). """
    return sorted(category["id"] for category in annotation_data["categories"])

def check_class_mapping(model: YOLO, annotation_data: dict[str, Any]) -> None:
    """ Stop if YOLOs class name do not match the COCO names they are mapped to. """
    names = {c["id"]: c["name"] for c in annotation_data["categories"]}
    category_ids = coco_category_ids(annotation_data)
    mismatched = [(index, name, name[category_ids[index]])
                  for index, name in model.name.items() if name[category_ids[index]] != name]
    if len(model.name) != len(category_ids) or mismatched: 
        raise ValueError(f"Class mapping mismatch (index, YOLO name, COCO name): {mismatched}")

# -------------------------------
# Predictions
# -------------------------------
def to_coco_predictions(image_id: int, result: Any, scale: float, category_ids: list[int]) -> list[dict[str, Any]]:
    """ Convert one Ultralytics result to official COCO result dictionaries. """
    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    labels = result.boxes.cls.cpu().tolist()

    predictions = []

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [coordinate * scale for coordinate in box]
        predictions.append({
            "image_id": image_id, 
            "category_id": category_ids[int(label)],
            "bbox": [round(x1, 2), round(y1, 2), round(max(0.0, x2 - x1), 2), round(max(0.0, y2 - y1), 2)],
            "score": round(float(score), 5),
        })

    return predictions

def validate_predictions(predictions: list[dict[str, Any]], annotation_data: dict[str, Any]) -> None:
    """ Stop with a clear error if any prediction is inconsistent with the subset. """
    sizes = {image["id"]: (image["width"], image["height"]) for image in annotation_data["images"]}
    category_ids = set(coco_category_ids(annotation_data))
    per_image: dict[int, int] = {}
    for p in predictions:
        if p["image_id"] not in sizes:
            raise ValueError(f"Prediction for unknown image {p['image_id']}")
        if p["category_id"] not in category_ids:
            raise ValueError(f"Unknown category {p['category_id']}")
        x, y, w, h = p["bbox"]
        width, height = sizes[p["image_id"]]
        if w < 0 or h < 0 or x < -1 or y < -1 or x + w > width + 1 or y + h > height + 1:
            raise ValueError(f"Box outside image {p['image_id']}: {p['bbox']}")   # 1 px tolerance for rounding
        per_image[p["image_id"]] = per_image.get(p["image_id"], 0) + 1
    if per_image and max(per_image.values()) > C.MAX_DETECTIONS:
        raise ValueError("More than MAX_DETECTIONS predictions for an image")

# -------------------------------
# Run YOLO
# -------------------------------
def run_condition(model: YOLO, condition: str, annotation_data: dict[str, Any]) -> list[dict[str, Any]]:
    """ Run YOLO11m once per cached image in one condition. """
    category_ids = coco_category_ids(annotation_data)
    predictions: list[dict[str, Any]] = []

    for image in annotation_data["images"]:
        path = image_path(cfg.CONDITION_DIRS[condition], image["id"])
        pil_image = Image.open(path).convert("RGB")
        result = model.predict(pil_image, imgsz = cfg.DETECTOR_SIZES[condition], conf = cfg.DETECTOR_CONFIDENCE, max_det = cfg.MAX_DETECTIONS, verbose = False)[0]
        predictions.extend(to_coco_predictions(image["id"], result, cfg.COORDINATE_SCALES[condition], category_ids))
    return predictions

# -------------------------------
# Main
# -------------------------------
def main() -> None:
    """ Run all four conditions and save four immutable, validated prediction files. """
    cfg.make_directories()

    annotation_data = read_json(cfg.SUBSET_ANNOTATIONS)
    model = YOLO(cfg.DETECTOR_WEIGHTS)
    check_class_mapping(model, annotation_data)

    for condition in cfg.CONDITIONS:
        output_path = cfg.PREDICTIONS_DIR / f"{cfg.DETECTOR_NAME}_{condition}.json"
        if output_path.exists():
            print(f"Skipping {condition} because {output_path} already exists")
            continue
        predictions = run_condition(model, condition, annotation_data)
        validate_predictions(predictions, annotation_data)
        write_json(output_path, predictions, indent = None)
        print(f"{condtion}: {len(predictions)} predictions")

if __name__ == "__main__":
    main()