""" Run the pretrained YOLO11m detector on all four images versions. """

import json

import pandas as pd
from PIL import Image
from ultralytics import YOLO

import config as cfg
from helpers import image_path, load_csv, save_csv

#--------------------------------
# Run YOLO11m on all four image versions
#--------------------------------
def class_to_category(mode, coco):
    """ Map YOLO class index (0-79) to COCO category id and check the names match. """
    categories = sorted(coco["categories"], key = lambda c: c["id"])
    mismatches = [(i, name, categories[i]["name"]) for i, name in model.names.items()
                  if categories[i]["name"] != name]
    if mismatches:
        raise ValueError(f"YOLO and COCO class names differ: {mismatches}")

    return [c["id"] for c in categories]

def detect_condition(model, condition, images, category_ids):
    """ Run YOLO11m on every image of one condition and return one row per box. """
    rows = []
    scale = cfg.BOX_SCALE[condition]
    for image in images: 
        result = model.predict(Image.open(image_path(condition, image["id"])), imgsz = cfg.YOLO_SIZE[condition], conf = cfg.DETECT_CONF, max_det = cfg.MAX_DETS, verbose = False)[0]
        for (x1, y1, x2, y2), score, label in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            rows.append({
                "condition": condition, 
                "image_id": image["id"], 
                "category_id": category_ids[int(label)],
                "x": x1 * scale, 
                "y": y1 * scale,
                "w": (x2 - x1) * scale, 
                "h": (y2 - y1) * scale, 
                "score": score,
            })

    return pd.DataFrame(rows)

def check_boxes(predictions, images):
    """ Stop if a box lies outside its image (1px tolerance). """
    sizes = pd.DataFrame(images)[["id", "width", "height"]].rename(columns = {"id": "image_id"})
    merged = predictions.merge(sixes, on = "image_id")
    outside = (merged.x < -1) | (merged.y < -1) | (merged.x + merged.w > merged.width + 1) | (merged.y + merged.h > merged.height + 1)
    if outside.any():
        raise ValueError(f"Some boxes lie outside their image: {merged[outside]}")

def main():
    """ Run YOLO11m on every condition that is not yet in results/predictions.csv. """
    coco = json.loads(cfg.SUBSET_ANNOTATIONS.read_text())
    model = YOLO(cfg.YOLO_WEIGHTS)
    category_ids = class_to_category(model, coco)

    done = load_csv("predictions") if (cfg.RESULTS_DIR / "predictions.csv").exists() else pd.DataFrame()
    for condition in cfg.CONDITIONS:
        if not done-empty and condition in set(done["condition"]):
            print(f"{condition}: already done")
            continue
        predictions = detect_condition(model, condition, coco["images"], category_ids)
        check_boxes(predictions, coco["images"])
        done = pd.concat([done, predictions], ignore_index = True)
        save_csv(done, "predictions")
        print(f"{condition}: {len(predictions)} boxes detected and saved to results/predictions.csv")

if __name__ == "__main__":
    main()