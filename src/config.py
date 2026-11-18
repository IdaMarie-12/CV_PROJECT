""" Central configuration for the reprodcibility SR objects detection experiment. """

from __future__ import annotations
from pathlib import Path

# -------------------------------
# Paths
# -------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "preds"
FIGURES_DIR = ROOT / "figures"

COCO_DIR = DATA_DIR / "coco"
COCO_IMAGES_DIR = COCO_DIR / "val2017"
COCO_ANNOTATIONS = COCO_DIR / "annotations" / "instances_val2017.json"
SUBSET_ANNOTATIONS = COCO_DIR / "annotations_subset.json"
SELECTED_IMAGES = CACHE_DIR / "selected_images_ids.json"
QUALIFYING_CATEGORIES = CACHE_DIR / "qualifying_categories.json"
OBJECT_COUNTS = CACHE_DIR / "object_counts.json"

COCO_URLS = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

# -------------------------------
# Subset rules
# -------------------------------
SEED = 42
IMAGE_COUNT = 500
MIN_INSTANCES_FOR_REPORT = 100

# COCO size groups, by object area in orginial-image pixels
SMALL_MAX_AREA = 32 ** 2
MEDIUM_MAX_AREA = 96 ** 2
SIZE_GROUPS = ("small", "medium", "large")

# -------------------------------
# Experiment settings
# -------------------------------
SCALE = 4
HR_SIZE = 640
NATIVE_LR_SIZE = 160
DETECTOR_CONFIDENCE = 0.001
REPORT_CONFIDENCE = 0.25
IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 100
MIN_OBJECT_PER_IMAGE_RECALL = 3

BOOSTRAP_RESAMPLES = 1_000
DEBUG_BOOTSTRAP_RESAMPLES = 200
RECOVERY_MIN_DENOMINATOR = 0.005
MEANINGFUL_EFFECT = 0.01

TIMING_WARMUPS = 10
TIMING_REPEATS = 100

# -------------------------------
# Models
# -------------------------------
DETECTOR_NAME = "yolo11m"
DETECTOR_WEIGHTS = "yolo11m.pt"
SR_MODEL_NAME = "eugenesiow/edsr-base"

# -------------------------------
# Input conditions
# -------------------------------
CONDITIONS = ("hr", "lr_native", "lr_bicubic", "lr_sr")
CONDITION_DIRS = {condition: CACHE_DIR / "images" / condition for condition in CONDITIONS}
DETECTOR_SIZES = {
    "hr": HR_SIZE,
    "lr_native": NATIVE_LR_SIZE,
    "lr_bicubic": HR_SIZE,
    "lr_sr": HR_SIZE,
}
COORDINATE_SCALES = {
    "hr": 1.0, 
    "lr_native": float(SCALE),
    "lr_bicubic": 1.0,
    "lr_sr": 1.0,
}

def size_group(area: float) -> str:
    """ Returns COCOs size group (small, medium, large) for a given object area. """
    if area < SMALL_MAX_AREA:
        return "small"
    if area < MEDIUM_MAX_AREA:
        return "medium"
    return "large"

def make_directories() -> None:
    """ Creates all project output directories. """
    for path in (DATA_DIR, CACHE_DIR, RESULTS_DIR, PREDICTIONS_DIR, FIGURES_DIR, *CONDITION_DIRS.values()):
        path.mkdir(parents = True, exist_ok = True)