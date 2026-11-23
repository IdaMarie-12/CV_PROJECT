""" Settings for the experiment. """

from pathlib import Path

#--------------------------------
# Folders
#--------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"                # COCO download
CACHE_DIR = ROOT / "cache"              # The four image versions
RESULTS_DIR = ROOT / "results"          # CSV files with all results
FIGURES_DIR = ROOT / "figures"          # PNG figures for the report

# COCO image information
COCO_IMAGES_DIR = DATA_DIR / "val2017"
COCO_ANNOTATIONS = DATA_DIR / "annotations" / "instances_val2017.json"
COCO_URLS = [
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
]

SUBSET_ANNOTATIONS = CACHE_DIR / "subset_annotations.json"

#--------------------------------
# Subset
#--------------------------------
SEED = 42
IMAGE_COUNT = 500                      # Number of images used for the experiment
MIN_INSTANCES = 100                    # Groups (size, supercategory and category) with fewer objects are "exploratory"

#--------------------------------
# The four input conditions
#--------------------------------
SCALE = 4                              # Scale factor for up- and downsampling, 640px -> 160px -> 640px
CONDITIONS = [ 
    "hr",                              # High-resolution images
    "lr_native",                       # Low-resolution images (native downsampling)
    "lr_bicubic",                      # Low-resolution images (bicubic interpolation)
    "lr_sr",                           # Low-resolution images (super-resolution)
]
YOLO_SIZE = {
    "hr": 640,
    "lr_native": 160,
    "lr_bicubic": 640,
    "lr_sr": 640,
}
BOX_SCALE = {
    "hr": 1,
    "lr_native": SCALE,
    "lr_bicubic": 1,
    "lr_sr": 1,
}

#--------------------------------
# Models
#--------------------------------
YOLO_WEIGHTS = "yolo11m.pt"
EDSR_MODEL = "eugenesiow/edsr-base"

#--------------------------------
# Detection and evaluation
#--------------------------------
DETECT_CINF = 0.001
REPORT_CINF = 0.25
MAX_DETS = 100
MIN_OBJETS_PER_IMAGE = 3

BOOTSTRAP_RESAMPLES = 1000
MIN_RECOVERY_DENOMINATOR = 0.005
MEANINGFUL_EFFECT = 0.01

TIMING_WARMUP = 10
TIMING_REPEATS = 100