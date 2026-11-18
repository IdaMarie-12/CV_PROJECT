""" Shared settings for the project. """

from pathlib import Path

# ----------------------------------
# Reproducibility
# ----------------------------------
SEED = 42
N_IMAGES = 500
SCALE = 4

# ----------------------------------
# Paths
# ----------------------------------
COCO_DIR = Path("/content/coco")
VAL_IMAGES = COCO_DIR / "val2017"
VAL_ANN = COCO_DIR / "annotations" / "instances_val2017.json"

# ----------------------------------
# Input conditions
# ----------------------------------
CONDITIONS = ["hr", "lr", "bicubic", "sr_edsr"]
IMG_DIRS = {cond: CACHE / c for c in CONDITIONS}
DET_IMGSZ = {"hr": 640, "lr": 160, "bicubic": 640, "sr_edsr": 640}
BOX_SCALE = {"hr": 1, "lr": SCALE, "bicubic": 1, "sr_edsr": 1}

# ----------------------------------
# Models
# ----------------------------------
DETECTORS = {"yolo11m": "yolo11m.pt"}
SR_MODELS = "eugenesiow/edsr-base"

# ----------------------------------
# Evaluation settings
# ----------------------------------
SCORE_THR_EVAL = 0.001
SCORE_THR_VIS = 0.25
MAX_DETS = 100
N_BOOTSTRAP = 200
SMALL_MAX = 32 ** 2
MEDIUM_MAX = 96 ** 2

def size_group(area: floet) -> str:
    """Return COCOs size group for an object. """
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"

def make_dirs() -> None:
    """ Create every output folder if it does not exist yet. """
    for p in (CACHE, RESULTS, PREDS, FIGURES, *IMG_DIRS.values()):
        p.mkdir(parents = True, exists_ok = True)
        