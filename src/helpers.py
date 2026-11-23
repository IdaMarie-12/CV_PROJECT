""" Helpers shared by the experiment. """

import contextlib
import io

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import config as cfg

def image_path(condition, image_id):
    """ Path of one cached PNG image. """
    return cfg.CACHE_DIR / condition / f"{image_id:012d}.png"

def size_group(area):
    """ COCOs size group of an object, from its area in originial-image pixels. """
    if area < 32 ** 2:
        return "small"
    elif area < 96 ** 2:
        return "medium"
    else:
        return "large"

def save_csv(table, name):
    """ Save tables as results/<name>.csv. """
    cfg.RESULTS_DIR.mkdir(parents = True, exist_ok = True)
    table.to_csv(cfg.RESULTS_DIR / f"{name}.csv", index = False)

def load_csv(name):
    """ Load results/<name>.csv as a table. """
    return pd.read_csv(cfg.RESULTS_DIR / f"{name}.csv")

@contextlib.contextmanager
def quiet():
    """ Hide printed output inside a 'with quiet():' block. pycocotools prints a lot. """
    with contextlib.redirect_stdout(io.StringIO()):
        yield

def load_coco(dataset):
    """ Build a pycocotools 'COCO' ground-truth object from COCO-format dict. """
    ground_truth = COCO()
    ground_truth.dataset = dataset
    with quiet():
        ground_truth.createIndex()

    return ground_truth

def coco_eval(ground_truth, predictions, category_ids = None):
    """ Run the official COCO box evaluation and return the finished 'COCOeval'. """
    if predictions.empty: 
        return None

    # pycocotools accepts an Nx7 array
    array = predictions[["image_id", "x", "y", "w", "h", "score", "category_id"]].to_numpy()
    with quiet():
        detections = ground_truth.loadRes(array)
        evaluator = COCOeval(ground_truth, detections, "bbox")
        if category_ids is not None:
            evaluator.params.catIds = list(category_ids)
        
        evaluator.evaluate()            # Match predictions to objects (per image, class, size range and IoU)
        evaluator.accumulate()          # Build precision-recall curves
        evaluator.summarize()           # Average them into 12 numbers in evaluator.stats

    return evaluator