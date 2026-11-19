""" Evaluate saved predictions and compute bootstrap intervals. """

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import config as cfg
from common import quiet, read_json, write_json

# Index of each reported metris in COCOevals 12-element 'stats' vector
METRICS = {"map_50_95": 0, "map_50": 1, "ap_small": 3, "ap_medium": 4, "ap_large": 5, "ar_100": 8}
IOU_75_INDEX = 5

# -------------------------------
# Get COCOeval information
# -------------------------------
def load_ground_truth(data, dict[str, Any]) -> COCO:
    """ Load COCO ground truth from a dict. """
    ground_truth = COCO()
    ground_truth.dataset = data
    quiet(ground_truth.createIndex)

    return ground_truth

# -------------------------------
# Evaluate predictions
# -------------------------------
def run_coco_eval(ground_truth: COCO, predictions: list[dict[str, Any]], category_ids: list[int] | None = None) -> COCOeval | None:
    """ Run COCOeval for boxes and return the finished evaluator. """
    if not predictions:
        return None

    detection = quiet(ground_truth.loadRes, copy.deepcopy(predictions))
    evaluator = COCOeval(ground_truth, detection, "bbox")
    evaluator.params.imgIds = sorted(ground_truth.getImgIds())
    if category_ids is not None:
        evaluator.params.catIds = category_ids
    evaluator.params.maxDets = [1, 10, cfg.MAX_DETECTIONS]
    quiet(evaluator.evaluate)
    quiet(evaluator.accumulate)
    quiet(evaluator.summarize)

    return evaluator

def stats_of(evaluator: COCOeval | None) -> np.ndarray:
    """ The reported metrics from an evalutor, in 'METRICS' order (zeros if no predictions). """
    if evaluator is None:
        return np.zeros(len(METRICS))
    return np.array([evaluator.stats[index] for index in METRICS.values()])

def per_image_counts(evalutor: COCOeval) -> dict[int, dict[str, int]]:
    """ Per-inage TP/FP at IoU 0,05, TP at IoU 0.75 and ground-truth count, for score >= 0.25. """
    counts: dict[int, dict[str, int]] = defaultdict(lambda: {"n_gt": 0, "tp50": 0, "tp75": 0})
    all_range = evalutor.params.areaRng[0]
    for entry in evalutor.evalImgs:
        if entry is None or entry["aRng"] != all_range:
            continue
        keep = np.asarray(entry["dtScores"]) >= cfg.REPORT_CONFIDENCE
        matches = np.asarray(entry["dtMatches"])
        ignored = np.asarray(entry["dtIgnore"]).astype(bool)
        row = counts[entry["image_id"]]
        row["n_gt"] += int((~np.asarray(entry["gtIgnore"]).astype(bool)).sum())
        if matches.size:
            row["tp50"] += int((keep & (matches[0] > 0) & ~ignored[0]).sum())
            row["fp50"] += int((keep & (matches[0] == 0) & ~ignored[0]).sum())
            row["tp75"] += int((keep & (matches[IOU_75_INDEX] > 0) & ~ignored[IOU_75_INDEX]).sum())
    return counts

def recovery(hr: float, bicubic: float, sr: float) -> float:
    """ SR recovery fraction; NaN when the denominator is below the pre-declared mninimum. """
    denominator = hr - bicubic

    return (sr - bicubic) / denominator if denominator >= cfg.RECOVERY_NIN_DENOMINATOR else float("nan")

