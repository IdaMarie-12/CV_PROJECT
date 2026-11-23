""" Evaluate the predictions and estimate uncertainty with a paired bootstrap. """

import json

import numpy as np
import pandas as pd

import config as cfg
from helpers import coco_eval, load_coco, load_csv, save_csv

#--------------------------------
# Prep
#--------------------------------
# Position of each reported metric in COCOeval 12-number 'stats' list
METRICS = {
    "map_50_95": 0,
    "map_50": 1,
    "ap_small": 3,
    "ap_medium": 4,
    "ap_large": 5,
    "ar_100": 8
}

# Name: (condition a, condition b) -> a minus b
DIFFERENCES = {
    "lr_native_minus_hr": ("lr_native", "hr"),
    "lr_bicubic_minus_hr": ("lr_bicubic", "hr"),
    "lr_sr_minus_lr_bicubic": ("lr_sr", "lr_bicubic"),
    "lr_sr_minus_hr": ("lr_sr", "hr"),    
}

#--------------------------------
# Evaluate predictions
#--------------------------------
def summary(evaluator):
    """ The six reported COCO metrics as a dict (all zero if there were no predictions). """
    if evaluator is None: 
        return {name: 0.0 for name in METRICS}
    return {name: float(evaluator.stats[index]) for name, index in METRICS.items()}

def per_image(evaluator):
    """ Per-image counts at score >= 0.25, using COCOevas own matching. """
    rows = []
    for entry in evaluator.evalImgs:
        if entry is None or entry["aRng"] != evaluator.params.areaRng[0]:
            continue
        confident = np.asarray(entry["dtScores"]) >= cfg.REPORT_CONF
        matched = np.asarray(entry["dtMatches"]) > 0
        ignored = np.asarray(entry["dtIgnore"]).astype(bool)
        counts = {
            "image_id": entry["image_id"],
            "n_objects": int((~np.asarray(entry["gtIgnore"]).astype(bool)).sum()),
            "tp50": 0,
            "fp50": 0,
            "tp75": 0,
        }
        if mathced.size:
            counts["tp50"] = int((confident & matched[0] & ~ignored[0]).sum())
            counts["fp50"] = int((confident & ~matched[0] & ~ignored[0]).sum())
            counts["tp75"] = int((confident & matched[1] & ~ignored[5]).sum())
        rows.append(counts)

    return pd.DataFrame(rows).groupby("image_id", as_index = False).sum()

def recovery(hr, bicubic, sr):
    """ Share of the gap left by bicubic that SR closes; NaN if the gap is below 0.005. """
    gap = hr - bicubic
    return (sr - bicubic) / gap if gap >= cfg.MIN_RECOVERY_DENOMINATOR else np.nan

def point_estimates(ground_truth, predictions, counts):
    """ Metrics, group metrics and per-image counts for every condition on all 500 images. """
    metrics_rows, group_rows, image_tables = [], [], []
    supercategories = {}
    for c in ground_truth.dataset["categories"]:
        supercategories(c["supercategory"], []).append(c["id"])
    qualifying = counts[(counts.group_type == "category") & ~counts.exploratory]

    for condition in cfg.CONDITIONS:
        boxes = predictions[predictions.condition == condition]
        evaluator = coco_eval(ground_truth, boxes)
        if evaluator is None:
            raise ValueError(f"No predictions for {condition}: run detect.py first")

        metrics = summary(evaluator)

        images = per_image(evaluator)
        images["condition"] = condition
        image_tables.append(images)
        tp, fp, n = images.tp50.sum(), images.fp50.sum(), images.n_objects.sum()
        metrics_rows.append({"condition": condition, **metrics, "precision_025": tp / max(tp + fp, 1), "recall_025": tp / max(n, 1)})

        groups = [("size", size, metrics[f"ap_{size}"]) for size in ("small", "medium", "large")]
        groups += [("supercategory", name, summary(coco_eval(ground_truth, boxes, ids))["map_50_95"]) for name, ids in supercategories.items()]
        groups += [("category", row.group, summary(coco_eval(ground_truth, boxes, [int(row.category_ids)]))["map_50_95"]) for row in qualifying.itertuples()]
        group_rows += [{"condition": condition, "group_type": t, "group": g, "ap": ap} for t, g, ap in groups]

    per_image_table = pd.concat(image_tables, ignore_index = True)
    enough = per_image_table.n_objects >= cfg.MIN_OBJECTS_PER_IMAGE
    per_image_table["recall"] = (per_image_table.tp50 / per_image_table.n_objects).where(enough)
    groupe_table = pd.DataFrame(group_rows).merge(counts[["group_type", "group", "n_instances", "exploratory"]], on = ["group_type", "group"], how = "left")

    return pd.DataFrame(metrics_rows), group_table, per_image_table

def comparisons(metrics):
    """ Absolute differences first, then retention and recovery, for every metric. """
    m = metrics.set_index("condition")
    rows = []
    for name in METRICS:
        hr, native, bicubic, sr = (m.loc[c, name] for c in cfg.CONDITIONS)
        rows.append({
            "metrics": name, 
            "native_drop": hr - native, 
            "bicubic_drop": hr - bicubic,
            "sr_minus_bicubic": sr - bicubic,
            "hr_minus_sr": hr - sr,
            "retention_lr_native": native / hr, 
            "retention_lr_bicubic": bicubic / hr,
            "retention_lr_sr": sr / hr,
            "rocovery": recovery(hr, bicubic, sr),
        })

    return pd.DataFrame(rows)

def bootstrap(coco, predictions, resamples):
    """ Paired bootstrap over images: one row per (resample, condition) with the six metrics. """
    rng = np.random.default_rng(cfg.SEED)
    images = {i["id"]: i for i in coco["images"]}
    objects = {}
    for a in coco["annotations"]:
        objects.setdefault(a["image_id"], []).append(a)
    ids = sorted(images)

    rows = []
    for r in range(resamples):
        draw = pd.DataFrame({"new_id": np.arange(len(ids)), "image_id": rng.choice(ids, len(ids))})
        new_images, new_objects = [], []
        for new_id, old_id in zip(draw.new_id, draw.image_id):
            new_images.append({**images[old_id], "id": int(new_id)})
            for a in objects.get(old_id, []):
                new_objects.append({**a, "id": len(new_objects) + 1, "image_id": int(new_id)})
        ground_truth = load_coco({"images": new_images, "annotations": new_objects,
                                  "categories": coco["categories"]})
        for condition in cfg.CONDITIONS:
            boxes = draw.merge(predictions[predictions.condition == condition], on="image_id")
            boxes = boxes.drop(columns="image_id").rename(columns={"new_id": "image_id"})
            rows.append({"resample": r, "condition": condition, **summary(coco_eval(ground_truth, boxes))})
        if r % 50 == 0:
            print(f"bootstrap {r}/{resamples}")

    return pd.DataFrame(rows)

def interval(values):
    """ 95% percentile interval, ignoring undefined (Nan) values. """
    values = np.asarray(values, dtype = np.float)
    values = values[~np.isnan(values)]
    return (np.percentile(values, 2.5), np.percentile(values, 97.5)) if values.size else (np.nan, np.nan)

def bootstrap_intervals(samples, metrics):
    """ Turn the bootstrap samples into point estimates with 95% intervals. """
    wide = samples.pivot(index = "resample", columns = "condition")
    m = metrics.set_index("condition")
    rows = []
    for name in METRICS:
        for condition in cfg.CONDITIONS:
            low, high = interval(wide[name][condition])
            rows.append({"metrics": name, "quantity": condition, "estimate": m.loc[condition, name], "ci_low": low, "ci_high": high})
            for label, (a, b) in DIFFERENCES.items():
                low, high = interval(wide[name][a] - wide[name][b])
                rows.append({"metrics": name, "quantity": label, "estimate": m.loc[a, name] - m.loc[b, name], "ci_low": low, "ci_high": high})

    main = wide["map_50_95"]
    values = [recovery(h, b, s) for h, b, s in zip(main.hr, main.lr_bicubic, main.lr_sr)]
    low, high = interval(values)
    rows.append({
        "metric": "map_50_95",
        "quantity": "recovery",
        "estimate": recovery(m.loc["hr", "map_50_95"], m.loc["lr_bicubic", "map_50_95"], m.loc["lr_sr", "map_50_95"]),
        "ci_low": low,
        "ci_high": high,
        "excluded_share": float(np.isnan(values).mean())
    })
    
    tables = pd.DataFrame(rows)
    table["resamples"] = len(samples) // len(cfg.CONDITIONS)

    return tables

def interpret(intervals):
    """ Apply the pre-declared rule to LR-SR minus LR-BICUBIC in mAP. """
    rows = intervals[(intervals.metrics == "map_50_95") & (intervals.quantity == "lr_sr_minus_lr_bicubic")].iloc[0]
    if row.ci_low <= 0 <= row.ci_high:
        verdict = "no clear difference"
    elif row.estimate >= cfg.MEANINGFUL_EFFECT:
        verdict = "SR helps"
    elif row.estimate > 0:
        verdict = "Detectable but small effect"
    else:
        verdict = "SR is worse than bicubic"

    return pd.DataFrame([{"difference": row.estimate, "ci_low": row.ci_low, "ci_high": row.ci_high, "threshold": cfg.MEANINGFUL_EFFECT, "verdict": verdict}])

def main(resamples = cfg.BOOTSTRAP_RESAMPLES):
    """ Run the pre-declared rule to LR-SR minus LR-Bicubic in mAP. """
    coco = json.loads(cfg.SUBSET_ANNOTATIONS.read_text())
    predictions = load_csv("predictions")
    ground_truth = load_coco(coco)

    metrics, groups, images = point_estimates(ground_truth, predictions, load_csv("object_counts"))
    save_csv(metrics, "metrics")
    save_csv(groups, "group_metrics")
    save_csv(images, "per_image")
    save_csv(comparisons(metrics), "comparisons")
    print(metrics.round(3).to_string(index = False))

    intervals = bootstrap_intervals(bootstrap(coco, predictions, resamples), metrics)
    save_csv(intervals, "bootstrap")
    verdict = interpret(intervals)
    save_csv(verdict, "interpretation")
    print(verdict.round(4).to_string(index = False))

if __name__ == "__main__":
    main()

