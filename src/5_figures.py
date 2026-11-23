""" Create figures for the results. """

import json

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import config as cfg
from helpers import image_path, load_csv, save_csv, size_group

#--------------------------------
# Colors & Labels
#--------------------------------
COLORS = {
    "hr": "#2a78d6", 
    "lr_native": "#eb6834", 
    "lr_bicubic": "#1baf7a", 
    "lr_sr": "#eda100"
}

LABELS = {
    "hr": "HR", 
    "lr_native": "LR-native", 
    "lr_bicubic": "LR-bicubic", 
    "lr_sr": "LR-SR"
}

plt.rcParams.update({
    "axes.spines.top": False, 
    "axes.spines.right": False, 
    "axes.grid": True, 
})

#--------------------------------
# Helpers
#--------------------------------
def save_figures(figure, name):
    """ Save figures as results/<name>.png and results/<name>.pdf. """
    cfg.FIGURES_DIR.mkdir(parents = True, exist_ok = True)
    figure.tight_layout()
    figure.savefig(cfg.FIGURES_DIR / f"{name}.png", dpi = 200)
    plt.close(figure)

def bars_with_intervals(axis, rows, colors):
    """ Draw one bar per row of 'row' with error bars. """
    positions = range(len(rows))
    axis.bar(positions, rows.estimate, color = colors, width = 0.6)
    below = (rows.estimate - rows.ci_low).clip(lower = 0)
    aboce = (rows.ci_high - rows.estimate).clip(lower = 0)
    axis.errorbar(positions, rows.estimate, yerr = [below, above], fmt = "none", ecolor = "#3d3d3a", capsize = 4)

#--------------------------------
# Result figures
#--------------------------------
def plot_map_by_condition(intervals):
    """ mAP per condition iwth 95%. paired bootstrap intervals. """
    rows = intervals[intervals.metric == "map_50_95"].set_index("quantity").loc[cfg.CONDITIONS]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars_with_intervals(axis, rows, [COLORS[c] for c in cfg.CONDITIONS])
    axis.set_xticks(range(4), [LABELS[c] for c in cfg.CONDITIONS])
    axis.set_ylabel("mAP@[0.50:0.95]")
    axis.set_title("Detection performance by input condition")
    
    save_figure(figure, "map_by_condition")

def plot_ap_by_size(groups):
    """ AP per size group, one line per condition. """
    sizes = ["small", "medium", "large"]
    table = groups[groups.group_type == "size"]
    counts = table.drop_duplicates("group").set_index("group").n_instances
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for condition in cfg.CONDITIONS:
        ap = table[table.condition == condition].set_index("group").loc[sizes, "ap"]
        axis.plot(range(3), ap, marker="o", linewidth=2, color=COLORS[condition], label=LABELS[condition])
    axis.set_xticks(range(3), [f"{s}\n(n = {counts[s]})" for s in sizes])
    axis.set_ylabel("AP@[0.50:0.95]")
    axis.legend(frameon=False)
    
    save_figure(figure, "ap_by_size")

def plot_sr_gain_by_size(intervals):
    """ LR-SR minus LR-bicubic, overall and per size, with the 0.01 threshold. """
    metrics = ["map_50_95", "ap_small", "ap_medium", "ap_large"]
    rows = intervals[intervals.quantity == "lr_sr_minus_lr_bicubic"].set_index("metric").loc[metrics]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars_with_intervals(axis, rows, COLORS["lr_sr"])
    axis.axhline(0, color="#9a9a95", linewidth=1)
    axis.axhline(cfg.MEANINGFUL_EFFECT, color="#3d3d3a", linestyle="--", linewidth=1, label="pre-declared threshold (0.010)")
    axis.set_xticks(range(4), ["all objects", "small", "medium", "large"])
    axis.set_ylabel("LR-SR minus LR-bicubic (AP)")
    axis.legend(frameon=False)
    
    save_figure(figure, "sr_gain_by_size")

def plot_group_gain(groups, group_type):
    """ LR-SR minus LR-bicubic per supercategory or category (point estimates, with counts). """
    table = groups[groups.group_type == group_type]
    if table.empty:
        return
    ap = table.pivot(index="group", columns="condition", values="ap")
    info = table.drop_duplicates("group").set_index("group")
    gain = (ap["lr_sr"] - ap["lr_bicubic"]).sort_values()
    figure, axis = plt.subplots(figsize=(7, 0.4 * len(gain) + 1.5))
    axis.barh(range(len(gain)), gain, height=0.6, color=["#9a9a95" if info.loc[g, "exploratory"] else COLORS["lr_sr"] for g in gain.index])
    axis.set_yticks(range(len(gain)), [f"{g} (n = {info.loc[g, 'n_instances']})" for g in gain.index])
    axis.axvline(0, color="#9a9a95", linewidth=1)
    axis.grid(axis="x", color="#e5e5e2")
    axis.set_xlabel("LR-SR minus LR-bicubic (AP)")
    axis.set_title(f"SR gain per {group_type} (grey = exploratory, fewer than {cfg.MIN_INSTANCES} objects)")
    
    save_figure(figure, f"{group_type}_gain")

def plot_quality_vs_detection(quality, images):
    """ Does a sharper-looking image (higher PSNR) also give better detection? one dot per image. """
    recall = images.pivot(index="image_id", columns="condition", values="recall")
    table = quality.set_index("image_id").assign(
        psnr_gain=lambda t: t.psnr_sr - t.psnr_bicubic,
        recall_gain=recall.lr_sr - recall.lr_bicubic).dropna(subset=["recall_gain"])
    
    # Spearman's rank correlation = ordinary correlation of the ranks (no straight line assumed)
    rho = table.psnr_gain.rank().corr(table.recall_gain.rank())
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter(table.psnr_gain, table.recall_gain, s=15, alpha=0.6, color=COLORS["lr_sr"])
    axis.axhline(0, color="#9a9a95", linewidth=1)
    axis.set_xlabel("PSNR gain of LR-SR over LR-bicubic (dB)")
    axis.set_ylabel("Recall gain at score >= 0.25")
    axis.set_title(f"Image quality vs. detection (Spearman rho = {rho:.2f}, n = {len(table)})")
    
    save_figure(figure, "quality_vs_detection")

def plot_accuracy_vs_latency(metrics, timing):
    """ mAP against end-to-end time per image. """
    table = metrics.merge(timing, on="condition")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for row in table.itertuples():
        axis.scatter(row.total_ms, row.map_50_95, s=80, color=COLORS[row.condition], zorder=3)
        axis.annotate(LABELS[row.condition], (row.total_ms, row.map_50_95), xytext=(6, 4), textcoords="offset points")
    axis.set_xlabel(f"Time per image, upscaling + YOLO (ms, {timing.gpu.iloc[0]})")
    axis.set_ylabel("mAP@[0.50:0.95]")

    save_figure(figure, "accuracy_vs_latency")

#--------------------------------
# Qualitative examples
#--------------------------------
def pick_examples(images):
    """ Choose one image per rule from per_image.csv (images with at least 3 objects). """
    table = images[images.n_objects >= cfg.MIN_OBJECTS_PER_IMAGE]
    wide = table.pivot(index="image_id", columns="condition")          # columns: (value, condition)
    gain = wide.recall.lr_sr - wide.recall.lr_bicubic
    same = wide.tp50.lr_sr == wide.tp50.lr_bicubic
    examples = {
        "sr_helps": gain.idxmax(),
        "equivalent": wide.n_objects.hr[same].idxmax(),
        "sr_hurts": gain.idxmin(),
        "native_misses": (wide.tp50.hr - wide.tp50.lr_native).idxmax(),
        "localization": (wide.tp75.lr_sr - wide.tp75.lr_bicubic).abs()[same].idxmax(),
    }

    return pd.DataFrame({"rule": list(examples), "image_id": [int(i) for i in examples.values()]})

def draw_example(image_id, rule, coco, prediction):
    """ HR, LR-native, LR-bicubic, LR-SR side by side. """
    names = {c["id"]: c["name"] for c in coco["categories"]}
    truth = [a for a in coco["annotations"] if a["image_id"] == image_id and not a["iscrowd"]]
    boxes = predictions[(predictions.image_id == image_id) & (predictions.score >= cfg.REPORT_CONF)]
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    for axis, condition in zip(axes, cfg.CONDITIONS):
        image = Image.open(image_path(condition, image_id))
        if condition == "lr_native":
            image = image.resize((image.width * cfg.SCALE, image.height * cfg.SCALE), Image.NEAREST)
        axis.imshow(image)
        for a in truth:
            x, y, w, h = a["bbox"]
            axis.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="#9a9a95", linestyle="--"))
            axis.text(x + w, y + h, size_group(a["area"])[0].upper(), fontsize=7, color="#3d3d3a", ha="right", va="bottom")
        shown = boxes[boxes.condition == condition]
        for b in shown.itertuples():
            axis.add_patch(plt.Rectangle((b.x, b.y), b.w, b.h, fill=False, edgecolor="#2a78d6", linewidth=2))
            axis.text(b.x, b.y - 2, f"{names[b.category_id]} {b.score:.2f}", fontsize=7, color="#2a78d6")
        axis.set_title(f"{LABELS[condition]} ({len(shown)} boxes)")
        axis.axis("off")
    figure.suptitle(f"Image {image_id}: {rule.replace('_', ' ')}")

    save_figure(figure, f"example_{rule}_{image_id}")

def main():
    """ Make every figure and qualitative example. """
    intervals, groups, images = load_csv("bootstrap"), load_csv("group_metrics"), load_csv("per_image")
    plot_map_by_condition(intervals)
    plot_ap_by_size(groups)
    plot_sr_gain_by_size(intervals)
    plot_group_gain(groups, "supercategory")
    plot_group_gain(groups, "category")
    plot_quality_vs_detection(load_csv("image_quality"), images)
    plot_accuracy_vs_latency(load_csv("metrics"), load_csv("timing"))

    examples = pick_examples(images)
    save_csv(examples, "qualitative_examples")          # documents which image each rule chose
    coco = json.loads(cfg.SUBSET_ANNOTATIONS.read_text())
    predictions = load_csv("predictions")
    for row in examples.itertuples():
        draw_example(row.image_id, row.rule, coco, predictions)
    print("Figures saved to", cfg.FIGURES_DIR)


if __name__ == "__main__":
    main()