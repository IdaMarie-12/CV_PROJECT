""" Measure how long each pipeline takes, and how big the models are. """

import json
import statistics
import time

import pandas as pd 
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCountAnalysis
from ultralytics import YOLO

import config as cfg
from helpers import image_path, save_csv
from prepare_data import load_edsr, super_resolve

def miadian_ms(function, images, gpu):
    """ Median time in ms of 'finction(image)' over 100 different images. """
    for image in image[:cfg.TIMING_WARMUP]:
        function(image)
    times = []
    for image in images[cfg.TIMING_WARMUP:cfg.TIMING_WARMUPS + cfg.TIMING_REPEATS]:
        if gpu:
            torch.cuda.synchronize()
        start = time.perf_counter()
        function(image)
        if gpu:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)

def gflops(network, example):
    """ Floating-point operations of one forward pass in billions. """
    with FlopCounterMode(display = False) as counter, torch.no_grad():
        network(example)
    return counter.get_total_flops() / 1e9

def main():
    """ Time every condition and count parameters and GFLOPS. """
    gpu = torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")
    coco = json.loads(cfg.SUBSET_ANNOTATIONS.read_text())
    ids = [i["id"] for i in coco["images"][:cfg.TIMING_WARMUPS + cfg.TIMING_REPEATS]]
    images = {c: [Image.open(image_path(c, i)).convert("RGB") for i in ids] for c in cfg.CONDITIONS}

    edsr = load_edsr(device)
    yolo = YOLO(cfg.YOLO_WEIGHTS)

    # Upscaling time per method (none for hr and lr_native)
    upscale = {
        "hr": 0, 
        "lr_native": 0,
        "lr_bicubic": median_ms(lambda im: im.resize((im.width * cfg.SCALE, im.height * cfg.SCALE), Image.BICUBIC), images["lr_native"], gpu = False), 
        "lr_sr": median_ms(lambda im: super_resolver(edsr, im, device), images["lr_native"], gpu)
    }

    rows = []
    for condition in cfg.CONDITIONS:
        yolo_ms = median_ms(lambda im: yolo.predict(im, imgsz = cfg.YOLO_SIZE[condition], conf = cfg.DETECT_CONF, max_det = cfg.MAX_DETS, verbose = False), images[condition], gpu)
        rows.append({
            "condition": condition, 
            "yolo_size": cfg.YOLO_SIZE[condition],
            "upscale_ms": upscale[condition], 
            "yolo_ms": yolo_ms,
            "total_ms": upscale[condition] + yolo_ms,
            "gpu": torch.cuda.get_device_name(0) if gpu else "cpu",
            "batch_size": 1,
            "torch_version": torch.__version__,
        })

    save_csv(pd.DataFrame(rows), "timing")

    # Model size: parameters (millions) and GLFPOS for the input size actually used
    network = yolo.model.to(device).eval()
    lr_width, lr_height = images["lr_native"][0].size
    cost = pd.DataFrame([
        {"model": "yolo11m", "input": "640x640", "params_millions": sum(p.numel() for p in network.parameters()) / 1e6,
         "gflops": gflops(network, torch.zeros(1, 3, 640, 640, device=device))},
        {"model": "yolo11m", "input": "160x160", "params_millions": sum(p.numel() for p in network.parameters()) / 1e6,
         "gflops": gflops(network, torch.zeros(1, 3, 160, 160, device=device))},
        {"model": "edsr_x4", "input": f"{lr_width}x{lr_height}", "params_millions": sum(p.numel() for p in edsr.parameters()) / 1e6,
         "gflops": gflops(edsr, torch.zeros(1, 3, lr_height, lr_width, device=device))},
    ])
    save_csv(cost, "model_cost")
    print(pd.DataFrame(rows)[["condition", "upscale_ms", "yolo_ms", "total_ms"]].round(1).to_string(index=False))
    print(cost.round(2).to_string(index=False))


if __name__ == "__main__":
    main()