""" Evaluate saved predictions and compute bootstrap intervals. """

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import config as C
from common import quiet, read_json, write_json

# Index of each reported metris in COCOevals 12-element 'stats' vector
METRICS = {"map_50_95": 0, "map_50": 1, "ap_small": 3, "ap_medium": 4, "ap_large": 5, "ar_100": 8}
IOU_75_INDEX = 5

