""" Small helpers shared by the experiment. """

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Callable
from PIL import Image

def read_json(path: Path) -> Any:
    """ Read JSON file from 'path' and fail clearly if it is missing or invalid. """
    with path.open(encoding = "utf-8") as handle:
        return json.load(handle)

def write_json(path: Path, value: Any, indent: int | None = 2) -> None:
    """ Write JSON with stable formatting for reproducibility. """
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("w", encoding = "utf-8") as handle:
        json.dump(value, handle, indent = indent, sort_keys = True)

def image_path(directory: Path, image_id: int) -> Path:
    """ Return the deterministic PNG path used for one COCO image ID. """
    return directory / f"{image_id:012d}.png"

def validate_image(path: Path, expected_size: tuple[int, int]) -> None:
    """ Verify that a chached image is readable, RGB, and correctly sized. """
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"{path} has mode {image.mode}, expected RGB")
        if image.size != expected_size:
            raise ValueError(f"{path} has size {image.size}, expected {expected_size}") 

def quiet(function: Callable, *args: Any, **kwargs: Any) -> Any:
    """ Call 'function' with its printed output suppressed. """
    with contectlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)

