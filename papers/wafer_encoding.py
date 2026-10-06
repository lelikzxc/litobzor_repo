"""Decode categorical WM-811K PNGs without guessing from observed levels."""

import numpy as np
from PIL import Image


def decode_die_map(image):
    arr = np.asarray(image.convert("L") if isinstance(image, Image.Image) else image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    levels = set(np.unique(arr).tolist())
    if levels <= {0, 1, 2}:
        return arr.astype(np.int64)
    if levels <= {0, 127, 254}:
        return (arr // 127).astype(np.int64)
    if levels <= {0, 128, 255}:
        return np.where(arr == 255, 2, np.where(arr == 128, 1, 0)).astype(np.int64)
    raise ValueError(
        f"Unknown wafer PNG encoding: levels={sorted(levels)[:12]}; expected categorical 0/1/2"
    )
