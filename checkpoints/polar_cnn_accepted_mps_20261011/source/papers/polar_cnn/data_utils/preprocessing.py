"""CPU-only categorical preprocessing, identical on MPS and CUDA."""

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

from papers.wafer_encoding import decode_die_map

RADIAL_EDGES = np.arange(0, 15, 2, dtype=np.float64)
# Seven sectors, as stated in Sec. 3.3: the last sector covers 300..360.
ANGULAR_EDGES = np.array([0, 50, 100, 150, 200, 250, 300, 360], dtype=np.float64)


def denoise(die_map, threshold=3):
    """Simultaneously remove defect centres having < N defects in 3x3.

    The centre itself is included. Passed dies/background are unchanged;
    removed defects become passed dies (1), never background (0).
    """
    if not isinstance(threshold, int) or not 0 <= threshold <= 9:
        raise ValueError("denoise_threshold must be an integer in [0,9]")
    result = np.array(die_map, dtype=np.uint8, copy=True)
    defects = result == 2
    neighbours = convolve(
        defects.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant", cval=0
    )
    result[defects & (neighbours < threshold)] = 1
    return result


def polar_counts(die_map):
    """Count defects by radius (rows) and CCW angle (columns).

    Centre is (size-1)/2; positive y points upwards. Radius >14 is outside
    the paper's stated range and excluded. No density normalisation.
    """
    arr = np.asarray(die_map)
    if arr.shape != (26, 26):
        raise ValueError("polar_counts expects a 26x26 categorical map")
    y, x = np.nonzero(arr == 2)
    dx, dy = x - 12.5, 12.5 - y
    radii = np.hypot(dx, dy)
    angles = np.degrees(np.arctan2(dy, dx)) % 360
    hist, _, _ = np.histogram2d(radii, angles, bins=(RADIAL_EDGES, ANGULAR_EDGES))
    return hist.astype(np.float32)


def native_polar_counts(die_map):
    """Diagnostic: preserve native defects, map pixel centres into 26x26 space.

    Area weighting makes counts comparable across native map sizes. This is
    an experimental alternative to lossy nearest resizing, not a paper claim.
    """
    arr = np.asarray(die_map)
    height, width = arr.shape
    y, x = np.nonzero(arr == 2)
    dx, dy = (x + 0.5) * 26 / width - 13, 13 - (y + 0.5) * 26 / height
    hist, _, _ = np.histogram2d(
        np.hypot(dx, dy),
        np.degrees(np.arctan2(dy, dx)) % 360,
        bins=(RADIAL_EDGES, ANGULAR_EDGES),
        weights=np.full(len(x), 26 * 26 / (height * width)),
    )
    return hist.astype(np.float32)


def preprocess(
    image,
    threshold=3,
    angle=0,
    denoise_order=None,
    count_transform="raw",
    map_resolution="resized26",
):
    """Baseline: decode, nearest resize, denoise once, optionally rotate.

    Rotation interpolation/border policy is unspecified in the article;
    nearest interpolation and background fill are local explicit choices.
    Experimental options filter before resizing, log-scale the counts, or
    preserve native defects by scaling their coordinates instead of the map.
    """
    if denoise_order is None:
        denoise_order = "before_resize" if map_resolution == "native_scaled" else "after_resize"
    if denoise_order not in {"after_resize", "before_resize"}:
        raise ValueError("Invalid denoise_order")
    if count_transform not in {"raw", "log1p"}:
        raise ValueError("Invalid count_transform")
    if map_resolution not in {"resized26", "native_scaled"}:
        raise ValueError("Invalid map_resolution")
    if map_resolution == "native_scaled" and denoise_order != "before_resize":
        raise ValueError("Native sectors require native denoising before coordinate scaling")
    arr = decode_die_map(image).astype(np.uint8)
    if map_resolution == "native_scaled":
        image = Image.fromarray(denoise(arr, threshold))
        if angle % 360:
            image = image.rotate(float(angle), resample=Image.Resampling.NEAREST, fillcolor=0)
        counts = native_polar_counts(np.asarray(image))
        return np.log1p(counts) if count_transform == "log1p" else counts
    if denoise_order == "before_resize":
        arr = denoise(arr, threshold)
    image = Image.fromarray(arr).resize((26, 26), Image.Resampling.NEAREST)
    if denoise_order == "after_resize":
        image = Image.fromarray(denoise(np.asarray(image), threshold))
    if angle % 360:
        image = image.rotate(float(angle), resample=Image.Resampling.NEAREST, fillcolor=0)
    counts = polar_counts(np.asarray(image))
    return np.log1p(counts) if count_transform == "log1p" else counts


def preprocessing_options(config):
    """Optional diagnostic choices; omitted keys preserve the paper baseline."""
    return {
        "denoise_order": config.get(
            "denoise_order",
            "before_resize" if config.get("map_resolution") == "native_scaled" else "after_resize",
        ),
        "count_transform": config.get("count_transform", "raw"),
        "map_resolution": config.get("map_resolution", "resized26"),
    }
