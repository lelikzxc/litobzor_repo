#!/usr/bin/env python3
"""Analyze the distribution of image sizes in datasets/wm811k/images.

For every image the script reads its dimensions and reports:
  * width  (w)
  * height (h)
  * area   (h * w)

Two views are printed for each metric:
  1. A clean distribution: summary stats + a histogram over distinct sizes.
  2. A binarized distribution: sizes grouped into 4 bins split at the
     quartiles (0.25, 0.5, 0.75) of the metric, with counts / percentages.

Usage:
    python scripts/image_size_distribution.py
    python scripts/image_size_distribution.py --images-dir datasets/wm811k/images

Dependencies:
    Pillow  (pip install Pillow)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, OrderedDict
from pathlib import Path

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        """Minimal fallback progress reporter when tqdm is unavailable."""
        total = kwargs.get("total")
        for i, item in enumerate(iterable):
            if total:
                print(f"\r  {i + 1}/{total} ({(i + 1) / total * 100:5.1f}%)", end="", flush=True)
            yield item
        if total:
            print()

IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distribution of image sizes (w, h, h*w) in a folder."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("datasets/wm811k/images"),
        help="Directory containing the images.",
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="Quantile cut points for the binarized view.",
    )
    return parser.parse_args()


def collect_sizes(images_dir: Path) -> list[tuple[int, int]]:
    """Return [(w, h), ...] for every image found in images_dir (non-recursive)."""
    if not images_dir.is_dir():
        sys.exit(f"Directory not found: {images_dir}")

    # os.scandir is much faster than pathlib.iterdir().is_file(): it uses the
    # cached directory-entry type instead of doing a separate stat per file,
    # which matters for folders with hundreds of thousands of files. Sorting is
    # skipped because it is not needed for the histogram.
    try:
        entries = [
            e.path
            for e in os.scandir(images_dir)
            if e.is_file() and Path(e.name).suffix.lower() in IMAGE_EXTS
        ]
    except OSError as exc:
        sys.exit(f"Could not list {images_dir}: {exc}")

    if not entries:
        sys.exit(f"No images found in: {images_dir}")

    sizes: list[tuple[int, int]] = []
    for path in tqdm(entries, desc="Reading sizes", unit="img", total=len(entries)):
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not stop the run
            print(f"\n  ! skipped {Path(path).name}: {exc}", file=sys.stderr)
            continue
        sizes.append((w, h))

    return sizes


def _quantile(sorted_vals: list[int], q: float) -> float:
    """Linear-interpolated quantile for a sorted list."""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def print_clean(name: str, values: list[int]) -> None:
    """Print summary stats and a histogram over distinct values."""
    n = len(values)
    sorted_vals = sorted(values)
    uniq = Counter(values)

    mean = sum(values) / n
    median = _quantile(sorted_vals, 0.5)

    print(f"\n=== {name} — clean distribution ===")
    print(f"  count        : {n}")
    print(f"  unique values: {len(uniq)}")
    print(f"  min          : {min(values)}")
    print(f"  max          : {max(values)}")
    print(f"  mean         : {mean:.2f}")
    print(f"  median       : {median:.2f}")

    # Histogram over distinct values, most common first.
    width = max(2, max(len(str(v)) for v in uniq))
    max_count = max(uniq.values())
    bar_scale = 60 / max_count if max_count else 1
    print("  value" + " " * (width - len("value")) + " | count | share   | histogram")
    for value, count in sorted(uniq.items(), key=lambda kv: -kv[1]):
        share = count / n * 100
        bar = "#" * max(1, round(count * bar_scale))
        print(f"  {value:>{width}} | {count:>5} | {share:6.2f}% | {bar}")


def print_binarized(name: str, values: list[int], quantiles: list[float]) -> None:
    """Bin values by the given quantiles and print counts / shares per bin."""
    n = len(values)
    sorted_vals = sorted(values)

    # Build a sorted list of cut points: 0, q..., 1.
    cut_points: OrderedDict[str, float] = OrderedDict()
    cut_points["min"] = float(min(values))
    for q in sorted(quantiles):
        cut_points[f"q{q:g}"] = _quantile(sorted_vals, q)
    cut_points["max"] = float(max(values))

    names = list(cut_points)
    bounds = list(cut_points.values())

    print(f"\n=== {name} — binarized by quantiles {', '.join(f'{q:g}' for q in sorted(quantiles))} ===")
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        lo_name, hi_name = names[i], names[i + 1]

        def _fmt(v: float) -> str:
            """Render as integer when exact, otherwise with 3 significant digits."""
            return str(int(v)) if v == int(v) else f"{v:.3g}"

        lo_str = _fmt(lo)
        hi_str = _fmt(hi)
        if i == 0:
            label = f"[min={lo_str}, {hi_name}={hi_str})"
        elif i == len(bounds) - 2:
            label = f"[{lo_name}={lo_str}, max={hi_str}]"
        else:
            label = f"[{lo_name}={lo_str}, {hi_name}={hi_str})"

        # Count values in [lo, hi), except the last bin which is closed on both sides.
        if i == len(bounds) - 2:
            count = sum(1 for v in values if lo <= v <= hi)
        else:
            count = sum(1 for v in values if lo <= v < hi)

        share = count / n * 100
        bar = "#" * max(1, round(share / 2))
        print(f"  {label:<32} | {count:>5} | {share:6.2f}% | {bar}")


def main() -> None:
    args = parse_args()
    sizes = collect_sizes(args.images_dir)
    if not sizes:
        sys.exit("No images could be read.")

    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    areas = [w * h for w, h in sizes]

    print(f"Total images: {len(sizes)}")
    print(f"Total distinct (w, h) pairs: {len(set(sizes))}")

    for name, values in (("width (w)", widths), ("height (h)", heights), ("area (h*w)", areas)):
        print_clean(name, values)
        print_binarized(name, values, args.quantiles)


if __name__ == "__main__":
    main()