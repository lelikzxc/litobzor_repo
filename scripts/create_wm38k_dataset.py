"""Prepare the WM-38k wafer map dataset for the ViT-Tiny pipeline.

Reads ``datasets/wm38k.npz`` (images ``[N, 52, 52]`` + one-hot labels
``[N, 8]``) and writes a structured dataset to ``datasets/wm38k/``:

    datasets/wm38k/
        images/          # PNG wafer maps (52x52, grayscale)
        labels.csv       # image,label  (label = 0..37 class index)

The 8 one-hot defect labels are mapped to 38 classes (each unique
combination of defects is a distinct class), ordered by the number of
defects (0, 1, 2, 3, 4) to match the paper's Table III grouping.

Usage:
    python scripts/create_wm38k_dataset.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = PROJECT_ROOT / "datasets" / "wm38k.npz"
OUT_DIR = PROJECT_ROOT / "datasets" / "wm38k"
IMAGE_DIR = OUT_DIR / "images"
LABELS_PATH = OUT_DIR / "labels.csv"


def build_class_mapping(labels: np.ndarray) -> dict[tuple[int, ...], int]:
    """Map each unique one-hot combination to a class index.

    Classes are ordered by the number of defects (0, 1, 2, 3, 4) to
    mirror the paper's Table III grouping (normal, single, 2-mixed,
    3-mixed, 4-mixed). Within the same defect count, combinations are
    ordered by their binary value for determinism.

    Args:
        labels: One-hot label matrix ``[N, 8]``.

    Returns:
        Mapping from ``tuple[int, ...]`` (8-bit combination) to class index.
    """
    unique = np.unique(labels, axis=0)
    # Sort by (number of defects, binary value)
    unique_sorted = sorted(
        unique.tolist(),
        key=lambda combo: (sum(combo), int("".join(map(str, combo)), 2)),
    )
    return {tuple(combo): idx for idx, combo in enumerate(unique_sorted)}


def main() -> None:
    """Run the dataset preparation."""
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

    print(f"Loading WM-38k from: {NPZ_PATH}")
    data = np.load(NPZ_PATH, allow_pickle=True)
    images = data["arr_0"]  # [N, 52, 52] int32
    labels = data["arr_1"]  # [N, 8] int32

    print(f"  Images: {images.shape}, Labels: {labels.shape}")

    class_mapping = build_class_mapping(labels)
    num_classes = len(class_mapping)
    print(f"  Unique defect combinations (classes): {num_classes}")

    # Create output directories
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Write images and labels
    rows: list[tuple[str, int]] = []
    for i in range(len(images)):
        combo = tuple(labels[i].tolist())
        class_idx = class_mapping[combo]

        # Normalize wafer map to [0, 255] uint8 for PNG storage
        img = images[i].astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)
        img_uint8 = img.astype(np.uint8)

        filename = f"wafer_{i:06d}.png"
        Image.fromarray(img_uint8, mode="L").save(IMAGE_DIR / filename)
        rows.append((filename, class_idx))

    print(f"  Wrote {len(images)} images to {IMAGE_DIR}")

    # Write labels.csv
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        for filename, class_idx in rows:
            writer.writerow([filename, class_idx])

    print(f"  Wrote labels.csv -> {LABELS_PATH}")

    print(f"\nDataset ready at: {OUT_DIR}")
    print(f"  Total samples: {len(images)}")
    print(f"  Classes: {num_classes}")


if __name__ == "__main__":
    main()