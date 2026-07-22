"""WM-811K wafer map dataset loader for FCS-VMamba.

Provides ``WaferWM811KDataset`` that reads ``labels.csv`` and loads
grayscale wafer map images from the ``images/`` directory, converting
them to 3-channel RGB (required by VMamba).

The WM-811K dataset contains 9 defect classes:
    Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, none, Random, Scratch

Each sample is a dict with ``"image"`` (``torch.Tensor [3, H, W]`` RGB)
and ``"label"`` (``int`` class index).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize

from papers.vit_tiny.data_utils.base import BaseDataset, DatasetType

# Label-to-index mapping for WM-811K (9 classes)
WM811K_CLASSES: list[str] = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]

WM811K_LABEL_TO_IDX: dict[str, int] = {
    label: idx for idx, label in enumerate(WM811K_CLASSES)
}


class WaferWM811KDataset(BaseDataset):
    """WM-811K wafer map classification dataset for FCS-VMamba.

    Reads ``labels.csv`` for image-to-label mapping and loads grayscale
    PNG images from ``images/``, converting to 3-channel RGB.

    Args:
        data_root: Root directory containing ``labels.csv`` and ``images/``.
        image_size: Target image size (assumed square, default 224).
        transform: Optional transform to apply to images (applied after resize).
    """

    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 224,
        transform: callable | None = None,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION, transform=transform)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.image_dir = self.data_root / "images"
        self.labels_path = self.data_root / "labels.csv"

        # Resize transform for WM-811K images (native 128x128 → target size)
        self._resize = Resize(image_size, interpolation=Image.BILINEAR)

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.image_dir}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        # Load labels
        self._samples: list[tuple[str, int]] = []  # (filename, label_idx)
        self._load_labels()

    def _load_labels(self) -> None:
        """Parse labels.csv and build (filename, label_idx) pairs."""
        with open(self.labels_path, "r", encoding="utf-8") as f:
            header = f.readline().strip()  # skip header: image,label
            if header != "image,label":
                f.seek(0)

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                filename = parts[0].strip()
                label_str = parts[1].strip()
                label_idx = WM811K_LABEL_TO_IDX.get(label_str, 0)
                self._samples.append((filename, label_idx))

        if not self._samples:
            raise ValueError(f"No samples loaded from {self.labels_path}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename, label_idx = self._samples[index]
        image_path = self.image_dir / filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load as grayscale, then convert to 3-channel RGB
        image = Image.open(image_path).convert("L")
        # Resize to target size
        image = self._resize(image)
        # Convert grayscale to RGB: [H, W] → [H, W, 3]
        image = image.convert("RGB")

        # Apply additional transform if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: PIL RGB → [3, H, W] float32 tensor
            image = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)) / 255.0

        return {
            "image": image,
            "label": label_idx,
        }

    @property
    def num_classes(self) -> int:
        """Return the number of classes (9 for WM-811K)."""
        return len(WM811K_CLASSES)

    @property
    def class_names(self) -> list[str]:
        """Return the list of class names."""
        return list(WM811K_CLASSES)