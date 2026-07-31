"""WM-811K wafer map dataset loader for SemiWaferNet.

Loads wafer map images from the WM-811K dataset (labels.csv + PNG images)
and returns multitask samples compatible with SemiWaferNet:
    ``{"image": [1, H, W], "label": int, "mask": [H, W]}``

SemiWaferNet operates on **grayscale** (1-channel) images per the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from papers.semiwafernet.data_utils.base import BaseDataset, DatasetType

# WM-811K class names (9 classes)
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


class WaferWM811KDataset(BaseDataset):
    """WM-811K wafer map classification dataset for SemiWaferNet.

    Reads ``labels.csv`` from the dataset root and loads PNG images.
    Returns multitask samples with image, label, and a dummy mask
    (since WM-811K only has classification labels, not segmentation masks).

    Args:
        data_root: Root directory containing ``labels.csv`` and ``images/``.
        image_size: Target image size (assumed square, default 32 for classification).
        num_classes: Number of classes (default 9 for WM-811K).
    """

    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 32,
        num_classes: int = 9,
        transform: callable | None = None,
    ) -> None:
        super().__init__(dataset_type=DatasetType.MULTITASK)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_names = WM811K_CLASSES
        self.transform = transform

        self.labels_path = self.data_root / "labels.csv"
        self.images_dir = self.data_root / "images"

        self._samples: list[tuple[str, int]] = []  # (filename, label_idx)
        self._load_labels()

    def _load_labels(self) -> None:
        """Parse labels.csv and build (filename, label_idx) pairs."""
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        with open(self.labels_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().lower()
            if "image" in header and "label" in header:
                # Standard format: image,label
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        filename = parts[0].strip()
                        label_str = parts[1].strip()
                        try:
                            label_idx = int(label_str)
                        except ValueError:
                            # Try matching class name
                            label_idx = WM811K_CLASSES.index(label_str) if label_str in WM811K_CLASSES else 0
                        self._samples.append((filename, label_idx))
            else:
                # Fallback: try to parse anyway
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        self._samples.append((parts[0].strip(), int(parts[1].strip())))

        if not self._samples:
            raise ValueError(f"No samples loaded from {self.labels_path}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename, label = self._samples[index]
        image_path = self.images_dir / filename

        # Load image as grayscale (1 channel) — WM-811K is grayscale
        if image_path.exists():
            image = Image.open(image_path).convert("L")
        else:
            # Fallback: try with .png extension
            png_path = self.images_dir / f"{Path(filename).stem}.png"
            if png_path.exists():
                image = Image.open(png_path).convert("L")
            else:
                raise FileNotFoundError(f"Image not found: {image_path} or {png_path}")

        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Apply augmentation transform (if any) BEFORE converting to tensor
        if self.transform is not None:
            # Convert to PIL RGB for torchvision transforms, then back to grayscale
            image_rgb = image.convert("RGB")
            image_rgb = self.transform(image_rgb)
            # Convert back to grayscale tensor [1, H, W]
            image = image_rgb.mean(dim=0, keepdim=True)  # [1, H, W]
        else:
            # Convert to tensor [1, H, W], normalize to [0, 1]
            image = torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0

        # Dummy mask (zeros) — WM-811K has no segmentation masks
        mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)

        return {
            "image": image,      # [1, H, W]
            "label": label,      # int
            "mask": mask,        # [H, W] dummy
        }

    @property
    def num_samples(self) -> int:
        return len(self._samples)