"""WM-38k wafer map dataset loader with augmentations.

Provides ``WaferWM38KDataset`` that reads ``labels.csv`` and loads
grayscale wafer map images from the ``images/`` directory.

The WM-38k dataset contains 38 defect classes (each unique combination
of the 8 base defect types is a distinct class), matching the paper:

    "Semiconductor Wafer Map Defect Classification with
     Tiny Vision Transformers" (arXiv:2504.02494)

Preprocessing follows Section III-A of the paper:
    1. Data splitting (80:20 train/test)
    2. Normalization (pixel values scaled to [0, 1])
    3. Resizing to 224x224 via bilinear interpolation
    4. Grayscale -> RGB conversion (done in the model via 1x1 Conv)
    5. Augmentation (rotation, flipping, zooming)

Each sample is a dict with ``"image"`` (``torch.Tensor [3, 224, 224]``
RGB, ImageNet-normalized) and ``"label"`` (``int`` class index 0..37).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import (
    Compose,
    Grayscale,
    Normalize,
    RandomAffine,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from papers.vit_tiny.data_utils.base import BaseDataset, DatasetType

# Number of classes in the WM-38k dataset (Table III of the paper)
WM38K_NUM_CLASSES = 38

# ImageNet normalization used by the pretrained ViT (ViTImageProcessor)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_train_transform(image_size: int = 224) -> Compose:
    """Build default training augmentation pipeline for WM-38k wafer maps.

    Applies (Section III-A.5):
        - Random rotation (rotation)
        - Random horizontal flip (flipping)
        - Random resized crop / zoom (zooming via RandomResizedCrop)
        - Resize to target size
        - Convert to tensor + ImageNet normalization

    Args:
        image_size: Target spatial size (H == W).

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    return Compose([
        RandomAffine(degrees=15, scale=(0.9, 1.1)),
        RandomHorizontalFlip(p=0.5),
        RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        Grayscale(num_output_channels=3),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def default_val_transform(image_size: int = 224) -> Compose:
    """Build default validation/test transform (no augmentation).

    Args:
        image_size: Target spatial size (H == W).

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    return Compose([
        Resize(image_size, interpolation=Image.BILINEAR),
        Grayscale(num_output_channels=3),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class WaferWM38KDataset(BaseDataset):
    """WM-38k wafer map classification dataset (38 classes).

    Reads ``labels.csv`` for image-to-label mapping and loads grayscale
    PNG images from ``images/``. Images are resized to ``image_size``
    and converted to RGB (3-channel) for the pretrained ViT.

    Args:
        data_root: Root directory containing ``labels.csv`` and ``images/``.
        image_size: Target image size (assumed square, default 224).
        transform: Optional transform to apply to images.
            If ``None``, uses a default ToTensor conversion (no augmentation).
        train: If ``True``, applies training augmentations by default
            (when ``transform`` is not explicitly set).
    """

    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 224,
        transform: callable | None = None,
        train: bool = True,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION, transform=transform)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.image_dir = self.data_root / "images"
        self.labels_path = self.data_root / "labels.csv"

        # Use default transforms if none provided
        if transform is None:
            if train:
                self.transform = default_train_transform(image_size)
            else:
                self.transform = default_val_transform(image_size)

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
                # Try to handle files without header
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
                try:
                    label_idx = int(label_str)
                except ValueError:
                    continue
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

        # Load as grayscale
        image = Image.open(image_path).convert("L")

        # Apply transform (includes resize + ToTensor + normalize)
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Fallback: manual conversion to RGB [3, H, H] tensor
            image = torch.from_numpy(np.array(image, dtype=np.float32)) / 255.0
            image = image.unsqueeze(0)  # [H, W] -> [1, H, W]
            image = image.repeat(3, 1, 1)  # grayscale -> RGB

        return {
            "image": image,
            "label": label_idx,
        }

    @property
    def num_classes(self) -> int:
        """Return the number of classes (38 for WM-38k)."""
        return WM38K_NUM_CLASSES

    @property
    def class_names(self) -> list[str]:
        """Return the list of class names (C1..C38)."""
        return [f"C{i+1}" for i in range(WM38K_NUM_CLASSES)]