"""Segmentation dataset for SemiWaferNet (ConvoFormer-UNet).

Reads the pre-generated WM-811K segmentation dataset (``datasets/wm811k_seg``)
produced by ``scripts/create_wm811k_segmentation_dataset.py``. That script:

    - Excludes the ``None`` and ``Random`` classes (paper Section 4.1).
    - Generates binary masks from defective die (waferMap == 2 / value 254).

Per the paper Section 4.1, wafer maps are resized to ``64x64`` and the
corresponding binary masks are resized using **nearest-neighbor** interpolation
to preserve label consistency.

Each sample is a dict:
    ``{"image": [1, 64, 64], "mask": [64, 64] (long, 0/1)}``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from papers.semiwafernet.data_utils.base import BaseDataset, DatasetType


class WaferSegmentationDataset(BaseDataset):
    """Binary wafer-defect segmentation dataset from ``datasets/wm811k_seg``.

    Args:
        data_root: Root of the segmentation dataset (contains ``train/``,
            ``val/``, ``test/``, each with ``images/`` and ``masks/``).
        split: One of ``"train"``, ``"val"``, ``"test"``.
        image_size: Target spatial size (default 64 per paper Section 4.1).
        train: If ``True``, apply training augmentations.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        image_size: int = 64,
        train: bool = True,
    ) -> None:
        super().__init__(dataset_type=DatasetType.SEGMENTATION)
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.train = train

        self.images_dir = self.data_root / split / "images"
        self.masks_dir = self.data_root / split / "masks"

        if not self.images_dir.exists() or not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Segmentation split '{split}' not found under {self.data_root}. "
                f"Run scripts/create_wm811k_segmentation_dataset.py first."
            )

        self._image_paths = sorted(self.images_dir.glob("*.png"))
        if not self._image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self._image_paths[index]
        mask_path = self.masks_dir / image_path.name

        # Load image as grayscale [0, 255]
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Resize image (bilinear) and mask (nearest-neighbor) to target size.
        # Nearest-neighbor preserves label consistency (paper Section 4.1).
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # Training augmentations (dynamic augmentation, paper Section 4.1)
        if self.train:
            image, mask = self._augment(image, mask)

        # Image tensor [1, H, W] in [0, 1]
        img_arr = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)

        # Mask tensor [H, W] long with values {0, 1}
        mask_arr = np.array(mask, dtype=np.int64)
        # Normalise mask values: 255 (defect) -> 1, else 0
        mask_tensor = torch.from_numpy((mask_arr > 0).astype(np.int64))

        return {
            "image": img_tensor,
            "mask": mask_tensor,
        }

    def _augment(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        """Apply identical geometric augmentation to image and mask."""
        import random

        # Random horizontal flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Random rotation in {-90, 0, 90, 180}
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
            mask = mask.rotate(angle, resample=Image.NEAREST, expand=False)

        return image, mask


__all__ = ["WaferSegmentationDataset"]