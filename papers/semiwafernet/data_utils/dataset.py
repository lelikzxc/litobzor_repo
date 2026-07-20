"""Multitask dataset adapter for SemiWaferNet built on common.datasets.

Provides ``LabeledWaferDataset`` and ``UnlabeledWaferDataset`` that extend
``common.datasets.BaseDataset`` for the SemiWaferNet semi-supervised pipeline.

Labeled samples return ``{"image": [3, H, W], "label": int, "mask": [H, W]}``
— compatible with ``common.datasets.multitask_collate``.

Unlabeled samples return ``{"image": [3, H, W]}`` — for semi-supervised
training where targets are generated as pseudo-labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from common.datasets import BaseDataset, DatasetType


class LabeledWaferDataset(BaseDataset):
    """Labeled multitask dataset for SemiWaferNet.

    Extends ``common.datasets.BaseDataset`` with a multitask interface.
    Each sample is a dict with ``"image"`` (``torch.Tensor [3, H, W]``),
    ``"label"`` (``int``), and ``"mask"`` (``torch.Tensor [H, W]``).

    SemiWaferNet operates on **RGB** (3-channel) images. Synthetic samples
    are generated as ``[3, H, W]`` tensors for images, ``[H, W]`` for masks
    (integer class indices), and ``int`` for labels.

    For synthetic / test usage, pass ``synthetic_size`` to generate random
    tensors without real files.

    Args:
        image_dir: Directory containing input images.
        mask_dir: Directory containing segmentation mask files.
        image_size: Target image size (assumed square, default 512).
        transform: Optional transform to apply to images.
        target_transform: Optional transform to apply to masks.
        synthetic_size: If set, generate this many synthetic samples
            instead of reading from disk.
        num_classes: Number of classes (default 6).
    """

    def __init__(
        self,
        image_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        image_size: int = 512,
        transform: callable | None = None,
        target_transform: callable | None = None,
        synthetic_size: int | None = None,
        num_classes: int = 6,
    ) -> None:
        super().__init__(
            dataset_type=DatasetType.MULTITASK,
            transform=transform,
            target_transform=target_transform,
        )
        self.image_dir = Path(image_dir) if image_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.image_size = image_size
        self.num_classes = num_classes
        self.synthetic_size = synthetic_size

        self._image_paths: list[Path] = []
        if synthetic_size is not None:
            pass
        elif self.image_dir is not None and self.image_dir.exists():
            self._image_paths = sorted(
                p
                for p in self.image_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            )

    def __len__(self) -> int:
        if self.synthetic_size is not None:
            return self.synthetic_size
        return len(self._image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.synthetic_size is not None:
            return self._synthetic_sample()

        image_path = self._image_paths[index]
        image = Image.open(image_path).convert("RGB")

        # Load corresponding mask if available
        mask: torch.Tensor | None = None
        if self.mask_dir is not None:
            mask_path = self.mask_dir / f"{image_path.stem}.png"
            if mask_path.exists():
                mask_pil = Image.open(mask_path)
                mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
            else:
                mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)) / 255.0

        if mask is not None and self.target_transform is not None:
            mask = self.target_transform(mask)

        if mask is None:
            mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)

        # Label derived from mask majority class, or index modulo num_classes
        label = int(torch.mode(mask.flatten()).values) if mask.numel() > 0 else 0

        return {
            "image": image,
            "label": label,
            "mask": mask,
        }

    def _synthetic_sample(self) -> dict[str, Any]:
        """Generate a synthetic multitask sample."""
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "label": torch.randint(0, self.num_classes, (1,)).item(),
            "mask": torch.randint(
                0, self.num_classes, (self.image_size, self.image_size), dtype=torch.long
            ),
        }


class UnlabeledWaferDataset(BaseDataset):
    """Unlabeled dataset for SemiWaferNet semi-supervised training.

    Extends ``common.datasets.BaseDataset``. Each sample is a dict with
    only ``"image"`` (``torch.Tensor [3, H, W]``) — no labels or masks,
    since targets are generated as pseudo-labels during training.

    For synthetic / test usage, pass ``synthetic_size`` to generate random
    tensors without real files.

    Args:
        image_dir: Directory containing input images.
        image_size: Target image size (assumed square, default 512).
        transform: Optional transform to apply to images.
        synthetic_size: If set, generate this many synthetic samples
            instead of reading from disk.
    """

    def __init__(
        self,
        image_dir: str | Path | None = None,
        image_size: int = 512,
        transform: callable | None = None,
        synthetic_size: int | None = None,
    ) -> None:
        super().__init__(
            dataset_type=DatasetType.CLASSIFICATION,
            transform=transform,
        )
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_size = image_size
        self.synthetic_size = synthetic_size

        self._image_paths: list[Path] = []
        if synthetic_size is not None:
            pass
        elif self.image_dir is not None and self.image_dir.exists():
            self._image_paths = sorted(
                p
                for p in self.image_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            )

    def __len__(self) -> int:
        if self.synthetic_size is not None:
            return self.synthetic_size
        return len(self._image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.synthetic_size is not None:
            return self._synthetic_sample()

        image_path = self._image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)) / 255.0

        return {"image": image}

    def _synthetic_sample(self) -> dict[str, Any]:
        """Generate a synthetic unlabeled sample (image only)."""
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
        }