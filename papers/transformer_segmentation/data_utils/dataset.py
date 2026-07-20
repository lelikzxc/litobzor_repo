"""Segmentation dataset adapter for SegFormer + Atrous built on common.datasets.

Provides a ``SegFormerDataset`` that extends ``common.datasets.BaseDataset``
and returns synthetic / real segmentation samples compatible with the
SegFormer segmentation pipeline.

Each sample is a dict with ``"image"`` (``torch.Tensor [3, H, W]`` RGB) and
``"mask"`` (``torch.Tensor [H, W]`` with integer class indices), which is
directly compatible with ``common.datasets.segmentation_collate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from common.datasets import BaseDataset, DatasetType


class SegFormerDataset(BaseDataset):
    """Segmentation dataset adapter for SegFormer + Atrous.

    Extends ``common.datasets.BaseDataset`` with a segmentation-specific
    interface. Each sample is a dict with ``"image"`` (``torch.Tensor``)
    and ``"mask"`` (``torch.Tensor``).

    SegFormer operates on **RGB** (3-channel) images. Synthetic samples
    are generated as ``[3, H, W]`` tensors for images and ``[H, W]``
    tensors for masks (integer class indices).

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
        num_classes: Number of segmentation classes (default 8).
    """

    def __init__(
        self,
        image_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        image_size: int = 512,
        transform: callable | None = None,
        target_transform: callable | None = None,
        synthetic_size: int | None = None,
        num_classes: int = 8,
    ) -> None:
        super().__init__(
            dataset_type=DatasetType.SEGMENTATION,
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
            # Synthetic mode: no files needed
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
                # Fallback: zero mask
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

        return {
            "image": image,
            "mask": mask,
        }

    def _synthetic_sample(self) -> dict[str, Any]:
        """Generate a synthetic segmentation sample.

        Returns:
            dict with ``"image"`` (``[3, H, W]``) and ``"mask"``
            (``[H, W]`` with integer class indices).
        """
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "mask": torch.randint(
                0, self.num_classes, (self.image_size, self.image_size), dtype=torch.long
            ),
        }