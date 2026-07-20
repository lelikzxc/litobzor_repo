"""Classification dataset adapter for ViT-Tiny built on common.datasets.

Provides a ``ViTTinyDataset`` that extends ``common.datasets.BaseDataset``
and returns synthetic / real classification samples compatible with the
ViT-Tiny classification pipeline.

Each sample is a dict with ``"image"`` (``torch.Tensor [1, H, W]`` grayscale)
and ``"label"`` (``int``), which is directly compatible with
``common.datasets.classification_collate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from common.datasets import BaseDataset, DatasetType


class ViTTinyDataset(BaseDataset):
    """Classification dataset adapter for ViT-Tiny.

    Extends ``common.datasets.BaseDataset`` with a classification-specific
    interface. Each sample is a dict with ``"image"`` (``torch.Tensor``)
    and ``"label"`` (``int``).

    ViT-Tiny operates on **grayscale** (1-channel) images of size 32×32
    by default. Synthetic samples are generated as ``[1, H, W]`` tensors.
    Real images loaded from disk are converted to grayscale via
    ``Image.open(...).convert("L")`` and expanded to ``[1, H, W]``.

    For synthetic / test usage, pass ``synthetic_size`` to generate random
    tensors without real files.

    Args:
        image_dir: Directory containing input images.
        image_size: Target image size (assumed square, default 32).
        transform: Optional transform to apply to images.
        synthetic_size: If set, generate this many synthetic samples
            instead of reading from disk.
        num_classes: Number of classification classes (default 8).
    """

    def __init__(
        self,
        image_dir: str | Path | None = None,
        image_size: int = 32,
        transform: callable | None = None,
        synthetic_size: int | None = None,
        num_classes: int = 8,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION, transform=transform)
        self.image_dir = Path(image_dir) if image_dir else None
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
        # Convert to grayscale (L mode) — ViT-Tiny uses 1-channel input
        image = Image.open(image_path).convert("L")

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert PIL grayscale to [1, H, W] tensor
            image = torch.from_numpy(np.array(image, dtype=np.float32)) / 255.0
            image = image.unsqueeze(0)  # [H, W] → [1, H, W]

        # Use index modulo num_classes as a placeholder label
        label = index % self.num_classes

        return {
            "image": image,
            "label": label,
        }

    def _synthetic_sample(self) -> dict[str, Any]:
        """Generate a synthetic classification sample.

        Returns a grayscale image tensor of shape ``[1, H, W]``.
        """
        return {
            "image": torch.randn(1, self.image_size, self.image_size),
            "label": torch.randint(0, self.num_classes, (1,)).item(),
        }