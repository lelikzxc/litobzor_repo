"""Classification dataset adapter for FCS-VMamba built on common.datasets.

Provides a ``VMambaDataset`` that extends ``common.datasets.BaseDataset``
and returns synthetic / real classification samples compatible with the
FCS-VMamba classification pipeline.

Each sample is a dict with ``"image"`` (``torch.Tensor [3, H, W]``) and
``"label"`` (``int``), which is directly compatible with
``common.datasets.classification_collate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from common.datasets import BaseDataset, DatasetType


class VMambaDataset(BaseDataset):
    """Classification dataset adapter for FCS-VMamba.

    Extends ``common.datasets.BaseDataset`` with a classification-specific
    interface. Each sample is a dict with ``"image"`` (``torch.Tensor``)
    and ``"label"`` (``int``).

    For synthetic / test usage, pass ``synthetic_size`` to generate random
    tensors without real files.

    Args:
        image_dir: Directory containing input images.
        image_size: Target image size (assumed square, default 224).
        transform: Optional transform to apply to images.
        synthetic_size: If set, generate this many synthetic samples
            instead of reading from disk.
        num_classes: Number of classification classes (default 8).
    """

    def __init__(
        self,
        image_dir: str | Path | None = None,
        image_size: int = 224,
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
        image = Image.open(image_path).convert("RGB")

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Use index modulo num_classes as a placeholder label
        label = index % self.num_classes

        return {
            "image": image,
            "label": label,
        }

    def _synthetic_sample(self) -> dict[str, Any]:
        """Generate a synthetic classification sample."""
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "label": torch.randint(0, self.num_classes, (1,)).item(),
        }