"""Detection dataset adapter for CTM-YOLOv10 built on common.datasets.

Provides a ``DetectionDataset`` that extends ``common.datasets.BaseDataset``
and returns synthetic / real detection samples compatible with the YOLOv10
detection pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from common.datasets import BaseDataset, DatasetType


class DetectionDataset(BaseDataset):
    """Detection dataset adapter for CTM-YOLOv10.

    Extends ``common.datasets.BaseDataset`` with a detection-specific
    interface. Each sample is a dict with ``"image"`` (``torch.Tensor``)
    and ``"label"`` (``torch.Tensor`` of shape ``[N, 5]`` for YOLO-format
    boxes, or a placeholder for synthetic data).

    For synthetic / test usage, pass ``synthetic_size`` to generate random
    tensors without real files.

    Args:
        image_dir: Directory containing input images.
        label_dir: Directory containing YOLO-format label files.
        image_size: Target image size (assumed square, default 640).
        transform: Optional transform to apply to images.
        synthetic_size: If set, generate this many synthetic samples
            instead of reading from disk.
        num_classes: Number of detection classes (default 80).
    """

    def __init__(
        self,
        image_dir: str | Path | None = None,
        label_dir: str | Path | None = None,
        image_size: int = 640,
        transform: callable | None = None,
        synthetic_size: int | None = None,
        num_classes: int = 80,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION, transform=transform)
        self.image_dir = Path(image_dir) if image_dir else None
        self.label_dir = Path(label_dir) if label_dir else None
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

        # Load YOLO-format label if available
        label = None
        if self.label_dir is not None:
            label_path = self.label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                label = self._load_yolo_label(label_path)

        return {
            "image": image,
            "label": label if label is not None else torch.zeros(0, 5),
        }

    def _synthetic_sample(self) -> dict[str, Any]:
        """Generate a synthetic detection sample."""
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "label": torch.zeros(0, 5),  # no boxes in synthetic data
        }

    @staticmethod
    def _load_yolo_label(path: Path) -> torch.Tensor:
        """Load a YOLO-format label file.

        YOLO format: one row per object with ``class_id cx cy w h``
        (all normalized to [0, 1]).

        Returns:
            Tensor of shape ``[N, 5]``.
        """
        boxes: list[list[float]] = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append([float(x) for x in parts[:5]])
        return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 5)