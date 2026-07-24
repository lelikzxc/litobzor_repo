"""Magnetic tile defect detection dataset for CTM-YOLOv10.

Loads images and YOLO-format labels from the magnetic_tile dataset
(pre-split into train/valid/test by Roboflow).

Each sample returns:
    ``{"image": [3, H, W] tensor, "label": [N, 5] tensor (YOLO format)}``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from papers.ctm_yolov10.data_utils.base import BaseDataset, DatasetType


class MagneticTileDataset(BaseDataset):
    """Magnetic tile defect detection dataset.

    Loads images and YOLO-format labels from a Roboflow-exported dataset
    with pre-split train/valid/test directories.

    Args:
        data_root: Root directory of the magnetic_tile dataset.
        split: One of ``"train"``, ``"valid"``, ``"test"``.
        image_size: Target image size (assumed square, default 640).
        transform: Optional transform to apply to images.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        image_size: int = 640,
        transform: callable | None = None,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION, transform=transform)
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size

        self.image_dir = self.data_root / split / "images"
        self.label_dir = self.data_root / split / "labels"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self._image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        )

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self._image_paths[index]
        image = Image.open(image_path).convert("RGB")

        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(
                __import__("numpy").array(image, dtype="float32").transpose(2, 0, 1)
            ) / 255.0

        # Load YOLO-format label
        label_path = self.label_dir / f"{image_path.stem}.txt"
        label = self._load_yolo_label(label_path) if label_path.exists() else torch.zeros(0, 5)

        return {
            "image": image,       # [3, H, W]
            "label": label,       # [N, 5] YOLO format
            "img_path": str(image_path),
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