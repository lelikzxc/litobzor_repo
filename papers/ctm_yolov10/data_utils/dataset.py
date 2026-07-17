"""Minimal detection dataset interface for CTM-YOLOv10."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class DetectionDataset(Dataset[Any]):
    """Minimal detection dataset.

    Reads images and labels from the given directories.
    This is a base interface — augmentation and preprocessing
    will be added in a future stage.

    Args:
        image_dir: Directory containing input images.
        label_dir: Directory containing label files.
    """

    def __init__(self, image_dir: str | Path, label_dir: str | Path) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        # Collect image paths (common formats)
        self.image_paths = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Return image and corresponding label.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image, label).
        """
        image_path = self.image_paths[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        # Placeholder: actual loading will be implemented with augmentation
        image = None  # will be replaced with cv2.imread / PIL
        label = None  # will be replaced with parsed YOLO-format labels

        return image, label