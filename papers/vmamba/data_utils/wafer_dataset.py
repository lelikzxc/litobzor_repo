"""WM-811K wafer map dataset loader for FCS-VMamba.

Locked Subset-A baseline encoding (~73.9% Top-1):

  1. Optional geometric aug on the categorical map (NEAREST).
  2. One-hot die states ``{0,1,2}`` → 3 channels.
  3. Bilinear resize to ``image_size`` (paper 224).

Do not switch back to grayscale ``/255``: NEAREST upsampling creates long
constant runs that explode selective-scan grads (~1e18 on LayerNorm).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import (
    Compose,
    RandomAffine,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

from papers.vit_tiny.data_utils.base import BaseDataset, DatasetType

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

WM811K_LABEL_TO_IDX: dict[str, int] = {
    label: idx for idx, label in enumerate(WM811K_CLASSES)
}


def _is_unlabeled(failure: str) -> bool:
    """Empty / NaN failureType → unlabeled (not the defect class ``none``)."""
    failure = failure.strip()
    return failure == "" or failure.lower() == "nan"


def parse_wm811k_labeled_samples(
    labels_path: str | Path,
    split: str | None = None,
) -> list[tuple[str, int]]:
    """Parse LS-WMD / WM-811K ``labels.csv`` into ``(filename, class_idx)``."""
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    samples: list[tuple[str, int]] = []
    split_filter = split.lower() if split else None

    with open(labels_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        cols = [c.strip().lower() for c in header.split(",")]

        if "failuretype" not in cols:
            raise ValueError(
                f"Unrecognized labels.csv header in {labels_path}: {header!r}. "
                "Expected WM-811K columns including failureType."
            )

        ft_idx = cols.index("failuretype")
        fn_idx = cols.index("filename") if "filename" in cols else 0
        split_idx = cols.index("triantestlabel") if "triantestlabel" in cols else None

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) <= ft_idx:
                continue
            failure = parts[ft_idx].strip()
            if _is_unlabeled(failure):
                continue
            label_idx = WM811K_LABEL_TO_IDX.get(failure)
            if label_idx is None:
                try:
                    label_idx = int(failure)
                except ValueError:
                    continue
            if split_filter is not None and split_idx is not None:
                row_split = parts[split_idx].strip().lower()
                if row_split != split_filter:
                    continue
            samples.append((parts[fn_idx].strip(), label_idx))

    return samples

def die_map_to_onehot(image: Image.Image | np.ndarray) -> torch.Tensor:
    """Convert categorical die map ``{0,1,2}`` to one-hot ``[3, H, W]`` float."""
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.int64)
    else:
        arr = np.asarray(image, dtype=np.int64)
        if arr.ndim == 3:
            arr = arr[..., 0]
    arr = np.clip(arr, 0, 2)
    oh = np.zeros((3, arr.shape[0], arr.shape[1]), dtype=np.float32)
    for c in (0, 1, 2):
        oh[c] = (arr == c).astype(np.float32)
    return torch.from_numpy(oh)


def resize_onehot(onehot: torch.Tensor, image_size: int) -> torch.Tensor:
    """Bilinear resize one-hot map to square ``image_size`` → ``[3, S, S]``."""
    x = onehot.unsqueeze(0)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x.squeeze(0)


def build_train_augment(image_size: int = 224) -> Compose:
    """Geometric aug on categorical die maps (NEAREST) — locked baseline."""
    del image_size
    return Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomAffine(degrees=15, translate=(0.1, 0.1), interpolation=Image.NEAREST),
        ]
    )


class WaferWM811KDataset(BaseDataset):
    """WM-811K wafer map classification dataset (locked Subset-A baseline).

    Args:
        data_root: Root with ``labels.csv`` and ``images/``.
        image_size: Target square size (paper: 224).
        transform: Optional PIL geometric transform on the categorical map.
        split: Optional ``training`` / ``test`` filter on ``trianTestLabel``.
        labeled_only: Skip empty ``failureType`` (default True).
    """

    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 224,
        transform: Callable | None = None,
        split: str | None = None,
        labeled_only: bool = True,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION, transform=transform)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.image_dir = self.data_root / "images"
        self.labels_path = self.data_root / "labels.csv"
        self.split = split

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.image_dir}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        if not labeled_only:
            raise ValueError("labeled_only=False is not supported for classification")

        self._samples = parse_wm811k_labeled_samples(self.labels_path, split=split)
        if not self._samples:
            raise ValueError(f"No labeled samples loaded from {self.labels_path}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename, label_idx = self._samples[index]
        image_path = self.image_dir / filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        tensor = resize_onehot(die_map_to_onehot(image), self.image_size)

        return {
            "image": tensor,
            "label": label_idx,
        }

    @property
    def num_classes(self) -> int:
        return len(WM811K_CLASSES)

    @property
    def class_names(self) -> list[str]:
        return list(WM811K_CLASSES)


__all__ = [
    "WM811K_CLASSES",
    "WM811K_LABEL_TO_IDX",
    "WaferWM811KDataset",
    "parse_wm811k_labeled_samples",
    "build_train_augment",
    "die_map_to_onehot",
    "resize_onehot",
]
