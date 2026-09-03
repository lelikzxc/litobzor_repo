"""WM-811K wafer map dataset loader for SemiWaferNet with augmentations and hybrid sampling.

Loads wafer map images from the WM-811K dataset (labels.csv + PNG images)
and returns multitask samples compatible with SemiWaferNet:
    ``{"image": [1, H, W], "label": int, "mask": [H, W]}``

Implements:
    - Data augmentation pipeline (Section 4.1: "dynamic augmentation")
    - Hybrid sampling: downsampling None class to 30% (Section 4.1)

SemiWaferNet operates on **grayscale** (1-channel) images per the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import (
    ColorJitter,
    Compose,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
)

from papers.semiwafernet.data_utils.base import BaseDataset, DatasetType

# WM-811K class names (9 classes)
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

# Label-to-index mapping
WM811K_LABEL_TO_IDX: dict[str, int] = {
    label: idx for idx, label in enumerate(WM811K_CLASSES)
}

# Index of "none" class (class 0)
NONE_CLASS_IDX: int = 0


def default_train_transform(image_size: int) -> Compose:
    """Build training augmentation pipeline per SemiWaferNet paper Section 4.1.

    Applies:
        - Random horizontal flip (p=0.5)
        - Random rotation (±10°)
        - Color jitter (brightness=0.2, contrast=0.2)
        - Resize to target size

    Returns PIL images (not tensors) — tensor conversion happens in __getitem__.

    Args:
        image_size: Target spatial size (H == W).

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),
        ColorJitter(brightness=0.2, contrast=0.2),
        Resize(image_size, interpolation=Image.BILINEAR),
    ])


def apply_hybrid_sampling(
    samples: list[tuple[str, int]],
    none_downsample_ratio: float = 0.30,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """Apply hybrid sampling: downsample the majority 'none' class.

    From SemiWaferNet paper Section 4.1:
        "To address class imbalance, we employ hybrid sampling that
         downsamples the majority None class to 30% of its original size."

    Args:
        samples: List of (filename, label_idx) pairs.
        none_downsample_ratio: Fraction of None class to keep (default 0.30).
        seed: Random seed for reproducibility.

    Returns:
        Downsampled list of (filename, label_idx) pairs.
    """
    rng = np.random.RandomState(seed)
    none_samples = [(f, l) for f, l in samples if l == NONE_CLASS_IDX]
    other_samples = [(f, l) for f, l in samples if l != NONE_CLASS_IDX]

    # Downsample None class
    keep_none = int(len(none_samples) * none_downsample_ratio)
    rng.shuffle(none_samples)
    none_downsampled = none_samples[:keep_none]

    print(f"  Hybrid sampling: None {len(none_samples)} -> {keep_none}, "
          f"others {len(other_samples)}, total {len(other_samples) + keep_none}")

    return other_samples + none_downsampled


def _is_unlabeled_failure_type(value: str) -> bool:
    """WM-811K unlabeled rows have empty failureType (parsed as NaN in pandas).

    Note: the string ``\"none\"`` is a valid labeled class (no defect pattern),
    not an unlabeled marker.
    """
    v = value.strip()
    return (not v) or v.lower() == "nan"


def parse_wm811k_labeled_rows(
    labels_path: Path,
    split: str | None = None,
) -> list[tuple[str, int]]:
    """Parse labeled ``(filename, class_idx)`` rows from WM-811K ``labels.csv``.

    Supports:
      - Official CSV: ``filename,...,trianTestLabel,failureType``
      - Simple CSV: ``image,label`` / ``filename,label``

    Args:
        labels_path: Path to ``labels.csv``.
        split: If ``\"training\"`` or ``\"test\"``, keep only that official
            ``trianTestLabel`` partition. ``None`` keeps all labeled rows.

    Returns:
        List of ``(filename, label_idx)`` for rows with a valid failureType.
    """
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    samples: list[tuple[str, int]] = []
    split_filter = split.lower() if split else None

    with open(labels_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        header_l = header.lower()
        cols = [c.strip().lower() for c in header.split(",")]

        # Official WM-811K layout
        if "failuretype" in cols:
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
                if _is_unlabeled_failure_type(failure):
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

        # Simple image,label layout
        elif ("image" in header_l or "filename" in header_l) and "label" in header_l:
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
                    label_idx = WM811K_LABEL_TO_IDX.get(label_str, 0)
                samples.append((filename, label_idx))
        else:
            raise ValueError(
                f"Unrecognized labels.csv header in {labels_path}: {header!r}. "
                "Expected WM-811K columns (failureType) or image,label."
            )

    return samples


def parse_wm811k_unlabeled_filenames(
    labels_path: str | Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[str]:
    """Return filenames with empty/NaN ``failureType`` (unlabeled pool).

    Paper Section 4.1 uses 150,000 unlabeled samples subsampled from the
    full unlabeled pool (~638k).
    """
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    filenames: list[str] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        cols = [c.strip().lower() for c in header.split(",")]
        if "failuretype" not in cols:
            raise ValueError(
                f"Unlabeled WM-811K rows require a failureType column; got {header!r}"
            )
        ft_idx = cols.index("failuretype")
        fn_idx = cols.index("filename") if "filename" in cols else 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) <= ft_idx:
                continue
            if _is_unlabeled_failure_type(parts[ft_idx]):
                filenames.append(parts[fn_idx].strip())

    if max_samples is not None and max_samples < len(filenames):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(filenames), size=max_samples, replace=False)
        filenames = [filenames[i] for i in sorted(idx.tolist())]

    return filenames


class WaferWM811KDataset(BaseDataset):
    """WM-811K wafer map classification dataset for SemiWaferNet.

    Reads ``labels.csv`` from the dataset root and loads PNG images.
    Only **labeled** rows (non-empty ``failureType``) are included.
    Returns samples with image, label, and a dummy mask
    (WM-811K has classification labels, not segmentation masks).

    Args:
        data_root: Root directory containing ``labels.csv`` and ``images/``.
        image_size: Target image size (assumed square, default 32 for classification).
        num_classes: Number of classes (default 9 for WM-811K).
        transform: Optional transform to apply to images (PIL → PIL).
            If ``None`` and ``train=True``, uses default augmentations.
            If ``None`` and ``train=False``, uses resize only.
        train: If ``True``, applies training augmentations by default.
        hybrid_sampling: If ``True``, downsamples None class (Section 4.1).
        none_downsample_ratio: Fraction of None class to keep (default 0.30).
        split: Official WM-811K partition filter: ``\"training\"``, ``\"test\"``,
            or ``None`` for all labeled samples (paper Section 4.1).
    """

    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 32,
        num_classes: int = 9,
        transform: callable | None = None,
        train: bool = True,
        hybrid_sampling: bool = True,
        none_downsample_ratio: float = 0.30,
        split: str | None = None,
    ) -> None:
        super().__init__(dataset_type=DatasetType.MULTITASK)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_names = WM811K_CLASSES
        self.split = split

        # Default transforms
        if transform is None:
            if train:
                self.transform = default_train_transform(image_size)
            else:
                self.transform = Compose([
                    Resize(image_size, interpolation=Image.BILINEAR),
                ])
        else:
            self.transform = transform

        self.labels_path = self.data_root / "labels.csv"
        self.images_dir = self.data_root / "images"

        self._samples: list[tuple[str, int]] = parse_wm811k_labeled_rows(
            self.labels_path, split=split
        )
        if not self._samples:
            raise ValueError(
                f"No labeled samples loaded from {self.labels_path} "
                f"(split={split!r})"
            )

        # Apply hybrid sampling (downsample None class) for training
        if train and hybrid_sampling:
            self._samples = apply_hybrid_sampling(
                self._samples,
                none_downsample_ratio=none_downsample_ratio,
                seed=42,
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename, label = self._samples[index]
        image_path = self.images_dir / filename

        # Load image as grayscale (1 channel) — WM-811K is grayscale
        if image_path.exists():
            image = Image.open(image_path).convert("L")
        else:
            png_path = self.images_dir / f"{Path(filename).stem}.png"
            if png_path.exists():
                image = Image.open(png_path).convert("L")
            else:
                raise FileNotFoundError(f"Image not found: {image_path} or {png_path}")

        # Apply transform (includes resize + augmentations)
        if self.transform is not None:
            # Convert to RGB for torchvision transforms (they expect 3-channel)
            image_rgb = image.convert("RGB")
            image_rgb = self.transform(image_rgb)
            # Convert back to grayscale tensor [1, H, W]
            image = torch.from_numpy(np.array(image_rgb, dtype=np.float32)).mean(dim=2, keepdim=True).permute(2, 0, 1) / 255.0
        else:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            image = torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0

        # Dummy mask (zeros) — WM-811K has no segmentation masks
        mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)

        return {
            "image": image,      # [1, H, W]
            "label": label,      # int
            "mask": mask,        # [H, W] dummy
        }

    @property
    def num_samples(self) -> int:
        return len(self._samples)


class SMOTEDataset(BaseDataset):
    """WM-811K dataset with SMOTE oversampling of minority classes.

    Applies SMOTE (Synthetic Minority Over-sampling Technique) to the
    minority classes to construct a balanced training set, per the paper
    Section 4.1: "a balanced labeled training set is constructed via hybrid
    sampling by downsampling the majority None class and applying SMOTE to
    minority classes".

    SMOTE operates on flattened image vectors (image_size^2 features) and
    generates synthetic samples by interpolating between nearest neighbours
    of the same class. The synthetic images are stored in memory.

    Args:
        data_root: Root directory containing ``labels.csv`` and ``images/``.
        samples: List of ``(filename, label_idx)`` pairs (already downsampled).
        image_size: Target image size (assumed square).
        num_classes: Number of classes.
        smote_k_neighbors: Number of nearest neighbours for SMOTE (default 5).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        data_root: str | Path,
        samples: list[tuple[str, int]],
        image_size: int = 32,
        num_classes: int = 9,
        smote_k_neighbors: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_names = WM811K_CLASSES
        self.images_dir = self.data_root / "images"

        # Dynamic augmentation (Section 4.1)
        self.transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=10),
            ColorJitter(brightness=0.2, contrast=0.2),
        ])

        # Load all images as flattened vectors
        images: list[np.ndarray] = []
        labels: list[int] = []
        for filename, label in samples:
            img = self._load_image(filename)
            images.append(img)
            labels.append(label)

        X = np.stack(images)  # [N, image_size^2]
        y = np.array(labels)  # [N]

        # Apply SMOTE to minority classes (only if there are >= 2 classes
        # and every class has enough samples for k-neighbours)
        from imblearn.over_sampling import SMOTE

        unique_classes = np.unique(y)
        min_count = min(np.bincount(y)[unique_classes])
        if len(unique_classes) < 2 or min_count < smote_k_neighbors + 1:
            # Not enough classes/samples for SMOTE — keep original data
            X_res, y_res = X, y
            print(f"  SMOTE skipped (classes={len(unique_classes)}, min_count={min_count})")
        else:
            smote = SMOTE(
                k_neighbors=smote_k_neighbors,
                random_state=seed,
            )
            X_res, y_res = smote.fit_resample(X, y)

        # Store resampled data
        self._X = X_res.reshape(-1, 1, image_size, image_size).astype(np.float32)
        self._y = y_res.astype(np.int64)

        print(f"  SMOTE: {len(X)} -> {len(X_res)} samples")
        counts = np.bincount(self._y, minlength=num_classes)
        print(f"  SMOTE class counts: {counts.tolist()}")

    def _load_image(self, filename: str) -> np.ndarray:
        """Load a single image as a flattened float vector in [0, 1]."""
        image_path = self.images_dir / filename
        if not image_path.exists():
            png_path = self.images_dir / f"{Path(filename).stem}.png"
            if png_path.exists():
                image_path = png_path
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(image, dtype=np.float32) / 255.0
        return arr.flatten()

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # image: [1, H, W] float in [0, 1]
        image = self._X[index]

        # Apply dynamic augmentation (Section 4.1)
        if self.transform is not None:
            # Convert to PIL for torchvision transforms
            img_pil = Image.fromarray((image[0] * 255.0).astype(np.uint8), mode="L")
            img_rgb = img_pil.convert("RGB")
            img_rgb = self.transform(img_rgb)
            # Convert back to grayscale tensor [1, H, W]
            image = torch.from_numpy(np.array(img_rgb, dtype=np.float32)).mean(dim=2, keepdim=True).permute(2, 0, 1) / 255.0
        else:
            image = torch.from_numpy(image)  # [1, H, W]

        label = int(self._y[index])
        mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
        return {
            "image": image,
            "label": label,
            "mask": mask,
        }

    @property
    def num_samples(self) -> int:
        return len(self._y)


class UnlabeledWM811KDataset(BaseDataset):
    """Unlabeled WM-811K maps (empty ``failureType``) for SSL Stages 2–3.

    Paper Section 4.1: subsample 150,000 unlabeled wafers and add them to the
    balanced labeled training set for progressive pseudo-labeling.

    Each sample is ``{\"image\": [1, H, W]}`` (no label / mask).
    """

    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 32,
        max_samples: int = 150_000,
        transform: callable | None = None,
        train: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.images_dir = self.data_root / "images"
        self.labels_path = self.data_root / "labels.csv"

        if transform is None:
            if train:
                self.transform = default_train_transform(image_size)
            else:
                self.transform = Compose([
                    Resize(image_size, interpolation=Image.BILINEAR),
                ])
        else:
            self.transform = transform

        self._filenames = parse_wm811k_unlabeled_filenames(
            self.labels_path,
            max_samples=max_samples,
            seed=seed,
        )
        if not self._filenames:
            raise ValueError(f"No unlabeled samples found in {self.labels_path}")

    def __len__(self) -> int:
        return len(self._filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename = self._filenames[index]
        image_path = self.images_dir / filename
        if not image_path.exists():
            png_path = self.images_dir / f"{Path(filename).stem}.png"
            if png_path.exists():
                image_path = png_path
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("L")
        if self.transform is not None:
            image_rgb = image.convert("RGB")
            image_rgb = self.transform(image_rgb)
            image = (
                torch.from_numpy(np.array(image_rgb, dtype=np.float32))
                .mean(dim=2, keepdim=True)
                .permute(2, 0, 1)
                / 255.0
            )
        else:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            image = torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0

        return {"image": image}

    @property
    def num_samples(self) -> int:
        return len(self._filenames)