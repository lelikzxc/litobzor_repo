"""WM-811K wafer map dataset loader for SemiWaferNet.

WM-811K PNGs store categorical die states ``{0, 1, 2}`` (background / normal /
defect), NOT natural-image intensities. The paper specifies
``X ∈ R^{3×H×W}`` but not its channel encoding. One-hot channels after
nearest-neighbor resize are an explicit reproduction choice.

Augmentations are geometric only (flip / 90° rotations) so categories stay
intact — no ColorJitter / bilinear.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from papers.semiwafernet.data_utils.base import BaseDataset, DatasetType
from papers.semiwafernet.training.progress import progress_bar

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

WM811K_LABEL_TO_IDX: dict[str, int] = {label: idx for idx, label in enumerate(WM811K_CLASSES)}

NONE_CLASS_IDX: int = 0


def load_wafer_die_map(image_path: Path) -> np.ndarray:
    """Load a wafer PNG as an integer die-state map ``{0,1,2}``."""
    from papers.wafer_encoding import decode_die_map

    with Image.open(image_path) as image:
        return decode_die_map(image)


def resize_die_map(die_map: np.ndarray, image_size: int) -> np.ndarray:
    """Nearest-neighbor resize — preserves categorical die states."""
    img = Image.fromarray(die_map.astype(np.uint8), mode="L")
    img = img.resize((image_size, image_size), Image.NEAREST)
    return np.array(img, dtype=np.int64)


def die_map_to_onehot(die_map: np.ndarray) -> torch.Tensor:
    """Convert ``[H,W]`` die states to ``[3,H,W]`` float one-hot (paper R^{3×H×W})."""
    oh = np.eye(3, dtype=np.float32)[die_map.astype(np.int64)]
    return torch.from_numpy(oh).permute(2, 0, 1).contiguous()


def geometric_augment(x: torch.Tensor, rng: np.random.RandomState | None = None) -> torch.Tensor:
    """Flip / 90°-rotate a ``[C,H,W]`` tensor (category-preserving)."""
    if rng is None:
        rng = np.random
    if rng.rand() < 0.5:
        x = torch.flip(x, dims=[-1])
    if rng.rand() < 0.5:
        x = torch.flip(x, dims=[-2])
    k = int(rng.randint(0, 4))
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])
    return x


def encode_wafer_image(image_path: Path, image_size: int) -> torch.Tensor:
    """Full encode path: load → nearest resize → one-hot ``[3,H,W]``."""
    die = load_wafer_die_map(image_path)
    die = resize_die_map(die, image_size)
    return die_map_to_onehot(die)


def apply_hybrid_sampling(
    samples: list[tuple[str, int]],
    none_downsample_ratio: float = 0.30,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """Downsample the majority 'none' class (paper Section 4.1)."""
    rng = np.random.RandomState(seed)
    none_samples = [(f, l) for f, l in samples if l == NONE_CLASS_IDX]
    other_samples = [(f, l) for f, l in samples if l != NONE_CLASS_IDX]

    keep_none = int(len(none_samples) * none_downsample_ratio)
    rng.shuffle(none_samples)
    none_downsampled = none_samples[:keep_none]

    print(
        f"  Hybrid sampling: None {len(none_samples)} -> {keep_none}, "
        f"others {len(other_samples)}, total {len(other_samples) + keep_none}"
    )
    return other_samples + none_downsampled


def inspect_wm811k_labels(labels_path: str | Path) -> dict[str, bool | str | int]:
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # utf-8-sig also accepts ordinary UTF-8 and removes the BOM added by
    # Windows spreadsheet exports. csv.reader handles quoted values and CRLF.
    with open(labels_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        cols = _labels_columns(reader, labels_path)
        has_failure_type = "failuretype" in cols
        has_official_split = "triantestlabel" in cols
        fmt = "official" if has_failure_type else "simple"
        unlabeled_count = 0
        if has_failure_type:
            ft_idx = cols.index("failuretype")
            for parts in reader:
                if parts and len(parts) > ft_idx and _is_unlabeled_failure_type(parts[ft_idx]):
                    unlabeled_count += 1

    return {
        "format": fmt,
        "has_failure_type": has_failure_type,
        "has_official_split": has_official_split,
        "unlabeled_count": unlabeled_count,
    }


def _is_unlabeled_failure_type(value: str) -> bool:
    v = value.strip()
    return (not v) or v.lower() == "nan"


def _labels_columns(reader, labels_path: Path) -> list[str]:
    header = next(reader, None)
    if not header:
        raise ValueError(f"Empty labels file: {labels_path}")
    return [column.strip().lower() for column in header]


def _parse_label(value: str, labels_path: Path, row_number: int) -> int:
    value = value.strip()
    label = WM811K_LABEL_TO_IDX.get(value)
    if label is None:
        try:
            label = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Unknown WM-811K label {value!r} in {labels_path}, row {row_number}"
            ) from exc
    if not 0 <= label < len(WM811K_CLASSES):
        raise ValueError(
            f"WM-811K label {label} is outside 0..8 in {labels_path}, row {row_number}"
        )
    return label


def parse_wm811k_labeled_rows(
    labels_path: Path,
    split: str | None = None,
) -> list[tuple[str, int]]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    samples: list[tuple[str, int]] = []
    split_filter = split.lower() if split else None

    with open(labels_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        cols = _labels_columns(reader, labels_path)

        if "failuretype" in cols:
            ft_idx = cols.index("failuretype")
            fn_idx = cols.index("filename") if "filename" in cols else 0
            split_idx = cols.index("triantestlabel") if "triantestlabel" in cols else None

            if split_filter is not None and split_idx is None:
                raise ValueError(
                    f"Official split {split_filter!r} requested but {labels_path} "
                    "has no trianTestLabel column. Run scripts/unpack_lswmd.py on "
                    "LSWMD.pkl or set data.use_official_split: false."
                )

            for row_number, parts in enumerate(reader, start=2):
                if not parts:
                    continue
                required_idx = max(ft_idx, fn_idx, split_idx if split_idx is not None else 0)
                if len(parts) <= required_idx:
                    raise ValueError(f"Incomplete WM-811K row in {labels_path}, row {row_number}")
                failure = parts[ft_idx].strip()
                if _is_unlabeled_failure_type(failure):
                    continue
                if split_filter is not None and split_idx is not None:
                    row_split = parts[split_idx].strip().lower()
                    if row_split != split_filter:
                        continue
                label_idx = _parse_label(failure, labels_path, row_number)
                filename = parts[fn_idx].strip()
                if not filename:
                    raise ValueError(f"Empty image filename in {labels_path}, row {row_number}")
                samples.append((filename, label_idx))

        elif ("image" in cols or "filename" in cols) and "label" in cols:
            if split_filter is not None:
                raise ValueError(
                    f"Official split {split_filter!r} requested but {labels_path} "
                    "has no trianTestLabel column"
                )
            fn_idx = cols.index("filename") if "filename" in cols else cols.index("image")
            label_idx = cols.index("label")
            for row_number, parts in enumerate(reader, start=2):
                if not parts:
                    continue
                if len(parts) <= max(fn_idx, label_idx):
                    raise ValueError(f"Incomplete WM-811K row in {labels_path}, row {row_number}")
                filename = parts[fn_idx].strip()
                if not filename:
                    raise ValueError(f"Empty image filename in {labels_path}, row {row_number}")
                samples.append((filename, _parse_label(parts[label_idx], labels_path, row_number)))
        else:
            raise ValueError(
                f"Unrecognized labels.csv header in {labels_path}: {cols!r}. "
                "Expected WM-811K columns (failureType) or image,label."
            )

    return samples


def parse_wm811k_unlabeled_filenames(
    labels_path: str | Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[str]:
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be nonnegative or None")

    filenames: list[str] = []
    with open(labels_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        cols = _labels_columns(reader, labels_path)
        if "failuretype" not in cols:
            return []
        ft_idx = cols.index("failuretype")
        fn_idx = cols.index("filename") if "filename" in cols else 0
        with progress_bar(reader, desc="  Parsing unlabeled WM-811K rows", unit="row") as rows:
            for row_number, parts in enumerate(rows, start=2):
                if not parts:
                    continue
                if len(parts) <= max(ft_idx, fn_idx):
                    raise ValueError(f"Incomplete WM-811K row in {labels_path}, row {row_number}")
                if _is_unlabeled_failure_type(parts[ft_idx]):
                    filename = parts[fn_idx].strip()
                    if not filename:
                        raise ValueError(f"Empty image filename in {labels_path}, row {row_number}")
                    filenames.append(filename)

    if max_samples is not None and max_samples < len(filenames):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(filenames), size=max_samples, replace=False)
        filenames = [filenames[i] for i in sorted(idx.tolist())]

    return filenames


def _resolve_image_path(images_dir: Path, filename: str) -> Path:
    image_path = images_dir / filename
    if image_path.exists():
        return image_path
    png_path = images_dir / f"{Path(filename).stem}.png"
    if png_path.exists():
        return png_path
    raise FileNotFoundError(f"Image not found: {image_path} or {png_path}")


def validate_image_inventory(
    images_dir: str | Path,
    samples: Iterable[str | tuple[str, int]],
) -> int:
    """Check selected files once before training, without decoding PNGs.

    A single directory scan avoids hundreds of thousands of individual
    filesystem calls for the usual flat WM-811K export. Nested/absolute paths
    keep the same resolution rules as the lazy dataset. Return the number of
    distinct manifest names checked; an incomplete copy fails with examples.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"WM-811K images directory not found: {images_dir}")
    with os.scandir(images_dir) as entries:
        flat_names = {os.path.normcase(entry.name) for entry in entries if entry.is_file()}

    seen: set[str] = set()
    missing_count = 0
    missing_examples: list[str] = []
    for sample in samples:
        filename = sample if isinstance(sample, str) else sample[0]
        if filename in seen:
            continue
        seen.add(filename)
        path = Path(filename)
        if len(path.parts) == 1 and not path.is_absolute():
            exists = (
                os.path.normcase(filename) in flat_names
                or os.path.normcase(f"{path.stem}.png") in flat_names
            )
        else:
            try:
                exists = _resolve_image_path(images_dir, filename).is_file()
            except FileNotFoundError:
                exists = False
        if not exists:
            missing_count += 1
            if len(missing_examples) < 3:
                missing_examples.append(filename)
    if missing_count:
        raise FileNotFoundError(
            f"Missing {missing_count} of {len(seen)} selected WM-811K images in {images_dir}; "
            f"examples: {missing_examples!r}. Copy the complete images directory before training."
        )
    return len(seen)


class WaferWM811KDataset(BaseDataset):
    """Labeled WM-811K classification dataset (3-channel one-hot die maps)."""

    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 32,
        num_classes: int = 9,
        transform: callable | None = None,  # unused; kept for API compat
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
        self.train = train
        self._aug_rng = np.random.RandomState(42)
        _ = transform

        self.labels_path = self.data_root / "labels.csv"
        self.images_dir = self.data_root / "images"

        self._samples: list[tuple[str, int]] = parse_wm811k_labeled_rows(
            self.labels_path, split=split
        )
        if not self._samples:
            raise ValueError(f"No labeled samples loaded from {self.labels_path} (split={split!r})")

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
        image = encode_wafer_image(_resolve_image_path(self.images_dir, filename), self.image_size)
        if self.train:
            image = geometric_augment(image, self._aug_rng)
        mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
        return {"image": image, "label": label, "mask": mask}

    @property
    def num_samples(self) -> int:
        return len(self._samples)


class SMOTEDataset(BaseDataset):
    """SMOTE over flattened one-hot inputs; random-copy ablation is explicit."""

    def __init__(
        self,
        data_root: str | Path,
        samples: list[tuple[str, int]],
        image_size: int = 32,
        num_classes: int = 9,
        smote_k_neighbors: int = 5,
        seed: int = 42,
        method: str = "smote",
        augment: bool = True,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_names = WM811K_CLASSES
        self.images_dir = self.data_root / "images"
        self._aug_rng = np.random.RandomState(seed + 7)
        self.augment = augment
        method = str(method).lower().strip()
        if method not in {"random", "smote"}:
            raise ValueError(f"Unknown oversample method: {method}")
        if not samples:
            raise ValueError("Cannot oversample an empty training set")
        if image_size <= 0 or num_classes <= 0:
            raise ValueError("image_size and num_classes must be positive")
        y = np.asarray([label for _, label in samples], dtype=np.int64)
        if np.any(y < 0) or np.any(y >= num_classes):
            raise ValueError(f"Training labels must be within 0..{num_classes - 1}")
        if method == "smote" and smote_k_neighbors < 1:
            raise ValueError("smote_k_neighbors must be positive")
        unique_classes = np.unique(y)
        min_count = int(min(np.bincount(y)[unique_classes]))
        if method == "smote" and (len(unique_classes) < 2 or min_count < 2):
            raise ValueError("SMOTE needs at least two samples in each of two classes")

        # Store categorical maps in one compact host array instead of keeping
        # float32 copies in a list and another stacked array during SMOTE.
        # The encoded SMOTE features below remain unchanged and float32.
        die = np.empty((len(samples), image_size, image_size), dtype=np.uint8)
        with progress_bar(
            samples, desc="  Loading images for oversample", unit="img"
        ) as rows:
            for index, (filename, _) in enumerate(rows):
                raw_die = load_wafer_die_map(_resolve_image_path(self.images_dir, filename))
                die[index] = resize_die_map(raw_die, image_size)
        if method == "random":
            from imblearn.over_sampling import RandomOverSampler

            # Store each real map once; oversampling duplicates indices only.
            ros = RandomOverSampler(random_state=seed)
            indices, y_res = ros.fit_resample(np.arange(len(y)).reshape(-1, 1), y)
            self._indices = indices[:, 0]
            self._die = die
            self._X = None
        else:
            from imblearn.over_sampling import SMOTE

            # SMOTE in encoded input space. Keep interpolated features: rounding
            # removes the interpolation and is not the algorithm in the paper.
            X = np.eye(3, dtype=np.float32)[die].transpose(0, 3, 1, 2).reshape(len(y), -1)
            smote = SMOTE(k_neighbors=min(smote_k_neighbors, min_count - 1), random_state=seed)
            X_res, y_res = smote.fit_resample(X, y)
            self._X = X_res.reshape(-1, 3, image_size, image_size)
            print(f"  SMOTE: {len(y)} -> {len(y_res)} encoded samples")
        self._y = y_res.astype(np.int64)

        counts = np.bincount(self._y, minlength=num_classes)
        print(f"  Balanced class counts: {counts.tolist()}")

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._X is None:
            image = die_map_to_onehot(self._die[self._indices[index]])
        else:
            image = torch.from_numpy(self._X[index].copy())
        if self.augment:
            image = geometric_augment(image, self._aug_rng)
        label = int(self._y[index])
        mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
        return {"image": image, "label": label, "mask": mask}

    @property
    def num_samples(self) -> int:
        return len(self._y)


class UnlabeledWM811KDataset(BaseDataset):
    """Unlabeled WM-811K maps for SSL (empty failureType)."""

    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 32,
        max_samples: int = 150_000,
        transform: callable | None = None,
        train: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.images_dir = self.data_root / "images"
        self.labels_path = self.data_root / "labels.csv"
        self.train = train
        self._aug_rng = np.random.RandomState(seed + 11)
        _ = transform

        self._filenames = parse_wm811k_unlabeled_filenames(
            self.labels_path,
            max_samples=max_samples,
            seed=seed,
        )
        if not self._filenames:
            raise ValueError(
                f"No unlabeled samples found in {self.labels_path}. "
                "Full WM-811K from LSWMD.pkl (scripts/unpack_lswmd.py) is required "
                "for semi-supervised training — rows with empty failureType are unlabeled."
            )

    def __len__(self) -> int:
        return len(self._filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename = self._filenames[index]
        image = encode_wafer_image(_resolve_image_path(self.images_dir, filename), self.image_size)
        if self.train:
            image = geometric_augment(image, self._aug_rng)
        return {"image": image}

    @property
    def num_samples(self) -> int:
        return len(self._filenames)


class UnlabeledWM811KDirectoryDataset(BaseDataset):
    """Categorical maps from an explicit external unlabeled image directory.

    Match the labeled classifier's nearest-resized, three-channel encoding.
    MC pseudo-label generation always receives deterministic unaugmented maps.
    """

    def __init__(
        self,
        image_dir: str | Path,
        image_size: int = 32,
        max_samples: int | None = 150_000,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION)
        self.images_dir = Path(image_dir)
        self.image_size = image_size
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None")
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Unlabeled image directory not found: {self.images_dir}")
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        with os.scandir(self.images_dir) as entries:
            self._filenames = sorted(
                entry.name for entry in entries
                if entry.is_file() and Path(entry.name).suffix.lower() in extensions
            )
        if not self._filenames:
            raise ValueError(f"No unlabeled wafer images found in {self.images_dir}")
        if max_samples is not None and max_samples < len(self._filenames):
            indices = np.random.RandomState(seed).choice(
                len(self._filenames), size=max_samples, replace=False
            )
            self._filenames = [self._filenames[index] for index in sorted(indices.tolist())]

    def __len__(self) -> int:
        return len(self._filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"image": encode_wafer_image(
            self.images_dir / self._filenames[index], self.image_size
        )}

    @property
    def num_samples(self) -> int:
        return len(self._filenames)


# Back-compat alias used by older imports / docs
def default_train_transform(image_size: int):
    """Deprecated: categorical maps use geometric_augment on tensors."""
    _ = image_size
    return None
