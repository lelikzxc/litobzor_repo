"""WM-811K wafer map dataset loader for SemiWaferNet.

WM-811K PNGs store categorical die states ``{0, 1, 2}`` (background / normal /
defect), NOT natural-image intensities. Encoding follows the paper's
``X ∈ R^{3×H×W}`` by one-hot channels after nearest-neighbor resize.

Augmentations are geometric only (flip / 90° rotations) so categories stay
intact — no ColorJitter / bilinear.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from papers.semiwafernet.data_utils.base import BaseDataset, DatasetType

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

NONE_CLASS_IDX: int = 0


def load_wafer_die_map(image_path: Path) -> np.ndarray:
    """Load a wafer PNG as an integer die-state map ``{0,1,2}``."""
    arr = np.array(Image.open(image_path).convert("L"), dtype=np.int64)
    # Some exports may stretch to 0..255 — snap to 3-level die states.
    uniq = np.unique(arr)
    if uniq.max() > 2:
        # Map lowest→0, mid→1, highest→2 when possible
        levels = np.unique(arr)
        if len(levels) == 1:
            arr = np.zeros_like(arr)
        elif len(levels) == 2:
            arr = (arr == levels[1]).astype(np.int64)
        else:
            # tertile-like: use 0, mid, max present
            lo, hi = levels.min(), levels.max()
            mid = levels[len(levels) // 2]
            out = np.zeros_like(arr)
            out[arr == mid] = 1
            out[arr == hi] = 2
            arr = out
    return np.clip(arr, 0, 2)


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
        rng = np.random.RandomState()
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

    with open(labels_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    cols = [c.strip().lower() for c in header.split(",")]

    has_failure_type = "failuretype" in cols
    has_official_split = "triantestlabel" in cols
    fmt = "official" if has_failure_type else "simple"

    unlabeled_count = 0
    if has_failure_type:
        ft_idx = cols.index("failuretype")
        with open(labels_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) > ft_idx and _is_unlabeled_failure_type(parts[ft_idx]):
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


def parse_wm811k_labeled_rows(
    labels_path: Path,
    split: str | None = None,
) -> list[tuple[str, int]]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    samples: list[tuple[str, int]] = []
    split_filter = split.lower() if split else None

    with open(labels_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        header_l = header.lower()
        cols = [c.strip().lower() for c in header.split(",")]

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
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    filenames: list[str] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        cols = [c.strip().lower() for c in header.split(",")]
        if "failuretype" not in cols:
            return []
        ft_idx = cols.index("failuretype")
        fn_idx = cols.index("filename") if "filename" in cols else 0
        lines = f.readlines()

    for line in tqdm(
        lines,
        desc="  Parsing unlabeled WM-811K rows",
        unit="row",
        leave=False,
    ):
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


def _resolve_image_path(images_dir: Path, filename: str) -> Path:
    image_path = images_dir / filename
    if image_path.exists():
        return image_path
    png_path = images_dir / f"{Path(filename).stem}.png"
    if png_path.exists():
        return png_path
    raise FileNotFoundError(f"Image not found: {image_path} or {png_path}")


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
            raise ValueError(
                f"No labeled samples loaded from {self.labels_path} (split={split!r})"
            )

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
    """Balance minority classes for HybridCNN-ViT training (paper Section 4.1).

    Paper says SMOTE; on categorical die states ``{0,1,2}`` soft SMOTE creates
    invalid maps. Default ``method="random"`` uses RandomOverSampler (exact
    copies of real maps + online geometric augs). ``method="smote"`` keeps the
    interpolate-then-round path for ablations.
    """

    def __init__(
        self,
        data_root: str | Path,
        samples: list[tuple[str, int]],
        image_size: int = 32,
        num_classes: int = 9,
        smote_k_neighbors: int = 5,
        seed: int = 42,
        method: str = "random",
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_names = WM811K_CLASSES
        self.images_dir = self.data_root / "images"
        self._aug_rng = np.random.RandomState(seed + 7)
        method = str(method).lower().strip()

        images: list[np.ndarray] = []
        labels: list[int] = []
        for filename, label in tqdm(
            samples,
            desc="  Loading images for oversample",
            unit="img",
        ):
            die = load_wafer_die_map(_resolve_image_path(self.images_dir, filename))
            die = resize_die_map(die, image_size)
            images.append(die.astype(np.float32).reshape(-1))
            labels.append(label)

        X = np.stack(images)
        y = np.array(labels)

        unique_classes = np.unique(y)
        min_count = int(min(np.bincount(y)[unique_classes]))
        if len(unique_classes) < 2 or min_count < 2:
            X_res, y_res = X, y
            print(f"  Oversample skipped (classes={len(unique_classes)}, min_count={min_count})")
        elif method == "smote":
            from imblearn.over_sampling import SMOTE

            k = min(smote_k_neighbors, max(1, min_count - 1))
            smote = SMOTE(k_neighbors=k, random_state=seed)
            X_res, y_res = smote.fit_resample(X, y)
            print(f"  SMOTE (interp+round): {len(X)} -> {len(X_res)} samples")
        else:
            from imblearn.over_sampling import RandomOverSampler

            ros = RandomOverSampler(random_state=seed)
            X_res, y_res = ros.fit_resample(X, y)
            print(f"  RandomOverSampler (real copies): {len(X)} -> {len(X_res)} samples")

        die = np.rint(X_res).astype(np.int64).clip(0, 2).reshape(-1, image_size, image_size)
        oh = np.eye(3, dtype=np.float32)[die]
        self._X = np.transpose(oh, (0, 3, 1, 2)).astype(np.float32)
        self._y = y_res.astype(np.int64)

        counts = np.bincount(self._y, minlength=num_classes)
        print(f"  Balanced class counts: {counts.tolist()}")

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = torch.from_numpy(self._X[index].copy())
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
        image = encode_wafer_image(
            _resolve_image_path(self.images_dir, filename), self.image_size
        )
        if self.train:
            image = geometric_augment(image, self._aug_rng)
        return {"image": image}

    @property
    def num_samples(self) -> int:
        return len(self._filenames)


# Back-compat alias used by older imports / docs
def default_train_transform(image_size: int):
    """Deprecated: categorical maps use geometric_augment on tensors."""
    _ = image_size
    return None
