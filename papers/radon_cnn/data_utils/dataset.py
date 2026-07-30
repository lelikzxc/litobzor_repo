"""WM-811K wafer map dataset for RadonCNN.

Loads wafer map images from the WM-811K dataset (labels.csv + PNG images)
and returns classification samples compatible with RadonCNN:

    ``{"inputs": [1, 64, 64], "targets": int}``

Preprocessing per the paper (Section "Experimental setup"):
    1. Resize wafer map to (64, 64)
    2. Remove wafer map background, retaining only defect points
    3. Apply Radon transform (rotation → translation)
    4. Use 7 classes (exclude 'Near-full' and 'None')
    5. Balanced sampling for each class

The Radon transform is applied at dataset load time (not in the model forward
pass) because:
    - It is a deterministic non-learnable operation (skimage, numpy)
    - Applying it in the model would break gradient flow (detach → numpy → back)
    - Pre-computing is ~3x faster (no redundant computation per epoch)
    - This matches the paper: Radon is a preprocessing step
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from skimage.transform import radon, resize

from papers.radon_cnn.data_utils.base import BaseDataset, DatasetType

# RadonCNN uses 7 classes (excludes 'Near-full' and 'None')
RADONCNN_CLASSES: list[str] = [
    "Center",       # 0
    "Donut",        # 1
    "Edge-Loc",     # 2
    "Edge-Ring",    # 3
    "Loc",          # 4
    "Random",       # 5
    "Scratch",      # 6
]

# Mapping from WM-811K 9-class labels to RadonCNN 7-class labels
# WM-811K: 0=None, 1=Center, 2=Donut, 3=Edge-Loc, 4=Edge-Ring,
#           5=Loc, 6=Near-full, 7=Random, 8=Scratch
WM811K_TO_RADONCNN: dict[int, int] = {
    1: 0,  # Center    → Center
    2: 1,  # Donut     → Donut
    3: 2,  # Edge-Loc  → Edge-Loc
    4: 3,  # Edge-Ring → Edge-Ring
    5: 4,  # Loc       → Loc
    7: 5,  # Random    → Random
    8: 6,  # Scratch   → Scratch
}


def remove_background(wafer_map: np.ndarray) -> np.ndarray:
    """Remove wafer map background, retaining only defect points.

    Per the paper: "removed the wafer map background, retaining only the
    defect points due to varying wafer map sizes, which can lead to slightly
    different shapes on the sides after resizing, thus affecting model
    training negatively."

    In WM-811K, background pixels are 0, defect pixels are 1 or 2.
    We set background to 0 and defect to 1.
    """
    binary = (wafer_map > 0).astype(np.float32)
    return binary


def apply_radon_transform(
    image: np.ndarray,
    theta: int = 64,
    image_size: int = 64,
) -> np.ndarray:
    """Apply Radon transform to a single wafer map image.

    Per the paper (Eq. 11-12):
        r = x*cos(theta) + y*sin(theta)
        P_theta(r) = sum_x sum_y f(x,y) * delta(x*cos(theta) + y*sin(theta) - r)

    The Radon transform converts rotation in the original image to translation
    in the Radon feature space, serving as a rotation-equivariant bridge for
    translation-invariant CNNs.

    Args:
        image: Input image [H, W] as float32 numpy array.
        theta: Number of projection angles (default 64).
        image_size: Target size for the output sinogram (default 64).

    Returns:
        Radon-transformed image [image_size, image_size] as float32 numpy array.
    """
    theta_range = np.linspace(0.0, 180.0, theta, endpoint=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sinogram = radon(image, theta=theta_range, circle=False)
    # Resize to (image_size, image_size) for consistent CNN input
    sinogram = resize(
        sinogram,
        (image_size, image_size),
        preserve_range=True,
        anti_aliasing=True,
    )
    return sinogram.astype(np.float32)


class WaferRadonDataset(BaseDataset):
    """WM-811K wafer map dataset for RadonCNN (7 classes, balanced).

    Reads ``labels.csv`` from the dataset root and loads PNG images.
    Applies Radon transform at load time so the model receives pre-computed
    Radon features directly, avoiding gradient graph breaks.

    Args:
        data_root: Root directory containing ``labels.csv`` and ``images/``.
        image_size: Target image size (default 64 per paper).
        num_classes: Number of classes (default 7 for RadonCNN).
        balanced: If True, samples are balanced per class (default True).
        apply_radon: If True, apply Radon transform to images (default True).
        transform: Optional transform to apply to images.
    """

    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 64,
        num_classes: int = 7,
        balanced: bool = True,
        apply_radon: bool = True,
        transform: callable | None = None,
    ) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION, transform=transform)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.num_classes = num_classes
        self.apply_radon = apply_radon
        self.class_names = RADONCNN_CLASSES[:num_classes]

        self.labels_path = self.data_root / "labels.csv"
        self.images_dir = self.data_root / "images"

        self._samples: list[tuple[str, int]] = []  # (filename, label_idx)
        self._load_labels()

        if balanced and len(self._samples) > 0:
            self._samples = self._balance_samples(self._samples)

    def _load_labels(self) -> None:
        """Parse labels.csv and build (filename, label_idx) pairs for 7 classes."""
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        with open(self.labels_path, "r", encoding="utf-8") as f:
            f.readline()  # skip header

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
                    from papers.semiwafernet.data_utils.wafer_dataset import (
                        WM811K_CLASSES,
                    )
                    label_idx = (
                        WM811K_CLASSES.index(label_str)
                        if label_str in WM811K_CLASSES
                        else -1
                    )

                if label_idx in WM811K_TO_RADONCNN:
                    mapped_label = WM811K_TO_RADONCNN[label_idx]
                    self._samples.append((filename, mapped_label))

        if not self._samples:
            raise ValueError(
                f"No valid samples loaded from {self.labels_path} "
                f"(no samples from the 7 RadonCNN classes found)"
            )

    @staticmethod
    def _balance_samples(
        samples: list[tuple[str, int]],
    ) -> list[tuple[str, int]]:
        """Balance samples by downsampling to the minimum class count."""
        from collections import Counter

        class_counts = Counter(label for _, label in samples)
        min_count = min(class_counts.values())

        balanced: list[tuple[str, int]] = []
        class_buckets: dict[int, list[tuple[str, int]]] = {}
        for s in samples:
            class_buckets.setdefault(s[1], []).append(s)

        for label, bucket in class_buckets.items():
            rng = np.random.RandomState(42)
            indices = rng.choice(len(bucket), size=min_count, replace=False)
            balanced.extend([bucket[i] for i in indices])

        return balanced

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename, label = self._samples[index]
        image_path = self.images_dir / filename

        # Load image as grayscale (1 channel)
        if image_path.exists():
            image = Image.open(image_path).convert("L")
        else:
            png_path = self.images_dir / f"{Path(filename).stem}.png"
            if png_path.exists():
                image = Image.open(png_path).convert("L")
            else:
                raise FileNotFoundError(f"Image not found: {image_path} or {png_path}")

        # Resize to (image_size, image_size) per paper
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to numpy for background removal
        img_array = np.array(image, dtype=np.float32)

        # Remove background (keep only defect points) per paper
        img_array = remove_background(img_array)

        # Apply Radon transform at dataset level (not in model forward pass)
        if self.apply_radon:
            img_array = apply_radon_transform(
                img_array,
                theta=64,
                image_size=self.image_size,
            )

        # Convert to tensor [1, H, W]
        image_tensor = torch.from_numpy(img_array).unsqueeze(0)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return {
            "inputs": image_tensor,  # [1, H, W]
            "targets": label,         # int
        }

    @property
    def num_samples(self) -> int:
        return len(self._samples)