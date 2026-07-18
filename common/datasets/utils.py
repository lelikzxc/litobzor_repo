"""Utility functions for dataset inspection and I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def read_image(path: str | Path) -> Image.Image:
    """Read an image from disk.

    Parameters
    ----------
    path : str | Path
        Path to the image file.

    Returns
    -------
    Image.Image
        PIL Image in RGB mode.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def read_mask(path: str | Path) -> np.ndarray:
    """Read a segmentation mask from disk.

    Parameters
    ----------
    path : str | Path
        Path to the mask file.

    Returns
    -------
    np.ndarray
        Mask as a 2-D numpy array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = Image.open(path)
    return np.array(mask)


def verify_dataset(
    dataset: Dataset,
    num_samples: int = 5,
) -> list[str]:
    """Verify a dataset by sampling a few items and checking structure.

    Parameters
    ----------
    dataset : Dataset
        The dataset to verify.
    num_samples : int
        Number of samples to check (default 5).

    Returns
    -------
    list[str]
        List of issue descriptions. Empty if no issues found.
    """
    issues: list[str] = []
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            if not isinstance(sample, dict):
                issues.append(f"Sample {i}: expected dict, got {type(sample).__name__}")
            elif "image" not in sample:
                issues.append(f"Sample {i}: missing 'image' key")
        except Exception as e:
            issues.append(f"Sample {i}: raised {type(e).__name__}: {e}")
    return issues


def count_classes(
    dataset: Dataset,
    label_key: str = "label",
) -> dict[Any, int]:
    """Count class occurrences in a dataset.

    Iterates over the entire dataset and counts occurrences of each class.

    Parameters
    ----------
    dataset : Dataset
        The dataset to scan.
    label_key : str
        Key to access the label in each sample dict (default ``"label"``).

    Returns
    -------
    dict[Any, int]
        Mapping from class label to count.

    Raises
    ------
    KeyError
        If a sample does not contain the specified ``label_key``.
    """
    counts: dict[Any, int] = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        if label_key not in sample:
            raise KeyError(f"Sample {i} has no key '{label_key}'")
        label = sample[label_key]
        # For mask tensors, count unique pixel values
        if isinstance(label, np.ndarray) or (hasattr(label, "unique")):
            if hasattr(label, "cpu"):
                unique = label.cpu().numpy()
            else:
                unique = np.unique(label)
            for cls in unique.ravel():
                counts[int(cls)] = counts.get(int(cls), 0) + 1
        else:
            counts[label] = counts.get(label, 0) + 1
    return counts