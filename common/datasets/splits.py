"""Dataset splitting utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import random_split


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = False,
    labels: list[Any] | None = None,
) -> dict[str, Subset]:
    """Split a dataset into train/val/test subsets.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    train_ratio : float
        Proportion for training (default 0.7).
    val_ratio : float
        Proportion for validation (default 0.15).
    test_ratio : float
        Proportion for testing (default 0.15).
    seed : int
        Random seed for reproducibility (default 42).
    stratify : bool
        Whether to perform stratified split (default False).
    labels : list[Any] | None
        Labels for stratification. Required if ``stratify=True``.

    Returns
    -------
    dict[str, Subset]
        Dict with keys ``"train"``, ``"val"``, ``"test"``.

    Raises
    ------
    ValueError
        If ratios do not sum to 1.0, or if ``stratify=True`` without labels.
    """
    total = len(dataset)

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    if stratify:
        if labels is None:
            raise ValueError("labels must be provided when stratify=True")
        return _stratified_split(dataset, labels, [train_size, val_size, test_size], seed)

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator,
    )

    return {"train": train_subset, "val": val_subset, "test": test_subset}


def _stratified_split(
    dataset: Dataset,
    labels: list[Any],
    sizes: list[int],
    seed: int,
) -> dict[str, Subset]:
    """Perform a stratified split using sklearn's ``train_test_split``.

    Falls back to random split if sklearn is not available.
    """
    try:
        from sklearn.model_selection import train_test_split as sk_split

        indices = list(range(len(dataset)))
        train_idx, temp_idx = sk_split(
            indices,
            train_size=sizes[0],
            test_size=sizes[1] + sizes[2],
            random_state=seed,
            stratify=labels,
        )
        val_labels = [labels[i] for i in temp_idx]
        val_idx, test_idx = sk_split(
            temp_idx,
            train_size=sizes[1],
            test_size=sizes[2],
            random_state=seed,
            stratify=val_labels,
        )
        return {
            "train": Subset(dataset, train_idx),
            "val": Subset(dataset, val_idx),
            "test": Subset(dataset, test_idx),
        }
    except ImportError:
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset, test_subset = random_split(
            dataset, sizes, generator=generator,
        )
        return {"train": train_subset, "val": val_subset, "test": test_subset}