"""Collate functions for classification, segmentation, and multitask batching."""

from __future__ import annotations

from typing import Any

import torch


def classification_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch of classification samples.

    Each sample is a dict with ``"image"`` (``torch.Tensor``) and
    ``"label"`` (``int``). Images are stacked; labels are converted to a
    1-D tensor.

    Parameters
    ----------
    batch : list[dict[str, Any]]
        List of sample dicts from a classification dataset.

    Returns
    -------
    dict[str, Any]
        Batched dict with ``"image"`` and ``"label"`` tensors.
    """
    images = torch.stack([sample["image"] for sample in batch])
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    return {"image": images, "label": labels}


def segmentation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch of segmentation samples.

    Each sample is a dict with ``"image"`` (``torch.Tensor``) and
    ``"mask"`` (``torch.Tensor``). Images and masks are stacked.

    Parameters
    ----------
    batch : list[dict[str, Any]]
        List of sample dicts from a segmentation dataset.

    Returns
    -------
    dict[str, Any]
        Batched dict with ``"image"`` and ``"mask"`` tensors.
    """
    images = torch.stack([sample["image"] for sample in batch])
    masks = torch.stack([sample["mask"] for sample in batch])
    return {"image": images, "mask": masks}


def multitask_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a batch of multitask samples.

    Each sample is a dict with ``"image"`` (``torch.Tensor``),
    ``"label"`` (``int``), and ``"mask"`` (``torch.Tensor``).

    Parameters
    ----------
    batch : list[dict[str, Any]]
        List of sample dicts from a multitask dataset.

    Returns
    -------
    dict[str, Any]
        Batched dict with ``"image"``, ``"label"``, and ``"mask"`` tensors.
    """
    images = torch.stack([sample["image"] for sample in batch])
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    masks = torch.stack([sample["mask"] for sample in batch])
    return {"image": images, "label": labels, "mask": masks}


_COLLATE_REGISTRY: dict[str, callable] = {
    "classification": classification_collate,
    "segmentation": segmentation_collate,
    "multitask": multitask_collate,
}


def build_collate_fn(dataset_type: str) -> callable:
    """Return the appropriate collate function for a dataset type.

    Parameters
    ----------
    dataset_type : str
        One of ``"classification"``, ``"segmentation"``, ``"multitask"``.

    Returns
    -------
    callable
        The corresponding collate function.

    Raises
    ------
    ValueError
        If ``dataset_type`` is unknown.
    """
    if dataset_type not in _COLLATE_REGISTRY:
        raise ValueError(
            f"Unknown dataset type: '{dataset_type}'. "
            f"Expected one of: {list(_COLLATE_REGISTRY.keys())}"
        )
    return _COLLATE_REGISTRY[dataset_type]