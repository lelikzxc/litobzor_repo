"""Shared dataset utilities and base classes."""

from __future__ import annotations

from common.datasets.augmentation import build_augmentations
from common.datasets.base_dataset import BaseDataset, DatasetType
from common.datasets.collate import (
    build_collate_fn,
    classification_collate,
    multitask_collate,
    segmentation_collate,
)
from common.datasets.datamodule import DataModule
from common.datasets.splits import split_dataset
from common.datasets.transforms import build_transforms
from common.datasets.utils import count_classes, read_image, read_mask, verify_dataset
from common.datasets.visualization import (
    overlay_mask,
    show_image,
    show_mask,
    visualize_batch,
)

__all__ = [
    "BaseDataset",
    "DatasetType",
    "DataModule",
    "build_transforms",
    "build_augmentations",
    "classification_collate",
    "segmentation_collate",
    "multitask_collate",
    "build_collate_fn",
    "split_dataset",
    "show_image",
    "show_mask",
    "overlay_mask",
    "visualize_batch",
    "read_image",
    "read_mask",
    "verify_dataset",
    "count_classes",
]
