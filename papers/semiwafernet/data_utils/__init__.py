"""Data loading utilities for SemiWaferNet.

Provides dataset adapters built on top of ``common.datasets``:

- ``LabeledWaferDataset`` — multitask dataset returning images, labels, and masks
- ``UnlabeledWaferDataset`` — image-only dataset for semi-supervised training
"""

from __future__ import annotations

from papers.semiwafernet.data_utils.dataset import LabeledWaferDataset, UnlabeledWaferDataset

__all__ = [
    "LabeledWaferDataset",
    "UnlabeledWaferDataset",
]