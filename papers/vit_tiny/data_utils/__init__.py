"""Data loading utilities for Tiny Vision Transformer.

Provides dataset adapters for synthetic data and real WM-38k wafer data.
"""

from __future__ import annotations

from papers.vit_tiny.data_utils.dataset import ViTTinyDataset
from papers.vit_tiny.data_utils.wafer_dataset import WaferWM38KDataset

__all__ = [
    "ViTTinyDataset",
    "WaferWM38KDataset",
]