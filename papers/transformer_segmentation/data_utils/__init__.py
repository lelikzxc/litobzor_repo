"""Data loading utilities for SegFormer + Atrous.

Provides dataset adapters built on top of ``common.datasets``.
"""

from __future__ import annotations

from papers.transformer_segmentation.data_utils.dataset import SegFormerDataset

__all__ = [
    "SegFormerDataset",
]