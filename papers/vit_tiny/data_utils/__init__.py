"""Data loading utilities for Tiny Vision Transformer.

Provides dataset adapters built on top of ``common.datasets``.
"""

from __future__ import annotations

from papers.vit_tiny.data_utils.dataset import ViTTinyDataset

__all__ = [
    "ViTTinyDataset",
]