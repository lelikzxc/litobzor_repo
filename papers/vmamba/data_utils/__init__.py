"""Data loading utilities for FCS-VMamba.

Provides dataset adapters built on top of ``common.datasets``.
"""

from __future__ import annotations

from papers.vmamba.data_utils.dataset import VMambaDataset

__all__ = [
    "VMambaDataset",
]