"""Data loading utilities for FCS-VMamba.

Provides dataset adapters for real WM-811K wafer data.
"""

from __future__ import annotations

from papers.vmamba.data_utils.wafer_dataset import WaferWM811KDataset

__all__ = [
    "WaferWM811KDataset",
]