"""Data loading utilities for CTM-YOLOv10.

Provides ``MagneticTileDataset`` for loading the magnetic tile defect dataset.
"""

from __future__ import annotations

from papers.ctm_yolov10.data_utils.magnetic_dataset import MagneticTileDataset

__all__ = [
    "MagneticTileDataset",
]