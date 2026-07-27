"""Data loading utilities for CTM-IYOLOv10.

Provides ``MagneticTileDataset`` for loading the magnetic tile defect dataset,
and ``DetectionDataset`` for synthetic detection data.
"""

from __future__ import annotations

from papers.ctm_yolov10.data_utils.dataset import DetectionDataset
from papers.ctm_yolov10.data_utils.magnetic_dataset import MagneticTileDataset

__all__ = [
    "DetectionDataset",
    "MagneticTileDataset",
]