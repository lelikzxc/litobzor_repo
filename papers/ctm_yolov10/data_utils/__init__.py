"""Data loading utilities for CTM-YOLOv10.

Provides dataset adapters built on top of ``common.datasets``.
"""

from __future__ import annotations

from papers.ctm_yolov10.data_utils.dataset import DetectionDataset

__all__ = [
    "DetectionDataset",
]