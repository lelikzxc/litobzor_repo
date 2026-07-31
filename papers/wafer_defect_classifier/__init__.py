"""Wafer Defect Classifier — классификация дефектов пластин.

Пайплайн из двух этапов:
1. Сегментация дефектов (SegmentationModel)
2. Классификация типов дефектов (ClassificationModel)

Основан на ноутбуке pipeline (3).ipynb.
"""

from __future__ import annotations

from papers.wafer_defect_classifier.models import ClassificationModel, SegmentationModel

__all__ = ["SegmentationModel", "ClassificationModel"]