"""Common inference infrastructure.

Provides:
    - Predictor: high-level inference wrapper with postprocessing.
    - device utilities: ``get_best_device``, ``model_size_mb``, etc.
"""

from common.inference.predictor import Predictor

__all__ = [
    "Predictor",
]