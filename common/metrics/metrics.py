"""Metric tracking and sklearn-based evaluation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

AverageStrategy = Literal["binary", "micro", "macro", "weighted"]


def accuracy(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
) -> float:
    """Compute classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def precision(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    average: AverageStrategy = "macro",
) -> float:
    """Compute classification precision."""
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))


def recall(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    average: AverageStrategy = "macro",
) -> float:
    """Compute classification recall."""
    return float(recall_score(y_true, y_pred, average=average, zero_division=0))


def f1(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    average: AverageStrategy = "macro",
) -> float:
    """Compute classification F1 score."""
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


@dataclass
class MetricTracker:
    """Container for accumulating and reporting training/evaluation metrics."""

    metrics: dict[str, list[float]] = field(default_factory=dict)

    def update(self, name: str, value: float) -> None:
        """Record a metric value.

        Args:
            name: Metric identifier.
            value: Metric value for the current step or epoch.
        """
        self.metrics.setdefault(name, []).append(value)

    def compute(self) -> dict[str, float]:
        """Compute aggregated metric statistics.

        Returns:
            Dictionary mapping each metric name to its mean value.
        """
        return {
            name: float(np.mean(values))
            for name, values in self.metrics.items()
            if values
        }

    def reset(self) -> None:
        """Clear all recorded metric values."""
        self.metrics.clear()
