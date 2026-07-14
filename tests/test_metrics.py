"""Tests for common.metrics.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from common.metrics.metrics import MetricTracker, accuracy, f1, precision, recall


Y_TRUE = np.array([0, 1, 2, 0, 1, 2])
Y_PRED = np.array([0, 2, 1, 0, 1, 2])


def test_accuracy_perfect_predictions() -> None:
    labels = np.array([0, 1, 1, 0])
    assert accuracy(labels, labels) == pytest.approx(1.0)


def test_sklearn_metric_wrappers() -> None:
    assert 0.0 <= accuracy(Y_TRUE, Y_PRED) <= 1.0
    assert 0.0 <= precision(Y_TRUE, Y_PRED) <= 1.0
    assert 0.0 <= recall(Y_TRUE, Y_PRED) <= 1.0
    assert 0.0 <= f1(Y_TRUE, Y_PRED) <= 1.0


def test_metric_wrappers_return_float() -> None:
    assert isinstance(accuracy(Y_TRUE, Y_PRED), float)
    assert isinstance(precision(Y_TRUE, Y_PRED), float)
    assert isinstance(recall(Y_TRUE, Y_PRED), float)
    assert isinstance(f1(Y_TRUE, Y_PRED), float)


def test_metric_tracker_update_and_compute() -> None:
    tracker = MetricTracker()
    tracker.update("loss", 1.0)
    tracker.update("loss", 3.0)
    tracker.update("accuracy", 0.8)

    result = tracker.compute()
    assert result["loss"] == pytest.approx(2.0)
    assert result["accuracy"] == pytest.approx(0.8)


def test_metric_tracker_reset() -> None:
    tracker = MetricTracker()
    tracker.update("f1", 0.5)
    tracker.reset()
    assert tracker.compute() == {}
