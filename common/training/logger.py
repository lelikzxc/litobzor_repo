"""Lightweight training logger for recording losses, metrics, learning rate, and epoch time.

Output is easily printable and serializable.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any


class TrainingLogger:
    """Records per-epoch training and validation metrics.

    Usage::

        logger = TrainingLogger()
        logger.log_epoch(
            train_loss=0.5, val_loss=0.4, train_acc=0.85, val_acc=0.88, lr=1e-3
        )
        print(logger.summary())
        df = logger.to_dataframe()  # requires pandas
    """

    def __init__(self) -> None:
        self._history: list[dict[str, float]] = []
        self._current_epoch_start: float | None = None

    @property
    def history(self) -> list[dict[str, float]]:
        """List of per-epoch metric dictionaries."""
        return list(self._history)

    def begin_epoch(self) -> None:
        """Mark the start of an epoch (for timing)."""
        self._current_epoch_start = time.time()

    def log_epoch(self, **kwargs: float) -> None:
        """Record metrics for the current epoch.

        Args:
            **kwargs: Metric name → value pairs. Common keys:
                ``train_loss``, ``val_loss``, ``train_<metric>``, ``val_<metric>``,
                ``lr``, ``epoch_time``.
        """
        record = dict(kwargs)

        # Auto-record epoch time if begin_epoch was called
        if self._current_epoch_start is not None and "epoch_time" not in record:
            record["epoch_time"] = time.time() - self._current_epoch_start
            self._current_epoch_start = None

        self._history.append(record)

    def summary(self) -> str:
        """Return a human-readable summary of all recorded metrics.

        Returns:
            A formatted string with per-epoch and aggregated statistics.
        """
        if not self._history:
            return "No training history recorded."

        lines: list[str] = []
        lines.append("=" * 72)
        lines.append(f"{'Epoch':>6} | {'Train Loss':>10} {'Val Loss':>10} {'LR':>10} {'Time':>8}")
        lines.append("-" * 72)

        for i, record in enumerate(self._history):
            train_loss = record.get("train_loss", float("nan"))
            val_loss = record.get("val_loss", float("nan"))
            lr = record.get("lr", float("nan"))
            epoch_time = record.get("epoch_time", float("nan"))
            lines.append(
                f"{i:>6} | {train_loss:>10.4f} {val_loss:>10.4f} {lr:>10.2e} {epoch_time:>7.2f}s"
            )

        lines.append("-" * 72)

        # Aggregated statistics
        agg: dict[str, list[float]] = defaultdict(list)
        for record in self._history:
            for key, value in record.items():
                agg[key].append(value)

        if agg:
            lines.append("Aggregated:")
            for key, values in sorted(agg.items()):
                mean_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                lines.append(f"  {key:>20}: mean={mean_val:.4f}  min={min_val:.4f}  max={max_val:.4f}")

        lines.append("=" * 72)
        return "\n".join(lines)

    def latest(self) -> dict[str, float]:
        """Return the most recent epoch's metrics.

        Returns:
            Dictionary of the last recorded metrics, or empty dict if none.
        """
        if self._history:
            return dict(self._history[-1])
        return {}

    def to_dict(self) -> dict[str, list[float]]:
        """Convert history to a dictionary of lists (one key per metric).

        Returns:
            Dictionary mapping metric names to lists of per-epoch values.
        """
        result: dict[str, list[float]] = defaultdict(list)
        for record in self._history:
            for key, value in record.items():
                result[key].append(value)
        return dict(result)

    def to_dataframe(self) -> Any:
        """Convert history to a pandas DataFrame.

        Returns:
            A ``pandas.DataFrame`` with one row per epoch.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            )
        return pd.DataFrame(self._history)

    def reset(self) -> None:
        """Clear all recorded history."""
        self._history.clear()
        self._current_epoch_start = None