"""EngineState dataclass for tracking training progress.

Tracks epoch, best metric, current metric, global step,
training finished flag, and checkpoint path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EngineState:
    """Mutable state container for the training Engine.

    Attributes:
        epoch: Current epoch number (1-based).
        best_metric: Best metric value observed so far.
        best_metric_name: Name of the tracked metric.
        current_metric: Most recent metric value.
        global_step: Total number of optimizer steps taken.
        training_finished: Whether training has completed.
        checkpoint_path: Path to the most recent checkpoint file.
        mode: ``"min"`` (lower is better) or ``"max"`` (higher is better).
        extra: Additional user-defined state values.
    """

    epoch: int = 0
    best_metric: float | None = None
    best_metric_name: str = "val_loss"
    current_metric: float | None = None
    global_step: int = 0
    training_finished: bool = False
    checkpoint_path: str | Path | None = None
    mode: str = "min"

    extra: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset all state fields to their defaults."""
        self.epoch = 0
        self.best_metric = None
        self.current_metric = None
        self.global_step = 0
        self.training_finished = False
        self.checkpoint_path = None
        self.extra.clear()

    def update_metric(self, value: float) -> bool:
        """Update the current metric and check if it is the best so far.

        Args:
            value: The latest metric value.

        Returns:
            ``True`` if the new value is better than the previous best.
        """
        self.current_metric = value
        if self.best_metric is None:
            self.best_metric = value
            return True
        is_better = (
            value < self.best_metric
            if self.mode == "min"
            else value > self.best_metric
        )
        if is_better:
            self.best_metric = value
        return is_better

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a dictionary.

        Returns:
            Dictionary with all state fields.
        """
        return {
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name,
            "current_metric": self.current_metric,
            "global_step": self.global_step,
            "training_finished": self.training_finished,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "mode": self.mode,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineState:
        """Create an EngineState from a dictionary.

        Args:
            data: Dictionary with state fields.

        Returns:
            A new ``EngineState`` instance.
        """
        extra = data.get("extra", {})
        checkpoint_path = data.get("checkpoint_path")
        return cls(
            epoch=data.get("epoch", 0),
            best_metric=data.get("best_metric"),
            best_metric_name=data.get("best_metric_name", "val_loss"),
            current_metric=data.get("current_metric"),
            global_step=data.get("global_step", 0),
            training_finished=data.get("training_finished", False),
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
            mode=data.get("mode", "min"),
            extra=dict(extra),
        )