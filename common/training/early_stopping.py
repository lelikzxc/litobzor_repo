"""Generic early stopping with patience and best-weight restoration."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs with no improvement after which training stops.
        min_delta: Minimum change in the monitored metric to qualify as improvement.
        mode: ``"min"`` (lower is better, e.g. loss) or ``"max"`` (higher is better).
        restore_best_weights: If ``True``, restore model parameters from the best epoch.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_metric: float = (
            float("inf") if mode == "min" else -float("inf")
        )
        self.best_epoch: int = 0
        self.counter: int = 0
        self.early_stop: bool = False
        self._best_state: dict[str, torch.Tensor] | None = None

    def step(self, metric: float, model: nn.Module | None = None) -> bool:
        """Check if training should stop.

        Call after each validation epoch.

        Args:
            metric: Current metric value.
            model: Optional model whose state to save for later restoration.
                Required if ``restore_best_weights`` is ``True``.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        is_better = (
            metric < self.best_metric - self.min_delta
            if self.mode == "min"
            else metric > self.best_metric + self.min_delta
        )

        if is_better:
            self.best_metric = metric
            self.best_epoch += self.counter
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self._best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore(self, model: nn.Module) -> None:
        """Restore model parameters from the best epoch.

        Args:
            model: The model whose parameters will be overwritten.

        Raises:
            RuntimeError: If no best state has been saved.
        """
        if self._best_state is None:
            raise RuntimeError(
                "No best state to restore. Call step() at least once "
                "with a model argument."
            )
        model.load_state_dict(self._best_state)

    def reset(self) -> None:
        """Reset the early stopping state for a new training run."""
        self.best_metric = float("inf") if self.mode == "min" else -float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self._best_state = None