"""Checkpoint manager for saving and loading model checkpoints.

Stores model state dict, optimizer state dict, scheduler state dict,
current epoch, and best metric value.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim


class CheckpointManager:
    """Manages saving and loading of training checkpoints.

    Args:
        save_dir: Directory where checkpoints are stored.
        metric_name: Name of the tracked metric (for ``save_best``).
        mode: ``"max"`` (higher is better) or ``"min"`` (lower is better).
    """

    def __init__(
        self,
        save_dir: str | Path,
        metric_name: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.metric_name = metric_name
        self.mode = mode

        self.best_path = self.save_dir / "best.pt"
        self.last_path = self.save_dir / "last.pt"

        self._best_metric: float | None = None
        self._best_metric = -float("inf") if mode == "max" else float("inf")

        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def best_metric(self) -> float | None:
        """Best metric value seen so far."""
        return self._best_metric

    def save_best(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int = 0,
        metric: float | None = None,
    ) -> bool:
        """Save a checkpoint if the metric is the best so far.

        Args:
            model: The model to checkpoint.
            optimizer: Optional optimizer state.
            scheduler: Optional scheduler state.
            epoch: Current epoch number.
            metric: Metric value to compare. If ``None``, always saves.

        Returns:
            ``True`` if the checkpoint was saved (new best), ``False`` otherwise.
        """
        if metric is not None:
            is_better = (
                metric > self._best_metric
                if self.mode == "max"
                else metric < self._best_metric
            )
            if not is_better:
                return False
            self._best_metric = metric

        state = self._build_state(model, optimizer, scheduler, epoch, metric)
        torch.save(state, self.best_path)
        return True

    def save_last(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int = 0,
        metric: float | None = None,
    ) -> None:
        """Save the latest checkpoint (always overwrites).

        Args:
            model: The model to checkpoint.
            optimizer: Optional optimizer state.
            scheduler: Optional scheduler state.
            epoch: Current epoch number.
            metric: Optional metric value.
        """
        state = self._build_state(model, optimizer, scheduler, epoch, metric)
        torch.save(state, self.last_path)

    def load(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint.

        Args:
            model: The model to load state into.
            optimizer: Optional optimizer to load state into.
            scheduler: Optional scheduler to load state into.
            checkpoint_path: Path to checkpoint file. If ``None``, loads
                ``best.pt`` from ``save_dir``.

        Returns:
            The full checkpoint dictionary (contains ``"model"``, ``"optimizer"``,
            ``"scheduler"``, ``"epoch"``, ``"metric"``, ``"metric_name"``).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(checkpoint_path) if checkpoint_path else self.best_path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])

        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])

        if scheduler is not None and "scheduler" in state:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau stores internal state differently
                if "scheduler_state" in state:
                    scheduler._last_lr = state["scheduler_state"].get(
                        "_last_lr", scheduler._last_lr
                    )
                    scheduler.best = state["scheduler_state"].get(
                        "best", scheduler.best
                    )
                    scheduler.cooldown_counter = state["scheduler_state"].get(
                        "cooldown_counter", scheduler.cooldown_counter
                    )
                    scheduler.num_bad_epochs = state["scheduler_state"].get(
                        "num_bad_epochs", scheduler.num_bad_epochs
                    )
            else:
                scheduler.load_state_dict(state["scheduler"])

        self._best_metric = state.get("metric", None)
        return state

    def load_last(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> dict[str, Any]:
        """Load the last checkpoint (``last.pt``).

        Args:
            model: The model to load state into.
            optimizer: Optional optimizer to load state into.
            scheduler: Optional scheduler to load state into.

        Returns:
            The full checkpoint dictionary.

        Raises:
            FileNotFoundError: If ``last.pt`` does not exist.
        """
        return self.load(model, optimizer, scheduler, checkpoint_path=self.last_path)

    def load_best(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> dict[str, Any]:
        """Load the best checkpoint (``best.pt``).

        Args:
            model: The model to load state into.
            optimizer: Optional optimizer to load state into.
            scheduler: Optional scheduler to load state into.

        Returns:
            The full checkpoint dictionary.

        Raises:
            FileNotFoundError: If ``best.pt`` does not exist.
        """
        return self.load(model, optimizer, scheduler, checkpoint_path=self.best_path)

    def _build_state(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None,
        scheduler: Any | None,
        epoch: int,
        metric: float | None,
    ) -> dict[str, Any]:
        """Build a checkpoint dictionary."""
        state: dict[str, Any] = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metric": metric if metric is not None else self._best_metric,
            "metric_name": self.metric_name,
        }
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                state["scheduler_state"] = {
                    "_last_lr": scheduler._last_lr,
                    "best": scheduler.best,
                    "cooldown_counter": scheduler.cooldown_counter,
                    "num_bad_epochs": scheduler.num_bad_epochs,
                }
            else:
                state["scheduler"] = scheduler.state_dict()
        return state