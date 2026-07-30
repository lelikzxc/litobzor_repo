"""Generic, paper-agnostic Trainer for supervised deep learning.

Works with any ``nn.Module``. Provides:
- ``fit(train_loader, val_loader, epochs)`` — full training loop
- ``train_one_epoch(loader)`` — single epoch of training
- ``validate(loader)`` — single epoch of validation
- ``predict(loader)`` — inference on a dataloader
- ``save_checkpoint()`` / ``load_checkpoint()`` — checkpoint persistence
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.training.checkpoint import CheckpointManager
from common.training.early_stopping import EarlyStopping
from common.training.logger import TrainingLogger
from common.training.utils import NativeScaler, clip_gradients, move_batch_to_device


class Trainer:
    """Generic supervised trainer for any ``nn.Module``.

    Args:
        model: The model to train.
        optimizer: Optimizer instance.
        loss_fn: Loss function (callable accepting ``(logits, targets)``).
        scheduler: Optional LR scheduler.
        device: Device string (``"cpu"``, ``"cuda"``, ``"auto"``).
        metric_fns: Optional dict of metric name → callable ``(logits, targets) -> float``.
        early_stopping: Optional ``EarlyStopping`` instance.
        checkpoint_manager: Optional ``CheckpointManager`` instance.
        logger: Optional ``TrainingLogger`` instance.
        scaler: Optional ``NativeScaler`` instance for mixed precision.
        grad_max_norm: Max gradient norm for clipping (``None`` = no norm clip).
        grad_max_value: Max gradient value for clipping (``None`` = no value clip).
        verbose: If ``True``, show progress bars.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: Any | None = None,
        device: str = "auto",
        metric_fns: dict[str, Callable] | None = None,
        early_stopping: EarlyStopping | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        logger: TrainingLogger | None = None,
        scaler: NativeScaler | None = None,
        grad_max_norm: float | None = None,
        grad_max_value: float | None = None,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.metric_fns = metric_fns or {}
        self.early_stopping = early_stopping
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger or TrainingLogger()
        self.scaler = scaler or NativeScaler(enabled=True)
        self.grad_max_norm = grad_max_norm
        self.grad_max_value = grad_max_value
        self.verbose = verbose

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Internal state
        self.current_epoch: int = 0
        self._best_val_loss: float = float("inf")

    def resume_from_checkpoint(
        self,
        checkpoint_path: str | Path | None = None,
        load_last: bool = True,
    ) -> int:
        """Load model, optimizer, and scheduler state from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file. If ``None``, uses
                ``last.pt`` (when ``load_last=True``) or ``best.pt`` from
                the checkpoint manager's save directory.
            load_last: If ``True`` (default), loads ``last.pt``. Otherwise
                loads ``best.pt``. Ignored if ``checkpoint_path`` is given.

        Returns:
            The epoch number stored in the checkpoint (0 if no checkpoint).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if self.checkpoint_manager is None:
            raise RuntimeError(
                "Cannot resume: no CheckpointManager configured. "
                "Pass checkpoint_manager to Trainer()."
            )

        if checkpoint_path is not None:
            state = self.checkpoint_manager.load(
                self.model, self.optimizer, self.scheduler,
                checkpoint_path=checkpoint_path,
            )
        elif load_last:
            state = self.checkpoint_manager.load_last(
                self.model, self.optimizer, self.scheduler,
            )
        else:
            state = self.checkpoint_manager.load_best(
                self.model, self.optimizer, self.scheduler,
            )

        resumed_epoch = state.get("epoch", 0)
        self.current_epoch = resumed_epoch
        self.model = self.model.to(self.device)
        return resumed_epoch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
    ) -> TrainingLogger:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            epochs: Number of epochs to train.

        Returns:
            The ``TrainingLogger`` with all recorded metrics.
        """
        for _ in range(epochs):
            self.current_epoch += 1
            self.logger.begin_epoch()

            # Training
            train_metrics = self.train_one_epoch(train_loader)

            # Validation
            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

            # Learning rate
            current_lr = self._get_current_lr()

            # Log
            log_entry: dict[str, float] = {
                "train_loss": train_metrics.get("loss", 0.0),
                "lr": current_lr,
            }
            for key, value in train_metrics.items():
                if key != "loss":
                    log_entry[f"train_{key}"] = value

            if val_metrics:
                log_entry["val_loss"] = val_metrics.get("loss", 0.0)
                for key, value in val_metrics.items():
                    if key != "loss":
                        log_entry[f"val_{key}"] = value

            self.logger.log_epoch(**log_entry)

            # Print progress
            if self.verbose:
                self._print_epoch_summary()

            # Scheduler step (plateau uses val_loss, others step every epoch)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    val_loss = val_metrics.get("loss", 0.0) if val_metrics else 0.0
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Checkpoint (save last every epoch)
            if self.checkpoint_manager is not None:
                val_loss = val_metrics.get("loss", None) if val_metrics else None
                self.checkpoint_manager.save_last(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=self.current_epoch,
                    metric=val_loss,
                )
                if val_loss is not None:
                    val_accuracy = val_metrics.get("accuracy", None)
                    self.checkpoint_manager.save_best(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=self.current_epoch,
                        metric=val_accuracy,
                    )

            # Early stopping
            if self.early_stopping is not None and val_metrics:
                val_loss = val_metrics.get("loss", 0.0)
                should_stop = self.early_stopping.step(val_loss, self.model)
                if should_stop:
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {self.current_epoch}")
                    if self.early_stopping.restore_best_weights:
                        self.early_stopping.restore(self.model)
                    break

        return self.logger

    def train_one_epoch(
        self, loader: DataLoader
    ) -> dict[str, float]:
        """Train the model for one epoch.

        Args:
            loader: DataLoader yielding ``(inputs, targets)`` or ``dict``.

        Returns:
            Dictionary with ``"loss"`` and any configured metric values.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Per-batch metric accumulation to avoid OOM on large datasets
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        iterator = tqdm(loader, desc="Train", disable=not self.verbose)
        for batch in iterator:
            inputs, targets = self._unpack_batch(batch)
            inputs = move_batch_to_device(inputs, self.device)
            targets = move_batch_to_device(targets, self.device)

            self.optimizer.zero_grad()

            with self.scaler.autocast():
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)

            self.scaler.backward(loss, self.optimizer)
            clip_gradients(self.model, self.grad_max_norm, self.grad_max_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            # Compute metrics per-batch and accumulate
            if self.metric_fns:
                for name, fn in self.metric_fns.items():
                    try:
                        val = fn(logits.detach(), targets.detach())
                        metric_sums[name] = metric_sums.get(name, 0.0) + val
                        metric_counts[name] = metric_counts.get(name, 0) + 1
                    except Exception:
                        pass

            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics: dict[str, float] = {"loss": total_loss / max(num_batches, 1)}

        # Average per-batch metrics
        for name in metric_sums:
            metrics[name] = metric_sums[name] / max(metric_counts.get(name, 1), 1)

        return metrics

    def validate(
        self, loader: DataLoader
    ) -> dict[str, float]:
        """Evaluate the model on a validation set.

        Args:
            loader: DataLoader yielding ``(inputs, targets)`` or ``dict``.

        Returns:
            Dictionary with ``"loss"`` and any configured metric values.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Per-batch metric accumulation to avoid OOM on large datasets
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        with torch.no_grad():
            iterator = tqdm(loader, desc="Val", disable=not self.verbose)
            for batch in iterator:
                inputs, targets = self._unpack_batch(batch)
                inputs = move_batch_to_device(inputs, self.device)
                targets = move_batch_to_device(targets, self.device)

                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)

                total_loss += loss.item()
                num_batches += 1

                # Compute metrics per-batch and accumulate
                if self.metric_fns:
                    for name, fn in self.metric_fns.items():
                        try:
                            val = fn(logits.detach(), targets.detach())
                            metric_sums[name] = metric_sums.get(name, 0.0) + val
                            metric_counts[name] = metric_counts.get(name, 0) + 1
                        except Exception:
                            pass

        metrics: dict[str, float] = {"loss": total_loss / max(num_batches, 1)}

        # Average per-batch metrics
        for name in metric_sums:
            metrics[name] = metric_sums[name] / max(metric_counts.get(name, 1), 1)

        return metrics

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> list[torch.Tensor]:
        """Run inference on a dataloader.

        Args:
            loader: DataLoader yielding ``(inputs, ...)`` or ``dict``.

        Returns:
            List of output tensors, one per batch.
        """
        self.model.eval()
        predictions: list[torch.Tensor] = []

        iterator = tqdm(loader, desc="Predict", disable=not self.verbose)
        for batch in iterator:
            inputs, _ = self._unpack_batch(batch)
            inputs = move_batch_to_device(inputs, self.device)
            logits = self.model(inputs)
            predictions.append(logits.cpu())

        return predictions

    def save_checkpoint(self, path: str | Path) -> None:
        """Save a full training checkpoint.

        Args:
            path: Path to save the checkpoint file.
        """
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_val_loss": self._best_val_loss,
        }
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                state["scheduler_state"] = {
                    "_last_lr": self.scheduler._last_lr,
                    "best": self.scheduler.best,
                    "cooldown_counter": self.scheduler.cooldown_counter,
                    "num_bad_epochs": self.scheduler.num_bad_epochs,
                }
            else:
                state["scheduler"] = self.scheduler.state_dict()
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()

        os.makedirs(Path(path).parent, exist_ok=True)
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a full training checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The epoch number at which training was saved.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.current_epoch = state.get("epoch", 0)
        self._best_val_loss = state.get("best_val_loss", float("inf"))

        if self.scheduler is not None:
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            elif "scheduler_state" in state:
                sched = self.scheduler
                sched._last_lr = state["scheduler_state"].get(
                    "_last_lr", sched._last_lr
                )
                sched.best = state["scheduler_state"].get("best", sched.best)
                sched.cooldown_counter = state["scheduler_state"].get(
                    "cooldown_counter", sched.cooldown_counter
                )
                sched.num_bad_epochs = state["scheduler_state"].get(
                    "num_bad_epochs", sched.num_bad_epochs
                )

        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        return self.current_epoch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> tuple[Any, torch.Tensor]:
        """Unpack a batch into ``(inputs, targets)``.

        Supports:
        - ``(inputs, targets)`` tuple
        - ``{"inputs": ..., "targets": ...}`` dict
        - Single tensor (assumed to be inputs, no targets)
        """
        if isinstance(batch, dict):
            return batch.get("inputs", batch.get("input")), batch.get(
                "targets", batch.get("target")
            )
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        return batch, torch.tensor(0)

    def _get_current_lr(self) -> float:
        """Get the current learning rate from the optimizer."""
        for param_group in self.optimizer.param_groups:
            return float(param_group["lr"])
        return 0.0

    def _print_epoch_summary(self) -> None:
        """Print a one-line summary of the current epoch."""
        latest = self.logger.latest()
        parts = [f"Epoch {self.current_epoch}"]
        if "train_loss" in latest:
            parts.append(f"Train Loss: {latest['train_loss']:.4f}")
        if "val_loss" in latest:
            parts.append(f"Val Loss: {latest['val_loss']:.4f}")
        if "lr" in latest:
            parts.append(f"LR: {latest['lr']:.2e}")
        if "epoch_time" in latest:
            parts.append(f"Time: {latest['epoch_time']:.2f}s")
        print(" | ".join(parts))
        print("-" * 60)