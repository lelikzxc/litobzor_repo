"""High-level Engine class that connects common/data, common/training, and common/inference.

The Engine provides a unified API for training, validation, testing, prediction,
checkpointing, and inference across all paper implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from common.engine.builder import Builder
from common.engine.config import EngineConfig
from common.engine.state import EngineState
from common.inference import Predictor
from common.training import (
    CheckpointManager,
    EarlyStopping,
    NativeScaler,
    Trainer,
    TrainingLogger,
)


class Engine:
    """High-level interface connecting data, training, and inference.

    The Engine orchestrates the full lifecycle: model creation, training,
    validation, testing, prediction, checkpointing, and inference.

    Args:
        model: An ``nn.Module`` instance or a registered model name.
        config: Optional configuration (``EngineConfig``, dict, or YAML path).
        device: Device string (``"auto"``, ``"cpu"``, ``"cuda"``).
        **kwargs: Additional keyword arguments passed to the ``Builder``
            for component construction. See ``Builder.build_all()``.
    """

    def __init__(
        self,
        model: nn.Module | str,
        config: EngineConfig | dict[str, Any] | str | Path | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.config: EngineConfig | None = None
        if config is not None:
            if isinstance(config, EngineConfig):
                self.config = config
            elif isinstance(config, dict):
                self.config = EngineConfig.from_dict(config)
            elif isinstance(config, (str, Path)):
                self.config = EngineConfig.from_yaml(config)

        self.device = device
        self.builder = Builder(self.config)

        # Build all components
        components = self.builder.build_all(model=model, device=device)

        self.model: nn.Module = components["model"]
        self.optimizer: optim.Optimizer = components["optimizer"]
        self.scheduler: Any = components["scheduler"]
        self.loss_fn: nn.Module = components["loss_fn"]
        self.metric_fns: dict[str, Callable] = components["metric_fns"]
        self.checkpoint_manager: CheckpointManager | None = components["checkpoint_manager"]
        self.early_stopping: EarlyStopping | None = components["early_stopping"]
        self.logger: TrainingLogger = components["logger"]
        self.scaler: NativeScaler = components["scaler"]
        self.trainer: Trainer = components["trainer"]
        self.predictor: Predictor = components["predictor"]
        self.state: EngineState = components["state"]

        # Re-enable gradients for training (Predictor.disable_gradients disables them)
        for param in self.model.parameters():
            param.requires_grad_(True)

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
        self.state.training_finished = False
        self.state.epoch = 0

        logger = self.trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
        )

        # Sync state from trainer
        self.state.epoch = self.trainer.current_epoch
        self.state.training_finished = True

        # Update best metric from checkpoint manager
        if self.checkpoint_manager is not None and self.checkpoint_manager.best_metric is not None:
            self.state.best_metric = self.checkpoint_manager.best_metric

        return logger

    def resume(
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
        if checkpoint_path is not None:
            epoch = self.load(checkpoint_path)
        elif load_last:
            if self.checkpoint_manager is None:
                raise RuntimeError(
                    "Cannot resume: no CheckpointManager configured."
                )
            epoch = self.load(self.checkpoint_manager.last_path)
        else:
            if self.checkpoint_manager is None:
                raise RuntimeError(
                    "Cannot resume: no CheckpointManager configured."
                )
            epoch = self.load(self.checkpoint_manager.best_path)

        # Sync trainer's current_epoch so training continues from here
        self.trainer.current_epoch = epoch
        return epoch

    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a validation set.

        Args:
            loader: DataLoader for validation data.

        Returns:
            Dictionary with ``"loss"`` and any configured metric values.
        """
        return self.trainer.validate(loader)

    def test(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a test set.

        Alias for ``validate()``.

        Args:
            loader: DataLoader for test data.

        Returns:
            Dictionary with ``"loss"`` and any configured metric values.
        """
        return self.trainer.validate(loader)

    def predict(self, loader: DataLoader) -> list[dict[str, Any]]:
        """Run inference on a DataLoader using the Predictor.

        Args:
            loader: DataLoader yielding inputs.

        Returns:
            List of result dicts, one per batch, with keys
            ``"logits"``, ``"probs"``, ``"prediction"``.
        """
        return self.predictor.predict(loader)

    def predict_single(self, image: torch.Tensor) -> dict[str, Any]:
        """Run inference on a single image.

        Args:
            image: Input tensor ``[C, H, W]`` or ``[B, C, H, W]``.

        Returns:
            Dict with keys ``"logits"``, ``"probs"``, ``"prediction"``.
        """
        return self.predictor.predict_single(image)

    def save(self, path: str | Path | None = None) -> Path:
        """Save a full training checkpoint.

        Args:
            path: Path to save the checkpoint. If ``None``, uses
                ``checkpoint.save_dir / "last.pt"`` from config.

        Returns:
            The resolved path where the checkpoint was saved.
        """
        if path is None:
            if self.checkpoint_manager is not None:
                path = self.checkpoint_manager.last_path
            else:
                path = Path("checkpoints/last.pt")

        path = Path(path)
        self.trainer.save_checkpoint(path)

        # Also save engine state
        state_path = path.with_suffix(".state.json")
        import json
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2, default=str)

        self.state.checkpoint_path = path
        return path

    def load(self, path: str | Path) -> int:
        """Load a full training checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The epoch number at which training was saved.
        """
        path = Path(path)
        epoch = self.trainer.load_checkpoint(path)

        # Try to load engine state (preferred over trainer epoch)
        state_path = path.with_suffix(".state.json")
        if state_path.exists():
            import json
            with open(state_path) as f:
                data = json.load(f)
            loaded_state = EngineState.from_dict(data)
            self.state.epoch = loaded_state.epoch
            self.state.best_metric = loaded_state.best_metric
            self.state.current_metric = loaded_state.current_metric
            self.state.global_step = loaded_state.global_step
            self.state.training_finished = loaded_state.training_finished
            self.state.checkpoint_path = path
        else:
            self.state.epoch = epoch

        return epoch

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    def set_model(self, model: nn.Module) -> None:
        """Replace the model (e.g. after loading weights).

        Args:
            model: New model instance.
        """
        self.model = model
        self.trainer.model = model
        self.predictor = Predictor(model, device=self.device)

    def set_device(self, device: str) -> None:
        """Change the device.

        Args:
            device: Device string (``"cpu"``, ``"cuda"``, ``"auto"``).
        """
        self.device = device
        self.trainer.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)
        self.model = self.model.to(self.trainer.device)
        self.predictor = Predictor(self.model, device=device)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the engine state for a new training run."""
        self.state.reset()
        self.trainer.current_epoch = 0
        self.trainer._best_val_loss = float("inf")
        self.logger.reset()
        if self.early_stopping is not None:
            self.early_stopping.reset()

    def summary(self) -> dict[str, Any]:
        """Return a summary of the engine configuration and state.

        Returns:
            Dictionary with engine configuration and current state.
        """
        return {
            "device": self.device,
            "model": self.model.__class__.__name__,
            "optimizer": type(self.optimizer).__name__,
            "scheduler": type(self.scheduler).__name__ if self.scheduler else None,
            "loss_fn": type(self.loss_fn).__name__,
            "metrics": list(self.metric_fns.keys()),
            "checkpoint_manager": self.checkpoint_manager is not None,
            "early_stopping": self.early_stopping is not None,
            "state": self.state.to_dict(),
        }