"""High-level Engine for model training, evaluation, and inference.

The ``Engine`` ties together the ``Builder``, model, optimiser, scheduler,
loss, metrics, checkpointing, and logging into a single ``fit`` / ``validate``
/ ``test`` / ``predict`` API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from common.engine.builder import Builder
from common.engine.config import EngineConfig
from common.engine.registry import build_model as _registry_build_model


class Engine:
    """High-level training and evaluation engine.

    Args:
        model: A ``torch.nn.Module`` instance or a registered model name.
        config: An ``EngineConfig`` with the full experiment configuration.
        device: Device string (e.g. ``"cpu"``, ``"cuda:0"``). If ``None``,
            auto-detected.
    """

    def __init__(
        self,
        model: torch.nn.Module | str,
        config: EngineConfig,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.builder = Builder(config)

        # Resolve model
        if isinstance(model, str):
            self.model = _registry_build_model(
                model,
                num_classes=config.get("model.num_classes", 80),
            )
        else:
            self.model = model

        # Resolve device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # Components (built lazily)
        self.optimizer: Any = None
        self.scheduler: Any = None
        self.loss_fn: Any = None
        self.metrics: list[dict[str, Any]] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def build_all(self) -> None:
        """Build all training components from config."""
        self.optimizer = self.builder.build_optimizer(self.model)
        if self.optimizer is not None:
            self.scheduler = self.builder.build_scheduler(self.optimizer)
        self.loss_fn = self.builder.build_loss()
        self.metrics = self.builder.build_metrics()

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, train_loader: Any, val_loader: Any | None = None) -> dict[str, Any]:
        """Run a full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.

        Returns:
            Training history dict.
        """
        self.model.train()
        history: dict[str, list[float]] = {"train_loss": []}

        num_epochs = self.config.get("training.num_epochs", 10)

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                loss = self._train_step(batch)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_loss)

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                for key, value in val_metrics.items():
                    history.setdefault(key, []).append(value)

            if self.scheduler is not None:
                self.scheduler.step()

        return history

    def _train_step(self, batch: Any) -> float:
        """Perform a single training step.

        Args:
            batch: A batch from the DataLoader.

        Returns:
            Loss value as a float.
        """
        self.optimizer.zero_grad()

        # Unpack batch
        if isinstance(batch, (list, tuple)):
            x, y = batch
        elif isinstance(batch, dict):
            x = batch.get("image", batch.get("x"))
            y = batch.get("label", batch.get("y"))
        else:
            x, y = batch, None

        x = x.to(self.device) if isinstance(x, torch.Tensor) else x
        y = y.to(self.device) if isinstance(y, torch.Tensor) else y

        output = self.model(x)
        loss = self.loss_fn(output, y) if self.loss_fn is not None else 0.0

        if isinstance(loss, torch.Tensor):
            loss.backward()
            self.optimizer.step()
            return float(loss.item())

        return 0.0

    # ── Validation / Evaluation ───────────────────────────────────────────

    @torch.no_grad()
    def validate(self, val_loader: Any) -> dict[str, float]:
        """Run validation.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Dict of metric name → value.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            elif isinstance(batch, dict):
                x = batch.get("image", batch.get("x"))
                y = batch.get("label", batch.get("y"))
            else:
                x, y = batch, None

            x = x.to(self.device) if isinstance(x, torch.Tensor) else x
            y = y.to(self.device) if isinstance(y, torch.Tensor) else y

            output = self.model(x)
            loss = self.loss_fn(output, y) if self.loss_fn is not None else 0.0
            total_loss += float(loss) if isinstance(loss, torch.Tensor) else loss
            num_batches += 1

        self.model.train()
        return {"val_loss": total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def test(self, test_loader: Any) -> dict[str, float]:
        """Run test evaluation.

        Args:
            test_loader: DataLoader for test data.

        Returns:
            Dict of metric name → value.
        """
        return self.validate(test_loader)

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Any:
        """Run inference on a single tensor or batch.

        Args:
            x: Input tensor.

        Returns:
            Model output.
        """
        self.model.eval()
        x = x.to(self.device)
        return self.model(x)

    def predict_single(self, x: torch.Tensor) -> Any:
        """Run inference on a single sample (adds batch dim if needed).

        Args:
            x: Input tensor of shape ``[C, H, W]`` or ``[B, C, H, W]``.

        Returns:
            Model output.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.predict(x)

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model state dict to disk.

        Args:
            path: Path to save the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load model state dict from disk.

        Args:
            path: Path to the checkpoint file.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    # ── Utilities ─────────────────────────────────────────────────────────

    def set_model(self, model: torch.nn.Module) -> None:
        """Replace the model (e.g. after loading weights)."""
        self.model = model
        self.model.to(self.device)

    def set_device(self, device: str) -> None:
        """Move model to a different device."""
        self.device = torch.device(device)
        self.model.to(self.device)

    def reset(self) -> None:
        """Reset optimiser, scheduler, and loss to initial state."""
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.metrics = []

    def summary(self) -> dict[str, Any]:
        """Return a summary of the engine state.

        Returns:
            Dict with model name, parameter count, device, etc.
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return {
            "model_name": type(self.model).__name__,
            "total_params": num_params,
            "trainable_params": num_trainable,
            "device": str(self.device),
            "optimizer": type(self.optimizer).__name__ if self.optimizer else None,
            "scheduler": type(self.scheduler).__name__ if self.scheduler else None,
            "loss_fn": type(self.loss_fn).__name__ if self.loss_fn else None,
        }