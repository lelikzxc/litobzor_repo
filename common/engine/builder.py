"""Builder utilities for constructing engine components.

Supports creation from model instance, config dictionary, or YAML configuration.
Automatically constructs all required common components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from common.engine.config import EngineConfig
from common.engine.registry import (
    build_loss as _build_loss,
    build_metric as _build_metric,
    build_model as _build_model,
    build_optimizer as _build_optimizer,
    build_scheduler as _build_scheduler,
)
from common.engine.state import EngineState
from common.inference import Predictor
from common.training import (
    CheckpointManager,
    EarlyStopping,
    NativeScaler,
    Trainer,
    TrainingLogger,
)


class Builder:
    """Factory for constructing engine components from configuration.

    Supports three input modes:
    - From a model instance + config dict
    - From a config dictionary (model name must be registered)
    - From a YAML file path
    """

    def __init__(self, config: EngineConfig | dict[str, Any] | str | Path | None = None) -> None:
        """Initialize the builder.

        Args:
            config: Optional configuration. Can be an ``EngineConfig``,
                a dictionary, or a path to a YAML file.
        """
        self.config: EngineConfig | None = None
        if config is not None:
            self._load_config(config)

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    def _load_config(self, config: EngineConfig | dict[str, Any] | str | Path) -> None:
        """Load configuration from various formats."""
        if isinstance(config, EngineConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = EngineConfig.from_dict(config)
        elif isinstance(config, (str, Path)):
            self.config = EngineConfig.from_yaml(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

    def with_config(self, config: EngineConfig | dict[str, Any] | str | Path) -> Builder:
        """Set or update the builder configuration.

        Args:
            config: Configuration source.

        Returns:
            ``self`` for chaining.
        """
        self._load_config(config)
        return self

    # ------------------------------------------------------------------
    # Component builders
    # ------------------------------------------------------------------

    def build_model(self, model: nn.Module | str | None = None, **kwargs: Any) -> nn.Module:
        """Build or return a model.

        Args:
            model: An ``nn.Module`` instance, a registered model name, or
                ``None`` to use the config's ``model.name``.
            **kwargs: Overrides for model constructor arguments.

        Returns:
            An ``nn.Module`` instance.
        """
        if isinstance(model, nn.Module):
            return model
        if isinstance(model, str):
            return _build_model(model, **kwargs)

        # Try config
        if self.config is not None:
            model_name = self.config.get("model.name")
            if model_name is not None:
                model_kwargs = dict(self.config.get("model.kwargs", {}))
                model_kwargs.update(kwargs)
                return _build_model(model_name, **model_kwargs)

        raise ValueError(
            "No model provided. Pass an nn.Module, a registered name, "
            "or set 'model.name' in the config."
        )

    def build_optimizer(
        self,
        model: nn.Module,
        name: str | None = None,
        lr: float | None = None,
        **kwargs: Any,
    ) -> optim.Optimizer:
        """Build an optimizer for the given model.

        Args:
            model: The model whose parameters will be optimised.
            name: Optimizer name. Uses config's ``optimizer.name`` if ``None``.
            lr: Learning rate. Uses config's ``optimizer.lr`` if ``None``.
            **kwargs: Additional optimizer arguments.

        Returns:
            A ``torch.optim.Optimizer`` instance.
        """
        if name is None and self.config is not None:
            name = self.config.get("optimizer.name", "adamw")
        if lr is None and self.config is not None:
            lr = self.config.get("optimizer.lr", 1e-3)

        opt_kwargs: dict[str, Any] = {}
        if self.config is not None:
            opt_kwargs = dict(self.config.get("optimizer.kwargs", {}))
        opt_kwargs.update(kwargs)

        return _build_optimizer(model, name=name or "adamw", lr=lr or 1e-3, **opt_kwargs)

    def build_scheduler(
        self,
        optimizer: optim.Optimizer,
        name: str | None = None,
        **kwargs: Any,
    ) -> optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau | None:
        """Build a learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule.
            name: Scheduler name. Uses config's ``scheduler.name`` if ``None``.
                Pass ``None`` and no config to skip scheduler creation.
            **kwargs: Additional scheduler arguments.

        Returns:
            A scheduler instance, or ``None`` if no scheduler is configured.
        """
        if name is None and self.config is not None:
            name = self.config.get("scheduler.name")
        if name is None:
            return None

        sched_kwargs: dict[str, Any] = {}
        if self.config is not None:
            sched_kwargs = dict(self.config.get("scheduler.kwargs", {}))
        sched_kwargs.update(kwargs)

        return _build_scheduler(optimizer, name=name, **sched_kwargs)

    def build_loss(self, name: str | None = None, **kwargs: Any) -> nn.Module:
        """Build a loss function.

        Args:
            name: Loss name. Uses config's ``loss.name`` if ``None``.
            **kwargs: Additional loss arguments.

        Returns:
            An ``nn.Module`` loss instance.
        """
        if name is None and self.config is not None:
            name = self.config.get("loss.name", "cross_entropy")

        loss_kwargs: dict[str, Any] = {}
        if self.config is not None:
            loss_kwargs = dict(self.config.get("loss.kwargs", {}))
        loss_kwargs.update(kwargs)

        return _build_loss(name=name or "cross_entropy", **loss_kwargs)

    def build_metrics(
        self,
        names: list[str] | None = None,
    ) -> dict[str, Callable]:
        """Build metric functions.

        Args:
            names: List of metric names. Uses config's ``metrics`` list if ``None``.

        Returns:
            Dictionary mapping metric names to callables.
        """
        if names is None and self.config is not None:
            names = self.config.get("metrics", [])
        if not names:
            return {}
        return {name: _build_metric(name) for name in names}

    def build_checkpoint_manager(
        self,
        save_dir: str | Path | None = None,
        metric_name: str | None = None,
        mode: str | None = None,
    ) -> CheckpointManager | None:
        """Build a checkpoint manager.

        Args:
            save_dir: Directory for checkpoints. Uses config's
                ``checkpoint.save_dir`` if ``None``. If still ``None``,
                no manager is created.
            metric_name: Metric name for best-checkpoint tracking.
            mode: ``"min"`` or ``"max"``.

        Returns:
            A ``CheckpointManager`` instance, or ``None``.
        """
        if save_dir is None and self.config is not None:
            save_dir = self.config.get("checkpoint.save_dir")
        if save_dir is None:
            return None

        if metric_name is None and self.config is not None:
            metric_name = self.config.get("checkpoint.metric_name", "val_loss")
        if mode is None and self.config is not None:
            mode = self.config.get("checkpoint.mode", "min")

        return CheckpointManager(
            save_dir=save_dir,
            metric_name=metric_name or "val_loss",
            mode=mode or "min",
        )

    def build_early_stopping(
        self,
        patience: int | None = None,
        mode: str | None = None,
    ) -> EarlyStopping | None:
        """Build an early stopping handler.

        Args:
            patience: Patience in epochs. Uses config's
                ``early_stopping.patience`` if ``None``. If still ``None``,
                no early stopping is created.
            mode: ``"min"`` or ``"max"``.

        Returns:
            An ``EarlyStopping`` instance, or ``None``.
        """
        if patience is None and self.config is not None:
            patience = self.config.get("early_stopping.patience")
        if patience is None:
            return None

        if mode is None and self.config is not None:
            mode = self.config.get("early_stopping.mode", "min")
        min_delta = self.config.get("early_stopping.min_delta", 0.0) if self.config else 0.0
        restore_best_weights = (
            self.config.get("early_stopping.restore_best_weights", True)
            if self.config
            else True
        )

        return EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            mode=mode or "min",
            restore_best_weights=restore_best_weights,
        )

    def build_logger(self) -> TrainingLogger:
        """Build a training logger.

        Returns:
            A new ``TrainingLogger`` instance.
        """
        return TrainingLogger()

    def build_scaler(self, enabled: bool | None = None) -> NativeScaler:
        """Build a mixed-precision scaler.

        Args:
            enabled: Whether to enable mixed precision. Uses config's
                ``training.amp`` if ``None``.

        Returns:
            A ``NativeScaler`` instance.
        """
        if enabled is None and self.config is not None:
            enabled = self.config.get("training.amp", True)
        return NativeScaler(enabled=enabled if enabled is not None else True)

    def build_trainer(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: Any = None,
        device: str = "auto",
        metric_fns: dict[str, Callable] | None = None,
        early_stopping: EarlyStopping | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        logger: TrainingLogger | None = None,
        scaler: NativeScaler | None = None,
        **kwargs: Any,
    ) -> Trainer:
        """Build a Trainer with all components.

        Args:
            model: The model to train.
            optimizer: The optimizer.
            loss_fn: The loss function.
            scheduler: Optional scheduler.
            device: Device string.
            metric_fns: Optional metric functions.
            early_stopping: Optional early stopping handler.
            checkpoint_manager: Optional checkpoint manager.
            logger: Optional logger.
            scaler: Optional mixed-precision scaler.
            **kwargs: Additional Trainer arguments (``grad_max_norm``,
                ``grad_max_value``, ``verbose``).

        Returns:
            A configured ``Trainer`` instance.
        """
        if logger is None:
            logger = self.build_logger()
        if scaler is None:
            scaler = self.build_scaler()

        # Resolve device from config if not specified
        if device == "auto" and self.config is not None:
            device = self.config.get("training.device", "auto")

        return Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=device,
            metric_fns=metric_fns,
            early_stopping=early_stopping,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            scaler=scaler,
            **kwargs,
        )

    def build_predictor(
        self,
        model: nn.Module,
        device: str = "auto",
    ) -> Predictor:
        """Build a Predictor for inference.

        Args:
            model: The model to use for inference.
            device: Device string.

        Returns:
            A configured ``Predictor`` instance.
        """
        if device == "auto" and self.config is not None:
            device = self.config.get("inference.device", "auto")
        return Predictor(model=model, device=device)

    def build_state(self) -> EngineState:
        """Build an EngineState from configuration.

        Returns:
            An ``EngineState`` with defaults from config.
        """
        state = EngineState()
        if self.config is not None:
            state.best_metric_name = self.config.get(
                "checkpoint.metric_name", state.best_metric_name
            )
            state.mode = self.config.get("checkpoint.mode", state.mode)
        return state

    # ------------------------------------------------------------------
    # All-in-one
    # ------------------------------------------------------------------

    def build_all(
        self,
        model: nn.Module | str | None = None,
        device: str = "auto",
    ) -> dict[str, Any]:
        """Build all standard engine components from configuration.

        This is a convenience method that constructs model, optimizer,
        scheduler, loss, metrics, checkpoint manager, early stopping,
        logger, scaler, trainer, predictor, and state in one call.

        Args:
            model: An ``nn.Module`` instance or registered model name.
            device: Device string.

        Returns:
            Dictionary with keys: ``"model"``, ``"optimizer"``,
            ``"scheduler"``, ``"loss_fn"``, ``"metric_fns"``,
            ``"checkpoint_manager"``, ``"early_stopping"``, ``"logger"``,
            ``"scaler"``, ``"trainer"``, ``"predictor"``, ``"state"``.
        """
        built_model = self.build_model(model)
        optimizer = self.build_optimizer(built_model)
        scheduler = self.build_scheduler(optimizer)
        loss_fn = self.build_loss()
        metric_fns = self.build_metrics()
        checkpoint_manager = self.build_checkpoint_manager()
        early_stopping = self.build_early_stopping()
        logger = self.build_logger()
        scaler = self.build_scaler()
        trainer = self.build_trainer(
            model=built_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=device,
            metric_fns=metric_fns,
            early_stopping=early_stopping,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            scaler=scaler,
        )
        predictor = self.build_predictor(built_model, device=device)
        state = self.build_state()

        return {
            "model": built_model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "loss_fn": loss_fn,
            "metric_fns": metric_fns,
            "checkpoint_manager": checkpoint_manager,
            "early_stopping": early_stopping,
            "logger": logger,
            "scaler": scaler,
            "trainer": trainer,
            "predictor": predictor,
            "state": state,
        }