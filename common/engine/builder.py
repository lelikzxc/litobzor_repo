"""Factory that constructs all training components from an ``EngineConfig``.

The ``Builder`` takes a config and provides ``build_*`` methods for each
component (model, optimizer, scheduler, loss, metrics, checkpoint manager,
early stopping, logger, scaler, trainer, predictor, state).

Components are resolved in this order:

1. If the component name is registered in the engine registry, use it.
2. Otherwise, fall back to built-in factories (e.g. ``torch.optim.Adam``).
"""

from __future__ import annotations

from typing import Any

import torch

from common.engine.config import EngineConfig
from common.engine.registry import (
    build_model as _registry_build_model,
    build_optimizer as _registry_build_optimizer,
    build_scheduler as _registry_build_scheduler,
    build_loss as _registry_build_loss,
    is_registered,
)


class Builder:
    """Factory for constructing training components from a config.

    Args:
        config: An ``EngineConfig`` instance with the full experiment config.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    # ── Model ─────────────────────────────────────────────────────────────

    def build_model(self, **extra_kwargs: Any) -> torch.nn.Module:
        """Build the model from config.

        Reads ``model.name`` and ``model.num_classes`` from config.
        Extra kwargs override config values.

        Returns:
            An instantiated ``torch.nn.Module``.
        """
        name: str = extra_kwargs.pop("name", self.config.get("model.name", ""))
        num_classes: int = extra_kwargs.get(
            "num_classes", self.config.get("model.num_classes", 80)
        )

        if is_registered("models", name):
            return _registry_build_model(
                name, num_classes=num_classes, **extra_kwargs
            )

        raise KeyError(
            f"Model {name!r} is not registered. "
            f"Available: {self._list_registered('models')}"
        )

    # ── Optimizer ─────────────────────────────────────────────────────────

    def build_optimizer(
        self, model: torch.nn.Module, **extra_kwargs: Any
    ) -> torch.optim.Optimizer:
        """Build the optimizer from config.

        Reads ``training.optimizer`` from config (dict with ``name``, ``lr``,
        ``weight_decay``, etc.).

        Args:
            model: The model whose parameters will be optimised.
            **extra_kwargs: Override config values.

        Returns:
            A ``torch.optim.Optimizer`` instance.
        """
        opt_cfg = extra_kwargs.pop(
            "optimizer", self.config.get("training.optimizer", {})
        )
        if isinstance(opt_cfg, dict):
            name = opt_cfg.get("name", "adamw")
            lr = float(opt_cfg.get("lr", 1e-3))
            weight_decay = float(opt_cfg.get("weight_decay", 0.0))
            kwargs = {k: v for k, v in opt_cfg.items() if k not in ("name",)}
        else:
            name = str(opt_cfg)
            lr = float(self.config.get("training.learning_rate", 1e-3))
            weight_decay = float(self.config.get("training.weight_decay", 0.0))
            kwargs = {}

        kwargs.update(extra_kwargs)
        kwargs.setdefault("lr", lr)
        kwargs.setdefault("weight_decay", weight_decay)

        if is_registered("optimizers", name):
            return _registry_build_optimizer(name, params=model.parameters(), **kwargs)

        # Built-in fallbacks
        optim_cls = _BUILTIN_OPTIMIZERS.get(name)
        if optim_cls is not None:
            return optim_cls(params=model.parameters(), **kwargs)

        raise KeyError(
            f"Optimizer {name!r} is not registered and has no built-in fallback. "
            f"Built-in: {list(_BUILTIN_OPTIMIZERS.keys())}"
        )

    # ── Scheduler ─────────────────────────────────────────────────────────

    def build_scheduler(
        self, optimizer: torch.optim.Optimizer, **extra_kwargs: Any
    ) -> Any:
        """Build the learning rate scheduler from config.

        Reads ``training.scheduler`` from config.

        Args:
            optimizer: The optimiser to schedule.
            **extra_kwargs: Override config values.

        Returns:
            A scheduler instance.
        """
        sched_cfg = extra_kwargs.pop(
            "scheduler", self.config.get("training.scheduler", {})
        )
        if isinstance(sched_cfg, dict):
            name = sched_cfg.get("name", "cosine")
            kwargs = {k: v for k, v in sched_cfg.items() if k != "name"}
        else:
            name = str(sched_cfg)
            kwargs = {}

        kwargs.update(extra_kwargs)

        if is_registered("schedulers", name):
            return _registry_build_scheduler(name, optimizer=optimizer, **kwargs)

        sched_cls = _BUILTIN_SCHEDULERS.get(name)
        if sched_cls is not None:
            return sched_cls(optimizer=optimizer, **kwargs)

        raise KeyError(
            f"Scheduler {name!r} is not registered and has no built-in fallback. "
            f"Built-in: {list(_BUILTIN_SCHEDULERS.keys())}"
        )

    # ── Loss ──────────────────────────────────────────────────────────────

    def build_loss(self, **extra_kwargs: Any) -> Any:
        """Build the loss function from config.

        Reads ``training.loss`` from config.

        Args:
            **extra_kwargs: Override config values.

        Returns:
            A loss module or function.
        """
        loss_cfg = extra_kwargs.pop("loss", self.config.get("training.loss", {}))
        if isinstance(loss_cfg, dict):
            name = loss_cfg.get("name", "cross_entropy")
            kwargs = {k: v for k, v in loss_cfg.items() if k != "name"}
        else:
            name = str(loss_cfg)
            kwargs = {}

        kwargs.update(extra_kwargs)

        if is_registered("losses", name):
            return _registry_build_loss(name, **kwargs)

        loss_cls = _BUILTIN_LOSSES.get(name)
        if loss_cls is not None:
            return loss_cls(**kwargs)

        raise KeyError(
            f"Loss {name!r} is not registered and has no built-in fallback. "
            f"Built-in: {list(_BUILTIN_LOSSES.keys())}"
        )

    # ── Metrics ───────────────────────────────────────────────────────────

    def build_metrics(self) -> list[dict[str, Any]]:
        """Build the list of metric configs from config.

        Reads ``evaluation.metrics`` from config.

        Returns:
            List of metric descriptor dicts (e.g. ``[{"name": "map"}]``).
        """
        metrics: list[str] | list[dict[str, Any]] = self.config.get(
            "evaluation.metrics", []
        )
        result: list[dict[str, Any]] = []
        for m in metrics:
            if isinstance(m, str):
                result.append({"name": m})
            else:
                result.append(m)
        return result

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _list_registered(category: str) -> list[str]:
        from common.engine.registry import list_registered

        return list_registered(category)


# ── Built-in mappings ─────────────────────────────────────────────────────

_BUILTIN_OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

_BUILTIN_SCHEDULERS: dict[str, Any] = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cosineannealinglr": torch.optim.lr_scheduler.CosineAnnealingLR,
    "step": torch.optim.lr_scheduler.StepLR,
    "steplr": torch.optim.lr_scheduler.StepLR,
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
    "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "reduceplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}

_BUILTIN_LOSSES: dict[str, Any] = {
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "bce": torch.nn.BCEWithLogitsLoss,
    "mse": torch.nn.MSELoss,
    "l1": torch.nn.L1Loss,
}