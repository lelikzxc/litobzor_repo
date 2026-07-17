"""Lightweight registry for models, datasets, optimizers, schedulers, losses, and metrics.

Allows future paper modules to register themselves without modifying the engine.
"""

from __future__ import annotations

from typing import Any, Callable

from torch import nn, optim

from common.training.losses import build_loss as _build_loss
from common.training.metrics import build_metric as _build_metric
from common.training.optim import build_optimizer as _build_optimizer
from common.training.scheduler import build_scheduler as _build_scheduler

# ---------------------------------------------------------------------------
# Registry storage
# ---------------------------------------------------------------------------

_registries: dict[str, dict[str, Any]] = {
    "models": {},
    "datasets": {},
    "optimizers": {
        "adam": "adam",
        "adamw": "adamw",
        "sgd": "sgd",
    },
    "schedulers": {
        "cosine": "cosine",
        "step": "step",
        "plateau": "plateau",
        "onecycle": "onecycle",
    },
    "losses": {
        "cross_entropy": "cross_entropy",
        "bce": "bce",
        "focal": "focal",
        "dice": "dice",
        "iou": "iou",
        "bce_dice": "bce_dice",
    },
    "metrics": {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "iou": "iou",
        "dice": "dice",
        "pixel_accuracy": "pixel_accuracy",
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_model(name: str, model_cls: type[nn.Module]) -> None:
    """Register a model class by name.

    Args:
        name: Unique model name.
        model_cls: The model class (must be an ``nn.Module`` subclass).

    Raises:
        ValueError: If the name is already registered.
    """
    if name in _registries["models"]:
        raise ValueError(f"Model '{name}' is already registered.")
    _registries["models"][name] = model_cls


def register_dataset(name: str, dataset_cls: type) -> None:
    """Register a dataset class by name.

    Args:
        name: Unique dataset name.
        dataset_cls: The dataset class.

    Raises:
        ValueError: If the name is already registered.
    """
    if name in _registries["datasets"]:
        raise ValueError(f"Dataset '{name}' is already registered.")
    _registries["datasets"][name] = dataset_cls


def register_optimizer(name: str, optimizer_cls: type[optim.Optimizer]) -> None:
    """Register an optimizer class by name.

    Args:
        name: Optimizer name.
        optimizer_cls: The optimizer class.

    Raises:
        ValueError: If the name is already registered.
    """
    if name in _registries["optimizers"]:
        raise ValueError(f"Optimizer '{name}' is already registered.")
    _registries["optimizers"][name] = optimizer_cls


def register_scheduler(name: str, scheduler_cls: type) -> None:
    """Register a scheduler class by name.

    Args:
        name: Scheduler name.
        scheduler_cls: The scheduler class.

    Raises:
        ValueError: If the name is already registered.
    """
    if name in _registries["schedulers"]:
        raise ValueError(f"Scheduler '{name}' is already registered.")
    _registries["schedulers"][name] = scheduler_cls


def register_loss(name: str, loss_cls: type[nn.Module]) -> None:
    """Register a loss class by name.

    Args:
        name: Loss name.
        loss_cls: The loss class (must be an ``nn.Module`` subclass).

    Raises:
        ValueError: If the name is already registered.
    """
    if name in _registries["losses"]:
        raise ValueError(f"Loss '{name}' is already registered.")
    _registries["losses"][name] = loss_cls


def register_metric(name: str, metric_fn: Callable) -> None:
    """Register a metric function by name.

    Args:
        name: Metric name.
        metric_fn: The metric callable.

    Raises:
        ValueError: If the name is already registered.
    """
    if name in _registries["metrics"]:
        raise ValueError(f"Metric '{name}' is already registered.")
    _registries["metrics"][name] = metric_fn


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Build a model by name.

    Args:
        name: Registered model name.
        **kwargs: Arguments forwarded to the model constructor.

    Returns:
        An ``nn.Module`` instance.

    Raises:
        ValueError: If the model name is not registered.
    """
    if name not in _registries["models"]:
        available = ", ".join(sorted(_registries["models"]))
        raise ValueError(f"Unknown model: '{name}'. Registered: {available}")
    return _registries["models"][name](**kwargs)


def build_dataset(name: str, **kwargs: Any) -> Any:
    """Build a dataset by name.

    Args:
        name: Registered dataset name.
        **kwargs: Arguments forwarded to the dataset constructor.

    Returns:
        A dataset instance.

    Raises:
        ValueError: If the dataset name is not registered.
    """
    if name not in _registries["datasets"]:
        available = ", ".join(sorted(_registries["datasets"]))
        raise ValueError(f"Unknown dataset: '{name}'. Registered: {available}")
    return _registries["datasets"][name](**kwargs)


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> optim.Optimizer:
    """Build an optimizer by name.

    Delegates to ``common.training.optim.build_optimizer`` for built-in
    optimizers. Custom registered optimizers are constructed directly.

    Args:
        model: The model whose parameters will be optimised.
        name: Optimizer name.
        lr: Learning rate.
        weight_decay: Weight decay.
        **kwargs: Additional arguments.

    Returns:
        A ``torch.optim.Optimizer`` instance.
    """
    # Built-in optimizers use the factory
    if name.lower() in ("adam", "adamw", "sgd"):
        return _build_optimizer(model, name=name, lr=lr, weight_decay=weight_decay, **kwargs)
    # Custom registered optimizers
    if name in _registries["optimizers"]:
        cls = _registries["optimizers"][name]
        if isinstance(cls, str):
            return _build_optimizer(model, name=name, lr=lr, weight_decay=weight_decay, **kwargs)
        return cls(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    available = ", ".join(sorted(_registries["optimizers"]))
    raise ValueError(f"Unknown optimizer: '{name}'. Registered: {available}")


def build_scheduler(
    optimizer: optim.Optimizer,
    name: str = "cosine",
    **kwargs: Any,
) -> optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau:
    """Build a scheduler by name.

    Delegates to ``common.training.scheduler.build_scheduler`` for built-in
    schedulers. Custom registered schedulers are constructed directly.

    Args:
        optimizer: The optimizer to schedule.
        name: Scheduler name.
        **kwargs: Additional arguments.

    Returns:
        A scheduler instance.
    """
    if name.lower() in ("cosine", "step", "plateau", "onecycle"):
        return _build_scheduler(optimizer, name=name, **kwargs)
    if name in _registries["schedulers"]:
        cls = _registries["schedulers"][name]
        if isinstance(cls, str):
            return _build_scheduler(optimizer, name=name, **kwargs)
        return cls(optimizer, **kwargs)
    available = ", ".join(sorted(_registries["schedulers"]))
    raise ValueError(f"Unknown scheduler: '{name}'. Registered: {available}")


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    """Build a loss by name.

    Delegates to ``common.training.losses.build_loss`` for built-in losses.
    Custom registered losses are constructed directly.

    Args:
        name: Loss name.
        **kwargs: Arguments forwarded to the loss constructor.

    Returns:
        An ``nn.Module`` loss instance.
    """
    key = name.lower().replace("-", "_")
    if key in ("cross_entropy", "bce", "focal", "dice", "iou", "bce_dice"):
        return _build_loss(name=name, **kwargs)
    if name in _registries["losses"]:
        cls = _registries["losses"][name]
        if isinstance(cls, str):
            return _build_loss(name=name, **kwargs)
        return cls(**kwargs)
    available = ", ".join(sorted(_registries["losses"]))
    raise ValueError(f"Unknown loss: '{name}'. Registered: {available}")


def build_metric(name: str) -> Callable:
    """Build a metric function by name.

    Delegates to ``common.training.metrics.build_metric`` for built-in metrics.

    Args:
        name: Metric name.

    Returns:
        A metric callable.
    """
    key = name.lower().replace("-", "_")
    if key in ("accuracy", "precision", "recall", "f1", "iou", "dice", "pixel_accuracy"):
        return _build_metric(name=name)
    if name in _registries["metrics"]:
        fn = _registries["metrics"][name]
        if isinstance(fn, str):
            return _build_metric(name=name)
        return fn
    available = ", ".join(sorted(_registries["metrics"]))
    raise ValueError(f"Unknown metric: '{name}'. Registered: {available}")


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def list_registered(category: str) -> list[str]:
    """List all registered names in a category.

    Args:
        category: One of ``"models"``, ``"datasets"``, ``"optimizers"``,
            ``"schedulers"``, ``"losses"``, ``"metrics"``.

    Returns:
        Sorted list of registered names.

    Raises:
        ValueError: If the category is unknown.
    """
    if category not in _registries:
        cats = ", ".join(sorted(_registries))
        raise ValueError(f"Unknown category: '{category}'. Available: {cats}")
    return sorted(_registries[category])


def is_registered(category: str, name: str) -> bool:
    """Check if a name is registered in a category.

    Args:
        category: Registry category.
        name: Name to check.

    Returns:
        ``True`` if registered.
    """
    return name in _registries.get(category, {})