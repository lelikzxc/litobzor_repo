"""Lightweight registry for models, datasets, optimizers, schedulers, losses, and metrics.

Allows paper modules to register their components so the common ``Builder``
and ``Engine`` can instantiate them by name without manual imports.

Typical usage::

    from common.engine.registry import register_model, build_model

    # In a paper's __init__.py:
    register_model("ctm_yolov10", CTMYOLOv10)

    # Anywhere in the codebase:
    model = build_model("ctm_yolov10", num_classes=8, ...)
"""

from __future__ import annotations

from typing import Any, Callable

# ── Internal storage ──────────────────────────────────────────────────────

_registries: dict[str, dict[str, Any]] = {
    "models": {},
    "datasets": {},
    "optimizers": {},
    "schedulers": {},
    "losses": {},
    "metrics": {},
}


# ── Registration helpers ──────────────────────────────────────────────────


def register_model(name: str, model_cls: type) -> None:
    """Register a model class under *name*.

    Args:
        name: Unique identifier (e.g. ``"ctm_yolov10"``).
        model_cls: A ``torch.nn.Module`` subclass.
    """
    _registries["models"][name] = model_cls


def register_dataset(name: str, dataset_cls: type) -> None:
    """Register a dataset class under *name*."""
    _registries["datasets"][name] = dataset_cls


def register_optimizer(name: str, optimizer_cls: type) -> None:
    """Register an optimizer class under *name*."""
    _registries["optimizers"][name] = optimizer_cls


def register_scheduler(name: str, scheduler_cls: type) -> None:
    """Register a scheduler class under *name*."""
    _registries["schedulers"][name] = scheduler_cls


def register_loss(name: str, loss_cls: type) -> None:
    """Register a loss class under *name*."""
    _registries["losses"][name] = loss_cls


def register_metric(name: str, metric_fn: Callable) -> None:
    """Register a metric function under *name*."""
    _registries["metrics"][name] = metric_fn


# ── Build helpers ─────────────────────────────────────────────────────────


def build_model(name: str, **kwargs: Any) -> Any:
    """Build a model instance by registered name.

    Args:
        name: Registered model name.
        **kwargs: Keyword arguments forwarded to the model constructor.

    Returns:
        An instantiated model.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in _registries["models"]:
        registered = list(_registries["models"].keys())
        raise KeyError(
            f"Model {name!r} is not registered. "
            f"Registered models: {registered}"
        )
    return _registries["models"][name](**kwargs)


def build_dataset(name: str, **kwargs: Any) -> Any:
    """Build a dataset instance by registered name."""
    if name not in _registries["datasets"]:
        raise KeyError(f"Dataset {name!r} is not registered.")
    return _registries["datasets"][name](**kwargs)


def build_optimizer(name: str, **kwargs: Any) -> Any:
    """Build an optimizer instance by registered name."""
    if name not in _registries["optimizers"]:
        raise KeyError(f"Optimizer {name!r} is not registered.")
    return _registries["optimizers"][name](**kwargs)


def build_scheduler(name: str, **kwargs: Any) -> Any:
    """Build a scheduler instance by registered name."""
    if name not in _registries["schedulers"]:
        raise KeyError(f"Scheduler {name!r} is not registered.")
    return _registries["schedulers"][name](**kwargs)


def build_loss(name: str, **kwargs: Any) -> Any:
    """Build a loss instance by registered name."""
    if name not in _registries["losses"]:
        raise KeyError(f"Loss {name!r} is not registered.")
    return _registries["losses"][name](**kwargs)


# ── Query helpers ─────────────────────────────────────────────────────────


def is_registered(category: str, name: str) -> bool:
    """Check whether *name* is registered in *category*.

    Args:
        category: One of ``"models"``, ``"datasets"``, ``"optimizers"``,
            ``"schedulers"``, ``"losses"``, ``"metrics"``.
        name: Registered name to check.

    Returns:
        ``True`` if the name is registered.
    """
    return name in _registries.get(category, {})


def list_registered(category: str) -> list[str]:
    """List all registered names in *category*.

    Args:
        category: One of the supported registry categories.

    Returns:
        Sorted list of registered names.
    """
    return sorted(_registries.get(category, {}).keys())