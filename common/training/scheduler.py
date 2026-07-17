"""Scheduler factory supporting CosineAnnealingLR, StepLR, ReduceLROnPlateau, and OneCycleLR."""

from __future__ import annotations

from typing import Any

from torch import optim


def build_scheduler(
    optimizer: optim.Optimizer,
    name: str = "cosine",
    **kwargs: Any,
) -> optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau:
    """Build a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        name: Scheduler name (case-insensitive). One of:
            ``"cosine"``, ``"step"``, ``"plateau"``, ``"onecycle"``.
        **kwargs: Arguments forwarded to the scheduler constructor.
            Common keys:
            - ``T_max`` (cosine): Maximum number of iterations.
            - ``step_size`` (step): Period of learning rate decay.
            - ``gamma`` (step, plateau): Multiplicative factor.
            - ``patience`` (plateau): Number of epochs with no improvement.
            - ``max_lr`` (onecycle): Upper learning rate boundary.

    Returns:
        A ``torch.optim.lr_scheduler`` instance.

    Raises:
        ValueError: If ``name`` is not supported.
    """
    key = name.lower().replace("-", "_")

    if key == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

    if key == "step":
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)

    if key == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    if key == "onecycle":
        return optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)

    raise ValueError(
        f"Unknown scheduler: '{name}'. Allowed: cosine, step, plateau, onecycle"
    )