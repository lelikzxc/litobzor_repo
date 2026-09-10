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
            ``"cosine"``, ``"cosine_warmup"``, ``"step"``, ``"plateau"``,
            ``"onecycle"``.
        **kwargs: Arguments forwarded to the scheduler constructor.
            Common keys:
            - ``T_max`` (cosine): Maximum number of iterations.
            - ``eta_min`` (cosine): Minimum learning rate.
            - ``warmup_epochs`` (cosine_warmup): Linear warmup length.
            - ``start_factor`` (cosine_warmup): LR multiplier at warmup start.
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

    if key == "cosine_warmup":
        t_max = int(kwargs.pop("T_max"))
        warmup_epochs = int(kwargs.pop("warmup_epochs", 5))
        eta_min = float(kwargs.pop("eta_min", 0.0))
        start_factor = float(kwargs.pop("start_factor", 0.01))
        warmup_epochs = max(0, min(warmup_epochs, t_max - 1))
        if warmup_epochs <= 0:
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min, **kwargs
            )
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max - warmup_epochs),
            eta_min=eta_min,
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    if key == "step":
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)

    if key == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    if key == "onecycle":
        return optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)

    raise ValueError(
        "Unknown scheduler: "
        f"'{name}'. Allowed: cosine, cosine_warmup, step, plateau, onecycle"
    )
