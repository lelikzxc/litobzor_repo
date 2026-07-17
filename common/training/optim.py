"""Optimizer factory supporting Adam, AdamW, and SGD."""

from __future__ import annotations

from typing import Any

from torch import nn, optim


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> optim.Optimizer:
    """Build an optimizer for a model's parameters.

    Args:
        model: The model whose parameters will be optimised.
        name: Optimizer name (case-insensitive). One of:
            ``"adam"``, ``"adamw"``, ``"sgd"``.
        lr: Learning rate.
        weight_decay: Weight decay (L2 penalty).
        **kwargs: Additional arguments forwarded to the optimizer constructor
            (e.g. ``momentum`` for SGD, ``betas`` for Adam/AdamW).

    Returns:
        A ``torch.optim.Optimizer`` instance.

    Raises:
        ValueError: If ``name`` is not supported.
    """
    key = name.lower().replace("-", "_")

    if key == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )
    if key == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )
    if key == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )

    raise ValueError(
        f"Unknown optimizer: '{name}'. Allowed: adam, adamw, sgd"
    )