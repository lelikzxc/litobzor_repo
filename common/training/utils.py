"""Helper utilities for training loops.

Provides:
- ``move_batch_to_device``: transfer tensors/dicts/lists to a device
- ``clip_gradients``: gradient norm/value clipping
- ``NativeScaler``: mixed precision (AMP) helper
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler


def move_batch_to_device(
    batch: Any,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:
    """Recursively move a batch of data to the specified device.

    Supports ``torch.Tensor``, ``dict``, ``list``, ``tuple``, and nested
    combinations thereof.

    Args:
        batch: A single tensor, dict of tensors, or nested structure.
        device: Target device.
        non_blocking: Whether to use asynchronous transfer (CPU → GPU).

    Returns:
        The batch with all tensors moved to ``device``.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)

    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device, non_blocking)
            for key, value in batch.items()
        }

    if isinstance(batch, (list, tuple)):
        return type(batch)(
            move_batch_to_device(item, device, non_blocking) for item in batch
        )

    return batch


def clip_gradients(
    model: nn.Module,
    max_norm: float | None = None,
    max_value: float | None = None,
) -> float:
    """Clip gradients of a model's parameters.

    Args:
        model: The model whose gradients will be clipped.
        max_norm: Maximum gradient norm (``clip_grad_norm_``). ``None`` disables.
        max_value: Maximum gradient value (``clip_grad_value_``). ``None`` disables.

    Returns:
        The total gradient norm before clipping (or ``0.0`` if no clipping).
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    if max_value is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), max_value)

    return total_norm


class NativeScaler:
    """Mixed precision (AMP) gradient scaler helper.

    Wraps ``torch.cuda.amp.GradScaler`` with a no-op fallback when CUDA
    is not available, so callers do not need conditional logic.

    Usage::

        scaler = NativeScaler()
        for batch in loader:
            with scaler.autocast():
                loss = model(batch)
            scaler.backward(loss, optimizer)
            scaler.step(optimizer)
            scaler.update()
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and torch.cuda.is_available()
        self._scaler = GradScaler(enabled=self.enabled) if self.enabled else None

    def autocast(self) -> torch.amp.autocast:
        """Return an autocast context manager.

        Returns:
            ``torch.amp.autocast`` context when CUDA is available and enabled,
            otherwise a no-op context.
        """
        if self.enabled:
            return torch.amp.autocast(device_type="cuda")
        return torch.amp.autocast(device_type="cpu", enabled=False)

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        retain_graph: bool = False,
    ) -> None:
        """Scale loss and call backward.

        Args:
            loss: The loss tensor.
            optimizer: The optimizer (needed for unscaling).
            retain_graph: Passed to ``backward()``.
        """
        if self.enabled:
            self._scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients and step the optimizer.

        Args:
            optimizer: The optimizer to step.
        """
        if self.enabled:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Update the scale factor."""
        if self.enabled:
            self._scaler.update()

    def state_dict(self) -> dict[str, Any]:
        """Return the scaler's state dict (empty if disabled)."""
        if self.enabled:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scaler's state dict."""
        if self.enabled:
            self._scaler.load_state_dict(state_dict)