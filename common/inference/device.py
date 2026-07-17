"""Device utilities for inference.

Provides helpers for device selection, model sizing, and memory estimation.
"""

from __future__ import annotations

import torch
from torch import nn


def get_best_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU).

    Returns:
        A ``torch.device`` instance.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(
    model: nn.Module,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Move a model to the specified device.

    Args:
        model: The model to move.
        device: Target device. If ``None``, uses ``get_best_device()``.

    Returns:
        The model on the target device.
    """
    if device is None:
        device = get_best_device()
    return model.to(device)


def model_size_mb(model: nn.Module) -> float:
    """Compute the model's parameter size in megabytes.

    Args:
        model: The model to measure.

    Returns:
        Size in MB (based on parameter data type).
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 * 1024)


def parameter_count(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a model.

    Args:
        model: The model to count.
        trainable_only: If ``True``, only count parameters that require gradients.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def inference_memory(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device | None = None,
) -> dict[str, float]:
    """Estimate memory usage for a single forward pass.

    Args:
        model: The model to measure.
        input_shape: Shape of a single input tensor ``(C, H, W)`` or ``(B, C, H, W)``.
        device: Device to run on. If ``None``, uses ``get_best_device()``.

    Returns:
        Dict with keys ``"model_mb"``, ``"input_mb"``, ``"output_mb"``,
        and ``"total_mb"`` (estimate).
    """
    if device is None:
        device = get_best_device()

    model = move_to_device(model, device)

    # Build a dummy input
    if len(input_shape) == 3:
        batch_shape = (1, *input_shape)
    else:
        batch_shape = input_shape

    dummy = torch.randn(batch_shape, device=device)

    # Model size
    model_mb = model_size_mb(model)

    # Input size
    input_bytes = dummy.numel() * dummy.element_size()
    input_mb = input_bytes / (1024 * 1024)

    # Forward pass to get output size
    model.eval()
    with torch.no_grad():
        output = model(dummy)

    output_bytes = output.numel() * output.element_size()
    output_mb = output_bytes / (1024 * 1024)

    return {
        "model_mb": round(model_mb, 2),
        "input_mb": round(input_mb, 2),
        "output_mb": round(output_mb, 2),
        "total_mb": round(model_mb + input_mb + output_mb, 2),
    }
