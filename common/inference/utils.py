"""Helper utilities for inference.

Provides checkpoint loading, eval mode, gradient disabling, and seeding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from common.utils.seed import set_seed


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load model weights from a checkpoint file.

    Args:
        model: The model to load weights into.
        checkpoint_path: Path to the checkpoint file.
        strict: Whether to enforce strict key matching.
        device: Device to load the checkpoint onto.

    Returns:
        The full checkpoint dictionary (may contain optimizer, epoch, etc.).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device, weights_only=False)

    if "model" in state:
        model.load_state_dict(state["model"], strict=strict)
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=strict)
    else:
        # Assume the checkpoint is the state dict itself
        model.load_state_dict(state, strict=strict)

    return state


def set_eval_mode(model: nn.Module) -> nn.Module:
    """Set a model to evaluation mode.

    Args:
        model: The model to set to eval mode.

    Returns:
        The model in eval mode.
    """
    model.eval()
    return model


def disable_gradients(model: nn.Module) -> nn.Module:
    """Disable gradient computation for all parameters.

    Args:
        model: The model whose gradients will be disabled.

    Returns:
        The model with ``requires_grad=False`` on all parameters.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all frameworks.

    Args:
        seed: Global random seed value.
    """
    set_seed(seed)
