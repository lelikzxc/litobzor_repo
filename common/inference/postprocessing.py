"""Reusable postprocessing utilities for classification and segmentation.

All functions accept torch tensors and return torch tensors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logits_to_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Convert logits to probabilities via softmax.

    Args:
        logits: Raw class scores ``[B, C]`` or ``[B, C, H, W]``.
        temperature: Softmax temperature (``> 0``). Higher values produce
            softer distributions.

    Returns:
        Probability tensor with the same shape as ``logits``.
    """
    return F.softmax(logits / temperature, dim=1)


def logits_to_class(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to predicted class indices via argmax.

    Args:
        logits: Raw class scores ``[B, C]`` or ``[B, C, H, W]``.

    Returns:
        Predicted class indices ``[B]`` or ``[B, H, W]``.
    """
    return logits.argmax(dim=1)


def topk_predictions(
    logits: torch.Tensor, k: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top-k predicted class indices and probabilities.

    Args:
        logits: Raw class scores ``[B, C]``.
        k: Number of top predictions to return.

    Returns:
        Tuple of ``(indices, probabilities)``, each ``[B, k]``.
    """
    probs = F.softmax(logits, dim=1)
    k = min(k, logits.shape[1])
    return torch.topk(probs, k, dim=1)


def logits_to_mask(
    logits: torch.Tensor,
    threshold: float | None = None,
) -> torch.Tensor:
    """Convert segmentation logits to a class mask.

    Args:
        logits: Raw pixel scores ``[B, C, H, W]``.
        threshold: If ``None``, uses argmax (multi-class). If a float,
            applies threshold to sigmoid (binary segmentation).

    Returns:
        Predicted mask ``[B, H, W]`` with integer class indices.
    """
    if threshold is not None:
        # Binary segmentation with threshold
        probs = torch.sigmoid(logits)
        if logits.shape[1] == 1:
            return (probs.squeeze(1) >= threshold).long()
        # Multi-class with confidence threshold
        mask = logits.argmax(dim=1)
        max_probs = probs.max(dim=1).values
        mask[max_probs < threshold] = 0
        return mask
    return logits.argmax(dim=1)
