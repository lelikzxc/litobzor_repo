"""Visualization helpers for inference results.

Supports classification (label, probability table, top-k) and
segmentation (predicted mask, colored overlay, side-by-side comparison).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.inference.postprocessing import logits_to_probs, topk_predictions


def plot_classification_result(
    image: torch.Tensor | np.ndarray,
    logits: torch.Tensor,
    class_names: list[str] | None = None,
    top_k: int = 5,
    title: str | None = None,
    ax: Any = None,
) -> None:
    """Plot a classification result with image and top-k probability bar chart.

    Args:
        image: Input image ``[C, H, W]`` or ``[H, W, C]``.
        logits: Raw class scores ``[C]`` or ``[1, C]``.
        class_names: Optional list of class names.
        top_k: Number of top predictions to display.
        title: Optional title.
        ax: Optional matplotlib axis. If ``None``, creates a new figure.
    """
    if ax is None:
        _, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        # If only one axis provided, split into two
        fig = ax.figure
        ax_img, ax_bar = ax, fig.add_subplot(1, 2, 2)

    # Display image
    img = _to_displayable(image)
    ax_img.imshow(img)
    ax_img.axis("off")
    if title:
        ax_img.set_title(title)

    # Top-k bar chart
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs, indices = topk_predictions(logits, k=top_k)
    indices = indices.squeeze(0).cpu().tolist()
    probs = probs.squeeze(0).cpu().numpy()

    labels = [class_names[i] if class_names else f"Class {i}" for i in indices]
    colors = plt.cm.Blues(probs / probs.max() + 0.2)

    ax_bar.barh(range(len(labels)), probs, color=colors)
    ax_bar.set_yticks(range(len(labels)))
    ax_bar.set_yticklabels(labels)
    ax_bar.set_xlabel("Probability")
    ax_bar.set_title("Top-k Predictions")
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.0)

    plt.tight_layout()


def plot_segmentation_result(
    image: torch.Tensor | np.ndarray,
    logits: torch.Tensor,
    alpha: float = 0.5,
    title: str | None = None,
    ax: Any = None,
    cmap: str = "tab10",
) -> None:
    """Plot a segmentation result with image, predicted mask, and overlay.

    Args:
        image: Input image ``[C, H, W]`` or ``[H, W, C]``.
        logits: Raw pixel scores ``[C, H, W]`` or ``[1, C, H, W]``.
        alpha: Overlay transparency.
        title: Optional title.
        ax: Optional matplotlib axis.
        cmap: Colormap for the mask.
    """
    if ax is None:
        _, (ax_img, ax_mask, ax_overlay) = plt.subplots(1, 3, figsize=(12, 4))
    else:
        fig = ax.figure
        ax_img = ax
        ax_mask = fig.add_subplot(1, 3, 2)
        ax_overlay = fig.add_subplot(1, 3, 3)

    # Image
    img = _to_displayable(image)
    ax_img.imshow(img)
    ax_img.set_title("Image")
    ax_img.axis("off")

    # Predicted mask
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    ax_mask.imshow(mask, cmap=cmap, interpolation="nearest")
    ax_mask.set_title("Predicted Mask")
    ax_mask.axis("off")

    # Overlay
    ax_overlay.imshow(img)
    ax_overlay.imshow(mask, cmap=cmap, alpha=alpha, interpolation="nearest")
    ax_overlay.set_title("Overlay")
    ax_overlay.axis("off")

    if title:
        ax_img.set_title(title)

    plt.tight_layout()


def plot_segmentation_comparison(
    image: torch.Tensor | np.ndarray,
    logits: torch.Tensor,
    ground_truth: torch.Tensor | np.ndarray,
    alpha: float = 0.5,
    title: str | None = None,
    cmap: str = "tab10",
) -> None:
    """Plot a side-by-side comparison of predicted vs ground-truth segmentation.

    Args:
        image: Input image ``[C, H, W]`` or ``[H, W, C]``.
        logits: Raw pixel scores ``[C, H, W]`` or ``[1, C, H, W]``.
        ground_truth: Ground-truth mask ``[H, W]``.
        alpha: Overlay transparency.
        title: Optional title.
        cmap: Colormap for masks.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    img = _to_displayable(image)
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    gt = _to_numpy(ground_truth)

    # Row 1: Image, Predicted, Ground Truth
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_mask, cmap=cmap, interpolation="nearest")
    axes[0, 1].set_title("Predicted")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(gt, cmap=cmap, interpolation="nearest")
    axes[0, 2].set_title("Ground Truth")
    axes[0, 2].axis("off")

    # Row 2: Overlays
    axes[1, 0].imshow(img)
    axes[1, 0].imshow(pred_mask, cmap=cmap, alpha=alpha, interpolation="nearest")
    axes[1, 0].set_title("Predicted Overlay")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img)
    axes[1, 1].imshow(gt, cmap=cmap, alpha=alpha, interpolation="nearest")
    axes[1, 1].set_title("Ground Truth Overlay")
    axes[1, 1].axis("off")

    # Difference map
    diff = (pred_mask != gt).astype(float)
    axes[1, 2].imshow(diff, cmap="Reds", interpolation="nearest")
    axes[1, 2].set_title("Difference (errors)")
    axes[1, 2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()


def _to_numpy(tensor_or_array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a tensor to a numpy array on CPU."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return tensor_or_array


def _to_displayable(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert an image tensor to a displayable numpy array ``[H, W, C]``."""
    img = _to_numpy(image)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(2)
    if img.max() <= 1.0 and img.min() >= -0.5:
        img = np.clip(img, 0, 1)
    return img
