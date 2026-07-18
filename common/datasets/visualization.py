"""Visualization helpers for dataset inspection."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def show_image(
    image: torch.Tensor | np.ndarray,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Display an image tensor or array.

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        Image tensor (C, H, W) or numpy array (H, W, C).
    title : str | None
        Optional title for the plot.
    ax : plt.Axes | None
        Optional matplotlib axes to plot on.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if isinstance(image, torch.Tensor):
        # Convert (C, H, W) -> (H, W, C)
        image = image.permute(1, 2, 0).numpy()

    # Clip to valid range for display
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return fig


def show_mask(
    mask: torch.Tensor | np.ndarray,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Display a segmentation mask.

    Parameters
    ----------
    mask : torch.Tensor | np.ndarray
        Mask tensor (H, W) or numpy array (H, W).
    title : str | None
        Optional title for the plot.
    ax : plt.Axes | None
        Optional matplotlib axes to plot on.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    ax.imshow(mask, cmap="tab10", interpolation="nearest")
    if title:
        ax.set_title(title)
    ax.axis("off")
    return fig


def overlay_mask(
    image: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    title: str | None = None,
    ax: plt.Axes | None = None,
    alpha: float = 0.5,
) -> Figure:
    """Overlay a segmentation mask on an image.

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        Image tensor (C, H, W) or numpy array (H, W, C).
    mask : torch.Tensor | np.ndarray
        Mask tensor (H, W) or numpy array (H, W).
    title : str | None
        Optional title for the plot.
    ax : plt.Axes | None
        Optional matplotlib axes to plot on.
    alpha : float
        Transparency of the overlay (default 0.5).

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    image = np.clip(image, 0, 1)
    ax.imshow(image)
    ax.imshow(mask, cmap="tab10", interpolation="nearest", alpha=alpha)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return fig


def visualize_batch(
    batch: dict[str, Any],
    num_samples: int = 4,
    task: str = "classification",
) -> Figure:
    """Visualize a batch of samples.

    Parameters
    ----------
    batch : dict[str, Any]
        Batched dict with keys like ``"image"``, ``"label"``, ``"mask"``.
    num_samples : int
        Number of samples to display (default 4).
    task : str
        Task type: ``"classification"``, ``"segmentation"``, or ``"multitask"``.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    images = batch["image"]
    n = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)

        title_parts: list[str] = []
        if task in ("classification", "multitask") and "label" in batch:
            title_parts.append(f"Label: {batch['label'][i].item()}")
        if task in ("segmentation", "multitask") and "mask" in batch:
            title_parts.append(f"Mask: {batch['mask'][i].unique().tolist()}")
        axes[i].set_title("\n".join(title_parts))
        axes[i].axis("off")

    fig.tight_layout()
    return fig