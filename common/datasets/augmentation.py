"""Augmentation factory for building data augmentation pipelines."""

from __future__ import annotations

from typing import Any

import torch
from torchvision import transforms as T


def build_augmentations(
    h_flip_prob: float | None = None,
    v_flip_prob: float | None = None,
    rotation_degrees: float | None = None,
    crop_size: tuple[int, int] | None = None,
    center_crop_size: tuple[int, int] | None = None,
    color_jitter_params: dict[str, Any] | None = None,
) -> T.Compose | None:
    """Build a data augmentation pipeline.

    Parameters
    ----------
    h_flip_prob : float | None
        Probability of random horizontal flip.
    v_flip_prob : float | None
        Probability of random vertical flip.
    rotation_degrees : float | None
        Degrees for random rotation.
    crop_size : tuple[int, int] | None
        Size for random crop.
    center_crop_size : tuple[int, int] | None
        Size for center crop.
    color_jitter_params : dict[str, Any] | None
        Parameters dict for ColorJitter (e.g. ``{"brightness": 0.2}``).

    Returns
    -------
    T.Compose | None
        Composed augmentation pipeline, or None if no augmentations requested.
    """
    augments: list[Any] = []

    if h_flip_prob is not None:
        augments.append(T.RandomHorizontalFlip(p=h_flip_prob))

    if v_flip_prob is not None:
        augments.append(T.RandomVerticalFlip(p=v_flip_prob))

    if rotation_degrees is not None:
        augments.append(T.RandomRotation(degrees=rotation_degrees))

    if crop_size is not None:
        augments.append(T.RandomCrop(crop_size))

    if center_crop_size is not None:
        augments.append(T.CenterCrop(center_crop_size))

    if color_jitter_params is not None:
        augments.append(T.ColorJitter(**color_jitter_params))

    if not augments:
        return None

    return T.Compose(augments)