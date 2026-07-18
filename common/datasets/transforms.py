"""Transform factory for building common torchvision transform pipelines."""

from __future__ import annotations

from typing import Any

from torchvision import transforms as T


def build_transforms(
    resize_size: tuple[int, int] | None = None,
    normalize: bool = True,
    to_tensor: bool = True,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    extra: list[Any] | None = None,
) -> T.Compose:
    """Build a standard image transform pipeline.

    Parameters
    ----------
    resize_size : tuple[int, int] | None
        If provided, adds ``Resize`` to the pipeline.
    normalize : bool
        Whether to add ``Normalize`` (default True).
    to_tensor : bool
        Whether to add ``ToTensor`` (default True).
    mean : list[float] | None
        Mean for normalization. Defaults to ImageNet stats.
    std : list[float] | None
        Std for normalization. Defaults to ImageNet stats.
    extra : list[Any] | None
        Additional transforms to prepend before ToTensor/Normalize.

    Returns
    -------
    T.Compose
        Composed transform pipeline.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    transforms: list[Any] = []

    if extra is not None:
        transforms.extend(extra)

    if resize_size is not None:
        transforms.append(T.Resize(resize_size))

    if to_tensor:
        transforms.append(T.ToTensor())

    if normalize:
        transforms.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transforms)