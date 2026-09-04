"""Progress bar helpers for SemiWaferNet training."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def cycle_loader(loader: Any) -> Iterator[Any]:
    """Yield batches from ``loader`` indefinitely (restarts each epoch)."""
    while True:
        yield from loader


def epoch_progress(
    num_epochs: int,
    *,
    stage: int,
    title: str,
    disable: bool = False,
) -> tqdm:
    """Outer progress bar for an SSL stage."""
    return tqdm(
        range(num_epochs),
        desc=f"[SSL Stage {stage}] {title}",
        unit="epoch",
        disable=disable,
    )


def batch_progress(
    iterable: Any,
    *,
    desc: str,
    total: int | None = None,
    disable: bool = False,
) -> tqdm:
    """Inner progress bar for batches within one epoch."""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=False,
        unit="batch",
        disable=disable,
    )


__all__ = ["cycle_loader", "epoch_progress", "batch_progress"]
