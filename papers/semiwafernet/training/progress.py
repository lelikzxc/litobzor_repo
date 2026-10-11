"""Terminal-aware, single-line progress for SemiWaferNet.

Bars use stderr only when it is an interactive terminal. Redirected output
(including a PowerShell pipeline) receives ordinary stage/epoch summaries,
without carriage returns or a new line for every percentage update.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from typing import Any, TextIO

from tqdm import tqdm

def _is_terminal(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def progress_bar(
    iterable: Any = None,
    *,
    desc: str = "",
    total: int | None = None,
    unit: str = "batch",
    disable: bool = False,
    leave: bool = False,
    file: TextIO | None = None,
    **kwargs: Any,
) -> tqdm:
    """Create one ASCII bar; suppress redraws when the stream is redirected."""
    stream = sys.stderr if file is None else file
    options = {
        "ascii": True,
        "dynamic_ncols": True,
        "position": 0,
        "mininterval": 1.0,
        "maxinterval": 10.0,
    }
    options.update(kwargs)
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        disable=disable or not _is_terminal(stream),
        leave=leave,
        file=stream,
        **options,
    )


def progress_write(message: str, *, file: TextIO | None = None) -> None:
    """Print a summary without corrupting an active bar, and allow stdout logs."""
    tqdm.write(message, file=sys.stdout if file is None else file)


class ProgressLoader:
    """Wrap a common-trainer loader with one bar without changing its batches.

    The underlying loader remains available for dataset/sampler introspection;
    SSL dataset unions and MC collection should use their own unwrapped loader.
    A callable description can include the current epoch on each iteration.
    """

    def __init__(
        self, loader: Any, *, desc: str | Callable[[], str], disable: bool = False
    ) -> None:
        self.loader = loader
        self.desc = desc
        self.disable = disable

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "loader"), name)

    def __iter__(self) -> Iterator[Any]:
        desc = self.desc() if callable(self.desc) else self.desc
        total = len(self.loader) if hasattr(self.loader, "__len__") else None
        with progress_bar(self.loader, desc=desc, total=total, disable=self.disable) as batches:
            yield from batches


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
    """Standalone stage bar (do not nest with a batch bar)."""
    return progress_bar(
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
    file: TextIO | None = None,
) -> tqdm:
    """The sole progress bar for one training epoch."""
    return progress_bar(
        iterable,
        desc=desc,
        total=total,
        leave=False,
        unit="batch",
        disable=disable,
        file=file,
    )


__all__ = [
    "cycle_loader",
    "epoch_progress",
    "batch_progress",
    "progress_bar",
    "progress_write",
    "ProgressLoader",
]
