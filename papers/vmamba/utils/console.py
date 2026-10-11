"""Single-line terminal progress and UTF-8 logs for VMamba entry points."""

from __future__ import annotations

import logging
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

from common.training.logger import TrainingLogger


def progress_bar(iterable, *, desc=""):
    try:
        interactive = bool(sys.stderr.isatty())
    except (AttributeError, OSError, ValueError):
        interactive = False
    return tqdm(
        iterable,
        desc=desc,
        file=sys.stderr,
        disable=not interactive,
        ascii=True,
        dynamic_ncols=True,
        position=0,
        leave=False,
        mininterval=1.0,
    )


class ProgressLoader:
    def __init__(self, loader, *, desc):
        self.loader = loader
        self.desc = desc

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "loader"), name)

    def __iter__(self):
        desc = self.desc() if callable(self.desc) else self.desc
        with progress_bar(self.loader, desc=desc) as batches:
            yield from batches


class EpochLogger(TrainingLogger):
    """Record the common trainer's metrics and print one line per epoch."""

    def __init__(self, current_epoch):
        super().__init__()
        self.current_epoch = current_epoch

    def log_epoch(self, **metrics):
        super().log_epoch(**metrics)
        fields = [f"Epoch {self.current_epoch()}"]
        for key in ("train_loss", "val_loss", "val_accuracy", "val_f1", "lr"):
            if key in metrics:
                fields.append(f"{key}={metrics[key]:.6g}")
        tqdm.write(" | ".join(fields), file=sys.stdout)


class ConsoleLog:
    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, value):
        written = self.console.write(value)
        self.log.write(value)
        self.log.flush()
        return written

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return self.console.isatty()

    def __getattr__(self, name):
        return getattr(self.console, name)


def _loggers():
    return [
        logging.getLogger(),
        *[
            logger
            for logger in logging.Logger.manager.loggerDict.values()
            if isinstance(logger, logging.Logger)
        ],
    ]


@contextmanager
def training_output(log_file: str | Path | None):
    """Keep stderr interactive; save ordinary output and exception tracebacks."""
    if log_file is None:
        yield
        return
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = sys.stdout
    with path.open("a", encoding="utf-8", newline="") as log:
        output = ConsoleLog(original, log)
        sys.stdout = output
        for logger in _loggers():
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream is original:
                    handler.setStream(output)
        try:
            yield
        except BaseException:
            traceback.print_exc(file=log)
            raise
        finally:
            try:
                output.flush()
            finally:
                sys.stdout = original
                for logger in _loggers():
                    for handler in logger.handlers:
                        if isinstance(handler, logging.StreamHandler) and handler.stream is output:
                            handler.setStream(original)
