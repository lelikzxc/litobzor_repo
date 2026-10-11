"""Keep the terminal interactive while recording ordinary training output."""

from __future__ import annotations

import logging
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


class ConsoleLog:
    """Copy stdout to a UTF-8 log; tqdm stays on the untouched stderr."""

    def __init__(self, console: TextIO, log: TextIO) -> None:
        self.console = console
        self.log = log

    def write(self, value: str) -> int:
        written = self.console.write(value)
        self.log.write(value)
        self.log.flush()
        return written

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.console.isatty()

    def __getattr__(self, name: str):
        return getattr(self.console, name)


@contextmanager
def training_output(log_file: str | Path | None) -> Iterator[None]:
    """Log prints and existing stdout loggers without piping the terminal."""
    if log_file is None:
        yield
        return
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = sys.stdout
    with path.open("a", encoding="utf-8", newline="") as log:
        output = ConsoleLog(original, log)
        sys.stdout = output
        # Some repository loggers are configured during module import.
        loggers = [logging.getLogger(), *[
            value for value in logging.Logger.manager.loggerDict.values()
            if isinstance(value, logging.Logger)
        ]]
        for logger in loggers:
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
                # Include handlers created while the context was active.
                current_loggers = [logging.getLogger(), *[
                    value for value in logging.Logger.manager.loggerDict.values()
                    if isinstance(value, logging.Logger)
                ]]
                for logger in current_loggers:
                    for handler in logger.handlers:
                        if isinstance(handler, logging.StreamHandler) and handler.stream is output:
                            handler.setStream(original)


__all__ = ["ConsoleLog", "training_output"]
