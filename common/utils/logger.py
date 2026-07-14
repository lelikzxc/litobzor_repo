"""Logging utilities."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LEVEL = logging.INFO

_configured = False


def _configure_root_logger(level: int = _DEFAULT_LEVEL) -> None:
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _configured = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Optional logging level override for this logger.

    Returns:
        A logger instance ready for use across the project.
    """
    _configure_root_logger(level or _DEFAULT_LEVEL)
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
