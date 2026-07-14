"""Tests for common.utils.logger."""

from __future__ import annotations

import logging

from common.utils.logger import get_logger


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("litobzor.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "litobzor.test"


def test_get_logger_has_handler(capsys: object) -> None:
    logger = get_logger("litobzor.handler_test")
    logger.info("logger smoke test")
    assert logger.hasHandlers()


def test_get_logger_level_override() -> None:
    logger = get_logger("litobzor.level_test", level=logging.DEBUG)
    assert logger.level == logging.DEBUG
