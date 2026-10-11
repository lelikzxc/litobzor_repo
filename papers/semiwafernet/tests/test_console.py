"""The internal log preserves terminal tqdm and restores console/logger state."""

from __future__ import annotations

import io
import logging
import sys

import pytest

from papers.semiwafernet.training.console import ConsoleLog, training_output
from papers.semiwafernet.training.progress import progress_bar, progress_write


class _Console(io.StringIO):
    def __init__(self, *, terminal: bool = True) -> None:
        super().__init__()
        self.terminal = terminal

    @property
    def encoding(self) -> str:
        return "cp1251"

    def fileno(self) -> int:
        return 123

    def isatty(self) -> bool:
        return self.terminal


@pytest.fixture
def logger(request):
    name = f"tests.semiwafernet.console.{request.node.name}"
    instance = logging.getLogger(name)
    previous = (instance.handlers[:], instance.level, instance.propagate, instance.disabled)
    instance.handlers = []
    instance.setLevel(logging.INFO)
    instance.propagate = False
    instance.disabled = False
    try:
        yield instance
    finally:
        for handler in instance.handlers:
            handler.close()
        instance.handlers, level, propagate, disabled = previous
        instance.setLevel(level)
        instance.propagate, instance.disabled = propagate, disabled


@pytest.mark.parametrize("terminal", [False, True])
def test_console_log_preserves_terminal_attributes_and_unicode(terminal) -> None:
    console, log = _Console(terminal=terminal), io.StringIO()
    output = ConsoleLog(console, log)
    assert output.isatty() is terminal
    assert output.encoding == "cp1251"
    assert output.fileno() == 123
    message = "Эпоха 1: loss=0.1234\n"
    assert output.write(message) == len(message)
    output.flush()
    assert console.getvalue() == message
    assert log.getvalue() == message


def test_internal_log_captures_prints_and_stdout_logger_but_excludes_tqdm(
    tmp_path, monkeypatch, logger
) -> None:
    console, stderr = _Console(), _Console()
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "stderr", stderr)
    stdout_handler = logging.StreamHandler(console)
    stderr_handler = logging.StreamHandler(stderr)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    path = tmp_path / "training.log"

    with training_output(path):
        wrapped = sys.stdout
        assert isinstance(wrapped, ConsoleLog)
        assert wrapped.isatty()
        assert wrapped.encoding == console.encoding
        assert stdout_handler.stream is wrapped
        assert stderr_handler.stream is stderr
        assert sys.stderr is stderr
        print("Training started")
        logger.info("Epoch metrics")
        with progress_bar(range(3), desc="MC pseudo-labels") as bar:
            for _ in bar:
                pass
        progress_write("Training finished")
        assert path.read_text(encoding="utf-8").endswith("Training finished\n")

    assert sys.stdout is console
    assert sys.stderr is stderr
    assert stdout_handler.stream is console
    assert stderr_handler.stream is stderr
    assert wrapped.log.closed
    assert not console.closed
    expected = "Training started\nEpoch metrics\nTraining finished\n"
    assert console.getvalue() == expected
    assert path.read_text(encoding="utf-8") == expected
    assert "MC pseudo-labels" in stderr.getvalue()
    assert "\r" in stderr.getvalue()
    assert "\r" not in path.read_text(encoding="utf-8")


def test_no_log_leaves_streams_and_handlers_untouched(monkeypatch, logger) -> None:
    console, stderr = _Console(), _Console()
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "stderr", stderr)
    handler = logging.StreamHandler(console)
    logger.addHandler(handler)
    with training_output(None):
        assert sys.stdout is console
        assert sys.stderr is stderr
        assert handler.stream is console
        print("No file requested")
    assert console.getvalue() == "No file requested\n"


@pytest.mark.parametrize("error", [RuntimeError("failed epoch"), KeyboardInterrupt("stopped")])
def test_exception_is_logged_and_console_and_handlers_are_restored(
    tmp_path, monkeypatch, logger, error
) -> None:
    console, stderr = _Console(), _Console()
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "stderr", stderr)
    handler = logging.StreamHandler(console)
    logger.addHandler(handler)
    path = tmp_path / "failed.log"
    with pytest.raises(type(error), match=str(error)):
        with training_output(path):
            wrapped = sys.stdout
            logger.info("Before failure")
            raise error

    assert sys.stdout is console
    assert sys.stderr is stderr
    assert handler.stream is console
    assert wrapped.log.closed
    text = path.read_text(encoding="utf-8")
    assert text.startswith("Before failure\n")
    assert text.count("Traceback (most recent call last)") == 1
    assert f"{type(error).__name__}: {error}" in text
    assert console.getvalue() == "Before failure\n"
    logger.info("After context")
    assert console.getvalue().endswith("After context\n")
    assert "After context" not in path.read_text(encoding="utf-8")


def test_log_appends_utf8_and_creates_parent_directory(tmp_path, monkeypatch) -> None:
    console = _Console()
    monkeypatch.setattr(sys, "stdout", console)
    path = tmp_path / "nested" / "run.log"
    for message in ("Первый запуск", "Второй запуск"):
        with training_output(path):
            print(message)
    assert path.read_bytes() == "Первый запуск\nВторой запуск\n".encode("utf-8")
    assert sys.stdout is console


def test_logger_created_during_training_is_restored_before_log_closes(
    tmp_path, monkeypatch, logger
) -> None:
    console = _Console()
    monkeypatch.setattr(sys, "stdout", console)
    path = tmp_path / "created_handler.log"
    with training_output(path):
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        logger.info("During training")
    assert handler.stream is console
    logger.info("After training")
    assert console.getvalue() == "During training\nAfter training\n"
    assert path.read_text(encoding="utf-8") == "During training\n"


def test_nested_training_logs_restore_each_previous_stream(tmp_path, monkeypatch, logger) -> None:
    console = _Console()
    monkeypatch.setattr(sys, "stdout", console)
    handler = logging.StreamHandler(console)
    logger.addHandler(handler)
    outer_path, inner_path = tmp_path / "outer.log", tmp_path / "inner.log"
    with training_output(outer_path):
        outer = sys.stdout
        logger.info("Outer before")
        with training_output(inner_path):
            logger.info("Inner")
        assert sys.stdout is outer
        assert handler.stream is outer
        logger.info("Outer after")
    assert sys.stdout is console
    assert handler.stream is console
    assert outer_path.read_text(encoding="utf-8") == "Outer before\nInner\nOuter after\n"
    assert inner_path.read_text(encoding="utf-8") == "Inner\n"
