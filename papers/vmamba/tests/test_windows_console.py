"""Windows console progress stays bounded and UTF-8 logging restores streams."""

from __future__ import annotations

import io
import logging
import sys

import pytest

from papers.vmamba.utils import console as console_module


class _Console(io.StringIO):
    def __init__(self, *, terminal=False, broken=False):
        super().__init__()
        self.terminal = terminal
        self.broken = broken

    @property
    def encoding(self):
        return "cp1251"

    def isatty(self):
        if self.broken:
            raise OSError("The console handle is closed")
        return self.terminal


class _Progress:
    def __init__(self, iterable):
        self.iterable = iterable
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def __iter__(self):
        yield from self.iterable


@pytest.fixture
def logger(request):
    instance = logging.getLogger(f"tests.vmamba.console.{request.node.name}")
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


@pytest.mark.parametrize("terminal,broken,disabled", [
    (True, False, False), (False, False, True), (True, True, True),
])
def test_progress_chooses_one_terminal_line_and_disables_redirected_updates(
    monkeypatch, terminal, broken, disabled
):
    stderr = _Console(terminal=terminal, broken=broken)
    monkeypatch.setattr(sys, "stderr", stderr)
    calls = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return _Progress(iterable)

    monkeypatch.setattr(console_module, "tqdm", fake_tqdm)
    with console_module.progress_bar(range(3), desc="Epoch 1/50") as bar:
        assert list(bar) == [0, 1, 2]
    assert bar.closed
    assert len(calls) == 1
    options = calls[0]
    assert options["file"] is stderr
    assert options["disable"] is disabled
    assert options["desc"] == "Epoch 1/50"
    assert options["ascii"] and options["dynamic_ncols"]
    assert options["position"] == 0
    assert options["leave"] is False
    assert options["mininterval"] >= 1.0
    assert stderr.getvalue() == ""


def test_real_redirected_progress_emits_no_percentages(monkeypatch):
    stderr = _Console(terminal=False)
    monkeypatch.setattr(sys, "stderr", stderr)
    with console_module.progress_bar(range(1000), desc="Training") as bar:
        assert bar.disable
        assert list(bar) == list(range(1000))
    assert stderr.getvalue() == ""


def test_progress_loader_retains_batches_metadata_and_closes_each_iteration(monkeypatch):
    batches = [[1, 2], [3]]

    class Loader:
        dataset = object()

        def __len__(self):
            return len(batches)

        def __iter__(self):
            yield from batches

    descriptions, bars = [], []

    def fake_progress(iterable, *, desc):
        descriptions.append(desc)
        result = _Progress(iterable)
        bars.append(result)
        return result

    monkeypatch.setattr(console_module, "progress_bar", fake_progress)
    epoch = 1
    loader = Loader()
    wrapped = console_module.ProgressLoader(loader, desc=lambda: f"Epoch {epoch}/50")
    assert len(wrapped) == len(loader)
    assert wrapped.dataset is loader.dataset
    assert list(wrapped) == batches
    epoch = 2
    assert list(wrapped) == batches
    assert descriptions == ["Epoch 1/50", "Epoch 2/50"]
    assert all(bar.closed for bar in bars)
    partial = iter(wrapped)
    assert next(partial) is batches[0]
    partial.close()
    assert bars[-1].closed


def test_epoch_logger_prints_exactly_one_line_and_retains_complete_history(monkeypatch):
    stdout, stderr = _Console(), _Console()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    times = iter([100.0, 101.0, 102.0, 104.0])
    monkeypatch.setattr("common.training.logger.time.time", lambda: next(times))
    epoch = 4
    logger = console_module.EpochLogger(lambda: epoch)
    first = dict(train_loss=1.25, val_loss=0.5, val_accuracy=0.8,
                 val_f1=0.7, lr=0.001, train_f1=0.65)
    second = dict(train_loss=1.0, val_loss=0.4, val_accuracy=0.85,
                  val_f1=0.75, lr=0.0009, train_f1=0.7)
    logger.begin_epoch()
    logger.log_epoch(**first)
    epoch = 5
    logger.begin_epoch()
    logger.log_epoch(**second)
    assert logger.history == [dict(first, epoch_time=1.0), dict(second, epoch_time=2.0)]
    assert logger.latest() == dict(second, epoch_time=2.0)
    assert stdout.getvalue().splitlines() == [
        "Epoch 4 | train_loss=1.25 | val_loss=0.5 | val_accuracy=0.8 | val_f1=0.7 | lr=0.001",
        "Epoch 5 | train_loss=1 | val_loss=0.4 | val_accuracy=0.85 | val_f1=0.75 | lr=0.0009",
    ]
    assert "\r" not in stdout.getvalue()
    assert stderr.getvalue() == ""


@pytest.mark.parametrize("terminal", [True, False])
def test_utf8_training_log_preserves_console_and_restores_existing_handlers(
    tmp_path, monkeypatch, logger, terminal
):
    stdout, stderr = _Console(terminal=terminal), _Console(terminal=terminal)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    stdout_handler = logging.StreamHandler(stdout)
    stderr_handler = logging.StreamHandler(stderr)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    path = tmp_path / "nested" / "train.log"
    with console_module.training_output(path):
        wrapped = sys.stdout
        assert isinstance(wrapped, console_module.ConsoleLog)
        assert wrapped.isatty() is terminal
        assert wrapped.encoding == "cp1251"
        assert stdout_handler.stream is wrapped
        assert stderr_handler.stream is stderr
        assert sys.stderr is stderr
        print("Начало обучения")
        logger.info("Epoch metrics")
        stderr.write("terminal progress redraw\r")
        console_module.EpochLogger(lambda: 1).log_epoch(train_loss=0.25)
    assert sys.stdout is stdout and sys.stderr is stderr
    assert stdout_handler.stream is stdout and stderr_handler.stream is stderr
    assert wrapped.log.closed
    assert not stdout.closed
    expected = "Начало обучения\nEpoch metrics\nEpoch 1 | train_loss=0.25\n"
    assert stdout.getvalue() == expected
    assert path.read_bytes() == expected.encode("utf-8")
    assert "progress" not in path.read_text(encoding="utf-8")
    assert "\r" not in path.read_text(encoding="utf-8")
    logger.info("After context")
    assert stdout.getvalue().endswith("After context\n")
    assert path.read_text(encoding="utf-8") == expected


@pytest.mark.parametrize("error", [RuntimeError("Ошибка эпохи"), KeyboardInterrupt("Остановка")])
def test_failed_context_records_traceback_and_restores_stdout_and_logging(
    tmp_path, monkeypatch, logger, error
):
    stdout, stderr = _Console(), _Console()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    handler = logging.StreamHandler(stdout)
    logger.addHandler(handler)
    path = tmp_path / "failed.log"
    with pytest.raises(type(error), match=str(error)):
        with console_module.training_output(path):
            wrapped = sys.stdout
            logger.info("Перед ошибкой")
            raise error
    assert sys.stdout is stdout and sys.stderr is stderr
    assert handler.stream is stdout
    assert wrapped.log.closed
    saved = path.read_text(encoding="utf-8")
    assert saved.startswith("Перед ошибкой\n")
    assert saved.count("Traceback (most recent call last)") == 1
    assert f"{type(error).__name__}: {error}" in saved
    assert stdout.getvalue() == "Перед ошибкой\n"
    logger.info("После ошибки")
    assert "После ошибки" in stdout.getvalue()
    assert "После ошибки" not in path.read_text(encoding="utf-8")


def test_new_handler_inside_context_is_restored_and_log_appends_utf8(
    tmp_path, monkeypatch, logger
):
    stdout = _Console()
    monkeypatch.setattr(sys, "stdout", stdout)
    path = tmp_path / "train.log"
    with console_module.training_output(path):
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        logger.info("Первый запуск")
    assert handler.stream is stdout
    with console_module.training_output(path):
        logger.info("Второй запуск")
    assert handler.stream is stdout
    expected = "Первый запуск\nВторой запуск\n"
    assert stdout.getvalue() == expected
    assert path.read_bytes() == expected.encode("utf-8")


def test_no_file_preserves_streams_and_logging(monkeypatch, logger):
    stdout, stderr = _Console(), _Console()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    handler = logging.StreamHandler(stdout)
    logger.addHandler(handler)
    with console_module.training_output(None):
        assert sys.stdout is stdout and sys.stderr is stderr
        assert handler.stream is stdout
        logger.info("No file requested")
    assert stdout.getvalue() == "No file requested\n"
