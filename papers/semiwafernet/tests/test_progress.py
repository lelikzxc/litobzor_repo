"""Progress output stays readable in a terminal and in redirected Windows logs."""

from __future__ import annotations

import io
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from papers.semiwafernet.training.progress import (
    ProgressLoader,
    batch_progress,
    progress_bar,
    progress_write,
)
from papers.semiwafernet.training.stage_manager import StageManager
from papers.semiwafernet.training.trainer import Trainer


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


class _BrokenTerminal(io.StringIO):
    def isatty(self) -> bool:
        raise OSError("The console handle is closed")


class _Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"classification": self.fc(images.flatten(1))}


def _trainer() -> Trainer:
    model = _Classifier()
    manager = StageManager(model, num_classes=2, mc_passes=2)
    return Trainer(
        model,
        manager,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        supervised_loss_fn=lambda out, targets: nn.functional.cross_entropy(
            out["classification"], targets["classification"]
        ),
        device=torch.device("cpu"),
        batch_size=2,
    )


@pytest.mark.parametrize("stream", [io.StringIO(), _BrokenTerminal()])
def test_redirected_stream_emits_no_batch_redraws(stream: io.StringIO) -> None:
    with progress_bar(range(1000), file=stream, desc="Training") as bar:
        assert bar.disable
        for index in bar:
            bar.set_postfix(loss=str(index), refresh=False)
    assert stream.getvalue() == ""


def test_explicit_quiet_suppresses_terminal_bar() -> None:
    stream = _Terminal()
    with batch_progress(range(10), desc="Quiet", disable=True, file=stream) as bar:
        assert list(bar) == list(range(10))
    assert stream.getvalue() == ""


def test_terminal_uses_one_line_and_throttles_postfix_redraws() -> None:
    stream = _Terminal()
    with batch_progress(range(100), desc="Training", file=stream) as bar:
        assert not bar.disable
        bar.refresh = Mock(wraps=bar.refresh)
        for index in bar:
            bar.set_postfix(loss=str(index), refresh=False)
        # Fast batches must not force a redraw for every new loss value.
        assert bar.refresh.call_count == 0
    output = stream.getvalue()
    assert "Training" in output
    assert "\r" in output
    assert "\n" not in output
    assert "\x1b[A" not in output  # No cursor-up codes for nested bars.
    assert output.isascii()


def test_summary_survives_redirection_without_progress_characters(monkeypatch) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)
    with progress_bar(range(2), desc="Hidden") as bar:
        for _ in bar:
            pass
        progress_write("[SSL Stage 1] epoch 1/1 loss=0.1234")
    assert stdout.getvalue() == "[SSL Stage 1] epoch 1/1 loss=0.1234\n"
    assert stderr.getvalue() == ""


def test_exception_closes_the_terminal_bar() -> None:
    existing = set(tqdm._instances)
    with pytest.raises(RuntimeError, match="bad batch"):
        with progress_bar(range(2), file=_Terminal()) as bar:
            for _ in bar:
                raise RuntimeError("bad batch")
    assert set(tqdm._instances) == existing


def test_ssl_epochs_write_only_bounded_summaries_when_redirected(monkeypatch) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)
    loader = DataLoader(
        TensorDataset(torch.randn(6, 1, 2, 2), torch.tensor([0, 1, 0, 1, 0, 1])),
        batch_size=2,
    )
    trainer = _trainer()
    metrics = trainer.train_stage1(loader, num_epochs=3, val_eval_fn=lambda: 0.5)
    lines = stdout.getvalue().splitlines()
    assert sum("epoch " in line and "loss=" in line for line in lines) == 3
    assert len(lines) == 5  # Three epochs, best teacher, stage loss.
    assert metrics["best_val_metric"] == 0.5
    assert trainer.current_epoch == 3
    assert stderr.getvalue() == ""
    assert "\r" not in stdout.getvalue()


def test_stage2_and_mc_are_quiet_under_redirection(monkeypatch) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)
    labeled = DataLoader(
        TensorDataset(torch.randn(4, 1, 2, 2), torch.tensor([0, 1, 0, 1])),
        batch_size=2,
    )
    unlabeled = DataLoader(TensorDataset(torch.randn(4, 1, 2, 2)), batch_size=2)
    trainer = _trainer()
    metrics = trainer.train_stage2(labeled, unlabeled, num_epochs=2)
    lines = stdout.getvalue().splitlines()
    assert sum("epoch " in line and "loss=" in line for line in lines) == 2
    assert len(lines) <= 5  # Pseudo-set summary(s), epochs, stage loss.
    assert metrics["loss"] >= 0
    assert stderr.getvalue() == ""
    assert "\r" not in stdout.getvalue()


def test_mc_error_closes_terminal_bar(monkeypatch) -> None:
    monkeypatch.setattr("sys.stderr", _Terminal())
    trainer = _trainer()
    existing = set(tqdm._instances)
    with pytest.raises(TypeError, match="Expected"):
        trainer.stage_manager._mc_collect(
            [torch.zeros(1)], torch.device("cpu"), None, "MC", True, with_labels=True
        )
    assert set(tqdm._instances) == existing


@pytest.mark.parametrize("stream", [io.StringIO(), _Terminal()])
def test_progress_loader_preserves_batches_and_reiterates(monkeypatch, stream) -> None:
    monkeypatch.setattr("sys.stderr", stream)
    loader = DataLoader(
        TensorDataset(torch.arange(6)), batch_size=2, shuffle=False
    )
    epoch = 1
    wrapped = ProgressLoader(loader, desc=lambda: f"Epoch {epoch}")
    assert len(wrapped) == len(loader)
    assert wrapped.dataset is loader.dataset
    assert wrapped.sampler is loader.sampler
    assert torch.equal(torch.cat([batch[0] for batch in wrapped]), torch.arange(6))
    epoch = 2
    assert torch.equal(torch.cat([batch[0] for batch in wrapped]), torch.arange(6))
    if stream.isatty():
        assert "Epoch 1" in stream.getvalue()
        assert "Epoch 2" in stream.getvalue()
        assert "\x1b[A" not in stream.getvalue()
    else:
        assert stream.getvalue() == ""
