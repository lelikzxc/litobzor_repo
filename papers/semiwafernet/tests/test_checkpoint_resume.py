"""CPU integrations for SemiWaferNet's checkpoint compatibility adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from common.engine.engine import Engine
from papers.semiwafernet.utils.checkpoint import (
    normalize_semiwafernet_state_dict,
    resume_semiwafernet_engine,
)


class TinyClassificationWrapper(nn.Module):
    """Retain the real wrapper/head key layout without running the backbone."""

    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.classifier = nn.ModuleDict({"head": nn.Linear(3, 2)})

    def forward(self, inputs):
        return self.base_model.classifier["head"](inputs)


def _engine(save_dir: Path, mode: str = "max") -> Engine:
    engine = Engine(
        model=TinyClassificationWrapper(),
        config={
            "optimizer": {"name": "adamw", "lr": 0.01},
            "scheduler": {"name": "step", "kwargs": {"step_size": 1, "gamma": 0.8}},
            "checkpoint": {
                "save_dir": str(save_dir),
                "metric_name": "val_f1" if mode == "max" else "val_loss",
                "mode": mode,
            },
            "loss": {"name": "cross_entropy"},
            "metrics": [],
        },
        device="cpu",
    )
    engine.trainer.verbose = False
    return engine


def _step(engine: Engine) -> None:
    inputs = torch.tensor([[1.0, 0.2, -0.4], [0.0, -0.5, 0.8]])
    targets = torch.tensor([0, 1])
    engine.optimizer.zero_grad(set_to_none=True)
    engine.loss_fn(engine.model(inputs), targets).backward()
    engine.optimizer.step()
    engine.scheduler.step()


@pytest.mark.parametrize("mode,best,worse", [("max", 0.9, 0.8), ("min", 0.2, 0.3)])
def test_resume_best_without_sidecar_preserves_selection(tmp_path, mode, best, worse):
    original = _engine(tmp_path, mode)
    _step(original)
    manager = original.checkpoint_manager
    manager.save_best(original.model, original.optimizer, original.scheduler, 7, best)
    before = hashlib.sha256(manager.best_path.read_bytes()).hexdigest()

    resumed = _engine(tmp_path, mode)
    assert resume_semiwafernet_engine(resumed, manager.best_path) == 7
    assert resumed.state.best_metric == best
    assert resumed.checkpoint_manager.best_metric == best
    assert not resumed.checkpoint_manager.save_best(
        resumed.model, resumed.optimizer, resumed.scheduler, 8, worse
    )
    assert hashlib.sha256(manager.best_path.read_bytes()).hexdigest() == before


def test_resume_last_uses_historical_best_not_mislabeled_loss(tmp_path):
    original = _engine(tmp_path)
    _step(original)
    manager = original.checkpoint_manager
    manager.save_best(original.model, original.optimizer, original.scheduler, 1, 0.9)
    # This reproduces shared Trainer.fit's payload: metric is loss, while
    # metric_name still names the configured selection metric, val_f1.
    manager.save_last(original.model, original.optimizer, original.scheduler, 2, 2.5)

    resumed = _engine(tmp_path)
    assert resume_semiwafernet_engine(resumed, manager.last_path) == 2
    assert resumed.checkpoint_manager.best_metric == 0.9
    assert resumed.checkpoint_manager.save_best(
        resumed.model, resumed.optimizer, resumed.scheduler, 3, 0.95
    )


def test_full_resume_matches_uninterrupted_next_optimizer_step(tmp_path):
    original = _engine(tmp_path)
    _step(original)
    original.trainer.current_epoch = 7
    original.state.epoch = 7
    original.state.best_metric = 0.95
    original.state.global_step = 123
    original.state.extra = {"run": "windows-compatible"}
    original.checkpoint_manager.save_best(
        original.model, original.optimizer, original.scheduler, 7, 0.95
    )
    path = original.save()
    # A write interrupted between .pt and .state.json leaves a stale epoch.
    sidecar = path.with_suffix(".state.json")
    payload = json.loads(sidecar.read_text())
    payload["epoch"] = 6
    sidecar.write_text(json.dumps(payload))

    resumed = _engine(tmp_path)
    assert resume_semiwafernet_engine(resumed, path) == 7
    assert resumed.state.epoch == resumed.trainer.current_epoch == 7
    assert resumed.state.global_step == 123
    assert resumed.state.extra == {"run": "windows-compatible"}
    assert resumed.state.checkpoint_path == path
    assert resumed.checkpoint_manager.best_metric == 0.95

    _step(original)
    _step(resumed)
    for expected, actual in zip(original.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert resumed.scheduler.state_dict() == original.scheduler.state_dict()


def test_legacy_nested_classifier_loads_and_continues_training(tmp_path):
    original = _engine(tmp_path)
    _step(original)
    path = original.save()
    state = torch.load(path, weights_only=False)
    state["model"] = {
        key.replace(".classifier.head.", ".classifier."): value
        for key, value in state["model"].items()
    }
    torch.save(state, path)

    resumed = _engine(tmp_path)
    resume_semiwafernet_engine(resumed, path)
    _step(original)
    _step(resumed)
    for expected, actual in zip(original.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_legacy_raw_classifier_keys_are_normalized():
    legacy = {"classifier.weight": torch.ones(2, 3), "classifier.bias": torch.zeros(2)}
    assert set(normalize_semiwafernet_state_dict(legacy)) == {
        "classifier.head.weight", "classifier.head.bias"
    }


def test_conflicting_classifier_keys_fail_clearly():
    weights = {
        "base_model.classifier.weight": torch.ones(2, 3),
        "base_model.classifier.head.weight": torch.zeros(2, 3),
    }
    with pytest.raises(ValueError, match="conflicting classifier keys"):
        normalize_semiwafernet_state_dict(weights)


@pytest.mark.parametrize("mode,previous,new_score", [("max", 0.9, 0.8), ("min", 0.2, 0.3)])
def test_fresh_boundary_resume_can_save_best_without_historical_artifact(
    tmp_path, mode, previous, new_score
):
    source = _engine(tmp_path / "source", mode)
    _step(source)
    source.checkpoint_manager.save_best(
        source.model, source.optimizer, source.scheduler, 1, previous
    )
    source.state.best_metric = previous
    source.state.epoch = source.trainer.current_epoch = 1
    boundary = source.save(tmp_path / "source" / "ssl_stage1.pt")

    resumed = _engine(tmp_path / "fresh destination", mode)
    assert resume_semiwafernet_engine(resumed, boundary) == 1
    assert resumed.state.best_metric is None
    assert not resumed.checkpoint_manager.best_path.exists()
    _step(resumed)
    assert resumed.checkpoint_manager.save_best(
        resumed.model, resumed.optimizer, resumed.scheduler, 2, new_score
    )
    assert resumed.checkpoint_manager.best_path.is_file()
    saved = torch.load(resumed.checkpoint_manager.best_path, weights_only=False)
    assert saved["metric"] == new_score
    for key, value in resumed.model.state_dict().items():
        torch.testing.assert_close(saved["model"][key], value, rtol=0, atol=0)


def test_external_best_resume_copies_selected_artifact_into_fresh_run(tmp_path):
    source = _engine(tmp_path / "source")
    _step(source)
    source.checkpoint_manager.save_best(
        source.model, source.optimizer, source.scheduler, 1, 0.9
    )
    original_bytes = source.checkpoint_manager.best_path.read_bytes()
    resumed = _engine(tmp_path / "fresh destination")
    resume_semiwafernet_engine(resumed, source.checkpoint_manager.best_path)
    assert resumed.state.best_metric == resumed.checkpoint_manager.best_metric == 0.9
    assert resumed.checkpoint_manager.best_path.read_bytes() == original_bytes
    assert not resumed.checkpoint_manager.save_best(
        resumed.model, resumed.optimizer, resumed.scheduler, 2, 0.8
    )


def test_existing_destination_best_defines_available_historical_score(tmp_path):
    source = _engine(tmp_path / "source")
    source.state.best_metric = 0.99
    source.state.epoch = source.trainer.current_epoch = 1
    boundary = source.save(tmp_path / "source" / "ssl_stage1.pt")
    resumed = _engine(tmp_path / "destination")
    resumed.checkpoint_manager.save_best(
        resumed.model, resumed.optimizer, resumed.scheduler, 1, 0.8
    )
    digest = hashlib.sha256(resumed.checkpoint_manager.best_path.read_bytes()).hexdigest()
    resume_semiwafernet_engine(resumed, boundary)
    assert resumed.state.best_metric == resumed.checkpoint_manager.best_metric == 0.8
    assert hashlib.sha256(resumed.checkpoint_manager.best_path.read_bytes()).hexdigest() == digest
