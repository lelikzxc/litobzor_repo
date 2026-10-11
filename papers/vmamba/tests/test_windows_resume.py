"""CPU checkpoint continuations using the actual FCS-VMamba model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from common.engine.config import EngineConfig
from common.engine.engine import Engine
from papers.reproduction import save_protocol
from papers.vmamba.models.vmamba import FCSVMamba
from papers.vmamba.train import configure_ssm_optimizer
from papers.vmamba.utils.checkpoint import (
    checkpoint_model_state,
    resume_vmamba_engine,
    validate_resume_protocol,
)


def _engine(directory: Path, mode: str = "max") -> Engine:
    config = EngineConfig.from_dict({
        "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.05},
        "scheduler": {"name": "cosine", "kwargs": {"T_max": 50}},
        "checkpoint": {
            "save_dir": str(directory),
            "metric_name": "val_accuracy" if mode == "max" else "val_loss",
            "mode": mode,
        },
        "training": {"amp": False},
        "loss": {"name": "cross_entropy"},
        "metrics": [],
    })
    model = FCSVMamba(
        image_size=32, embed_dim=8, depths=[1, 1, 1, 1],
        num_heads=[1, 1, 1, 1], ssm_ratio=1.0, d_state=2,
        drop_path_rate=0.0, num_classes=3, clca_num_heads=2,
    )
    engine = Engine(model, config, device="cpu")
    configure_ssm_optimizer(engine, config)
    engine.trainer.verbose = False
    return engine


def _step(engine: Engine) -> None:
    engine.model.train()
    inputs = torch.linspace(-1.0, 1.0, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    targets = torch.tensor([0, 2])
    engine.optimizer.zero_grad(set_to_none=True)
    loss = engine.loss_fn(engine.model(inputs), targets)
    assert torch.isfinite(loss)
    loss.backward()
    engine.optimizer.step()
    engine.scheduler.step()


def test_resume_actual_fcs_matches_next_adamw_and_cosine_step(tmp_path):
    original = _engine(tmp_path)
    _step(original)
    original.state.epoch = original.trainer.current_epoch = 7
    original.state.global_step = 19
    original.state.extra = {"run": "windows cuda"}
    original.checkpoint_manager.save_best(
        original.model, original.optimizer, original.scheduler, epoch=7, metric=0.9
    )
    path = original.save()
    sidecar = path.with_suffix(".state.json")
    payload = json.loads(sidecar.read_text())
    payload["epoch"] = 6
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    resumed = _engine(tmp_path)
    assert resume_vmamba_engine(resumed, path) == 7
    assert resumed.state.epoch == resumed.trainer.current_epoch == 7
    assert resumed.state.global_step == 19
    assert resumed.state.extra == {"run": "windows cuda"}
    assert resumed.state.checkpoint_path == path
    assert resumed.checkpoint_manager.best_metric == resumed.state.best_metric == 0.9
    assert len(resumed.optimizer.param_groups) == 2
    assert resumed.optimizer.param_groups[1]["weight_decay"] == 0.0
    for parameter, value in resumed.optimizer.state.items():
        assert value["exp_avg"].device == parameter.device
        assert value["exp_avg_sq"].device == parameter.device

    _step(original)
    _step(resumed)
    for expected, actual in zip(original.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert original.scheduler.state_dict() == resumed.scheduler.state_dict()


@pytest.mark.parametrize("mode,best,worse", [("max", 0.9, 0.8), ("min", 0.2, 0.3)])
def test_resume_last_preserves_best_weights_and_ignores_its_loss(tmp_path, mode, best, worse):
    original = _engine(tmp_path, mode)
    manager = original.checkpoint_manager
    manager.save_best(original.model, original.optimizer, original.scheduler, 1, best)
    before = manager.best_path.read_bytes()
    # Shared Trainer.fit records loss here, even with metric_name=val_accuracy.
    manager.save_last(original.model, original.optimizer, original.scheduler, 2, 3.5)
    resumed = _engine(tmp_path, mode)
    assert resume_vmamba_engine(resumed, manager.last_path) == 2
    assert resumed.checkpoint_manager.best_metric == best
    assert not resumed.checkpoint_manager.save_best(
        resumed.model, resumed.optimizer, resumed.scheduler, 3, worse
    )
    assert manager.best_path.read_bytes() == before


def test_external_selected_best_is_available_for_final_evaluation(tmp_path):
    original = _engine(tmp_path / "source")
    original.checkpoint_manager.save_best(
        original.model, original.optimizer, original.scheduler, 4, 0.95
    )
    selected = original.checkpoint_manager.best_path
    resumed = _engine(tmp_path / "destination with spaces")
    assert resume_vmamba_engine(resumed, selected) == 4
    assert resumed.checkpoint_manager.best_path.read_bytes() == selected.read_bytes()
    assert resumed.checkpoint_manager.best_metric == 0.95
    assert not resumed.checkpoint_manager.save_best(
        resumed.model, resumed.optimizer, resumed.scheduler, 5, 0.8
    )


def test_external_last_cannot_keep_a_score_without_selected_weights(tmp_path):
    original = _engine(tmp_path / "source")
    original.state.best_metric = 0.99
    original.trainer.current_epoch = original.state.epoch = 4
    path = original.save()
    resumed = _engine(tmp_path / "destination")
    resume_vmamba_engine(resumed, path)
    assert resumed.state.best_metric is None
    assert not resumed.checkpoint_manager.best_path.exists()
    assert resumed.checkpoint_manager.save_best(
        resumed.model, resumed.optimizer, resumed.scheduler, 5, 0.8
    )


def test_model_state_container_supports_legacy_format_without_remapping_keys(tmp_path):
    original = _engine(tmp_path)
    path = original.save()
    payload = torch.load(path, weights_only=False)
    payload["model_state_dict"] = payload.pop("model")
    torch.save(payload, path)
    resumed = _engine(tmp_path)
    resume_vmamba_engine(resumed, path)
    for expected, actual in zip(original.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert checkpoint_model_state(original.model.state_dict()) is not None


def test_weights_only_file_is_evaluable_but_not_resumable(tmp_path):
    engine = _engine(tmp_path)
    path = tmp_path / "weights.pt"
    torch.save(engine.model.state_dict(), path)
    assert set(checkpoint_model_state(torch.load(path, weights_only=False))) == set(engine.model.state_dict())
    with pytest.raises(ValueError, match="weights-only.*optimizer state"):
        resume_vmamba_engine(engine, path)


@pytest.mark.parametrize("bad_epoch", [-1, 1.5, True])
def test_malformed_epoch_is_rejected_before_model_load(tmp_path, bad_epoch):
    engine = _engine(tmp_path)
    path = engine.save()
    payload = torch.load(path, weights_only=False)
    payload["epoch"] = bad_epoch
    torch.save(payload, path)
    with pytest.raises(ValueError, match="nonnegative integer"):
        resume_vmamba_engine(engine, path)


def test_resume_does_not_accept_missing_cosine_scheduler_state(tmp_path):
    engine = _engine(tmp_path)
    path = engine.save()
    payload = torch.load(path, weights_only=False)
    del payload["scheduler"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="scheduler state is missing"):
        resume_vmamba_engine(engine, path)


def test_incompatible_previous_architecture_is_not_silently_remapped(tmp_path):
    engine = _engine(tmp_path)
    path = engine.save()
    payload = torch.load(path, weights_only=False)
    payload["model"]["classifier.weight"] = payload["model"].pop("head.weight")
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key.*head.weight|Unexpected key.*classifier.weight"):
        resume_vmamba_engine(engine, path)


def _protocol_fixture(directory):
    config = EngineConfig.from_dict({
        "model": {"name": "fcs_vmamba", "num_classes": 9, "depths": (2, 2, 2, 2)},
        "data": {"source": "author_archive", "image_size": 224, "augmentation": True},
        "training": {"num_epochs": 50, "batch_size": 4},
        "optimizer": {"name": "adamw", "lr": 0.001},
        "checkpoint": {"save_dir": str(directory), "metric_name": "val_accuracy", "mode": "max"},
        "logging": {"level": "INFO"},
    })
    samples = [("Center/a.jpg", 0), ("Center/b.jpg", 0), ("Edge/c.jpg", 1)]
    splits = {"train": [0, 2], "holdout": [1]}
    save_protocol(directory, config, samples, splits, seed=42)
    checkpoint = directory / "last.pt"
    checkpoint.write_bytes(b"Protocol validation precedes loading the checkpoint.")
    return checkpoint, config, samples, splits


def test_resume_protocol_accepts_new_output_directory_and_json_normalization(tmp_path):
    path, config, samples, splits = _protocol_fixture(tmp_path / "source")
    config.merge_deep({"checkpoint": {"save_dir": str(tmp_path / "new output")}, "logging": {"level": "DEBUG"}})
    assert validate_resume_protocol(path, config, samples, splits, 42) == path.parent / "protocol.json"


@pytest.mark.parametrize("change,error", [
    ({"model": {"num_classes": 8}}, "config.model.num_classes"),
    ({"data": {"augmentation": False}}, "config.data.augmentation"),
    ({"training": {"batch_size": 8}}, "config.training.batch_size"),
    ({"optimizer": {"lr": 0.01}}, "config.optimizer.lr"),
])
def test_resume_protocol_rejects_changed_scientific_recipe(tmp_path, change, error):
    path, config, samples, splits = _protocol_fixture(tmp_path)
    config.merge_deep(change)
    with pytest.raises(ValueError, match=error):
        validate_resume_protocol(path, config, samples, splits, 42)


def test_resume_protocol_rejects_changed_sample_order_and_holdout(tmp_path):
    path, config, samples, splits = _protocol_fixture(tmp_path)
    with pytest.raises(ValueError, match="sample_digest"):
        validate_resume_protocol(path, config, samples[::-1], splits, 42)
    with pytest.raises(ValueError, match="splits.holdout"):
        validate_resume_protocol(path, config, samples, {**splits, "holdout": [0]}, 42)
    with pytest.raises(ValueError, match="seed"):
        validate_resume_protocol(path, config, samples, splits, 43)


def test_resume_protocol_requires_original_manifest(tmp_path):
    path = tmp_path / "last.pt"
    path.write_bytes(b"checkpoint")
    with pytest.raises(FileNotFoundError, match="original training protocol"):
        validate_resume_protocol(path, {}, [], {}, 42)
