"""Evaluation must reconstruct the saved inference recipe, not today's YAML."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.engine.config import EngineConfig
from papers.vmamba.evaluate import checkpoint_config, parse_args


ROOT = Path(__file__).resolve().parents[3]


def _saved_run(directory: Path):
    baseline = EngineConfig.from_yaml(ROOT / "papers/vmamba/configs/config.yaml")
    effective = EngineConfig.from_dict(baseline.to_dict())
    effective.merge_deep({
        "seed": 7,
        "data": {"augmentation_enabled": False, "train_subset": 0.5},
        "training": {"num_epochs": 12, "batch_size": 2},
        "checkpoint": {"save_dir": str(directory / "CUDA results")},
    })
    checkpoint = directory / "best.pt"
    checkpoint.write_bytes(b"Configuration is validated before checkpoint tensors are loaded.")
    (directory / "protocol.json").write_text(
        json.dumps({"config": effective.to_dict()}), encoding="utf-8",
    )
    current_yaml = directory / "current configuration.yaml"
    baseline.to_yaml(current_yaml)
    return checkpoint, effective, baseline, current_yaml


def test_default_evaluation_uses_effective_saved_config_after_yaml_changes(tmp_path):
    checkpoint, effective, current, current_yaml = _saved_run(tmp_path)
    current.merge_deep({
        "model": {"num_classes": 8},
        "data": {"source": "raw_maps", "image_size": 64, "mean": [0, 0, 0]},
        "seed": 99,
    })
    current.to_yaml(current_yaml)
    args = parse_args(["--checkpoint", str(checkpoint), "--device", "cpu"])
    assert args.config is None
    loaded = checkpoint_config(args)
    assert loaded.to_dict() == effective.to_dict()
    assert loaded.get("model.num_classes") == 9
    assert loaded.get("data.image_size") == 224
    assert loaded.get("seed") == 7
    assert loaded.get("data.augmentation_enabled") is False


def test_optional_matching_yaml_accepts_training_only_differences_and_keeps_saved_recipe(tmp_path):
    checkpoint, effective, current, current_yaml = _saved_run(tmp_path)
    # These fields differ naturally between a base YAML and the run's effective
    # CLI overrides. They do not change deterministic inference pixels/weights.
    current.merge_deep({
        "seed": 101,
        "data": {"augmentation_enabled": True, "train_subset": 1.0},
        "training": {"num_epochs": 100, "batch_size": 16},
        "optimizer": {"lr": 0.01},
        "checkpoint": {"save_dir": str(tmp_path / "different training output")},
    })
    current.to_yaml(current_yaml)
    loaded = checkpoint_config(parse_args([
        "--checkpoint", str(checkpoint), "--config", str(current_yaml), "--device", "cpu",
    ]))
    assert loaded.to_dict() == effective.to_dict()


@pytest.mark.parametrize("change,message", [
    ({"model": {"num_classes": 8}}, "Model config differs"),
    ({"data": {"source": "raw_maps"}}, "data.source differs"),
    ({"data": {"image_size": 128}}, "data.image_size differs"),
    ({"data": {"mean": [0.0, 0.0, 0.0]}}, "data.mean differs"),
])
def test_optional_yaml_rejects_changes_that_would_change_predictions(tmp_path, change, message):
    checkpoint, _, current, current_yaml = _saved_run(tmp_path)
    current.merge_deep(change)
    current.to_yaml(current_yaml)
    with pytest.raises(ValueError, match=message):
        checkpoint_config(parse_args([
            "--checkpoint", str(checkpoint), "--config", str(current_yaml), "--device", "cpu",
        ]))


@pytest.mark.parametrize("missing,message", [
    ("checkpoint", "Checkpoint not found"),
    ("protocol", "evaluation requires the training split"),
])
def test_missing_training_artifacts_fail_before_any_model_or_dataset_load(tmp_path, missing, message):
    checkpoint, _, _, _ = _saved_run(tmp_path)
    (checkpoint if missing == "checkpoint" else tmp_path / "protocol.json").unlink()
    with pytest.raises(FileNotFoundError, match=message):
        checkpoint_config(parse_args(["--checkpoint", str(checkpoint), "--device", "cpu"]))
