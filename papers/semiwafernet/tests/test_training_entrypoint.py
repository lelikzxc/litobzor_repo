"""Exercise the actual file-based training/evaluation and restart entry points.

Tiny artificial categorical maps test execution and provenance, not accuracy.
The classifier is the complete default 788,169-parameter network. Permissive
pseudo-label gates are fixture-only and force the lazy Dl/pseudo union path.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from common.engine.config import EngineConfig
from common.training.checkpoint import CheckpointManager
from papers.semiwafernet import evaluate, train
from papers.semiwafernet.data_utils.wafer_dataset import WM811K_CLASSES
from papers.semiwafernet.training.console import training_output


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def wafer_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("official wafer maps")
    images = root / "images"
    images.mkdir()
    rows = []
    random = np.random.RandomState(123)
    yy, xx = np.mgrid[:32, :32]
    active = (yy - 15.5) ** 2 + (xx - 15.5) ** 2 < 15 ** 2
    for label, name in enumerate(WM811K_CLASSES):
        for partition, count in (("Training", 30), ("Test", 4)):
            for index in range(count):
                filename = f"{partition}, class {label} map {index} ä.png"
                die = active.astype(np.uint8)
                defect = (random.rand(32, 32) < (0.02 + label * 0.015)) & active
                die[defect] = 2
                Image.fromarray(die).save(images / filename)
                rows.append((filename, partition, name))
    for index in range(16):
        filename = f"unlabeled {index}.png"
        die = active.astype(np.uint8)
        die[(random.rand(32, 32) < 0.1) & active] = 2
        Image.fromarray(die).save(images / filename)
        rows.append((filename, "", ""))
    # Exercise a Windows-style BOM/CRLF CSV, quoted commas and Unicode paths.
    with (root / "labels.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "trianTestLabel", "failureType"))
        writer.writerows(rows)
    return root


def _config(wafer_files, directory, *, ssl=True):
    source = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(source.read_text())
    config["data"]["data_root"] = str(wafer_files)
    config["data"]["augmentation"]["enabled"] = False
    config["training"].update(num_epochs=3, batch_size=16)
    config["evaluation"]["batch_size"] = 16
    config["checkpoint"]["save_dir"] = str(directory)
    config["semi_supervised"].update(
        enabled=ssl, epochs_per_stage=[1, 1, 1], unlabeled_max_samples=16,
        mc_passes=2, confidence_threshold=0.0, alpha=0.0, beta=0.0,
        entropy_threshold=10.0, mutual_information_threshold=10.0,
    )
    path = directory.parent / f"{directory.name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(config_path, *extra):
    stdout, stderr = io.StringIO(), io.StringIO()
    args = train.parse_args(["--config", str(config_path), "--device", "cpu", *map(str, extra)])
    log = Path(config_path).with_suffix(".log")
    engines = []
    original_engine = train.Engine

    class ObservedEngine(original_engine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.validation_calls = 0
            engines.append(self)

        def validate(self, loader):
            self.validation_calls += 1
            return super().validate(loader)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "Engine", ObservedEngine)
        with redirect_stdout(stdout), redirect_stderr(stderr), training_output(log):
            train._run_training(args)
    output = stdout.getvalue() + stderr.getvalue()
    assert "\r" not in output
    assert not re.search(r"\d+%\|", output)
    assert "\r" not in log.read_text(encoding="utf-8")
    assert not re.search(r"\d+%\|", log.read_text(encoding="utf-8"))
    return engines[0], output


@pytest.fixture(scope="module")
def completed_ssl(wafer_files, tmp_path_factory):
    directory = tmp_path_factory.mktemp("ssl run") / "checkpoints"
    config_path = _config(wafer_files, directory)
    engine, output = _run(config_path)
    return directory, config_path, engine, output


def test_actual_three_stage_entrypoint_and_one_validation_per_epoch(completed_ssl):
    directory, _, engine, output = completed_ssl
    assert sum(parameter.numel() for parameter in engine.model.parameters()) == 788_169
    assert engine.trainer.current_epoch == engine.state.epoch == 3
    # Three epoch validations and the explicit final post-SSL validation.
    assert engine.validation_calls == 4
    epoch_history = [row for row in engine.logger.history if "ssl_stage" in row]
    assert [row["ssl_stage"] for row in epoch_history] == [1, 2, 3]
    assert all("val_f1" in row and "val_loss" in row for row in epoch_history)
    for name, epoch in (("ssl_stage1.pt", 1), ("ssl_stage2.pt", 2), ("last.pt", 3)):
        checkpoint = torch.load(directory / name, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == epoch
        assert "optimizer" in checkpoint
        assert json.loads((directory / name).with_suffix(".state.json").read_text())["epoch"] == epoch
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    assert best["metric_name"] == "val_f1"
    assert best["metric"] == engine.checkpoint_manager.best_metric
    assert output.count("Pseudo-set ready: 16/16") == 2
    assert "Test Results:" in output
    assert re.search(r"Train Loss:\s+\d+\.\d+", output)
    assert re.search(r"Val Loss:\s+\d+\.\d+", output)


@pytest.mark.parametrize("start_stage", [2, 3])
def test_actual_ssl_boundary_restart(completed_ssl, start_stage):
    directory, config_path, _, _ = completed_ssl
    previous = directory / f"ssl_stage{start_stage - 1}.pt"
    digest = hashlib.sha256(previous.read_bytes()).hexdigest()
    protocol_digest = hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest()
    engine, output = _run(config_path, "--resume", previous, "--ssl-start-stage", start_stage)
    assert engine.trainer.current_epoch == engine.state.epoch == 3
    assert engine.validation_calls == (4 - start_stage) + 1
    assert [row["ssl_stage"] for row in engine.logger.history if "ssl_stage" in row] == list(range(start_stage, 4))
    assert hashlib.sha256(previous.read_bytes()).hexdigest() == digest
    assert hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest() == protocol_digest
    assert torch.load(directory / "last.pt", map_location="cpu", weights_only=False)["epoch"] == 3
    assert "Skipping Stage 1" in output


def test_standalone_evaluate_saved_best_and_exact_test_manifest(completed_ssl):
    directory, _, _, _ = completed_ssl
    stdout, stderr = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", str(directory / "best.pt"), "--device", "cpu"])
        with redirect_stdout(stdout), redirect_stderr(stderr):
            evaluate.main()
    output = stdout.getvalue()
    assert "Samples: 36;" in output
    assert all(f"{metric}:" in output for metric in ("accuracy", "f1", "precision", "recall"))
    assert "\r" not in stderr.getvalue()
    saved = json.loads((directory / "protocol.json").read_text())
    assert saved["splits"]["test"] == list(range(36))
    assert all("Training" in pair[0] for pair in saved["splits"]["train_samples"])


def test_supervised_resume_trains_only_remaining_epochs(wafer_files, tmp_path):
    directory = tmp_path / "supervised"
    config_path = _config(wafer_files, directory, ssl=False)
    first_epoch = directory / "interrupted.pt"
    original_save = CheckpointManager.save_last

    def preserve_first_epoch(manager, *args, **kwargs):
        original_save(manager, *args, **kwargs)
        if kwargs.get("epoch") == 1 and manager.save_dir == directory:
            shutil.copy2(manager.last_path, first_epoch)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CheckpointManager, "save_last", preserve_first_epoch)
        original, _ = _run(config_path)
    assert original.trainer.current_epoch == 3
    assert torch.load(first_epoch, map_location="cpu", weights_only=False)["epoch"] == 1
    resumed, output = _run(config_path, "--resume", first_epoch)
    assert resumed.trainer.current_epoch == resumed.state.epoch == 3
    assert len(resumed.logger.history) == 2
    assert "Training 2 remaining epochs (target total: 3)" in output
    completed, output = _run(config_path, "--resume", "last")
    assert completed.trainer.current_epoch == 3
    assert completed.logger.history == []
    assert "already completed" in output


def test_real_du_preparation_is_deterministic_and_unaugmented(wafer_files, tmp_path):
    config_path = _config(wafer_files, tmp_path / "unused")
    config = EngineConfig.from_yaml(config_path)
    args = train.parse_args(["--config", str(config_path), "--device", "cpu"])
    dataset = train._prepare_unlabeled_dataset(
        config, args, str(wafer_files), 32, 123, {"has_failure_type": True}
    )
    assert dataset.train is False
    assert len(dataset) == 16
    for index in range(len(dataset)):
        torch.testing.assert_close(dataset[index]["image"], dataset[index]["image"], rtol=0, atol=0)


@pytest.mark.parametrize("field,value,match", [
    ("training.batch_size", 0, "batch_size"),
    ("optimizer.lr", float("nan"), "optimizer.lr"),
    ("model.num_classes", 8, "num_classes"),
    ("semi_supervised.mc_passes", 1, "mc_passes"),
])
def test_invalid_configuration_fails_before_loading_dataset(tmp_path, field, value, match):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text())
    section, key = field.split(".")
    config[section][key] = value
    config["data"]["data_root"] = str(tmp_path / "missing dataset")
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    args = train.parse_args(["--config", str(path), "--device", "cpu", "--save-dir", str(tmp_path / "run")])
    with pytest.raises(ValueError, match=match):
        train._run_training(args)
    assert not (tmp_path / "run").exists()


def test_existing_run_and_wrong_stage_restart_fail_early(completed_ssl):
    directory, path, _, _ = completed_ssl
    config = EngineConfig.from_yaml(path)
    args = train.parse_args(["--config", str(path), "--device", "cpu"])
    with pytest.raises(ValueError, match="already exist"):
        train.resume_path(config, args)
    args.resume = str(directory / "last.pt")
    args.ssl_start_stage = 2
    with pytest.raises(ValueError, match="ssl_stage1.pt"):
        train.resume_path(config, args)
    with pytest.raises(ValueError, match="epoch 1"):
        train.validate_boundary_epoch(2, 2, (1, 1, 1))


def test_changed_restart_protocol_fails_before_smote(completed_ssl, tmp_path, monkeypatch):
    directory, config_path, _, _ = completed_ssl
    config = yaml.safe_load(config_path.read_text())
    config["optimizer"]["lr"] *= 2
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(config))
    protocol = directory / "protocol.json"
    before = protocol.read_bytes()

    def forbid_smote(*args, **kwargs):
        raise AssertionError("Changed protocol must be rejected before SMOTE")

    monkeypatch.setattr(train, "SMOTEDataset", forbid_smote)
    args = train.parse_args([
        "--config", str(changed), "--device", "cpu", "--resume",
        str(directory / "ssl_stage1.pt"), "--ssl-start-stage", "2",
    ])
    with pytest.raises(ValueError, match=r"config\.optimizer\.lr"):
        train._run_training(args)
    assert protocol.read_bytes() == before


def test_missing_selected_du_image_fails_before_smote_and_protocol(
    wafer_files, tmp_path, monkeypatch
):
    damaged = tmp_path / "damaged dataset"
    shutil.copytree(wafer_files, damaged)
    (damaged / "images" / "unlabeled 0.png").unlink()
    directory = tmp_path / "run"
    config_path = _config(damaged, directory)

    def forbid_smote(*args, **kwargs):
        raise AssertionError("Missing image must be rejected before SMOTE")

    monkeypatch.setattr(train, "SMOTEDataset", forbid_smote)
    args = train.parse_args(["--config", str(config_path), "--device", "cpu"])
    with pytest.raises(FileNotFoundError, match="unlabeled 0.png"):
        train._run_training(args)
    assert not directory.exists()

