"""Portable VMamba CLI, interrupted training and saved-holdout integration."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from common.engine.config import EngineConfig
from common.training.trainer import Trainer
from papers.vmamba import evaluate as vm_evaluate
from papers.vmamba import train as vm_train
from papers.vmamba.data_utils.author_dataset import FOLDER_LABELS
from papers.vmamba.kernels import csms6s


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def cpu_threads_and_backend(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(csms6s, "_SCAN_BACKEND", "auto")
    yield
    torch.set_num_threads(previous)


def _config_fixture(directory: Path, *, small_backbone: bool = False) -> Path:
    """Synthetic source; check-only keeps the complete default architecture."""
    archive_path = directory / "author data.zip"
    generator = np.random.default_rng(13)
    with zipfile.ZipFile(archive_path, "w") as archive:
        for folder, label in FOLDER_LABELS.items():
            for index in range(6):
                pixels = generator.integers(0, 255, (28, 29, 3), dtype=np.uint8)
                pixels[4:10, label:label + 6, label % 3] = 255
                buffer = io.BytesIO()
                Image.fromarray(pixels).save(buffer, format="JPEG")
                archive.writestr(f"WM811k_Dataset/{folder}/{index}.jpg", buffer.getvalue())

    config = EngineConfig.from_yaml(ROOT / "papers/vmamba/configs/config.yaml").to_dict()
    config["data"].update({
        "data_root": str(archive_path), "verify_author_archive": False, "image_size": 32,
    })
    config["model"]["input"]["image_size"] = 32
    if small_backbone:
        config["model"]["backbone"].update({
            "embed_dim": 16, "depths": [1, 1, 1, 1], "ssm_ratio": 1.0,
            "d_state": 4, "drop_path_rate": 0.0,
        })
    config["training"].update({"num_epochs": 2, "batch_size": 2, "amp": False})
    config["evaluation"]["batch_size"] = 4
    config["checkpoint"]["save_dir"] = str(directory / "unused default output")
    config_path = directory / "diagnostic config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _args(config_path: Path, *extra: str):
    return vm_train.parse_args(["--config", str(config_path), "--device", "cpu", *extra])


def _main(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["papers/vmamba/train.py", *args])
    vm_train.main()


class TrainingInterrupted(RuntimeError):
    pass


def test_full_fcs_cli_interrupt_resume_and_standalone_saved_holdout(
    tmp_path, monkeypatch, capsys
):
    config_path = _config_fixture(tmp_path, small_backbone=True)
    output_dir = tmp_path / "CUDA run with spaces"
    log_path = output_dir / "train.log"
    argv = [
        "--config", str(config_path), "--device", "cpu", "--save-dir", str(output_dir),
        "--log-file", str(log_path), "--no-aug", "--scan-backend", "chunked",
    ]
    original_train_epoch = Trainer.train_one_epoch

    def interrupt_before_second_epoch(self, loader):
        if self.current_epoch >= 2:
            raise TrainingInterrupted("Simulated interruption after epoch 1 checkpoint")
        return original_train_epoch(self, loader)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(Trainer, "train_one_epoch", interrupt_before_second_epoch)
        with pytest.raises(TrainingInterrupted, match="after epoch 1"):
            _main(interrupted, argv)

    first_capture = capsys.readouterr()
    assert "Epoch 1 |" in first_capture.out
    assert "Epoch 2 |" not in first_capture.out
    assert "\r" not in first_capture.out + first_capture.err
    assert "%|" not in first_capture.out + first_capture.err
    first_last = torch.load(output_dir / "last.pt", weights_only=False)
    assert first_last["epoch"] == 1
    first_best = (output_dir / "best.pt").read_bytes()
    protocol_bytes = (output_dir / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol["config"]["training"]["num_epochs"] == 2
    assert protocol["config"]["model"]["backbone"]["embed_dim"] == 16
    assert protocol["config"]["model"]["backbone"]["depths"] == [1, 1, 1, 1]
    assert protocol["config"]["data"]["augmentation_enabled"] is False
    assert protocol["config"]["checkpoint"]["save_dir"] == str(output_dir)
    assert len(protocol["splits"]["train"]) == 43
    assert len(protocol["splits"]["holdout"]) == 11
    assert not set(protocol["splits"]["train"]).intersection(protocol["splits"]["holdout"])
    assert "TrainingInterrupted" in log_path.read_text(encoding="utf-8")

    original_validate = Trainer.validate

    def no_better_selection_score(self, loader, desc="Val"):
        metrics = original_validate(self, loader, desc=desc)
        if desc != "Test":
            # Exercise a valid no-improvement continuation without depending
            # on recognition quality of random synthetic images.
            metrics["accuracy"] = 0.0
        return metrics

    with monkeypatch.context() as resumed:
        resumed.setattr(Trainer, "validate", no_better_selection_score)
        _main(resumed, [*argv, "--resume"])
    resumed_capture = capsys.readouterr()
    assert "Resumed at epoch 1" in resumed_capture.out
    assert "Starting training for 1 epochs" in resumed_capture.out
    assert "Epoch 2 |" in resumed_capture.out
    assert "Epoch 3 |" not in resumed_capture.out
    assert "\r" not in resumed_capture.out + resumed_capture.err
    assert "%|" not in resumed_capture.out + resumed_capture.err
    assert (output_dir / "best.pt").read_bytes() == first_best
    assert (output_dir / "protocol.json").read_bytes() == protocol_bytes
    last = torch.load(output_dir / "last.pt", weights_only=False)
    best = torch.load(output_dir / "best.pt", weights_only=False)
    assert last["epoch"] == 2
    assert best["epoch"] == 1
    assert last["scheduler"]["last_epoch"] == 2
    assert best["scheduler"]["last_epoch"] == 1
    assert any(not torch.equal(last["model"][key], best["model"][key]) for key in best["model"])
    assert any(not torch.equal(last["model"][key], first_last["model"][key]) for key in last["model"])
    assert "Epoch 2 |" in log_path.read_text(encoding="utf-8")
    assert "\r" not in log_path.read_text(encoding="utf-8")
    final_last_bytes = (output_dir / "last.pt").read_bytes()

    # Evaluation should reconstruct the effective training configuration from
    # protocol.json, including a CLI-selected output directory and image size.
    monkeypatch.setattr(sys, "argv", [
        "papers/vmamba/evaluate.py", "--checkpoint", str(output_dir / "best.pt"),
        "--device", "cpu", "--scan-backend", "chunked",
    ])
    vm_evaluate.main()
    evaluation = capsys.readouterr()
    assert "Saved holdout samples: 11" in evaluation.out
    assert "Holdout Results" in evaluation.out
    for metric in ("accuracy", "f1", "precision", "recall"):
        inline = re.search(rf"^\s+{metric}: ([0-9.]+)$", resumed_capture.out, re.MULTILINE)
        standalone = re.search(rf"^\s+{metric}:\s+([0-9.]+)$", evaluation.out, re.MULTILINE)
        assert inline and standalone
        assert inline.group(1) == standalone.group(1)
    assert (output_dir / "last.pt").read_bytes() == final_last_bytes


def test_check_only_full_cli_subprocess_does_not_create_experiment(tmp_path):
    config_path = _config_fixture(tmp_path)
    output_dir = tmp_path / "check output"
    env = {
        **os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
    }
    completed = subprocess.run(
        [sys.executable, "-u", str(ROOT / "papers/vmamba/train.py"),
         "--config", str(config_path), "--device", "cpu", "--save-dir", str(output_dir),
         "--check-only", "--no-aug", "--scan-backend", "chunked"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Training update OK: batch=2" in completed.stdout
    assert "Evaluation batch OK: batch=4" in completed.stdout
    assert "No checkpoints or training protocol were written" in completed.stdout
    assert "\r" not in completed.stdout + completed.stderr
    assert "%|" not in completed.stdout + completed.stderr
    assert not output_dir.exists()
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("protocol.json"))


@pytest.mark.parametrize("options,error", [
    (["--epochs", "0"], "training.num_epochs"),
    (["--batch-size", "0"], "training.batch_size"),
    (["--lr", "nan"], "optimizer.lr"),
    (["--per-class", "-1"], "nonnegative integer"),
    (["--subset", "0"], "train_subset"),
    (["--seed", "-1"], "seed must"),
    (["--check-only", "--resume"], "omit --resume"),
    (["--allow-slow-scan", "--scan-backend", "chunked"], "either --scan-backend"),
])
def test_invalid_cli_fails_before_dataset_or_model(tmp_path, monkeypatch, options, error):
    config_path = _config_fixture(tmp_path)
    monkeypatch.setattr(vm_train, "build_dataset", lambda *a, **kw: pytest.fail("Invalid CLI must fail before data"))
    with pytest.raises(ValueError, match=error):
        vm_train._run_training(_args(config_path, *options))


def test_missing_resume_checkpoint_and_existing_fresh_weights_fail_before_data(tmp_path, monkeypatch):
    config_path = _config_fixture(tmp_path)
    output_dir = tmp_path / "existing run"
    monkeypatch.setattr(vm_train, "build_dataset", lambda *a, **kw: pytest.fail("Guard must run before data"))
    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        vm_train._run_training(_args(config_path, "--save-dir", str(output_dir), "--resume"))
    output_dir.mkdir()
    (output_dir / "best.pt").write_bytes(b"existing checkpoint")
    with pytest.raises(ValueError, match="new --save-dir or --resume"):
        vm_train._run_training(_args(config_path, "--save-dir", str(output_dir)))


def test_effective_cli_protocol_records_flags_and_rejects_aug_change_on_resume(tmp_path, monkeypatch):
    config_path = _config_fixture(tmp_path)
    output_dir = tmp_path / "CLI selected output"

    class ModelNotAllocated(RuntimeError):
        pass

    def skip_model(config):
        raise ModelNotAllocated("Protocol was written before model allocation")

    monkeypatch.setattr(vm_train.FCSVMamba, "from_config", skip_model)
    flags = [
        "--save-dir", str(output_dir), "--no-aug", "--per-class", "5", "--subset", "0.5",
        "--seed", "7", "--lr", "0.002", "--batch-size", "3",
    ]
    with pytest.raises(ModelNotAllocated):
        vm_train._run_training(_args(config_path, *flags))
    protocol = json.loads((output_dir / "protocol.json").read_text())
    assert protocol["seed"] == protocol["config"]["seed"] == 7
    assert protocol["config"]["data"]["per_class"] == 5
    assert protocol["config"]["data"]["train_subset"] == 0.5
    assert protocol["config"]["data"]["augmentation_enabled"] is False
    assert protocol["config"]["training"]["batch_size"] == 3
    assert protocol["config"]["optimizer"]["lr"] == 0.002
    assert protocol["config"]["scheduler"]["kwargs"]["T_max"] == 50
    assert len(protocol["splits"]["train"]) == 18
    assert len(protocol["splits"]["holdout"]) == 9
    (output_dir / "last.pt").write_bytes(b"Must reject protocol before loading this file")
    with pytest.raises(ValueError, match="config.data.augmentation_enabled"):
        vm_train._run_training(_args(config_path, *[flag for flag in flags if flag != "--no-aug"], "--resume"))
