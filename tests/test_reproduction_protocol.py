"""Repeated runs must compare JSON protocols without tuple/list false mismatches."""

import copy
import json
from argparse import Namespace

import numpy as np
import pytest
import yaml

from common.engine.config import EngineConfig
from papers.reproduction import load_holdout, save_protocol


def protocol_inputs():
    config = EngineConfig({"model": {"depths": [1, 1]}, "seed": 42})
    samples = [("test0.png", 0), ("test1.png", 1)]
    splits = {
        "test": [0, 1],
        "train_samples": [("train0.png", 0), ("train1.png", 1)],
        "pseudo_eval": [("pe0.png", 1)],
    }
    return config, samples, splits


def test_same_semiwafernet_protocol_can_be_saved_twice(tmp_path):
    config, samples, splits = protocol_inputs()
    path = save_protocol(tmp_path, config, samples, splits, 42)
    saved = json.loads(path.read_text())
    assert saved["splits"]["train_samples"][0] == ["train0.png", 0]
    assert isinstance(splits["train_samples"][0], tuple)
    assert save_protocol(tmp_path, config, samples, splits, 42) == path
    indices, saved_config = load_holdout(tmp_path / "ssl_stage1.pt", samples, "test")
    assert indices == [0, 1] and saved_config == config.to_dict()


def test_matching_protocol_preserves_original_metadata(tmp_path):
    config, samples, splits = protocol_inputs()
    path = save_protocol(tmp_path, config, samples, splits, 42)
    saved = json.loads(path.read_text())
    saved["torch_version"] = "first-run-version"
    path.write_text(json.dumps(saved, sort_keys=True))
    before = path.read_bytes()
    save_protocol(tmp_path, config, samples, splits, 42)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change, field", [
    ("config", "config.model.depths"),
    ("samples", "sample_digest"),
    ("train", "splits.train_samples"),
    ("pseudo_eval", "splits.pseudo_eval"),
    ("test", "splits.test"),
    ("seed", "seed"),
])
def test_real_protocol_changes_still_rejected(tmp_path, change, field):
    config, samples, splits = protocol_inputs()
    path = save_protocol(tmp_path, config, samples, splits, 42)
    before = path.read_bytes()
    config = EngineConfig(copy.deepcopy(config.to_dict()))
    splits = copy.deepcopy(splits)
    seed = 42
    if change == "config":
        config._data["model"]["depths"][0] = 2
    elif change == "samples":
        samples = samples[::-1]
    elif change == "train":
        splits["train_samples"][0] = ("train0.png", 1)
    elif change == "pseudo_eval":
        splits["pseudo_eval"] = [("different.png", 1)]
    elif change == "test":
        splits["test"] = [1, 0]
    else:
        seed = 43
    with pytest.raises(ValueError, match=field):
        save_protocol(tmp_path, config, samples, splits, seed)
    assert path.read_bytes() == before


def test_semiwafernet_entrypoint_reuses_protocol_on_stage2_resume(tmp_path, monkeypatch):
    from papers.semiwafernet import train
    from papers.semiwafernet.data_utils.wafer_dataset import WM811K_CLASSES

    class DataPrepared(Exception):
        pass

    root = tmp_path / "wm811k"
    root.mkdir()
    rows = ["filename,failureType,trianTestLabel"]
    for label in WM811K_CLASSES:
        rows.extend(f"{label}_{i}.png,{label},Training" for i in range(60))
        rows.extend(f"{label}_test_{i}.png,{label},Test" for i in range(10))
    (root / "labels.csv").write_text("\n".join(rows))
    run = tmp_path / "checkpoints"
    config = {
        "seed": 42,
        "model": {"mode": "classification", "num_classes": 9},
        "training": {"batch_size": 8, "num_epochs": 50},
        "checkpoint": {"save_dir": str(run)},
        "data": {"data_root": str(root), "use_official_split": True,
                 "pseudo_eval_fraction": 0.05},
        "semi_supervised": {"enabled": True},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    class NoImageOversample:
        def __init__(self, **kwargs):
            self._y = np.array([label for _, label in kwargs["samples"]])

        def __len__(self):
            return len(self._y)

    def stop_after_protocol(*args, **kwargs):
        raise DataPrepared

    monkeypatch.setattr(train, "SMOTEDataset", NoImageOversample)
    monkeypatch.setattr(train, "DataLoader", stop_after_protocol)
    before = None
    for resume, stage in [(None, 1), (str(run / "ssl_stage1.pt"), 2)]:
        monkeypatch.setattr(train, "parse_args", lambda: Namespace(
            config=str(config_path), device="cpu", epochs=None, batch_size=None,
            lr=None, mode=None, ssl_fast=False, no_ssl=False, resume=resume,
            ssl_start_stage=stage, data_fraction=None,
        ))
        with pytest.raises(DataPrepared):
            train.main()
        current = (run / "protocol.json").read_bytes()
        if before is not None:
            assert current == before
        before = current
