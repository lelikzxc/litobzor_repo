"""Exercise entry-point data preparation without loading images or training."""

from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from papers.semiwafernet import train
from papers.semiwafernet.data_utils.wafer_dataset import WM811K_CLASSES


class _DataPrepared(Exception):
    """Stop before SSL when every seeded data operation has completed."""


@pytest.mark.parametrize("official_split", [True, False])
def test_config_seed_controls_all_classification_data(tmp_path, monkeypatch, official_split):
    root = tmp_path / "wm811k"
    root.mkdir()
    rows = ["filename,failureType,trianTestLabel"]
    for label in WM811K_CLASSES:
        rows.extend(f"{label}_{i}.png,{label},Training" for i in range(100))
        rows.extend(f"{label}_test_{i}.png,{label},Test" for i in range(20))
    rows.extend(f"unlabeled_{i}.png,," for i in range(600))
    (root / "labels.csv").write_text("\n".join(rows) + "\n")

    original_loader = train.DataLoader
    original_unlabeled = train.UnlabeledWM811KDataset
    original_hybrid = train.apply_hybrid_sampling

    def prepare(seed):
        config = {
            "model": {"mode": "classification", "num_classes": 9},
            "training": {"batch_size": 8, "num_epochs": 50},
            "checkpoint": {"save_dir": str(tmp_path / "checkpoints")},
            "data": {
                "data_root": str(root),
                "use_official_split": official_split,
                "pseudo_eval_fraction": 0.05,
                "hybrid_sampling": {"enabled": True, "none_downsample_ratio": 0.3},
            },
            "semi_supervised": {"enabled": True, "unlabeled_max_samples": 500},
        }
        if seed is not None:
            config["seed"] = seed
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result = {"eval_samples": []}

        class _Oversampled(torch.utils.data.Dataset):
            def __init__(self, **kwargs):
                result["smote_seed"] = kwargs["seed"]
                result["train_samples"] = kwargs["samples"]
                self._y = np.array([label for _, label in kwargs["samples"]])

            def __len__(self):
                return len(self._y)

            def __getitem__(self, index):
                raise AssertionError("Data preparation should not load image tensors")

        def record_loader(dataset, **kwargs):
            if not kwargs["shuffle"]:
                indices = dataset.indices if isinstance(dataset, train.Subset) else range(len(dataset))
                source = dataset.dataset if isinstance(dataset, train.Subset) else dataset
                result["eval_samples"].append([source._samples[int(i)] for i in indices])
            return original_loader(dataset, **kwargs)

        def record_hybrid(samples, **kwargs):
            result["hybrid_seed"] = kwargs["seed"]
            return original_hybrid(samples, **kwargs)

        def record_protocol(directory, config, samples, splits, seed):
            result["protocol_seed"] = seed
            result["pseudo_eval"] = splits["pseudo_eval"]

        def prepare_unlabeled(**kwargs):
            dataset = original_unlabeled(**kwargs)
            result["unlabeled_seed"] = kwargs["seed"]
            result["unlabeled_samples"] = dataset._filenames
            raise _DataPrepared

        with monkeypatch.context() as patches:
            patches.setattr(train, "parse_args", lambda: Namespace(
                config=str(config_path), device="cpu", epochs=None, batch_size=None,
                lr=None, mode=None, ssl_fast=False, no_ssl=False, resume=None,
                ssl_start_stage=1, data_fraction=0.5,
            ))
            patches.setattr(train, "SMOTEDataset", _Oversampled)
            patches.setattr(train, "DataLoader", record_loader)
            patches.setattr(train, "apply_hybrid_sampling", record_hybrid)
            patches.setattr(train, "save_protocol", record_protocol)
            patches.setattr(train.SemiWaferNet, "from_config", lambda config: torch.nn.Linear(1, 9))
            patches.setattr(train, "Engine", lambda **kwargs: SimpleNamespace(trainer=SimpleNamespace()))
            patches.setattr(train, "UnlabeledWM811KDataset", prepare_unlabeled)
            with pytest.raises(_DataPrepared):
                train.main()
        return result

    default = prepare(None)
    assert default == prepare(42)
    changed = prepare(137)
    assert changed == prepare(137)
    for key in ("hybrid_seed", "smote_seed", "unlabeled_seed", "protocol_seed"):
        assert default[key] == 42
        assert changed[key] == 137
    assert default["train_samples"] != changed["train_samples"]
    assert default["unlabeled_samples"] != changed["unlabeled_samples"]
    assert all(old != new for old, new in zip(default["eval_samples"], changed["eval_samples"], strict=True))
    if official_split:
        assert default["pseudo_eval"] != changed["pseudo_eval"]
