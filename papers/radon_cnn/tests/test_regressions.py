"""Regression checks for actual WM-811K encoding and experiment persistence."""

import contextlib
import copy
import json
import sys

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from common.engine.config import EngineConfig
from common.training.checkpoint import CheckpointManager
from common.training.trainer import Trainer
from common.training.utils import NativeScaler
from papers.radon_cnn.data_utils.dataset import WaferRadonDataset, remove_background
from papers.radon_cnn.data_utils.protocol import load_manifest, make_manifest, model_state, subsets
from papers.radon_cnn.models.radon_cnn import RadonCNN, _ConvBlockKF


@pytest.fixture
def real_schema_dataset(tmp_path):
    (tmp_path / "images").mkdir()
    lines = ["filename,dieSize,lotName,waferIndex,trianTestLabel,failureType"]
    classes = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch"]
    for label, name in enumerate(classes):
        for i in range(12 + label * 3):
            fn = f"{label}_{i}.png"
            lines.append(f"{fn},1683,lot1,{i},Training,{name}")
            arr = np.ones((8, 8), dtype=np.uint8)
            arr[0] = 0
            arr[2, label + 1] = 2
            Image.fromarray(arr).save(tmp_path / "images" / fn)
    lines.extend(["none.png,3,lot2,1,Training,none", "unlabelled.png,3,lot2,2,Training,"])
    (tmp_path / "labels.csv").write_text("\n".join(lines))
    return WaferRadonDataset(tmp_path, balanced=False)


def test_actual_csv_schema_ignores_die_size(real_schema_dataset):
    d = real_schema_dataset
    assert len(d) == sum(12 + i * 3 for i in range(7))
    assert {label for _, label in d._samples} == set(range(7))
    assert all("none" not in fn for fn, _ in d._samples)
    assert d[0]["inputs"].shape == (1, 64, 64)
    assert torch.isfinite(d[0]["inputs"]).all()


@pytest.mark.parametrize("levels", [(0, 1, 2), (0, 127, 254), (0, 128, 255)])
def test_defect_mask_keeps_only_failed_dies(levels):
    arr = np.array([levels], dtype=np.uint8)
    assert np.array_equal(remove_background(arr), [[0, 0, 1]])


def test_resize_does_not_turn_good_dies_into_defects(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels.csv").write_text("filename,label\ngood.png,1\n")
    Image.fromarray(np.ones((9, 13), dtype=np.uint8)).save(tmp_path / "images/good.png")
    d = WaferRadonDataset(tmp_path, balanced=False)
    assert torch.count_nonzero(d[0]["inputs"]) == 0


def test_disjoint_balanced_splits_roundtrip(real_schema_dataset, tmp_path):
    d = real_schema_dataset
    config = EngineConfig.from_dict(
        {"data": {"protocol": "balanced", "train_size": 140, "allow_train_replacement": True}}
    )
    m = make_manifest(d, config, seed=17)
    assert len(m["indices"]["train"]) == 140
    assert len(set(m["indices"]["train"])) < 140
    assert m == make_manifest(d, config, seed=17)
    assert m["indices"] != make_manifest(d, config, seed=18)["indices"]
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(m["indices"][a]) & set(m["indices"][b])
    path = tmp_path / "split.json"
    path.write_text(json.dumps(m))
    restored = load_manifest(d, path)
    assert subsets(d, restored)[2].indices == m["indices"]["test"]
    restored["indices"]["train"].append(restored["indices"]["test"][0])
    path.write_text(json.dumps(restored))
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(d, path)


def test_refuse_implicit_oversampling(real_schema_dataset):
    config = EngineConfig.from_dict({"data": {"protocol": "balanced", "train_size": 140}})
    with pytest.raises(ValueError, match="replacement"):
        make_manifest(real_schema_dataset, config, 42)


def test_evaluation_loads_common_checkpoint(tmp_path):
    model = RadonCNN().eval()
    manager = CheckpointManager(tmp_path)
    manager.save_best(model, metric=1.0)
    checkpoint = torch.load(manager.best_path, weights_only=False)
    restored = RadonCNN().eval()
    restored.load_state_dict(model_state(checkpoint))
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x))


def test_kernel_maxout_follows_batchnorm_and_pooling():
    block = _ConvBlockKF(1, 1).eval()
    with torch.no_grad():
        block.bn.weight.fill_(-1)  # BN need not preserve max-out ordering.
        block.kernel_flip.conv.weight.copy_(torch.arange(9.0).reshape(1, 1, 3, 3))
        x = torch.arange(16.0).reshape(1, 1, 4, 4)
        branches = block.kernel_flip.branches(x)
        expected = torch.stack(
            [block.pool(block.bn(block.relu(branches[:, i]))) for i in range(2)], dim=1
        ).amax(dim=1)
        old_order = block.pool(block.bn(block.relu(branches.amax(dim=1))))
        assert not torch.allclose(expected, old_order)
        torch.testing.assert_close(block(x), expected)


def test_evaluation_entrypoint_uses_saved_test_split(
    real_schema_dataset, tmp_path, monkeypatch, capsys
):
    from papers.radon_cnn.evaluate import main

    d = real_schema_dataset
    config = EngineConfig.from_dict({"data": {"data_root": str(d.data_root)}})
    manifest = make_manifest(d, config, 42)
    run = tmp_path / "run"
    manager = CheckpointManager(run)
    model = RadonCNN().eval()
    manager.save_best(model, metric=1.0)
    (run / "experiment.yaml").write_text(yaml.safe_dump(config.to_dict()))
    (run / "split.json").write_text(json.dumps(manifest))
    samples = [d[i] for i in manifest["indices"]["test"]]
    with torch.no_grad():
        logits = model(torch.stack([s["inputs"] for s in samples]))
        expected = float(
            (logits.argmax(1) == torch.tensor([s["targets"] for s in samples])).float().mean()
        )
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--checkpoint", str(manager.best_path), "--device", "cpu"]
    )
    main()
    output = capsys.readouterr().out
    assert f"Test samples: {len(samples)}" in output
    assert f"Accuracy:  {expected:.4f}" in output


def test_amp_unscaled_before_clipping_on_cpu():
    # CPU GradScaler exercises real scale/unscale logic without a CUDA device.
    class CPUScaler(NativeScaler):
        def __init__(self):
            self.enabled = True
            self._scaler = torch.amp.GradScaler("cpu", init_scale=1024)

        def autocast(self):
            return contextlib.nullcontext()

    model = torch.nn.Linear(2, 2)
    reference = copy.deepcopy(model)
    loader = DataLoader(
        TensorDataset(torch.tensor([[100.0, -100.0], [-100.0, 100.0]]), torch.tensor([0, 1])),
        batch_size=2,
    )
    for current, scaler in [(model, CPUScaler()), (reference, NativeScaler(enabled=False))]:
        trainer = Trainer(
            current,
            torch.optim.SGD(current.parameters(), lr=0.1),
            torch.nn.CrossEntropyLoss(),
            device="cpu",
            scaler=scaler,
            grad_max_norm=1.0,
            verbose=False,
        )
        trainer.train_one_epoch(loader)
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b)
