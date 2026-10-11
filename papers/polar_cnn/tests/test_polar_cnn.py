"""Geometry, leakage, backend and checkpoint regression checks."""

import csv
import json
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from papers.polar_cnn.data_utils import CLASSES, PolarDataset, WaferInventory
from papers.polar_cnn.data_utils.preprocessing import denoise, polar_counts, preprocess
from papers.polar_cnn.data_utils.protocol import make_manifest, validate_manifest
from papers.polar_cnn.models import PolarCNN
from papers.polar_cnn.optim import KerasAdam
from papers.polar_cnn.train import ROOT, load_config, main, make_loader
from papers.polar_cnn.trainer import PolarTrainer


def test_log_counts_and_preprocessing_order_are_explicit():
    wafer = np.ones((52, 52), np.uint8)
    wafer[24:27, 24:27] = 2
    image = Image.fromarray(wafer)
    raw = preprocess(image, threshold=0)
    np.testing.assert_allclose(
        preprocess(image, threshold=0, count_transform="log1p"), np.log1p(raw)
    )
    native = preprocess(image, denoise_order="before_resize")
    resized = preprocess(image, denoise_order="after_resize")
    assert not np.array_equal(native, resized)
    with pytest.raises(ValueError):
        preprocess(image, count_transform="unknown")


def test_native_sectors_preserve_thin_lines_and_match_26_geometry():
    wafer = np.ones((26, 26), np.uint8)
    wafer[12, 5:21] = 2
    np.testing.assert_array_equal(
        preprocess(Image.fromarray(wafer), threshold=0, map_resolution="native_scaled"),
        preprocess(Image.fromarray(wafer), threshold=0),
    )
    thin = np.ones((52, 52), np.uint8)
    thin[24, 10:42] = 2
    original = Image.fromarray(thin)
    assert preprocess(original, threshold=0).sum() == 0
    assert preprocess(original, threshold=0, map_resolution="native_scaled").sum() == 8


def test_minimum_class_augmentation_retains_originals_and_only_augments_train(fixture_data):
    inventory = WaferInventory(fixture_data)
    config = config_for(fixture_data)
    config["data"]["train_min_per_class"] = 20
    manifest = make_manifest(inventory, config, 42)
    validate_manifest(inventory, manifest, config)
    assert {i for i, _ in manifest["records"]["train"]} == set(
        manifest["original_indices"]["train"]
    )
    assert any(angle for _, angle in manifest["records"]["train"])
    assert all(count >= 20 for count in manifest["class_counts"]["train"].values())
    for part in ("val", "test"):
        assert all(angle == 0 for _, angle in manifest["records"][part])
    config["data"]["train_min_per_class"] = 1
    manifest = make_manifest(inventory, config, 42)
    assert len(manifest["records"]["train"]) == len(manifest["original_indices"]["train"])
    config["data"]["train_per_class"] = 2
    with pytest.raises(ValueError):
        make_manifest(inventory, config, 42)


def test_experimental_checkpoint_resume_and_independent_evaluation(fixture_data, tmp_path):
    config = config_for(fixture_data)
    config["data"].update(
        map_resolution="native_scaled",
        denoise_order="before_resize",
        count_transform="log1p",
        train_min_per_class=20,
    )
    config["training"].update(balanced_sampling=True, selection_metric="f1", learning_rate=0.001)
    path = tmp_path / "experimental.yaml"
    path.write_text(yaml.safe_dump(config))
    run = tmp_path / "resumed_experiment"
    args = ["--config", str(path), "--device", "cpu", "--save-dir", str(run)]
    main(args)
    main([*args, "--resume", "--epochs", "2"])
    other = tmp_path / "uninterrupted_experiment"
    main(["--config", str(path), "--device", "cpu", "--save-dir", str(other), "--epochs", "2"])
    resumed = torch.load(run / "last.pt", weights_only=False)
    uninterrupted = torch.load(other / "last.pt", weights_only=False)
    for name, value in resumed["model"].items():
        torch.testing.assert_close(value, uninterrupted["model"][name], rtol=0, atol=0)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "papers/polar_cnn/evaluate.py"),
            "--checkpoint",
            str(run / "best.pt"),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    internal = json.loads((run / "test_metrics.json").read_text())
    independent = json.loads((run / "best_test_metrics.json").read_text())
    assert internal["f1"] == independent["f1"]
    assert internal["confusion_matrix"] == independent["confusion_matrix"]


def test_balanced_sampler_equalizes_class_mass_without_touching_eval(fixture_data):
    inventory = WaferInventory(fixture_data)
    records = [[i, 0] for i in range(48)] + [[label * 24, 0] for label in range(2, 8)]
    dataset = PolarDataset(inventory, records)
    generator = torch.Generator().manual_seed(42)
    loader = make_loader(dataset, 16, "cpu", generator=generator, balanced=True)
    masses = [loader.sampler.weights[dataset.targets == label].sum().item() for label in range(8)]
    np.testing.assert_allclose(masses, np.ones(8))
    assert len(loader.sampler) == len(dataset)
    eval_loader = make_loader(dataset, 16, "cpu")
    assert list(eval_loader.sampler) == list(range(len(dataset)))


def test_selection_prefers_macro_f1_instead_of_lower_loss(tmp_path, monkeypatch):
    trainer = PolarTrainer(PolarCNN(), "cpu")
    rows = iter(
        [
            {"loss": 1.0},
            {"loss": 0.8, "f1": 0.9, "accuracy": 0.8},
            {"loss": 0.5},
            {"loss": 0.4, "f1": 0.7, "accuracy": 0.9},
        ]
    )
    monkeypatch.setattr(trainer, "epoch", lambda *args, **kwargs: next(rows))
    config = load_config("papers/polar_cnn/configs/config.yaml")
    config["training"]["selection_metric"] = "f1"
    trainer.fit([], [], 2, tmp_path, config, {}, torch.Generator().manual_seed(42))
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["epoch"] == 1
    assert best["checkpoint_selection"] == "validation_f1"


def test_study_never_decodes_test_during_selection_and_keeps_completed_runs(
    fixture_data, tmp_path, monkeypatch
):
    from papers.polar_cnn import experiments

    config = config_for(fixture_data)
    inventory = WaferInventory(fixture_data)
    test_ids = set(make_manifest(inventory, config, 42)["original_indices"]["test"])
    visited = []
    original_dataset = experiments.PolarDataset

    def dataset_without_test(inventory, records, *args, **kwargs):
        assert not {i for i, _ in records} & test_ids
        visited.extend(i for i, _ in records)
        return original_dataset(inventory, records, *args, **kwargs)

    monkeypatch.setattr(experiments, "load_config", lambda path: deepcopy(config))
    monkeypatch.setattr(experiments, "PolarDataset", dataset_without_test)
    monkeypatch.setattr(
        experiments, "VARIANTS", [("tiny", {"count_transform": "log1p", "balanced_sampling": True})]
    )
    output = tmp_path / "study"
    args = [
        "--device",
        "cpu",
        "--output-dir",
        str(output),
        "--screening-epochs",
        "2",
        "--skip-final",
    ]
    experiments.main(args)
    before = (output / "tiny" / "best.pt").read_bytes()
    assert visited and not (output / "tiny" / "test_metrics.json").exists()
    experiments.main([*args, "--resume-study"])
    assert (output / "tiny" / "best.pt").read_bytes() == before
    assert json.loads((output / "selection.json").read_text())["test_used_for_selection"] is False


@pytest.fixture
def fixture_data(tmp_path):
    root = tmp_path / "wm811k"
    (root / "images").mkdir(parents=True)
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "failureType"])
        for label, name in enumerate(CLASSES):
            for index in range(24):
                wafer = np.ones((26, 26), np.uint8)
                y, x = 2 + label * 2, 2 + index % 15
                wafer[y : y + 3, x : x + 3] = 2
                wafer[0, 0] = 0
                filename = f"{label}_{index}.png"
                Image.fromarray(wafer).save(root / "images" / filename)
                writer.writerow([filename, name])
        writer.writerow(["ignored.png", "none"])
    return root


def config_for(root):
    config = load_config("papers/polar_cnn/configs/config.yaml")
    config["data"]["data_root"] = str(root)
    config["training"].update(batch_size=32, num_epochs=1)
    return config


def test_architecture_matches_parameter_counts():
    model = PolarCNN()
    assert sum(p.numel() for p in model.parameters()) == 93192
    assert model.features(torch.ones(2, 1, 26, 26)).shape == (2, 64, 3, 3)
    assert model(torch.ones(1, 1, 7, 7)).shape == (1, 8)
    with pytest.raises(ValueError):
        model(torch.ones(2, 1, 26, 26))


def test_denoise_uses_original_neighbourhood_and_preserves_background():
    image = np.zeros((26, 26), np.uint8)
    image[10, 10:13] = 2
    image[20, 20] = 2
    output = denoise(image)
    assert output[10, 11] == 2
    assert output[10, 10] == 1 and output[10, 12] == 1
    assert output[20, 20] == 1 and output[0, 0] == 0
    assert image[20, 20] == 2


def test_angles_use_counterclockwise_and_cover_last_sector():
    image = np.ones((26, 26), np.uint8)
    image[12, 20] = 2  # +x, slight positive y.
    image[5, 13] = 2  # ~86deg, second angular sector.
    image[13, 20] = 2  # ~356deg, last angular sector.
    hist = polar_counts(image)
    assert hist.shape == (7, 7)
    assert hist.sum() == 3
    assert hist[3, 0] == hist[3, 1] == hist[3, 6] == 1


@pytest.mark.parametrize("levels", [(0, 1, 2), (0, 127, 254), (0, 128, 255)])
def test_encoding_is_independent_of_png_scale(levels):
    wafer = np.ones((26, 26), np.uint8)
    wafer[10:15, 10:15] = 2
    wafer[0, 0] = 0
    encoded = np.array(levels, dtype=np.uint8)[wafer]
    np.testing.assert_equal(
        preprocess(Image.fromarray(encoded)), preprocess(Image.fromarray(wafer))
    )


def test_unknown_encoding_is_rejected():
    with pytest.raises(ValueError, match="encoding"):
        preprocess(Image.fromarray(np.full((26, 26), 77, np.uint8)))


def test_manifest_splits_originals_before_training_rotations(fixture_data):
    inventory = WaferInventory(fixture_data)
    config = config_for(fixture_data)
    config["data"]["train_per_class"] = 30
    manifest = make_manifest(inventory, config, 42)
    validate_manifest(inventory, manifest, config)
    assert len(manifest["records"]["train"]) == 8 * 30
    assert any(angle for _, angle in manifest["records"]["train"])
    for name in ("val", "test"):
        assert all(angle == 0 for _, angle in manifest["records"][name])
    tampered = deepcopy(manifest)
    tampered["records"]["test"][0] = tampered["records"]["train"][0]
    with pytest.raises(ValueError):
        validate_manifest(inventory, tampered)


def test_threefold_test_partitions_cover_each_original_once(fixture_data):
    inventory = WaferInventory(fixture_data)
    config = config_for(fixture_data)
    folds = [make_manifest(inventory, config, 42, fold=i) for i in range(3)]
    heldouts = [set(fold["original_indices"]["test"]) for fold in folds]
    assert len(set.union(*heldouts)) == len(inventory.samples)
    assert not heldouts[0] & heldouts[1]
    assert not heldouts[1] & heldouts[2]
    for fold in folds:
        validate_manifest(inventory, fold)


def test_changed_png_invalidates_saved_manifest(fixture_data):
    inventory = WaferInventory(fixture_data)
    manifest = make_manifest(inventory, config_for(fixture_data), 42)
    Image.fromarray(np.ones((26, 26), np.uint8)).save(fixture_data / "images" / "0_0.png")
    with pytest.raises(ValueError, match="inventory"):
        validate_manifest(WaferInventory(fixture_data), manifest)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("adapter", ["nearest", "bilinear"])
def test_backend_forward_backward_and_optimizer_step(device, adapter):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable in current sandbox")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = PolarCNN(input_adapter=adapter).to(device)
    trainer = PolarTrainer(model, device)
    inputs = torch.rand(16, 1, 7, 7, device=device) * 8
    targets = torch.arange(16, device=device) % 8
    before = model.features[0].weight.detach().clone()
    loss = trainer.loss_fn(model(inputs), targets)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    trainer.optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.features[0].weight)


def test_spawn_dataloader_is_picklable(fixture_data):
    inventory = WaferInventory(fixture_data)
    dataset = PolarDataset(inventory, [[i, 0] for i in range(10)])
    loader = make_loader(dataset, 4, "cpu", workers=1)
    assert sum(len(batch["targets"]) for batch in loader) == 10


def test_end_to_end_train_resume_and_evaluate(fixture_data, tmp_path):
    config = config_for(fixture_data)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run = tmp_path / "run"
    args = ["--config", str(config_path), "--save-dir", str(run), "--device", "cpu"]
    main([*args, "--prepare-only"])
    assert not run.exists()
    main(args)
    initial = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert initial["epoch"] == 1
    with pytest.raises(FileExistsError):
        main(args)
    main([*args, "--resume", "--epochs", "2"])
    resumed = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2 and len(resumed["history"]) == 2
    # Compare resuming to an uninterrupted run, including shuffled batches.
    uninterrupted = tmp_path / "uninterrupted"
    main(
        [
            "--config",
            str(config_path),
            "--save-dir",
            str(uninterrupted),
            "--device",
            "cpu",
            "--epochs",
            "2",
        ]
    )
    other = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=False)
    for name, value in resumed["model"].items():
        torch.testing.assert_close(value, other["model"][name], rtol=0, atol=0)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "papers/polar_cnn/evaluate.py"),
            "--checkpoint",
            str(run / "best.pt"),
            "--device",
            "cpu",
            "--batch-size",
            "19",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    metrics = json.loads((run / "best_test_metrics.json").read_text())
    assert len(metrics["confusion_matrix"]) == 8
    assert 0 <= metrics["accuracy"] <= 1


def test_entrypoints_work_outside_repo(tmp_path):
    for script in ("train.py", "evaluate.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "papers/polar_cnn" / script), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_keras_adam_matches_epsilon_hat_formula_and_differs_from_pytorch():
    parameter = torch.nn.Parameter(torch.tensor([0.3, -0.8], dtype=torch.float64))
    optimizer = KerasAdam([parameter])
    expected = parameter.detach().numpy().copy()
    m, v = np.zeros(2), np.zeros(2)
    gradients = [np.array([1e-9, -1e-8]), np.array([0.1, -0.2]), np.array([-0.3, 0.4])]
    for step, gradient in enumerate(gradients, 1):
        parameter.grad = torch.tensor(gradient, dtype=torch.float64)
        optimizer.step()
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient**2
        alpha = 0.01 * np.sqrt(1 - 0.999**step) / (1 - 0.9**step)
        expected -= alpha * m / (np.sqrt(v) + 1e-7)
        np.testing.assert_allclose(parameter.detach().numpy(), expected, atol=1e-15)
    original = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    keras = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    pytorch_optimizer = torch.optim.Adam([original], lr=0.01, eps=1e-7)
    keras_optimizer = KerasAdam([keras])
    original.grad = keras.grad = torch.tensor([1e-9], dtype=torch.float64)
    pytorch_optimizer.step()
    keras_optimizer.step()
    assert abs(original.item() - keras.item()) > 1e-5


def test_fixed_epoch_70_30_protocol_has_no_validation_selection(fixture_data, tmp_path):
    config = config_for(fixture_data)
    config["data"]["val_fraction_of_development"] = 0.0
    inventory = WaferInventory(fixture_data)
    manifest = make_manifest(inventory, config, 42)
    validate_manifest(inventory, manifest, config)
    assert manifest["original_indices"]["val"] == []
    assert len(manifest["original_indices"]["train"]) == int(len(inventory.samples) * 0.7)
    path = tmp_path / "paper.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run = tmp_path / "paper_run"
    args = ["--config", str(path), "--save-dir", str(run), "--device", "cpu"]
    main(args)
    assert (run / "final.pt").exists() and not (run / "best.pt").exists()
    assert json.loads((run / "test_metrics.json").read_text())["checkpoint"] == "final.pt"
    main([*args, "--resume", "--epochs", "2"])
    saved = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 2
    assert saved["checkpoint_selection"] == "final_epoch"
    assert "val_loss" not in saved["history"][0]
