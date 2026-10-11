"""Bounded validation-only ablations, then one selected full training/test run.

These are diagnostic departures from the paper, not recovered author settings.
The historical 30% test identities are preserved and never used for selection.
"""

# ruff: noqa: E402 -- Standalone entry point bootstraps the repository path.

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.training.utils import resolve_device
from common.utils.seed import set_seed
from papers.polar_cnn.data_utils import PolarDataset
from papers.polar_cnn.data_utils.preprocessing import preprocessing_options
from papers.polar_cnn.data_utils.protocol import make_manifest, validate_manifest
from papers.polar_cnn.models import PolarCNN
from papers.polar_cnn.train import inventory_from_config, load_config, make_loader, project_path
from papers.polar_cnn.trainer import PolarTrainer

VARIANTS = [
    ("baseline", {}),
    ("lower_lr", {"learning_rate": 0.001}),
    ("no_denoise", {"learning_rate": 0.001, "denoise_threshold": 0}),
    ("threshold2", {"learning_rate": 0.001, "denoise_threshold": 2}),
    ("native_denoise", {"learning_rate": 0.001, "denoise_order": "before_resize"}),
    ("bilinear", {"learning_rate": 0.001, "input_adapter": "bilinear"}),
    ("log_counts", {"learning_rate": 0.001, "count_transform": "log1p"}),
    ("balanced", {"learning_rate": 0.001, "balanced_sampling": True}),
    (
        "log_balanced",
        {"learning_rate": 0.001, "count_transform": "log1p", "balanced_sampling": True},
    ),
    (
        "log_balanced_no_denoise",
        {
            "learning_rate": 0.001,
            "count_transform": "log1p",
            "balanced_sampling": True,
            "denoise_threshold": 0,
        },
    ),
]
VARIANTS.extend(
    [
        (
            "native_log",
            {"learning_rate": 0.001, "denoise_order": "before_resize", "count_transform": "log1p"},
        ),
        (
            "native_balanced",
            {"learning_rate": 0.001, "denoise_order": "before_resize", "balanced_sampling": True},
        ),
        (
            "native_log_balanced",
            {
                "learning_rate": 0.001,
                "denoise_order": "before_resize",
                "count_transform": "log1p",
                "balanced_sampling": True,
            },
        ),
    ]
)
# Rotations only on training originals; retain every majority-class original.
VARIANTS.extend(
    [
        (
            "native_augmented",
            {
                "learning_rate": 0.001,
                "denoise_order": "before_resize",
                "train_min_per_class": 1500,
                "balanced_sampling": True,
            },
        ),
        (
            "native_sectors_augmented",
            {
                "learning_rate": 0.001,
                "map_resolution": "native_scaled",
                "denoise_order": "before_resize",
                "train_min_per_class": 1500,
                "balanced_sampling": True,
            },
        ),
        (
            "native_sectors_log_augmented",
            {
                "learning_rate": 0.001,
                "map_resolution": "native_scaled",
                "denoise_order": "before_resize",
                "count_transform": "log1p",
                "train_min_per_class": 1500,
                "balanced_sampling": True,
            },
        ),
    ]
)
VARIANTS.extend(
    [
        (
            "native_sectors",
            {
                "learning_rate": 0.001,
                "map_resolution": "native_scaled",
                "denoise_order": "before_resize",
            },
        ),
        (
            "native_sectors_log",
            {
                "learning_rate": 0.001,
                "map_resolution": "native_scaled",
                "denoise_order": "before_resize",
                "count_transform": "log1p",
            },
        ),
    ]
)


def variant_config(base, changes):
    config = deepcopy(base)
    for key, value in changes.items():
        section = (
            "model"
            if key == "input_adapter"
            else ("training" if key in {"learning_rate", "balanced_sampling"} else "data")
        )
        config[section][key] = value
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", default="checkpoints/polar_cnn_f1_study_20261011")
    parser.add_argument("--screening-epochs", type=int, default=40)
    parser.add_argument("--final-epochs", type=int, default=300)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--resume-study", action="store_true")
    args = parser.parse_args(argv)
    if min(args.screening_epochs, args.final_epochs) < 1:
        raise ValueError("Epoch counts must be positive")
    output = project_path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume_study:
        raise FileExistsError(f"Refusing to overwrite study: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    base = load_config("papers/polar_cnn/configs/config.yaml")
    base["data"]["val_fraction_of_development"] = 0.20
    base["training"].update(selection_metric="f1", num_epochs=args.screening_epochs)
    torch.set_num_threads(base["training"]["cpu_threads"])
    inventory = inventory_from_config(base)
    canonical = make_manifest(inventory, base, 42)
    # Verify that selection uses exactly the already established test identities.
    historical = make_manifest(
        inventory, load_config("papers/polar_cnn/configs/config_paper_all_maps.yaml"), 42
    )
    assert canonical["original_indices"]["test"] == historical["original_indices"]["test"]
    cached = {}
    results_path = output / "screening_results.json"
    results = (
        json.loads(results_path.read_text()) if args.resume_study and results_path.exists() else []
    )

    def fit(config, name, epochs):
        run = output / name
        run.mkdir(exist_ok=args.resume_study)
        config["checkpoint"]["save_dir"] = str(run)
        config["training"]["num_epochs"] = epochs
        manifest = make_manifest(inventory, config, 42)
        validate_manifest(inventory, manifest, config)
        assert manifest["original_indices"] == canonical["original_indices"]
        key = json.dumps(
            {
                "threshold": config["data"]["denoise_threshold"],
                "train_min_per_class": config["data"].get("train_min_per_class"),
                **preprocessing_options(config["data"]),
            },
            sort_keys=True,
        )
        if key not in cached:
            cached[key] = {
                part: PolarDataset(
                    inventory,
                    manifest["records"][part],
                    config["data"]["denoise_threshold"],
                    **preprocessing_options(config["data"]),
                )
                for part in ("train", "val")
            }
        datasets = cached[key]
        set_seed(42)
        generator = torch.Generator().manual_seed(42)
        train = make_loader(
            datasets["train"],
            256,
            device,
            shuffle=True,
            generator=generator,
            balanced=config["training"].get("balanced_sampling", False),
        )
        val = make_loader(datasets["val"], 256, device)
        trainer = PolarTrainer(
            PolarCNN(input_adapter=config["model"]["input_adapter"]),
            device,
            config["training"]["learning_rate"],
        )
        (run / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (run / "split.json").write_text(json.dumps(manifest))
        print(
            f"\nVARIANT {name}; epochs={epochs}; train={len(datasets['train'])}; val={len(datasets['val'])}; test NOT evaluated",
            flush=True,
        )
        resume_path = run / "last.pt"
        if args.resume_study and resume_path.exists():
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
            if saved["manifest"] != manifest:
                raise ValueError("Study resume preprocessing/split differs")
            if saved["epoch"] < epochs:
                trainer.fit(train, val, epochs, run, config, manifest, generator, resume_path)
        else:
            trainer.fit(train, val, epochs, run, config, manifest, generator)
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        trainer.model.load_state_dict(checkpoint["model"])
        metrics = trainer.epoch(val)
        (run / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
        return (
            {
                "variant": name,
                "val_f1": metrics["f1"],
                "val_accuracy": metrics["accuracy"],
                "val_scratch_f1": metrics["per_class"]["Scratch"]["f1-score"],
                "selected_epoch": checkpoint["epoch"],
                "changes": config,
            },
            trainer,
            manifest,
        )

    for name, changes in VARIANTS:
        completed = next((row for row in results if row["variant"] == name), None)
        if completed is not None:
            expected = variant_config(base, changes)
            if any(
                completed["changes"][section] != expected[section]
                for section in ("data", "model", "training")
            ):
                raise ValueError(f"Completed study settings differ for {name}")
            print(f"Already completed {name}; keeping saved validation result", flush=True)
            continue
        result, _, _ = fit(variant_config(base, changes), name, args.screening_epochs)
        results.append(result)
        (output / "screening_results.json").write_text(json.dumps(results, indent=2))
        print(
            f"SCREENING RESULT {name}: val macro F1={result['val_f1']:.5f}, Scratch={result['val_scratch_f1']:.5f}",
            flush=True,
        )
    winner = max(results, key=lambda row: row["val_f1"])
    (output / "selection.json").write_text(
        json.dumps(
            {
                "criterion": "validation macro F1 only",
                "winner": winner["variant"],
                "test_used_for_selection": False,
                "historical_test_already_seen": True,
                "note": "Existing test was previously reported; final comparison is not a new blind benchmark.",
            },
            indent=2,
        )
    )
    config = deepcopy(winner["changes"])
    config["training"]["num_epochs"] = args.final_epochs
    config["checkpoint"]["save_dir"] = str(output / "selected_full")
    (output / "selected_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"SELECTED {winner['variant']}: validation macro F1={winner['val_f1']:.5f}", flush=True)
    if args.skip_final:
        return
    result, trainer, manifest = fit(config, "selected_full", args.final_epochs)
    test = PolarDataset(
        inventory,
        manifest["records"]["test"],
        config["data"]["denoise_threshold"],
        **preprocessing_options(config["data"]),
    )
    metrics = trainer.epoch(make_loader(test, 256, device), desc="Final test")
    metrics.update(
        checkpoint="best.pt",
        selected_epoch=result["selected_epoch"],
        dataset_sha256=inventory.identity,
        diagnostic=False,
        variant=winner["variant"],
        protocol="56/14/30; selection by validation macro F1",
        historical_test_already_seen=True,
    )
    (output / "selected_full" / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"FINAL TEST accuracy={metrics['accuracy']:.5f}; macro F1={metrics['f1']:.5f}", flush=True
    )


if __name__ == "__main__":
    main()
