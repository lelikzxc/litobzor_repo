"""Evaluate a saved Polar CNN using its exact original-wafer test manifest."""

# ruff: noqa: E402 -- Standalone entry point bootstraps the repository path.

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.training.utils import resolve_device
from papers.polar_cnn.data_utils import CLASSES, PolarDataset
from papers.polar_cnn.data_utils.preprocessing import preprocessing_options
from papers.polar_cnn.data_utils.protocol import validate_manifest
from papers.polar_cnn.models import PolarCNN
from papers.polar_cnn.train import inventory_from_config, make_loader, project_path
from papers.polar_cnn.trainer import PolarTrainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--data-root", help="Relocate dataset between operating systems")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = resolve_device(args.device)
    path = project_path(args.checkpoint)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config, manifest = checkpoint["config"], checkpoint["manifest"]
    torch.set_num_threads(config["training"]["cpu_threads"])
    inventory = inventory_from_config(config, args.data_root)
    validate_manifest(inventory, manifest, config)
    dataset = PolarDataset(
        inventory,
        manifest["records"]["test"],
        config["data"]["denoise_threshold"],
        **preprocessing_options(config["data"]),
    )
    model = PolarCNN(num_classes=8, input_adapter=config["model"]["input_adapter"])
    model.load_state_dict(checkpoint["model"])
    trainer = PolarTrainer(model, device)
    loader = make_loader(dataset, args.batch_size, device)
    metrics = trainer.epoch(loader, desc="Test")
    metrics.update(
        checkpoint=path.name,
        selected_epoch=checkpoint["epoch"],
        dataset_sha256=inventory.identity,
        diagnostic=manifest["diagnostic_max_per_class"] is not None,
    )
    output = project_path(args.output_dir) if args.output_dir else path.parent
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{path.stem}_test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (output / f"{path.stem}_confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual/predicted", *CLASSES])
        writer.writerows(
            [name, *row] for name, row in zip(CLASSES, metrics["confusion_matrix"], strict=True)
        )
    print(f"Using device: {device}; test originals: {len(dataset)}")
    for metric in ("loss", "accuracy", "f1", "precision", "recall", "weighted_f1"):
        print(f"{metric}: {metrics[metric]:.5f}")
    for name in CLASSES:
        row = metrics["per_class"][name]
        print(
            f"{name}: F1={row['f1-score']:.4f}, recall={row['recall']:.4f}, support={row['support']:.0f}"
        )


if __name__ == "__main__":
    main()
