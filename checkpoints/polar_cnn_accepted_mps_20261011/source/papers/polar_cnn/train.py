"""Train Model 4 Polar CNN on original-wafer-disjoint WM-811K splits."""

# ruff: noqa: E402 -- Standalone entry point bootstraps the repository path.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.training.utils import resolve_device
from common.utils.seed import set_seed
from papers.polar_cnn.data_utils import CLASSES, PolarDataset, WaferInventory
from papers.polar_cnn.data_utils.preprocessing import preprocessing_options
from papers.polar_cnn.data_utils.protocol import make_manifest, validate_manifest
from papers.polar_cnn.models import PolarCNN
from papers.polar_cnn.trainer import PolarTrainer


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    config = yaml.safe_load(project_path(path).read_text(encoding="utf-8"))
    if config["model"]["num_classes"] != 8:
        raise ValueError("This reproduction requires exactly eight defect classes")
    return config


def inventory_from_config(config, data_root=None):
    data = config["data"]
    reviewed = data.get("reviewed_ids")
    return WaferInventory(
        project_path(data_root or data["data_root"]),
        data["selection"],
        project_path(reviewed) if reviewed else None,
    )


def make_loader(
    dataset, batch_size, device, workers=0, shuffle=False, generator=None, balanced=False
):
    sampler = None
    if balanced:
        counts = torch.bincount(dataset.targets, minlength=8).double()
        if (counts == 0).any():
            raise ValueError("Balanced sampling requires all eight training classes")
        sampler = WeightedRandomSampler(
            1 / counts[dataset.targets], len(dataset), replacement=True, generator=generator
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        generator=generator,
        num_workers=workers,
        multiprocessing_context="spawn" if workers else None,
        pin_memory=device == "cuda",
        drop_last=False,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="papers/polar_cnn/configs/config.yaml")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--save-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume last.pt in save-dir")
    parser.add_argument(
        "--fold", type=int, choices=[0, 1, 2], help="True stratified 3-fold CV (test=1/3)"
    )
    parser.add_argument(
        "--max-per-class", type=int, help="Bounded diagnostic inventory, >=10 originals/class"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Verify/decode all selected maps, without training",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    for key, value in (
        ("num_epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("seed", args.seed),
        ("num_workers", args.num_workers),
    ):
        if value is not None:
            config["training"][key] = value
    if args.data_root:
        config["data"]["data_root"] = args.data_root
    training = config["training"]
    if any(training[k] <= 0 for k in ("num_epochs", "batch_size", "learning_rate", "cpu_threads")):
        raise ValueError("epochs/batch_size/lr/cpu_threads must be positive")
    if training["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative")
    device = resolve_device(args.device)
    set_seed(training["seed"])
    torch.set_num_threads(training["cpu_threads"])
    run_dir = project_path(args.save_dir or config["checkpoint"]["save_dir"])
    config["checkpoint"]["save_dir"] = str(run_dir)
    if args.resume:
        if not (run_dir / "last.pt").is_file():
            raise FileNotFoundError("--resume requires last.pt in save-dir")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}; use --resume or a new --save-dir"
        )
    inventory = inventory_from_config(config)
    print(f"Using device: {device}\nOriginal defect wafers: {len(inventory.samples)}", flush=True)
    print(
        f"Classes: {CLASSES}\nInput adapter: {config['model']['input_adapter']} (local documented assumption)"
    )
    if args.resume:
        manifest = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
        validate_manifest(inventory, manifest, config)
        if training["seed"] != manifest["seed"]:
            raise ValueError("Resume seed differs from the saved split")
        if args.fold is not None and args.fold != manifest["fold"]:
            raise ValueError("Resume fold differs")
        if (
            args.max_per_class is not None
            and args.max_per_class != manifest["diagnostic_max_per_class"]
        ):
            raise ValueError("Resume diagnostic inventory differs")
    else:
        manifest = make_manifest(inventory, config, training["seed"], args.fold, args.max_per_class)
        validate_manifest(inventory, manifest, config)
    datasets = {
        name: PolarDataset(
            inventory,
            manifest["records"][name],
            config["data"]["denoise_threshold"],
            progress=sys.stderr.isatty(),
            **preprocessing_options(config["data"]),
        )
        for name in ("train", "val", "test")
        if manifest["records"][name]
    }
    print("Prepared inputs:", {name: len(dataset) for name, dataset in datasets.items()})
    print("Class counts:", manifest["class_counts"], flush=True)
    if args.prepare_only:
        print("Preparation verified. No model training or checkpoints created.")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "experiment.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    generator = torch.Generator().manual_seed(training["seed"])
    loaders = {
        name: make_loader(
            dataset,
            training["batch_size"],
            device,
            training["num_workers"],
            name == "train",
            generator if name == "train" else None,
            balanced=name == "train" and training.get("balanced_sampling", False),
        )
        for name, dataset in datasets.items()
    }
    model = PolarCNN(num_classes=8, input_adapter=config["model"]["input_adapter"])
    print(
        f"Parameters: {sum(p.numel() for p in model.parameters()):,}; FP32; Keras Adam epsilon-hat=1e-7"
    )
    trainer = PolarTrainer(model, device, training["learning_rate"])
    trainer.fit(
        loaders["train"],
        loaders.get("val"),
        training["num_epochs"],
        run_dir,
        config,
        manifest,
        generator,
        run_dir / "last.pt" if args.resume else None,
    )
    selected_checkpoint = "best.pt" if "val" in loaders else "final.pt"
    checkpoint = torch.load(run_dir / selected_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = trainer.epoch(loaders["test"], desc="Test")
    result.update(
        checkpoint=selected_checkpoint,
        selected_epoch=checkpoint["epoch"],
        dataset_sha256=inventory.identity,
        diagnostic=manifest["diagnostic_max_per_class"] is not None,
    )
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Test accuracy={result['accuracy']:.4f}, macro F1={result['f1']:.4f}")


if __name__ == "__main__":
    main()
