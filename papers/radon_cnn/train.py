"""Training entry point for RadonCNN on WM-811K.

Usage:
    # Train from scratch
    python papers/radon_cnn/train.py

    # Resume from last checkpoint
    python papers/radon_cnn/train.py --resume

    # Resume from a specific checkpoint file
    python papers/radon_cnn/train.py --resume checkpoints/radon_cnn/best.pt

    # Train for 100 more epochs (resumed or from scratch)
    python papers/radon_cnn/train.py --epochs 100 --resume

Trains RadonCNN using the common training loop and sample-weighted validation.

Hyperparameters (from paper):
    - lr=0.0003, Adam optimizer
    - lr_decay=0.99 per epoch (ExponentialLR)
    - Early stopping with patience=30 epochs
    - CrossEntropyLoss
    - 20 repeats with different random seeds (for full experiment)
    - Balanced 7 classes (excludes Near-full and None)
    - Image size: 64x64
    - Background removal preprocessing
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.training.checkpoint import CheckpointManager
from common.training.early_stopping import EarlyStopping
from common.training.logger import TrainingLogger
from common.training.metrics import accuracy, f1, precision, recall
from common.training.utils import NativeScaler, resolve_device
from papers.radon_cnn.data_utils import WaferRadonDataset
from papers.radon_cnn.data_utils.protocol import (
    dataset_options,
    load_manifest,
    make_manifest,
    subsets,
)
from papers.radon_cnn.models.radon_cnn import RadonCNN
from papers.radon_cnn.trainer import RadonTrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train RadonCNN on WM-811K wafer map dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="papers/radon_cnn/configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training (auto: CUDA, then Apple MPS, then CPU)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for experiment repeat (paper uses 20 repeats)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="last",
        default=None,
        help=(
            "Resume training from a checkpoint. "
            "Use --resume (loads last.pt), --resume best (loads best.pt), "
            "or --resume /path/to/checkpoint.pt"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the training loop."""
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # ── Load config ──────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    config = EngineConfig.from_yaml(config_path)

    # Override config with CLI args
    num_epochs = args.epochs or config.get("training.num_epochs", 200)
    batch_size = args.batch_size or config.get("training.batch_size", 64)
    learning_rate = args.lr or config.get("training.learning_rate", 0.0003)
    lr_decay = config.get("training.lr_decay", 0.99)
    early_stopping_patience = config.get("training.early_stopping_patience", 30)
    weight_decay = config.get("training.weight_decay", 0.0)
    grad_max_norm = config.get("training.grad_max_norm", None)

    # Data params
    data_root = config.get("data.data_root", "datasets/wm811k")
    num_classes = config.get("model.num_classes", 7)

    # Checkpoint
    save_dir = config.get("checkpoint.save_dir", "checkpoints/radon_cnn")

    # ── Seed ─────────────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else config.get("training.seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if device == "cpu":
        torch.set_num_threads(config.get("training.cpu_threads", 2))
    print(f"Using seed: {seed}")

    # ── Dataset ──────────────────────────────────────────────────────────
    print(f"Loading WM-811K dataset from: {data_root}")
    full_dataset = WaferRadonDataset(**dataset_options(config))
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Classes: {full_dataset.class_names}")

    split_path = Path(save_dir) / "split.json"
    if args.resume is not None:
        if args.resume not in {"last", "best"}:
            split_path = Path(args.resume).parent / "split.json"
        manifest = load_manifest(full_dataset, split_path)
        if args.seed is not None and args.seed != manifest["seed"]:
            raise ValueError("--seed differs from the resumed split")
        seed = manifest["seed"]
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    else:
        if (Path(save_dir) / "last.pt").exists() or (Path(save_dir) / "best.pt").exists():
            raise FileExistsError(f"Existing run in {save_dir}; use --resume or a new save_dir")
        manifest = make_manifest(full_dataset, config, seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(manifest, indent=2))
    train_dataset, val_dataset, test_dataset = subsets(full_dataset, manifest)
    print(f"  Protocol: {manifest['protocol']}; unique wafers: {manifest['unique_counts']}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # WeightedRandomSampler for balanced batches (compensates for class imbalance)
    from collections import Counter

    # Get labels from train_dataset via full_dataset indices
    train_labels = [full_dataset._samples[i][1] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    # Weight = 1 / count for each class
    weights = [1.0 / class_counts[full_dataset._samples[i][1]] for i in train_dataset.indices]
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if manifest["protocol"] == "full" and config.get("data.balance_train_batches", True):
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_dataset), replacement=True, generator=generator
        )
    if batch_size < 2 or len(train_dataset) < batch_size:
        raise ValueError("Training requires batch_size >= 2 and at least one full batch")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        generator=generator,
        drop_last=True,  # FC BatchNorm cannot train on a singleton final batch.
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────
    print("Creating RadonCNN model...")
    model = RadonCNN(in_channels=1, num_classes=num_classes)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ── Scheduler (ExponentialLR with gamma=0.99 per epoch) ──────────────
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=lr_decay,
    )

    # ── Loss ─────────────────────────────────────────────────────────────
    loss_fn = nn.CrossEntropyLoss()

    # ── Metrics ──────────────────────────────────────────────────────────
    metric_fns = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    # ── Callbacks ────────────────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        mode="min",  # Track val_loss: lower is better
    )
    checkpoint_manager = CheckpointManager(
        save_dir=save_dir,
        metric_name="val_loss",
        mode="min",
    )
    logger = TrainingLogger()
    scaler = NativeScaler(enabled=(device == "cuda" and config.get("training.amp", False)))

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = RadonTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        metric_fns=metric_fns,
        early_stopping=early_stopping,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        scaler=scaler,
        grad_max_norm=grad_max_norm,
        verbose=True,
    )

    # ── Resume from checkpoint ───────────────────────────────────────────
    resumed_epoch = 0
    if args.resume is not None:
        if args.resume == "last":
            # Load last.pt from the checkpoint directory
            checkpoint_path = checkpoint_manager.last_path
            print(f"\nResuming from last checkpoint: {checkpoint_path}")
            resumed_epoch = trainer.resume_from_checkpoint(load_last=True)
        elif args.resume == "best":
            # Load best.pt from the checkpoint directory
            checkpoint_path = checkpoint_manager.best_path
            print(f"\nResuming from best checkpoint: {checkpoint_path}")
            resumed_epoch = trainer.resume_from_checkpoint(load_last=False)
        else:
            # Load a specific checkpoint file
            checkpoint_path = Path(args.resume)
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            resumed_epoch = trainer.resume_from_checkpoint(
                checkpoint_path=checkpoint_path,
            )
        print(f"  Resumed at epoch {resumed_epoch}")
        print(f"  Current LR: {trainer._get_current_lr():.6f}")
        best_path = Path(checkpoint_path).parent / "best.pt"
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            checkpoint_manager._best_metric = best["metric"]
            early_stopping.best_metric = best["metric"]
            early_stopping._best_state = best["model"]
            if best_path.resolve() != checkpoint_manager.best_path.resolve():
                if checkpoint_manager.best_path.exists():
                    raise FileExistsError(
                        "Resume destination already has a different best checkpoint"
                    )
                shutil.copy2(best_path, checkpoint_manager.best_path)

    # Save the effective CLI overrides, so evaluate.py can recover the run.
    config._data["training"].update(
        seed=manifest["seed"],
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    (Path(save_dir) / "experiment.yaml").write_text(yaml.safe_dump(config._data))
    if split_path != Path(save_dir) / "split.json":
        (Path(save_dir) / "split.json").write_text(json.dumps(manifest, indent=2))

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  LR: {learning_rate}, Decay: {lr_decay}/epoch")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Checkpoints: {save_dir}")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
    )

    # ── Final evaluation on test set ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final evaluation on test set")
    print("=" * 60)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    checkpoint_manager.load_best(model)
    test_metrics = trainer.validate(test_loader)
    print(f"  Test accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
    print(f"  Test f1:       {test_metrics.get('f1', 0.0):.4f}")
    print(f"  Test precision:{test_metrics.get('precision', 0.0):.4f}")
    print(f"  Test recall:   {test_metrics.get('recall', 0.0):.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
