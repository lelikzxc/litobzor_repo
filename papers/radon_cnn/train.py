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

Trains RadonCNN on the WM-811K wafer map dataset using the common Trainer.

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
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.training.metrics import accuracy, f1, precision, recall
from common.training.trainer import Trainer
from common.training.checkpoint import CheckpointManager
from common.training.early_stopping import EarlyStopping
from common.training.logger import TrainingLogger
from common.training.utils import NativeScaler
from papers.radon_cnn.data_utils import WaferRadonDataset
from papers.radon_cnn.models.radon_cnn import RadonCNN


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RadonCNN on WM-811K wafer map dataset"
    )
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
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training (auto=use CUDA if available)",
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
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
    grad_max_norm = config.get("training.grad_max_norm", 1.0)

    # Data params
    data_root = config.get("data.data_root", "datasets/wm811k")
    image_size = config.get("data.image_size", 64)
    num_classes = config.get("model.num_classes", 7)
    balanced = config.get("data.balanced", True)
    train_split = config.get("data.train_split", 0.8)
    val_split = config.get("data.val_split", 0.1)

    # Checkpoint
    save_dir = config.get("checkpoint.save_dir", "checkpoints/radon_cnn")

    # ── Seed ─────────────────────────────────────────────────────────────
    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        print(f"Using seed: {seed}")

    # ── Dataset ──────────────────────────────────────────────────────────
    print(f"Loading WM-811K dataset from: {data_root}")
    full_dataset = WaferRadonDataset(
        data_root=data_root,
        image_size=image_size,
        num_classes=num_classes,
        balanced=False,  # Use all 25k samples; WeightedRandomSampler balances batches
    )
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Classes: {full_dataset.class_names}")

    total = len(full_dataset)
    train_len = int(total * train_split)
    val_len = int(total * val_split)
    test_len = total - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # WeightedRandomSampler for balanced batches (compensates for class imbalance)
    from torch.utils.data import WeightedRandomSampler
    from collections import Counter

    # Get labels from train_dataset via full_dataset indices
    train_labels = [full_dataset._samples[i][1] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    # Weight = 1 / count for each class
    weights = [1.0 / class_counts[full_dataset._samples[i][1]] for i in train_dataset.indices]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
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
        metric_name="val_accuracy",
        mode="max",  # Track val_accuracy: higher is better
    )
    logger = TrainingLogger()
    scaler = NativeScaler(enabled=(device == "cuda"))

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
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
            # Reset scheduler LR to initial value so the model can escape
            # local minima instead of continuing to decay from ~0.0002.
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
            # Recreate scheduler with fresh state
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=lr_decay,
            )
            trainer.scheduler = scheduler
            print(f"  LR reset to {learning_rate} (was decaying from ~0.0002)")
        else:
            # Load a specific checkpoint file
            checkpoint_path = Path(args.resume)
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            resumed_epoch = trainer.resume_from_checkpoint(
                checkpoint_path=checkpoint_path,
            )
        print(f"  Resumed at epoch {resumed_epoch}")
        print(f"  Current LR: {trainer._get_current_lr():.6f}")

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  LR: {learning_rate}, Decay: {lr_decay}/epoch")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Checkpoints: {save_dir}")

    history = trainer.fit(
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

    test_metrics = trainer.validate(test_loader)
    print(f"  Test accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
    print(f"  Test f1:       {test_metrics.get('f1', 0.0):.4f}")
    print(f"  Test precision:{test_metrics.get('precision', 0.0):.4f}")
    print(f"  Test recall:   {test_metrics.get('recall', 0.0):.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()