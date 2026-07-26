"""Training entry point for FCS-VMamba on WM-811K.

Usage:
    python papers/vmamba/train.py --config papers/vmamba/configs/config.yaml

Trains FCS-VMamba on the WM-811K wafer map dataset using the common engine.
Supports CUDA automatically when available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, random_split

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.engine.engine import Engine
from papers.vmamba.data_utils import WaferWM811KDataset
from papers.vmamba.models.vmamba import FCSVMamba


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train FCS-VMamba on WM-811K wafer map dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="papers/vmamba/configs/config.yaml",
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
        "--subset",
        type=float,
        default=None,
        help="Use only a fraction of the training dataset (e.g. 0.2 = 20%%), "
             "with stratified sampling to preserve class balance",
    )
    return parser.parse_args()


def _stratified_subset(
    dataset: Subset,
    subset_ratio: float,
    seed: int = 42,
) -> Subset:
    """Create a stratified subset of a ``Subset`` (from ``random_split``).

    Uses ``StratifiedShuffleSplit`` to preserve class distribution,
    which is critical for imbalanced datasets like WM-811K.

    Args:
        dataset: A ``Subset`` wrapping the original ``WaferWM811KDataset``.
        subset_ratio: Fraction of the dataset to keep (0, 1].
        seed: Random seed for reproducibility.

    Returns:
        A new ``Subset`` with stratified sampling.
    """
    # Get labels from the underlying full dataset via the Subset indices
    full_dataset = dataset.dataset  # type: ignore[union-attr]
    labels = np.array([full_dataset._samples[i][1] for i in dataset.indices])

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=subset_ratio,
        random_state=seed,
    )
    subset_idx, _ = next(sss.split(np.zeros(len(labels)), labels))

    # Map back to original dataset indices
    original_indices = [dataset.indices[i] for i in subset_idx]
    return Subset(full_dataset, original_indices)


def main() -> None:
    """Run the training loop."""
    args = parse_args()

    # ── Load configuration ──────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = EngineConfig.from_yaml(config_path)

    # Apply CLI overrides
    if args.epochs is not None:
        config._data.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config._data.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config._data.setdefault("training", {})["learning_rate"] = args.lr
        config._data.setdefault("training", {}).setdefault("optimizer", {})["lr"] = args.lr
        config._data.setdefault("optimizer", {})["lr"] = args.lr

    # ── Resolve device ──────────────────────────────────────────────────
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Create dataset ──────────────────────────────────────────────────
    data_root = config.get("data.data_root", "datasets/wm811k")
    image_size = config.get("data.image_size", 224)
    train_split = config.get("data.train_split", 0.8)
    val_split = config.get("data.val_split", 0.1)

    print(f"Loading WM-811K dataset from: {data_root}")
    full_dataset = WaferWM811KDataset(
        data_root=data_root,
        image_size=image_size,
    )
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Classes: {full_dataset.class_names}")

    # ── Split dataset ───────────────────────────────────────────────────
    total = len(full_dataset)
    train_len = int(total * train_split)
    val_len = int(total * val_split)
    test_len = total - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42),
    )

    # ── Apply stratified subset for faster iteration ────────────────────
    if args.subset is not None:
        subset_ratio = float(args.subset)
        if subset_ratio <= 0.0 or subset_ratio > 1.0:
            print(f"Error: --subset must be in (0, 1], got {subset_ratio}")
            sys.exit(1)
        train_dataset = _stratified_subset(train_dataset, subset_ratio, seed=42)
        print(f"  Using stratified subset: {len(train_dataset)} train samples ({subset_ratio*100:.0f}%)")

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ── Create DataLoaders ──────────────────────────────────────────────
    batch_size = config.get("training.batch_size", 64)
    eval_batch_size = config.get("evaluation.batch_size", 128)
    num_workers = 0  # safe default on Windows

    def collate_fn(batch):
        """Custom collate for dict-based samples."""
        images = torch.stack([item["image"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return images, labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # ── Create model ────────────────────────────────────────────────────
    print("Creating FCS-VMamba model...")
    model = FCSVMamba.from_config(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── Create Engine ───────────────────────────────────────────────────
    print("Initializing engine...")
    engine = Engine(
        model=model,
        config=config,
        device=device,
    )

    # ── Train ───────────────────────────────────────────────────────────
    epochs = config.get("training.num_epochs", 50)
    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'='*60}")

    logger = engine.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )

    # ── Final metrics ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    # Print final training metrics
    history = logger.history
    if history:
        final = history[-1]
        print(f"\nFinal training metrics:")
        if "train_loss" in final:
            print(f"  Train Loss: {final['train_loss']:.4f}")
        if "val_loss" in final:
            print(f"  Val Loss:   {final['val_loss']:.4f}")
        for key in ["train_accuracy", "train_f1", "train_recall", "train_precision"]:
            if key in final:
                print(f"  {key}: {final[key]:.4f}")
        for key in ["val_accuracy", "val_f1", "val_recall", "val_precision"]:
            if key in final:
                print(f"  {key}: {final[key]:.4f}")

    # ── Evaluate on test set ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")

    test_metrics = engine.test(test_loader)
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics.get('loss', 'N/A'):.4f}")
    for name, value in test_metrics.items():
        if name != "loss":
            print(f"  {name}: {value:.4f}")

    # ── Save final checkpoint ───────────────────────────────────────────
    checkpoint_path = engine.save()
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # ── Best metric ─────────────────────────────────────────────────────
    if engine.state.best_metric is not None:
        print(f"Best validation metric: {engine.state.best_metric:.4f}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()