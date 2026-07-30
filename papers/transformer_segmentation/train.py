"""Training entry point for SegFormer + Atrous on WM-811K segmentation.

Usage:
    python papers/transformer_segmentation/train.py --config papers/transformer_segmentation/configs/config.yaml

Trains SegFormer on the WM-811K 7-class segmentation dataset using the common engine.
Supports CUDA automatically when available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.engine.engine import Engine
from common.training.metrics import iou_score, dice_score, pixel_accuracy, f1
from papers.transformer_segmentation.data_utils import SegFormerDataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SegFormer + Atrous on WM-811K segmentation dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="papers/transformer_segmentation/configs/config.yaml",
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


def _segmentation_collate(batch):
    """Collate function for segmentation samples.

    Converts dict samples from SegFormerDataset into (inputs, targets) tuple
    expected by the common Engine/Trainer.
    """
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    return images, masks


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

    # ── Resolve device ──────────────────────────────────────────────────
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Create datasets ─────────────────────────────────────────────────
    data_root = config.get("data.data_root", "datasets/wm811k_seg")
    image_size = config.get("data.image_size", 512)
    num_classes = config.get("model.num_classes", 7)

    train_image_dir = Path(data_root) / "train" / "images"
    train_mask_dir = Path(data_root) / "train" / "masks"
    val_image_dir = Path(data_root) / "val" / "images"
    val_mask_dir = Path(data_root) / "val" / "masks"
    test_image_dir = Path(data_root) / "test" / "images"
    test_mask_dir = Path(data_root) / "test" / "masks"

    print(f"\nLoading WM-811K segmentation dataset from: {data_root}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Num classes: {num_classes} (6 defect types + background)")

    train_dataset = SegFormerDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        image_size=image_size,
        num_classes=num_classes,
    )
    val_dataset = SegFormerDataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        image_size=image_size,
        num_classes=num_classes,
    )
    test_dataset = SegFormerDataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        image_size=image_size,
        num_classes=num_classes,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    # ── Create DataLoaders ──────────────────────────────────────────────
    batch_size = config.get("training.batch_size", 10)
    eval_batch_size = config.get("evaluation.batch_size", 10)
    num_workers = 0  # safe default on Windows

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_segmentation_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_segmentation_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_segmentation_collate,
    )

    # ── Create model ────────────────────────────────────────────────────
    print("\nCreating SegFormer model...")
    from papers.transformer_segmentation.models.segformer import SegFormer
    model = SegFormer.from_config(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Variant: {model.variant}")
    print(f"  Encoder: Hybrid (conv stages 1-2 + transformer stages 3-4)")

    # ── Create Engine ───────────────────────────────────────────────────
    print("Initializing engine...")
    engine = Engine(
        model=model,
        config=config,
        device=device,
    )

    # ── Resume from checkpoint ───────────────────────────────────────────
    if args.resume is not None:
        if args.resume == "last":
            checkpoint_path = engine.checkpoint_manager.last_path
            print(f"\nResuming from last checkpoint: {checkpoint_path}")
            resumed_epoch = engine.resume(load_last=True)
        elif args.resume == "best":
            checkpoint_path = engine.checkpoint_manager.best_path
            print(f"\nResuming from best checkpoint: {checkpoint_path}")
            resumed_epoch = engine.resume(load_last=False)
        else:
            checkpoint_path = Path(args.resume)
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            resumed_epoch = engine.resume(checkpoint_path=checkpoint_path)
        print(f"  Resumed at epoch {resumed_epoch}")

    # ── Train ───────────────────────────────────────────────────────────
    epochs = config.get("training.num_epochs", 100)
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
        for key in ["train_iou", "train_dice", "train_pixel_accuracy", "train_f1"]:
            if key in final:
                print(f"  {key}: {final[key]:.4f}")
        for key in ["val_iou", "val_dice", "val_pixel_accuracy", "val_f1"]:
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