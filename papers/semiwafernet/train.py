"""Training entry point for SemiWaferNet on WM-811K.

Usage:
    python papers/semiwafernet/train.py --config papers/semiwafernet/configs/config.yaml

Trains SemiWaferNet on the WM-811K wafer map dataset using the common engine.
Supports CUDA automatically when available.

Classification mode (HybridCNN-ViT):
    - Weighted Cross-Entropy loss with w_c = 1/sqrt(n_c)
    - Batch size 256, lr=5e-5, weight_decay=4e-4

Segmentation mode (ConvoFormer-UNet):
    - Dice + 0.5*Focal loss
    - Deep supervision: L_total = L_main + 0.3*L_aux1 + 0.2*L_aux2
    - Batch size 32, lr=1e-4, weight_decay=0.01

Semi-supervised (3-stage progressive pseudo-labeling):
    - Stage 1: supervised warm-up on D_l
    - Stage 2: pseudo-label generation + adaptive thresholding + uncertainty filtering
    - Stage 3: refresh teacher + regenerate pseudo-labels + retrain
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.engine.engine import Engine
from common.training.losses import FocalLoss, DiceLoss
from common.training.metrics import accuracy, f1, precision, recall
from papers.semiwafernet.data_utils import WaferWM811KDataset
from papers.semiwafernet.models.semiwafernet import SemiWaferNet


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SemiWaferNet on WM-811K wafer map dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="papers/semiwafernet/configs/config.yaml",
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
        "--mode",
        type=str,
        default=None,
        choices=["classification", "segmentation"],
        help="Override model mode from config",
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


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross-Entropy loss with w_c = 1 / sqrt(n_c).

    From SemiWaferNet paper Section 2.3: weights are inversely proportional
    to the square root of class frequencies to handle class imbalance.
    """

    def __init__(self, num_classes: int = 9, class_counts: list[int] | None = None) -> None:
        super().__init__()
        if class_counts is not None:
            # w_c = 1 / sqrt(n_c)
            counts = torch.tensor(class_counts, dtype=torch.float32)
            weights = 1.0 / torch.sqrt(counts + 1e-8)
            weights = weights / weights.sum() * num_classes  # normalize
            self.register_buffer("weight", weights)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, targets, weight=self.weight)


def _compute_class_counts(dataset: WaferWM811KDataset) -> list[int]:
    """Compute per-class sample counts from the dataset.

    Returns:
        List of counts indexed by class label (0..num_classes-1).
    """
    counts: list[int] = [0] * dataset.num_classes
    for _, label in dataset._samples:
        counts[label] += 1
    return counts


def _build_balanced_train_dataset(
    dataset: WaferWM811KDataset,
    none_downsample_ratio: float = 0.30,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """Build a partially balanced training set via downsampling None.

    Paper Section 4.1: downsampling the majority None class + SMOTE.
    Here we use only downsampling (SMOTE for 32x32 grayscale creates artifacts).
    The class weights in WeightedCrossEntropyLoss handle the rest.

    Args:
        dataset: The full WM-811K dataset.
        none_downsample_ratio: Fraction of None samples to keep (0.30 = ~30%).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (indices, labels).
    """
    rng = np.random.RandomState(random_state)

    # Collect per-class indices
    class_indices: dict[int, list[int]] = {c: [] for c in range(dataset.num_classes)}
    for idx, (_, label) in enumerate(dataset._samples):
        class_indices[label].append(idx)

    # Downsample None (class 0) — keep ~30% (paper uses aggressive downsampling)
    none_indices = class_indices[0]
    n_keep = max(1, int(len(none_indices) * none_downsample_ratio))
    kept_none = rng.choice(none_indices, size=n_keep, replace=False).tolist()

    # Keep all minority class samples
    minority_indices: list[int] = []
    minority_labels: list[int] = []
    for c in range(1, dataset.num_classes):
        minority_indices.extend(class_indices[c])
        minority_labels.extend([c] * len(class_indices[c]))

    # Final set: kept None + all minority
    all_indices = kept_none + minority_indices
    all_labels = [0] * len(kept_none) + minority_labels

    print(f"  Hybrid sampling (downsample None):")
    print(f"    None: {len(none_indices)} -> {n_keep}")
    print(f"    Minority: {len(minority_indices)}")
    print(f"    Total: {len(all_indices)}")

    return all_indices, all_labels


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal loss for binary segmentation.

    From SemiWaferNet paper Equation (16): L_seg = Dice + 0.5 * Focal
    """

    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0) -> None:
        super().__init__()
        self.dice = DiceLoss(smooth=1.0)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 1, H, W] or [B, 2, H, W], targets: [B, H, W]
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        return dice_loss + 0.5 * focal_loss


class DeepSupervisionLoss(nn.Module):
    """Deep supervision loss for segmentation decoder.

    From SemiWaferNet paper Equation (17):
        L_total = L_main + 0.3 * L_aux1 + 0.2 * L_aux2
    """

    def __init__(self, base_loss: nn.Module) -> None:
        super().__init__()
        self.base_loss = base_loss

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute deep supervision loss.

        Args:
            logits: Dict with "main", "aux1", "aux2" keys.
            targets: Ground truth mask [B, H, W].
        """
        main_loss = self.base_loss(logits["main"], targets)
        aux1_loss = self.base_loss(logits["aux1"], targets)
        aux2_loss = self.base_loss(logits["aux2"], targets)
        return main_loss + 0.3 * aux1_loss + 0.2 * aux2_loss


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
    if args.mode is not None:
        config._data.setdefault("model", {})["mode"] = args.mode

    # ── Resolve device ──────────────────────────────────────────────────
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Determine mode ──────────────────────────────────────────────────
    model_mode = config.get("model.mode", "classification")
    is_segmentation = model_mode == "segmentation"
    print(f"Model mode: {model_mode}")

    # ── Create dataset ──────────────────────────────────────────────────
    data_root = config.get("data.data_root", "datasets/wm811k")
    image_size = config.get("data.image_size", 32)
    num_classes = config.get("model.num_classes", 9)
    train_split = config.get("data.train_split", 0.8)
    val_split = config.get("data.val_split", 0.1)

    print(f"Loading WM-811K dataset from: {data_root}")
    full_dataset = WaferWM811KDataset(
        data_root=data_root,
        image_size=image_size,
        num_classes=num_classes,
    )
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Classes: {full_dataset.class_names}")

    # ── Compute class distribution ──────────────────────────────────────
    class_counts = _compute_class_counts(full_dataset)
    print(f"  Class distribution: {dict(zip(full_dataset.class_names, class_counts))}")

    # ── Data augmentation ───────────────────────────────────────────────
    use_augmentation = config.get("data.augmentation.enabled", True)
    if use_augmentation and not is_segmentation:
        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=0),
            transforms.ToTensor(),  # PIL -> Tensor [C, H, W]
        ])
        print(f"  Using augmentation: RandomFlip + Rotation(15°) + Affine(10%)")
    else:
        aug_transform = None

    # ── Stratified split ────────────────────────────────────────────────
    from sklearn.model_selection import StratifiedShuffleSplit

    total = len(full_dataset)
    train_len = int(total * train_split)
    val_len = int(total * val_split)
    test_len = total - train_len - val_len

    all_labels = [label for _, label in full_dataset._samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_len + test_len, random_state=42)
    train_idx, temp_idx = next(sss.split(np.arange(total), all_labels))

    temp_labels = [all_labels[i] for i in temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_len / (val_len + test_len), random_state=42)
    val_idx, test_idx = next(sss2.split(np.arange(len(temp_idx)), temp_labels))
    val_idx = [temp_idx[i] for i in val_idx]
    test_idx = [temp_idx[i] for i in test_idx]

    # Create train dataset with augmentation, val/test without
    train_dataset = WaferWM811KDataset(
        data_root=data_root,
        image_size=image_size,
        num_classes=num_classes,
        transform=aug_transform,
    )
    # Use only the training indices
    train_dataset._samples = [full_dataset._samples[i] for i in train_idx]

    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ── Hybrid sampling (downsample None) ───────────────────────────────
    # Paper Section 4.1: downsampling majority None class
    # Class weights in WeightedCrossEntropyLoss handle the imbalance further
    use_hybrid_sampling = config.get("data.hybrid_sampling.enabled", True)
    if use_hybrid_sampling and not is_segmentation:
        none_downsample_ratio = config.get("data.hybrid_sampling.none_downsample_ratio", 0.30)

        balanced_indices, balanced_labels = _build_balanced_train_dataset(
            full_dataset,
            none_downsample_ratio=none_downsample_ratio,
            random_state=42,
        )

        # Filter to only training indices
        train_idx_set = set(train_idx)
        balanced_train_indices = [i for i in balanced_indices if i in train_idx_set]
        balanced_train_labels = [balanced_labels[balanced_indices.index(i)] for i in balanced_train_indices]

        print(f"  Balanced train set: {len(balanced_train_indices)} samples")

        # Override train_dataset samples with balanced subset
        train_dataset._samples = [full_dataset._samples[i] for i in balanced_train_indices]
        train_shuffle = True
    else:
        train_shuffle = True

    # ── Create DataLoaders ──────────────────────────────────────────────
    if is_segmentation:
        batch_size = config.get("seg_training.batch_size", 32)
    else:
        batch_size = config.get("training.batch_size", 256)
    eval_batch_size = config.get("evaluation.batch_size", 64)
    num_workers = 0  # safe default on Windows

    def collate_fn(batch):
        """Custom collate for multitask dict-based samples."""
        images = torch.stack([item["image"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        masks = torch.stack([item["mask"] for item in batch])
        return images, labels, masks

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
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
    print("Creating SemiWaferNet model...")
    model = SemiWaferNet.from_config(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── Create loss function ────────────────────────────────────────────
    if is_segmentation:
        # Dice + 0.5*Focal with deep supervision
        base_loss = DiceFocalLoss(focal_alpha=0.25, focal_gamma=2.0)
        loss_fn = DeepSupervisionLoss(base_loss)

        # Wrap model to return segmentation output
        class SegmentationWrapper(nn.Module):
            """Wraps SemiWaferNet to return only segmentation output."""

            def __init__(self, base_model: nn.Module) -> None:
                super().__init__()
                self.base_model = base_model

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                outputs = self.base_model(x, return_aux=True)
                return outputs["segmentation"]

        model = SegmentationWrapper(model)
    else:
        # Weighted Cross-Entropy for classification with real class counts
        loss_fn = WeightedCrossEntropyLoss(num_classes=num_classes, class_counts=class_counts)

        # Wrap model to extract only classification output
        class ClassificationWrapper(nn.Module):
            """Wraps SemiWaferNet to return only classification logits."""

            def __init__(self, base_model: nn.Module) -> None:
                super().__init__()
                self.base_model = base_model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                outputs = self.base_model(x)
                return outputs["classification"]

        model = ClassificationWrapper(model)

    # ── Create Engine ───────────────────────────────────────────────────
    print("Initializing engine...")
    engine = Engine(
        model=model,
        config=config,
        device=device,
    )

    # Override loss function in engine
    engine.loss_fn = loss_fn

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
    if is_segmentation:
        epochs = config.get("seg_training.num_epochs", 50)
    else:
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
        print(f"  Train Loss: {final.get('train_loss', 'N/A'):.4f}" if "train_loss" in final else "")
        print(f"  Val Loss:   {final.get('val_loss', 'N/A'):.4f}" if "val_loss" in final else "")
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