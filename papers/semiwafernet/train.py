"""Training entry point for SemiWaferNet on WM-811K.

Usage:
    python papers/semiwafernet/train.py --config papers/semiwafernet/configs/config.yaml

Trains SemiWaferNet on the WM-811K wafer map dataset using the common engine.
Supports CUDA automatically when available.

Classification mode (HybridCNN-ViT):
    - Weighted Cross-Entropy loss with w_c = 1/sqrt(n_c)
    - Batch size 256, lr=5e-5, weight_decay=4e-4
    - Data augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter
    - Hybrid sampling: None class downsampled to 30%

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
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.engine.engine import Engine
from common.training.losses import FocalLoss, DiceLoss
from common.training.metrics import accuracy, f1, precision, recall
from common.utils.cache import cache_class_counts, cache_stratified_split
from papers.semiwafernet.data_utils import (
    WaferWM811KDataset,
    SMOTEDataset,
    UnlabeledWM811KDataset,
    WaferSegmentationDataset,
)
from papers.semiwafernet.data_utils.wafer_dataset import apply_hybrid_sampling, inspect_wm811k_labels
from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.utils.checkpoint import resume_semiwafernet_engine
from papers.semiwafernet.training.stage_manager import StageManager
from papers.semiwafernet.training.trainer import Trainer as SemiWaferTrainer


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
    parser.add_argument(
        "--ssl-fast",
        action="store_true",
        help=(
            "Smoke-test SSL: 1 epoch/stage, 5k unlabeled, 5 MC passes "
            "(verifies pseudo-label accept rate and metrics quickly)"
        ),
    )
    parser.add_argument(
        "--ssl-start-stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help=(
            "Start SSL from this stage (1/2/3). Use with --resume pointing to "
            "the previous stage checkpoint, e.g. --resume checkpoints/semiwafernet/ssl_stage2.pt "
            "--ssl-start-stage 3"
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
            # w_c = 1 / sqrt(n_c)  — paper Section 2.3
            counts = torch.tensor(class_counts, dtype=torch.float32)
            weights = 1.0 / torch.sqrt(counts + 1e-8)
            weights = weights / weights.sum() * num_classes  # normalize so mean(weight) ≈ 1
            self.register_buffer("weight", weights)
            print(f"  Loss class weights (1/sqrt(n_c)): {weights.numpy()}")
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, targets, weight=self.weight)


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal loss for binary segmentation.

    From SemiWaferNet paper Equation (16): L_seg = Dice + 0.5 * Focal
    """

    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0) -> None:
        super().__init__()
        self.dice = DiceLoss(smooth=1.0)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
        # Auxiliary decoder outputs are at reduced resolutions (H/2, H/4).
        # Upsample them to the full target resolution so the base loss can
        # compare logits and targets of matching spatial size.
        # targets: [B, H, W] (pixel class indices).
        _, h, w = targets.shape
        main = logits["main"]
        aux1 = torch.nn.functional.interpolate(
            logits["aux1"], size=(h, w), mode="bilinear", align_corners=False
        )
        aux2 = torch.nn.functional.interpolate(
            logits["aux2"], size=(h, w), mode="bilinear", align_corners=False
        )
        main_loss = self.base_loss(main, targets)
        aux1_loss = self.base_loss(aux1, targets)
        aux2_loss = self.base_loss(aux2, targets)
        return main_loss + 0.3 * aux1_loss + 0.2 * aux2_loss


def compute_class_counts(
    labels: np.ndarray,
    num_classes: int,
    cache_dir: str | Path,
) -> list[int]:
    """Count samples per class in the dataset (cached on disk).

    Args:
        labels: Array of class labels for every sample.
        num_classes: Number of classes.
        cache_dir: Cache directory.

    Returns:
        List of counts per class (index = class index).
    """
    counts = cache_class_counts(labels, num_classes, cache_dir)
    print(f"  Class counts: {counts}")
    return counts


def stratified_split(
    dataset: WaferWM811KDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> tuple[Subset, Subset, Subset]:
    """Perform stratified train/val/test split (cached on disk).

    Args:
        dataset: The full dataset.
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        seed: Random seed.
        cache_dir: Cache directory. Defaults to ``<data_root>/cache``.

    Returns:
        Tuple of ``(train_subset, val_subset, test_subset)``.
    """
    if cache_dir is None:
        cache_dir = dataset.data_root / "cache"

    if hasattr(dataset, "_samples"):
        labels = np.array([lab for _, lab in dataset._samples])
    else:
        labels = np.array([dataset[i]["label"] for i in range(len(dataset))])
    train_idx, val_idx, test_idx = cache_stratified_split(
        labels,
        train_ratio,
        val_ratio,
        seed,
        cache_dir,
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def collate_fn(batch):
    """Custom collate for multitask dict-based samples (classification).

    Returns ``(images, labels)`` so the generic trainer's ``_unpack_batch``
    correctly treats labels as targets.
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return images, labels


def seg_collate_fn(batch):
    """Custom collate for segmentation samples.

    Returns ``(images, masks)`` so the generic trainer's ``_unpack_batch``
    correctly treats masks as targets.
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
    if args.mode is not None:
        config._data.setdefault("model", {})["mode"] = args.mode
    if args.ssl_fast:
        ssl = config._data.setdefault("semi_supervised", {})
        ssl["enabled"] = True
        ssl["epochs_per_stage"] = [1, 1, 1]
        ssl["unlabeled_max_samples"] = 5_000
        ssl["mc_passes"] = 5
        # Relax filters so smoke can exercise the pseudo-label path
        # (paper thresholds need a warm model; 1 epoch is not enough).
        ssl["confidence_threshold"] = 0.50
        ssl["entropy_threshold"] = 0.90
        ssl["mutual_information_threshold"] = 0.30
        config._data.setdefault("training", {})["num_epochs"] = 3
        print(
            "[SSL-FAST] Smoke mode: 1 epoch/stage, 5k unlabeled, 5 MC passes, "
            "relaxed filters (tau=0.50, H=0.90, MI=0.30)"
        )

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

    if is_segmentation:
        # Segmentation uses the pre-generated WM-811K segmentation dataset
        # (datasets/wm811k_seg) with masks derived from defective die
        # (waferMap == 2), excluding None/Random classes (paper Section 4.1).
        seg_root = config.get("data.seg_data_root", "datasets/wm811k_seg")
        seg_image_size = config.get("data.seg_image_size", 64)
        print(f"Loading WM-811K segmentation dataset from: {seg_root}")
        print(f"  Image size: {seg_image_size}x{seg_image_size}")

        train_dataset = WaferSegmentationDataset(
            data_root=seg_root, split="train", image_size=seg_image_size, train=True,
        )
        val_dataset = WaferSegmentationDataset(
            data_root=seg_root, split="val", image_size=seg_image_size, train=False,
        )
        test_dataset = WaferSegmentationDataset(
            data_root=seg_root, split="test", image_size=seg_image_size, train=False,
        )
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    else:
        # Read augmentation and hybrid_sampling config
        aug_cfg = config.get("data.augmentation", {})
        aug_enabled = aug_cfg.get("enabled", True) if isinstance(aug_cfg, dict) else True
        hybrid_cfg = config.get("data.hybrid_sampling", {})
        hybrid_enabled = hybrid_cfg.get("enabled", True) if isinstance(hybrid_cfg, dict) else True
        none_downsample_ratio = hybrid_cfg.get("none_downsample_ratio", 0.30) if isinstance(hybrid_cfg, dict) else 0.30
        use_official_split = config.get("data.use_official_split", True)
        labels_path = Path(data_root) / "labels.csv"
        label_info = inspect_wm811k_labels(labels_path)

        print(f"Loading WM-811K dataset from: {data_root}")
        print(f"  labels.csv format: {label_info['format']}")
        if label_info["format"] == "simple":
            print(
                "  NOTE: Simple image,label CSV detected. For paper-faithful SSL and "
                "official train/test split, unpack full WM-811K:\n"
                "        python scripts/unpack_lswmd.py  (requires datasets/LSWMD.pkl)"
            )
        if use_official_split and not label_info["has_official_split"]:
            print(
                "  WARNING: No trianTestLabel column - disabling official split; "
                "using stratified train/val/test instead."
            )
            use_official_split = False

        print(f"  Augmentations: {'enabled' if aug_enabled else 'disabled'}")
        print(f"  Hybrid sampling (None downsampling): {'enabled' if hybrid_enabled else 'disabled'} "
              f"(ratio={none_downsample_ratio})")
        print(f"  Official train/test partition: {'enabled' if use_official_split else 'disabled'}")

        if use_official_split:
            # Paper Section 4.1: official Training/Test partition; validation is
            # stratified from the official Training portion before re-sampling.
            # Table 1: Train(raw) 48,919 / Val 5,436 / Test 118,595
            official_train = WaferWM811KDataset(
                data_root=data_root,
                image_size=image_size,
                num_classes=num_classes,
                train=False,
                hybrid_sampling=False,
                split="training",
            )
            test_dataset = WaferWM811KDataset(
                data_root=data_root,
                image_size=image_size,
                num_classes=num_classes,
                train=False,
                hybrid_sampling=False,
                split="test",
            )
            print(f"  Official Training labeled: {len(official_train)}")
            print(f"  Official Test labeled: {len(test_dataset)}")
            print(f"  Classes: {official_train.class_names}")

            labels = np.array([lab for _, lab in official_train._samples])
            cache_dir = official_train.data_root / "cache"
            print("  Computing class counts for weighted loss (cached)...")
            class_counts = compute_class_counts(labels, num_classes, cache_dir)

            # Val ratio relative to official Training ≈ 5436/54355 ≈ 0.1 (Table 1)
            val_from_train = float(config.get("data.val_from_train", 0.1))
            print(f"  Splitting official Training -> train/val "
                  f"(val_from_train={val_from_train})...")
            from sklearn.model_selection import StratifiedShuffleSplit

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=val_from_train, random_state=42
            )
            train_idx, val_idx = next(
                sss.split(np.zeros(len(labels)), labels)
            )
            train_samples = [official_train._samples[int(i)] for i in train_idx]
            val_dataset = Subset(official_train, val_idx.tolist())
        else:
            # Legacy random stratified split over all labeled samples
            full_dataset_no_aug = WaferWM811KDataset(
                data_root=data_root,
                image_size=image_size,
                num_classes=num_classes,
                train=False,
                hybrid_sampling=False,
            )
            print(f"  Total labeled samples: {len(full_dataset_no_aug)}")
            print(f"  Classes: {full_dataset_no_aug.class_names}")

            labels = np.array([lab for _, lab in full_dataset_no_aug._samples])
            cache_dir = full_dataset_no_aug.data_root / "cache"

            print("  Computing class counts for weighted loss (cached)...")
            class_counts = compute_class_counts(labels, num_classes, cache_dir)

            print("  Performing stratified train/val/test split (cached)...")
            train_idx_subset, val_dataset, test_dataset = stratified_split(
                full_dataset_no_aug,
                train_ratio=train_split,
                val_ratio=val_split,
                seed=42,
                cache_dir=cache_dir,
            )
            train_samples = [
                full_dataset_no_aug._samples[i] for i in train_idx_subset.indices
            ]

        # Apply hybrid sampling: downsample the majority None class (Section 4.1)
        if hybrid_enabled:
            train_samples = apply_hybrid_sampling(
                train_samples,
                none_downsample_ratio=none_downsample_ratio,
                seed=42,
            )

        # Apply SMOTE to minority classes to construct a balanced training set
        # (Section 4.1: "downsampling the majority None class and applying SMOTE
        #  to minority classes")
        train_dataset = SMOTEDataset(
            data_root=data_root,
            samples=train_samples,
            image_size=image_size,
            num_classes=num_classes,
            seed=42,
        )
        # After SMOTE all classes are balanced — recompute weights from actual
        # training counts. Using pre-SMOTE frequencies makes 'none' cheap
        # (weight≈0.13) and the model over-predicts defects on imbalanced test.
        class_counts = np.bincount(train_dataset._y, minlength=num_classes).astype(int).tolist()
        print(f"  SMOTE-balanced counts for loss weights: {class_counts}")

        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ── Create DataLoaders ──────────────────────────────────────────────
    if is_segmentation:
        batch_size = config.get("seg_training.batch_size", 32)
    else:
        batch_size = config.get("training.batch_size", 256)
    eval_batch_size = config.get("evaluation.batch_size", 64)
    num_workers = 0  # safe default on Windows

    # Segmentation batches are (images, masks); classification batches are
    # (images, labels). The generic trainer's _unpack_batch uses batch[1] as
    # targets, so the correct collate must be selected per mode.
    use_seg_collate = is_segmentation
    _collate = seg_collate_fn if use_seg_collate else collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    # ── Create model ────────────────────────────────────────────────────
    print("Creating SemiWaferNet model...")
    model = SemiWaferNet.from_config(config)
    # Keep a reference to the raw multitask model (returns a dict with
    # "classification" and "segmentation" keys). The SSL pipeline (MC Dropout,
    # consistency loss) requires this dict interface, so it uses ``base_model``
    # rather than the task-specific wrapper below.
    base_model = model

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── Create loss function ────────────────────────────────────────────
    if is_segmentation:
        base_loss = DiceFocalLoss(focal_alpha=0.25, focal_gamma=2.0)
        loss_fn = DeepSupervisionLoss(base_loss)

        class SegmentationWrapper(nn.Module):
            def __init__(self, base_model: nn.Module) -> None:
                super().__init__()
                self.base_model = base_model

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                # return_aux=True returns a dict {"main", "aux1", "aux2"}
                # for deep supervision (Equation 17).
                outputs = self.base_model(x, return_aux=True)
                return outputs["segmentation"]

        model = SegmentationWrapper(model)
    else:
        # Weighted Cross-Entropy with REAL class counts (paper Section 2.3)
        loss_fn = WeightedCrossEntropyLoss(
            num_classes=num_classes,
            class_counts=class_counts,
        )

        class ClassificationWrapper(nn.Module):
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

    # Override loss function in engine AND trainer (so the trainer actually
    # uses the weighted loss), and move it to the device.
    engine.loss_fn = loss_fn
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn.to(device)
    engine.trainer.loss_fn = loss_fn

    # ── Resume from checkpoint ───────────────────────────────────────────
    if args.resume is not None:
        if args.resume == "last":
            checkpoint_path = engine.checkpoint_manager.last_path
            print(f"\nResuming from last checkpoint: {checkpoint_path}")
        elif args.resume == "best":
            checkpoint_path = engine.checkpoint_manager.best_path
            print(f"\nResuming from best checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = Path(args.resume)
            print(f"\nResuming from checkpoint: {checkpoint_path}")
        resumed_epoch = resume_semiwafernet_engine(engine, checkpoint_path)
        print(f"  Resumed at epoch {resumed_epoch}")

        if args.lr is not None:
            for param_group in engine.optimizer.param_groups:
                param_group["lr"] = args.lr
            print(f"  Overrode learning rate to: {args.lr}")

    # -- Train -----------------------------------------------------------
    if is_segmentation:
        epochs = config.get("seg_training.num_epochs", 50)
    else:
        epochs = config.get("training.num_epochs", 50)
    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'='*60}")

    # ── Semi-supervised pipeline (paper Section 2.2 / 4.1) ───────────────
    # Three-stage progressive pseudo-labeling on Dl ∪ Du.
    # Unlabeled samples come from WM-811K rows with empty failureType
    # (638,507 total; paper uses a 150,000 subsample).
    ssl_cfg = config.get("semi_supervised", {})
    ssl_enabled = ssl_cfg.get("enabled", False) if isinstance(ssl_cfg, dict) else False
    unlabeled_loader = None

    if ssl_enabled and not is_segmentation:
        unlabeled_max = int(ssl_cfg.get("unlabeled_max_samples", 150_000))
        unlabeled_root = config.get("data.unlabeled_root", None)

        if not label_info.get("has_failure_type") and not (
            unlabeled_root and Path(unlabeled_root).exists()
        ):
            print(
                "\n[SSL] DISABLED: labels.csv has no failureType column (no unlabeled pool).\n"
                "      Re-create the dataset from LSWMD.pkl for semi-supervised training:\n"
                "        python scripts/unpack_lswmd.py\n"
                "      Continuing with supervised-only training (Stage 1 only).\n"
            )
            ssl_enabled = False

        def unlabeled_collate(batch):
            # Yield plain image tensors for the SSL trainer.
            return torch.stack([item["image"] for item in batch])

        if ssl_enabled:
            if unlabeled_root and Path(unlabeled_root).exists():
                from papers.semiwafernet.data_utils import UnlabeledWaferDataset

                unlabeled_ds = UnlabeledWaferDataset(
                    image_dir=unlabeled_root,
                    image_size=image_size,
                )
                print(f"[SSL] Loaded {len(unlabeled_ds)} unlabeled samples from {unlabeled_root}")
            else:
                # Default: unlabeled pool from the same WM-811K labels.csv
                unlabeled_ds = UnlabeledWM811KDataset(
                    data_root=data_root,
                    image_size=image_size,
                    max_samples=unlabeled_max,
                    train=True,
                    seed=42,
                )
                print(
                    f"[SSL] Loaded {len(unlabeled_ds)} unlabeled WM-811K samples "
                    f"(max={unlabeled_max}) from {data_root}"
                )

            unlabeled_loader = DataLoader(
                unlabeled_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=unlabeled_collate,
            )

    if ssl_enabled and not is_segmentation and unlabeled_loader is not None:
        ssl_student = base_model

        class LabeledSSLAdapter:
            """Re-iterable labeled batches as (images, targets_dict)."""

            def __init__(self, loader: DataLoader, size: int) -> None:
                self.loader = loader
                self.size = size

            def __len__(self) -> int:
                return len(self.loader)

            def __iter__(self):
                for images, labels in self.loader:
                    seg_targets = torch.zeros(
                        images.shape[0], self.size, self.size, dtype=torch.long
                    )
                    targets = {
                        "classification": labels,
                        "segmentation": seg_targets,
                    }
                    yield images, targets

        class SSLSupervisedLoss(nn.Module):
            """Adapt WeightedCrossEntropyLoss to the SSL dict interface."""

            def __init__(self, base_loss: nn.Module) -> None:
                super().__init__()
                self.base_loss = base_loss

            def forward(
                self,
                student_output: dict[str, torch.Tensor],
                targets: dict[str, torch.Tensor],
            ) -> dict[str, torch.Tensor]:
                class_loss = self.base_loss(
                    student_output["classification"],
                    targets["classification"],
                )
                return {"classification": class_loss}

        # Split total epochs across the three SSL stages (paper: 50 epochs total)
        stage_epochs_cfg = ssl_cfg.get("epochs_per_stage", None)
        if isinstance(stage_epochs_cfg, list) and len(stage_epochs_cfg) == 3:
            e1, e2, e3 = [int(x) for x in stage_epochs_cfg]
        else:
            e1 = e2 = epochs // 3
            e3 = epochs - e1 - e2

        stage_manager = StageManager(
            student=ssl_student,
            num_classes=num_classes,
            ema_decay=ssl_cfg.get("ema_decay", 0.999),
            base_threshold=ssl_cfg.get("confidence_threshold", 0.94),
            alpha=ssl_cfg.get("alpha", 0.08),
            beta=ssl_cfg.get("beta", 0.02),
            mc_passes=ssl_cfg.get("mc_passes", 20),
            entropy_threshold=ssl_cfg.get("entropy_threshold", 0.08),
            mi_threshold=ssl_cfg.get("mutual_information_threshold", 0.12),
            consistency_weight=ssl_cfg.get("consistency_weight", 0.0),
        )
        grad_max_norm = config.get("training.grad_max_norm", 1.0)
        ssl_trainer = SemiWaferTrainer(
            student=ssl_student,
            stage_manager=stage_manager,
            optimizer=engine.optimizer,
            supervised_loss_fn=SSLSupervisedLoss(loss_fn),
            scheduler=engine.scheduler,
            device=torch.device(device),
            grad_max_norm=grad_max_norm,
        )
        labeled_ssl = LabeledSSLAdapter(train_loader, image_size)
        ssl_ckpt_dir = engine.checkpoint_manager.last_path.parent if engine.checkpoint_manager else Path("checkpoints/semiwafernet")
        start_stage = int(args.ssl_start_stage)
        if start_stage > 1 and args.resume is None:
            print(
                f"[SSL] WARNING: --ssl-start-stage {start_stage} without --resume "
                "starts from a randomly initialized model. Prefer:\n"
                "  --resume checkpoints/semiwafernet/ssl_stage{start_stage - 1}.pt "
                f"--ssl-start-stage {start_stage}"
            )
        print(
            f"[SSL] Running progressive pseudo-labeling from stage {start_stage} "
            f"(epochs/stage={e1}/{e2}/{e3})"
        )

        stage1_metrics: dict[str, float] = {"loss": float("nan"), "skipped": True}
        stage2_metrics: dict[str, float] = {"loss": float("nan"), "skipped": True}
        stage3_metrics: dict[str, float] = {"loss": float("nan"), "skipped": True}

        if start_stage <= 1:
            print("\n[SSL] Stage 1: supervised warm-up on labeled data")
            stage1_metrics = ssl_trainer.train_stage1(
                labeled_data=labeled_ssl, num_epochs=e1
            )
            stage1_ckpt = ssl_ckpt_dir / "ssl_stage1.pt"
            engine.save(stage1_ckpt)
            print(f"[SSL] Stage 1 checkpoint saved: {stage1_ckpt}")
        else:
            # Weights loaded via --resume; sync teacher before later stages.
            ssl_trainer.stage_manager.install_teacher_from_student()
            print(f"[SSL] Skipping Stage 1 (start_stage={start_stage})")

        if start_stage <= 2:
            print("\n[SSL] Stage 2: pseudo-labels on unlabeled + train Dl U Dpseudo")
            stage2_metrics = ssl_trainer.train_stage2(
                labeled_data=labeled_ssl,
                unlabeled_data=unlabeled_loader,
                num_epochs=e2,
                consistency_weight=ssl_cfg.get("consistency_weight", 0.0),
            )
            stage2_ckpt = ssl_ckpt_dir / "ssl_stage2.pt"
            engine.save(stage2_ckpt)
            print(f"[SSL] Stage 2 checkpoint saved: {stage2_ckpt}")
        else:
            print(f"[SSL] Skipping Stage 2 (start_stage={start_stage})")

        if start_stage <= 3:
            print("\n[SSL] Stage 3: refresh teacher + regenerate + retrain")
            stage3_metrics = ssl_trainer.train_stage3(
                labeled_data=labeled_ssl,
                unlabeled_data=unlabeled_loader,
                num_epochs=e3,
                consistency_weight=ssl_cfg.get("consistency_weight", 0.0),
            )
        ssl_metrics = {
            "stage1": stage1_metrics,
            "stage2": stage2_metrics,
            "stage3": stage3_metrics,
        }
        print(f"\n[SSL] Training complete: {ssl_metrics}")

        print(f"\n{'='*60}")
        print("Validating after SSL...")
        print(f"{'='*60}")
        val_metrics = engine.validate(val_loader)
        print(f"  Val Loss: {val_metrics.get('loss', 'N/A'):.4f}")
        for key, value in val_metrics.items():
            if key != "loss":
                print(f"  Val {key}: {value:.4f}")

        engine.model.to(device)
        engine.model.eval()
        logger = engine.logger
        if hasattr(logger, "log_epoch"):
            def _ssl_loss(metrics: dict) -> float:
                value = metrics.get("loss", 0.0)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return 0.0
                return value if value == value else 0.0  # NaN -> 0

            logger.log_epoch(
                ssl_stage1_loss=_ssl_loss(stage1_metrics),
                ssl_stage2_loss=_ssl_loss(stage2_metrics),
                ssl_stage3_loss=_ssl_loss(stage3_metrics),
            )
        # SSL path skips engine.fit(); ensure history exists for downstream logging
        if not getattr(logger, "history", None):
            logger.history = []
    else:
        logger = engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
        )

    # -- Final metrics ---------------------------------------------------
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

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

    # -- Evaluate on test set --------------------------------------------
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