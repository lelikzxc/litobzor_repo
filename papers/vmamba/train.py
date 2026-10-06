"""Training entry point for FCS-VMamba on WM-811K.

Usage (paper Subset A, ~902 images, 50 epochs):
    python papers/vmamba/train.py --config papers/vmamba/configs/config.yaml

Trains FCS-VMamba on the WM-811K wafer map dataset using the common engine.
Default protocol matches the paper Sec. 4.2 balanced benchmark (Subset A).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.engine.engine import Engine
from common.utils.seed import set_seed
from papers.reproduction import save_protocol
from papers.vmamba.data_utils.author_dataset import build_dataset
from papers.vmamba.data_utils.protocol import (
    balanced_indices as _balanced_indices,
)
from papers.vmamba.data_utils.protocol import (
    stratified_split as _stratified_split,
)
from papers.vmamba.models.vmamba import FCSVMamba


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train FCS-VMamba on WM-811K wafer map dataset")
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
        "--per-class",
        type=int,
        default=None,
        help="Paper Subset A: max samples per class (default 100 → ~902). "
        "Use 0 to keep all labeled samples.",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Extra stratified fraction of the train split after balancing "
        "(e.g. 0.2). Ignored for the default paper protocol.",
    )
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="Disable paper training augmentations",
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
        "--allow-slow-scan",
        action="store_true",
        help="Allow the slow PyTorch selective scan on CUDA without compiled kernels",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _stratified_subset(
    indices: list[int],
    labels: list[int],
    subset_ratio: float,
    seed: int = 42,
) -> list[int]:
    y = np.array([labels[i] for i in indices])
    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=subset_ratio,
        random_state=seed,
    )
    keep, _ = next(sss.split(np.zeros(len(indices)), y))
    return [indices[i] for i in keep]


def configure_ssm_optimizer(engine, config):
    """Respect the SS2D A/D parameter flags instead of decaying its state dynamics."""
    from common.engine.builder import Builder

    decay, no_decay = [], []
    for parameter in engine.model.parameters():
        (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(parameter)
    kwargs = dict(config.get("optimizer.kwargs", {}) or {})
    kwargs.pop("weight_decay", None)
    engine.optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config.get("optimizer.weight_decay", 0.05))},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config.get("optimizer.lr", 0.001)),
        **kwargs,
    )
    engine.scheduler = Builder(config).build_scheduler(engine.optimizer)
    engine.trainer.optimizer = engine.optimizer
    engine.trainer.scheduler = engine.scheduler


def main() -> None:
    """Run the training loop."""
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = EngineConfig.from_yaml(config_path)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)
    per_class = (
        args.per_class
        if args.per_class is not None
        else int(
            config.get(
                "data.per_class", 0 if config.get("data.source") == "author_archive" else 100
            )
        )
    )

    if args.epochs is not None:
        config._data.setdefault("training", {})["num_epochs"] = args.epochs
        # Do not rewrite scheduler T_max: short smokes keep the paper cosine
        # schedule (e.g. T_max=50) so the first N epochs match a full run.
    if args.batch_size is not None:
        config._data.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config._data.setdefault("training", {})["learning_rate"] = args.lr
        config._data.setdefault("training", {}).setdefault("optimizer", {})["lr"] = args.lr
        config._data.setdefault("optimizer", {})["lr"] = args.lr

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not args.allow_slow_scan:
        from papers.vmamba.kernels import csms6s

        if not (
            csms6s.WITH_SELECTIVESCAN_OFLEX
            or csms6s.WITH_SELECTIVESCAN_CORE
            or csms6s.WITH_SELECTIVESCAN_MAMBA
        ):
            raise RuntimeError(
                "CUDA training requires a compiled selective-scan kernel; see papers/vmamba/README.md or explicitly use --allow-slow-scan"
            )

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_root = config.get("data.data_root", "datasets/wm811k")

    print(f"Loading WM-811K labeled maps from: {data_root}")
    # Locked baseline encoding: one-hot die states + bilinear → image_size.
    base_dataset = build_dataset(config, train=False)
    labels = [y for _, y in base_dataset._samples]
    print(f"  Labeled samples: {len(base_dataset)}")
    print(f"  Classes: {base_dataset.class_names}")

    if per_class > 0:
        pool = _balanced_indices(labels, per_class=per_class, seed=seed)
        print(f"  Balanced subset (<={per_class}/class): {len(pool)} images")
    else:
        pool = list(range(len(base_dataset)))
        print(f"  Using all labeled samples: {len(pool)}")

    train_idx, val_idx = _stratified_split(
        pool, labels, train_ratio=float(config.get("data.train_split", 0.8)), seed=seed
    )

    if args.subset is not None:
        ratio = float(args.subset)
        if ratio <= 0.0 or ratio > 1.0:
            print(f"Error: --subset must be in (0, 1], got {ratio}")
            sys.exit(1)
        if ratio < 1.0:
            train_idx = _stratified_subset(train_idx, labels, ratio, seed=seed)
        print(f"  Extra train subset: {len(train_idx)} ({ratio * 100:.0f}%)")

    # Train dataset with paper augmentations
    if args.no_aug:
        train_dataset = Subset(base_dataset, train_idx)
    else:
        aug_dataset = build_dataset(config, train=True)
        train_dataset = Subset(aug_dataset, train_idx)

    # Paper Sec. 4.1: stratified 8:2 — holdout is both val (during fit) and test.
    val_dataset = Subset(base_dataset, val_idx)
    test_dataset = val_dataset

    print(
        f"  Train: {len(train_dataset)}, Val/Test: {len(val_dataset)} (paper 8:2 on balanced pool)"
    )

    save_protocol(
        config.get("checkpoint.save_dir"),
        config,
        base_dataset._samples,
        {"train": train_idx, "holdout": val_idx},
        seed,
    )

    batch_size = config.get("training.batch_size", 4)
    eval_batch_size = config.get("evaluation.batch_size", 32)
    num_workers = 0

    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        labels_t = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return images, labels_t

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

    print("Creating FCS-VMamba model...")
    model = FCSVMamba.from_config(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    print("Initializing engine...")
    engine = Engine(
        model=model,
        config=config,
        device=device,
    )

    if config.get("optimizer.name", "adamw").lower() == "adamw":
        configure_ssm_optimizer(engine, config)

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
        # Common Engine.load restores trainer state, not the manager's best score.
        if engine.checkpoint_manager.best_path.exists():
            best_state = torch.load(
                engine.checkpoint_manager.best_path, map_location="cpu", weights_only=False
            )
            if best_state.get("metric") is not None:
                engine.checkpoint_manager._best_metric = float(best_state["metric"])

    epochs = max(0, int(config.get("training.num_epochs", 50)) - engine.trainer.current_epoch)
    print(f"\n{'=' * 60}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'=' * 60}")

    logger = engine.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")

    history = logger.history
    if history:
        final = history[-1]
        print("\nFinal training metrics:")
        if "train_loss" in final:
            print(f"  Train Loss: {final['train_loss']:.4f}")
        if "val_loss" in final:
            print(f"  Val Loss:   {final['val_loss']:.4f}")
        for key in [
            "train_accuracy",
            "train_f1",
            "train_recall",
            "train_precision",
            "val_accuracy",
            "val_f1",
            "val_recall",
            "val_precision",
        ]:
            if key in final:
                print(f"  {key}: {final[key]:.4f}")

    print(f"\n{'=' * 60}")
    print("Evaluating on test set...")
    print(f"{'=' * 60}")

    # Prefer best val checkpoint for final numbers.
    best_path = engine.checkpoint_manager.best_path
    if best_path is not None and Path(best_path).exists():
        print(f"Loading best checkpoint: {best_path}")
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        engine.model.load_state_dict(state["model"])

    test_metrics = engine.test(test_loader)
    print("\nTest Results (best ckpt):")
    loss_v = test_metrics.get("loss", None)
    if isinstance(loss_v, (int, float)):
        print(f"  Loss: {loss_v:.4f}")
    for name, value in test_metrics.items():
        if name != "loss" and isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}")

    # Preserve last.pt as the resumable final epoch; best weights are for reporting.
    checkpoint_path = engine.checkpoint_manager.last_path
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    if engine.state.best_metric is not None:
        print(f"Best validation metric: {engine.state.best_metric:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
