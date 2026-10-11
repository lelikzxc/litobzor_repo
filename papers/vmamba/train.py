"""Training entry point for FCS-VMamba on WM-811K.

Usage (paper Subset A, ~902 images, 50 epochs):
    python papers/vmamba/train.py --config papers/vmamba/configs/config.yaml

Trains FCS-VMamba on the WM-811K wafer map dataset using the common engine.
Default protocol matches the paper Sec. 4.2 balanced benchmark (Subset A).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
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
from common.training.utils import resolve_device
from common.utils.seed import set_seed
from papers.reproduction import save_protocol
from papers.vmamba.data_utils.author_dataset import build_dataset, validate_dataset_images
from papers.vmamba.data_utils.protocol import (
    balanced_indices as _balanced_indices,
)
from papers.vmamba.data_utils.protocol import (
    stratified_split as _stratified_split,
)
from papers.vmamba.kernels import csms6s
from papers.vmamba.models.vmamba import FCSVMamba
from papers.vmamba.utils.checkpoint import resume_vmamba_engine, validate_resume_protocol
from papers.vmamba.utils.console import EpochLogger, ProgressLoader, training_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training (auto: CUDA, then MPS, then CPU)",
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
        help="Maximum samples per class; default keeps the complete 902-image author archive. "
        "Use 0 to keep all samples.",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Extra stratified fraction of the train split (e.g. 0.2).",
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
        help="Legacy diagnostic option: force the slow sequential reference scan",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--save-dir", type=Path, default=None, help="Checkpoint directory for this run"
    )
    parser.add_argument(
        "--log-file", type=Path, default=None, help="UTF-8 log without progress redraws"
    )
    parser.add_argument(
        "--scan-backend",
        choices=["auto", "chunked", "compiled"],
        default="auto",
        help="auto prefers a compiled CUDA scan, then the portable checkpointed PyTorch scan",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check data, a training update and evaluation batch; do not save or train epochs",
    )
    return parser.parse_args(argv)


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


def ssm_optimizer(model, config):
    """Respect the SS2D A/D parameter flags instead of decaying state dynamics."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(parameter)
    kwargs = dict(config.get("optimizer.kwargs", {}) or {})
    kwargs.pop("weight_decay", None)
    kwargs.pop("lr", None)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config.get("optimizer.weight_decay", 0.05))},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config.get("optimizer.lr", 0.001)),
        **kwargs,
    )


def configure_ssm_optimizer(engine, config):
    from common.engine.builder import Builder

    engine.optimizer = ssm_optimizer(engine.model, config)
    engine.scheduler = Builder(config).build_scheduler(engine.optimizer)
    engine.trainer.optimizer = engine.optimizer
    engine.trainer.scheduler = engine.scheduler


def validate_run_config(config, args, seed, per_class):
    for key, default in (
        ("training.num_epochs", 50),
        ("training.batch_size", 4),
        ("evaluation.batch_size", 16),
    ):
        value = config.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(per_class) is not int or per_class < 0:
        raise ValueError("data.per_class / --per-class must be a nonnegative integer")
    for key, default, positive in (
        ("optimizer.lr", 0.001, True),
        ("optimizer.weight_decay", 0.05, False),
    ):
        value = float(config.get(key, default))
        if not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(
                f"{key} must be finite and {'positive' if positive else 'nonnegative'}"
            )
    image_size = config.get("data.image_size", 224)
    if type(image_size) is not int or image_size < 32 or image_size % 32:
        raise ValueError("data.image_size must be a positive multiple of 32")
    if config.get("model.input.image_size", image_size) != image_size:
        raise ValueError("model.input.image_size and data.image_size must match")
    if config.get("model.num_classes", 9) != 9:
        raise ValueError("WM-811K classification requires model.num_classes=9")
    if config.get("model.input.channels", 3) != 3:
        raise ValueError("VMamba RGB maps require model.input.channels=3")
    if config.get("model.backbone.pretrained", False):
        raise ValueError(
            "This FCS recipe trains from scratch; no pretrained weights are configured"
        )
    if not 0 < float(config.get("data.train_split", 0.8)) < 1:
        raise ValueError("data.train_split must be in (0, 1)")
    if not 0 < float(config.get("data.train_subset", 1.0)) <= 1:
        raise ValueError("data.train_subset / --subset must be in (0, 1]")
    if args.allow_slow_scan and args.scan_backend != "auto":
        raise ValueError("Use either --scan-backend or legacy --allow-slow-scan")
    if args.check_only and args.resume is not None:
        raise ValueError("--check-only checks a fresh model; omit --resume")


def _resume_path(config, args):
    directory = Path(config.get("checkpoint.save_dir", "checkpoints/vmamba_fcs_reproduction"))
    if args.resume is not None:
        path = (
            directory / f"{args.resume}.pt"
            if args.resume in {"last", "best"}
            else Path(args.resume)
        )
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path
    if not args.check_only and directory.is_dir() and any(directory.glob("*.pt")):
        raise ValueError(
            f"Checkpoints already exist in {directory}; use a new --save-dir or --resume"
        )
    return None


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def check_model_batches(model, config, device, train_loader, val_loader):
    """Probe the actual configured shapes without modifying saved experiment state."""
    from common.engine.builder import Builder

    model.to(device).train()
    optimizer = (
        ssm_optimizer(model, config)
        if config.get("optimizer.name", "adamw").lower() == "adamw"
        else Builder(config).build_optimizer(model)
    )
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    started = time.perf_counter()
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Preflight training loss is not finite")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
        ):
            raise RuntimeError(f"Preflight gradient is missing or nonfinite: {name}")
    max_norm = config.get("training.grad_max_norm", 1.0)
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Preflight optimizer produced nonfinite weights")
    synchronize(device)
    seconds = time.perf_counter() - started
    print(f"Training update OK: batch={len(labels)}, loss={loss.item():.6f}, seconds={seconds:.3f}")
    model.eval()
    images, labels = next(iter(val_loader))
    with torch.no_grad():
        logits = model(images.to(device))
        eval_loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
        if not torch.isfinite(logits).all() or not torch.isfinite(eval_loss):
            raise RuntimeError("Preflight evaluation is not finite")
    synchronize(device)
    print(f"Evaluation batch OK: batch={len(labels)}, loss={eval_loss.item():.6f}")
    if device == "cuda":
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"CUDA peak allocated/reserved: {allocated:.2f}/{reserved:.2f} GiB")
    print("Preflight complete. No checkpoints or training protocol were written.")


def main() -> None:
    args = parse_args()
    with training_output(args.log_file):
        _run_training(args)


def _run_training(args) -> None:

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = EngineConfig.from_yaml(config_path)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    per_class = (
        args.per_class
        if args.per_class is not None
        else config.get(
            "data.per_class", 0 if config.get("data.source") == "author_archive" else 100
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
    config._data["seed"] = seed
    config._data.setdefault("data", {})["per_class"] = per_class
    config._data["data"]["augmentation_enabled"] = (
        bool(config.get("data.augmentation_enabled", True)) and not args.no_aug
    )
    if args.subset is not None:
        config._data["data"]["train_subset"] = args.subset
    if args.save_dir is not None:
        config._data.setdefault("checkpoint", {})["save_dir"] = str(args.save_dir)
    validate_run_config(config, args, seed, per_class)
    checkpoint_path = _resume_path(config, args)
    set_seed(seed)

    device = resolve_device(args.device)

    csms6s.set_scan_backend("torch" if args.allow_slow_scan else args.scan_backend)
    backend = csms6s.scan_backend_info(device)

    print(f"Using device: {device}")
    print(f"Selective scan backend: {backend['backend']}")
    if backend["import_errors"]:
        print(f"  Unavailable optional CUDA extensions: {backend['import_errors']}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_root = config.get("data.data_root", "datasets/wm811k")

    print(f"Loading WM-811K labeled maps from: {data_root}")
    # Paper default: the pinned author RGB archive, resized to 224 pixels.
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

    ratio = float(config.get("data.train_subset", 1.0))
    if ratio < 1.0:
        train_idx = _stratified_subset(train_idx, labels, ratio, seed=seed)
        print(f"  Extra train subset: {len(train_idx)} ({ratio * 100:.0f}%)")

    checked = validate_dataset_images(base_dataset, [*train_idx, *val_idx])
    print(f"  Image decoding verified: {checked} selected images")

    # Train dataset with paper augmentations
    if not config.get("data.augmentation_enabled"):
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

    splits = {"train": train_idx, "holdout": val_idx}
    if checkpoint_path is not None:
        validate_resume_protocol(checkpoint_path, config, base_dataset._samples, splits, seed)
    if not args.check_only:
        save_protocol(
            config.get("checkpoint.save_dir"), config, base_dataset._samples, splits, seed
        )

    batch_size = config.get("training.batch_size", 4)
    eval_batch_size = config.get("evaluation.batch_size", 16)
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

    if args.check_only:
        try:
            check_model_batches(model, config, device, train_loader, val_loader)
        except RuntimeError as exc:
            if device == "cuda" and backend["backend"] in {"oflex", "core", "mamba"}:
                raise RuntimeError(
                    "CUDA preflight failed with the compiled scan; retry --scan-backend chunked. "
                    f"Original error: {exc}"
                ) from exc
            raise
        return

    print("Initializing engine...")
    engine = Engine(
        model=model,
        config=config,
        device=device,
    )
    engine.trainer.verbose = False
    engine.logger = EpochLogger(lambda: engine.trainer.current_epoch)
    engine.trainer.logger = engine.logger

    if config.get("optimizer.name", "adamw").lower() == "adamw":
        configure_ssm_optimizer(engine, config)

    if args.resume is not None:
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        resumed_epoch = resume_vmamba_engine(engine, checkpoint_path)
        print(f"  Resumed at epoch {resumed_epoch}")

    epochs = max(0, int(config.get("training.num_epochs", 50)) - engine.trainer.current_epoch)
    print(f"\n{'=' * 60}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'=' * 60}")

    if epochs:
        total_epochs = config.get("training.num_epochs", 50)
        logger = engine.fit(
            train_loader=ProgressLoader(
                train_loader,
                desc=lambda: f"Epoch {engine.trainer.current_epoch}/{total_epochs}",
            ),
            val_loader=ProgressLoader(val_loader, desc="Validation"),
            epochs=epochs,
        )
    else:
        print("Configured total epochs already completed; evaluating the saved run")
        logger = engine.logger

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
    print("Evaluating on the saved validation holdout...")
    print(f"{'=' * 60}")

    # Prefer best val checkpoint for final numbers.
    best_path = engine.checkpoint_manager.best_path
    if best_path is not None and Path(best_path).exists():
        print(f"Loading best checkpoint: {best_path}")
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        engine.model.load_state_dict(state["model"])

    test_metrics = engine.test(ProgressLoader(test_loader, desc="Holdout"))
    print("\nHoldout Results (best checkpoint, used for selection):")
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
