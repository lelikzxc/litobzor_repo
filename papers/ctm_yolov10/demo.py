#!/usr/bin/env python3
"""Reproducible demo script for YOLOv10 baseline and CTM-YOLOv10.

Creates both models, runs dummy forward passes, and prints a structured
comparison including parameter counts, CTM insertion details, and output
shapes. Uses ``experiment.py`` for metadata formatting.

Also demonstrates dataset integration with ``common.datasets``.

Usage:
    python papers/ctm_yolov10/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from common.datasets import (
    DataModule,
    build_transforms,
    split_dataset,
)
from papers.ctm_yolov10.data_utils import DetectionDataset
from papers.ctm_yolov10.models.yolov10 import CTMYOLOv10, YOLOv10Baseline
from papers.ctm_yolov10.utils.experiment import (
    build_experiment_info,
    format_experiment_info,
)


def main() -> None:
    """Run YOLOv10 baseline and CTM-YOLOv10 comparison demo."""
    print("=" * 60)
    print("  YOLOv10 Baseline  vs  CTM-YOLOv10  —  Reproducible Demo")
    print("=" * 60)

    # ── Configuration ──────────────────────────────────────────────────
    MODEL_NAME = "yolov10n"
    PRETRAINED = False
    NUM_CLASSES = 80
    BATCH_SIZE = 1
    IMAGE_SIZE = 640

    CTM_CONFIG = {
        "dim": 256,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
    }

    # ── Create models ──────────────────────────────────────────────────
    print(f"\n[1] Creating models ({MODEL_NAME}, classes={NUM_CLASSES})...")
    baseline = YOLOv10Baseline(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
        num_classes=NUM_CLASSES,
    )
    ctm_model = CTMYOLOv10(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
        num_classes=NUM_CLASSES,
        ctm_enabled=True,
        **CTM_CONFIG,
    )
    baseline.eval()
    ctm_model.eval()

    # ── Experiment metadata ────────────────────────────────────────────
    baseline_info = build_experiment_info(baseline)
    ctm_info = build_experiment_info(ctm_model)

    print(f"\n[2] Baseline metadata:")
    print(format_experiment_info(baseline_info))

    print(f"\n[3] CTM-YOLOv10 metadata:")
    print(format_experiment_info(ctm_info))

    # ── Parameter comparison ───────────────────────────────────────────
    added_params = ctm_info.total_params - baseline_info.total_params
    print(f"\n[4] Parameter comparison:")
    print(f"    Baseline params:       {baseline_info.total_params:>10,}")
    print(f"    CTM-YOLOv10 params:    {ctm_info.total_params:>10,}")
    print(f"    CTM module params:     {ctm_info.ctm_params:>10,}")
    print(f"    Added by CTM:          {added_params:>10,}")
    print(f"    Relative increase:     {added_params / baseline_info.total_params * 100:>9.2f}%")

    # ── Forward pass comparison ────────────────────────────────────────
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    print(f"\n[5] Forward pass (input: [{BATCH_SIZE}, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]):")

    with torch.no_grad():
        baseline_out = baseline(x)
        ctm_out = ctm_model(x)

    def describe_output(out: torch.Tensor | dict | tuple | list, label: str) -> None:
        """Print a structured description of a model output."""
        print(f"\n    {label}:")
        print(f"      Type:  {type(out).__name__}")
        if isinstance(out, dict):
            print(f"      Keys:  {list(out.keys())}")
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    print(f"      {k}:    {list(v.shape)}")
        elif isinstance(out, (list, tuple)):
            print(f"      Length: {len(out)}")
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    print(f"      [{i}]    {list(o.shape)}")
                else:
                    print(f"      [{i}]    {type(o).__name__}")

    describe_output(baseline_out, "YOLOv10Baseline")
    describe_output(ctm_out, "CTMYOLOv10")

    # ── Architecture summary ───────────────────────────────────────────
    print(f"\n[6] CTM insertion point:")
    print(f"    YOLOv10n backbone layers: 0-10")
    print(f"      [0-8]   Conv → C2f → SCDown (backbone)")
    print(f"      [9]     SPPF (multi-scale pooling)")
    print(f"      [10]    PSA (position-sensitive attention)")
    print(f"    CTM inserted:             after PSA (layer 10), before neck (layer 11)")
    print(f"    CTM input resolution:     [B, {CTM_CONFIG['dim']}, 20, 20] (P5 feature map)")
    print(f"    CTM output resolution:    [B, {CTM_CONFIG['dim']}, 20, 20] (preserved)")
    print(f"    CTM tokens:               400 (20 × 20)")
    print(f"    CTM attention heads:      {CTM_CONFIG['num_heads']}")
    print(f"    CTM head dimension:       {CTM_CONFIG['dim'] // CTM_CONFIG['num_heads']}")

    # ── Ablation note ──────────────────────────────────────────────────
    print(f"\n[7] Ablation support:")
    print(f"    CTM can be disabled via ``ctm_enabled=False``")
    print(f"    Disabled CTM → params match baseline exactly")
    print(f"    Enables fair comparison of CTM contribution")

    # ── Dataset integration demo ──────────────────────────────────────
    print(f"\n[8] Dataset integration (common.datasets):")
    print(f"    Creating synthetic DetectionDataset (100 samples)...")
    dataset = DetectionDataset(synthetic_size=100, image_size=640, num_classes=80)
    print(f"    Dataset type:  {dataset.dataset_type.value}")
    print(f"    Dataset len:   {len(dataset)}")
    sample = dataset[0]
    print(f"    Sample keys:   {list(sample.keys())}")
    print(f"    Image shape:   {list(sample['image'].shape)}")
    print(f"    Label shape:   {list(sample['label'].shape)}")

    print(f"\n    Building transforms...")
    transform = build_transforms(resize_size=(640, 640))
    print(f"    Transform:     {type(transform).__name__}")

    print(f"\n    Splitting dataset...")
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(f"    Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    print(f"\n    Creating DataModule...")
    dm = DataModule(
        dataset_type="classification",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=default_collate,
    )
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"    Batch image shape: {list(batch['image'].shape)}")
    print(f"    Batch label shape: {list(batch['label'].shape)}")

    # ── Training integration demo ────────────────────────────────────────
    print(f"\n[9] Training integration (common.training):")
    from papers.ctm_yolov10.utils.training import (
        YOLOLoss,
        patch_eval,
        training_collate,
    )
    from common.training import (
        Trainer,
        build_optimizer,
        build_scheduler,
        CheckpointManager,
        EarlyStopping,
        NativeScaler,
    )

    # Patch eval to keep model in train mode (YOLO eval triggers _inference)
    print(f"    Patching model.eval() for Trainer compatibility...")
    training_model = patch_eval(baseline)

    print(f"    Creating YOLOLoss adapter...")
    loss_fn = YOLOLoss(training_model)
    print(f"    Loss adapter: {type(loss_fn).__name__}")

    print(f"\n    Building optimizer (SGD, lr=1e-3, momentum=0.9)...")
    optimizer = build_optimizer(
        training_model, name="sgd", lr=1e-3, momentum=0.9, weight_decay=0.0005
    )
    print(f"    Optimizer: {type(optimizer).__name__}")

    print(f"\n    Building scheduler (cosine, T_max=10)...")
    scheduler = build_scheduler(optimizer, name="cosine", T_max=10)
    print(f"    Scheduler: {type(scheduler).__name__}")

    print(f"\n    Creating training DataLoader with training_collate...")
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=training_collate,
        shuffle=True,
    )
    batch = next(iter(train_loader))
    print(f"    Batch inputs shape:  {list(batch['inputs'].shape)}")
    print(f"    Batch targets keys:  {list(batch['targets'].keys())}")

    print(f"\n    Creating Trainer...")
    trainer = Trainer(
        model=training_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device="cpu",
        verbose=False,
    )
    print(f"    Trainer device: {trainer.device}")

    print(f"\n    Running 2 training epochs...")
    log = trainer.fit(train_loader, epochs=2)
    latest = log.latest()
    print(f"    Final train loss: {latest.get('train_loss', 'N/A'):.4f}")
    print(f"    Final LR:         {latest.get('lr', 'N/A'):.2e}")

    print(f"\n    Validating...")
    val_loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=training_collate,
        shuffle=False,
    )
    val_metrics = trainer.validate(val_loader)
    print(f"    Val loss: {val_metrics.get('loss', 'N/A'):.4f}")

    print(f"\n    Gradient flow check...")
    total_grad_norm = 0.0
    for name, param in training_model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    print(f"    Total grad norm: {total_grad_norm:.4f}")
    print(f"    Gradients flow:  {'YES' if total_grad_norm > 0 else 'NO'}")

    print(f"\n    Checkpoint save/load...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    trainer.save_checkpoint(ckpt_path)
    print(f"    Checkpoint saved: {ckpt_path}")
    loaded_epoch = trainer.load_checkpoint(ckpt_path)
    print(f"    Checkpoint loaded (epoch {loaded_epoch})")
    Path(ckpt_path).unlink(missing_ok=True)

    print(f"\n    Mixed precision (AMP) compatibility...")
    scaler = NativeScaler(enabled=True)
    trainer_amp = Trainer(
        model=training_model,
        optimizer=build_optimizer(training_model, name="adamw", lr=1e-3),
        loss_fn=YOLOLoss(training_model),
        scaler=scaler,
        device="cpu",
        verbose=False,
    )
    amp_metrics = trainer_amp.train_one_epoch(train_loader)
    print(f"    AMP train loss: {amp_metrics.get('loss', 'N/A'):.4f}")

    print(f"\n    Early stopping...")
    es = EarlyStopping(patience=3, min_delta=100.0)
    trainer_es = Trainer(
        model=training_model,
        optimizer=build_optimizer(training_model, name="sgd", lr=1e-3),
        loss_fn=YOLOLoss(training_model),
        early_stopping=es,
        device="cpu",
        verbose=False,
    )
    trainer_es.fit(train_loader, val_loader, epochs=10)
    print(f"    Stopped at epoch: {trainer_es.current_epoch} (< 10 = early stopping triggered)")

    print(f"\n    CheckpointManager with DataModule...")
    from common.datasets import DataModule as _DataModule
    from common.datasets import split_dataset as _split_dataset
    splits = _split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm = _DataModule(
        dataset_type="classification",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=4,
        collate_fn=training_collate,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(save_dir=tmpdir)
        trainer_dm = Trainer(
            model=training_model,
            optimizer=build_optimizer(training_model, name="sgd", lr=1e-3),
            loss_fn=YOLOLoss(training_model),
            checkpoint_manager=ckpt_mgr,
            device="cpu",
            verbose=False,
        )
        trainer_dm.fit(dm.train_dataloader(), dm.val_dataloader(), epochs=2)
        saved = list(Path(tmpdir).glob("*.pt"))
        print(f"    Checkpoints saved: {len(saved)}")

    print(f"\n{'=' * 60}")
    print("  Demo completed successfully.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()