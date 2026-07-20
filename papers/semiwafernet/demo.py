#!/usr/bin/env python3
"""SemiWaferNet — baseline architecture demo with engine integration.

Demonstrates model creation, parameter counts, forward pass
shapes, experiment metadata, and engine integration for the
hybrid CNN–Transformer baseline.
"""

from __future__ import annotations

import torch

from common.engine.builder import Builder
from common.engine.config import EngineConfig
from common.engine.engine import Engine
from common.engine.registry import build_model, is_registered, list_registered
from common.inference.predictor import Predictor
from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.utils.experiment import (
    build_experiment_info,
    count_params,
    format_experiment_info,
)


def semiwafernet_postprocess(logits: dict) -> dict:
    """Postprocess SemiWaferNet multitask output for Predictor compatibility.

    SemiWaferNet returns a dict with ``"classification"`` and ``"segmentation"``
    keys. This function extracts the classification logits and applies the
    standard postprocessing (softmax + argmax).

    Returns a dict with ``"logits"``, ``"probs"``, ``"prediction"`` keys.
    """
    from common.inference.postprocessing import logits_to_probs, logits_to_class

    cls_logits = logits["classification"]
    return {
        "logits": cls_logits,
        "probs": logits_to_probs(cls_logits),
        "prediction": logits_to_class(cls_logits),
    }


def main() -> None:
    print("=" * 60)
    print("SemiWaferNet — Baseline Architecture Demo")
    print("=" * 60)
    print()
    print("Paper: SemiWaferNet: Efficient Semi-Supervised Hybrid")
    print("        CNN–Transformer Models for Wafer Defect")
    print("        Classification and Segmentation")
    print()

    # ── Model creation ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    model = SemiWaferNet(
        in_channels=3,
        backbone_channels=[64, 128, 256, 512],
        backbone_depths=[2, 2, 6, 2],
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4,
        dropout=0.1,
        fusion_dim=256,
        num_classes=6,
    ).to(device)

    # ── Architecture summary ────────────────────────────────────────
    print("Architecture Summary:")
    print(f"  CNN backbone:    4 stages, channels=[64, 128, 256, 512]")
    print(f"  Transformer:     embed_dim=256, heads=8, layers=4")
    print(f"  Feature fusion:  multi-scale concat + 3x3 conv")
    print(f"  Classifier:      GAP → LayerNorm → Linear(256→6)")
    print(f"  Segmentation:    ×2 upsample → conv → ×2 upsample → 1x1 conv")
    print()

    # ── Parameter breakdown ─────────────────────────────────────────
    total_params = count_params(model)
    backbone_params = count_params(model.backbone)
    transformer_params = count_params(model.transformer)
    fusion_params = count_params(model.fusion)
    classifier_params = count_params(model.classifier)
    decoder_params = count_params(model.decoder)

    print("Parameter Breakdown:")
    print(f"  Total:              {total_params:>10,}")
    print(f"  CNN backbone:       {backbone_params:>10,}  ({100 * backbone_params / total_params:.1f}%)")
    print(f"  Transformer:        {transformer_params:>10,}  ({100 * transformer_params / total_params:.1f}%)")
    print(f"  Feature fusion:     {fusion_params:>10,}  ({100 * fusion_params / total_params:.1f}%)")
    print(f"  Classifier head:    {classifier_params:>10,}  ({100 * classifier_params / total_params:.1f}%)")
    print(f"  Segmentation dec:   {decoder_params:>10,}  ({100 * decoder_params / total_params:.1f}%)")
    print()

    # ── Experiment metadata ─────────────────────────────────────────
    info = build_experiment_info(model)
    print(format_experiment_info(info))
    print()

    # ── Forward pass ────────────────────────────────────────────────
    B, C, H, W = 2, 3, 512, 512
    x = torch.randn(B, C, H, W, device=device)
    print(f"Input shape: [{B}, {C}, {H}, {W}]")
    print()

    model.eval()
    with torch.no_grad():
        output = model(x)

    class_logits = output["classification"]
    seg_logits = output["segmentation"]

    print("Output shapes:")
    print(f"  classification: {list(class_logits.shape)}  (expected: [{B}, 6])")
    print(f"  segmentation:   {list(seg_logits.shape)}   (expected: [{B}, 6, {H}, {W}])")
    print()

    # ── Verify shapes ───────────────────────────────────────────────
    assert class_logits.shape == (B, 6), f"Classification shape mismatch: {class_logits.shape}"
    assert seg_logits.shape == (B, 6, H, W), f"Segmentation shape mismatch: {seg_logits.shape}"
    print("Shape verification: ✅ All output shapes match expectations")
    print()

    # ── Engine integration demos ────────────────────────────────────
    print("─" * 60)
    print("Engine Integration Demos")
    print("─" * 60)
    print()

    # 1. Registry
    print("1. Registry:")
    print(f"   is_registered('models', 'semiwafernet'): {is_registered('models', 'semiwafernet')}")
    print(f"   Registered models: {list_registered('models')}")
    model_from_registry = build_model("semiwafernet", num_classes=6)
    print(f"   build_model('semiwafernet'): {type(model_from_registry).__name__}")
    print()

    # 2. EngineConfig
    print("2. EngineConfig:")
    config = EngineConfig.from_yaml("papers/semiwafernet/configs/config.yaml")
    print(f"   model.name:          {config.get('model.name')}")
    print(f"   model.num_classes:   {config.get('model.num_classes')}")
    print(f"   model.backbone.channels: {config.get('model.backbone.channels')}")
    print(f"   model.transformer.embed_dim: {config.get('model.transformer.embed_dim')}")
    print()

    # 3. from_config
    print("3. from_config with EngineConfig:")
    model_from_cfg = SemiWaferNet.from_config(config)
    print(f"   Model type: {type(model_from_cfg).__name__}")
    print(f"   num_classes: {model_from_cfg.classifier.head.out_features}")
    print()

    # 4. Builder
    print("4. Builder:")
    builder = Builder(config)
    model_from_builder = builder.build_model()
    print(f"   Model type: {type(model_from_builder).__name__}")
    print()

    # 5. Predictor
    print("5. Predictor:")
    model.eval()
    predictor = Predictor(model, device="cpu", postprocess_fn=semiwafernet_postprocess)
    x_single = torch.randn(3, 128, 128)
    with torch.no_grad():
        result = predictor.predict_single(x_single)
    print(f"   predict_single keys: {list(result.keys())}")
    print(f"   logits shape: {list(result['logits'].shape)}")
    print(f"   probs shape:  {list(result['probs'].shape)}")
    print(f"   prediction:   {result['prediction']}")
    print()

    # 6. Engine
    print("6. Engine:")
    engine = Engine(model, config, device="cpu")
    summary = engine.summary()
    print(f"   Engine summary: {summary}")
    x_engine = torch.randn(3, 128, 128)
    with torch.no_grad():
        engine_result = engine.predict_single(x_engine)
    print(f"   predict_single keys: {list(engine_result.keys())}")
    print(f"   logits shape: {list(engine_result['logits'].shape)}")
    print()

    # ── Status ──────────────────────────────────────────────────────
    print("Status: ✅ Baseline architecture implemented with engine integration")
    print()

    # ═══════════════════════════════════════════════════════════════════
    #  Dataset Integration Demo
    # ═══════════════════════════════════════════════════════════════════
    print("─" * 60)
    print("Dataset Integration Demo")
    print("─" * 60)
    print()

    from common.datasets import (
        DataModule,
        build_transforms,
        multitask_collate,
        split_dataset,
    )
    from papers.semiwafernet.data_utils import LabeledWaferDataset, UnlabeledWaferDataset

    IMG_SIZE = 128  # smaller for demo speed

    # ── 1. Labeled dataset ──────────────────────────────────────────
    print("1. LabeledWaferDataset (synthetic):")
    ds_labeled = LabeledWaferDataset(synthetic_size=50, image_size=IMG_SIZE, num_classes=6)
    print(f"   Dataset: {repr(ds_labeled)}")
    sample = ds_labeled[0]
    print(f"   Keys: {list(sample.keys())}")
    print(f"   Image shape: {list(sample['image'].shape)}")
    print(f"   Label:       {sample['label']}")
    print(f"   Mask shape:  {list(sample['mask'].shape)}")
    print(f"   Mask dtype:  {sample['mask'].dtype}")
    print()

    # ── 2. Unlabeled dataset ────────────────────────────────────────
    print("2. UnlabeledWaferDataset (synthetic):")
    ds_unlabeled = UnlabeledWaferDataset(synthetic_size=100, image_size=IMG_SIZE)
    print(f"   Dataset: {repr(ds_unlabeled)}")
    u_sample = ds_unlabeled[0]
    print(f"   Keys: {list(u_sample.keys())}")
    print(f"   Image shape: {list(u_sample['image'].shape)}")
    print()

    # ── 3. Transforms ───────────────────────────────────────────────
    print("3. Transforms via common.datasets.build_transforms:")
    transform = build_transforms(resize_size=(IMG_SIZE, IMG_SIZE))
    ds_transformed = LabeledWaferDataset(
        synthetic_size=10, image_size=IMG_SIZE, num_classes=6, transform=transform,
    )
    t_sample = ds_transformed[0]
    print(f"   Transform applied: {t_sample['image'].shape}")
    print()

    # ── 4. Collation ────────────────────────────────────────────────
    print("4. Collation via common.datasets.multitask_collate:")
    batch = [ds_labeled[i] for i in range(4)]
    collated = multitask_collate(batch)
    print(f"   Batched image shape: {list(collated['image'].shape)}")
    print(f"   Batched label shape: {list(collated['label'].shape)}")
    print(f"   Batched mask shape:  {list(collated['mask'].shape)}")
    print(f"   Labels:              {collated['label'].tolist()}")
    print()

    # ── 5. Splitting ────────────────────────────────────────────────
    print("5. Splitting via common.datasets.split_dataset:")
    splits = split_dataset(ds_labeled, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(f"   Train size: {len(splits['train'])}")
    print(f"   Val size:   {len(splits['val'])}")
    print(f"   Test size:  {len(splits['test'])}")
    print()

    # ── 6. DataModule ───────────────────────────────────────────────
    print("6. DataModule integration:")
    dm = DataModule(
        dataset_type="multitask",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=multitask_collate,
    )
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    print(f"   Batch image shape: {list(batch['image'].shape)}")
    print(f"   Batch label shape: {list(batch['label'].shape)}")
    print(f"   Batch mask shape:  {list(batch['mask'].shape)}")
    print(f"   Batch size:        {batch['image'].shape[0]}")
    print()

    # ── 7. Semi-supervised pipeline ─────────────────────────────────
    print("7. Semi-supervised pipeline (labeled + unlabeled):")
    ds_labeled_full = LabeledWaferDataset(synthetic_size=30, image_size=IMG_SIZE, num_classes=6)
    ds_unlabeled_full = UnlabeledWaferDataset(synthetic_size=70, image_size=IMG_SIZE)
    splits_l = split_dataset(ds_labeled_full, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    splits_u = split_dataset(ds_unlabeled_full, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    dm_l = DataModule(
        dataset_type="multitask",
        train_dataset=splits_l["train"],
        val_dataset=splits_l["val"],
        test_dataset=splits_l["test"],
        batch_size=4,
        collate_fn=multitask_collate,
    )
    dm_u = DataModule(
        dataset_type="classification",
        train_dataset=splits_u["train"],
        val_dataset=splits_u["val"],
        test_dataset=splits_u["test"],
        batch_size=4,
    )
    labeled_batch = next(iter(dm_l.train_dataloader()))
    unlabeled_batch = next(iter(dm_u.train_dataloader()))
    print(f"   Labeled batch:   image {list(labeled_batch['image'].shape)}, "
          f"label {list(labeled_batch['label'].shape)}, "
          f"mask {list(labeled_batch['mask'].shape)}")
    print(f"   Unlabeled batch: image {list(unlabeled_batch['image'].shape)}")
    print()

    print("─" * 60)
    print("Dataset Integration Demo — Complete.")
    print("─" * 60)

    # ═══════════════════════════════════════════════════════════════════
    #  Training Integration Demo
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("Training Integration Demo")
    print("=" * 60)
    print()

    from common.training import (
        CheckpointManager,
        Trainer,
        TrainingLogger,
        accuracy,
        build_loss,
        build_optimizer,
        build_scheduler,
        f1,
    )
    from torch.utils.data import DataLoader

    # ── MultiTaskLoss adapter ────────────────────────────────────────
    class MultiTaskLoss(torch.nn.Module):
        """Loss adapter for SemiWaferNet's dict output."""

        def __init__(self, cls_loss_fn, seg_loss_fn,
                     cls_weight=1.0, seg_weight=1.0):
            super().__init__()
            self.cls_loss_fn = cls_loss_fn
            self.seg_loss_fn = seg_loss_fn
            self.cls_weight = cls_weight
            self.seg_weight = seg_weight

        def forward(self, logits, targets):
            cls_loss = self.cls_loss_fn(logits["classification"], targets["label"])
            seg_loss = self.seg_loss_fn(logits["segmentation"], targets["mask"])
            return self.cls_weight * cls_loss + self.seg_weight * seg_loss

    # ── Training collate adapter ─────────────────────────────────────
    def _training_collate(batch):
        collated = multitask_collate(batch)
        return {
            "inputs": collated["image"],
            "targets": {"label": collated["label"], "mask": collated["mask"]},
        }

    # ── Create a small model for training demo ───────────────────────
    TRAIN_IMG_SIZE = 64
    train_model = SemiWaferNet(
        in_channels=3,
        backbone_channels=[16, 32, 64, 128],
        backbone_depths=[1, 1, 1, 1],
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2,
        dropout=0.0,
        fusion_dim=64,
        num_classes=6,
    )

    # ── Synthetic dataset ────────────────────────────────────────────
    print("1. Synthetic dataset:")
    train_ds = LabeledWaferDataset(
        synthetic_size=32, image_size=TRAIN_IMG_SIZE, num_classes=6,
    )
    val_ds = LabeledWaferDataset(
        synthetic_size=16, image_size=TRAIN_IMG_SIZE, num_classes=6,
    )
    train_loader = DataLoader(
        train_ds, batch_size=8, collate_fn=_training_collate, shuffle=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, collate_fn=_training_collate, shuffle=False,
    )
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples:   {len(val_ds)}")
    print()

    # ── Training components ──────────────────────────────────────────
    print("2. Training components:")
    opt = build_optimizer(train_model, name="adamw", lr=1e-3, weight_decay=0.05)
    sched = build_scheduler(opt, name="cosine", T_max=5)
    loss_fn = MultiTaskLoss(
        cls_loss_fn=build_loss("cross_entropy"),
        seg_loss_fn=build_loss("cross_entropy"),
        cls_weight=1.0,
        seg_weight=1.0,
    )
    metric_fns = {"accuracy": accuracy, "f1": f1}
    print(f"   Optimizer: AdamW (lr=1e-3, weight_decay=0.05)")
    print(f"   Scheduler: CosineAnnealingLR (T_max=5)")
    print(f"   Loss:      MultiTaskLoss (cls_weight=1.0, seg_weight=1.0)")
    print(f"   Metrics:   accuracy, f1")
    print()

    # ── Trainer ──────────────────────────────────────────────────────
    print("3. Trainer:")
    trainer = Trainer(
        model=train_model,
        optimizer=opt,
        scheduler=sched,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        device="cpu",
        verbose=False,
    )
    print(f"   Device: {trainer.device}")
    print()

    # ── Train one epoch ──────────────────────────────────────────────
    print("4. Training (1 epoch):")
    train_metrics = trainer.train_one_epoch(train_loader)
    print(f"   Train loss: {train_metrics['loss']:.4f}")
    print(f"   Accuracy:   {train_metrics.get('accuracy', 'N/A')}")
    print(f"   F1:         {train_metrics.get('f1', 'N/A')}")
    print()

    # ── Validate ─────────────────────────────────────────────────────
    print("5. Validation:")
    val_metrics = trainer.validate(val_loader)
    print(f"   Val loss: {val_metrics['loss']:.4f}")
    print(f"   Accuracy: {val_metrics.get('accuracy', 'N/A')}")
    print()

    # ── Checkpoint ───────────────────────────────────────────────────
    print("6. Checkpoint save/load:")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "semiwafernet_demo.pt"
        trainer.save_checkpoint(ckpt_path)
        print(f"   Checkpoint saved: {ckpt_path.exists()}")

        # Modify weights and restore
        for p in train_model.parameters():
            p.data.add_(0.1)
        trainer.load_checkpoint(ckpt_path)
        print(f"   Checkpoint loaded: weights restored")
    print()

    # ── Full training loop ───────────────────────────────────────────
    print("7. Full training loop (3 epochs):")
    # Reset model
    train_model = SemiWaferNet(
        in_channels=3,
        backbone_channels=[16, 32, 64, 128],
        backbone_depths=[1, 1, 1, 1],
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2,
        dropout=0.0,
        fusion_dim=64,
        num_classes=6,
    )
    opt = build_optimizer(train_model, name="adamw", lr=1e-3)
    sched = build_scheduler(opt, name="cosine", T_max=5)
    logger = TrainingLogger()
    trainer = Trainer(
        model=train_model,
        optimizer=opt,
        scheduler=sched,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        logger=logger,
        device="cpu",
        verbose=False,
    )
    trainer.fit(train_loader, val_loader, epochs=3)
    print(f"   Epochs completed: {trainer.current_epoch}")
    latest = logger.latest()
    print(f"   Final train loss: {latest.get('train_loss', 'N/A'):.4f}")
    print(f"   Final val loss:   {latest.get('val_loss', 'N/A'):.4f}")
    print()

    print("─" * 60)
    print("Training Integration Demo — Complete.")
    print("─" * 60)


if __name__ == "__main__":
    main()