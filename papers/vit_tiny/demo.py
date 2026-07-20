#!/usr/bin/env python3
"""Demo script for Tiny Vision Transformer (ViT-Tiny).

Creates the ViT-Tiny model, runs a dummy forward pass, and prints
architecture details including output shape and parameter count.

Also demonstrates integration with the common engine infrastructure:
registry, EngineConfig, Predictor, and Engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from papers.vit_tiny.models.vit_tiny import ViTTiny


def main() -> None:
    """Run ViT-Tiny demo."""
    print("=" * 56)
    print("  Tiny Vision Transformer Demo")
    print("=" * 56)

    # ── Configuration ──────────────────────────────────────────────────
    IMAGE_SIZE = 32
    NUM_CLASSES = 8
    BATCH_SIZE = 4
    IN_CHANNELS = 1

    # ── Create model ───────────────────────────────────────────────────
    print(f"\n[1] Creating model...")
    model = ViTTiny(
        image_size=IMAGE_SIZE,
        patch_size=4,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        embed_dim=192,
        num_layers=4,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
        emb_dropout=0.1,
    )
    model.eval()

    # ── Architecture summary ───────────────────────────────────────────
    print(f"\n[2] Architecture summary:")
    print(f"    Model:              ViT-Tiny")
    print(f"    Image size:         {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"    Patch size:         {model.patch_size}")
    print(f"    In channels:        {model.patch_embed.proj.in_channels}")
    print(f"    Embed dim:          {model.embed_dim}")
    print(f"    Num layers:         {model.num_layers}")
    print(f"    Num heads:          {model.num_heads}")
    print(f"    MLP ratio:          4.0")
    print(f"    Num classes:        {model.num_classes}")
    print(f"    Parameters:         {sum(p.numel() for p in model.parameters()):,}")

    # ── Forward pass ───────────────────────────────────────────────────
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    print(f"\n[3] Forward pass:")
    print(f"    Input shape:        [{BATCH_SIZE}, {IN_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}]")

    with torch.no_grad():
        out = model(x)

    print(f"    Output shape:       {list(out.shape)}")
    print(f"    Output type:        logits (no Softmax)")
    print(f"    Sample logits:\n{out}")

    # ═══════════════════════════════════════════════════════════════════
    #  Engine Integration Demo
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 56}")
    print("  Engine Integration Demo")
    print(f"{'=' * 56}")

    # ── 1. Registry construction ──────────────────────────────────────
    print(f"\n[E1] Registry construction:")
    from common.engine.registry import build_model, is_registered, list_registered

    assert is_registered("models", "vit_tiny"), "vit_tiny should be registered"
    registered_models = list_registered("models")
    print(f"    Registered models: {registered_models}")
    print(f"    vit_tiny in registry: {'vit_tiny' in registered_models}")

    # ── 2. EngineConfig loading ───────────────────────────────────────
    print(f"\n[E2] EngineConfig loading:")
    from common.engine.config import EngineConfig

    config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")
    print(f"    model.name:           {config.get('model.name')}")
    print(f"    model.num_classes:    {config.get('model.num_classes')}")
    print(f"    model.arch.embed_dim: {config.get('model.arch.embed_dim')}")
    print(f"    training.optimizer:   {config.get('training.optimizer')}")
    print(f"    training.scheduler:   {config.get('training.scheduler')}")
    print(f"    training.loss:        {config.get('training.loss')}")
    print(f"    dataset.name:         {config.get('dataset.name')}")

    # ── 3. build_model() by registered name ───────────────────────────
    print(f"\n[E3] build_model() by registered name:")
    model_from_registry = build_model(
        "vit_tiny",
        image_size=IMAGE_SIZE,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
    )
    model_from_registry.eval()
    print(f"    Model type: {type(model_from_registry).__name__}")
    print(f"    num_classes: {model_from_registry.num_classes}")

    # ── 4. from_config() with EngineConfig ────────────────────────────
    print(f"\n[E4] from_config() with EngineConfig:")
    model_from_cfg = ViTTiny.from_config(config)
    model_from_cfg.eval()
    print(f"    Model type: {type(model_from_cfg).__name__}")
    print(f"    num_classes: {model_from_cfg.num_classes}")
    print(f"    embed_dim: {model_from_cfg.embed_dim}")

    # ── 5. Predictor inference ────────────────────────────────────────
    print(f"\n[E5] Predictor inference:")
    from common.inference.predictor import Predictor

    predictor = Predictor(model, device="cpu")
    x_single = torch.randn(IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    result = predictor.predict_single(x_single)
    print(f"    logits shape:    {list(result['logits'].shape)}")
    print(f"    probs shape:     {list(result['probs'].shape)}")
    print(f"    prediction:      {result['prediction'].item()}")
    print(f"    probs sum:       {result['probs'].sum().item():.4f}")

    # ── 6. Batch inference via Predictor ──────────────────────────────
    print(f"\n[E6] Batch inference via Predictor:")
    x_batch = torch.randn(BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    batch_result = predictor.predict_batch(x_batch)
    print(f"    batch logits shape: {list(batch_result['logits'].shape)}")
    print(f"    batch predictions:  {batch_result['prediction'].tolist()}")

    # ── 7. Engine instantiation ───────────────────────────────────────
    print(f"\n[E7] Engine instantiation:")
    from common.engine.engine import Engine

    engine = Engine(model, config, device="cpu")
    summary = engine.summary()
    print(f"    Engine model: {summary['model']}")
    print(f"    Engine device: {summary['device']}")

    # ── 8. Engine predict_single ──────────────────────────────────────
    print(f"\n[E8] Engine predict_single:")
    engine_result = engine.predict_single(x_single)
    print(f"    logits shape:    {list(engine_result['logits'].shape)}")
    print(f"    probs shape:     {list(engine_result['probs'].shape)}")
    print(f"    prediction:      {engine_result['prediction'].item()}")

    # ── 9. Forward pass output shapes ─────────────────────────────────
    print(f"\n[E9] Forward pass output shapes:")
    with torch.no_grad():
        logits = model(x_batch)
    print(f"    Input:  [{BATCH_SIZE}, {IN_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}]")
    print(f"    Output: {list(logits.shape)}  (logits, no Softmax)")
    print(f"    Expected: [{BATCH_SIZE}, {NUM_CLASSES}]")

    print(f"\n{'=' * 56}")
    print("  ViT-Tiny Demo — Complete.")
    print(f"{'=' * 56}")

    # ═══════════════════════════════════════════════════════════════════
    #  Dataset Integration Demo
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 56}")
    print("  Dataset Integration Demo")
    print(f"{'=' * 56}")

    from common.datasets import (
        DataModule,
        build_transforms,
        classification_collate,
        split_dataset,
    )
    from papers.vit_tiny.data_utils import ViTTinyDataset

    # ── 1. Synthetic dataset ─────────────────────────────────────────
    print(f"\n[D1] Synthetic dataset creation:")
    ds_synthetic = ViTTinyDataset(synthetic_size=100, image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    print(f"    Dataset: {repr(ds_synthetic)}")
    sample = ds_synthetic[0]
    print(f"    Sample keys: {list(sample.keys())}")
    print(f"    Image shape: {list(sample['image'].shape)}")
    print(f"    Label:       {sample['label']}")

    # ── 2. Transforms ────────────────────────────────────────────────
    print(f"\n[D2] Transforms via common.datasets.build_transforms:")
    transform = build_transforms(resize_size=(IMAGE_SIZE, IMAGE_SIZE))
    ds_transformed = ViTTinyDataset(
        synthetic_size=10,
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        transform=transform,
    )
    sample_t = ds_transformed[0]
    print(f"    Transform applied: {sample_t['image'].shape}")

    # ── 3. Collation ─────────────────────────────────────────────────
    print(f"\n[D3] Collation via common.datasets.classification_collate:")
    batch = [ds_synthetic[i] for i in range(4)]
    collated = classification_collate(batch)
    print(f"    Batched image shape: {list(collated['image'].shape)}")
    print(f"    Batched label shape: {list(collated['label'].shape)}")
    print(f"    Labels:              {collated['label'].tolist()}")

    # ── 4. Splitting ─────────────────────────────────────────────────
    print(f"\n[D4] Splitting via common.datasets.split_dataset:")
    splits = split_dataset(ds_synthetic, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(f"    Train size: {len(splits['train'])}")
    print(f"    Val size:   {len(splits['val'])}")
    print(f"    Test size:  {len(splits['test'])}")

    # ── 5. DataModule ────────────────────────────────────────────────
    print(f"\n[D5] DataModule integration:")
    dm = DataModule(
        dataset_type="classification",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=classification_collate,
    )
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"    Batch image shape: {list(batch['image'].shape)}")
    print(f"    Batch label shape: {list(batch['label'].shape)}")
    print(f"    Batch size:        {batch['image'].shape[0]}")

    print(f"\n{'=' * 56}")
    print("  Dataset Integration Demo — Complete.")
    print(f"{'=' * 56}")

    # ═══════════════════════════════════════════════════════════════════
    #  Training Integration Demo
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 56}")
    print("  Training Integration Demo")
    print(f"{'=' * 56}")

    from common.training import (
        CheckpointManager,
        EarlyStopping,
        NativeScaler,
        Trainer,
        TrainingLogger,
        accuracy,
        build_loss,
        build_optimizer,
        build_scheduler,
        f1,
    )
    from common.datasets import (
        DataModule,
        classification_collate,
        split_dataset,
    )
    from papers.vit_tiny.data_utils import ViTTinyDataset

    # Adapter collate: maps classification_collate output to Trainer's expected format
    def _training_collate(batch):
        collated = classification_collate(batch)
        return {"inputs": collated["image"], "targets": collated["label"]}

    # ── 1. Synthetic dataset ─────────────────────────────────────────
    print(f"\n[T1] Synthetic dataset creation:")
    ds = ViTTinyDataset(synthetic_size=64, image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    splits = split_dataset(ds, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(f"    Train size: {len(splits['train'])}")
    print(f"    Val size:   {len(splits['val'])}")
    print(f"    Test size:  {len(splits['test'])}")

    # ── 2. DataModule ────────────────────────────────────────────────
    print(f"\n[T2] DataModule creation:")
    dm = DataModule(
        dataset_type="classification",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=_training_collate,
    )
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches:   {len(val_loader)}")

    # ── 3. Trainer creation ──────────────────────────────────────────
    print(f"\n[T3] Trainer creation:")
    model_for_train = ViTTiny(
        image_size=IMAGE_SIZE,
        patch_size=4,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        embed_dim=192,
        num_layers=4,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
        emb_dropout=0.1,
    )
    opt = build_optimizer(model_for_train, name="adamw", lr=1e-3, weight_decay=0.05)
    sched = build_scheduler(opt, name="cosine", T_max=10)
    loss_fn = build_loss("cross_entropy")
    trainer = Trainer(
        model=model_for_train,
        optimizer=opt,
        loss_fn=loss_fn,
        scheduler=sched,
        device="cpu",
        metric_fns={"accuracy": accuracy, "f1": f1},
        verbose=False,
    )
    print(f"    Trainer created on device: {trainer.device}")

    # ── 4. One epoch of training ─────────────────────────────────────
    print(f"\n[T4] One epoch of training:")
    train_metrics = trainer.train_one_epoch(train_loader)
    print(f"    Train loss: {train_metrics['loss']:.4f}")
    print(f"    Train accuracy: {train_metrics.get('accuracy', 'N/A')}")

    # ── 5. Validation ────────────────────────────────────────────────
    print(f"\n[T5] Validation:")
    val_metrics = trainer.validate(val_loader)
    print(f"    Val loss: {val_metrics['loss']:.4f}")
    print(f"    Val accuracy: {val_metrics.get('accuracy', 'N/A')}")

    # ── 6. Full fit with checkpointing ───────────────────────────────
    print(f"\n[T6] Full fit with checkpointing (3 epochs):")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(save_dir=tmpdir)
        es = EarlyStopping(patience=5, min_delta=0.001)
        trainer2 = Trainer(
            model=ViTTiny(
                image_size=IMAGE_SIZE,
                patch_size=4,
                in_channels=IN_CHANNELS,
                num_classes=NUM_CLASSES,
            ),
            optimizer=build_optimizer(
                ViTTiny(image_size=IMAGE_SIZE, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES),
                name="adamw", lr=1e-3,
            ),
            loss_fn=build_loss("cross_entropy"),
            scheduler=build_scheduler(opt, name="cosine", T_max=10),
            device="cpu",
            metric_fns={"accuracy": accuracy},
            checkpoint_manager=ckpt_mgr,
            early_stopping=es,
            verbose=False,
        )
        log = trainer2.fit(train_loader, val_loader, epochs=3)
        latest = log.latest()
        print(f"    Final train loss: {latest.get('train_loss', 'N/A'):.4f}")
        print(f"    Final val loss:   {latest.get('val_loss', 'N/A'):.4f}")
        saved = list(Path(tmpdir).glob("*.pt"))
        print(f"    Checkpoints saved: {len(saved)}")

    # ── 7. Checkpoint save and resume ────────────────────────────────
    print(f"\n[T7] Checkpoint save and resume:")
    model_for_ckpt = ViTTiny(
        image_size=IMAGE_SIZE, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES
    )
    opt_ckpt = build_optimizer(model_for_ckpt, name="adamw", lr=1e-3)
    loss_ckpt = build_loss("cross_entropy")
    trainer_ckpt = Trainer(
        model_for_ckpt, opt_ckpt, loss_ckpt, device="cpu", verbose=False
    )
    trainer_ckpt.fit(train_loader, epochs=2)
    assert trainer_ckpt.current_epoch == 2
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    trainer_ckpt.save_checkpoint(ckpt_path)
    # Resume
    model_resume = ViTTiny(
        image_size=IMAGE_SIZE, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES
    )
    opt_resume = build_optimizer(model_resume, name="adamw", lr=1e-3)
    loss_resume = build_loss("cross_entropy")
    trainer_resume = Trainer(
        model_resume, opt_resume, loss_resume, device="cpu", verbose=False
    )
    loaded_epoch = trainer_resume.load_checkpoint(ckpt_path)
    print(f"    Saved at epoch: {trainer_ckpt.current_epoch}")
    print(f"    Loaded at epoch: {loaded_epoch}")
    Path(ckpt_path).unlink(missing_ok=True)

    print(f"\n{'=' * 56}")
    print("  Training Integration Demo — Complete.")
    print(f"{'=' * 56}")


if __name__ == "__main__":
    main()