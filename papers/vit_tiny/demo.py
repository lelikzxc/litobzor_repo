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
    print(f"    MLP ratio:          {model.mlp_ratio}")
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


if __name__ == "__main__":
    main()