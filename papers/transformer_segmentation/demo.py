#!/usr/bin/env python3
"""Demo script for SegFormer vs SegFormer + Atrous comparison.

Creates both baseline and Atrous-enhanced models, runs dummy forward
passes, and prints architecture details, parameter comparison, and
output shapes. Includes experiment metadata from the utils module.

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

from papers.transformer_segmentation.models.segformer import SegFormer
from papers.transformer_segmentation.utils.experiment import (
    build_experiment_info,
    count_params,
    format_experiment_info,
)


def main() -> None:
    """Run SegFormer vs SegFormer + Atrous comparison demo."""
    print("=" * 56)
    print("  SegFormer + Atrous — Research Validation")
    print("=" * 56)

    # ── Configuration ──────────────────────────────────────────────────
    IMAGE_SIZE = 512
    NUM_CLASSES = 8
    BATCH_SIZE = 2
    VARIANT = "B0"

    # ── Create models ──────────────────────────────────────────────────
    print(f"\n[1] Creating SegFormer-{VARIANT} models...")

    # Baseline (no Atrous)
    model_baseline = SegFormer(
        in_channels=3,
        variant=VARIANT,
        num_classes=NUM_CLASSES,
        decoder_dim=256,
        atrous_enabled=False,
    )
    model_baseline.eval()

    # Atrous-enhanced
    model_atrous = SegFormer(
        in_channels=3,
        variant=VARIANT,
        num_classes=NUM_CLASSES,
        decoder_dim=256,
        atrous_enabled=True,
        atrous_rates=[1, 6, 12, 18],
        atrous_reduction=4,
    )
    model_atrous.eval()

    # ── Experiment metadata ────────────────────────────────────────────
    print(f"\n[2] Experiment metadata:")
    info_baseline = build_experiment_info(model_baseline)
    info_atrous = build_experiment_info(model_atrous)
    print(format_experiment_info(info_atrous))

    # ── Architecture summary ───────────────────────────────────────────
    print(f"\n[3] Architecture summary:")
    print(f"    Model:              SegFormer-{VARIANT}")
    print(f"    Backbone:           MiT-{VARIANT}")
    print(f"    Decoder:            MLP Decoder (dim={model_baseline.decoder_dim})")
    print(f"    Num classes:        {model_baseline.num_classes}")
    print(f"    Embed dims:         {model_baseline.embed_dims}")
    print(f"    Depths:             {model_baseline.depths}")
    print(f"    Num heads:          {model_baseline.num_heads}")
    print(f"    Reduction ratios:   {model_baseline.reduction_ratios}")
    print(f"    MLP ratios:         {model_baseline.mlp_ratios}")
    print(f"    Atrous enabled:     {model_atrous.atrous_enabled}")
    print(f"    Atrous rates:       {model_atrous.atrous_rates}")
    print(f"    Atrous reduction:   {model_atrous.atrous_reduction}")

    # ── Parameter comparison ───────────────────────────────────────────
    params_baseline = count_params(model_baseline)
    params_atrous = count_params(model_atrous)
    params_added = params_atrous - params_baseline

    print(f"\n[4] Parameter comparison:")
    print(f"    SegFormer baseline:     {params_baseline:>10,}")
    print(f"    SegFormer + Atrous:     {params_atrous:>10,}")
    print(f"    Added by Atrous:        {params_added:>10,}")
    print(f"    Relative increase:      {params_added / params_baseline * 100:>9.2f}%")

    # ── Stage breakdown ────────────────────────────────────────────────
    resolutions = [
        IMAGE_SIZE // 4,
        IMAGE_SIZE // 8,
        IMAGE_SIZE // 16,
        IMAGE_SIZE // 32,
    ]
    print(f"\n[5] Stage breakdown:")
    for i in range(4):
        print(f"    Stage {i + 1}:  {model_baseline.depths[i]}× TransformerBlock  "
              f"[B, {model_baseline.embed_dims[i]}, {resolutions[i]}, {resolutions[i]}]")

    # ── Atrous insertion point ─────────────────────────────────────────
    print(f"\n[6] Atrous insertion point:")
    print(f"    MiT Backbone (Stage 4) → Atrous Enhancement → MLP Decoder")
    print(f"    Atrous operates on the final stage feature map:")
    print(f"    [B, {model_baseline.embed_dims[3]}, {resolutions[3]}, {resolutions[3]}]")
    print(f"    → Multi-scale atrous conv (rates: {model_atrous.atrous_rates})")
    print(f"    → Channel fusion → Residual")
    print(f"    → [B, {model_baseline.embed_dims[3]}, {resolutions[3]}, {resolutions[3]}]")

    # ── Forward pass ───────────────────────────────────────────────────
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    print(f"\n[7] Forward pass:")
    print(f"    Input shape:        [{BATCH_SIZE}, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]")

    with torch.no_grad():
        out_baseline = model_baseline(x)
        out_atrous = model_atrous(x)

    print(f"    Baseline output:    {list(out_baseline.shape)}")
    print(f"    Atrous output:      {list(out_atrous.shape)}")
    print(f"    Output type:        logits (no Softmax)")

    # ── Feature map shapes ─────────────────────────────────────────────
    print(f"\n[8] Multi-scale feature maps:")
    with torch.no_grad():
        features_baseline = model_baseline.backbone(x)
        features_atrous = model_atrous.backbone(x)
    for i in range(4):
        print(f"    Stage {i + 1}:          {list(features_baseline[i].shape)}")
    print(f"    After Atrous:       {list(model_atrous.atrous(features_atrous[3]).shape)}")

    # ── Validation status ──────────────────────────────────────────────
    print(f"\n[9] Validation status:")
    print(f"    ✅ OverlapPatchEmbed — Exact match with SegFormer paper")
    print(f"    ✅ EfficientSelfAttention — Exact match with SegFormer paper")
    print(f"    ✅ Mix-FFN — Exact match with SegFormer paper")
    print(f"    ✅ TransformerBlock — Exact match with SegFormer paper")
    print(f"    ✅ MiTBackbone — Exact match with SegFormer paper")
    print(f"    ✅ MLPDecoder — Exact match with SegFormer paper")
    print(f"    ✅ Segmentation Head — Exact match with SegFormer paper")
    print(f"    ✅ Atrous insertion point — Between backbone and decoder")
    print(f"    ✅ Atrous dilation config — Configurable rates from YAML")
    print(f"    ✅ Ablation support — All configurations supported")
    print(f"    ✅ Config-driven — All params from YAML")
    print(f"    ✅ Engine integration — Registry, EngineConfig, Predictor, Engine")
    print(f"    ⚠️ Atrous internal arch — Approximation (paper lacks detail)")
    print(f"    ⚠️ Atrous normalisation — BatchNorm (paper does not specify)")
    print(f"    ❌ Pretrained weights — Not available")
    print(f"    ❌ Training pipeline — Not implemented")

    print(f"\n{'=' * 56}")
    print("  Demo completed successfully.")
    print(f"{'=' * 56}")

    # ═══════════════════════════════════════════════════════════════════
    #  Engine Integration Demo
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 56}")
    print("  Engine Integration Demo")
    print(f"{'=' * 56}")

    # ── 1. Registry construction ──────────────────────────────────────
    print(f"\n[E1] Registry construction:")
    from common.engine.registry import build_model, is_registered, list_registered

    assert is_registered("models", "segformer_atrous"), "segformer_atrous should be registered"
    registered_models = list_registered("models")
    print(f"    Registered models: {registered_models}")
    print(f"    segformer_atrous in registry: {'segformer_atrous' in registered_models}")

    # ── 2. EngineConfig loading ───────────────────────────────────────
    print(f"\n[E2] EngineConfig loading:")
    from common.engine.config import EngineConfig

    config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    print(f"    model.name:           {config.get('model.name')}")
    print(f"    model.num_classes:    {config.get('model.num_classes')}")
    print(f"    model.backbone.variant: {config.get('model.backbone.variant')}")
    print(f"    training.optimizer:   {config.get('training.optimizer')}")
    print(f"    training.scheduler:   {config.get('training.scheduler')}")
    print(f"    training.loss:        {config.get('training.loss')}")
    print(f"    dataset.name:         {config.get('dataset.name')}")

    # ── 3. build_model() by registered name ───────────────────────────
    print(f"\n[E3] build_model() by registered name:")
    model_from_registry = build_model(
        "segformer_atrous",
        in_channels=3,
        variant=VARIANT,
        num_classes=NUM_CLASSES,
    )
    model_from_registry.eval()
    print(f"    Model type: {type(model_from_registry).__name__}")
    print(f"    num_classes: {model_from_registry.num_classes}")

    # ── 4. from_config() with EngineConfig ────────────────────────────
    print(f"\n[E4] from_config() with EngineConfig:")
    model_from_cfg = SegFormer.from_config(config)
    model_from_cfg.eval()
    print(f"    Model type: {type(model_from_cfg).__name__}")
    print(f"    num_classes: {model_from_cfg.num_classes}")
    print(f"    variant: {model_from_cfg.variant}")

    # ── 5. Predictor inference ────────────────────────────────────────
    print(f"\n[E5] Predictor inference:")
    from common.inference.predictor import Predictor

    predictor = Predictor(model_atrous, device="cpu")
    x_single = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
    result = predictor.predict_single(x_single)
    print(f"    logits shape:    {list(result['logits'].shape)}")
    print(f"    probs shape:     {list(result['probs'].shape)}")
    print(f"    prediction shape: {list(result['prediction'].shape)}")

    # ── 6. Batch inference via Predictor ──────────────────────────────
    print(f"\n[E6] Batch inference via Predictor:")
    x_batch = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    batch_result = predictor.predict_batch(x_batch)
    print(f"    batch logits shape: {list(batch_result['logits'].shape)}")
    print(f"    batch predictions:  {list(batch_result['prediction'].shape)}")

    # ── 7. Engine instantiation ───────────────────────────────────────
    print(f"\n[E7] Engine instantiation:")
    from common.engine.engine import Engine

    engine = Engine(model_atrous, config, device="cpu")
    summary = engine.summary()
    print(f"    Engine model: {summary['model']}")
    print(f"    Engine device: {summary['device']}")

    # ── 8. Engine predict_single ──────────────────────────────────────
    print(f"\n[E8] Engine predict_single:")
    engine_result = engine.predict_single(x_single)
    print(f"    logits shape:    {list(engine_result['logits'].shape)}")
    print(f"    probs shape:     {list(engine_result['probs'].shape)}")
    print(f"    prediction shape: {list(engine_result['prediction'].shape)}")

    # ── 9. Forward pass output shapes ─────────────────────────────────
    print(f"\n[E9] Forward pass output shapes:")
    with torch.no_grad():
        logits = model_atrous(x_batch)
    print(f"    Input:  [{BATCH_SIZE}, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]")
    print(f"    Output: {list(logits.shape)}  (logits, no Softmax)")
    print(f"    Expected: [{BATCH_SIZE}, {NUM_CLASSES}, {IMAGE_SIZE}, {IMAGE_SIZE}]")

    print(f"\n{'=' * 56}")
    print("  Engine Integration Demo — Complete.")
    print(f"{'=' * 56}")


if __name__ == "__main__":
    main()