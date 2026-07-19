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


if __name__ == "__main__":
    main()