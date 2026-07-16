#!/usr/bin/env python3
"""SemiWaferNet — baseline architecture demo.

Demonstrates model creation, parameter counts, forward pass
shapes, and experiment metadata for the hybrid CNN–Transformer baseline.
"""

from __future__ import annotations

import torch

from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.utils.experiment import (
    build_experiment_info,
    count_params,
    format_experiment_info,
)


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

    # ── Status ──────────────────────────────────────────────────────
    print("Status: ✅ Baseline architecture implemented")
    print()


if __name__ == "__main__":
    main()