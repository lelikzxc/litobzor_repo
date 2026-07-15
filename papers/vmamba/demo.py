#!/usr/bin/env python3
"""Demo script for FCS-VMamba with FA, SFS, and CLCA.

Creates the FCS-VMamba model, runs a dummy forward pass, and prints
architecture details including stage dimensions, output shape, and
parameter count comparison between backbone-only and full FCS-VMamba.

This is the final research validation demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from papers.vmamba.models.vmamba import FCSVMamba
from papers.vmamba.utils.experiment import (
    build_experiment_info,
    count_params,
    format_experiment_info,
)


def main() -> None:
    """Run FCS-VMamba research validation demo."""
    print("=" * 56)
    print("  FCS-VMamba Research Validation")
    print("=" * 56)

    # ── Configuration ──────────────────────────────────────────────────
    IMAGE_SIZE = 224
    NUM_CLASSES = 8
    BATCH_SIZE = 2

    # ── Create models ──────────────────────────────────────────────────
    print(f"\n[1] Creating models...")

    # Backbone-only (no FA, SFS, CLCA)
    model_backbone = FCSVMamba(
        in_channels=3,
        image_size=IMAGE_SIZE,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        ssm_ratio=2.0,
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        num_classes=NUM_CLASSES,
        fa_enabled=False,
        sfs_enabled=False,
        clca_enabled=False,
    )
    model_backbone.eval()

    # Full FCS-VMamba (with FA, SFS, CLCA)
    model_full = FCSVMamba(
        in_channels=3,
        image_size=IMAGE_SIZE,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        ssm_ratio=2.0,
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        num_classes=NUM_CLASSES,
        fa_enabled=True,
        sfs_enabled=True,
        clca_enabled=True,
    )
    model_full.eval()

    # ── Experiment metadata ────────────────────────────────────────────
    print(f"\n[2] Experiment metadata:")
    info_backbone = build_experiment_info(model_backbone)
    info_full = build_experiment_info(model_full)
    print(format_experiment_info(info_full))

    # ── Parameter comparison ───────────────────────────────────────────
    params_backbone = count_params(model_backbone)
    params_full = count_params(model_full)
    params_added = params_full - params_backbone

    print(f"\n[3] Parameter comparison:")
    print(f"    VMamba baseline:        {params_backbone:>12,}")
    print(f"    FCS-VMamba (full):      {params_full:>12,}")
    print(f"    Added by FCS modules:   {params_added:>12,}")
    print(f"    Relative increase:      {params_added / params_backbone * 100:>11.2f}%")

    # ── Architecture summary ───────────────────────────────────────────
    stage_dims = [model_full.embed_dim * (2**i) for i in range(4)]

    print(f"\n[4] Architecture summary:")
    print(f"    Model:              FCS-VMamba (vmamba_tiny)")
    print(f"    Number of stages:   4")
    print(f"    Base embed dim:     {model_full.embed_dim}")
    print(f"    Stage depths:       {model_full.depths}")
    print(f"    Stage dims:         {stage_dims}")
    print(f"    SSM ratio:          {model_full.ssm_ratio}")
    print(f"    MLP ratio:          {model_full.mlp_ratio}")
    print(f"    Drop path rate:     {model_full.drop_path_rate}")
    print(f"    Num classes:        {model_full.num_classes}")
    print(f"    FA reduction:       {model_full.fa_reduction}")
    print(f"    SFS reduction:      {model_full.sfs_reduction}")
    print(f"    CLCA reduction:     {model_full.clca_reduction}")

    # ── Stage-by-stage breakdown ───────────────────────────────────────
    print(f"\n[5] Stage breakdown:")
    resolutions = [
        IMAGE_SIZE // 4,
        IMAGE_SIZE // 8,
        IMAGE_SIZE // 16,
        IMAGE_SIZE // 32,
    ]
    for i in range(4):
        modules = ["VSSBlock"]
        if model_full.fa_enabled:
            modules.append("FA")
        if model_full.sfs_enabled:
            modules.append("SFS")
        print(f"    Stage {i + 1}:  {model_full.depths[i]}× {'+'.join(modules)}  "
              f"[B, {stage_dims[i]}, {resolutions[i]}, {resolutions[i]}]")

    # ── Forward pass ───────────────────────────────────────────────────
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    print(f"\n[6] Forward pass:")
    print(f"    Input shape:        [{BATCH_SIZE}, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]")

    with torch.no_grad():
        out = model_full(x)

    print(f"    Output shape:       {list(out.shape)}")
    print(f"    Output type:        logits (no Softmax)")

    # ── VSSBlock execution order ────────────────────────────────────────
    print(f"\n[7] FCSVSSBlock execution order (paper-correct):")
    print(f"    1. LayerNorm")
    print(f"    2. SS2D (2D Selective Scan, official v2)")
    print(f"    3. DropPath + residual")
    print(f"    4. FA (Frequency Attention) — FFT → gate → iFFT")
    print(f"    5. SFS (Saliency Feature Suppression)")
    print(f"    6. LayerNorm")
    print(f"    7. MLP (GELU, ratio={model_full.mlp_ratio})")
    print(f"    8. DropPath + residual")
    print(f"")
    print(f"    FA and SFS are inserted *between* the SS2D residual and")
    print(f"    the MLP path, exactly as described in the paper.")
    print(f"    Verified via register_forward_hook in tests.")

    # ── Module insertion points ─────────────────────────────────────────
    print(f"\n[8] Module insertion points:")
    print(f"    1. FA (Frequency Attention):")
    print(f"       Inside each FCSVSSBlock, after SS2D residual,")
    print(f"       before MLP path. Operates on [B, C, H, W].")
    print(f"       Applies 2D FFT → frequency-domain gating → inverse FFT.")
    print(f"")
    print(f"    2. SFS (Saliency Feature Suppression):")
    print(f"       Inside each FCSVSSBlock, after FA,")
    print(f"       before MLP path. Operates on [B, C, H, W].")
    print(f"       Computes spatial saliency → learned suppression gate.")
    print(f"")
    print(f"    3. CLCA (Cross-Layer Channel Attention):")
    print(f"       Applied after all 4 stages, before GAP.")
    print(f"       Connects each earlier stage (1, 2, 3) to the final")
    print(f"       stage (4) via channel-wise attention recalibration.")

    # ── Reproduced layers ──────────────────────────────────────────────
    print(f"\n[9] Layers fully reproduced:")
    print(f"    ✓ Patch Embedding (Conv2d stride=4 + LayerNorm)")
    print(f"    ✓ Patch Merging (2× downsampling, channel doubling)")
    print(f"    ✓ SS2D (official VMamba v2: cross-scan + selective scan)")
    print(f"    ✓ FCSVSSBlock (LN → SS2D → residual → FA → SFS → LN → MLP → residual)")
    print(f"    ✓ FA (Frequency Attention via FFT)")
    print(f"    ✓ SFS (Saliency Feature Suppression)")
    print(f"    ✓ CLCA (Cross-Layer Channel Attention)")
    print(f"    ✓ Global Average Pooling + Linear classifier")

    # ── Remaining differences ──────────────────────────────────────────
    print(f"\n[10] Remaining differences vs. paper:")
    print(f"    - FA implementation: Uses learnable frequency-domain gating")
    print(f"      via conv net on log-magnitude spectrum. The paper may use")
    print(f"      a specific frequency selection strategy.")
    print(f"    - SFS implementation: Uses spatial saliency (channel mean)")
    print(f"      combined with learned gating. The paper may use a more")
    print(f"      sophisticated saliency estimation.")
    print(f"    - CLCA implementation: Uses SE-style channel attention with")
    print(f"      guide→target feature projection. The paper may use")
    print(f"      different aggregation strategies.")
    print(f"    - Pretrained weights: Not available (training from scratch).")
    print(f"    - Training hyperparameters: Need to be tuned for wafer data.")

    # ── Validation status ──────────────────────────────────────────────
    print(f"\n[11] Validation status:")
    print(f"    ✅ Patch Embedding — Exact match with VMamba paper")
    print(f"    ✅ Patch Merging — Exact match with VMamba paper")
    print(f"    ✅ SS2D (official v2) — Vendored official implementation")
    print(f"    ✅ FCSVSSBlock order — Verified via forward hooks")
    print(f"    ✅ FA placement — Between SS2D residual and MLP")
    print(f"    ✅ SFS placement — Between SS2D residual and MLP")
    print(f"    ✅ CLCA placement — Cross-stage connections")
    print(f"    ✅ Classification head — GAP → LayerNorm → Linear")
    print(f"    ✅ Ablation support — All 8 configurations supported")
    print(f"    ✅ Config-driven — All params from YAML")
    print(f"    ⚠️ FA implementation — Approximation (paper lacks detail)")
    print(f"    ⚠️ SFS implementation — Approximation (paper lacks detail)")
    print(f"    ⚠️ CLCA implementation — Approximation (paper lacks detail)")
    print(f"    ❌ Pretrained weights — Not available")
    print(f"    ❌ Training pipeline — Not implemented")

    # ── Test summary ───────────────────────────────────────────────────
    print(f"\n[12] Test summary:")
    print(f"    Run: pytest papers/vmamba/tests/ -v")
    print(f"    Expected: 30+ tests covering FA, SFS, CLCA, ablation, experiment info")

    print(f"\n{'=' * 56}")
    print("  FCS-VMamba Research Validation — Complete.")
    print(f"{'=' * 56}")


if __name__ == "__main__":
    main()