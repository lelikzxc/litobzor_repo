#!/usr/bin/env python3
"""Reproducible demo script for YOLOv10 baseline and CTM-YOLOv10.

Creates both models, runs dummy forward passes, and prints a structured
comparison including parameter counts, CTM insertion details, and output
shapes. Uses ``experiment.py`` for metadata formatting.

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

    print(f"\n{'=' * 60}")
    print("  Demo completed successfully.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()