#!/usr/bin/env python3
"""Demo script for YOLOv10 baseline.

Creates a YOLOv10Baseline model, runs a dummy forward pass,
and prints architecture information.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from papers.ctm_yolov10.models.yolov10 import YOLOv10Baseline


def main() -> None:
    """Run YOLOv10 baseline demo."""
    print("=== YOLOv10 Baseline Demo ===\n")

    # Create model
    model = YOLOv10Baseline(model_name="yolov10n", pretrained=False)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create dummy input
    batch_size = 1
    channels = 3
    height, width = 640, 640
    x = torch.randn(batch_size, channels, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Print info
    print(f"Model name:          {model.model_name}")
    print(f"Input shape:         {list(x.shape)}")
    print(f"Output type:         {type(output).__name__}")
    if isinstance(output, (list, tuple)):
        print(f"Output length:       {len(output)}")
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor):
                print(f"  [{i}] Tensor shape:  {list(o.shape)}")
            else:
                print(f"  [{i}] {type(o).__name__}: {o.keys() if isinstance(o, dict) else o}")
    elif isinstance(output, dict):
        print(f"Output keys:         {list(output.keys())}")
    print(f"Number of parameters: {num_params:,}")
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()