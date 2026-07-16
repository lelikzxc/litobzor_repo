"""Segmentation decoder for SemiWaferNet.

Lightweight decoder that progressively upsamples fused features
to produce pixel-level segmentation logits.
"""

from __future__ import annotations

import torch
from torch import nn


class SegmentationDecoder(nn.Module):
    """Lightweight segmentation decoder.

    Progressively upsamples fused features to full input resolution.

    Architecture:
        Fused features [B, C, H/4, W/4]
        → 3×3 conv + BN + ReLU
        → ×2 upsample [B, C, H/2, W/2]
        → 3×3 conv + BN + ReLU
        → ×2 upsample [B, C, H, W]
        → 1×1 conv → [B, num_classes, H, W]

    Input:  [B, C, H/4, W/4] fused feature map
    Output: [B, num_classes, H, W] segmentation logits
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H/4, W/4]
        x = self.conv1(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv2(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.head(x)  # [B, num_classes, H, W]
        return x