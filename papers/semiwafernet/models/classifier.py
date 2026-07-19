"""Classification head for SemiWaferNet.

Global Average Pooling → LayerNorm → Linear classifier.
"""

from __future__ import annotations

import torch
from torch import nn


class ClassifierHead(nn.Module):
    """Classification head with GAP, LayerNorm, and linear projection.

    Input:  [B, C, H, W] fused feature map
    Output: [B, num_classes] classification logits
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(in_channels)
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.gap(x)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]
        x = self.norm(x)
        x = self.head(x)  # [B, num_classes]
        return x