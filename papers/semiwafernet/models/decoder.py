"""Segmentation decoder for SemiWaferNet (ConvoFormer-UNet).

Lightweight decoder that progressively upsamples fused features
to produce pixel-level binary segmentation logits.

Architecture (matches Section 3.1.2 and Equation 16-17 from the paper):
    Fused features [B, C, H/4, W/4]
    → 3×3 conv + BN + ReLU
    → ×2 upsample [B, C, H/2, W/2]  (auxiliary head 2)
    → 3×3 conv + BN + ReLU
    → ×2 upsample [B, C, H, W]      (auxiliary head 1)
    → 1×1 conv → [B, 1, H, W]       (main head)

Deep supervision:
    L_total = L_main + 0.3 * L_aux1 + 0.2 * L_aux2

Input:  [B, C, H/4, W/4] fused feature map
Output: [B, 1, H, W] binary segmentation logits
"""

from __future__ import annotations

import torch
from torch import nn


class SegmentationDecoder(nn.Module):
    """Lightweight segmentation decoder with deep supervision.

    Progressively upsamples fused features to full input resolution.
    Supports auxiliary outputs for deep supervision during training.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of output classes (default: 1 for binary).
    """

    def __init__(self, in_channels: int, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Decoder blocks (one per upsampling stage)
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
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Main output head
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        # Auxiliary heads for deep supervision (Equation 17)
        self.aux_head1 = nn.Conv2d(in_channels, num_classes, kernel_size=1)  # after 2nd upsample
        self.aux_head2 = nn.Conv2d(in_channels, num_classes, kernel_size=1)  # after 1st upsample

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass.

        The ConvoFormer-UNet encoder (ConvEmbed with 8x8 stride-8 projection)
        produces an ``H/8 x W/8`` token grid (8x8 for a 64x64 input). The decoder
        therefore applies three 2x upsampling stages to recover the full input
        resolution (8 -> 16 -> 32 -> 64).

        Args:
            x: Input feature map [B, C, H/8, W/8].
            return_aux: If True, return dict with main and auxiliary outputs.

        Returns:
            If return_aux=False: [B, num_classes, H, W] segmentation logits.
            If return_aux=True: dict with "main", "aux1", "aux2" keys.
        """
        # x: [B, C, H/8, W/8]

        # First upsample: ×2 → [B, C, H/4, W/4]
        x = self.conv1(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Auxiliary head 2 (after 1st upsample)
        aux2_out = self.aux_head2(x)  # [B, num_classes, H/4, W/4]

        # Second upsample: ×2 → [B, C, H/2, W/2]
        x = self.conv2(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Auxiliary head 1 (after 2nd upsample)
        aux1_out = self.aux_head1(x)  # [B, num_classes, H/2, W/2]

        # Third upsample: ×2 → [B, C, H, W]
        x = self.conv3(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Main output head
        main_out = self.head(x)  # [B, num_classes, H, W]

        if return_aux:
            return {
                "main": main_out,
                "aux1": aux1_out,
                "aux2": aux2_out,
            }

        return main_out


__all__ = ["SegmentationDecoder"]