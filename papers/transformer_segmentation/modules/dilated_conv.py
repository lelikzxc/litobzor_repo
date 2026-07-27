"""Dilated Convolution Block for the hybrid encoder.

Implements the convolution path from:
    "Region Segmentation for Efficient Semiconductor Inspection:
     A Deep Learning Approach with Transformers and Atrous Convolution"
     (Electronics 2025, Section 3.1.1)

The convolution path consists of two dilated convolution blocks that replace
the first two transformer stages in the hybrid encoder. Each block uses
dilated convolutions to expand the receptive field while preserving spatial
resolution, making them more efficient at extracting local features compared
to transformer blocks.

Architecture per the paper:
    - Two dilated convolution blocks producing feature maps with channel
      dimensions matching C₁ and C₂ of the transformer path
    - Dilated convolutions preserve spatial resolution while expanding
      receptive field
    - Outputs replace the first two transformer stages in the encoder output
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class DilatedConvBlock(nn.Module):
    """Dilated convolution block for the hybrid encoder's convolution path.

    Each block consists of:
        1. Overlapping patch embedding (7×7 conv, stride=4, padding=3 for stage 1;
           3×3 conv, stride=2, padding=2 for stage 2 with dilation=2)
        2. Multiple dilated convolution layers with residual connections
        3. Layer normalisation

    The dilated convolutions preserve spatial resolution while expanding
    the receptive field, allowing extraction of features across multiple
    scales without losing geometric integrity.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the initial patch embedding.
        kernel_size: Kernel size for the initial patch embedding.
        padding: Padding for the initial patch embedding.
        dilations: List of dilation rates for the dilated convolutions.
        depth: Number of dilated convolution layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 4,
        kernel_size: int = 7,
        padding: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2),
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilations = list(dilations)
        self.depth = depth

        # Overlapping patch embedding (like SegFormer's OverlapPatchEmbed)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Dilated convolution layers with residual connections
        # Each layer: 3×3 dilated conv → BN → ReLU
        self.layers = nn.ModuleList()
        for i in range(depth):
            dilation = self.dilations[i % len(self.dilations)]
            # Ensure spatial resolution is preserved with dilated conv
            # padding = dilation to keep same resolution
            conv_padding = dilation
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=conv_padding,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            Output tensor of shape ``[B, out_channels, H_out, W_out]``.
        """
        x = self.patch_embed(x)

        for layer in self.layers:
            identity = x
            x = layer(x)
            x = x + identity  # Residual connection

        return x


__all__ = ["DilatedConvBlock"]