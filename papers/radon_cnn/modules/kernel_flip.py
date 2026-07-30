"""Kernel flip module for flip-equivariant feature extraction.

Based on the paper:
    Jeong et al., "Wafer map failure pattern classification using geometric
    transformation-invariant convolutional neural network",
    Scientific Reports 2023, 13:8127.

The kernel flip module uses only two branches of flipped kernels:
    1. Original kernel
    2. Single-axis flipped kernel (horizontal flip)

Weight-sharing ensures no additional trainable parameters.
A max-out operation selects the most active features element-wise.

Per the paper:
    "Kernel flip modules aim to learn the flip equivariance through generated
    flipped copies of input features with multiple flip versions of kernels."
    "After generating a flipped feature set, the max-out module then takes the
    most active features element-by-element to pass to the CNN classifier module."
"""

from __future__ import annotations

import torch
from torch import nn


class KernelFlip(nn.Module):
    """Kernel flip module with weight-shared flipped kernels and max-out.

    Applies a convolution with the original kernel and a horizontally flipped
    version of the same kernel (weight-shared), then performs max-out
    element-wise over the two branches.

    Per the paper (Table 2):
        - Proposed Conv2: 16×16×64 ×2 (weight-shared kernel flip)
        - Max out: 16×16×64

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size (default 3).
        stride: Convolution stride (default 1).
        padding: Convolution padding (default 1 for 3×3 kernel).
        bias: Whether to use bias (default False, weight-shared).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # Single convolution layer — the flipped version shares weights
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with kernel flip and max-out.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Max-out tensor of shape [B, out_channels, H_out, W_out].
        """
        # Branch 1: original convolution
        out1 = self.conv(x)

        # Branch 2: flip input horizontally, apply same conv, flip back
        x_flipped = x.flip(-1)  # Horizontal flip
        out2 = self.conv(x_flipped)
        out2 = out2.flip(-1)  # Flip back to original orientation

        # Max-out: element-wise maximum over the two branches
        out = torch.maximum(out1, out2)

        return out