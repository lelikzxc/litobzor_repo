"""Atrous Convolution Enhancement module for SegFormer + Atrous.

Implements a multi-scale atrous convolution module that enhances the
final-stage feature map from the MiT backbone before passing it to the
MLP decoder. The module applies parallel atrous convolutions with
different dilation rates to capture multi-scale context, then fuses
the results.

This module is inserted between the MiT backbone and the MLP decoder:
    MiT Backbone → Atrous Enhancement → MLP Decoder → Segmentation Head

Architecture:
    1. Project input to a common channel dimension (with reduction)
    2. Apply parallel atrous convolutions with configurable dilation rates
    3. Concatenate multi-scale features
    4. Fuse via 1×1 convolution
    5. Residual connection with learnable scale

All hyperparameters come from ``configs/config.yaml``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class AtrousEnhancement(nn.Module):
    """Multi-scale Atrous Convolution Enhancement module.

    Enhances features by applying parallel atrous convolutions with
    different dilation rates, then fusing the multi-scale outputs.

    The module preserves spatial resolution (same padding) and channel
    dimensions (residual connection).

    Args:
        dim: Input/output channel dimension.
        rates: List of dilation rates for parallel atrous convolutions
            (default: ``[1, 6, 12, 18]``, from config ``model.atrous.rates``).
        reduction: Channel reduction ratio for the bottleneck projection
            (default: 4, from config ``model.atrous.reduction``).
    """

    def __init__(
        self,
        dim: int,
        rates: list[int] | tuple[int, ...] = (1, 6, 12, 18),
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rates = list(rates)
        self.reduction = reduction

        # Bottleneck projection: C → C/r
        reduced_dim = max(1, dim // reduction)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim, reduced_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True),
        )

        # Parallel atrous convolutions
        self.atrous_convs = nn.ModuleList([
            nn.Conv2d(
                reduced_dim, reduced_dim,
                kernel_size=3,
                padding=rate,
                dilation=rate,
                bias=False,
            )
            for rate in self.rates
        ])

        # Fusion: concatenate all atrous outputs → fuse back to reduced_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_dim * len(self.rates), reduced_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True),
        )

        # Output projection: C/r → C
        self.out_proj = nn.Sequential(
            nn.Conv2d(reduced_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        # Learnable scale for residual connection
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            Enhanced output tensor of shape ``[B, C, H, W]``.
        """
        identity = x

        # Bottleneck: C → C/r
        x = self.bottleneck(x)  # [B, C/r, H, W]

        # Parallel atrous convolutions
        atrous_outputs: list[torch.Tensor] = []
        for conv in self.atrous_convs:
            atrous_outputs.append(conv(x))  # each [B, C/r, H, W]

        # Concatenate along channel dimension
        x = torch.cat(atrous_outputs, dim=1)  # [B, C/r * N, H, W]

        # Fuse multi-scale features
        x = self.fusion(x)  # [B, C/r, H, W]

        # Output projection: C/r → C
        x = self.out_proj(x)  # [B, C, H, W]

        # Residual connection with learnable scale
        out = identity + self.scale * x

        return out


__all__ = ["AtrousEnhancement"]