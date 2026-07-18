"""Patch merging module for FCS-VMamba.

Downsamples spatial resolution by 2× while doubling the channel dimension,
following the hierarchical VMamba / Swin Transformer design.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchMerging(nn.Module):
    """Patch Merging layer.

    Downsamples the spatial resolution by 2× and projects the concatenated
    2×2 patch groups into a doubled embedding dimension.

    Args:
        dim: Input channel dimension.
        norm_layer: Normalisation layer (default ``nn.LayerNorm``).
    """

    def __init__(self, dim: int, norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Downsampled tensor of shape [B, 2*C, H/2, W/2].
        """
        B, C, H, W = x.shape
        # Ensure even spatial dimensions
        assert H % 2 == 0 and W % 2 == 0, f"Spatial dims ({H}, {W}) must be even"

        # Rearrange 2×2 patches into channel dimension
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, H // 2 * W // 2, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2 * W/2, 2*C]

        # Restore spatial structure
        x = x.transpose(1, 2).reshape(B, 2 * C, H // 2, W // 2)
        return x