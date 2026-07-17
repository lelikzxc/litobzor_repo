"""Patch embedding module for FCS-VMamba.

Converts an input image into a sequence of patch tokens using a
convolutional stem, following the VMamba / ViT patch embedding design.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding.

    Uses a convolutional layer with stride to project image patches into
    an embedding space, followed by LayerNorm.

    Args:
        in_channels: Number of input image channels (default 3).
        embed_dim: Patch embedding dimension.
        patch_size: Size of each patch (default 4).
        norm_layer: Normalisation layer (default ``nn.LayerNorm``).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        patch_size: int = 4,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Patch embedding tensor of shape [B, embed_dim, H/p, W/p]
            where p = patch_size.
        """
        x = self.proj(x)  # [B, embed_dim, H/p, W/p]
        # Apply LayerNorm over the channel dimension (permute → norm → permute)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W/p², C]
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        return x