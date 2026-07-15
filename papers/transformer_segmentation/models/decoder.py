"""MLP Decoder for SegFormer.

Implements the lightweight all-MLP decoder from:
    SegFormer: Simple and Efficient Design for Semantic Segmentation
    with Transformers (NeurIPS 2021)

The decoder aggregates multi-scale features from the MiT backbone:
    1. Project each stage feature to a common dimension C via MLP
    2. Upsample all features to 1/4 resolution
    3. Concatenate all features
    4. Fuse channels via another MLP
    5. Produce the final decoder feature map

This is a lightweight decoder with no attention or complex operations.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    """Lightweight all-MLP decoder for SegFormer.

    Aggregates multi-scale features from the MiT backbone stages into a
    single feature map at 1/4 resolution.

    Args:
        embed_dims: Channel dimensions of the 4 MiT stages (list of 4 ints).
        decoder_dim: Common projection dimension for all stages.
        num_classes: Number of output classes (for the segmentation head).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dims: list[int] | tuple[int, ...],
        decoder_dim: int = 256,
        num_classes: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dims = list(embed_dims)
        self.decoder_dim = decoder_dim
        self.num_classes = num_classes

        assert len(self.embed_dims) == 4, f"Expected 4 embed_dims, got {len(self.embed_dims)}"

        # Linear projections for each stage: C_i → decoder_dim
        self.linear_projs = nn.ModuleList([
            nn.Conv2d(dim, decoder_dim, kernel_size=1)
            for dim in self.embed_dims
        ])

        # Fusion MLP after concatenation
        self.fusion = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Segmentation head: decoder_dim → num_classes
        self.head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of 4 feature maps from MiT backbone
                ``[stage1, stage2, stage3, stage4]`` at resolutions
                1/4, 1/8, 1/16, 1/32 of input.

        Returns:
            Segmentation logits of shape ``[B, num_classes, H, W]``
            at original input resolution (upsampled from 1/4).
        """
        assert len(features) == 4, f"Expected 4 features, got {len(features)}"

        # Get target spatial size from stage 1 (1/4 resolution)
        _, _, H4, W4 = features[0].shape

        # Project each stage to decoder_dim and upsample to 1/4 resolution
        projected: list[torch.Tensor] = []
        for i, feat in enumerate(features):
            x = self.linear_projs[i](feat)  # [B, decoder_dim, H_i, W_i]
            x = F.interpolate(x, size=(H4, W4), mode="bilinear", align_corners=False)
            projected.append(x)

        # Concatenate along channel dimension
        x = torch.cat(projected, dim=1)  # [B, decoder_dim * 4, H4, W4]

        # Fuse channels
        x = self.fusion(x)  # [B, decoder_dim, H4, W4]

        # Segmentation head
        x = self.head(x)  # [B, num_classes, H4, W4]

        # Upsample to original input resolution
        # (The caller should provide target size; we upsample to 4× H4, W4)
        x = F.interpolate(x, scale_factor=4.0, mode="bilinear", align_corners=False)

        return x


__all__ = ["MLPDecoder"]