"""Feature fusion module for SemiWaferNet.

Combines multi-scale CNN features with transformer-enhanced features
for downstream classification and segmentation.

Architecture (updated for 2-stage CNN backbone):
    1. Align CNN stage features (stage1: 64ch, stage2: 128ch) to fusion_dim
    2. Reshape transformer tokens back to spatial feature map
    3. Project transformer features to fusion_dim
    4. Concatenate all aligned features (3 sources: stage1, stage2, transformer)
    5. Fuse with 3×3 conv + BN + ReLU
    6. Output fused feature map at stage1 resolution

For segmentation, the fusion also accepts raw input features from ConvEmbed.
"""

from __future__ import annotations

import torch
from torch import nn


class ChannelAlign(nn.Module):
    """Project a feature map to a target number of channels via 1×1 conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class FeatureFusion(nn.Module):
    """Fuse multi-scale CNN features with transformer-enhanced features.

    Architecture:
        1. Align CNN stage features to a common fusion dimension
        2. Reshape transformer tokens back to spatial feature map
        3. Project transformer features to fusion dimension
        4. Concatenate all aligned features
        5. Fuse with 3×3 conv + BN + ReLU
        6. Output fused feature map at the highest resolution (stage 1)

    The fused output preserves spatial information from the highest-resolution
    CNN stage while incorporating global context from the transformer.
    """

    def __init__(
        self,
        cnn_channels: list[int],
        transformer_dim: int,
        fusion_dim: int = 256,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim

        # Align each CNN stage to fusion_dim
        self.cnn_align = nn.ModuleList([
            ChannelAlign(c, fusion_dim) for c in cnn_channels
        ])

        # Project transformer features to fusion_dim
        self.transformer_proj = nn.Conv2d(transformer_dim, fusion_dim, kernel_size=1)

        # Fusion conv (after concat: (len(cnn_channels) + 1) * fusion_dim → fusion_dim)
        num_sources = len(cnn_channels) + 1  # CNN stages + transformer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_dim * num_sources, fusion_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )

        # Output projections for each task
        self.class_proj = nn.Conv2d(fusion_dim, fusion_dim, kernel_size=1)
        self.seg_proj = nn.Conv2d(fusion_dim, fusion_dim, kernel_size=1)

    def forward(
        self,
        cnn_features: list[torch.Tensor],
        transformer_tokens: torch.Tensor,
        transformer_spatial: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            cnn_features: List of feature maps from CNN backbone
                [stage1, stage2] (2 stages for HybridCNN-ViT).
            transformer_tokens: [B, N, embed_dim] transformer output.
            transformer_spatial: (H, W) spatial dims of transformer feature map.

        Returns:
            class_features: [B, fusion_dim, H/4, W/4] for classification.
            seg_features: [B, fusion_dim, H/4, W/4] for segmentation.
        """
        H, W = transformer_spatial

        # Align CNN features to fusion_dim and upsample to stage1 resolution
        aligned: list[torch.Tensor] = []
        target_h, target_w = cnn_features[0].shape[2:]  # stage 1 resolution

        for i, feat in enumerate(cnn_features):
            aligned_feat = self.cnn_align[i](feat)  # [B, fusion_dim, Hi, Wi]
            if aligned_feat.shape[2:] != (target_h, target_w):
                aligned_feat = torch.nn.functional.interpolate(
                    aligned_feat, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            aligned.append(aligned_feat)

        # Reshape transformer tokens back to spatial feature map
        transformer_map = transformer_tokens.transpose(1, 2).reshape(-1, transformer_tokens.shape[-1], H, W)

        # Upsample transformer map to stage1 resolution
        if (H, W) != (target_h, target_w):
            transformer_map = torch.nn.functional.interpolate(
                transformer_map, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        transformer_map = self.transformer_proj(transformer_map)  # [B, fusion_dim, H/4, W/4]

        # Concatenate all features
        all_features = torch.cat([*aligned, transformer_map], dim=1)

        # Fuse
        fused = self.fusion_conv(all_features)  # [B, fusion_dim, H/4, W/4]

        # Task-specific projections
        class_features = self.class_proj(fused)
        seg_features = self.seg_proj(fused)

        return class_features, seg_features


__all__ = ["ChannelAlign", "FeatureFusion"]