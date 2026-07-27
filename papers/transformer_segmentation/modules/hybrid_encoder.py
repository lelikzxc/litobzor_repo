"""Hybrid Encoder for the proposed segmentation model.

Implements the hybrid encoder design from:
    "Region Segmentation for Efficient Semiconductor Inspection:
     A Deep Learning Approach with Transformers and Atrous Convolution"
     (Electronics 2025, Section 3.1.1)

Architecture per the paper:
    - Two parallel paths:
        1. Convolution path: 2 dilated convolution blocks producing feature
           maps with channel dimensions C₁ and C₂
        2. Transformer path: 2 transformer stages (stages 3 and 4) from
           SegFormer's MiT backbone, producing feature maps with channel
           dimensions C₃ and C₄
    - The outputs of the convolutional path replace the first two transformer
      stages in the encoder's final output
    - All four feature maps are concatenated: [conv1, conv2, trans3, trans4]
      with shape H×W×4C

Key insight from the paper:
    "The transformer path in the proposed model closely follows the structure
     of SegFormer's encoder... However, in our design, only the outputs of
     the last two stages (with channels C₃ and C₄) are passed to the decoder.
     The first stages are replaced by convolution blocks that are more
     efficient at extracting local features compared to the transformer blocks."
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from papers.transformer_segmentation.modules.mit import MiTStage, MIT_CONFIGS
from papers.transformer_segmentation.modules.dilated_conv import DilatedConvBlock


# Default hybrid variant configuration (based on MiT-B0)
HYBRID_CONFIGS: dict[str, dict[str, Any]] = {
    "B0": {
        # Convolution path (stages 1-2)
        "conv_channels": [32, 64],
        "conv_strides": [4, 2],
        "conv_kernel_sizes": [7, 3],
        "conv_paddings": [3, 1],
        "conv_dilations": [[1, 2], [1, 2]],
        "conv_depths": [2, 2],
        # Transformer path (stages 3-4, from MiT-B0)
        "trans_embed_dims": [160, 256],
        "trans_depths": [2, 2],
        "trans_num_heads": [5, 8],
        "trans_reduction_ratios": [2, 1],
        "trans_mlp_ratios": [4, 4],
        "trans_strides": [2, 2],
        "trans_patch_sizes": [3, 3],
        "trans_paddings": [1, 1],
    },
    "B1": {
        "conv_channels": [64, 128],
        "conv_strides": [4, 2],
        "conv_kernel_sizes": [7, 3],
        "conv_paddings": [3, 1],
        "conv_dilations": [[1, 2], [1, 2]],
        "conv_depths": [2, 2],
        "trans_embed_dims": [320, 512],
        "trans_depths": [2, 2],
        "trans_num_heads": [5, 8],
        "trans_reduction_ratios": [2, 1],
        "trans_mlp_ratios": [4, 4],
        "trans_strides": [2, 2],
        "trans_patch_sizes": [3, 3],
        "trans_paddings": [1, 1],
    },
}


class HybridEncoder(nn.Module):
    """Hybrid encoder with convolution path (stages 1-2) and transformer path (stages 3-4).

    Produces 4 multi-scale feature maps:
        - Stage 1 (conv): 1/4 of input resolution, C₁ channels
        - Stage 2 (conv): 1/8 of input resolution, C₂ channels
        - Stage 3 (transformer): 1/16 of input resolution, C₃ channels
        - Stage 4 (transformer): 1/32 of input resolution, C₄ channels

    Args:
        in_channels: Number of input image channels.
        variant: Hybrid variant name (``"B0"`` or ``"B1"``).
        qkv_bias: Whether to use bias in QKV projection.
        qk_scale: Manual scale for QK.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        variant: str = "B0",
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.variant = variant

        config = HYBRID_CONFIGS.get(variant)
        if config is None:
            raise ValueError(
                f"Unknown hybrid variant '{variant}'. "
                f"Available: {list(HYBRID_CONFIGS.keys())}"
            )

        self.conv_channels = config["conv_channels"]
        self.trans_embed_dims = config["trans_embed_dims"]

        # ── Convolution path (stages 1-2) ──────────────────────────────
        self.conv_stages = nn.ModuleList()
        prev_channels = in_channels
        for i in range(2):
            stage = DilatedConvBlock(
                in_channels=prev_channels,
                out_channels=self.conv_channels[i],
                stride=config["conv_strides"][i],
                kernel_size=config["conv_kernel_sizes"][i],
                padding=config["conv_paddings"][i],
                dilations=config["conv_dilations"][i],
                depth=config["conv_depths"][i],
            )
            self.conv_stages.append(stage)
            prev_channels = self.conv_channels[i]

        # ── Transformer path (stages 3-4) ──────────────────────────────
        # Stage 3 takes input from conv stage 2 output
        self.trans_stages = nn.ModuleList()
        for i in range(2):
            stage = MiTStage(
                in_channels=self.conv_channels[1] if i == 0 else self.trans_embed_dims[0],
                embed_dim=self.trans_embed_dims[i],
                depth=config["trans_depths"][i],
                num_heads=config["trans_num_heads"][i],
                reduction_ratio=config["trans_reduction_ratios"][i],
                mlp_ratio=config["trans_mlp_ratios"][i],
                patch_size=config["trans_patch_sizes"][i],
                stride=config["trans_strides"][i],
                padding=config["trans_paddings"][i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                dropout=dropout,
            )
            self.trans_stages.append(stage)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass producing multi-scale features.

        Args:
            x: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            List of 4 feature maps ``[conv1, conv2, trans3, trans4]``
            at resolutions 1/4, 1/8, 1/16, 1/32 of input.
        """
        features: list[torch.Tensor] = []

        # Convolution path: stages 1-2
        for stage in self.conv_stages:
            x = stage(x)
            features.append(x)

        # Transformer path: stages 3-4
        for stage in self.trans_stages:
            x = stage(x)
            features.append(x)

        return features


__all__ = ["HybridEncoder", "HYBRID_CONFIGS", "DilatedConvBlock"]