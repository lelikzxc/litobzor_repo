"""Reusable modules for SemiWaferNet.

Components:
- CNN backbone: multi-scale convolutional feature extractor
- Transformer / ConvoFormer blocks for HybridCNN-ViT and ConvoFormer-UNet
"""

from __future__ import annotations

from papers.semiwafernet.modules.cnn_backbone import CNNBackbone, ConvBlock, ResidualBlock
from papers.semiwafernet.modules.transformer import (
    TransformerEncoder,
    TransformerEncoderBlock,
    ConvoFormerBlock,
    ConvEmbed,
    PatchProjection,
    MultiHeadSelfAttention,
    TransformerMLP,
    HybridViTEncoder,
)

__all__ = [
    "CNNBackbone",
    "ConvBlock",
    "ResidualBlock",
    "TransformerEncoder",
    "TransformerEncoderBlock",
    "ConvoFormerBlock",
    "ConvEmbed",
    "PatchProjection",
    "MultiHeadSelfAttention",
    "TransformerMLP",
    "HybridViTEncoder",
]
