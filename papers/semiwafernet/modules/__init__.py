"""Reusable modules for SemiWaferNet.

Components:
- CNN backbone: multi-scale convolutional feature extractor
- Transformer encoder: global context modeling via self-attention
- Feature fusion: multi-scale CNN + transformer feature integration
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
from papers.semiwafernet.modules.fusion import FeatureFusion, ChannelAlign

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
    "FeatureFusion",
    "ChannelAlign",
]