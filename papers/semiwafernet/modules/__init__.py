"""Reusable modules for SemiWaferNet.

Components:
- CNN backbone: multi-scale convolutional feature extractor
- Transformer encoder: global context modeling via self-attention
- Feature fusion: multi-scale CNN + transformer feature integration
"""

from __future__ import annotations

from papers.semiwafernet.modules.cnn_backbone import CNNBackbone, ConvBlock, CNNStage
from papers.semiwafernet.modules.transformer import (
    TransformerEncoder,
    TransformerEncoderBlock,
    PatchProjection,
    MultiHeadSelfAttention,
    TransformerMLP,
)
from papers.semiwafernet.modules.fusion import FeatureFusion, ChannelAlign

__all__ = [
    "CNNBackbone",
    "ConvBlock",
    "CNNStage",
    "TransformerEncoder",
    "TransformerEncoderBlock",
    "PatchProjection",
    "MultiHeadSelfAttention",
    "TransformerMLP",
    "FeatureFusion",
    "ChannelAlign",
]