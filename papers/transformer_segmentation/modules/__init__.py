"""Reusable modules for SegFormer + Atrous.

Exports:
    - OverlapPatchEmbed: Overlapping convolutional patch embedding
    - EfficientSelfAttention: Self-attention with spatial reduction
    - MixFFN: MLP with 3×3 depthwise convolution
    - TransformerBlock: SegFormer transformer encoder block
    - MiTStage: Single hierarchical stage
    - MiTBackbone: Full 4-stage MiT encoder
    - MIT_CONFIGS: Pre-defined variant configurations (B0-B5)
    - AtrousEnhancement: Multi-scale atrous convolution enhancement
    - DilatedConvBlock: Dilated convolution block for hybrid encoder
    - HybridEncoder: Hybrid conv + transformer encoder
    - HYBRID_CONFIGS: Hybrid encoder variant configurations
"""

from papers.transformer_segmentation.modules.mit import (
    OverlapPatchEmbed,
    EfficientSelfAttention,
    MixFFN,
    TransformerBlock,
    MiTStage,
    MiTBackbone,
    MIT_CONFIGS,
)
from papers.transformer_segmentation.modules.atrous import AtrousEnhancement
from papers.transformer_segmentation.modules.dilated_conv import DilatedConvBlock
from papers.transformer_segmentation.modules.hybrid_encoder import HybridEncoder, HYBRID_CONFIGS

__all__ = [
    "OverlapPatchEmbed",
    "EfficientSelfAttention",
    "MixFFN",
    "TransformerBlock",
    "MiTStage",
    "MiTBackbone",
    "MIT_CONFIGS",
    "AtrousEnhancement",
    "DilatedConvBlock",
    "HybridEncoder",
    "HYBRID_CONFIGS",
]