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

__all__ = [
    "OverlapPatchEmbed",
    "EfficientSelfAttention",
    "MixFFN",
    "TransformerBlock",
    "MiTStage",
    "MiTBackbone",
    "MIT_CONFIGS",
    "AtrousEnhancement",
]