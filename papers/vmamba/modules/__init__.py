"""Reusable modules for FCS-VMamba.

Available modules:
    - PatchEmbed2D: Convolutional image-to-patch embedding.
    - PatchMerging: Spatial downsampling with channel doubling.
    - SS2D: 2D Selective Scan for VMamba (official implementation).
    - VSSBlock / FCSVSSBlock: Core VMamba State-Space block with FA + SFS
      inserted *between* SS2D residual and MLP path (paper-correct order).
    - FrequencyAttention: Frequency-domain attention (FA).
    - SaliencySuppression: Saliency Feature Suppression (SFS).
    - CrossLayerChannelAttention: Cross-Layer Channel Attention (CLCA).
"""

from papers.vmamba.modules.patch_embed import PatchEmbed2D
from papers.vmamba.modules.patch_merging import PatchMerging
from papers.vmamba.modules.vss_block import SS2D, VSSBlock, FCSVSSBlock
from papers.vmamba.modules.fcs_modules import (
    FrequencyAttention,
    SaliencySuppression,
    CrossLayerChannelAttention,
    build_fa,
    build_sfs,
    build_clca,
)

__all__ = [
    "PatchEmbed2D",
    "PatchMerging",
    "SS2D",
    "VSSBlock",
    "FCSVSSBlock",
    "FrequencyAttention",
    "SaliencySuppression",
    "CrossLayerChannelAttention",
    "build_fa",
    "build_sfs",
    "build_clca",
]