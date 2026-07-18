"""Model definitions for SegFormer + Atrous.

Exports:
    - MLPDecoder: Lightweight all-MLP decoder
    - SegFormer: Full SegFormer baseline model
"""

from papers.transformer_segmentation.models.decoder import MLPDecoder
from papers.transformer_segmentation.models.segformer import SegFormer

__all__ = [
    "MLPDecoder",
    "SegFormer",
]
