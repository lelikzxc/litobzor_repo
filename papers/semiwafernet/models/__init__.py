"""Model definitions for SemiWaferNet.

Models:
- SemiWaferNet: HybridCNN-ViT or ConvoFormer-UNet (mode switch)
- ConvoFormerUNet: segmentation architecture (~7.11M)
"""

from __future__ import annotations

from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.models.convoformer_unet import ConvoFormerUNet

__all__ = [
    "SemiWaferNet",
    "ConvoFormerUNet",
]
