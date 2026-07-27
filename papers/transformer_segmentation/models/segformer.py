"""SegFormer baseline model for semantic segmentation.

Implements the full SegFormer architecture:
    - MiT backbone (4 hierarchical stages)
    - Optional Atrous Enhancement module (between backbone and decoder)
    - MLP decoder (lightweight all-MLP)
    - Segmentation head

All architecture parameters come from ``configs/config.yaml``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from papers.transformer_segmentation.modules.hybrid_encoder import (
    HybridEncoder,
    HYBRID_CONFIGS,
)
from papers.transformer_segmentation.models.decoder import MLPDecoder


class SegFormer(nn.Module):
    """SegFormer for semantic segmentation.

    Hybrid encoder (conv stages 1-2 + transformer stages 3-4) + MLP decoder.

    Args:
        in_channels: Number of input image channels.
        variant: Hybrid encoder variant name (``"B0"`` or ``"B1"``).
        num_classes: Number of output segmentation classes.
        decoder_dim: Common projection dimension for the MLP decoder.
        dropout: Dropout rate.
        qkv_bias: Whether to use bias in QKV projection.
        qk_scale: Manual scale for QK.
    """

    def __init__(
        self,
        in_channels: int = 3,
        variant: str = "B0",
        num_classes: int = 7,
        decoder_dim: int = 256,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.variant = variant
        self.num_classes = num_classes
        self.decoder_dim = decoder_dim

        # Resolve hybrid config for the chosen variant
        hybrid_config = HYBRID_CONFIGS.get(variant)
        if hybrid_config is None:
            raise ValueError(
                f"Unknown hybrid variant '{variant}'. "
                f"Available: {list(HYBRID_CONFIGS.keys())}"
            )

        # Store embed_dims for decoder (conv channels + trans embed_dims)
        self.embed_dims = (
            hybrid_config["conv_channels"]
            + hybrid_config["trans_embed_dims"]
        )

        # Hybrid encoder: conv path (stages 1-2) + transformer path (stages 3-4)
        self.backbone = HybridEncoder(
            in_channels=in_channels,
            variant=variant,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout=dropout,
        )

        # MLP decoder
        self.decoder = MLPDecoder(
            embed_dims=self.embed_dims,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            Segmentation logits of shape ``[B, num_classes, H, W]``
            at original input resolution. No Softmax applied.
        """
        # Hybrid encoder: [conv1, conv2, trans3, trans4]
        features = self.backbone(x)

        # MLP decoder: aggregate features → segmentation logits
        logits = self.decoder(features)  # [B, num_classes, H, W]

        return logits

    @classmethod
    def from_config(cls, config: Any) -> SegFormer:
        """Instantiate model from a config object.

        Args:
            config: Config object with ``model.encoder.*``,
                    ``model.decoder.*``, ``model.input.*``,
                    and ``data.num_classes``
                    accessible via ``config.get("...")``.

        Returns:
            Configured SegFormer instance.
        """
        return cls(
            in_channels=config.get("model.input.channels", 3),
            variant=config.get("model.encoder.variant", "B0"),
            num_classes=config.get("data.num_classes", 7),
            decoder_dim=config.get("model.decoder.decoder_dim", 256),
            dropout=config.get("model.decoder.dropout", 0.0),
            qkv_bias=config.get("model.encoder.qkv_bias", False),
            qk_scale=config.get("model.encoder.qk_scale", None),
        )


__all__ = ["SegFormer"]