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

from papers.transformer_segmentation.modules.mit import MiTBackbone, MIT_CONFIGS
from papers.transformer_segmentation.modules.atrous import AtrousEnhancement
from papers.transformer_segmentation.models.decoder import MLPDecoder


class SegFormer(nn.Module):
    """SegFormer for semantic segmentation.

    Hierarchical MiT encoder + optional Atrous Enhancement + MLP decoder.

    Args:
        in_channels: Number of input image channels.
        variant: MiT variant name (``"B0"`` through ``"B5"``).
        num_classes: Number of output segmentation classes.
        decoder_dim: Common projection dimension for the MLP decoder.
        dropout: Dropout rate.
        qkv_bias: Whether to use bias in QKV projection.
        qk_scale: Manual scale for QK.
        atrous_enabled: Whether to enable the Atrous Enhancement module.
        atrous_rates: Dilation rates for atrous convolutions.
        atrous_reduction: Channel reduction ratio for atrous bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 3,
        variant: str = "B0",
        num_classes: int = 8,
        decoder_dim: int = 256,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        atrous_enabled: bool = False,
        atrous_rates: list[int] | tuple[int, ...] = (1, 6, 12, 18),
        atrous_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.variant = variant
        self.num_classes = num_classes
        self.decoder_dim = decoder_dim
        self.atrous_enabled = atrous_enabled
        self.atrous_rates = list(atrous_rates)
        self.atrous_reduction = atrous_reduction

        # Resolve MiT config for the chosen variant
        mit_config = MIT_CONFIGS.get(variant)
        if mit_config is None:
            raise ValueError(
                f"Unknown MiT variant '{variant}'. "
                f"Available: {list(MIT_CONFIGS.keys())}"
            )

        self.embed_dims = mit_config["embed_dims"]
        self.depths = mit_config["depths"]
        self.num_heads = mit_config["num_heads"]
        self.reduction_ratios = mit_config["reduction_ratios"]
        self.mlp_ratios = mit_config["mlp_ratios"]
        self.strides = mit_config["strides"]
        self.patch_sizes = mit_config["patch_sizes"]
        self.paddings = mit_config["paddings"]

        # MiT backbone
        self.backbone = MiTBackbone(
            in_channels=in_channels,
            embed_dims=self.embed_dims,
            depths=self.depths,
            num_heads=self.num_heads,
            reduction_ratios=self.reduction_ratios,
            mlp_ratios=self.mlp_ratios,
            strides=self.strides,
            patch_sizes=self.patch_sizes,
            paddings=self.paddings,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout=dropout,
        )

        # Atrous Enhancement (optional, between backbone and decoder)
        # Operates on the final stage feature map [B, C4, H/32, W/32]
        self.atrous: nn.Module
        if atrous_enabled:
            self.atrous = AtrousEnhancement(
                dim=self.embed_dims[3],  # final stage dimension
                rates=self.atrous_rates,
                reduction=self.atrous_reduction,
            )
        else:
            self.atrous = nn.Identity()

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
        # MiT backbone: multi-scale features
        features = self.backbone(x)  # list of 4 tensors

        # Atrous Enhancement on the final stage feature
        # This enhances the highest-level semantic features before decoding
        features[3] = self.atrous(features[3])

        # MLP decoder: aggregate features → segmentation logits
        logits = self.decoder(features)  # [B, num_classes, H, W]

        return logits

    @classmethod
    def from_config(cls, config: Any) -> SegFormer:
        """Instantiate model from a config object.

        Args:
            config: Config object with ``model.backbone.*``,
                    ``model.decoder.*``, ``model.atrous.*``,
                    ``model.input.*``, and ``data.num_classes``
                    accessible via ``config.get("...")``.

        Returns:
            Configured SegFormer instance.
        """
        return cls(
            in_channels=config.get("model.input.channels", 3),
            variant=config.get("model.backbone.variant", "B0"),
            num_classes=config.get("data.num_classes", 8),
            decoder_dim=config.get("model.decoder.decoder_dim", 256),
            dropout=config.get("model.decoder.dropout", 0.0),
            qkv_bias=config.get("model.backbone.qkv_bias", False),
            qk_scale=config.get("model.backbone.qk_scale", None),
            atrous_enabled=config.get("model.atrous.enabled", False),
            atrous_rates=config.get("model.atrous.rates", [1, 6, 12, 18]),
            atrous_reduction=config.get("model.atrous.reduction", 4),
        )


__all__ = ["SegFormer"]