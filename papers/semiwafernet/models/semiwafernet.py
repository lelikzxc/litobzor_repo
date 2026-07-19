"""SemiWaferNet: Hybrid CNN–Transformer model for wafer defect analysis.

Combines a CNN backbone, transformer encoder, feature fusion,
classification head, and segmentation decoder into a single model.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from papers.semiwafernet.modules.cnn_backbone import CNNBackbone
from papers.semiwafernet.modules.transformer import TransformerEncoder
from papers.semiwafernet.modules.fusion import FeatureFusion
from papers.semiwafernet.models.classifier import ClassifierHead
from papers.semiwafernet.models.decoder import SegmentationDecoder


class SemiWaferNet(nn.Module):
    """Hybrid CNN–Transformer model for wafer defect classification and segmentation.

    Forward returns a dictionary:
        {
            "classification": [B, num_classes] logits,
            "segmentation":   [B, num_classes, H, W] logits,
        }
    """

    def __init__(
        self,
        in_channels: int = 3,
        backbone_channels: list[int] | None = None,
        backbone_depths: list[int] | None = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        fusion_dim: int = 256,
        num_classes: int = 6,
        norm: str = "bn",
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if backbone_channels is None:
            backbone_channels = [64, 128, 256, 512]
        if backbone_depths is None:
            backbone_depths = [2, 2, 6, 2]

        self.backbone = CNNBackbone(
            in_channels=in_channels,
            channels=backbone_channels,
            depths=backbone_depths,
            norm=norm,
            activation=activation,
        )

        self.transformer = TransformerEncoder(
            in_channels=backbone_channels[-1],  # Stage 4 output channels
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.fusion = FeatureFusion(
            cnn_channels=backbone_channels,
            transformer_dim=embed_dim,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
        )

        self.classifier = ClassifierHead(
            in_channels=fusion_dim,
            num_classes=num_classes,
        )

        self.decoder = SegmentationDecoder(
            in_channels=fusion_dim,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # CNN backbone: multi-scale features
        cnn_features = self.backbone(x)  # list of 4 tensors

        # Transformer: global context from stage 4 features
        transformer_tokens, transformer_spatial = self.transformer(cnn_features[-1])

        # Feature fusion
        class_features, seg_features = self.fusion(cnn_features, transformer_tokens, transformer_spatial)

        # Task heads
        class_logits = self.classifier(class_features)
        seg_logits = self.decoder(seg_features)

        return {
            "classification": class_logits,
            "segmentation": seg_logits,
        }

    @classmethod
    def from_config(cls, config: Any) -> SemiWaferNet:
        """Build SemiWaferNet from a configuration object.

        Args:
            config: A Config object (common.utils.config.Config) with
                dot-notation access via config.get(key, default).

        Returns:
            Configured SemiWaferNet instance.
        """
        backbone_cfg = config.get("model.backbone", {})
        transformer_cfg = config.get("model.transformer", {})
        decoder_cfg = config.get("model.decoder", {})
        input_cfg = config.get("model.input", {})
        model_cfg = config.get("model", {})

        return cls(
            in_channels=input_cfg.get("in_channels", 3),
            backbone_channels=backbone_cfg.get("channels", [64, 128, 256, 512]),
            backbone_depths=backbone_cfg.get("depths", [2, 2, 6, 2]),
            embed_dim=transformer_cfg.get("embed_dim", 256),
            num_heads=transformer_cfg.get("num_heads", 8),
            num_layers=transformer_cfg.get("num_layers", 4),
            mlp_ratio=transformer_cfg.get("mlp_ratio", 4),
            dropout=transformer_cfg.get("dropout", 0.1),
            fusion_dim=decoder_cfg.get("embed_dim", 256),
            num_classes=model_cfg.get("num_classes", 6),
            norm=backbone_cfg.get("norm", "bn"),
            activation=backbone_cfg.get("activation", "relu"),
        )