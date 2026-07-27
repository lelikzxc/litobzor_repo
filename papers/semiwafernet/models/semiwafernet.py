"""SemiWaferNet: Hybrid CNN–Transformer model for wafer defect analysis.

Supports two modes:
    1. Classification (HybridCNN-ViT): CNN backbone → Transformer → fusion → head
    2. Segmentation (ConvoFormer-UNet): ConvEmbed → ConvoFormer → decoder

Architecture matches the SemiWaferNet paper (Electronics 2026, 15, 1437).
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
            "segmentation":   [B, 1, H, W] logits,
        }

    Args:
        in_channels: Number of input image channels (1 for WM-811K).
        backbone_channels: Channel dimensions for CNN stages.
        embed_dim: Transformer embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        mlp_ratio: MLP hidden dimension ratio.
        dropout: Dropout rate.
        fusion_dim: Feature fusion dimension.
        num_classes: Number of classification classes (9 for WM-811K).
        seg_classes: Number of segmentation classes (1 for binary).
        norm: Normalization type ("bn" or "ln").
        activation: Activation type ("relu" or "gelu").
        mode: "classification" (HybridCNN-ViT) or "segmentation" (ConvoFormer-UNet).
    """

    def __init__(
        self,
        in_channels: int = 1,
        backbone_channels: list[int] | None = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.2,
        fusion_dim: int = 128,
        num_classes: int = 9,
        seg_classes: int = 1,
        norm: str = "bn",
        activation: str = "relu",
        mode: str = "classification",
    ) -> None:
        super().__init__()
        self.mode = mode

        if backbone_channels is None:
            backbone_channels = [64, 128]

        # CNN backbone (HybridCNN-ViT only)
        if mode == "classification":
            self.backbone = CNNBackbone(
                in_channels=in_channels,
                channels=backbone_channels,
                norm=norm,
                activation=activation,
            )

            # Transformer takes CNN stage 2 features (128ch → 8×8 at 32×32 input)
            transformer_in_channels = backbone_channels[-1]  # 128
        else:
            self.backbone = nn.Identity()
            # For segmentation, ConvEmbed takes raw input (1ch)
            transformer_in_channels = in_channels  # 1

        # Transformer encoder
        # Classification: PatchProjection + standard blocks
        # Segmentation: ConvEmbed + ConvoFormer blocks
        self.transformer = TransformerEncoder(
            in_channels=transformer_in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_conv_embed=(mode == "segmentation"),
        )

        # Feature fusion (classification only)
        if mode == "classification":
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

            # Segmentation head for classification mode (dummy)
            self.decoder = nn.Identity()
        else:
            # Segmentation mode: no fusion, direct decoder
            self.fusion = nn.Identity()
            self.classifier = nn.Identity()

            self.decoder = SegmentationDecoder(
                in_channels=embed_dim,
                num_classes=seg_classes,
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Dictionary with "classification" and "segmentation" logits.
        """
        if self.mode == "classification":
            # CNN backbone: multi-scale features
            cnn_features = self.backbone(x)  # list of 2 tensors

            # Transformer: global context from stage 2 features
            transformer_tokens, transformer_spatial = self.transformer(cnn_features[-1])

            # Feature fusion
            class_features, seg_features = self.fusion(cnn_features, transformer_tokens, transformer_spatial)

            # Task heads
            class_logits = self.classifier(class_features)
            seg_logits = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)

        else:  # segmentation
            # ConvEmbed + ConvoFormer
            transformer_tokens, transformer_spatial = self.transformer(x)

            # Reshape tokens to spatial feature map for decoder
            H, W = transformer_spatial
            B, N, D = transformer_tokens.shape
            seg_features = transformer_tokens.transpose(1, 2).reshape(B, D, H, W)

            # Decoder
            seg_logits = self.decoder(seg_features)
            class_logits = torch.zeros(x.shape[0], 1, device=x.device)

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
            in_channels=input_cfg.get("in_channels", 1),
            backbone_channels=backbone_cfg.get("channels", [64, 128]),
            embed_dim=transformer_cfg.get("embed_dim", 128),
            num_heads=transformer_cfg.get("num_heads", 8),
            num_layers=transformer_cfg.get("num_layers", 4),
            mlp_ratio=transformer_cfg.get("mlp_ratio", 2),
            dropout=transformer_cfg.get("dropout", 0.2),
            fusion_dim=decoder_cfg.get("embed_dim", 128),
            num_classes=model_cfg.get("num_classes", 9),
            seg_classes=model_cfg.get("seg_classes", 1),
            norm=backbone_cfg.get("norm", "bn"),
            activation=backbone_cfg.get("activation", "relu"),
            mode=model_cfg.get("mode", "classification"),
        )


__all__ = ["SemiWaferNet"]