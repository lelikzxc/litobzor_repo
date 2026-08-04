"""SemiWaferNet: Hybrid CNN–Transformer model for wafer defect analysis.

Supports two modes:
    1. Classification (HybridCNN-ViT): CNN backbone → Transformer (class token) → linear head
    2. Segmentation (ConvoFormer-UNet): ConvEmbed → ConvoFormer → decoder

Architecture matches the SemiWaferNet paper (Electronics 2026, 15, 1437).

Classification (HybridCNN-ViT, Section 2.1):
    - CNN backbone: Conv3×3(64) → BN → ReLU → MaxPool → ResBlock(64→128, stride=2)
    - AdaptiveAvgPool → F_c ∈ R^128×8×8
    - Flatten N=64 tokens → linear project to D=128
    - Positional embeddings + class token + dropout(0.5)
    - Transformer: L=4, 8 heads, dim 128, FFN 256, GELU, dropout 0.2
    - Class token → linear head: ŷ = Softmax(W_cls·z_cls + b_cls)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from papers.semiwafernet.modules.cnn_backbone import CNNBackbone
from papers.semiwafernet.modules.transformer import HybridViTEncoder, TransformerEncoder
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
        dropout: Dropout rate.
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
        dropout: float = 0.2,
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

        if mode == "classification":
            # CNN backbone (HybridCNN-ViT)
            self.backbone = CNNBackbone(
                in_channels=in_channels,
                channels=backbone_channels,
                norm=norm,
                activation=activation,
            )

            # Adaptive average pooling to fixed spatial size (8×8 at 32×32 input)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

            # Transformer takes CNN stage 2 features (128ch → 8×8 at 32×32 input)
            transformer_in_channels = backbone_channels[-1]  # 128

            # HybridCNN-ViT encoder with class token, positional embeddings,
            # dropout(0.5), L=4 layers, 8 heads, dim 128, FFN 256, dropout 0.2
            self.transformer = HybridViTEncoder(
                in_channels=transformer_in_channels,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                num_tokens=64,
                dropout_cls=0.5,
                dropout=dropout,
            )

            # Linear classification head on the class token:
            #   ŷ = Softmax(W_cls·z_cls + b_cls)  (Equation 4)
            self.classifier = nn.Linear(embed_dim, num_classes)

            # Segmentation head for classification mode (dummy)
            self.decoder = nn.Identity()
        else:
            # Segmentation mode: no CNN backbone, direct ConvEmbed + ConvoFormer
            self.backbone = nn.Identity()
            self.adaptive_pool = nn.Identity()

            # ConvEmbed takes raw input (1ch)
            self.transformer = TransformerEncoder(
                in_channels=in_channels,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=2,
                dropout=dropout,
                use_conv_embed=True,
            )

            self.classifier = nn.Identity()

            self.decoder = SegmentationDecoder(
                in_channels=embed_dim,
                num_classes=seg_classes,
            )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].
            return_aux: If True (segmentation mode), return auxiliary decoder
                outputs for deep supervision (Equation 17). Ignored in
                classification mode.

        Returns:
            Dictionary with "classification" and "segmentation" logits.
            In segmentation mode with ``return_aux=True``, the "segmentation"
            value is a dict with keys "main", "aux1", "aux2".
        """
        if self.mode == "classification":
            # CNN backbone: multi-scale features
            cnn_features = self.backbone(x)  # list of 2 tensors

            # Adaptive average pooling to fixed 8×8 spatial size
            pooled = self.adaptive_pool(cnn_features[-1])  # [B, 128, 8, 8]

            # Transformer: class token from pooled features
            class_token = self.transformer(pooled)  # [B, embed_dim]

            # Linear classification head on the class token
            class_logits = self.classifier(class_token)  # [B, num_classes]
            seg_logits = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)

        else:  # segmentation
            # ConvEmbed + ConvoFormer
            transformer_tokens, transformer_spatial = self.transformer(x)

            # Reshape tokens to spatial feature map for decoder
            H, W = transformer_spatial
            B, N, D = transformer_tokens.shape
            seg_features = transformer_tokens.transpose(1, 2).reshape(B, D, H, W)

            # Decoder (with auxiliary outputs for deep supervision)
            seg_logits = self.decoder(seg_features, return_aux=return_aux)
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
        input_cfg = config.get("model.input", {})
        model_cfg = config.get("model", {})

        return cls(
            in_channels=input_cfg.get("in_channels", 1),
            backbone_channels=backbone_cfg.get("channels", [64, 128]),
            embed_dim=transformer_cfg.get("embed_dim", 128),
            num_heads=transformer_cfg.get("num_heads", 8),
            num_layers=transformer_cfg.get("num_layers", 4),
            dropout=transformer_cfg.get("dropout", 0.2),
            num_classes=model_cfg.get("num_classes", 9),
            seg_classes=model_cfg.get("seg_classes", 1),
            norm=backbone_cfg.get("norm", "bn"),
            activation=backbone_cfg.get("activation", "relu"),
            mode=model_cfg.get("mode", "classification"),
        )


__all__ = ["SemiWaferNet"]