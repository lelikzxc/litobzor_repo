"""SemiWaferNet: two hybrid CNN–Transformer models from the paper.

Electronics 2026, 15, 1437 — *not* a single multitask network:

1. **HybridCNN-ViT** (classification, Section 2.1)
2. **ConvoFormer-UNet** (segmentation, Section 3.1)

``SemiWaferNet(mode=...)`` selects which model to build. Forward always
returns a dict with both keys for a stable training/engine interface; the
unused task is filled with zeros.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from papers.semiwafernet.modules.cnn_backbone import CNNBackbone
from papers.semiwafernet.modules.transformer import HybridViTEncoder
from papers.semiwafernet.models.convoformer_unet import ConvoFormerUNet


class SemiWaferNet(nn.Module):
    """Paper-faithful HybridCNN-ViT / ConvoFormer-UNet wrapper.

    Args:
        mode: ``"classification"`` or ``"segmentation"``.
        in_channels: Input channels (1 for WM-811K).
        backbone_channels: CNN stage widths for HybridCNN-ViT.
        embed_dim: Transformer width.
        num_heads / num_layers / dropout: Transformer hyperparams.
        num_classes: Classification classes (9).
        seg_classes: Segmentation output channels (1 = binary logits).
        mlp_ratio / fusion_dim: Accepted for API compatibility.
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
        mlp_ratio: int = 2,
        fusion_dim: int | None = None,  # unused; kept for test/API compat
        base_channels: int = 66,
        seg_embed_dim: int = 160,
        seg_mlp_ratio: int = 2,
        dropout_cls: float = 0.5,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.seg_classes = seg_classes

        if backbone_channels is None:
            backbone_channels = [64, 128]

        # Placeholders so hasattr(..., "fusion") etc. stay true for both modes
        self.fusion = nn.Identity()

        if mode == "classification":
            self.backbone = CNNBackbone(
                in_channels=in_channels,
                channels=backbone_channels,
                norm=norm,
                activation=activation,
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
            self.transformer = HybridViTEncoder(
                in_channels=backbone_channels[-1],
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                num_tokens=64,
                dropout_cls=dropout_cls,
                dropout=dropout,
            )
            # ModuleDict exposes ``classifier.head`` (Linear) for tests/API compat
            self.classifier = nn.ModuleDict({"head": nn.Linear(embed_dim, num_classes)})
            self.decoder = nn.Identity()
            self.seg_model = None
        else:
            self.backbone = nn.Identity()
            self.adaptive_pool = nn.Identity()
            self.transformer = nn.Identity()
            self.classifier = nn.ModuleDict({"head": nn.Linear(1, 1)})  # unused stub
            self.seg_model = ConvoFormerUNet(
                in_channels=in_channels,
                base_channels=base_channels,
                embed_dim=seg_embed_dim if seg_embed_dim else max(embed_dim, 256),
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=seg_mlp_ratio,
                dropout=dropout,
                num_classes=seg_classes,
            )
            self.decoder = self.seg_model  # alias for tests

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.mode == "classification":
            feats = self.backbone(x)
            pooled = self.adaptive_pool(feats[-1])
            class_token = self.transformer(pooled)
            class_logits = self.classifier["head"](class_token)
            seg_logits = torch.zeros(
                x.shape[0], self.seg_classes, x.shape[2], x.shape[3], device=x.device
            )
        else:
            assert self.seg_model is not None
            seg_logits = self.seg_model(x, return_aux=return_aux)
            class_logits = torch.zeros(x.shape[0], self.num_classes, device=x.device)

        return {"classification": class_logits, "segmentation": seg_logits}

    @classmethod
    def from_config(cls, config: Any) -> SemiWaferNet:
        backbone_cfg = config.get("model.backbone", {}) or {}
        transformer_cfg = config.get("model.transformer", {}) or {}
        decoder_cfg = config.get("model.decoder", {}) or {}
        input_cfg = config.get("model.input", {}) or {}
        model_cfg = config.get("model", {}) or {}

        mode = model_cfg.get("mode", "classification")
        default_embed = 160 if mode == "segmentation" else 128

        return cls(
            in_channels=input_cfg.get("in_channels", 1),
            backbone_channels=backbone_cfg.get("channels", [64, 128]),
            embed_dim=transformer_cfg.get("embed_dim", default_embed),
            num_heads=transformer_cfg.get("num_heads", 8),
            num_layers=transformer_cfg.get("num_layers", 4),
            dropout=transformer_cfg.get("dropout", 0.2 if mode == "classification" else 0.1),
            dropout_cls=transformer_cfg.get("dropout_cls", 0.5),
            num_classes=model_cfg.get("num_classes", 9),
            seg_classes=model_cfg.get("seg_classes", 1),
            norm=backbone_cfg.get("norm", "bn"),
            activation=backbone_cfg.get("activation", "relu"),
            mode=mode,
            mlp_ratio=transformer_cfg.get("mlp_ratio", 2),
            base_channels=decoder_cfg.get("base_channels", 66),
            seg_embed_dim=decoder_cfg.get("embed_dim", 160),
            seg_mlp_ratio=transformer_cfg.get("seg_mlp_ratio", decoder_cfg.get("mlp_ratio", 2)),
        )


__all__ = ["SemiWaferNet"]
