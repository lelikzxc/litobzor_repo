"""Experiment metadata utilities for SemiWaferNet."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from papers.semiwafernet.models.semiwafernet import SemiWaferNet


@dataclass
class ExperimentInfo:
    """Immutable snapshot of experiment metadata."""

    model_name: str = "semiwafernet"
    mode: str = "classification"
    num_classes: int = 9
    image_size: int = 32
    backbone_channels: list[int] = field(default_factory=lambda: [64, 128])
    transformer_embed_dim: int = 128
    transformer_layers: int = 4
    total_params: int = 0
    backbone_params: int = 0
    transformer_params: int = 0
    fusion_params: int = 0
    classifier_params: int = 0
    decoder_params: int = 0
    ema_enabled: bool = False
    pseudo_labels_enabled: bool = False
    consistency_enabled: bool = False
    architecture_summary: str = ""


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_experiment_info(
    model: SemiWaferNet,
    ema_enabled: bool = False,
    pseudo_labels_enabled: bool = False,
    consistency_enabled: bool = False,
) -> ExperimentInfo:
    """Build an ``ExperimentInfo`` snapshot from a SemiWaferNet instance."""
    total = count_params(model)
    mode = getattr(model, "mode", "classification")

    backbone_params = count_params(model.backbone) if not isinstance(model.backbone, torch.nn.Identity) else 0
    if hasattr(model, "transformer") and not isinstance(model.transformer, torch.nn.Identity):
        transformer_params = count_params(model.transformer)
    else:
        transformer_params = 0
    fusion_params = count_params(model.fusion) if hasattr(model, "fusion") else 0
    classifier_params = count_params(model.classifier) if hasattr(model, "classifier") else 0
    if mode == "segmentation" and getattr(model, "seg_model", None) is not None:
        decoder_params = count_params(model.seg_model)
    else:
        decoder_params = count_params(model.decoder) if not isinstance(model.decoder, torch.nn.Identity) else 0

    # Recover hyperparams
    if hasattr(model, "backbone") and hasattr(model.backbone, "out_channels"):
        backbone_channels = list(model.backbone.out_channels)
    else:
        backbone_channels = [64, 128]
    embed_dim = getattr(model.transformer, "embed_dim", 128)
    num_layers = len(getattr(model.transformer, "blocks", [])) if hasattr(model.transformer, "blocks") else 4
    if hasattr(model.classifier, "head"):
        num_classes = model.classifier.head.out_features
    else:
        num_classes = getattr(model, "num_classes", 9)

    arch = (
        f"SemiWaferNet({mode}) | CNN({backbone_channels}) | "
        f"Transformer(embed={embed_dim}, layers={num_layers}) | classes={num_classes}"
    )

    return ExperimentInfo(
        model_name="semiwafernet",
        mode=mode,
        num_classes=num_classes,
        image_size=32 if mode == "classification" else 64,
        backbone_channels=backbone_channels,
        transformer_embed_dim=embed_dim,
        transformer_layers=num_layers,
        total_params=total,
        backbone_params=backbone_params,
        transformer_params=transformer_params,
        fusion_params=fusion_params,
        classifier_params=classifier_params,
        decoder_params=decoder_params,
        ema_enabled=ema_enabled,
        pseudo_labels_enabled=pseudo_labels_enabled,
        consistency_enabled=consistency_enabled,
        architecture_summary=arch,
    )


def format_experiment_info(info: ExperimentInfo) -> str:
    lines = [
        "=" * 56,
        "  SemiWaferNet Experiment Metadata",
        "=" * 56,
        f"  Model:              {info.model_name} ({info.mode})",
        f"  Classes:            {info.num_classes}",
        f"  Image size:         {info.image_size}×{info.image_size}",
        f"  Backbone channels:  {info.backbone_channels}",
        f"  Transformer embed:  {info.transformer_embed_dim}",
        f"  Transformer layers: {info.transformer_layers}",
        "",
        "  Parameters:",
        f"    Total:            {info.total_params:>10,}",
        f"    CNN backbone:     {info.backbone_params:>10,}",
        f"    Transformer:      {info.transformer_params:>10,}",
        f"    Feature fusion:   {info.fusion_params:>10,}",
        f"    Classifier head:  {info.classifier_params:>10,}",
        f"    Segmentation dec: {info.decoder_params:>10,}",
        "",
        "  Semi-supervised:",
        f"    EMA teacher:      {info.ema_enabled}",
        f"    Pseudo labels:    {info.pseudo_labels_enabled}",
        f"    Consistency:      {info.consistency_enabled}",
        f"  Architecture:       {info.architecture_summary}",
        "=" * 56,
    ]
    return "\n".join(lines)
