"""Experiment metadata utilities for SemiWaferNet.

Provides a single source of truth for model metadata, parameter counting,
and architecture summaries used across training, evaluation, and logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from papers.semiwafernet.models.semiwafernet import SemiWaferNet


@dataclass
class ExperimentInfo:
    """Immutable snapshot of experiment metadata.

    Attributes:
        model_name: Model name (e.g. ``"semiwafernet"``).
        num_classes: Number of output classes.
        image_size: Input image size (assumed square).
        backbone_channels: Channel list for CNN backbone stages.
        transformer_embed_dim: Transformer embedding dimension.
        transformer_layers: Number of transformer encoder layers.
        total_params: Total trainable parameters.
        backbone_params: Parameters in the CNN backbone.
        transformer_params: Parameters in the transformer encoder.
        fusion_params: Parameters in the feature fusion module.
        classifier_params: Parameters in the classification head.
        decoder_params: Parameters in the segmentation decoder.
        ema_enabled: Whether EMA teacher is enabled.
        pseudo_labels_enabled: Whether pseudo-label generation is enabled.
        consistency_enabled: Whether consistency loss is enabled.
        architecture_summary: Human-readable architecture description.
    """

    model_name: str = "semiwafernet"
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
    """Count trainable parameters in a model.

    Args:
        model: A PyTorch module.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_module_params(model: torch.nn.Module, class_name: str) -> int:
    """Count parameters in all sub-modules of a given class name.

    Args:
        model: A PyTorch module.
        class_name: Class name to match (e.g. ``"CNNBackbone"``).

    Returns:
        Number of trainable parameters in matching sub-modules.
    """
    total = 0
    for module in model.modules():
        if type(module).__name__ == class_name:
            total += count_params(module)
    return total


def build_experiment_info(
    model: SemiWaferNet,
    ema_enabled: bool = False,
    pseudo_labels_enabled: bool = False,
    consistency_enabled: bool = False,
) -> ExperimentInfo:
    """Build an ``ExperimentInfo`` snapshot from a model instance.

    Args:
        model: A SemiWaferNet instance.
        ema_enabled: Whether EMA teacher is enabled.
        pseudo_labels_enabled: Whether pseudo-label generation is enabled.
        consistency_enabled: Whether consistency loss is enabled.

    Returns:
        Populated ``ExperimentInfo`` dataclass.
    """
    total = count_params(model)

    # Count parameters by component using class name matching
    backbone_params = _count_module_params(model, "CNNBackbone")
    transformer_params = _count_module_params(model, "TransformerEncoder")
    fusion_params = _count_module_params(model, "FeatureFusion")
    classifier_params = _count_module_params(model, "ClassifierHead")
    decoder_params = _count_module_params(model, "SegmentationDecoder")

    # Read attributes from model
    backbone_channels = getattr(model, "backbone_channels", [64, 128])
    embed_dim = getattr(model.transformer, "embed_dim", 128) if hasattr(model, "transformer") else 128
    num_layers = len(getattr(model.transformer, "blocks", [])) if hasattr(model, "transformer") else 4
    num_classes = getattr(model.classifier, "num_classes", 9) if hasattr(model, "classifier") else 9

    # Build architecture summary
    parts: list[str] = [
        f"SemiWaferNet",
        f"CNN({backbone_channels})",
        f"Transformer(embed={embed_dim}, layers={num_layers})",
        f"classes={num_classes}",
    ]
    arch_summary = " | ".join(parts)

    return ExperimentInfo(
        model_name="semiwafernet",
        num_classes=num_classes,
        image_size=32,
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
        architecture_summary=arch_summary,
    )


def format_experiment_info(info: ExperimentInfo) -> str:
    """Format ``ExperimentInfo`` as a human-readable string.

    Args:
        info: Experiment metadata.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        "=" * 56,
        "  SemiWaferNet — Experiment Metadata",
        "=" * 56,
        f"  Model:              {info.model_name}",
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
        "",
        f"  Architecture:       {info.architecture_summary}",
        "=" * 56,
    ]
    return "\n".join(lines)