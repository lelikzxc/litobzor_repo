"""Experiment metadata utilities for SegFormer + Atrous.

Provides a single source of truth for model metadata, parameter counting,
and architecture summaries used across training, evaluation, and logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ExperimentInfo:
    """Immutable snapshot of experiment metadata.

    Attributes:
        model_name: Model name (e.g. ``"segformer_atrous"``).
        backbone: Backbone variant (e.g. ``"MiT-B0"``).
        num_classes: Number of output segmentation classes.
        input_size: Input image size (assumed square).
        decoder_dim: Common projection dimension for the MLP decoder.
        atrous_enabled: Whether Atrous Enhancement is active.
        atrous_rates: Dilation rates for atrous convolutions (``None`` if disabled).
        atrous_reduction: Channel reduction ratio for atrous bottleneck (``None`` if disabled).
        total_params: Total trainable parameters.
        backbone_params: Parameters in the MiT backbone.
        decoder_params: Parameters in the MLP decoder.
        atrous_params: Parameters in the Atrous module (0 if disabled).
        architecture_summary: Human-readable architecture description.
    """

    model_name: str = "segformer_atrous"
    backbone: str = "MiT-B0"
    num_classes: int = 8
    input_size: int = 512
    decoder_dim: int = 256
    atrous_enabled: bool = False
    atrous_rates: list[int] | None = None
    atrous_reduction: int | None = None
    total_params: int = 0
    backbone_params: int = 0
    decoder_params: int = 0
    atrous_params: int = 0
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
        class_name: Class name to match (e.g. ``"AtrousEnhancement"``).

    Returns:
        Number of trainable parameters in matching sub-modules.
    """
    total = 0
    for module in model.modules():
        if type(module).__name__ == class_name:
            total += count_params(module)
    return total


def build_experiment_info(model: torch.nn.Module) -> ExperimentInfo:
    """Build an ``ExperimentInfo`` snapshot from a model instance.

    Works with ``SegFormer`` instances (both baseline and Atrous-enhanced).

    Args:
        model: A SegFormer instance.

    Returns:
        Populated ``ExperimentInfo`` dataclass.
    """
    total = count_params(model)

    # Count parameters by component
    atrous_params = _count_module_params(model, "AtrousEnhancement")
    decoder_params = _count_module_params(model, "MLPDecoder")
    backbone_params = total - atrous_params - decoder_params

    # Read attributes from model
    variant = getattr(model, "variant", "B0")
    num_classes = getattr(model, "num_classes", 8)
    input_size = getattr(model, "in_channels", 3)  # not image_size, but we use 512 default
    decoder_dim = getattr(model, "decoder_dim", 256)
    atrous_enabled = getattr(model, "atrous_enabled", False)
    atrous_rates = getattr(model, "atrous_rates", None) if atrous_enabled else None
    atrous_reduction = getattr(model, "atrous_reduction", None) if atrous_enabled else None

    # Build architecture summary
    parts: list[str] = [
        f"SegFormer ({variant})",
    ]
    if atrous_enabled:
        parts.append(f"+ Atrous(rates={atrous_rates}, r={atrous_reduction})")
    else:
        parts.append("baseline (no Atrous)")
    parts.append(f"classes={num_classes}")
    arch_summary = " | ".join(parts)

    return ExperimentInfo(
        model_name="segformer_atrous",
        backbone=f"MiT-{variant}",
        num_classes=num_classes,
        input_size=512,
        decoder_dim=decoder_dim,
        atrous_enabled=atrous_enabled,
        atrous_rates=list(atrous_rates) if atrous_rates else None,
        atrous_reduction=atrous_reduction,
        total_params=total,
        backbone_params=backbone_params,
        decoder_params=decoder_params,
        atrous_params=atrous_params,
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
        "  SegFormer + Atrous — Research Validation",
        "=" * 56,
        f"  Model:              {info.model_name} ({info.backbone})",
        f"  Classes:            {info.num_classes}",
        f"  Input size:         {info.input_size}×{info.input_size}",
        f"  Decoder dim:        {info.decoder_dim}",
        "",
        "  Atrous module:",
        f"    Enabled:          {info.atrous_enabled}",
    ]
    if info.atrous_enabled and info.atrous_rates is not None:
        lines += [
            f"    Rates:            {info.atrous_rates}",
            f"    Reduction:        {info.atrous_reduction}",
        ]
    lines += [
        "",
        "  Parameters:",
        f"    Total:            {info.total_params:>10,}",
        f"    Backbone:         {info.backbone_params:>10,}",
        f"    Decoder:          {info.decoder_params:>10,}",
        f"    Atrous:           {info.atrous_params:>10,}",
        f"  Architecture:       {info.architecture_summary}",
        "=" * 56,
    ]
    return "\n".join(lines)