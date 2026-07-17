"""Experiment metadata utilities for CTM-YOLOv10.

Provides a single source of truth for model metadata, parameter counting,
and architecture summaries used across training, evaluation, and logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ExperimentInfo:
    """Immutable snapshot of experiment metadata.

    Attributes:
        model_name: YOLOv10 variant name (e.g. ``"yolov10n"``).
        num_classes: Number of output classes.
        ctm_enabled: Whether CTM is active.
        ctm_dim: CTM embedding dimension (``None`` if disabled).
        ctm_num_heads: CTM attention heads (``None`` if disabled).
        ctm_mlp_ratio: CTM MLP hidden ratio (``None`` if disabled).
        total_params: Total trainable parameters.
        backbone_params: Parameters in the YOLOv10 backbone.
        ctm_params: Parameters in the CTM module (0 if disabled).
        architecture_summary: Human-readable architecture description.
    """

    model_name: str = "yolov10n"
    num_classes: int = 80
    ctm_enabled: bool = False
    ctm_dim: int | None = None
    ctm_num_heads: int | None = None
    ctm_mlp_ratio: float | None = None
    total_params: int = 0
    backbone_params: int = 0
    ctm_params: int = 0
    architecture_summary: str = ""


def count_params(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: A PyTorch module.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_experiment_info(model: torch.nn.Module) -> ExperimentInfo:
    """Build an ``ExperimentInfo`` snapshot from a model instance.

    Works with both ``YOLOv10Baseline`` and ``CTMYOLOv10``.

    Args:
        model: A YOLOv10Baseline or CTMYOLOv10 instance.

    Returns:
        Populated ``ExperimentInfo`` dataclass.
    """
    total = count_params(model)

    # Detect model type by checking for CTM attributes
    ctm_enabled = getattr(model, "ctm_enabled", False)
    ctm_module = getattr(model, "ctm", None)
    ctm_params = count_params(ctm_module) if ctm_module is not None else 0
    backbone_params = total - ctm_params

    ctm_dim = None
    ctm_num_heads = None
    ctm_mlp_ratio = None
    if ctm_module is not None:
        ctm_dim = getattr(ctm_module, "dim", None)
        ctm_num_heads = getattr(ctm_module, "num_heads", None)
        ctm_mlp_ratio = getattr(ctm_module, "mlp_ratio", None)

    model_name = getattr(model, "model_name", "unknown")
    num_classes = getattr(model, "num_classes", 80)

    # Build architecture summary
    parts: list[str] = [
        f"YOLOv10 ({model_name})",
    ]
    if ctm_enabled:
        parts.append(
            f"+ CTM(dim={ctm_dim}, heads={ctm_num_heads}, "
            f"mlp_ratio={ctm_mlp_ratio})"
        )
    parts.append(f"classes={num_classes}")
    arch_summary = " | ".join(parts)

    return ExperimentInfo(
        model_name=model_name,
        num_classes=num_classes,
        ctm_enabled=ctm_enabled,
        ctm_dim=ctm_dim,
        ctm_num_heads=ctm_num_heads,
        ctm_mlp_ratio=ctm_mlp_ratio,
        total_params=total,
        backbone_params=backbone_params,
        ctm_params=ctm_params,
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
        "  Experiment Metadata",
        "=" * 56,
        f"  Model:              {info.model_name}",
        f"  Classes:            {info.num_classes}",
        f"  CTM enabled:        {info.ctm_enabled}",
    ]
    if info.ctm_enabled and info.ctm_dim is not None:
        lines += [
            f"  CTM dim:            {info.ctm_dim}",
            f"  CTM heads:          {info.ctm_num_heads}",
            f"  CTM MLP ratio:      {info.ctm_mlp_ratio}",
        ]
    lines += [
        f"  Total params:       {info.total_params:>10,}",
        f"  Backbone params:    {info.backbone_params:>10,}",
        f"  CTM params:         {info.ctm_params:>10,}",
        f"  Architecture:       {info.architecture_summary}",
        "=" * 56,
    ]
    return "\n".join(lines)