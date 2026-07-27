"""Experiment metadata utilities for CTM-IYOLOv10.

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
        ghost_conv: Whether GhostConv is enabled.
        bifpn: Whether BiFPN is enabled.
        total_params: Total trainable parameters.
        backbone_params: Parameters in the YOLOv10 backbone.
        architecture_summary: Human-readable architecture description.
    """

    model_name: str = "yolov10n"
    num_classes: int = 80
    ghost_conv: bool = False
    bifpn: bool = False
    total_params: int = 0
    backbone_params: int = 0
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

    Works with both ``YOLOv10Baseline`` and ``CTMIYOLOv10``.

    Args:
        model: A YOLOv10Baseline or CTMIYOLOv10 instance.

    Returns:
        Populated ``ExperimentInfo`` dataclass.
    """
    total = count_params(model)

    ghost_conv = getattr(model, "ghost_conv", False)
    bifpn = getattr(model, "bifpn", False)

    model_name = getattr(model, "model_name", "unknown")
    num_classes = getattr(model, "num_classes", 80)

    # Build architecture summary
    parts: list[str] = [
        f"YOLOv10 ({model_name})",
    ]
    if ghost_conv:
        parts.append("+ GhostConv")
    if bifpn:
        parts.append("+ BiFPN")
    parts.append(f"classes={num_classes}")
    arch_summary = " | ".join(parts)

    return ExperimentInfo(
        model_name=model_name,
        num_classes=num_classes,
        ghost_conv=ghost_conv,
        bifpn=bifpn,
        total_params=total,
        backbone_params=total,
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
        f"  GhostConv:          {info.ghost_conv}",
        f"  BiFPN:              {info.bifpn}",
        f"  Total params:       {info.total_params:>10,}",
        f"  Architecture:       {info.architecture_summary}",
        "=" * 56,
    ]
    return "\n".join(lines)