"""Experiment metadata utilities for FCS-VMamba.

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
        model_name: Model variant name (e.g. ``"fcs_vmamba"``).
        backbone: Backbone architecture name (e.g. ``"vmamba_tiny"``).
        num_classes: Number of output classes.
        input_size: Input image size (assumed square).
        embed_dim: Base embedding dimension.
        depths: Number of VSSBlocks per stage (list of 4 ints).
        ssm_ratio: SSM expansion ratio.
        mlp_ratio: MLP hidden dimension ratio.
        fa_enabled: Whether Frequency Attention is active.
        sfs_enabled: Whether Saliency Feature Suppression is active.
        clca_enabled: Whether Cross-Layer Channel Attention is active.
        fa_reduction: Reduction ratio for FA (``None`` if disabled).
        sfs_reduction: Reduction ratio for SFS (``None`` if disabled).
        clca_reduction: Reduction ratio for CLCA (``None`` if disabled).
        total_params: Total trainable parameters.
        backbone_params: Parameters in the VMamba backbone (SS2D + MLP).
        fcs_params: Parameters in FCS modules (FA + SFS + CLCA).
        architecture_summary: Human-readable architecture description.
    """

    model_name: str = "fcs_vmamba"
    backbone: str = "vmamba_tiny"
    num_classes: int = 8
    input_size: int = 224
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2, 6, 2)
    ssm_ratio: float = 2.0
    mlp_ratio: float = 4.0
    fa_enabled: bool = False
    sfs_enabled: bool = False
    clca_enabled: bool = False
    fa_reduction: int | None = None
    sfs_reduction: int | None = None
    clca_reduction: int | None = None
    total_params: int = 0
    backbone_params: int = 0
    fcs_params: int = 0
    architecture_summary: str = ""


def count_params(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: A PyTorch module.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_fcs_params(model: torch.nn.Module) -> int:
    """Count parameters in FCS-specific modules (FA, SFS, CLCA).

    Iterates through all modules and sums parameters of ``FrequencyAttention``,
    ``SaliencySuppression``, and ``CrossLayerChannelAttention`` instances.

    Args:
        model: An FCSVMamba instance.

    Returns:
        Number of trainable parameters in FCS modules.
    """
    total = 0
    for module in model.modules():
        module_name = type(module).__name__
        if module_name in ("FrequencyAttention", "SaliencySuppression", "CrossLayerChannelAttention"):
            total += count_params(module)
    return total


def build_experiment_info(model: torch.nn.Module) -> ExperimentInfo:
    """Build an ``ExperimentInfo`` snapshot from a model instance.

    Works with ``FCSVMamba`` instances.

    Args:
        model: An FCSVMamba instance.

    Returns:
        Populated ``ExperimentInfo`` dataclass.
    """
    total = count_params(model)
    fcs_params = _count_fcs_params(model)
    backbone_params = total - fcs_params

    # Read attributes from model (with defaults for safety)
    model_name = getattr(model, "model_name", "fcs_vmamba")
    backbone = getattr(model, "backbone", "vmamba_tiny")
    num_classes = getattr(model, "num_classes", 8)
    input_size = getattr(model, "image_size", 224)
    embed_dim = getattr(model, "embed_dim", 96)
    depths = tuple(getattr(model, "depths", [2, 2, 6, 2]))
    ssm_ratio = getattr(model, "ssm_ratio", 2.0)
    mlp_ratio = getattr(model, "mlp_ratio", 4.0)
    fa_enabled = getattr(model, "fa_enabled", False)
    sfs_enabled = getattr(model, "sfs_enabled", False)
    clca_enabled = getattr(model, "clca_enabled", False)
    fa_reduction = getattr(model, "fa_reduction", None) if fa_enabled else None
    sfs_reduction = getattr(model, "sfs_reduction", None) if sfs_enabled else None
    clca_reduction = getattr(model, "clca_reduction", None) if clca_enabled else None

    # Build architecture summary
    parts: list[str] = [
        f"FCS-VMamba ({backbone})",
    ]
    enabled_modules: list[str] = []
    if fa_enabled:
        enabled_modules.append(f"FA(r={fa_reduction})")
    if sfs_enabled:
        enabled_modules.append(f"SFS(r={sfs_reduction})")
    if clca_enabled:
        enabled_modules.append(f"CLCA(r={clca_reduction})")
    if enabled_modules:
        parts.append("+".join(enabled_modules))
    else:
        parts.append("baseline (no FCS modules)")
    parts.append(f"classes={num_classes}")
    arch_summary = " | ".join(parts)

    return ExperimentInfo(
        model_name=model_name,
        backbone=backbone,
        num_classes=num_classes,
        input_size=input_size,
        embed_dim=embed_dim,
        depths=depths,
        ssm_ratio=ssm_ratio,
        mlp_ratio=mlp_ratio,
        fa_enabled=fa_enabled,
        sfs_enabled=sfs_enabled,
        clca_enabled=clca_enabled,
        fa_reduction=fa_reduction,
        sfs_reduction=sfs_reduction,
        clca_reduction=clca_reduction,
        total_params=total,
        backbone_params=backbone_params,
        fcs_params=fcs_params,
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
        "  FCS-VMamba Research Validation",
        "=" * 56,
        f"  Model:              {info.model_name} ({info.backbone})",
        f"  Classes:            {info.num_classes}",
        f"  Input size:         {info.input_size}×{info.input_size}",
        f"  Embed dim:          {info.embed_dim}",
        f"  Depths:             {list(info.depths)}",
        f"  SSM ratio:          {info.ssm_ratio}",
        f"  MLP ratio:          {info.mlp_ratio}",
        "",
        "  FCS modules:",
        f"    FA enabled:       {info.fa_enabled}",
        f"    FA reduction:     {info.fa_reduction}",
        f"    SFS enabled:      {info.sfs_enabled}",
        f"    SFS reduction:    {info.sfs_reduction}",
        f"    CLCA enabled:     {info.clca_enabled}",
        f"    CLCA reduction:   {info.clca_reduction}",
        "",
        "  Parameters:",
        f"    Total:            {info.total_params:>10,}",
        f"    Backbone:         {info.backbone_params:>10,}",
        f"    FCS modules:      {info.fcs_params:>10,}",
        f"    Added by FCS:     {info.fcs_params:>10,}",
        f"  Architecture:       {info.architecture_summary}",
        "=" * 56,
    ]
    return "\n".join(lines)