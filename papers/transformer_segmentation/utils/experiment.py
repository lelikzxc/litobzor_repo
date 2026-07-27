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
        backbone: Encoder variant (e.g. ``"Hybrid-B0"``).
        num_classes: Number of output segmentation classes.
        input_size: Input image size (assumed square).
        decoder_dim: Common projection dimension for the MLP decoder.
        total_params: Total trainable parameters.
        encoder_params: Parameters in the hybrid encoder.
        decoder_params: Parameters in the MLP decoder.
        architecture_summary: Human-readable architecture description.
    """

    model_name: str = "segformer_atrous"
    backbone: str = "Hybrid-B0"
    num_classes: int = 7
    input_size: int = 512
    decoder_dim: int = 256
    total_params: int = 0
    encoder_params: int = 0
    decoder_params: int = 0
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
        class_name: Class name to match (e.g. ``"MLPDecoder"``).

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

    Works with ``SegFormer`` instances (hybrid encoder).

    Args:
        model: A SegFormer instance.

    Returns:
        Populated ``ExperimentInfo`` dataclass.
    """
    total = count_params(model)

    # Count parameters by component
    decoder_params = _count_module_params(model, "MLPDecoder")
    encoder_params = total - decoder_params

    # Read attributes from model
    variant = getattr(model, "variant", "B0")
    num_classes = getattr(model, "num_classes", 7)
    decoder_dim = getattr(model, "decoder_dim", 256)

    # Build architecture summary
    parts: list[str] = [
        f"SegFormer Hybrid-{variant}",
        f"classes={num_classes}",
    ]
    arch_summary = " | ".join(parts)

    return ExperimentInfo(
        model_name="segformer_atrous",
        backbone=f"Hybrid-{variant}",
        num_classes=num_classes,
        input_size=512,
        decoder_dim=decoder_dim,
        total_params=total,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
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
        "  Parameters:",
        f"    Total:            {info.total_params:>10,}",
        f"    Encoder:          {info.encoder_params:>10,}",
        f"    Decoder:          {info.decoder_params:>10,}",
        f"  Architecture:       {info.architecture_summary}",
        "=" * 56,
    ]
    return "\n".join(lines)