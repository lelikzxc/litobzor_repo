"""Tests for Atrous Enhancement module (standalone)."""

from __future__ import annotations

import torch
import pytest

from papers.transformer_segmentation.modules.atrous import AtrousEnhancement
from papers.transformer_segmentation.models.segformer import SegFormer
from papers.transformer_segmentation.utils.experiment import (
    ExperimentInfo,
    build_experiment_info,
    count_params,
    format_experiment_info,
)


# ── Module import tests ──────────────────────────────────────────────────


def test_atrous_import() -> None:
    """Verify AtrousEnhancement can be imported."""
    assert AtrousEnhancement is not None


# ── Module creation tests ────────────────────────────────────────────────


def test_atrous_creation() -> None:
    """Verify AtrousEnhancement can be instantiated."""
    module = AtrousEnhancement(dim=256)
    assert isinstance(module, AtrousEnhancement)
    assert module.dim == 256
    assert module.rates == [1, 6, 12, 18]
    assert module.reduction == 4


def test_atrous_creation_custom() -> None:
    """Verify AtrousEnhancement with custom params."""
    module = AtrousEnhancement(dim=128, rates=[1, 2, 3], reduction=2)
    assert isinstance(module, AtrousEnhancement)
    assert module.dim == 128
    assert module.rates == [1, 2, 3]
    assert module.reduction == 2


# ── Forward pass tests ───────────────────────────────────────────────────


def test_atrous_forward_shape() -> None:
    """Verify AtrousEnhancement preserves input shape."""
    module = AtrousEnhancement(dim=256)
    x = torch.randn(2, 256, 16, 16)
    out = module(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_atrous_gradients_flow() -> None:
    """Verify gradients flow through AtrousEnhancement."""
    module = AtrousEnhancement(dim=256)
    x = torch.randn(1, 256, 16, 16, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient is None"
    assert x.grad.abs().sum() > 0, "Input gradient is zero"


def test_atrous_different_rates() -> None:
    """Verify AtrousEnhancement works with different dilation configurations."""
    configs = [
        [1, 2, 3],
        [1, 6, 12, 18, 24],
        [1],
        [1, 12],
    ]
    x = torch.randn(2, 256, 16, 16)
    for rates in configs:
        module = AtrousEnhancement(dim=256, rates=rates)
        out = module(x)
        assert out.shape == x.shape, (
            f"rates={rates}: expected {x.shape}, got {out.shape}"
        )


# ── Experiment metadata tests ────────────────────────────────────────────


def test_experiment_info_dataclass() -> None:
    """Verify ExperimentInfo dataclass can be created."""
    info = ExperimentInfo(
        model_name="segformer_atrous",
        backbone="Hybrid-B0",
        num_classes=7,
        input_size=512,
        decoder_dim=256,
        total_params=3_080_000,
        encoder_params=2_800_000,
        decoder_params=280_000,
    )
    assert info.model_name == "segformer_atrous"
    assert info.total_params == 3_080_000
    assert info.encoder_params == 2_800_000


def test_build_experiment_info() -> None:
    """Verify build_experiment_info works with SegFormer."""
    model = SegFormer()
    info = build_experiment_info(model)
    assert isinstance(info, ExperimentInfo)
    assert info.total_params > 0
    assert info.encoder_params > 0
    assert info.decoder_params > 0
    assert info.total_params == info.encoder_params + info.decoder_params


def test_format_experiment_info() -> None:
    """Verify format_experiment_info produces a non-empty string."""
    model = SegFormer()
    info = build_experiment_info(model)
    formatted = format_experiment_info(info)
    assert isinstance(formatted, str)
    assert len(formatted) > 0
    assert "SegFormer" in formatted
    assert "Total" in formatted
    assert "Encoder" in formatted
    assert "Decoder" in formatted