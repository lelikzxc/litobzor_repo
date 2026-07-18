"""Tests for Atrous Enhancement module and SegFormer + Atrous integration."""

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


# ── Integration tests ────────────────────────────────────────────────────


def test_segformer_atrous_creation() -> None:
    """Verify SegFormer with Atrous can be instantiated."""
    model = SegFormer(atrous_enabled=True)
    assert isinstance(model, SegFormer)
    assert model.atrous_enabled is True
    assert isinstance(model.atrous, AtrousEnhancement)


def test_segformer_atrous_forward_shape() -> None:
    """Verify SegFormer + Atrous produces correct output shape."""
    model = SegFormer(atrous_enabled=True, num_classes=8)
    model.eval()
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8, 512, 512), f"Expected (2, 8, 512, 512), got {out.shape}"


def test_segformer_atrous_gradients_flow() -> None:
    """Verify gradients flow through SegFormer + Atrous."""
    model = SegFormer(atrous_enabled=True, num_classes=8)
    x = torch.randn(1, 3, 512, 512, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient is None"
    assert x.grad.abs().sum() > 0, "Input gradient is zero"


def test_atrous_param_increase() -> None:
    """Verify SegFormer + Atrous has more params than baseline."""
    model_baseline = SegFormer(atrous_enabled=False)
    model_atrous = SegFormer(atrous_enabled=True)
    params_baseline = sum(p.numel() for p in model_baseline.parameters() if p.requires_grad)
    params_atrous = sum(p.numel() for p in model_atrous.parameters() if p.requires_grad)
    assert params_atrous > params_baseline, (
        f"Atrous ({params_atrous}) should have more params than baseline ({params_baseline})"
    )


def test_atrous_output_shape_equality() -> None:
    """Verify baseline and Atrous produce same output shape."""
    model_baseline = SegFormer(atrous_enabled=False, num_classes=8)
    model_atrous = SegFormer(atrous_enabled=True, num_classes=8)
    model_baseline.eval()
    model_atrous.eval()
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out_baseline = model_baseline(x)
        out_atrous = model_atrous(x)
    assert out_baseline.shape == out_atrous.shape, (
        f"Baseline {out_baseline.shape} != Atrous {out_atrous.shape}"
    )


def test_atrous_from_config() -> None:
    """Verify SegFormer.from_config() works with Atrous enabled."""
    from common.utils.config import Config

    config = Config.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    model = SegFormer.from_config(config)
    assert isinstance(model, SegFormer)
    assert model.atrous_enabled is True
    assert model.atrous_rates == [1, 6, 12, 18]
    assert model.atrous_reduction == 4
    assert isinstance(model.atrous, AtrousEnhancement)


def test_atrous_from_config_disabled() -> None:
    """Verify SegFormer.from_config() works with Atrous disabled."""
    from common.utils.config import Config

    config = Config.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    # Override to disable atrous
    config.raw["model"]["atrous"]["enabled"] = False
    model = SegFormer.from_config(config)
    assert isinstance(model, SegFormer)
    assert model.atrous_enabled is False
    assert not isinstance(model.atrous, AtrousEnhancement)


# ── Experiment metadata tests ────────────────────────────────────────────


def test_experiment_info_dataclass() -> None:
    """Verify ExperimentInfo dataclass can be created."""
    info = ExperimentInfo(
        model_name="segformer_atrous",
        backbone="MiT-B0",
        num_classes=8,
        input_size=512,
        decoder_dim=256,
        atrous_enabled=True,
        atrous_rates=[1, 6, 12, 18],
        atrous_reduction=4,
        total_params=3_770_568,
        backbone_params=3_500_000,
        decoder_params=250_000,
        atrous_params=20_568,
    )
    assert info.model_name == "segformer_atrous"
    assert info.total_params == 3_770_568
    assert info.atrous_params == 20_568


def test_build_experiment_info_baseline() -> None:
    """Verify build_experiment_info works with baseline SegFormer."""
    model = SegFormer(atrous_enabled=False)
    info = build_experiment_info(model)
    assert isinstance(info, ExperimentInfo)
    assert info.atrous_enabled is False
    assert info.atrous_rates is None
    assert info.atrous_reduction is None
    assert info.total_params > 0
    assert info.atrous_params == 0
    assert info.total_params == info.backbone_params + info.decoder_params


def test_build_experiment_info_atrous() -> None:
    """Verify build_experiment_info works with Atrous-enhanced SegFormer."""
    model = SegFormer(atrous_enabled=True)
    info = build_experiment_info(model)
    assert isinstance(info, ExperimentInfo)
    assert info.atrous_enabled is True
    assert info.atrous_rates == [1, 6, 12, 18]
    assert info.atrous_reduction == 4
    assert info.total_params > 0
    assert info.atrous_params > 0
    assert info.total_params == info.backbone_params + info.decoder_params + info.atrous_params


def test_format_experiment_info() -> None:
    """Verify format_experiment_info produces a non-empty string."""
    model = SegFormer(atrous_enabled=True)
    info = build_experiment_info(model)
    formatted = format_experiment_info(info)
    assert isinstance(formatted, str)
    assert len(formatted) > 0
    assert "SegFormer + Atrous" in formatted
    assert "Total" in formatted
    assert "Atrous" in formatted


# ── Ablation support tests ───────────────────────────────────────────────


def test_ablation_baseline_creation() -> None:
    """Verify baseline SegFormer (Atrous disabled) can be created."""
    model = SegFormer(atrous_enabled=False, num_classes=8)
    assert isinstance(model, SegFormer)
    assert model.atrous_enabled is False
    assert isinstance(model.atrous, torch.nn.Identity)


def test_ablation_atrous_creation() -> None:
    """Verify Atrous-enhanced SegFormer can be created."""
    model = SegFormer(atrous_enabled=True, num_classes=8)
    assert isinstance(model, SegFormer)
    assert model.atrous_enabled is True
    assert isinstance(model.atrous, AtrousEnhancement)


def test_ablation_custom_rates() -> None:
    """Verify SegFormer with custom atrous rates can be created."""
    model = SegFormer(atrous_enabled=True, atrous_rates=[1, 3, 6, 9], num_classes=8)
    assert isinstance(model, SegFormer)
    assert model.atrous_rates == [1, 3, 6, 9]


def test_ablation_param_difference() -> None:
    """Verify Atrous model has more params than baseline."""
    model_baseline = SegFormer(atrous_enabled=False)
    model_atrous = SegFormer(atrous_enabled=True)
    p_baseline = count_params(model_baseline)
    p_atrous = count_params(model_atrous)
    assert p_atrous > p_baseline, (
        f"Atrous ({p_atrous}) should have more params than baseline ({p_baseline})"
    )


def test_ablation_output_shape_equality() -> None:
    """Verify baseline and Atrous produce identical output shapes."""
    model_baseline = SegFormer(atrous_enabled=False, num_classes=8)
    model_atrous = SegFormer(atrous_enabled=True, num_classes=8)
    model_baseline.eval()
    model_atrous.eval()
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out_baseline = model_baseline(x)
        out_atrous = model_atrous(x)
    assert out_baseline.shape == out_atrous.shape
    assert out_baseline.shape == (2, 8, 512, 512)


def test_ablation_disabled_atrous_behaviour() -> None:
    """Verify disabled Atrous produces logits (not probabilities)."""
    model = SegFormer(atrous_enabled=False, num_classes=8)
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 8, 512, 512)
    assert (out < 0).any() or (out > 1).any(), (
        "Output appears to be probabilities, not logits"
    )


def test_ablation_gradients_preserved() -> None:
    """Verify gradients flow through both baseline and Atrous models."""
    for atrous_enabled in [False, True]:
        model = SegFormer(atrous_enabled=atrous_enabled, num_classes=8)
        x = torch.randn(1, 3, 512, 512, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, f"Gradient is None (atrous={atrous_enabled})"
        assert x.grad.abs().sum() > 0, f"Gradient is zero (atrous={atrous_enabled})"