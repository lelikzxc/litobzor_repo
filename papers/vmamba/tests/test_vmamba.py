"""Tests for FCS-VMamba backbone model and FCS modules (FA, SFS, CLCA)."""

from __future__ import annotations

import torch
import pytest

from papers.vmamba.models.vmamba import FCSVMamba
from papers.vmamba.modules import (
    PatchEmbed2D,
    PatchMerging,
    SS2D,
    VSSBlock,
    FCSVSSBlock,
    FrequencyAttention,
    SaliencySuppression,
    CrossLayerChannelAttention,
)
from papers.vmamba.utils.experiment import (
    ExperimentInfo,
    build_experiment_info,
    count_params,
    format_experiment_info,
)


# ── Module import tests ──────────────────────────────────────────────────


def test_import() -> None:
    """Verify all modules can be imported."""
    assert FCSVMamba is not None
    assert PatchEmbed2D is not None
    assert PatchMerging is not None
    assert SS2D is not None
    assert VSSBlock is not None
    assert FCSVSSBlock is not None
    assert FrequencyAttention is not None
    assert SaliencySuppression is not None
    assert CrossLayerChannelAttention is not None


# ── Model creation tests ─────────────────────────────────────────────────


def test_model_creation() -> None:
    """Verify FCSVMamba can be instantiated with default params."""
    model = FCSVMamba()
    assert isinstance(model, FCSVMamba)
    assert model.num_classes == 8
    assert model.embed_dim == 96
    assert len(model.stages) == 4
    assert len(model.mergings) == 3
    assert model.fa_enabled is True
    assert model.sfs_enabled is True
    assert model.clca_enabled is True


def test_model_creation_custom() -> None:
    """Verify FCSVMamba can be instantiated with custom params."""
    model = FCSVMamba(
        in_channels=3,
        image_size=224,
        embed_dim=64,
        depths=[1, 1, 3, 1],
        num_heads=[2, 4, 8, 16],
        ssm_ratio=1.0,
        mlp_ratio=2.0,
        drop_path_rate=0.1,
        num_classes=10,
        fa_enabled=False,
        sfs_enabled=False,
        clca_enabled=False,
    )
    assert isinstance(model, FCSVMamba)
    assert model.num_classes == 10
    assert model.embed_dim == 64
    assert model.depths == [1, 1, 3, 1]
    assert model.fa_enabled is False
    assert model.sfs_enabled is False
    assert model.clca_enabled is False


# ── Forward pass tests ───────────────────────────────────────────────────


def test_forward_shape() -> None:
    """Verify forward pass produces correct output shape.

    Input:  [2, 3, 224, 224]
    Output: [2, 8] (logits, no Softmax)
    """
    model = FCSVMamba(num_classes=8)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out is not None, "Forward pass returned None"
    assert out.shape == (2, 8), f"Expected (2, 8), got {out.shape}"


def test_forward_logits_only() -> None:
    """Verify model returns logits (not probabilities)."""
    model = FCSVMamba(num_classes=8)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 8)
    # Values should not be softmaxed (can be negative or > 1)
    assert (out < 0).any() or (out > 1).any(), (
        "Output appears to be probabilities, not logits"
    )


def test_gradients_flow() -> None:
    """Verify gradients flow through the entire model."""
    model = FCSVMamba(num_classes=8)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient is None"
    assert x.grad.abs().sum() > 0, "Input gradient is zero"


def test_parameter_count() -> None:
    """Verify model has trainable parameters."""
    model = FCSVMamba()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0, "Model has zero trainable parameters"


# ── Config-driven creation ───────────────────────────────────────────────


def test_from_config() -> None:
    """Verify FCSVMamba.from_config() works."""
    from common.utils.config import Config

    config = Config.from_yaml("papers/vmamba/configs/config.yaml")
    model = FCSVMamba.from_config(config)
    assert isinstance(model, FCSVMamba)
    assert model.num_classes == 8
    assert model.embed_dim == 96
    assert model.depths == [2, 2, 6, 2]
    assert model.num_heads == [3, 6, 12, 24]
    assert model.ssm_ratio == 2.0
    assert model.mlp_ratio == 4.0
    assert model.fa_enabled is True
    assert model.sfs_enabled is True
    assert model.clca_enabled is True


# ── FA (Frequency Attention) tests ───────────────────────────────────────


def test_fa_forward_shape() -> None:
    """Verify FA preserves input shape."""
    fa = FrequencyAttention(dim=96, reduction=16)
    x = torch.randn(2, 96, 56, 56)
    out = fa(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_fa_gradients_flow() -> None:
    """Verify gradients flow through FA."""
    fa = FrequencyAttention(dim=96, reduction=16)
    x = torch.randn(1, 96, 56, 56, requires_grad=True)
    out = fa(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient is None"
    assert x.grad.abs().sum() > 0, "Input gradient is zero"


# ── SFS (Saliency Feature Suppression) tests ─────────────────────────────


def test_sfs_forward_shape() -> None:
    """Verify SFS preserves input shape."""
    sfs = SaliencySuppression(dim=96, reduction=4)
    x = torch.randn(2, 96, 56, 56)
    out = sfs(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_sfs_gradients_flow() -> None:
    """Verify gradients flow through SFS."""
    sfs = SaliencySuppression(dim=96, reduction=4)
    x = torch.randn(1, 96, 56, 56, requires_grad=True)
    out = sfs(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient is None"
    assert x.grad.abs().sum() > 0, "Input gradient is zero"


# ── CLCA (Cross-Layer Channel Attention) tests ───────────────────────────


def test_clca_forward_shape() -> None:
    """Verify CLCA preserves target shape."""
    clca = CrossLayerChannelAttention(guide_dim=96, target_dim=768, reduction=16)
    guide = torch.randn(2, 96, 56, 56)
    target = torch.randn(2, 768, 7, 7)
    out = clca(guide, target)
    assert out.shape == target.shape, f"Expected {target.shape}, got {out.shape}"


def test_clca_gradients_flow() -> None:
    """Verify gradients flow through CLCA."""
    clca = CrossLayerChannelAttention(guide_dim=96, target_dim=768, reduction=16)
    guide = torch.randn(2, 96, 56, 56, requires_grad=True)
    target = torch.randn(2, 768, 7, 7, requires_grad=True)
    out = clca(guide, target)
    loss = out.sum()
    loss.backward()
    assert guide.grad is not None, "Guide gradient is None"
    assert guide.grad.abs().sum() > 0, "Guide gradient is zero"
    assert target.grad is not None, "Target gradient is None"
    assert target.grad.abs().sum() > 0, "Target gradient is zero"


# ── Experiment info tests ────────────────────────────────────────────────


def test_experiment_info_dataclass() -> None:
    """Verify ExperimentInfo dataclass can be created."""
    info = ExperimentInfo(
        model_name="fcs_vmamba",
        backbone="vmamba_tiny",
        num_classes=8,
        input_size=224,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        fa_enabled=True,
        sfs_enabled=True,
        clca_enabled=True,
        fa_reduction=16,
        sfs_reduction=4,
        clca_reduction=16,
        total_params=1_000_000,
        backbone_params=900_000,
        fcs_params=100_000,
    )
    assert info.model_name == "fcs_vmamba"
    assert info.total_params == 1_000_000
    assert info.fcs_params == 100_000


def test_build_experiment_info_full() -> None:
    """Verify build_experiment_info works with full FCS-VMamba."""
    model = FCSVMamba(fa_enabled=True, sfs_enabled=True, clca_enabled=True)
    info = build_experiment_info(model)
    assert isinstance(info, ExperimentInfo)
    assert info.fa_enabled is True
    assert info.sfs_enabled is True
    assert info.clca_enabled is True
    assert info.fa_reduction == 16
    assert info.sfs_reduction == 4
    assert info.clca_reduction == 16
    assert info.total_params > 0
    assert info.fcs_params > 0
    assert info.backbone_params > 0
    assert info.total_params == info.backbone_params + info.fcs_params


def test_build_experiment_info_baseline() -> None:
    """Verify build_experiment_info works with baseline VMamba."""
    model = FCSVMamba(fa_enabled=False, sfs_enabled=False, clca_enabled=False)
    info = build_experiment_info(model)
    assert isinstance(info, ExperimentInfo)
    assert info.fa_enabled is False
    assert info.sfs_enabled is False
    assert info.clca_enabled is False
    assert info.fa_reduction is None
    assert info.sfs_reduction is None
    assert info.clca_reduction is None
    assert info.total_params > 0
    assert info.fcs_params == 0
    assert info.total_params == info.backbone_params


def test_format_experiment_info() -> None:
    """Verify format_experiment_info produces a non-empty string."""
    model = FCSVMamba(fa_enabled=True, sfs_enabled=True, clca_enabled=True)
    info = build_experiment_info(model)
    formatted = format_experiment_info(info)
    assert isinstance(formatted, str)
    assert len(formatted) > 0
    assert "FCS-VMamba Research Validation" in formatted
    assert "Total" in formatted
    assert "Backbone" in formatted
    assert "FCS modules" in formatted


# ── Ablation support tests ───────────────────────────────────────────────


def test_ablation_baseline_creation() -> None:
    """Verify baseline VMamba (all FCS disabled) can be created."""
    model = FCSVMamba(
        fa_enabled=False, sfs_enabled=False, clca_enabled=False, num_classes=8,
    )
    assert isinstance(model, FCSVMamba)
    assert model.fa_enabled is False
    assert model.sfs_enabled is False
    assert model.clca_enabled is False


def test_ablation_full_fcs_creation() -> None:
    """Verify full FCS-VMamba (all FCS enabled) can be created."""
    model = FCSVMamba(
        fa_enabled=True, sfs_enabled=True, clca_enabled=True, num_classes=8,
    )
    assert isinstance(model, FCSVMamba)
    assert model.fa_enabled is True
    assert model.sfs_enabled is True
    assert model.clca_enabled is True


def test_ablation_param_increase() -> None:
    """Verify full FCS-VMamba has more params than baseline."""
    model_baseline = FCSVMamba(
        fa_enabled=False, sfs_enabled=False, clca_enabled=False,
    )
    model_full = FCSVMamba(
        fa_enabled=True, sfs_enabled=True, clca_enabled=True,
    )
    params_baseline = count_params(model_baseline)
    params_full = count_params(model_full)
    assert params_full > params_baseline, (
        f"Full ({params_full}) should have more params than baseline ({params_baseline})"
    )


def test_ablation_disabled_modules_output_shape() -> None:
    """Verify baseline and full model produce same output shape."""
    model_baseline = FCSVMamba(
        fa_enabled=False, sfs_enabled=False, clca_enabled=False, num_classes=8,
    )
    model_full = FCSVMamba(
        fa_enabled=True, sfs_enabled=True, clca_enabled=True, num_classes=8,
    )
    model_baseline.eval()
    model_full.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_baseline = model_baseline(x)
        out_full = model_full(x)
    assert out_baseline.shape == (2, 8), f"Baseline output shape: {out_baseline.shape}"
    assert out_full.shape == (2, 8), f"Full output shape: {out_full.shape}"


def test_ablation_partial_enabled() -> None:
    """Verify partial ablation configurations work."""
    configs = [
        ("FA only", True, False, False),
        ("SFS only", False, True, False),
        ("CLCA only", False, False, True),
        ("FA+SFS", True, True, False),
        ("FA+CLCA", True, False, True),
        ("SFS+CLCA", False, True, True),
    ]
    x = torch.randn(1, 3, 224, 224)
    for name, fa, sfs, clca in configs:
        model = FCSVMamba(
            fa_enabled=fa, sfs_enabled=sfs, clca_enabled=clca, num_classes=8,
        )
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 8), (
            f"{name}: expected (1, 8), got {out.shape}"
        )


# ── Integration tests ────────────────────────────────────────────────────


def test_fcs_vs_backbone_param_increase() -> None:
    """Verify FCS-VMamba has more params than backbone-only version."""
    model_full = FCSVMamba(
        fa_enabled=True, sfs_enabled=True, clca_enabled=True,
    )
    model_backbone = FCSVMamba(
        fa_enabled=False, sfs_enabled=False, clca_enabled=False,
    )
    params_full = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    params_backbone = sum(p.numel() for p in model_backbone.parameters() if p.requires_grad)
    assert params_full > params_backbone, (
        f"Full model ({params_full}) should have more params than backbone ({params_backbone})"
    )


def test_fcs_forward_with_disabled_modules() -> None:
    """Verify forward pass works with all FCS modules disabled."""
    model = FCSVMamba(
        fa_enabled=False, sfs_enabled=False, clca_enabled=False, num_classes=8,
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8), f"Expected (2, 8), got {out.shape}"


# ── FCSVSSBlock execution order tests ────────────────────────────────────


def test_fcsvssblock_has_fa_and_sfs_before_mlp() -> None:
    """Verify FA and SFS are called *before* the MLP in FCSVSSBlock.

    Uses ``register_forward_hook`` to trace the call order and confirm
    that FA and SFS execute between the SS2D residual and the MLP.
    """
    block = FCSVSSBlock(dim=96, fa_enabled=True, sfs_enabled=True)

    call_order: list[str] = []

    def make_hook(name: str):
        def hook(_module, _input, _output):
            call_order.append(name)
        return hook

    # Register hooks on the sub-modules in execution order
    handles = [
        block.op.register_forward_hook(make_hook("ss2d")),
        block.fa.register_forward_hook(make_hook("fa")),
        block.sfs.register_forward_hook(make_hook("sfs")),
        block.mlp.register_forward_hook(make_hook("mlp")),
    ]

    x = torch.randn(1, 96, 56, 56)
    _ = block(x)

    for h in handles:
        h.remove()

    # Verify the relative order: SS2D → FA → SFS → MLP
    ss2d_idx = call_order.index("ss2d")
    fa_idx = call_order.index("fa")
    sfs_idx = call_order.index("sfs")
    mlp_idx = call_order.index("mlp")

    assert ss2d_idx < fa_idx, f"SS2D ({ss2d_idx}) should execute before FA ({fa_idx})"
    assert fa_idx < sfs_idx, f"FA ({fa_idx}) should execute before SFS ({sfs_idx})"
    assert sfs_idx < mlp_idx, f"SFS ({sfs_idx}) should execute before MLP ({mlp_idx})"


def test_fcsvssblock_forward_shape() -> None:
    """Verify FCSVSSBlock preserves input shape."""
    block = FCSVSSBlock(dim=96, fa_enabled=True, sfs_enabled=True)
    x = torch.randn(2, 96, 56, 56)
    out = block(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_fcsvssblock_gradients_flow() -> None:
    """Verify gradients flow through FCSVSSBlock."""
    block = FCSVSSBlock(dim=96, fa_enabled=True, sfs_enabled=True)
    x = torch.randn(1, 96, 56, 56, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient is None"
    assert x.grad.abs().sum() > 0, "Input gradient is zero"


def test_fcsvssblock_parameter_count() -> None:
    """Verify FCSVSSBlock has trainable parameters."""
    block = FCSVSSBlock(dim=96, fa_enabled=True, sfs_enabled=True)
    num_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    assert num_params > 0, "Block has zero trainable parameters"


def test_fcsvssblock_disabled_fa_sfs() -> None:
    """Verify FCSVSSBlock forward works with FA and SFS disabled."""
    block = FCSVSSBlock(dim=96, fa_enabled=False, sfs_enabled=False)
    x = torch.randn(2, 96, 56, 56)
    out = block(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"