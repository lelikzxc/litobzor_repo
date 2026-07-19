"""Tests for the Context Transformer Module (CTM) and CTM-YOLOv10 integration."""

from __future__ import annotations

import torch
import pytest

from papers.ctm_yolov10.modules.ctm import CTM, ContextAttention, ContextMLP
from papers.ctm_yolov10.models.yolov10 import CTMYOLOv10, YOLOv10Baseline


# ── CTM module tests ─────────────────────────────────────────────────────


def test_ctm_import() -> None:
    """Verify CTM, ContextAttention, and ContextMLP can be imported."""
    assert CTM is not None
    assert ContextAttention is not None
    assert ContextMLP is not None


def test_ctm_forward_shape() -> None:
    """Verify CTM preserves spatial dimensions.

    Input:  [2, 256, 20, 20]
    Output: [2, 256, 20, 20]
    """
    ctm = CTM(dim=256, num_heads=4)
    x = torch.randn(2, 256, 20, 20)
    out = ctm(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_ctm_gradients_flow() -> None:
    """Verify gradients flow through the CTM module."""
    ctm = CTM(dim=128, num_heads=4)
    x = torch.randn(1, 128, 10, 10, requires_grad=True)
    out = ctm(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient is None"
    assert x.grad.shape == x.shape, f"Gradient shape mismatch: {x.grad.shape}"
    assert x.grad.abs().sum() > 0, "Input gradient is zero"


# ── CTM-YOLOv10 integration tests ────────────────────────────────────────


def test_ctm_yolov10_creation() -> None:
    """Verify CTMYOLOv10 can be instantiated."""
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=8)
    assert isinstance(model, CTMYOLOv10)
    assert model.model_name == "yolov10n"
    assert model.num_classes == 8


def test_ctm_yolov10_has_more_params_than_baseline() -> None:
    """Verify CTMYOLOv10 has more parameters than YOLOv10Baseline."""
    baseline = YOLOv10Baseline(model_name="yolov10n", pretrained=False)
    ctm_model = CTMYOLOv10(model_name="yolov10n", pretrained=False)

    baseline_params = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    ctm_params = sum(p.numel() for p in ctm_model.parameters() if p.requires_grad)

    assert ctm_params > baseline_params, (
        f"CTM model ({ctm_params:,}) should have more parameters "
        f"than baseline ({baseline_params:,})"
    )


def test_ctm_yolov10_forward_smoke() -> None:
    """Verify CTMYOLOv10 forward pass with dummy input."""
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80)
    model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(x)

    assert output is not None, "Forward pass returned None"


def test_ctm_yolov10_from_config() -> None:
    """Verify CTMYOLOv10.from_config() works."""
    from common.utils.config import Config

    config = Config.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    model = CTMYOLOv10.from_config(config)
    assert isinstance(model, CTMYOLOv10)
    assert model.num_classes == 8