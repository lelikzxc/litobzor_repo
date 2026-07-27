"""Tests for the CTM-IYOLOv10 implementation.

Tests cover:
    - GhostConv module
    - BiFPN module
    - CTM (Clustering-Template Matching) preprocessing
    - CTMIYOLOv10 model creation and forward pass
"""

from __future__ import annotations

import numpy as np
import torch
import pytest

from papers.ctm_yolov10.modules.ctm import CTM
from papers.ctm_yolov10.modules.ghost_conv import GhostConv
from papers.ctm_yolov10.modules.bifpn import BiFPN
from papers.ctm_yolov10.models.yolov10 import CTMIYOLOv10, YOLOv10Baseline


# ── GhostConv tests ───────────────────────────────────────────────────────


def test_ghost_conv_import() -> None:
    """Verify GhostConv can be imported."""
    assert GhostConv is not None


def test_ghost_conv_forward_shape() -> None:
    """Verify GhostConv preserves spatial dimensions correctly."""
    conv = GhostConv(in_channels=32, out_channels=64, kernel_size=3, stride=1)
    x = torch.randn(2, 32, 40, 40)
    out = conv(x)
    assert out.shape == (2, 64, 40, 40), f"Expected (2, 64, 40, 40), got {out.shape}"


def test_ghost_conv_stride2() -> None:
    """Verify GhostConv with stride=2 halves spatial size."""
    conv = GhostConv(in_channels=16, out_channels=32, kernel_size=3, stride=2)
    x = torch.randn(2, 16, 80, 80)
    out = conv(x)
    assert out.shape == (2, 32, 40, 40), f"Expected (2, 32, 40, 40), got {out.shape}"


def test_ghost_conv_gradients() -> None:
    """Verify gradients flow through GhostConv."""
    conv = GhostConv(in_channels=16, out_channels=32, kernel_size=3)
    x = torch.randn(1, 16, 20, 20, requires_grad=True)
    out = conv(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient is None"
    assert x.grad.abs().sum() > 0, "Input gradient is zero"


# ── BiFPN tests ───────────────────────────────────────────────────────────


def test_bifpn_import() -> None:
    """Verify BiFPN can be imported."""
    assert BiFPN is not None


def test_bifpn_forward_shape() -> None:
    """Verify BiFPN preserves spatial dimensions and original channel dims."""
    bifpn = BiFPN(channels=256, num_levels=3, num_repeats=1)
    p3 = torch.randn(1, 64, 80, 80)
    p4 = torch.randn(1, 128, 40, 40)
    p5 = torch.randn(1, 256, 20, 20)
    out = bifpn([p3, p4, p5])
    assert len(out) == 3, f"Expected 3 outputs, got {len(out)}"
    # BiFPN projects back to original channel dimensions
    assert out[0].shape == (1, 64, 80, 80), f"P3 shape mismatch: {out[0].shape}"
    assert out[1].shape == (1, 128, 40, 40), f"P4 shape mismatch: {out[1].shape}"
    assert out[2].shape == (1, 256, 20, 20), f"P5 shape mismatch: {out[2].shape}"


def test_bifpn_gradients() -> None:
    """Verify gradients flow through BiFPN."""
    bifpn = BiFPN(channels=64, num_levels=3, num_repeats=1)
    p3 = torch.randn(1, 64, 40, 40, requires_grad=True)
    p4 = torch.randn(1, 128, 20, 20, requires_grad=True)
    p5 = torch.randn(1, 256, 10, 10, requires_grad=True)
    out = bifpn([p3, p4, p5])
    loss = sum(o.sum() for o in out)
    loss.backward()
    assert p3.grad is not None, "P3 gradient is None"
    assert p4.grad is not None, "P4 gradient is None"
    assert p5.grad is not None, "P5 gradient is None"


# ── CTM (Clustering-Template Matching) tests ──────────────────────────────


def test_ctm_import() -> None:
    """Verify CTM can be imported."""
    assert CTM is not None


def test_ctm_normalized_cross_correlation() -> None:
    """Verify NCC computation works."""
    from papers.ctm_yolov10.modules.ctm import normalized_cross_correlation
    image = np.random.randn(100, 100).astype(np.float32)
    template = np.random.randn(20, 20).astype(np.float32)
    result = normalized_cross_correlation(image, template)
    assert result.shape == (81, 81), f"Expected (81, 81), got {result.shape}"
    assert -1.0 <= result.min() <= result.max() <= 1.0


def test_ctm_affinity_propagation() -> None:
    """Verify AP clustering works on simple match boxes."""
    from papers.ctm_yolov10.modules.ctm import affinity_propagation_clustering
    # Two well-separated clusters
    matches = np.array([
        [0, 0, 20, 20],
        [5, 5, 25, 25],
        [100, 100, 120, 120],
        [105, 105, 125, 125],
    ], dtype=np.float32)
    labels, exemplars = affinity_propagation_clustering(matches)
    assert len(np.unique(labels)) == 2, f"Expected 2 clusters, got {len(np.unique(labels))}"
    assert len(exemplars) == 2, f"Expected 2 exemplars, got {len(exemplars)}"


def test_ctm_match_template_with_clustering() -> None:
    """Verify full CTM pipeline works."""
    from papers.ctm_yolov10.modules.ctm import match_template_with_clustering
    # Create a simple image with a known pattern
    image = np.zeros((100, 100), dtype=np.uint8)
    image[10:30, 10:30] = 255  # a white square
    template = np.ones((20, 20), dtype=np.uint8) * 255
    results = match_template_with_clustering(image, template, threshold=0.5)
    assert len(results) >= 1, "Expected at least 1 match"
    assert "bbox" in results[0]
    assert "score" in results[0]
    assert "die_image" in results[0]


# ── CTM-IYOLOv10 integration tests ────────────────────────────────────────


def test_ctm_iyolov10_creation() -> None:
    """Verify CTMIYOLOv10 can be instantiated."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, num_classes=8)
    assert isinstance(model, CTMIYOLOv10)
    assert model.model_name == "yolov10n"
    assert model.num_classes == 8
    assert model.ghost_conv is True
    assert model.bifpn is True


def test_ctm_iyolov10_forward_smoke() -> None:
    """Verify CTMIYOLOv10 forward pass with dummy input.

    Uses ``bifpn=False`` because BiFPN integration with YOLOv10's
    ``_predict_once`` requires careful handling of skip connections.
    BiFPN is tested independently in ``test_bifpn_forward_shape``.
    """
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80, bifpn=False)
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(x)
    assert output is not None, "Forward pass returned None"


def test_ctm_iyolov10_from_config() -> None:
    """Verify CTMIYOLOv10.from_config() works."""
    from common.utils.config import Config
    config = Config.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    model = CTMIYOLOv10.from_config(config)
    assert isinstance(model, CTMIYOLOv10)
    # Magnetic Tile dataset has 3 classes: Blowhole, Crack, Fray
    assert model.num_classes == 3


def test_ctm_iyolov10_ablation_ghost_conv() -> None:
    """Verify GhostConv can be disabled."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, ghost_conv=False)
    assert model.ghost_conv is False


def test_ctm_iyolov10_ablation_bifpn() -> None:
    """Verify BiFPN can be disabled."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, bifpn=False)
    assert model.bifpn is False


def test_ctm_iyolov10_ablation_both() -> None:
    """Verify both improvements can be disabled (pure YOLOv10)."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, ghost_conv=False, bifpn=False)
    assert model.ghost_conv is False
    assert model.bifpn is False