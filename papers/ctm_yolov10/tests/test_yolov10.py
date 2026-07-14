"""Tests for YOLOv10 baseline model."""

from __future__ import annotations

import torch
import pytest

from papers.ctm_yolov10.models.yolov10 import YOLOv10Baseline


def test_model_import() -> None:
    """Verify YOLOv10Baseline can be imported."""
    from papers.ctm_yolov10.models import YOLOv10Baseline  # noqa: F811

    assert YOLOv10Baseline is not None


def test_model_creation() -> None:
    """Verify model can be instantiated without pretrained weights."""
    model = YOLOv10Baseline(model_name="yolov10n", pretrained=False)
    assert isinstance(model, YOLOv10Baseline)


def test_model_has_parameters() -> None:
    """Verify model has trainable parameters."""
    model = YOLOv10Baseline(model_name="yolov10n", pretrained=False)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0, "Model has zero trainable parameters"


@pytest.mark.slow
def test_forward_smoke() -> None:
    """Verify forward pass with dummy input."""
    model = YOLOv10Baseline(model_name="yolov10n", pretrained=False)
    model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(x)

    # Output should be a tuple or dict (not None)
    assert output is not None, "Forward pass returned None"


def test_forward_without_download() -> None:
    """Verify forward works without downloading weights (mock test).

    This test creates the model from config only (no .pt download).
    """
    model = YOLOv10Baseline(model_name="yolov10n", pretrained=False)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)

    assert output is not None


def test_model_name_configurability() -> None:
    """Verify different model variants can be created."""
    for variant in ["yolov10n", "yolov10s"]:
        model = YOLOv10Baseline(model_name=variant, pretrained=False)
        assert isinstance(model, YOLOv10Baseline)
        assert model.model_name == variant