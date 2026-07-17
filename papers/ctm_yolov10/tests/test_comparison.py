"""Comparison tests between YOLOv10Baseline and CTMYOLOv10.

Verifies that:
- Both models can be created with identical configurations.
- CTM model has strictly more parameters.
- Both models accept identical inputs.
- Output shapes are compatible.
- Config-driven ``ctm_enabled`` switching works correctly.
"""

from __future__ import annotations

import torch
import pytest

from papers.ctm_yolov10.models.yolov10 import CTMYOLOv10, YOLOv10Baseline
from papers.ctm_yolov10.utils.experiment import (
    ExperimentInfo,
    build_experiment_info,
    count_params,
    format_experiment_info,
)


def test_baseline_model_creation() -> None:
    """Verify YOLOv10Baseline can be created."""
    model = YOLOv10Baseline(model_name="yolov10n", pretrained=False, num_classes=80)
    assert isinstance(model, YOLOv10Baseline)
    assert count_params(model) > 0


def test_ctm_model_creation() -> None:
    """Verify CTMYOLOv10 can be created with CTM enabled."""
    model = CTMYOLOv10(
        model_name="yolov10n",
        pretrained=False,
        num_classes=80,
        ctm_enabled=True,
    )
    assert isinstance(model, CTMYOLOv10)
    assert model.ctm_enabled is True
    assert model.ctm is not None


def test_ctm_parameter_increase() -> None:
    """Verify CTM model has strictly more parameters than baseline."""
    baseline = YOLOv10Baseline(model_name="yolov10n", pretrained=False, num_classes=80)
    ctm_model = CTMYOLOv10(
        model_name="yolov10n",
        pretrained=False,
        num_classes=80,
        ctm_enabled=True,
    )

    baseline_params = count_params(baseline)
    ctm_params = count_params(ctm_model)

    assert ctm_params > baseline_params, (
        f"CTM model ({ctm_params:,}) should have more parameters "
        f"than baseline ({baseline_params:,})"
    )


def test_identical_input_compatibility() -> None:
    """Verify both models accept the same input tensor."""
    baseline = YOLOv10Baseline(model_name="yolov10n", pretrained=False, num_classes=80)
    ctm_model = CTMYOLOv10(
        model_name="yolov10n",
        pretrained=False,
        num_classes=80,
        ctm_enabled=True,
    )
    baseline.eval()
    ctm_model.eval()

    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        baseline_out = baseline(x)
        ctm_out = ctm_model(x)

    assert baseline_out is not None
    assert ctm_out is not None


def test_output_shape_equality() -> None:
    """Verify both models produce outputs with the same structure.

    Both models should return the same output type and shape
    (tuple of [detections, loss_dict]).
    """
    baseline = YOLOv10Baseline(model_name="yolov10n", pretrained=False, num_classes=80)
    ctm_model = CTMYOLOv10(
        model_name="yolov10n",
        pretrained=False,
        num_classes=80,
        ctm_enabled=True,
    )
    baseline.eval()
    ctm_model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        baseline_out = baseline(x)
        ctm_out = ctm_model(x)

    # Both should return the same type
    assert type(baseline_out) is type(ctm_out), (
        f"Output types differ: {type(baseline_out)} vs {type(ctm_out)}"
    )

    # If tuple, compare lengths and tensor shapes
    if isinstance(baseline_out, tuple) and isinstance(ctm_out, tuple):
        assert len(baseline_out) == len(ctm_out)
        for i, (b, c) in enumerate(zip(baseline_out, ctm_out)):
            if isinstance(b, torch.Tensor) and isinstance(c, torch.Tensor):
                assert b.shape == c.shape, (
                    f"Output [{i}] shape mismatch: {b.shape} vs {c.shape}"
                )


def test_config_switching() -> None:
    """Verify ``ctm_enabled`` flag correctly switches CTM on/off.

    - ``ctm_enabled=True``: CTM module is created, params > baseline.
    - ``ctm_enabled=False``: CTM module is ``None``, params == baseline.
    """
    baseline = YOLOv10Baseline(model_name="yolov10n", pretrained=False, num_classes=80)
    ctm_on = CTMYOLOv10(
        model_name="yolov10n",
        pretrained=False,
        num_classes=80,
        ctm_enabled=True,
    )
    ctm_off = CTMYOLOv10(
        model_name="yolov10n",
        pretrained=False,
        num_classes=80,
        ctm_enabled=False,
    )

    baseline_params = count_params(baseline)
    ctm_on_params = count_params(ctm_on)
    ctm_off_params = count_params(ctm_off)

    # CTM enabled → more params than baseline
    assert ctm_on_params > baseline_params
    # CTM disabled → same params as baseline
    assert ctm_off_params == baseline_params, (
        f"CTM disabled ({ctm_off_params:,}) should match "
        f"baseline ({baseline_params:,})"
    )
    # CTM module is None when disabled
    assert ctm_off.ctm is None
    assert ctm_on.ctm is not None


def test_experiment_info_baseline() -> None:
    """Verify ``build_experiment_info`` works for baseline model."""
    model = YOLOv10Baseline(model_name="yolov10n", pretrained=False, num_classes=80)
    info = build_experiment_info(model)

    assert isinstance(info, ExperimentInfo)
    assert info.model_name == "yolov10n"
    assert info.ctm_enabled is False
    assert info.ctm_params == 0
    assert info.total_params > 0
    assert info.backbone_params == info.total_params

    # Should produce a non-empty formatted string
    formatted = format_experiment_info(info)
    assert len(formatted) > 0
    assert "CTM enabled:        False" in formatted


def test_experiment_info_ctm() -> None:
    """Verify ``build_experiment_info`` works for CTM model."""
    model = CTMYOLOv10(
        model_name="yolov10n",
        pretrained=False,
        num_classes=80,
        ctm_enabled=True,
        dim=256,
        num_heads=4,
        mlp_ratio=4.0,
    )
    info = build_experiment_info(model)

    assert isinstance(info, ExperimentInfo)
    assert info.model_name == "yolov10n"
    assert info.ctm_enabled is True
    assert info.ctm_params > 0
    assert info.total_params > info.backbone_params
    assert info.ctm_dim == 256
    assert info.ctm_num_heads == 4

    formatted = format_experiment_info(info)
    assert len(formatted) > 0
    assert "CTM enabled:        True" in formatted
    assert "CTM dim:            256" in formatted