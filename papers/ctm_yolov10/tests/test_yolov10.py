"""Tests for YOLOv10 baseline model and CTM-IYOLOv10 integration."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from papers.ctm_yolov10.models.yolov10 import YOLOv10Baseline
from papers.ctm_yolov10.modules.ctm import CTM

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── YOLOv10 baseline tests ──────────────────────────────────────────────


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

    assert output is not None, "Forward pass returned None"


def test_forward_without_download() -> None:
    """Verify forward works without downloading weights."""
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


# ── CTM (Clustering-Template Matching) tests ────────────────────────────


def test_ctm_import() -> None:
    """Verify CTM can be imported from modules package."""
    from papers.ctm_yolov10.modules import CTM  # noqa: F811

    assert CTM is not None


def test_ctm_creation() -> None:
    """Verify CTM can be instantiated with a template."""
    import numpy as np
    template = np.ones((20, 20), dtype=np.uint8) * 255
    ctm = CTM(template=template, threshold=0.7)
    assert isinstance(ctm, CTM)
    assert ctm.threshold == 0.7


def test_ctm_call_no_template_raises() -> None:
    """Verify CTM raises ValueError when no template is set."""
    ctm = CTM()
    import numpy as np
    with pytest.raises(ValueError, match="template not set"):
        ctm(np.zeros((100, 100), dtype=np.uint8))


# ── Documentation tests ─────────────────────────────────────────────────


def test_yolov10_architecture_doc_exists() -> None:
    """Verify YOLOv10 architecture documentation exists."""
    doc_path = (
        REPO_ROOT / "papers" / "ctm_yolov10" / "docs" / "yolov10_architecture.md"
    )
    assert doc_path.is_file(), f"Missing: {doc_path}"


def test_ctm_integration_plan_doc_exists() -> None:
    """Verify CTM integration plan documentation exists."""
    doc_path = (
        REPO_ROOT / "papers" / "ctm_yolov10" / "docs" / "ctm_integration_plan.md"
    )
    assert doc_path.is_file(), f"Missing: {doc_path}"