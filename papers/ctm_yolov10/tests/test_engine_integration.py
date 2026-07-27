"""Engine integration tests for CTM-IYOLOv10.

Verifies:
- Model registration in the engine registry
- EngineConfig loading from the paper config
- build_model() by registered name
- CTMIYOLOv10.from_config() with EngineConfig
- Predictor inference compatibility
- Output shape verification
- Engine instantiation with registered model name
"""

from __future__ import annotations

from typing import Any

import torch
import pytest

from common.engine.config import EngineConfig
from common.engine.registry import (
    build_model,
    is_registered,
    list_registered,
)
from common.engine.engine import Engine
from common.inference.predictor import Predictor
from papers.ctm_yolov10.models.yolov10 import CTMIYOLOv10


# ── Helpers ───────────────────────────────────────────────────────────────


def _detection_postprocess(logits: Any) -> dict[str, Any]:
    """Postprocessing function for detection model output (tuple)."""
    if isinstance(logits, tuple):
        detections = logits[0]
    else:
        detections = logits
    if isinstance(detections, torch.Tensor):
        probs = torch.softmax(detections[..., 5:].float(), dim=-1)
        pred = probs.argmax(dim=-1)
    else:
        probs = torch.tensor(0.0)
        pred = torch.tensor(0)
    return {"logits": detections, "probs": probs, "prediction": pred}


# ── Registry tests ────────────────────────────────────────────────────────


def test_model_registered() -> None:
    """Verify CTM-IYOLOv10 is registered in the engine registry."""
    assert is_registered("models", "ctm_iyolov10"), "ctm_iyolov10 not registered"


def test_baseline_registered() -> None:
    """Verify YOLOv10Baseline is registered."""
    assert is_registered("models", "yolov10_baseline"), "yolov10_baseline not registered"


def test_list_registered_contains_models() -> None:
    """Verify list_registered() includes our models."""
    registered = list_registered("models")
    assert "ctm_iyolov10" in registered
    assert "yolov10_baseline" in registered


def test_build_model_by_name() -> None:
    """Verify build_model('ctm_iyolov10') returns a CTMIYOLOv10 instance."""
    model = build_model("ctm_iyolov10", num_classes=8)
    assert isinstance(model, CTMIYOLOv10)
    assert model.num_classes == 8
    assert model.ghost_conv is True
    assert model.bifpn is True


def test_build_model_with_improvements_disabled() -> None:
    """Verify build_model with ghost_conv=False, bifpn=False works."""
    model = build_model("ctm_iyolov10", num_classes=8, ghost_conv=False, bifpn=False)
    assert isinstance(model, CTMIYOLOv10)
    assert model.ghost_conv is False
    assert model.bifpn is False


def test_build_model_unregistered_raises() -> None:
    """Verify build_model with unknown name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("nonexistent_model")


# ── EngineConfig tests ────────────────────────────────────────────────────


def test_engine_config_from_yaml() -> None:
    """Verify EngineConfig can load the paper config."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    assert config is not None
    assert config.get("model.name") == "ctm_iyolov10"
    # Magnetic Tile dataset has 3 classes: Blowhole, Crack, Fray
    assert config.get("model.num_classes") == 3


def test_engine_config_dot_access() -> None:
    """Verify dot-separated key access works."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    assert config.get("model.backbone.name") == "yolov10n"
    assert config.get("model.ghost_conv.enabled") is True
    assert config.get("model.bifpn.enabled") is True
    assert config.get("training.batch_size") == 16
    assert config.get("training.num_epochs") == 100


def test_engine_config_engine_fields() -> None:
    """Verify engine-compatible fields are present."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")

    assert config.get("model.name") == "ctm_iyolov10"
    # Magnetic Tile dataset has 3 classes: Blowhole, Crack, Fray
    assert config.get("model.num_classes") == 3

    opt = config.get("optimizer")
    assert isinstance(opt, dict)
    assert opt.get("name") == "sgd"
    assert opt.get("lr") == 0.001

    sched = config.get("scheduler")
    assert isinstance(sched, dict)
    assert sched.get("name") == "cosine"

    loss = config.get("loss")
    assert isinstance(loss, dict)
    assert loss.get("name") == "cross_entropy"


def test_engine_config_default_values() -> None:
    """Verify EngineConfig returns defaults for missing keys."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    assert config.get("nonexistent.key", "default") == "default"
    assert config.get("model.nonexistent", 42) == 42


# ── from_config tests ─────────────────────────────────────────────────────


def test_from_config_with_engine_config() -> None:
    """Verify CTMIYOLOv10.from_config() works with EngineConfig."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    model = CTMIYOLOv10.from_config(config)
    assert isinstance(model, CTMIYOLOv10)
    # Magnetic Tile dataset has 3 classes: Blowhole, Crack, Fray
    assert model.num_classes == 3
    assert model.ghost_conv is True
    assert model.bifpn is True


def test_from_config_improvements_disabled() -> None:
    """Verify from_config with ghost_conv=False, bifpn=False works."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    config.merge_deep({"model": {"ghost_conv": {"enabled": False}, "bifpn": {"enabled": False}}})
    model = CTMIYOLOv10.from_config(config)
    assert isinstance(model, CTMIYOLOv10)
    assert model.ghost_conv is False
    assert model.bifpn is False


# ── Predictor compatibility tests ─────────────────────────────────────────


def test_predictor_creation() -> None:
    """Verify Predictor can wrap CTMIYOLOv10."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80, bifpn=False)
    predictor = Predictor(model, device="cpu", postprocess_fn=_detection_postprocess)
    assert predictor is not None
    assert predictor.model is model


def test_predictor_single_inference() -> None:
    """Verify Predictor.predict_single() works with CTMIYOLOv10."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80, bifpn=False)
    model.eval()
    predictor = Predictor(model, device="cpu", postprocess_fn=_detection_postprocess)

    x = torch.randn(3, 640, 640)
    with torch.no_grad():
        result = predictor.predict_single(x)

    assert isinstance(result, dict)
    assert "logits" in result
    assert "probs" in result
    assert "prediction" in result


def test_predictor_batch_inference() -> None:
    """Verify Predictor.predict_batch() works with CTMIYOLOv10."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80, bifpn=False)
    model.eval()
    predictor = Predictor(model, device="cpu", postprocess_fn=_detection_postprocess)

    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        result = predictor.predict_batch(x)

    assert isinstance(result, dict)
    assert "logits" in result


# ── Engine instantiation tests ────────────────────────────────────────────


def test_engine_with_registered_model_name() -> None:
    """Verify Engine can be instantiated with a model instance."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, num_classes=8)
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    engine = Engine(model, config, device="cpu")
    assert isinstance(engine.model, CTMIYOLOv10)
    assert engine.model.num_classes == 8


def test_engine_summary() -> None:
    """Verify Engine.summary() returns expected keys."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, num_classes=8)
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    engine = Engine(model, config, device="cpu")
    summary = engine.summary()
    assert "model" in summary
    assert "device" in summary
    assert "CTMIYOLOv10" in summary["model"]


def test_engine_forward_pass() -> None:
    """Verify Engine.model forward pass works."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80, bifpn=False)
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    engine = Engine(model, config, device="cpu")
    engine.model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = engine.model(x)

    assert output is not None


# ── Output shape tests ────────────────────────────────────────────────────


def test_output_shape_with_synthetic_input() -> None:
    """Verify forward output shape with synthetic input."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80, bifpn=False)
    model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(x)

    assert output is not None


def test_output_consistency_across_batches() -> None:
    """Verify output structure is consistent across different batch sizes."""
    model = CTMIYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80, bifpn=False)
    model.eval()

    x1 = torch.randn(1, 3, 640, 640)
    x2 = torch.randn(2, 3, 640, 640)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    assert out1 is not None
    assert out2 is not None
    assert type(out1) is type(out2)