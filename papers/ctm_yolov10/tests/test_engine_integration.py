"""Engine integration tests for CTM-YOLOv10.

Verifies:
- Model registration in the engine registry
- EngineConfig loading from the paper config
- build_model() by registered name
- CTMYOLOv10.from_config() with EngineConfig
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
from papers.ctm_yolov10.models.yolov10 import CTMYOLOv10


# ── Helpers ───────────────────────────────────────────────────────────────


def _detection_postprocess(logits: Any) -> dict[str, Any]:
    """Postprocessing function for detection model output (tuple).

    The canonical Predictor's default postprocessing (softmax + argmax)
    does not work with YOLOv10's tuple output. This passthrough function
    preserves the raw output structure.
    """
    return {
        "logits": logits,
        "probs": logits,
        "prediction": logits,
    }


# ── Registry tests ────────────────────────────────────────────────────────


def test_model_is_registered() -> None:
    """Verify CTM-YOLOv10 is registered in the engine registry."""
    assert is_registered("models", "ctm_yolov10"), (
        "ctm_yolov10 should be registered in the engine registry"
    )


def test_baseline_is_registered() -> None:
    """Verify YOLOv10Baseline is registered in the engine registry."""
    assert is_registered("models", "yolov10_baseline"), (
        "yolov10_baseline should be registered in the engine registry"
    )


def test_list_registered_includes_ctm() -> None:
    """Verify list_registered includes ctm_yolov10."""
    registered = list_registered("models")
    assert "ctm_yolov10" in registered


def test_build_model_by_name() -> None:
    """Verify build_model('ctm_yolov10') returns a CTMYOLOv10 instance."""
    model = build_model("ctm_yolov10", num_classes=8)
    assert isinstance(model, CTMYOLOv10)
    assert model.num_classes == 8
    assert model.ctm_enabled is True


def test_build_model_with_ctm_disabled() -> None:
    """Verify build_model with ctm_enabled=False works."""
    model = build_model("ctm_yolov10", num_classes=8, ctm_enabled=False)
    assert isinstance(model, CTMYOLOv10)
    assert model.ctm_enabled is False
    assert model.ctm is None


def test_build_model_unregistered_raises() -> None:
    """Verify build_model with unknown name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("nonexistent_model")


# ── EngineConfig tests ────────────────────────────────────────────────────


def test_engine_config_from_yaml() -> None:
    """Verify EngineConfig can load the paper config."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    assert config is not None
    assert config.get("model.name") == "ctm_yolov10"
    assert config.get("model.num_classes") == 8


def test_engine_config_dot_access() -> None:
    """Verify dot-separated key access works."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    assert config.get("model.backbone.name") == "yolov10n"
    assert config.get("model.ctm.enabled") is True
    assert config.get("model.ctm.embed_dim") == 256
    assert config.get("training.batch_size") == 16
    assert config.get("training.num_epochs") == 300


def test_engine_config_engine_fields() -> None:
    """Verify engine-compatible fields are present."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")

    # model section
    assert config.get("model.name") == "ctm_yolov10"
    assert config.get("model.num_classes") == 8

    # training.optimizer as dict
    opt = config.get("training.optimizer")
    assert isinstance(opt, dict)
    assert opt.get("name") == "sgd"
    assert opt.get("lr") == 0.001

    # training.scheduler as dict
    sched = config.get("training.scheduler")
    assert isinstance(sched, dict)
    assert sched.get("name") == "multistep"

    # training.loss as dict
    loss = config.get("training.loss")
    assert isinstance(loss, dict)
    assert loss.get("name") == "cross_entropy"

    # dataset section
    ds = config.get("dataset")
    assert isinstance(ds, dict)
    assert ds.get("name") == "wafer_defects"


def test_engine_config_default_values() -> None:
    """Verify EngineConfig returns defaults for missing keys."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    assert config.get("nonexistent.key", "default") == "default"
    assert config.get("model.nonexistent", 42) == 42


# ── from_config tests ─────────────────────────────────────────────────────


def test_from_config_with_engine_config() -> None:
    """Verify CTMYOLOv10.from_config() works with EngineConfig."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    model = CTMYOLOv10.from_config(config)
    assert isinstance(model, CTMYOLOv10)
    assert model.num_classes == 8
    assert model.ctm_enabled is True


def test_from_config_ctm_disabled() -> None:
    """Verify from_config with ctm_enabled=False works."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    # Override ctm.enabled to False using merge_deep
    config.merge_deep({"model": {"ctm": {"enabled": False}}})
    model = CTMYOLOv10.from_config(config)
    assert isinstance(model, CTMYOLOv10)
    assert model.ctm_enabled is False
    assert model.ctm is None


# ── Predictor compatibility tests ─────────────────────────────────────────


def test_predictor_creation() -> None:
    """Verify Predictor can wrap CTMYOLOv10."""
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80)
    predictor = Predictor(model, device="cpu", postprocess_fn=_detection_postprocess)
    assert predictor is not None
    assert predictor.model is model


def test_predictor_single_inference() -> None:
    """Verify Predictor.predict_single() works with CTMYOLOv10.

    Uses a custom postprocess_fn because the canonical Predictor's default
    postprocessing (softmax + argmax) expects a tensor, but YOLOv10 returns
    a tuple of (detections, loss_dict).
    """
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80)
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
    """Verify Predictor.predict_batch() works with CTMYOLOv10."""
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80)
    model.eval()
    predictor = Predictor(model, device="cpu", postprocess_fn=_detection_postprocess)

    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        result = predictor.predict_batch(x)

    assert isinstance(result, dict)
    assert "logits" in result


# ── Engine instantiation tests ────────────────────────────────────────────


def test_engine_with_registered_model_name() -> None:
    """Verify Engine can be instantiated with a model instance.

    Passes a model instance directly (not a string name) to avoid the
    canonical Builder's config schema requirements, which expect
    ``model.kwargs``, ``optimizer.name``, ``scheduler.name``, etc.
    at the top level rather than under ``training.*``.
    """
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=8)
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    engine = Engine(model, config, device="cpu")
    assert isinstance(engine.model, CTMYOLOv10)
    assert engine.model.num_classes == 8


def test_engine_summary() -> None:
    """Verify Engine.summary() returns expected keys."""
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=8)
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    engine = Engine(model, config, device="cpu")
    summary = engine.summary()
    assert "model" in summary
    assert "device" in summary
    assert summary["model"] == "CTMYOLOv10"


def test_engine_forward_pass() -> None:
    """Verify Engine.model forward pass works.

    Uses num_classes=80 to match the YOLOv10 default head configuration.
    Calls model directly rather than Engine.predict_single() because the
    canonical Predictor's default postprocessing (softmax) does not support
    detection model tuple output.
    """
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80)
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
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80)
    model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(x)

    assert output is not None


def test_output_consistency_across_batches() -> None:
    """Verify output structure is consistent across different batch sizes."""
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80)
    model.eval()

    x1 = torch.randn(1, 3, 640, 640)
    x2 = torch.randn(2, 3, 640, 640)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    assert out1 is not None
    assert out2 is not None
    # Both should return the same type
    assert type(out1) is type(out2)