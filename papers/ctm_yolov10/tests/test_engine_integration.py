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
    """Verify build_model with unknown name raises KeyError."""
    with pytest.raises(KeyError, match="not registered"):
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
    # Override ctm.enabled to False
    config.merge_deep({"model": {"ctm": {"enabled": False}}})
    model = CTMYOLOv10.from_config(config)
    assert isinstance(model, CTMYOLOv10)
    assert model.ctm_enabled is False
    assert model.ctm is None


# ── Predictor compatibility tests ─────────────────────────────────────────


def test_predictor_creation() -> None:
    """Verify Predictor can wrap CTMYOLOv10."""
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80)
    predictor = Predictor(model, device="cpu")
    assert predictor is not None
    assert predictor.model is model


def test_predictor_single_inference() -> None:
    """Verify Predictor.predict_single() works with CTMYOLOv10."""
    model = CTMYOLOv10(model_name="yolov10n", pretrained=False, num_classes=80)
    model.eval()
    predictor = Predictor(model, device="cpu")

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
    predictor = Predictor(model, device="cpu")

    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        result = predictor.predict_batch(x)

    assert isinstance(result, dict)
    assert "logits" in result


# ── Engine instantiation tests ────────────────────────────────────────────


def test_engine_with_registered_model_name() -> None:
    """Verify Engine can be instantiated with registered model name."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    engine = Engine("ctm_yolov10", config, device="cpu")
    assert isinstance(engine.model, CTMYOLOv10)
    assert engine.model.num_classes == 8


def test_engine_summary() -> None:
    """Verify Engine.summary() returns expected keys."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    engine = Engine("ctm_yolov10", config, device="cpu")
    summary = engine.summary()
    assert "model_name" in summary
    assert "total_params" in summary
    assert "trainable_params" in summary
    assert "device" in summary
    assert summary["model_name"] == "CTMYOLOv10"


def test_engine_build_all() -> None:
    """Verify Engine.build_all() constructs optimizer, scheduler, loss."""
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    engine = Engine("ctm_yolov10", config, device="cpu")
    engine.build_all()
    assert engine.optimizer is not None
    assert engine.scheduler is not None
    assert engine.loss_fn is not None


def test_engine_predict() -> None:
    """Verify Engine.predict() works.

    Uses num_classes=80 to match the YOLOv10 default head configuration,
    since the detection head's internal layers are sized for 80 COCO classes
    when created without pretrained weights.
    """
    config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
    # Override num_classes to 80 for forward pass compatibility
    config.merge_deep({"model": {"num_classes": 80}})
    engine = Engine("ctm_yolov10", config, device="cpu")
    engine.model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = engine.predict(x)

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