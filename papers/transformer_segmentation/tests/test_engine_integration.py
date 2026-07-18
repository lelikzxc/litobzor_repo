"""Engine integration tests for SegFormer + Atrous.

Verifies:
- Model registration in the engine registry
- EngineConfig loading from the paper config
- build_model() by registered name
- SegFormer.from_config() with EngineConfig
- Predictor inference compatibility
- Output shape verification
- Engine instantiation with model instance
- Batch inference
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
from papers.transformer_segmentation.models.segformer import SegFormer


# ── Registry tests ────────────────────────────────────────────────────────


def test_model_is_registered() -> None:
    """Verify SegFormer is registered in the engine registry."""
    assert is_registered("models", "segformer_atrous"), (
        "segformer_atrous should be registered in the engine registry"
    )


def test_list_registered_includes_segformer() -> None:
    """Verify list_registered includes segformer_atrous."""
    registered = list_registered("models")
    assert "segformer_atrous" in registered


def test_build_model_by_name() -> None:
    """Verify build_model('segformer_atrous') returns a SegFormer instance."""
    model = build_model("segformer_atrous", in_channels=3, variant="B0", num_classes=8)
    assert isinstance(model, SegFormer)
    assert model.num_classes == 8
    assert model.variant == "B0"


def test_build_model_custom_params() -> None:
    """Verify build_model with custom parameters works."""
    model = build_model(
        "segformer_atrous",
        in_channels=3,
        variant="B2",
        num_classes=10,
        decoder_dim=128,
        atrous_enabled=False,
    )
    assert isinstance(model, SegFormer)
    assert model.variant == "B2"
    assert model.num_classes == 10
    assert model.decoder_dim == 128
    assert model.atrous_enabled is False


def test_build_model_unregistered_raises() -> None:
    """Verify build_model with unknown name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("nonexistent_model")


# ── EngineConfig tests ────────────────────────────────────────────────────


def test_engine_config_from_yaml() -> None:
    """Verify EngineConfig can load the paper config."""
    config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    assert config is not None
    assert config.get("model.name") == "segformer_atrous"
    assert config.get("model.num_classes") == 8


def test_engine_config_dot_access() -> None:
    """Verify dot-separated key access works."""
    config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    assert config.get("model.backbone.variant") == "B0"
    assert config.get("model.backbone.qkv_bias") is False
    assert config.get("model.decoder.decoder_dim") == 256
    assert config.get("model.atrous.enabled") is True
    assert config.get("model.atrous.rates") == [1, 6, 12, 18]
    assert config.get("model.input.image_size") == 512
    assert config.get("model.input.channels") == 3


def test_engine_config_engine_fields() -> None:
    """Verify engine-compatible fields are present."""
    config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")

    # model section
    assert config.get("model.name") == "segformer_atrous"
    assert config.get("model.num_classes") == 8

    # training.optimizer as dict
    opt = config.get("training.optimizer")
    assert isinstance(opt, dict)
    assert opt.get("name") == "adamw"
    assert opt.get("lr") == 0.00006

    # training.scheduler as dict
    sched = config.get("training.scheduler")
    assert isinstance(sched, dict)
    assert sched.get("name") == "poly"

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
    config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    assert config.get("nonexistent.key", "default") == "default"
    assert config.get("model.nonexistent", 42) == 42


# ── from_config tests ─────────────────────────────────────────────────────


def test_from_config_with_engine_config() -> None:
    """Verify SegFormer.from_config() works with EngineConfig."""
    config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    model = SegFormer.from_config(config)
    assert isinstance(model, SegFormer)
    assert model.num_classes == 8
    assert model.variant == "B0"
    assert model.embed_dims == [32, 64, 160, 256]
    assert model.depths == [2, 2, 2, 2]
    assert model.decoder_dim == 256
    assert model.atrous_enabled is True


def test_from_config_atrous_disabled() -> None:
    """Verify from_config with atrous_enabled=False works."""
    config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    # Override atrous.enabled to False using merge_deep
    config.merge_deep({"model": {"atrous": {"enabled": False}}})
    model = SegFormer.from_config(config)
    assert isinstance(model, SegFormer)
    assert model.atrous_enabled is False
    assert isinstance(model.atrous, torch.nn.Identity)


# ── Predictor compatibility tests ─────────────────────────────────────────


def test_predictor_creation() -> None:
    """Verify Predictor can wrap SegFormer."""
    model = SegFormer(num_classes=8)
    predictor = Predictor(model, device="cpu")
    assert predictor is not None
    assert predictor.model is model


def test_predictor_single_inference() -> None:
    """Verify Predictor.predict_single() works with SegFormer.

    SegFormer returns a segmentation logits tensor [B, num_classes, H, W],
    which is compatible with the canonical Predictor's default postprocessing
    (softmax + argmax).
    """
    model = SegFormer(num_classes=8)
    model.eval()
    predictor = Predictor(model, device="cpu")

    x = torch.randn(3, 64, 64)  # smaller size for speed
    with torch.no_grad():
        result = predictor.predict_single(x)

    assert isinstance(result, dict)
    assert "logits" in result
    assert "probs" in result
    assert "prediction" in result
    assert result["logits"].shape == (1, 8, 64, 64)
    assert result["probs"].shape == (1, 8, 64, 64)
    assert result["prediction"].shape == (1, 64, 64)


def test_predictor_batch_inference() -> None:
    """Verify Predictor.predict_batch() works with SegFormer."""
    model = SegFormer(num_classes=8)
    model.eval()
    predictor = Predictor(model, device="cpu")

    x = torch.randn(2, 3, 64, 64)  # smaller size for speed
    with torch.no_grad():
        result = predictor.predict_batch(x)

    assert isinstance(result, dict)
    assert "logits" in result
    assert "probs" in result
    assert "prediction" in result
    assert result["logits"].shape == (2, 8, 64, 64)
    assert result["probs"].shape == (2, 8, 64, 64)
    assert result["prediction"].shape == (2, 64, 64)


# ── Engine instantiation tests ────────────────────────────────────────────


def test_engine_with_model_instance() -> None:
    """Verify Engine can be instantiated with a SegFormer model instance."""
    model = SegFormer(num_classes=8)
    config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    engine = Engine(model, config, device="cpu")
    assert isinstance(engine.model, SegFormer)
    assert engine.model.num_classes == 8


def test_engine_summary() -> None:
    """Verify Engine.summary() returns expected keys."""
    model = SegFormer(num_classes=8)
    config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    engine = Engine(model, config, device="cpu")
    summary = engine.summary()
    assert "model" in summary
    assert "device" in summary
    assert summary["model"] == "SegFormer"


def test_engine_predict_single() -> None:
    """Verify Engine.predict_single() works with SegFormer."""
    model = SegFormer(num_classes=8)
    model.eval()
    config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    engine = Engine(model, config, device="cpu")

    x = torch.randn(3, 64, 64)  # smaller size for speed
    with torch.no_grad():
        result = engine.predict_single(x)

    assert isinstance(result, dict)
    assert "logits" in result
    assert "probs" in result
    assert "prediction" in result
    assert result["logits"].shape == (1, 8, 64, 64)
    assert result["probs"].shape == (1, 8, 64, 64)


# ── Output shape tests ────────────────────────────────────────────────────


def test_output_shape_with_synthetic_input() -> None:
    """Verify forward output shape with synthetic input."""
    model = SegFormer(num_classes=8)
    model.eval()

    x = torch.randn(1, 3, 64, 64)  # smaller size for speed
    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 8, 64, 64)


def test_output_consistency_across_batches() -> None:
    """Verify output shape is consistent across different batch sizes."""
    model = SegFormer(num_classes=8)
    model.eval()

    x1 = torch.randn(1, 3, 64, 64)
    x2 = torch.randn(4, 3, 64, 64)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    assert out1.shape == (1, 8, 64, 64)
    assert out2.shape == (4, 8, 64, 64)


def test_output_is_logits_not_probs() -> None:
    """Verify forward output is raw logits (not softmax-applied)."""
    model = SegFormer(num_classes=8)
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        output = model(x)

    # Logits can be any value, not bounded to [0, 1]
    assert output.shape == (2, 8, 64, 64)
    # At least some values should be outside [0, 1] (raw logits)
    assert (output > 1.0).any() or (output < 0.0).any(), (
        "Output should be raw logits, not probabilities"
    )