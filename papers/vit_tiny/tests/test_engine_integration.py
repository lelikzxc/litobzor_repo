"""Engine integration tests for Tiny Vision Transformer (ViT-Tiny).

Verifies:
- Model registration in the engine registry
- EngineConfig loading from the paper config
- build_model() by registered name
- ViTTiny.from_config() with EngineConfig
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
from papers.vit_tiny.models.vit_tiny import ViTTiny


# ── Registry tests ────────────────────────────────────────────────────────


def test_model_is_registered() -> None:
    """Verify ViTTiny is registered in the engine registry."""
    assert is_registered("models", "vit_tiny"), (
        "vit_tiny should be registered in the engine registry"
    )


def test_list_registered_includes_vit_tiny() -> None:
    """Verify list_registered includes vit_tiny."""
    registered = list_registered("models")
    assert "vit_tiny" in registered


def test_build_model_by_name() -> None:
    """Verify build_model('vit_tiny') returns a ViTTiny instance."""
    model = build_model("vit_tiny", image_size=32, in_channels=1, num_classes=8)
    assert isinstance(model, ViTTiny)
    assert model.num_classes == 8
    assert model.embed_dim == 192


def test_build_model_custom_params() -> None:
    """Verify build_model with custom parameters works."""
    model = build_model(
        "vit_tiny",
        image_size=64,
        in_channels=3,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        num_classes=4,
    )
    assert isinstance(model, ViTTiny)
    assert model.image_size == 64
    assert model.patch_embed.proj.in_channels == 3
    assert model.embed_dim == 128
    assert model.num_layers == 2
    assert model.num_heads == 4
    assert model.num_classes == 4


def test_build_model_unregistered_raises() -> None:
    """Verify build_model with unknown name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("nonexistent_model")


# ── EngineConfig tests ────────────────────────────────────────────────────


def test_engine_config_from_yaml() -> None:
    """Verify EngineConfig can load the paper config."""
    config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")
    assert config is not None
    assert config.get("model.name") == "vit_tiny"
    assert config.get("model.num_classes") == 8


def test_engine_config_dot_access() -> None:
    """Verify dot-separated key access works."""
    config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")
    assert config.get("model.arch.image_size") == 32
    assert config.get("model.arch.patch_size") == 4
    assert config.get("model.arch.in_channels") == 1
    assert config.get("model.arch.embed_dim") == 192
    assert config.get("model.arch.num_layers") == 4
    assert config.get("model.arch.num_heads") == 3


def test_engine_config_engine_fields() -> None:
    """Verify engine-compatible fields are present."""
    config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")

    # model section
    assert config.get("model.name") == "vit_tiny"
    assert config.get("model.num_classes") == 8

    # training.optimizer as dict
    opt = config.get("training.optimizer")
    assert isinstance(opt, dict)
    assert opt.get("name") == "adamw"
    assert opt.get("lr") == 0.0001

    # training.scheduler as dict
    sched = config.get("training.scheduler")
    assert isinstance(sched, dict)
    assert sched.get("name") == "cosine"

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
    config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")
    assert config.get("nonexistent.key", "default") == "default"
    assert config.get("model.nonexistent", 42) == 42


# ── from_config tests ─────────────────────────────────────────────────────


def test_from_config_with_engine_config() -> None:
    """Verify ViTTiny.from_config() works with EngineConfig."""
    config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")
    model = ViTTiny.from_config(config)
    assert isinstance(model, ViTTiny)
    assert model.num_classes == 8
    assert model.image_size == 32
    assert model.patch_size == 4
    assert model.embed_dim == 192
    assert model.num_layers == 4
    assert model.num_heads == 3


# ── Predictor compatibility tests ─────────────────────────────────────────


def test_predictor_creation() -> None:
    """Verify Predictor can wrap ViTTiny."""
    model = ViTTiny(num_classes=8)
    predictor = Predictor(model, device="cpu")
    assert predictor is not None
    assert predictor.model is model


def test_predictor_single_inference() -> None:
    """Verify Predictor.predict_single() works with ViTTiny.

    ViT-Tiny returns a logits tensor [B, num_classes], which is compatible
    with the canonical Predictor's default postprocessing (softmax + argmax).
    """
    model = ViTTiny(num_classes=8)
    model.eval()
    predictor = Predictor(model, device="cpu")

    x = torch.randn(1, 32, 32)  # grayscale HWC
    with torch.no_grad():
        result = predictor.predict_single(x)

    assert isinstance(result, dict)
    assert "logits" in result
    assert "probs" in result
    assert "prediction" in result
    assert result["logits"].shape == (1, 8)
    assert result["probs"].shape == (1, 8)
    # probs should sum to 1 (softmax)
    assert abs(result["probs"].sum().item() - 1.0) < 1e-5


def test_predictor_batch_inference() -> None:
    """Verify Predictor.predict_batch() works with ViTTiny."""
    model = ViTTiny(num_classes=8)
    model.eval()
    predictor = Predictor(model, device="cpu")

    x = torch.randn(2, 1, 32, 32)  # NCHW
    with torch.no_grad():
        result = predictor.predict_batch(x)

    assert isinstance(result, dict)
    assert "logits" in result
    assert "probs" in result
    assert "prediction" in result
    assert result["logits"].shape == (2, 8)
    assert result["probs"].shape == (2, 8)
    assert result["prediction"].shape == (2,)


# ── Engine instantiation tests ────────────────────────────────────────────


def test_engine_with_model_instance() -> None:
    """Verify Engine can be instantiated with a ViTTiny model instance."""
    model = ViTTiny(num_classes=8)
    config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")
    engine = Engine(model, config, device="cpu")
    assert isinstance(engine.model, ViTTiny)
    assert engine.model.num_classes == 8


def test_engine_summary() -> None:
    """Verify Engine.summary() returns expected keys."""
    model = ViTTiny(num_classes=8)
    config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")
    engine = Engine(model, config, device="cpu")
    summary = engine.summary()
    assert "model" in summary
    assert "device" in summary
    assert summary["model"] == "ViTTiny"


def test_engine_predict_single() -> None:
    """Verify Engine.predict_single() works with ViTTiny."""
    model = ViTTiny(num_classes=8)
    model.eval()
    config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")
    engine = Engine(model, config, device="cpu")

    x = torch.randn(1, 32, 32)  # grayscale HWC
    with torch.no_grad():
        result = engine.predict_single(x)

    assert isinstance(result, dict)
    assert "logits" in result
    assert "probs" in result
    assert "prediction" in result
    assert result["logits"].shape == (1, 8)
    assert result["probs"].shape == (1, 8)


# ── Output shape tests ────────────────────────────────────────────────────


def test_output_shape_with_synthetic_input() -> None:
    """Verify forward output shape with synthetic input."""
    model = ViTTiny(num_classes=8)
    model.eval()

    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 8)


def test_output_consistency_across_batches() -> None:
    """Verify output shape is consistent across different batch sizes."""
    model = ViTTiny(num_classes=8)
    model.eval()

    x1 = torch.randn(1, 1, 32, 32)
    x2 = torch.randn(4, 1, 32, 32)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    assert out1.shape == (1, 8)
    assert out2.shape == (4, 8)


def test_output_is_logits_not_probs() -> None:
    """Verify forward output is raw logits (not softmax-applied)."""
    model = ViTTiny(num_classes=8)
    model.eval()

    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        output = model(x)

    # Logits can be any value, not bounded to [0, 1]
    assert output.shape == (2, 8)
    # At least some values should be outside [0, 1] (raw logits)
    assert (output > 1.0).any() or (output < 0.0).any(), (
        "Output should be raw logits, not probabilities"
    )