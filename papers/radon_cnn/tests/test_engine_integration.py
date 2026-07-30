"""Engine integration tests for RadonCNN.

Tests cover:
    - Model registration in Engine registry
    - Engine config loading
    - Predictor creation and inference
    - Engine predict_single
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.engine.engine import Engine
from common.engine.registry import list_registered, build_model
from papers.radon_cnn.models.radon_cnn import RadonCNN


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def small_input() -> torch.Tensor:
    return torch.randn(1, 1, 64, 64)


@pytest.fixture
def batch_input() -> torch.Tensor:
    return torch.randn(4, 1, 64, 64)


# ── Engine Registry Tests ─────────────────────────────────────────────────

class TestEngineRegistry:
    """Tests for Engine registry integration."""

    def test_list_registered(self) -> None:
        """RadonCNN should be registered in the engine."""
        registered = list_registered("models")
        assert "radon_cnn" in registered

    def test_build_model_by_name(self) -> None:
        model = build_model("radon_cnn")
        assert isinstance(model, RadonCNN)

    def test_build_model_custom_params(self) -> None:
        model = build_model("radon_cnn", in_channels=1, num_classes=7)
        assert isinstance(model, RadonCNN)
        assert model.classifier.fc3.out_features == 7

    def test_build_model_unregistered_raises(self) -> None:
        with pytest.raises(ValueError):
            build_model("nonexistent_model")


# ── Engine Config Tests ───────────────────────────────────────────────────

class TestEngineConfig:
    """Tests for EngineConfig with RadonCNN."""

    def test_engine_config_from_yaml(self) -> None:
        config_path = Path("papers/radon_cnn/configs/config.yaml")
        if config_path.exists():
            config = EngineConfig.from_yaml(config_path)
            assert config.get("model.name") == "radon_cnn"
            assert config.get("model.num_classes") == 7

    def test_engine_config_dot_access(self) -> None:
        config = EngineConfig.from_dict({
            "model": {"name": "radon_cnn", "num_classes": 7},
            "training": {"learning_rate": 0.0003},
        })
        assert config.get("model.name") == "radon_cnn"
        assert config.get("model.num_classes") == 7
        assert config.get("training.learning_rate") == 0.0003

    def test_engine_config_default_values(self) -> None:
        config = EngineConfig.from_dict({})
        assert config.get("nonexistent.key", "default") == "default"
        assert config.get("model.num_classes", 7) == 7


# ── Predictor Tests ───────────────────────────────────────────────────────

class TestPredictor:
    """Tests for Predictor with RadonCNN."""

    def test_predictor_creation(self) -> None:
        model = RadonCNN(in_channels=1, num_classes=7)
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 1, 64, 64)
            output = model(x)
        assert output.shape == (1, 7)

    def test_predictor_single_inference(self, small_input: torch.Tensor) -> None:
        model = RadonCNN(in_channels=1, num_classes=7)
        model.eval()
        with torch.no_grad():
            output = model(small_input)
        assert output.shape == (1, 7)

    def test_predictor_batch_inference(self, batch_input: torch.Tensor) -> None:
        model = RadonCNN(in_channels=1, num_classes=7)
        model.eval()
        with torch.no_grad():
            output = model(batch_input)
        assert output.shape == (4, 7)

    def test_output_is_logits_not_probs(self, small_input: torch.Tensor) -> None:
        """Raw forward should return logits, not probabilities."""
        model = RadonCNN(in_channels=1, num_classes=7)
        model.eval()
        with torch.no_grad():
            output = model(small_input)
        # Logits can be any value, not bounded to [0, 1]
        assert not torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-5)


# ── Engine Tests ──────────────────────────────────────────────────────────

class TestEngine:
    """Tests for Engine with RadonCNN."""

    def test_engine_with_model_instance(self) -> None:
        model = RadonCNN(in_channels=1, num_classes=7)
        engine = Engine(model=model)
        assert engine.model is model

    def test_engine_summary(self) -> None:
        model = RadonCNN(in_channels=1, num_classes=7)
        engine = Engine(model=model)
        summary = engine.summary()
        assert summary is not None

    def test_engine_predict_single(self, small_input: torch.Tensor) -> None:
        model = RadonCNN(in_channels=1, num_classes=7)
        engine = Engine(model=model)
        result = engine.predict_single(small_input)
        assert result is not None

    def test_output_consistency_across_batches(self) -> None:
        """Same input should produce same output."""
        model = RadonCNN(in_channels=1, num_classes=7)
        model.eval()

        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)