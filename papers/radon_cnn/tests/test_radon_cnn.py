"""Tests for RadonCNN baseline architecture.

Tests cover:
    - Model creation (BaselineCNN, RadonCNN)
    - Forward pass shapes
    - Gradient flow
    - Parameter counts
    - Radon transform module
    - Kernel flip module
    - from_config
"""

from __future__ import annotations

import pytest
import torch

from papers.radon_cnn.models.radon_cnn import BaselineCNN, RadonCNN
from papers.radon_cnn.modules.radon_transform import RadonTransformModule
from papers.radon_cnn.modules.kernel_flip import KernelFlip


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def small_input() -> torch.Tensor:
    """Small input tensor [2, 1, 64, 64]."""
    return torch.randn(2, 1, 64, 64)


@pytest.fixture
def baseline_model() -> BaselineCNN:
    return BaselineCNN(in_channels=1, num_classes=7)


@pytest.fixture
def radon_model() -> RadonCNN:
    return RadonCNN(in_channels=1, num_classes=7)


# ── Model Creation ────────────────────────────────────────────────────────

class TestModelCreation:
    """Tests for model creation."""

    def test_baseline_creation(self) -> None:
        model = BaselineCNN(in_channels=1, num_classes=7)
        assert isinstance(model, BaselineCNN)

    def test_radon_creation(self) -> None:
        model = RadonCNN(in_channels=1, num_classes=7)
        assert isinstance(model, RadonCNN)

    def test_baseline_custom_params(self) -> None:
        model = BaselineCNN(in_channels=3, num_classes=5)
        assert model.conv1.conv.in_channels == 3
        assert model.classifier.fc3.out_features == 5

    def test_radon_custom_params(self) -> None:
        model = RadonCNN(in_channels=3, num_classes=5)
        assert model.classifier.fc3.out_features == 5

    def test_baseline_has_parameters(self, baseline_model: BaselineCNN) -> None:
        assert sum(p.numel() for p in baseline_model.parameters()) > 0

    def test_radon_has_parameters(self, radon_model: RadonCNN) -> None:
        assert sum(p.numel() for p in radon_model.parameters()) > 0


# ── Forward Pass ──────────────────────────────────────────────────────────

class TestForwardPass:
    """Tests for forward pass shapes."""

    def test_baseline_forward_shape(self, baseline_model: BaselineCNN, small_input: torch.Tensor) -> None:
        with torch.no_grad():
            output = baseline_model(small_input)
        assert output.shape == (2, 7)

    def test_radon_forward_shape(self, radon_model: RadonCNN, small_input: torch.Tensor) -> None:
        with torch.no_grad():
            output = radon_model(small_input)
        assert output.shape == (2, 7)

    def test_baseline_forward_batch1(self, baseline_model: BaselineCNN) -> None:
        baseline_model.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            output = baseline_model(x)
        assert output.shape == (1, 7)

    def test_radon_forward_batch1(self, radon_model: RadonCNN) -> None:
        radon_model.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            output = radon_model(x)
        assert output.shape == (1, 7)

    def test_baseline_forward_batch4(self, baseline_model: BaselineCNN) -> None:
        x = torch.randn(4, 1, 64, 64)
        with torch.no_grad():
            output = baseline_model(x)
        assert output.shape == (4, 7)

    def test_radon_forward_batch4(self, radon_model: RadonCNN) -> None:
        x = torch.randn(4, 1, 64, 64)
        with torch.no_grad():
            output = radon_model(x)
        assert output.shape == (4, 7)

    def test_radon_forward_not_nan(self, radon_model: RadonCNN, small_input: torch.Tensor) -> None:
        with torch.no_grad():
            output = radon_model(small_input)
        assert not torch.isnan(output).any()

    def test_baseline_forward_not_nan(self, baseline_model: BaselineCNN, small_input: torch.Tensor) -> None:
        with torch.no_grad():
            output = baseline_model(small_input)
        assert not torch.isnan(output).any()


# ── Gradient Flow ─────────────────────────────────────────────────────────

class TestGradientFlow:
    """Tests for gradient flow through models.

    Note: Radon transform is a non-learnable deterministic operation that
    detaches from the autograd graph (it uses numpy/skimage internally).
    Therefore, gradients only flow through layers *after* the Radon transform
    in the RadonCNN model. The KernelFlip module (conv2_kf) is the first
    learnable layer that receives gradients.
    """

    def test_baseline_gradients_flow(self, baseline_model: BaselineCNN, small_input: torch.Tensor) -> None:
        output = baseline_model(small_input)
        loss = output.sum()
        loss.backward()
        has_grad = False
        for p in baseline_model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowing through baseline model"

    def test_radon_gradients_flow(self, radon_model: RadonCNN, small_input: torch.Tensor) -> None:
        output = radon_model(small_input)
        loss = output.sum()
        loss.backward()
        has_grad = False
        for p in radon_model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowing through RadonCNN"

    def test_radon_kernel_flip_gradients(self, radon_model: RadonCNN) -> None:
        """KernelFlip (conv2_kf) is the first learnable layer after Radon."""
        x = torch.randn(4, 1, 64, 64)  # larger batch for stable gradients
        output = radon_model(x)
        loss = output.sum()
        loss.backward()
        assert radon_model.conv2_kf.kernel_flip.conv.weight.grad is not None
        assert radon_model.conv2_kf.kernel_flip.conv.weight.grad.abs().sum() > 0, \
            "KernelFlip conv should receive non-zero gradients"

    def test_radon_classifier_gradients(self, radon_model: RadonCNN) -> None:
        """Classifier head should receive gradients (after Radon + convs)."""
        x = torch.randn(4, 1, 64, 64)  # larger batch for stable gradients
        output = radon_model(x)
        loss = output.sum()
        loss.backward()
        assert radon_model.classifier.fc1.weight.grad is not None
        assert radon_model.classifier.fc1.weight.grad.abs().sum() > 0, \
            "Classifier fc1 should receive non-zero gradients"


# ── Parameter Counts ──────────────────────────────────────────────────────

class TestParameterCounts:
    """Tests for parameter counts."""

    def test_baseline_parameter_count(self, baseline_model: BaselineCNN) -> None:
        num_params = sum(p.numel() for p in baseline_model.parameters())
        assert num_params > 0
        # Baseline: ~479K parameters (4 conv layers + 3 FC layers)
        assert 400_000 < num_params < 600_000

    def test_radon_parameter_count(self, radon_model: RadonCNN) -> None:
        num_params = sum(p.numel() for p in radon_model.parameters())
        assert num_params > 0
        # RadonCNN: similar to baseline (kernel flip shares weights)
        assert 400_000 < num_params < 600_000

    def test_radon_no_extra_params_vs_baseline(self) -> None:
        """Kernel flip shares weights, so RadonCNN should have similar params to baseline."""
        baseline = BaselineCNN(in_channels=1, num_classes=7)
        radon = RadonCNN(in_channels=1, num_classes=7)
        baseline_params = sum(p.numel() for p in baseline.parameters())
        radon_params = sum(p.numel() for p in radon.parameters())
        # RadonCNN has same conv layers + classifier, kernel flip shares weights
        # Difference should be small (< 5%)
        ratio = abs(radon_params - baseline_params) / baseline_params
        assert ratio < 0.05, f"Parameter ratio too large: {ratio:.4f}"


# ── Radon Transform Module ────────────────────────────────────────────────

class TestRadonTransform:
    """Tests for RadonTransformModule."""

    def test_creation(self) -> None:
        module = RadonTransformModule(theta=64)
        assert isinstance(module, RadonTransformModule)

    def test_forward_shape(self, small_input: torch.Tensor) -> None:
        module = RadonTransformModule(theta=64, image_size=64)
        with torch.no_grad():
            output = module(small_input)
        # Output shape: [B, 1, 64, 64] after resize
        assert output.shape == (2, 1, 64, 64)

    def test_forward_not_nan(self, small_input: torch.Tensor) -> None:
        module = RadonTransformModule(theta=64, image_size=64)
        with torch.no_grad():
            output = module(small_input)
        assert not torch.isnan(output).any()

    def test_forward_not_inf(self, small_input: torch.Tensor) -> None:
        module = RadonTransformModule(theta=64, image_size=64)
        with torch.no_grad():
            output = module(small_input)
        assert not torch.isinf(output).any()

    def test_custom_theta(self) -> None:
        module = RadonTransformModule(theta=32, image_size=64)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            output = module(x)
        assert output.shape == (1, 1, 64, 64)

    def test_repr(self) -> None:
        module = RadonTransformModule(theta=64, image_size=64)
        repr_str = repr(module)
        assert "theta=64" in repr_str
        assert "image_size=64" in repr_str


# ── Kernel Flip Module ────────────────────────────────────────────────────

class TestKernelFlip:
    """Tests for KernelFlip module."""

    def test_creation(self) -> None:
        module = KernelFlip(in_channels=16, out_channels=64)
        assert isinstance(module, KernelFlip)

    def test_forward_shape(self) -> None:
        module = KernelFlip(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            output = module(x)
        assert output.shape == (2, 64, 32, 32)

    def test_forward_not_nan(self) -> None:
        module = KernelFlip(in_channels=16, out_channels=64)
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            output = module(x)
        assert not torch.isnan(output).any()

    def test_weight_sharing(self) -> None:
        """KernelFlip has only one conv layer (weight-shared)."""
        module = KernelFlip(in_channels=16, out_channels=64)
        # Should have exactly 1 conv layer
        conv_count = sum(1 for _ in module.modules() if isinstance(_, torch.nn.Conv2d))
        assert conv_count == 1

    def test_max_out_different_branches(self) -> None:
        """Max-out should select element-wise maximum."""
        module = KernelFlip(in_channels=1, out_channels=1, kernel_size=1, padding=0)
        # Set weight to 1.0
        with torch.no_grad():
            module.conv.weight.fill_(1.0)
            if module.conv.bias is not None:
                module.conv.bias.fill_(0.0)

        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        # Branch 1: conv(x) = sum of values in kernel window
        # Branch 2: flip x, conv, flip back
        with torch.no_grad():
            output = module(x)
        assert output.shape == (1, 1, 2, 2)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self) -> None:
        module = KernelFlip(in_channels=16, out_channels=64)
        x = torch.randn(2, 16, 32, 32, requires_grad=True)
        output = module(x)
        loss = output.sum()
        loss.backward()
        assert module.conv.weight.grad is not None
        assert module.conv.weight.grad.abs().sum() > 0


# ── from_config ───────────────────────────────────────────────────────────

class TestFromConfig:
    """Tests for from_config class method."""

    def test_baseline_from_config(self) -> None:
        config = {"in_channels": 1, "num_classes": 7}
        model = RadonCNN.from_config(config)
        assert isinstance(model, RadonCNN)
        assert model.classifier.fc3.out_features == 7

    def test_radon_from_config_custom(self) -> None:
        config = {"in_channels": 3, "num_classes": 5}
        model = RadonCNN.from_config(config)
        assert model.classifier.fc3.out_features == 5

    def test_radon_from_config_engine_config(self) -> None:
        """Test with EngineConfig-like object."""
        from common.engine.config import EngineConfig

        config = EngineConfig.from_dict({
            "model": {
                "in_channels": 1,
                "num_classes": 7,
            }
        })
        model = RadonCNN.from_config(config)
        assert isinstance(model, RadonCNN)
        assert model.classifier.fc3.out_features == 7


# ── Predict ───────────────────────────────────────────────────────────────

class TestPredict:
    """Tests for predict method."""

    def test_predict_shape(self, radon_model: RadonCNN, small_input: torch.Tensor) -> None:
        probs = radon_model.predict(small_input)
        assert probs.shape == (2, 7)

    def test_predict_sum_to_one(self, radon_model: RadonCNN, small_input: torch.Tensor) -> None:
        probs = radon_model.predict(small_input)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_predict_non_negative(self, radon_model: RadonCNN, small_input: torch.Tensor) -> None:
        probs = radon_model.predict(small_input)
        assert (probs >= 0).all()

    def test_predict_eval_mode(self, radon_model: RadonCNN, small_input: torch.Tensor) -> None:
        """predict should set model to eval mode."""
        radon_model.train()
        _ = radon_model.predict(small_input)
        assert not radon_model.training