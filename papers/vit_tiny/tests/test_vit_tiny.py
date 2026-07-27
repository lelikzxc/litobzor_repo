"""Tests for Tiny Vision Transformer model."""

from __future__ import annotations

import torch

from common.utils.config import Config
from common.utils.seed import set_seed
from papers.vit_tiny.models.vit_tiny import ViTTiny

ROOT_CONFIG = "papers/vit_tiny/configs/config.yaml"


def test_model_import() -> None:
    """Verify ViTTiny can be imported."""
    from papers.vit_tiny.models import ViTTiny  # noqa: F811

    assert ViTTiny is not None


def test_model_creation() -> None:
    """Verify model can be instantiated with default parameters."""
    model = ViTTiny()
    assert isinstance(model, ViTTiny)


def test_model_creation_from_config() -> None:
    """Verify model can be instantiated from YAML config."""
    config = Config.from_yaml(ROOT_CONFIG)
    model = ViTTiny.from_config(config)
    assert isinstance(model, ViTTiny)


def test_forward_pass_shape() -> None:
    """Verify forward pass produces correct output shape."""
    config = Config.from_yaml(ROOT_CONFIG)
    model = ViTTiny.from_config(config)
    model.eval()

    batch_size = 4
    image_size = config.get("model.arch.image_size", 64)
    in_channels = config.get("model.arch.in_channels", 1)
    num_classes = config.get("model.arch.num_classes", 9)

    x = torch.randn(batch_size, in_channels, image_size, image_size)
    logits = model(x)

    assert logits.shape == (batch_size, num_classes), (
        f"Expected ({batch_size}, {num_classes}), got {logits.shape}"
    )


def test_forward_pass_deterministic() -> None:
    """Verify deterministic inference after set_seed()."""
    config = Config.from_yaml(ROOT_CONFIG)
    model = ViTTiny.from_config(config)
    model.eval()

    image_size = config.get("model.arch.image_size", 64)
    in_channels = config.get("model.arch.in_channels", 1)

    set_seed(42)
    x = torch.randn(2, in_channels, image_size, image_size)
    out1 = model(x)

    set_seed(42)
    x = torch.randn(2, in_channels, image_size, image_size)
    out2 = model(x)

    torch.testing.assert_close(out1, out2)


def test_model_has_trainable_parameters() -> None:
    """Verify model has trainable parameters."""
    model = ViTTiny()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0, "Model has zero trainable parameters"


def test_model_returns_logits_not_probabilities() -> None:
    """Verify model returns logits (not bounded to [0,1])."""
    config = Config.from_yaml(ROOT_CONFIG)
    model = ViTTiny.from_config(config)
    model.eval()

    image_size = config.get("model.arch.image_size", 64)
    in_channels = config.get("model.arch.in_channels", 1)

    x = torch.randn(2, in_channels, image_size, image_size)
    logits = model(x)

    # Logits can be any value, not just [0, 1]
    assert not torch.all((logits >= 0) & (logits <= 1)), (
        "Model appears to apply Softmax — output is bounded to [0,1]"
    )


def test_forward_pass_different_batch_sizes() -> None:
    """Verify forward pass works with different batch sizes."""
    config = Config.from_yaml(ROOT_CONFIG)
    model = ViTTiny.from_config(config)
    model.eval()

    image_size = config.get("model.arch.image_size", 64)
    in_channels = config.get("model.arch.in_channels", 1)
    num_classes = config.get("model.arch.num_classes", 9)

    for batch_size in [1, 2, 8]:
        x = torch.randn(batch_size, in_channels, image_size, image_size)
        logits = model(x)
        assert logits.shape == (batch_size, num_classes), (
            f"Batch size {batch_size}: expected ({batch_size}, {num_classes}), "
            f"got {logits.shape}"
        )


def test_gradients_flow() -> None:
    """Verify gradients flow through all parameters."""
    config = Config.from_yaml(ROOT_CONFIG)
    model = ViTTiny.from_config(config)
    model.train()

    image_size = config.get("model.arch.image_size", 64)
    in_channels = config.get("model.arch.in_channels", 1)

    x = torch.randn(2, in_channels, image_size, image_size)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert param.grad.abs().sum().item() > 0, (
            f"Parameter {name} has zero gradient"
        )