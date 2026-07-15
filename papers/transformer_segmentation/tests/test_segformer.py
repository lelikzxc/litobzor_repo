"""Tests for SegFormer baseline model."""

from __future__ import annotations

import torch
import pytest

from papers.transformer_segmentation.models.segformer import SegFormer
from papers.transformer_segmentation.models.decoder import MLPDecoder
from papers.transformer_segmentation.modules import (
    OverlapPatchEmbed,
    EfficientSelfAttention,
    MixFFN,
    TransformerBlock,
    MiTStage,
    MiTBackbone,
    MIT_CONFIGS,
)


# ── Module import tests ──────────────────────────────────────────────────


def test_import() -> None:
    """Verify all modules can be imported."""
    assert SegFormer is not None
    assert MLPDecoder is not None
    assert OverlapPatchEmbed is not None
    assert EfficientSelfAttention is not None
    assert MixFFN is not None
    assert TransformerBlock is not None
    assert MiTStage is not None
    assert MiTBackbone is not None
    assert MIT_CONFIGS is not None


# ── Model creation tests ─────────────────────────────────────────────────


def test_model_creation() -> None:
    """Verify SegFormer can be instantiated with default params."""
    model = SegFormer()
    assert isinstance(model, SegFormer)
    assert model.num_classes == 8
    assert model.variant == "B0"
    assert model.embed_dims == [32, 64, 160, 256]
    assert model.depths == [2, 2, 2, 2]


def test_model_creation_custom() -> None:
    """Verify SegFormer can be instantiated with custom params."""
    model = SegFormer(
        in_channels=3,
        variant="B2",
        num_classes=10,
        decoder_dim=128,
        dropout=0.1,
    )
    assert isinstance(model, SegFormer)
    assert model.num_classes == 10
    assert model.variant == "B2"
    assert model.decoder_dim == 128
    assert model.embed_dims == [64, 128, 320, 512]
    assert model.depths == [3, 4, 6, 3]


def test_model_creation_all_variants() -> None:
    """Verify all MiT variants can be instantiated."""
    for variant in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        model = SegFormer(variant=variant, num_classes=8)
        assert isinstance(model, SegFormer), f"Failed for variant {variant}"
        assert model.variant == variant


def test_model_creation_invalid_variant() -> None:
    """Verify invalid variant raises ValueError."""
    with pytest.raises(ValueError):
        SegFormer(variant="INVALID")


# ── Forward pass tests ───────────────────────────────────────────────────


def test_forward_shape() -> None:
    """Verify forward pass produces correct output shape.

    Input:  [2, 3, 512, 512]
    Output: [2, 8, 512, 512] (logits, no Softmax)
    """
    model = SegFormer(num_classes=8)
    model.eval()

    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    assert out is not None, "Forward pass returned None"
    assert out.shape == (2, 8, 512, 512), f"Expected (2, 8, 512, 512), got {out.shape}"


def test_forward_logits_only() -> None:
    """Verify model returns logits (not probabilities)."""
    model = SegFormer(num_classes=8)
    model.eval()

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 8, 512, 512)
    # Values should not be softmaxed (can be negative or > 1)
    assert (out < 0).any() or (out > 1).any(), (
        "Output appears to be probabilities, not logits"
    )


def test_gradients_flow() -> None:
    """Verify gradients flow through the entire model."""
    model = SegFormer(num_classes=8)
    x = torch.randn(1, 3, 512, 512, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient is None"
    assert x.grad.abs().sum() > 0, "Input gradient is zero"


def test_parameter_count() -> None:
    """Verify model has trainable parameters."""
    model = SegFormer()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0, "Model has zero trainable parameters"


# ── Config-driven creation ───────────────────────────────────────────────


def test_from_config() -> None:
    """Verify SegFormer.from_config() works."""
    from common.utils.config import Config

    config = Config.from_yaml("papers/transformer_segmentation/configs/config.yaml")
    model = SegFormer.from_config(config)
    assert isinstance(model, SegFormer)
    assert model.num_classes == 8
    assert model.variant == "B0"
    assert model.embed_dims == [32, 64, 160, 256]
    assert model.depths == [2, 2, 2, 2]
    assert model.decoder_dim == 256


# ── Module unit tests ────────────────────────────────────────────────────


def test_overlap_patch_embed_shape() -> None:
    """Verify OverlapPatchEmbed preserves expected output shape."""
    ope = OverlapPatchEmbed(in_channels=3, embed_dim=32, stride=4)
    x = torch.randn(2, 3, 512, 512)
    out = ope(x)
    assert out.shape == (2, 32, 128, 128), f"Got {out.shape}"


def test_efficient_self_attention_shape() -> None:
    """Verify EfficientSelfAttention preserves input shape."""
    esa = EfficientSelfAttention(dim=32, num_heads=1, reduction_ratio=8)
    x = torch.randn(2, 32, 128, 128)
    out = esa(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_mix_ffn_shape() -> None:
    """Verify MixFFN preserves input shape."""
    mffn = MixFFN(dim=32, hidden_dim=128)
    x = torch.randn(2, 32, 128, 128)
    out = mffn(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_transformer_block_shape() -> None:
    """Verify TransformerBlock preserves input shape."""
    block = TransformerBlock(dim=32, num_heads=1, reduction_ratio=8, mlp_ratio=4)
    x = torch.randn(2, 32, 128, 128)
    out = block(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_mit_stage_shape() -> None:
    """Verify MiTStage produces correct output shape."""
    stage = MiTStage(
        in_channels=3,
        embed_dim=32,
        depth=2,
        num_heads=1,
        reduction_ratio=8,
        stride=4,
    )
    x = torch.randn(2, 3, 512, 512)
    out = stage(x)
    assert out.shape == (2, 32, 128, 128), f"Got {out.shape}"


def test_mit_backbone_shapes() -> None:
    """Verify MiTBackbone produces 4 feature maps at correct resolutions."""
    backbone = MiTBackbone(in_channels=3)
    x = torch.randn(2, 3, 512, 512)
    features = backbone(x)
    assert len(features) == 4, f"Expected 4 features, got {len(features)}"
    expected_shapes = [
        (2, 32, 128, 128),   # 1/4
        (2, 64, 64, 64),     # 1/8
        (2, 160, 32, 32),    # 1/16
        (2, 256, 16, 16),    # 1/32
    ]
    for i, (feat, expected) in enumerate(zip(features, expected_shapes)):
        assert feat.shape == expected, (
            f"Stage {i + 1}: expected {expected}, got {feat.shape}"
        )


def test_mlp_decoder_shape() -> None:
    """Verify MLPDecoder produces correct output shape."""
    decoder = MLPDecoder(
        embed_dims=[32, 64, 160, 256],
        decoder_dim=256,
        num_classes=8,
    )
    features = [
        torch.randn(2, 32, 128, 128),
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 160, 32, 32),
        torch.randn(2, 256, 16, 16),
    ]
    out = decoder(features)
    assert out.shape == (2, 8, 512, 512), f"Got {out.shape}"


def test_mit_configs_structure() -> None:
    """Verify MIT_CONFIGS has all required keys for each variant."""
    required_keys = [
        "embed_dims", "depths", "num_heads", "reduction_ratios",
        "mlp_ratios", "strides", "patch_sizes", "paddings",
    ]
    for variant, config in MIT_CONFIGS.items():
        for key in required_keys:
            assert key in config, (
                f"Variant {variant} missing key '{key}'"
            )
        assert len(config["embed_dims"]) == 4
        assert len(config["depths"]) == 4
        assert len(config["num_heads"]) == 4