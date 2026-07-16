"""Tests for SemiWaferNet baseline architecture.

Covers model creation, forward pass shapes, gradient flow,
parameter counts, and config-driven instantiation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.modules.cnn_backbone import CNNBackbone, ConvBlock, CNNStage
from papers.semiwafernet.modules.transformer import (
    TransformerEncoder,
    TransformerEncoderBlock,
    PatchProjection,
    MultiHeadSelfAttention,
    TransformerMLP,
)
from papers.semiwafernet.modules.fusion import FeatureFusion, ChannelAlign
from papers.semiwafernet.models.classifier import ClassifierHead
from papers.semiwafernet.models.decoder import SegmentationDecoder

from common.utils.config import Config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


# ── Imports ────────────────────────────────────────────────────────────────────


def test_import_modules() -> None:
    """All module classes are importable."""
    assert CNNBackbone is not None
    assert ConvBlock is not None
    assert CNNStage is not None
    assert TransformerEncoder is not None
    assert TransformerEncoderBlock is not None
    assert PatchProjection is not None
    assert MultiHeadSelfAttention is not None
    assert TransformerMLP is not None
    assert FeatureFusion is not None
    assert ChannelAlign is not None
    assert ClassifierHead is not None
    assert SegmentationDecoder is not None
    assert SemiWaferNet is not None


# ── Model creation ─────────────────────────────────────────────────────────────


def test_model_creation() -> None:
    """SemiWaferNet can be created with default parameters."""
    model = SemiWaferNet()
    assert isinstance(model, SemiWaferNet)
    assert hasattr(model, "backbone")
    assert hasattr(model, "transformer")
    assert hasattr(model, "fusion")
    assert hasattr(model, "classifier")
    assert hasattr(model, "decoder")


def test_model_creation_custom() -> None:
    """SemiWaferNet can be created with custom parameters."""
    model = SemiWaferNet(
        in_channels=1,
        backbone_channels=[32, 64, 128, 256],
        backbone_depths=[1, 1, 2, 1],
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2,
        dropout=0.2,
        fusion_dim=128,
        num_classes=3,
    )
    assert isinstance(model, SemiWaferNet)
    # Verify custom config propagated
    assert model.classifier.head.out_features == 3
    assert model.decoder.head.out_channels == 3


# ── Forward pass ───────────────────────────────────────────────────────────────


def test_forward_pass() -> None:
    """Forward pass returns dict with classification and segmentation keys."""
    model = SemiWaferNet()
    x = torch.randn(2, 3, 128, 128)
    output = model(x)
    assert isinstance(output, dict)
    assert "classification" in output
    assert "segmentation" in output


def test_classification_shape() -> None:
    """Classification output has shape [B, num_classes]."""
    B, num_classes = 4, 6
    model = SemiWaferNet(num_classes=num_classes)
    x = torch.randn(B, 3, 128, 128)
    output = model(x)
    assert output["classification"].shape == (B, num_classes)


def test_segmentation_shape() -> None:
    """Segmentation output has shape [B, num_classes, H, W]."""
    B, num_classes, H, W = 2, 6, 128, 128
    model = SemiWaferNet(num_classes=num_classes)
    x = torch.randn(B, 3, H, W)
    output = model(x)
    assert output["segmentation"].shape == (B, num_classes, H, W)


def test_segmentation_full_resolution() -> None:
    """Segmentation output matches input spatial resolution."""
    B, C, H, W = 1, 3, 512, 512
    model = SemiWaferNet()
    x = torch.randn(B, C, H, W)
    output = model(x)
    assert output["segmentation"].shape[2:] == (H, W)


# ── Gradients ──────────────────────────────────────────────────────────────────


def test_gradients_flow() -> None:
    """Gradients flow through all parameters."""
    model = SemiWaferNet()
    x = torch.randn(2, 3, 64, 64)
    output = model(x)

    loss = output["classification"].sum() + output["segmentation"].sum()
    loss.backward()

    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradients found — backward pass may be broken"

    # Verify all parameters have gradients
    all_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert all_grad, "Not all parameters received gradients"


# ── Parameter count ────────────────────────────────────────────────────────────


def test_parameter_count() -> None:
    """Model has a reasonable number of parameters."""
    model = SemiWaferNet()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Expect roughly 10-30M parameters for the default config
    assert 5_000_000 < total < 50_000_000, f"Unexpected parameter count: {total:,}"


# ── from_config ────────────────────────────────────────────────────────────────


def test_from_config() -> None:
    """SemiWaferNet can be built from config file."""
    config = Config.from_yaml(CONFIG_PATH)
    model = SemiWaferNet.from_config(config)
    assert isinstance(model, SemiWaferNet)

    # Verify config values propagated
    assert model.classifier.head.out_features == 6
    assert model.decoder.head.out_channels == 6


def test_from_config_custom_values() -> None:
    """from_config respects custom config values."""
    config = Config.from_yaml(CONFIG_PATH)
    # Override via raw dict
    config.raw["model"]["num_classes"] = 3
    config.raw["model"]["backbone"]["channels"] = [32, 64, 128, 256]
    config.raw["model"]["transformer"]["embed_dim"] = 128
    config.raw["model"]["transformer"]["num_layers"] = 2

    model = SemiWaferNet.from_config(config)
    assert model.classifier.head.out_features == 3
    assert model.decoder.head.out_channels == 3


# ── Module unit tests ──────────────────────────────────────────────────────────


class TestCNNBackbone:
    """Unit tests for CNN backbone components."""

    def test_conv_block_shape(self) -> None:
        block = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_cnn_stage_shape(self) -> None:
        stage = CNNStage(64, 128, depth=2, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = stage(x)
        assert out.shape == (2, 128, 16, 16)

    def test_backbone_shapes(self) -> None:
        backbone = CNNBackbone(in_channels=3, channels=[64, 128, 256, 512], depths=[2, 2, 6, 2])
        x = torch.randn(2, 3, 256, 256)
        features = backbone(x)
        assert len(features) == 4
        expected = [
            (2, 64, 64, 64),    # H/4, W/4
            (2, 128, 32, 32),   # H/8, W/8
            (2, 256, 16, 16),   # H/16, W/16
            (2, 512, 8, 8),     # H/32, W/32
        ]
        for feat, exp in zip(features, expected):
            assert feat.shape == exp, f"Expected {exp}, got {feat.shape}"


class TestTransformer:
    """Unit tests for transformer encoder components."""

    def test_patch_projection_shape(self) -> None:
        proj = PatchProjection(in_channels=512, embed_dim=256)
        x = torch.randn(2, 512, 8, 8)
        tokens, (H, W) = proj(x)
        assert tokens.shape == (2, 64, 256)  # B, H*W, embed_dim
        assert H == 8
        assert W == 8

    def test_attention_shape(self) -> None:
        attn = MultiHeadSelfAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 64, 256)
        out = attn(x)
        assert out.shape == (2, 64, 256)

    def test_mlp_shape(self) -> None:
        mlp = TransformerMLP(embed_dim=256, mlp_ratio=4)
        x = torch.randn(2, 64, 256)
        out = mlp(x)
        assert out.shape == (2, 64, 256)

    def test_encoder_block_shape(self) -> None:
        block = TransformerEncoderBlock(embed_dim=256, num_heads=8, mlp_ratio=4)
        x = torch.randn(2, 64, 256)
        out = block(x)
        assert out.shape == (2, 64, 256)

    def test_encoder_shape(self) -> None:
        encoder = TransformerEncoder(in_channels=512, embed_dim=256, num_heads=8, num_layers=4)
        x = torch.randn(2, 512, 8, 8)
        tokens, (H, W) = encoder(x)
        assert tokens.shape == (2, 64, 256)
        assert H == 8
        assert W == 8


class TestFusion:
    """Unit tests for feature fusion module."""

    def test_channel_align_shape(self) -> None:
        align = ChannelAlign(512, 256)
        x = torch.randn(2, 512, 16, 16)
        out = align(x)
        assert out.shape == (2, 256, 16, 16)

    def test_fusion_shape(self) -> None:
        fusion = FeatureFusion(
            cnn_channels=[64, 128, 256, 512],
            transformer_dim=256,
            fusion_dim=256,
        )
        cnn_features = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 256, 8, 8),
            torch.randn(2, 512, 4, 4),
        ]
        transformer_tokens = torch.randn(2, 16, 256)  # 4x4 = 16 tokens
        class_feat, seg_feat = fusion(cnn_features, transformer_tokens, (4, 4))
        assert class_feat.shape == (2, 256, 32, 32)
        assert seg_feat.shape == (2, 256, 32, 32)


class TestClassifier:
    """Unit tests for classification head."""

    def test_classifier_shape(self) -> None:
        classifier = ClassifierHead(in_channels=256, num_classes=6)
        x = torch.randn(2, 256, 32, 32)
        out = classifier(x)
        assert out.shape == (2, 6)


class TestDecoder:
    """Unit tests for segmentation decoder."""

    def test_decoder_shape(self) -> None:
        decoder = SegmentationDecoder(in_channels=256, num_classes=6)
        x = torch.randn(2, 256, 32, 32)
        out = decoder(x)
        assert out.shape == (2, 6, 128, 128)  # ×4 upsample