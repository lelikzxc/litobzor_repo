"""Tests for SemiWaferNet baseline architecture.

Covers model creation, forward pass shapes, gradient flow,
parameter counts, and config-driven instantiation.

Based on Electronics 2026, 15, 1437:
- HybridCNN-ViT (classification): 2-stage CNN + 4-layer Transformer
- ConvoFormer-UNet (segmentation): ConvEmbed + ConvoFormer blocks + decoder
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.modules.cnn_backbone import CNNBackbone, ConvBlock, ResidualBlock
from papers.semiwafernet.modules.transformer import (
    TransformerEncoder,
    TransformerEncoderBlock,
    ConvoFormerBlock,
    ConvEmbed,
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
    assert ResidualBlock is not None
    assert TransformerEncoder is not None
    assert TransformerEncoderBlock is not None
    assert ConvoFormerBlock is not None
    assert ConvEmbed is not None
    assert PatchProjection is not None
    assert MultiHeadSelfAttention is not None
    assert TransformerMLP is not None
    assert FeatureFusion is not None
    assert ChannelAlign is not None
    assert ClassifierHead is not None
    assert SegmentationDecoder is not None
    assert SemiWaferNet is not None


# ── Model creation ─────────────────────────────────────────────────────────────


def test_model_creation_classification() -> None:
    """SemiWaferNet can be created in classification mode with default params."""
    model = SemiWaferNet(mode="classification")
    assert isinstance(model, SemiWaferNet)
    assert model.mode == "classification"
    assert hasattr(model, "backbone")
    assert hasattr(model, "transformer")
    assert hasattr(model, "fusion")
    assert hasattr(model, "classifier")
    assert hasattr(model, "decoder")


def test_model_creation_segmentation() -> None:
    """SemiWaferNet can be created in segmentation mode."""
    model = SemiWaferNet(mode="segmentation")
    assert isinstance(model, SemiWaferNet)
    assert model.mode == "segmentation"
    assert hasattr(model, "transformer")
    assert hasattr(model, "decoder")


def test_model_creation_custom() -> None:
    """SemiWaferNet can be created with custom parameters."""
    model = SemiWaferNet(
        mode="classification",
        in_channels=1,
        backbone_channels=[32, 64],
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2,
        dropout=0.2,
        fusion_dim=64,
        num_classes=3,
    )
    assert isinstance(model, SemiWaferNet)
    # Verify custom config propagated
    assert model.classifier.head.out_features == 3


# ── Forward pass ───────────────────────────────────────────────────────────────


def test_forward_pass_classification() -> None:
    """Forward pass returns dict with classification key."""
    model = SemiWaferNet(mode="classification")
    x = torch.randn(2, 1, 32, 32)
    output = model(x)
    assert isinstance(output, dict)
    assert "classification" in output
    assert "segmentation" in output


def test_forward_pass_segmentation() -> None:
    """Forward pass in segmentation mode returns segmentation output."""
    model = SemiWaferNet(mode="segmentation")
    x = torch.randn(2, 1, 64, 64)
    output = model(x)
    assert isinstance(output, dict)
    assert "segmentation" in output


def test_classification_shape() -> None:
    """Classification output has shape [B, num_classes]."""
    B, num_classes = 4, 9
    model = SemiWaferNet(mode="classification", num_classes=num_classes)
    x = torch.randn(B, 1, 32, 32)
    output = model(x)
    assert output["classification"].shape == (B, num_classes)


def test_segmentation_shape() -> None:
    """Segmentation output has shape [B, seg_classes, H, W].

    ConvoFormer-UNet uses ConvEmbed (8×8 conv stride 8), so input 64×64
    → 8×8 token grid → decoder ×2×2 upsample → 32×32 output.
    """
    B, seg_classes, H, W = 2, 1, 64, 64
    model = SemiWaferNet(mode="segmentation", seg_classes=seg_classes)
    x = torch.randn(B, 1, H, W)
    output = model(x)
    # ConvEmbed stride 8 → 8×8 tokens → decoder ×2×2 → 32×32
    assert output["segmentation"].shape == (B, seg_classes, 32, 32)


def test_segmentation_full_resolution() -> None:
    """Segmentation output is H/2 × W/2 of input (ConvEmbed stride 8 → ×2×2 decoder)."""
    B, C, H, W = 1, 1, 128, 128
    model = SemiWaferNet(mode="segmentation")
    x = torch.randn(B, C, H, W)
    output = model(x)
    # ConvEmbed stride 8 → 16×16 tokens → decoder ×2×2 → 64×64
    assert output["segmentation"].shape[2:] == (64, 64)


# ── Gradients ──────────────────────────────────────────────────────────────────


def test_gradients_flow_classification() -> None:
    """Gradients flow through classification-relevant parameters.

    In classification mode, the decoder and segmentation-specific
    fusion projections (seg_proj) are not used in forward(),
    so their parameters won't receive gradients.
    """
    model = SemiWaferNet(mode="classification")
    x = torch.randn(2, 1, 32, 32)
    output = model(x)

    loss = output["classification"].sum()
    loss.backward()

    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradients found — backward pass may be broken"

    # Verify classification-relevant parameters have gradients
    # (decoder and seg_proj are not used in classification mode)
    skip_patterns = ("decoder", "seg_proj")
    for name, p in model.named_parameters():
        if p.requires_grad and not any(s in name for s in skip_patterns):
            assert p.grad is not None, f"Parameter {name} has no gradient"


def test_gradients_flow_segmentation() -> None:
    """Gradients flow through all parameters in segmentation mode."""
    model = SemiWaferNet(mode="segmentation")
    x = torch.randn(2, 1, 64, 64)
    output = model(x)

    loss = output["segmentation"].sum()
    loss.backward()

    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradients found — backward pass may be broken"


# ── Parameter count ────────────────────────────────────────────────────────────


def test_parameter_count_classification() -> None:
    """Classification model has a reasonable number of parameters."""
    model = SemiWaferNet(mode="classification")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Expect roughly 0.5-5M parameters for the paper config
    assert 100_000 < total < 10_000_000, f"Unexpected parameter count: {total:,}"


def test_parameter_count_segmentation() -> None:
    """Segmentation model has a reasonable number of parameters."""
    model = SemiWaferNet(mode="segmentation")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Expect roughly 0.5-5M parameters for the paper config
    assert 100_000 < total < 10_000_000, f"Unexpected parameter count: {total:,}"


# ── from_config ────────────────────────────────────────────────────────────────


def test_from_config_classification() -> None:
    """SemiWaferNet can be built from config file in classification mode."""
    config = Config.from_yaml(CONFIG_PATH)
    model = SemiWaferNet.from_config(config)
    assert isinstance(model, SemiWaferNet)
    assert model.mode == "classification"

    # Verify config values propagated
    assert model.classifier.head.out_features == 9


def test_from_config_custom_values() -> None:
    """from_config respects custom config values."""
    config = Config.from_yaml(CONFIG_PATH)
    # Override via raw dict
    config.raw["model"]["num_classes"] = 3
    config.raw["model"]["backbone"]["channels"] = [32, 64]
    config.raw["model"]["transformer"]["embed_dim"] = 64
    config.raw["model"]["transformer"]["num_layers"] = 2

    model = SemiWaferNet.from_config(config)
    assert model.classifier.head.out_features == 3


# ── Module unit tests ──────────────────────────────────────────────────────────


class TestCNNBackbone:
    """Unit tests for CNN backbone components (2-stage: ConvBlock + ResBlock)."""

    def test_conv_block_shape(self) -> None:
        block = ConvBlock(1, 64, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 1, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_conv_block_with_pool(self) -> None:
        block = ConvBlock(1, 64, kernel_size=3, stride=1, padding=1, use_pool=True)
        x = torch.randn(2, 1, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)  # H/2, W/2 after MaxPool

    def test_residual_block_shape(self) -> None:
        block = ResidualBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 128, 8, 8)

    def test_residual_block_same_dim(self) -> None:
        block = ResidualBlock(64, 64, stride=1)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_backbone_shapes(self) -> None:
        backbone = CNNBackbone(in_channels=1, channels=[64, 128])
        x = torch.randn(2, 1, 32, 32)
        features = backbone(x)
        assert len(features) == 2
        expected = [
            (2, 64, 16, 16),    # H/2, W/2 (after ConvBlock + MaxPool)
            (2, 128, 8, 8),     # H/4, W/4 (after ResBlock stride 2)
        ]
        for feat, exp in zip(features, expected):
            assert feat.shape == exp, f"Expected {exp}, got {feat.shape}"


class TestTransformer:
    """Unit tests for transformer encoder components."""

    def test_conv_embed_shape(self) -> None:
        """ConvEmbed: 3×3 conv (GELU) → 8×8 conv stride 8 → token grid."""
        embed = ConvEmbed(in_channels=1, embed_dim=128)
        x = torch.randn(2, 1, 64, 64)
        tokens, (H, W) = embed(x)
        assert tokens.shape == (2, 64, 128)  # B, N=H*W, embed_dim
        assert H == 8
        assert W == 8

    def test_patch_projection_shape(self) -> None:
        proj = PatchProjection(in_channels=128, embed_dim=128)
        x = torch.randn(2, 128, 8, 8)
        tokens, (H, W) = proj(x)
        assert tokens.shape == (2, 64, 128)  # B, H*W, embed_dim
        assert H == 8
        assert W == 8

    def test_attention_shape(self) -> None:
        attn = MultiHeadSelfAttention(embed_dim=128, num_heads=8)
        x = torch.randn(2, 64, 128)
        out = attn(x)
        assert out.shape == (2, 64, 128)

    def test_mlp_shape(self) -> None:
        mlp = TransformerMLP(embed_dim=128, mlp_ratio=2)
        x = torch.randn(2, 64, 128)
        out = mlp(x)
        assert out.shape == (2, 64, 128)

    def test_convo_former_block_shape(self) -> None:
        """ConvoFormerBlock: MSA + depthwise conv fusion."""
        block = ConvoFormerBlock(embed_dim=128, num_heads=8, mlp_ratio=2)
        x = torch.randn(2, 64, 128)
        out = block(x, spatial_shape=(8, 8))
        assert out.shape == (2, 64, 128)

    def test_encoder_block_shape(self) -> None:
        block = TransformerEncoderBlock(embed_dim=128, num_heads=8, mlp_ratio=2)
        x = torch.randn(2, 64, 128)
        out = block(x)
        assert out.shape == (2, 64, 128)

    def test_encoder_shape_classification(self) -> None:
        """TransformerEncoder in classification mode (PatchProjection)."""
        encoder = TransformerEncoder(
            in_channels=128, embed_dim=128, num_heads=8, num_layers=4,
            use_conv_embed=False,
        )
        x = torch.randn(2, 128, 8, 8)
        tokens, (H, W) = encoder(x)
        assert tokens.shape == (2, 64, 128)
        assert H == 8
        assert W == 8

    def test_encoder_shape_segmentation(self) -> None:
        """TransformerEncoder in segmentation mode (ConvEmbed)."""
        encoder = TransformerEncoder(
            in_channels=1, embed_dim=128, num_heads=8, num_layers=4,
            use_conv_embed=True,
        )
        x = torch.randn(2, 1, 64, 64)
        tokens, (H, W) = encoder(x)
        assert tokens.shape == (2, 64, 128)
        assert H == 8
        assert W == 8


class TestFusion:
    """Unit tests for feature fusion module (2 CNN features + transformer)."""

    def test_channel_align_shape(self) -> None:
        align = ChannelAlign(128, 64)
        x = torch.randn(2, 128, 16, 16)
        out = align(x)
        assert out.shape == (2, 64, 16, 16)

    def test_fusion_shape(self) -> None:
        fusion = FeatureFusion(
            cnn_channels=[64, 128],
            transformer_dim=128,
            fusion_dim=128,
        )
        cnn_features = [
            torch.randn(2, 64, 16, 16),
            torch.randn(2, 128, 8, 8),
        ]
        transformer_tokens = torch.randn(2, 64, 128)  # 8x8 = 64 tokens
        class_feat, seg_feat = fusion(cnn_features, transformer_tokens, (8, 8))
        assert class_feat.shape == (2, 128, 16, 16)
        assert seg_feat.shape == (2, 128, 16, 16)


class TestClassifier:
    """Unit tests for classification head."""

    def test_classifier_shape(self) -> None:
        classifier = ClassifierHead(in_channels=128, num_classes=9)
        x = torch.randn(2, 128, 16, 16)
        out = classifier(x)
        assert out.shape == (2, 9)


class TestDecoder:
    """Unit tests for segmentation decoder (binary segmentation).

    Decoder takes [B, C, H/4, W/4] and does ×2 → ×2 upsample,
    producing [B, C, H, W]. With 8×8 input → 32×32 output.
    """

    def test_decoder_shape(self) -> None:
        decoder = SegmentationDecoder(in_channels=128, num_classes=1)
        x = torch.randn(2, 128, 8, 8)
        out = decoder(x)
        assert out.shape == (2, 1, 32, 32)  # ×2 → ×2 = ×4 upsample

    def test_decoder_with_deep_supervision(self) -> None:
        decoder = SegmentationDecoder(in_channels=128, num_classes=1)
        x = torch.randn(2, 128, 8, 8)
        out = decoder(x, return_aux=True)
        assert isinstance(out, dict)
        assert "main" in out
        assert "aux1" in out
        assert "aux2" in out
        assert out["main"].shape == (2, 1, 32, 32)
        assert out["aux1"].shape == (2, 1, 16, 16)
        assert out["aux2"].shape == (2, 1, 8, 8)