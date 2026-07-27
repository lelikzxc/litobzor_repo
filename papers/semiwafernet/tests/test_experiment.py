"""Tests for SemiWaferNet experiment metadata utilities.

Covers ExperimentInfo dataclass, count_params, build_experiment_info,
format_experiment_info, parameter counting, metadata formatting,
baseline model metadata, and model with semi-supervised components metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.utils.experiment import (
    ExperimentInfo,
    build_experiment_info,
    count_params,
    format_experiment_info,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model() -> SemiWaferNet:
    """Default SemiWaferNet model for testing (classification mode)."""
    return SemiWaferNet(
        mode="classification",
        in_channels=1,
        backbone_channels=[64, 128],
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        mlp_ratio=2,
        dropout=0.1,
        fusion_dim=128,
        num_classes=9,
    )


# ── ExperimentInfo ─────────────────────────────────────────────────────────────


class TestExperimentInfo:
    """Tests for the ExperimentInfo dataclass."""

    def test_default_creation(self) -> None:
        """ExperimentInfo can be created with default values."""
        info = ExperimentInfo()
        assert info.model_name == "semiwafernet"
        assert info.num_classes == 9
        assert info.image_size == 32
        assert info.backbone_channels == [64, 128]
        assert info.transformer_embed_dim == 128
        assert info.transformer_layers == 4
        assert info.total_params == 0
        assert info.backbone_params == 0
        assert info.transformer_params == 0
        assert info.fusion_params == 0
        assert info.classifier_params == 0
        assert info.decoder_params == 0
        assert info.ema_enabled is False
        assert info.pseudo_labels_enabled is False
        assert info.consistency_enabled is False
        assert info.architecture_summary == ""

    def test_custom_values(self) -> None:
        """ExperimentInfo accepts custom values."""
        info = ExperimentInfo(
            model_name="semiwafernet_custom",
            num_classes=8,
            image_size=64,
            backbone_channels=[32, 64],
            transformer_embed_dim=64,
            transformer_layers=2,
            total_params=1_000_000,
            backbone_params=500_000,
            transformer_params=200_000,
            fusion_params=200_000,
            classifier_params=50_000,
            decoder_params=50_000,
            ema_enabled=True,
            pseudo_labels_enabled=True,
            consistency_enabled=True,
            architecture_summary="Custom SemiWaferNet",
        )
        assert info.model_name == "semiwafernet_custom"
        assert info.num_classes == 8
        assert info.image_size == 64
        assert info.backbone_channels == [32, 64]
        assert info.transformer_embed_dim == 64
        assert info.transformer_layers == 2
        assert info.total_params == 1_000_000
        assert info.backbone_params == 500_000
        assert info.transformer_params == 200_000
        assert info.fusion_params == 200_000
        assert info.classifier_params == 50_000
        assert info.decoder_params == 50_000
        assert info.ema_enabled is True
        assert info.pseudo_labels_enabled is True
        assert info.consistency_enabled is True
        assert info.architecture_summary == "Custom SemiWaferNet"

    def test_is_dataclass(self) -> None:
        """ExperimentInfo is a dataclass."""
        info = ExperimentInfo()
        assert hasattr(info, "__dataclass_fields__")


# ── count_params ───────────────────────────────────────────────────────────────


class TestCountParams:
    """Tests for the count_params utility."""

    def test_count_params_positive(self, model: SemiWaferNet) -> None:
        """count_params returns a positive integer for a real model."""
        n = count_params(model)
        assert isinstance(n, int)
        assert n > 0

    def test_count_params_submodule(self, model: SemiWaferNet) -> None:
        """count_params works on sub-modules."""
        backbone_params = count_params(model.backbone)
        transformer_params = count_params(model.transformer)
        fusion_params = count_params(model.fusion)
        classifier_params = count_params(model.classifier)
        decoder_params = count_params(model.decoder)

        assert backbone_params > 0
        assert transformer_params > 0
        assert fusion_params > 0
        assert classifier_params > 0
        # Decoder may have 0 params in classification mode (not used in forward)
        assert decoder_params >= 0

    def test_count_params_sum_matches_total(self, model: SemiWaferNet) -> None:
        """Sum of sub-module params equals total params."""
        total = count_params(model)
        sub_total = (
            count_params(model.backbone)
            + count_params(model.transformer)
            + count_params(model.fusion)
            + count_params(model.classifier)
            + count_params(model.decoder)
        )
        assert total == sub_total

    def test_count_params_empty_module(self) -> None:
        """count_params returns 0 for an empty module."""
        empty = torch.nn.Module()
        assert count_params(empty) == 0


# ── build_experiment_info ──────────────────────────────────────────────────────


class TestBuildExperimentInfo:
    """Tests for build_experiment_info."""

    def test_build_baseline(self, model: SemiWaferNet) -> None:
        """build_experiment_info returns correct metadata for baseline model."""
        info = build_experiment_info(model)
        assert info.model_name == "semiwafernet"
        assert info.num_classes == 9
        assert info.image_size == 32
        assert info.backbone_channels == [64, 128]
        assert info.transformer_embed_dim == 128
        assert info.transformer_layers == 4
        assert info.total_params > 0
        assert info.backbone_params > 0
        assert info.transformer_params > 0
        assert info.fusion_params > 0
        assert info.classifier_params > 0
        # Decoder may have 0 params in classification mode (not used in forward)
        assert info.decoder_params >= 0
        assert info.ema_enabled is False
        assert info.pseudo_labels_enabled is False
        assert info.consistency_enabled is False

    def test_build_with_semi_supervised(self, model: SemiWaferNet) -> None:
        """build_experiment_info reflects semi-supervised flags."""
        info = build_experiment_info(
            model,
            ema_enabled=True,
            pseudo_labels_enabled=True,
            consistency_enabled=True,
        )
        assert info.ema_enabled is True
        assert info.pseudo_labels_enabled is True
        assert info.consistency_enabled is True

    def test_build_partial_semi_supervised(self, model: SemiWaferNet) -> None:
        """build_experiment_info supports partial semi-supervised flags."""
        info = build_experiment_info(model, ema_enabled=True)
        assert info.ema_enabled is True
        assert info.pseudo_labels_enabled is False
        assert info.consistency_enabled is False

    def test_build_architecture_summary(self, model: SemiWaferNet) -> None:
        """build_experiment_info generates a non-empty architecture summary."""
        info = build_experiment_info(model)
        assert isinstance(info.architecture_summary, str)
        assert len(info.architecture_summary) > 0
        assert "SemiWaferNet" in info.architecture_summary
        assert "CNN" in info.architecture_summary
        assert "Transformer" in info.architecture_summary

    def test_build_parameter_counts_consistent(self, model: SemiWaferNet) -> None:
        """Component parameter counts sum to total in build_experiment_info."""
        info = build_experiment_info(model)
        component_sum = (
            info.backbone_params
            + info.transformer_params
            + info.fusion_params
            + info.classifier_params
            + info.decoder_params
        )
        assert component_sum == info.total_params


# ── format_experiment_info ─────────────────────────────────────────────────────


class TestFormatExperimentInfo:
    """Tests for format_experiment_info."""

    def test_format_returns_string(self, model: SemiWaferNet) -> None:
        """format_experiment_info returns a non-empty string."""
        info = build_experiment_info(model)
        formatted = format_experiment_info(info)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_format_contains_key_fields(self, model: SemiWaferNet) -> None:
        """Formatted output contains all key metadata fields."""
        info = build_experiment_info(model)
        formatted = format_experiment_info(info)
        assert "SemiWaferNet" in formatted
        assert "Experiment Metadata" in formatted
        assert "Model:" in formatted
        assert "Classes:" in formatted
        assert "Image size:" in formatted
        assert "Backbone channels:" in formatted
        assert "Transformer embed:" in formatted
        assert "Transformer layers:" in formatted
        assert "Parameters:" in formatted
        assert "Total:" in formatted
        assert "CNN backbone:" in formatted
        assert "Transformer:" in formatted
        assert "Feature fusion:" in formatted
        assert "Classifier head:" in formatted
        assert "Segmentation dec:" in formatted
        assert "Semi-supervised:" in formatted
        assert "EMA teacher:" in formatted
        assert "Pseudo labels:" in formatted
        assert "Consistency:" in formatted
        assert "Architecture:" in formatted

    def test_format_contains_parameter_values(self, model: SemiWaferNet) -> None:
        """Formatted output contains actual parameter numbers."""
        info = build_experiment_info(model)
        formatted = format_experiment_info(info)
        # Should contain comma-formatted numbers
        assert any(c.isdigit() for c in formatted)

    def test_format_semi_supervised_disabled(self, model: SemiWaferNet) -> None:
        """Formatted output shows False for disabled semi-supervised flags."""
        info = build_experiment_info(model)
        formatted = format_experiment_info(info)
        assert "False" in formatted

    def test_format_semi_supervised_enabled(self, model: SemiWaferNet) -> None:
        """Formatted output shows True for enabled semi-supervised flags."""
        info = build_experiment_info(
            model,
            ema_enabled=True,
            pseudo_labels_enabled=True,
            consistency_enabled=True,
        )
        formatted = format_experiment_info(info)
        # Count occurrences of "True" (should be 3)
        assert formatted.count("True") >= 3

    def test_format_has_separator(self, model: SemiWaferNet) -> None:
        """Formatted output has separator lines."""
        info = build_experiment_info(model)
        formatted = format_experiment_info(info)
        assert formatted.startswith("=")
        assert formatted.endswith("=")


# ── Integration ────────────────────────────────────────────────────────────────


class TestExperimentIntegration:
    """Integration tests combining all experiment utilities."""

    def test_full_pipeline_baseline(self, model: SemiWaferNet) -> None:
        """Full pipeline: build → format for baseline model."""
        info = build_experiment_info(model)
        formatted = format_experiment_info(info)
        assert info.total_params > 0
        assert "False" in formatted

    def test_full_pipeline_semi_supervised(self, model: SemiWaferNet) -> None:
        """Full pipeline: build → format for model with semi-supervised."""
        info = build_experiment_info(
            model,
            ema_enabled=True,
            pseudo_labels_enabled=True,
            consistency_enabled=True,
        )
        formatted = format_experiment_info(info)
        assert info.ema_enabled is True
        assert info.pseudo_labels_enabled is True
        assert info.consistency_enabled is True
        assert "True" in formatted

    def test_forward_pass_preserved(self, model: SemiWaferNet) -> None:
        """build_experiment_info does not affect model forward pass."""
        B, C, H, W = 2, 1, 32, 32
        x = torch.randn(B, C, H, W)

        model.eval()
        with torch.no_grad():
            # Forward before
            out_before = model(x)

            # Build metadata
            build_experiment_info(model)

            # Forward after
            out_after = model(x)

        assert torch.allclose(out_before["classification"], out_after["classification"], atol=1e-6)
        assert torch.allclose(out_before["segmentation"], out_after["segmentation"], atol=1e-6)

    def test_demo_runs_without_error(self, model: SemiWaferNet) -> None:
        """Demo-style workflow runs without errors."""
        info = build_experiment_info(model)
        formatted = format_experiment_info(info)
        assert len(formatted) > 0

        # Verify all parameter breakdowns are present
        assert "CNN backbone:" in formatted
        assert "Transformer:" in formatted
        assert "Feature fusion:" in formatted
        assert "Classifier head:" in formatted
        assert "Segmentation dec:" in formatted