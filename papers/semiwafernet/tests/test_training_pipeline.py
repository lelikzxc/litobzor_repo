"""Tests for SemiWaferNet training pipeline components.

Covers:
- MonteCarloDropout: forward pass, shapes, entropy, mutual information
- AdaptiveThreshold: statistics update, threshold computation, reset
- UncertaintyFilter: classification/segmentation filtering, combined criteria
- StageManager: stage transitions, pseudo-label generation, consistency
- Trainer: stage training, optimizer/loss setup, error handling
- Config: new config parameters
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.training.mc_dropout import MonteCarloDropout
from papers.semiwafernet.training.adaptive_threshold import AdaptiveThreshold
from papers.semiwafernet.training.uncertainty import UncertaintyFilter
from papers.semiwafernet.training.stage_manager import StageManager
from papers.semiwafernet.training.trainer import Trainer
from common.utils.config import Config

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


@pytest.fixture
def small_input() -> torch.Tensor:
    """Small input tensor for fast tests (grayscale, 32x32)."""
    return torch.randn(2, 1, 32, 32)


@pytest.fixture
def mc_dropout() -> MonteCarloDropout:
    """Default MC Dropout instance."""
    return MonteCarloDropout(num_passes=5)


@pytest.fixture
def adaptive_threshold() -> AdaptiveThreshold:
    """Default AdaptiveThreshold instance."""
    return AdaptiveThreshold(num_classes=9, base_threshold=0.9, alpha=0.1, beta=0.05)


@pytest.fixture
def uncertainty_filter() -> UncertaintyFilter:
    """Default UncertaintyFilter instance."""
    return UncertaintyFilter(entropy_threshold=0.5, mi_threshold=0.3)


@pytest.fixture
def stage_manager(model: SemiWaferNet) -> StageManager:
    """Default StageManager instance."""
    return StageManager(
        student=model,
        num_classes=9,
        ema_decay=0.999,
        base_threshold=0.9,
        alpha=0.1,
        beta=0.05,
        mc_passes=5,
        entropy_threshold=0.5,
        mi_threshold=0.3,
        consistency_weight=0.1,
    )


@pytest.fixture
def trainer(model: SemiWaferNet, stage_manager: StageManager) -> Trainer:
    """Default Trainer instance."""
    return Trainer(student=model, stage_manager=stage_manager)


# ── MonteCarloDropout ─────────────────────────────────────────────────────────


class TestMonteCarloDropout:
    """Tests for MonteCarloDropout."""

    def test_creation(self) -> None:
        """MonteCarloDropout can be created with default params."""
        mcd = MonteCarloDropout()
        assert mcd.num_passes == 20

    def test_creation_custom(self) -> None:
        """MonteCarloDropout accepts custom num_passes."""
        mcd = MonteCarloDropout(num_passes=10)
        assert mcd.num_passes == 10

    def test_forward_returns_dict(self, mc_dropout: MonteCarloDropout, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """Forward returns a dictionary with expected keys."""
        model.eval()
        result = mc_dropout(model, small_input)
        assert isinstance(result, dict)
        expected_keys = {
            "mean_probs_class", "mean_probs_seg",
            "entropy_class", "entropy_seg",
            "mutual_info_class", "mutual_info_seg",
        }
        assert set(result.keys()) == expected_keys

    def test_mean_probs_class_shape(self, mc_dropout: MonteCarloDropout, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """mean_probs_class has shape [B, num_classes]."""
        model.eval()
        result = mc_dropout(model, small_input)
        assert result["mean_probs_class"].shape == (2, 9)

    def test_mean_probs_seg_shape(self, mc_dropout: MonteCarloDropout, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """mean_probs_seg has shape [B, seg_classes, H, W]."""
        model.eval()
        result = mc_dropout(model, small_input)
        assert result["mean_probs_seg"].shape == (2, 1, 32, 32)

    def test_entropy_class_shape(self, mc_dropout: MonteCarloDropout, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """entropy_class has shape [B]."""
        model.eval()
        result = mc_dropout(model, small_input)
        assert result["entropy_class"].shape == (2,)

    def test_entropy_seg_shape(self, mc_dropout: MonteCarloDropout, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """entropy_seg has shape [B, H, W]."""
        model.eval()
        result = mc_dropout(model, small_input)
        assert result["entropy_seg"].shape == (2, 32, 32)

    def test_mutual_info_class_shape(self, mc_dropout: MonteCarloDropout, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """mutual_info_class has shape [B]."""
        model.eval()
        result = mc_dropout(model, small_input)
        assert result["mutual_info_class"].shape == (2,)

    def test_mutual_info_seg_shape(self, mc_dropout: MonteCarloDropout, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """mutual_info_seg has shape [B, H, W]."""
        model.eval()
        result = mc_dropout(model, small_input)
        assert result["mutual_info_seg"].shape == (2, 32, 32)

    def test_mean_probs_sum_to_one(self, mc_dropout: MonteCarloDropout, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """Mean probabilities sum to 1 along class dimension."""
        model.eval()
        result = mc_dropout(model, small_input)
        assert torch.allclose(result["mean_probs_class"].sum(dim=1), torch.ones(2), atol=1e-5)

    def test_entropy_non_negative(self, mc_dropout: MonteCarloDropout, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """Entropy is non-negative."""
        model.eval()
        result = mc_dropout(model, small_input)
        assert (result["entropy_class"] >= 0).all()
        assert (result["entropy_seg"] >= 0).all()

    def test_mutual_info_non_negative(self, mc_dropout: MonteCarloDropout, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """Mutual information is non-negative."""
        model.eval()
        result = mc_dropout(model, small_input)
        assert (result["mutual_info_class"] >= -1e-6).all()
        assert (result["mutual_info_seg"] >= -1e-6).all()

    def test_restores_eval_mode(self, mc_dropout: MonteCarloDropout, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """Model is restored to eval mode after MC Dropout."""
        model.eval()
        mc_dropout(model, small_input)
        assert not model.training

    def test_num_passes_affects_result(self, model: SemiWaferNet, small_input: torch.Tensor) -> None:
        """Different num_passes produces different results."""
        model.eval()
        mcd1 = MonteCarloDropout(num_passes=2)
        mcd2 = MonteCarloDropout(num_passes=20)
        r1 = mcd1(model, small_input)
        r2 = mcd2(model, small_input)
        # Mean probs should differ with different number of passes
        assert not torch.allclose(r1["mean_probs_class"], r2["mean_probs_class"], atol=1e-3)


# ── AdaptiveThreshold ─────────────────────────────────────────────────────────


class TestAdaptiveThreshold:
    """Tests for AdaptiveThreshold."""

    def test_creation(self) -> None:
        """AdaptiveThreshold can be created with default params."""
        at = AdaptiveThreshold(num_classes=9)
        assert at.num_classes == 9
        assert at.base_threshold == 0.9
        assert at.alpha == 0.1
        assert at.beta == 0.05

    def test_creation_custom(self) -> None:
        """AdaptiveThreshold accepts custom params."""
        at = AdaptiveThreshold(num_classes=10, base_threshold=0.8, alpha=0.2, beta=0.1)
        assert at.num_classes == 10
        assert at.base_threshold == 0.8
        assert at.alpha == 0.2
        assert at.beta == 0.1

    def test_initial_statistics(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """Initial statistics are zeros/ones."""
        assert (adaptive_threshold.class_mean == 0).all()
        assert (adaptive_threshold.class_std == 1).all()
        assert (adaptive_threshold.class_count == 0).all()

    def test_update_statistics_classification(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """update_statistics works with 1D confidence and labels."""
        confidence = torch.tensor([0.95, 0.85, 0.92, 0.78, 0.99])
        pseudo_labels = torch.tensor([0, 1, 0, 2, 1])
        adaptive_threshold.update_statistics(confidence, pseudo_labels)
        assert adaptive_threshold.class_count[0].item() == 2
        assert adaptive_threshold.class_count[1].item() == 2
        assert adaptive_threshold.class_count[2].item() == 1

    def test_update_statistics_segmentation(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """update_statistics works with 2D confidence and labels."""
        confidence = torch.tensor([[0.9, 0.8], [0.7, 0.95]])
        pseudo_labels = torch.tensor([[0, 1], [2, 0]])
        adaptive_threshold.update_statistics(confidence, pseudo_labels)
        assert adaptive_threshold.class_count[0].item() == 2
        assert adaptive_threshold.class_count[1].item() == 1
        assert adaptive_threshold.class_count[2].item() == 1

    def test_update_statistics_with_mask(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """update_statistics respects mask."""
        confidence = torch.tensor([0.95, 0.85, 0.92])
        pseudo_labels = torch.tensor([0, 1, 0])
        mask = torch.tensor([True, False, True])
        adaptive_threshold.update_statistics(confidence, pseudo_labels, mask=mask)
        assert adaptive_threshold.class_count[0].item() == 2
        assert adaptive_threshold.class_count[1].item() == 0

    def test_update_statistics_empty(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """update_statistics handles empty input."""
        confidence = torch.tensor([])
        pseudo_labels = torch.tensor([], dtype=torch.long)
        adaptive_threshold.update_statistics(confidence, pseudo_labels)
        assert (adaptive_threshold.class_count == 0).all()

    def test_compute_threshold_default(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """compute_threshold returns base_threshold when no statistics."""
        tau = adaptive_threshold.compute_threshold()
        assert tau.item() == pytest.approx(0.9, abs=1e-6)

    def test_compute_threshold_with_statistics(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """compute_threshold adjusts based on statistics."""
        # Add high-confidence predictions
        confidence = torch.tensor([0.95, 0.98, 0.97, 0.96, 0.99])
        pseudo_labels = torch.tensor([0, 1, 0, 1, 0])
        adaptive_threshold.update_statistics(confidence, pseudo_labels)
        tau = adaptive_threshold.compute_threshold()
        # Low CV → threshold close to base
        assert tau.item() > 0.85
        assert tau.item() <= 1.0

    def test_compute_threshold_with_entropy(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """compute_threshold incorporates entropy bonus."""
        tau_no_entropy = adaptive_threshold.compute_threshold(entropy=None)
        tau_with_entropy = adaptive_threshold.compute_threshold(
            entropy=torch.tensor([0.1, 0.2])
        )
        # Low entropy → bonus → higher threshold
        assert tau_with_entropy.item() >= tau_no_entropy.item()

    def test_compute_threshold_clamped(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """compute_threshold clamps to [0, 1]."""
        adaptive_threshold.alpha = 100.0
        adaptive_threshold.beta = 100.0
        confidence = torch.tensor([0.5, 0.6])
        pseudo_labels = torch.tensor([0, 1])
        adaptive_threshold.update_statistics(confidence, pseudo_labels)
        tau = adaptive_threshold.compute_threshold()
        assert 0.0 <= tau.item() <= 1.0

    def test_get_threshold_value(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """get_threshold_value returns a float."""
        val = adaptive_threshold.get_threshold_value()
        assert isinstance(val, float)

    def test_reset(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """reset clears all statistics."""
        confidence = torch.tensor([0.95, 0.98])
        pseudo_labels = torch.tensor([0, 1])
        adaptive_threshold.update_statistics(confidence, pseudo_labels)
        assert adaptive_threshold.class_count[0].item() > 0
        adaptive_threshold.reset()
        assert (adaptive_threshold.class_count == 0).all()
        assert (adaptive_threshold.class_mean == 0).all()

    def test_multiple_updates(self, adaptive_threshold: AdaptiveThreshold) -> None:
        """Multiple updates accumulate statistics correctly."""
        for _ in range(3):
            confidence = torch.tensor([0.9, 0.8])
            pseudo_labels = torch.tensor([0, 1])
            adaptive_threshold.update_statistics(confidence, pseudo_labels)
        assert adaptive_threshold.class_count[0].item() == 3
        assert adaptive_threshold.class_count[1].item() == 3


# ── UncertaintyFilter ─────────────────────────────────────────────────────────


class TestUncertaintyFilter:
    """Tests for UncertaintyFilter."""

    def test_creation(self) -> None:
        """UncertaintyFilter can be created with default params."""
        uf = UncertaintyFilter()
        assert uf.entropy_threshold == 0.5
        assert uf.mi_threshold == 0.3

    def test_creation_custom(self) -> None:
        """UncertaintyFilter accepts custom params."""
        uf = UncertaintyFilter(entropy_threshold=0.3, mi_threshold=0.1)
        assert uf.entropy_threshold == 0.3
        assert uf.mi_threshold == 0.1

    def test_filter_classification_all_pass(self, uncertainty_filter: UncertaintyFilter) -> None:
        """All criteria pass for high-confidence, low-uncertainty predictions."""
        confidence = torch.tensor([0.95, 0.98])
        entropy = torch.tensor([0.1, 0.2])
        mi = torch.tensor([0.05, 0.1])
        mask = uncertainty_filter.filter_classification(confidence, 0.9, entropy, mi)
        assert mask.tolist() == [True, True]

    def test_filter_classification_confidence_fail(self, uncertainty_filter: UncertaintyFilter) -> None:
        """Low confidence predictions are rejected."""
        confidence = torch.tensor([0.5, 0.98])
        entropy = torch.tensor([0.1, 0.2])
        mi = torch.tensor([0.05, 0.1])
        mask = uncertainty_filter.filter_classification(confidence, 0.9, entropy, mi)
        assert mask.tolist() == [False, True]

    def test_filter_classification_entropy_fail(self, uncertainty_filter: UncertaintyFilter) -> None:
        """High entropy predictions are rejected."""
        confidence = torch.tensor([0.95, 0.98])
        entropy = torch.tensor([0.6, 0.2])
        mi = torch.tensor([0.05, 0.1])
        mask = uncertainty_filter.filter_classification(confidence, 0.9, entropy, mi)
        assert mask.tolist() == [False, True]

    def test_filter_classification_mi_fail(self, uncertainty_filter: UncertaintyFilter) -> None:
        """High mutual information predictions are rejected."""
        confidence = torch.tensor([0.95, 0.98])
        entropy = torch.tensor([0.1, 0.2])
        mi = torch.tensor([0.5, 0.1])
        mask = uncertainty_filter.filter_classification(confidence, 0.9, entropy, mi)
        assert mask.tolist() == [False, True]

    def test_filter_classification_all_fail(self, uncertainty_filter: UncertaintyFilter) -> None:
        """All predictions rejected when all criteria fail."""
        confidence = torch.tensor([0.5, 0.6])
        entropy = torch.tensor([0.6, 0.7])
        mi = torch.tensor([0.5, 0.6])
        mask = uncertainty_filter.filter_classification(confidence, 0.9, entropy, mi)
        assert mask.tolist() == [False, False]

    def test_filter_segmentation_shape(self, uncertainty_filter: UncertaintyFilter) -> None:
        """Segmentation filter preserves spatial shape."""
        confidence = torch.randn(2, 8, 8).abs()
        entropy = torch.rand(2, 8, 8) * 0.3
        mi = torch.rand(2, 8, 8) * 0.2
        mask = uncertainty_filter.filter_segmentation(confidence, 0.9, entropy, mi)
        assert mask.shape == (2, 8, 8)

    def test_filter_segmentation_values(self, uncertainty_filter: UncertaintyFilter) -> None:
        """Segmentation filter produces boolean mask."""
        confidence = torch.tensor([[[0.95, 0.5], [0.8, 0.99]]])
        entropy = torch.tensor([[[0.1, 0.6], [0.4, 0.2]]])
        mi = torch.tensor([[[0.05, 0.5], [0.3, 0.1]]])
        mask = uncertainty_filter.filter_segmentation(confidence, 0.9, entropy, mi)
        assert mask.dtype == torch.bool

    def test_forward_dict(self, uncertainty_filter: UncertaintyFilter) -> None:
        """Forward returns dict with classification and segmentation masks."""
        result = uncertainty_filter(
            confidence_class=torch.tensor([0.95, 0.5]),
            confidence_seg=torch.tensor([[[0.95]]]),
            adaptive_threshold=0.9,
            entropy_class=torch.tensor([0.1, 0.6]),
            entropy_seg=torch.tensor([[[0.1]]]),
            mutual_info_class=torch.tensor([0.05, 0.5]),
            mutual_info_seg=torch.tensor([[[0.05]]]),
        )
        assert "classification" in result
        assert "segmentation" in result

    def test_adaptive_threshold_tensor(self, uncertainty_filter: UncertaintyFilter) -> None:
        """Adaptive threshold can be a tensor."""
        confidence = torch.tensor([0.95, 0.5])
        entropy = torch.tensor([0.1, 0.6])
        mi = torch.tensor([0.05, 0.5])
        mask = uncertainty_filter.filter_classification(
            confidence, torch.tensor(0.9), entropy, mi
        )
        assert mask.tolist() == [True, False]


# ── StageManager ──────────────────────────────────────────────────────────────


class TestStageManager:
    """Tests for StageManager."""

    def test_creation(self, stage_manager: StageManager) -> None:
        """StageManager can be created with default params."""
        assert stage_manager.current_stage == 1
        assert stage_manager.num_classes == 9
        assert stage_manager.teacher is not None
        assert stage_manager.adaptive_threshold is not None
        assert stage_manager.mc_dropout is not None
        assert stage_manager.uncertainty_filter is not None
        assert stage_manager.consistency_loss is not None

    def test_set_stage(self, stage_manager: StageManager) -> None:
        """set_stage updates current_stage."""
        stage_manager.set_stage(2)
        assert stage_manager.current_stage == 2
        stage_manager.set_stage(3)
        assert stage_manager.current_stage == 3

    def test_set_stage_invalid(self, stage_manager: StageManager) -> None:
        """set_stage raises ValueError for invalid stage."""
        with pytest.raises(ValueError, match="Invalid stage"):
            stage_manager.set_stage(0)
        with pytest.raises(ValueError, match="Invalid stage"):
            stage_manager.set_stage(4)

    def test_get_stage(self, stage_manager: StageManager) -> None:
        """get_stage returns current stage."""
        assert stage_manager.get_stage() == 1
        stage_manager.set_stage(2)
        assert stage_manager.get_stage() == 2

    def test_is_semi_supervised_stage1(self, stage_manager: StageManager) -> None:
        """Stage 1 is not semi-supervised."""
        stage_manager.set_stage(1)
        assert not stage_manager.is_semi_supervised()

    def test_is_semi_supervised_stage2(self, stage_manager: StageManager) -> None:
        """Stage 2 is semi-supervised."""
        stage_manager.set_stage(2)
        assert stage_manager.is_semi_supervised()

    def test_is_semi_supervised_stage3(self, stage_manager: StageManager) -> None:
        """Stage 3 is semi-supervised."""
        stage_manager.set_stage(3)
        assert stage_manager.is_semi_supervised()

    def test_generate_pseudo_labels_shape(self, stage_manager: StageManager, small_input: torch.Tensor) -> None:
        """generate_pseudo_labels returns correct shapes."""
        stage_manager.set_stage(2)
        result = stage_manager.generate_pseudo_labels(small_input)
        assert "pseudo_labels_class" in result
        assert "pseudo_labels_seg" in result
        assert "mask_class" in result
        assert "mask_seg" in result
        assert "confidence_class" in result
        assert "confidence_seg" in result
        assert "adaptive_threshold" in result
        assert result["pseudo_labels_class"].shape == (2,)
        assert result["pseudo_labels_seg"].shape == (2, 32, 32)
        assert result["mask_class"].shape == (2,)
        assert result["mask_seg"].shape == (2, 32, 32)
        assert isinstance(result["adaptive_threshold"], float)

    def test_generate_pseudo_labels_adaptive_threshold_range(self, stage_manager: StageManager, small_input: torch.Tensor) -> None:
        """Adaptive threshold is in [0, 1]."""
        stage_manager.set_stage(2)
        result = stage_manager.generate_pseudo_labels(small_input)
        assert 0.0 <= result["adaptive_threshold"] <= 1.0

    def test_compute_consistency_loss(self, stage_manager: StageManager, small_input: torch.Tensor) -> None:
        """compute_consistency_loss returns dict with loss values."""
        stage_manager.set_stage(2)
        student_output = stage_manager.student(small_input)
        teacher_output = stage_manager.teacher(small_input)
        losses = stage_manager.compute_consistency_loss(student_output, teacher_output)
        assert "classification" in losses
        assert "segmentation" in losses
        assert losses["classification"].ndim == 0  # scalar
        assert losses["segmentation"].ndim == 0  # scalar

    def test_compute_consistency_loss_with_masks(self, stage_manager: StageManager, small_input: torch.Tensor) -> None:
        """compute_consistency_loss works with masks."""
        stage_manager.set_stage(2)
        student_output = stage_manager.student(small_input)
        teacher_output = stage_manager.teacher(small_input)
        class_mask = torch.tensor([True, False])
        seg_mask = torch.ones(2, 32, 32, dtype=torch.bool)
        losses = stage_manager.compute_consistency_loss(
            student_output, teacher_output,
            class_mask=class_mask, seg_mask=seg_mask,
        )
        assert "classification" in losses
        assert "segmentation" in losses

    def test_refresh_teacher(self, stage_manager: StageManager) -> None:
        """refresh_teacher creates a new teacher from current student."""
        old_teacher = stage_manager.teacher
        stage_manager.refresh_teacher()
        assert stage_manager.teacher is not old_teacher

    def test_reset_statistics(self, stage_manager: StageManager, small_input: torch.Tensor) -> None:
        """reset_statistics clears adaptive threshold stats."""
        stage_manager.set_stage(2)
        stage_manager.generate_pseudo_labels(small_input)
        assert stage_manager.adaptive_threshold.class_count.sum().item() > 0
        stage_manager.reset_statistics()
        assert stage_manager.adaptive_threshold.class_count.sum().item() == 0

    def test_teacher_forward(self, stage_manager: StageManager, small_input: torch.Tensor) -> None:
        """Teacher forward returns correct output dict."""
        output = stage_manager.teacher(small_input)
        assert "classification" in output
        assert "segmentation" in output
        assert output["classification"].shape == (2, 9)
        assert output["segmentation"].shape == (2, 1, 32, 32)


# ── Trainer ────────────────────────────────────────────────────────────────────


class TestTrainer:
    """Tests for Trainer."""

    def test_creation(self, trainer: Trainer) -> None:
        """Trainer can be created with default params."""
        assert trainer.student is not None
        assert trainer.stage_manager is not None
        assert trainer.optimizer is None
        assert trainer.scheduler is None
        assert trainer.supervised_loss_fn is None

    def test_set_optimizer(self, trainer: Trainer) -> None:
        """set_optimizer stores the optimizer."""
        opt = torch.optim.SGD(trainer.student.parameters(), lr=0.01)
        trainer.set_optimizer(opt)
        assert trainer.optimizer is opt

    def test_set_scheduler(self, trainer: Trainer) -> None:
        """set_scheduler stores the scheduler."""
        opt = torch.optim.SGD(trainer.student.parameters(), lr=0.01)
        trainer.set_optimizer(opt)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
        trainer.set_scheduler(scheduler)
        assert trainer.scheduler is scheduler

    def test_set_supervised_loss(self, trainer: Trainer) -> None:
        """set_supervised_loss stores the loss function."""
        def loss_fn(output, targets):
            return {"loss": torch.tensor(0.0)}
        trainer.set_supervised_loss(loss_fn)
        assert trainer.supervised_loss_fn is loss_fn

    def test_train_stage1_no_optimizer(self, trainer: Trainer) -> None:
        """train_stage1 raises RuntimeError without optimizer."""
        with pytest.raises(RuntimeError, match="Optimizer not set"):
            trainer.train_stage1(labeled_data=[])

    def test_train_stage1_no_loss(self, trainer: Trainer) -> None:
        """train_stage1 raises RuntimeError without loss function."""
        opt = torch.optim.SGD(trainer.student.parameters(), lr=0.01)
        trainer.set_optimizer(opt)
        with pytest.raises(RuntimeError, match="Loss function not set"):
            trainer.train_stage1(labeled_data=[])

    def test_train_stage1_sets_stage(self, trainer: Trainer) -> None:
        """train_stage1 sets stage to 1."""
        opt = torch.optim.SGD(trainer.student.parameters(), lr=0.01)
        trainer.set_optimizer(opt)
        trainer.set_supervised_loss(lambda o, t: {"loss": torch.tensor(0.0, requires_grad=True)})
        trainer.train_stage1(labeled_data=[(torch.randn(2, 1, 32, 32), {"classification": torch.randn(2, 9), "segmentation": torch.randn(2, 1, 32, 32)})])
        assert trainer.stage_manager.current_stage == 1

    def test_train_stage1_returns_metrics(self, trainer: Trainer) -> None:
        """train_stage1 returns metrics dict."""
        opt = torch.optim.SGD(trainer.student.parameters(), lr=0.01)
        trainer.set_optimizer(opt)
        trainer.set_supervised_loss(lambda o, t: {"loss": torch.tensor(0.0, requires_grad=True)})
        metrics = trainer.train_stage1(
            labeled_data=[(torch.randn(2, 1, 32, 32), {"classification": torch.randn(2, 9), "segmentation": torch.randn(2, 1, 32, 32)})],
            num_epochs=1,
        )
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_train_stage2_no_optimizer(self, trainer: Trainer) -> None:
        """train_stage2 raises RuntimeError without optimizer."""
        with pytest.raises(RuntimeError, match="Optimizer not set"):
            trainer.train_stage2(labeled_data=[], unlabeled_data=[])

    def test_train_stage2_no_loss(self, trainer: Trainer) -> None:
        """train_stage2 raises RuntimeError without loss function."""
        opt = torch.optim.SGD(trainer.student.parameters(), lr=0.01)
        trainer.set_optimizer(opt)
        with pytest.raises(RuntimeError, match="Loss function not set"):
            trainer.train_stage2(labeled_data=[], unlabeled_data=[])

    def test_train_stage2_sets_stage(self, trainer: Trainer) -> None:
        """train_stage2 sets stage to 2."""
        opt = torch.optim.SGD(trainer.student.parameters(), lr=0.01)
        trainer.set_optimizer(opt)
        trainer.set_supervised_loss(lambda o, t: {"loss": torch.tensor(0.0, requires_grad=True)})
        trainer.train_stage2(
            labeled_data=[(torch.randn(2, 1, 32, 32), {"classification": torch.randn(2, 9), "segmentation": torch.randn(2, 1, 32, 32)})],
            unlabeled_data=[torch.randn(2, 1, 32, 32)],
        )
        assert trainer.stage_manager.current_stage == 2

    def test_train_stage2_returns_metrics(self, trainer: Trainer) -> None:
        """train_stage2 returns metrics dict with all keys."""
        opt = torch.optim.SGD(trainer.student.parameters(), lr=0.01)
        trainer.set_optimizer(opt)
        trainer.set_supervised_loss(lambda o, t: {"loss": torch.tensor(0.0, requires_grad=True)})
        metrics = trainer.train_stage2(
            labeled_data=[(torch.randn(2, 1, 32, 32), {"classification": torch.randn(2, 9), "segmentation": torch.randn(2, 1, 32, 32)})],
            unlabeled_data=[torch.randn(2, 1, 32, 32)],
        )
        assert "loss" in metrics
        assert "supervised_loss" in metrics
        assert "consistency_loss" in metrics

    def test_train_stage3_sets_stage(self, trainer: Trainer) -> None:
        """train_stage3 sets stage to 2 (via train_stage2)."""
        opt = torch.optim.SGD(trainer.student.parameters(), lr=0.01)
        trainer.set_optimizer(opt)
        trainer.set_supervised_loss(lambda o, t: {"loss": torch.tensor(0.0, requires_grad=True)})
        trainer.train_stage3(
            labeled_data=[(torch.randn(2, 1, 32, 32), {"classification": torch.randn(2, 9), "segmentation": torch.randn(2, 1, 32, 32)})],
            unlabeled_data=[torch.randn(2, 1, 32, 32)],
        )
        assert trainer.stage_manager.current_stage == 2

    def test_generate_pseudo_labels(self, trainer: Trainer, small_input: torch.Tensor) -> None:
        """generate_pseudo_labels delegates to StageManager."""
        # Ensure input is on the same device as the model
        device = next(trainer.student.parameters()).device
        x = small_input.to(device)
        result = trainer.generate_pseudo_labels(x)
        assert "pseudo_labels_class" in result
        assert "pseudo_labels_seg" in result

    def test_refresh_teacher(self, trainer: Trainer) -> None:
        """refresh_teacher delegates to StageManager."""
        old_teacher = trainer.stage_manager.teacher
        trainer.refresh_teacher()
        assert trainer.stage_manager.teacher is not old_teacher


# ── Config ─────────────────────────────────────────────────────────────────────


class TestConfig:
    """Tests for config parameters."""

    def test_config_has_stages(self) -> None:
        """Config has training.stages."""
        config = Config.from_yaml(CONFIG_PATH)
        stages = config.get("semi_supervised.stages")
        assert stages is not None
        assert stages == 3

    def test_config_has_mc_passes(self) -> None:
        """Config has semi_supervised.mc_passes."""
        config = Config.from_yaml(CONFIG_PATH)
        mc_passes = config.get("semi_supervised.mc_passes")
        assert mc_passes is not None
        assert mc_passes == 20

    def test_config_has_base_threshold(self) -> None:
        """Config has semi_supervised.confidence_threshold (= base_threshold in paper)."""
        config = Config.from_yaml(CONFIG_PATH)
        bt = config.get("semi_supervised.confidence_threshold")
        assert bt is not None
        assert bt == 0.94

    def test_config_has_alpha(self) -> None:
        """Config has semi_supervised.alpha."""
        config = Config.from_yaml(CONFIG_PATH)
        alpha = config.get("semi_supervised.alpha")
        assert alpha is not None
        assert alpha == 0.08

    def test_config_has_beta(self) -> None:
        """Config has semi_supervised.beta."""
        config = Config.from_yaml(CONFIG_PATH)
        beta = config.get("semi_supervised.beta")
        assert beta is not None
        assert beta == 0.02

    def test_config_has_entropy_threshold(self) -> None:
        """Config has semi_supervised.entropy_threshold."""
        config = Config.from_yaml(CONFIG_PATH)
        et = config.get("semi_supervised.entropy_threshold")
        assert et is not None
        assert et == 0.08

    def test_config_has_mi_threshold(self) -> None:
        """Config has semi_supervised.mutual_information_threshold."""
        config = Config.from_yaml(CONFIG_PATH)
        mi = config.get("semi_supervised.mutual_information_threshold")
        assert mi is not None
        assert mi == 0.12

    def test_config_values_reasonable(self) -> None:
        """Config values are within reasonable ranges."""
        config = Config.from_yaml(CONFIG_PATH)
        assert 0 < config.get("semi_supervised.mc_passes", 0) <= 100
        assert 0 <= config.get("semi_supervised.confidence_threshold", -1) <= 1
        assert 0 <= config.get("semi_supervised.alpha", -1) <= 10
        assert 0 <= config.get("semi_supervised.beta", -1) <= 10
        assert 0 <= config.get("semi_supervised.entropy_threshold", -1) <= 10
        assert 0 <= config.get("semi_supervised.mutual_information_threshold", -1) <= 10
        assert config.get("semi_supervised.stages", 0) in (1, 2, 3)