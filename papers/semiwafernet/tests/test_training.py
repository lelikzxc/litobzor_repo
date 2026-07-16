"""Tests for semi-supervised training components.

Covers EMA teacher updates, pseudo-label generation with confidence
thresholding, consistency loss computation, and config loading.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from papers.semiwafernet.training.ema import EMATeacher
from papers.semiwafernet.training.pseudo_label import PseudoLabelGenerator
from papers.semiwafernet.training.consistency import ConsistencyLoss
from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from common.utils.config import Config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def student() -> SemiWaferNet:
    return SemiWaferNet(num_classes=6)


@pytest.fixture
def teacher(student: SemiWaferNet) -> EMATeacher:
    return EMATeacher(student, momentum=0.999)


@pytest.fixture
def pseudo_label_gen() -> PseudoLabelGenerator:
    return PseudoLabelGenerator(confidence_threshold=0.9)


@pytest.fixture
def consistency_loss() -> ConsistencyLoss:
    return ConsistencyLoss(reduction="mean")


# ── Imports ────────────────────────────────────────────────────────────────────


def test_import_training_modules() -> None:
    """All training module classes are importable."""
    assert EMATeacher is not None
    assert PseudoLabelGenerator is not None
    assert ConsistencyLoss is not None


# ── EMA Teacher ────────────────────────────────────────────────────────────────


class TestEMATeacher:
    """Tests for EMATeacher."""

    def test_creation(self, student: SemiWaferNet) -> None:
        """EMATeacher can be created from a student model."""
        teacher = EMATeacher(student, momentum=0.999)
        assert isinstance(teacher, EMATeacher)
        assert teacher.momentum == 0.999

    def test_teacher_has_no_gradients(self, student: SemiWaferNet) -> None:
        """Teacher parameters do not require gradients."""
        teacher = EMATeacher(student)
        for p in teacher.teacher.parameters():
            assert not p.requires_grad, "Teacher parameter requires grad"

    def test_teacher_is_eval_mode(self, student: SemiWaferNet) -> None:
        """Teacher is in eval mode after creation."""
        teacher = EMATeacher(student)
        assert not teacher.teacher.training, "Teacher should be in eval mode"

    def test_initial_parameters_match(self, student: SemiWaferNet) -> None:
        """Teacher parameters initially match student parameters."""
        teacher = EMATeacher(student, momentum=0.999)
        for t_param, s_param in zip(teacher.teacher.parameters(), student.parameters()):
            assert torch.equal(t_param, s_param), "Initial params should match"

    def test_ema_update_changes_teacher(self, student: SemiWaferNet) -> None:
        """EMA update changes teacher parameters."""
        teacher = EMATeacher(student, momentum=0.5)
        # Change student parameters
        for p in student.parameters():
            p.data.add_(1.0)
        old_params = [p.clone() for p in teacher.teacher.parameters()]
        teacher.update(student)
        # Teacher should have moved toward student
        for old_p, new_p in zip(old_params, teacher.teacher.parameters()):
            assert not torch.equal(old_p, new_p), "Teacher should change after update"

    def test_ema_update_formula(self, student: SemiWaferNet) -> None:
        """EMA update follows θ_t = m * θ_t + (1-m) * θ_s."""
        momentum = 0.8
        teacher = EMATeacher(student, momentum=momentum)
        # Change student parameters
        for p in student.parameters():
            p.data.fill_(2.0)
        # Record teacher before update
        teacher_before = [p.clone() for p in teacher.teacher.parameters()]
        teacher.update(student)
        # Verify: new_teacher = 0.8 * old_teacher + 0.2 * student
        for t_before, t_after, s_param in zip(
            teacher_before, teacher.teacher.parameters(), student.parameters()
        ):
            expected = momentum * t_before + (1 - momentum) * s_param
            assert torch.allclose(t_after, expected, atol=1e-6), (
                f"EMA formula mismatch"
            )

    def test_teacher_forward_shape(self, student: SemiWaferNet) -> None:
        """Teacher forward produces correct output shapes."""
        teacher = EMATeacher(student)
        x = torch.randn(2, 3, 128, 128)
        output = teacher(x)
        assert output["classification"].shape == (2, 6)
        assert output["segmentation"].shape == (2, 6, 128, 128)

    def test_teacher_no_grad_forward(self, student: SemiWaferNet) -> None:
        """Teacher forward does not require gradients."""
        teacher = EMATeacher(student)
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            output = teacher(x)
        assert output["classification"].shape == (2, 6)

    def test_multiple_updates(self, student: SemiWaferNet) -> None:
        """Multiple EMA updates converge teacher toward student."""
        teacher = EMATeacher(student, momentum=0.5)
        for _ in range(10):
            for p in student.parameters():
                p.data.add_(0.1)
            teacher.update(student)
        # After many updates with low momentum, teacher should be close to student
        for t_param, s_param in zip(teacher.teacher.parameters(), student.parameters()):
            assert torch.allclose(t_param, s_param, atol=0.5), (
                "Teacher should converge toward student"
            )

    def test_buffer_copy(self, student: SemiWaferNet) -> None:
        """EMA update copies buffers (e.g., BN running stats)."""
        teacher = EMATeacher(student, momentum=0.5)
        # Change student buffers (only float buffers, skip non-float like Long)
        for b in student.buffers():
            if b.numel() > 0 and b.dtype.is_floating_point:
                b.data.add_(1.0)
        teacher.update(student)
        for t_buf, s_buf in zip(teacher.teacher.buffers(), student.buffers()):
            assert torch.equal(t_buf, s_buf), "Buffers should be copied"


# ── Pseudo-Label Generation ────────────────────────────────────────────────────


class TestPseudoLabelGenerator:
    """Tests for PseudoLabelGenerator."""

    def test_creation(self) -> None:
        """PseudoLabelGenerator can be created with a threshold."""
        gen = PseudoLabelGenerator(confidence_threshold=0.9)
        assert gen.confidence_threshold == 0.9

    def test_classification_output_shapes(self, pseudo_label_gen: PseudoLabelGenerator) -> None:
        """Classification pseudo-labels have correct shapes."""
        logits = torch.randn(4, 6)
        labels, mask = pseudo_label_gen.generate_classification(logits)
        assert labels.shape == (4,)
        assert mask.shape == (4,)
        assert labels.dtype == torch.long
        assert mask.dtype == torch.bool

    def test_classification_high_confidence(self, pseudo_label_gen: PseudoLabelGenerator) -> None:
        """High-confidence predictions are kept."""
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # very confident class 0
        labels, mask = pseudo_label_gen.generate_classification(logits)
        assert labels[0] == 0
        assert mask[0] == True

    def test_classification_low_confidence(self, pseudo_label_gen: PseudoLabelGenerator) -> None:
        """Low-confidence predictions are masked out."""
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])  # uniform = low confidence
        labels, mask = pseudo_label_gen.generate_classification(logits)
        assert mask[0] == False

    def test_classification_confidence_threshold(self) -> None:
        """Predictions above the threshold are kept."""
        gen = PseudoLabelGenerator(confidence_threshold=0.5)
        # softmax([5, 0, 0, 0, 0, 0]) ≈ [0.993, 0.001, ...] — well above 0.5
        logits = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        labels, mask = gen.generate_classification(logits)
        assert mask[0] == True
        assert labels[0] == 0

    def test_segmentation_output_shapes(self, pseudo_label_gen: PseudoLabelGenerator) -> None:
        """Segmentation pseudo-labels have correct shapes."""
        logits = torch.randn(2, 6, 16, 16)
        labels, mask = pseudo_label_gen.generate_segmentation(logits)
        assert labels.shape == (2, 16, 16)
        assert mask.shape == (2, 16, 16)
        assert labels.dtype == torch.long
        assert mask.dtype == torch.bool

    def test_segmentation_high_confidence(self, pseudo_label_gen: PseudoLabelGenerator) -> None:
        """High-confidence segmentation pixels are kept."""
        logits = torch.randn(2, 6, 8, 8)
        # Make one pixel very confident
        logits[0, :, 0, 0] = 0.0
        logits[0, 0, 0, 0] = 100.0
        labels, mask = pseudo_label_gen.generate_segmentation(logits)
        assert mask[0, 0, 0] == True

    def test_forward_dict(self, pseudo_label_gen: PseudoLabelGenerator) -> None:
        """Forward returns dict with both tasks."""
        class_logits = torch.randn(4, 6)
        seg_logits = torch.randn(4, 6, 16, 16)
        result = pseudo_label_gen(class_logits, seg_logits)
        assert "classification" in result
        assert "segmentation" in result
        class_labels, class_mask = result["classification"]
        seg_labels, seg_mask = result["segmentation"]
        assert class_labels.shape == (4,)
        assert seg_labels.shape == (4, 16, 16)

    def test_custom_threshold(self) -> None:
        """Custom confidence threshold is respected."""
        gen = PseudoLabelGenerator(confidence_threshold=0.99)
        logits = torch.tensor([[5.0, 1.0, 0.0, 0.0, 0.0, 0.0]])  # high but not 0.99
        labels, mask = gen.generate_classification(logits)
        assert mask[0] == False  # below 0.99 threshold


# ── Consistency Loss ───────────────────────────────────────────────────────────


class TestConsistencyLoss:
    """Tests for ConsistencyLoss."""

    def test_creation(self) -> None:
        """ConsistencyLoss can be created."""
        loss = ConsistencyLoss(reduction="mean")
        assert isinstance(loss, ConsistencyLoss)

    def test_classification_loss_shape(self, consistency_loss: ConsistencyLoss) -> None:
        """Classification consistency loss returns a scalar."""
        student_logits = torch.randn(4, 6)
        teacher_logits = torch.randn(4, 6)
        loss = consistency_loss.classification_loss(student_logits, teacher_logits)
        assert loss.ndim == 0, "Loss should be a scalar tensor"

    def test_classification_loss_value(self, consistency_loss: ConsistencyLoss) -> None:
        """Identical student and teacher give zero loss."""
        logits = torch.randn(4, 6)
        loss = consistency_loss.classification_loss(logits, logits)
        assert loss.item() == 0.0

    def test_classification_loss_positive(self, consistency_loss: ConsistencyLoss) -> None:
        """Different student and teacher give positive loss."""
        student = torch.randn(4, 6)
        teacher = torch.randn(4, 6)
        loss = consistency_loss.classification_loss(student, teacher)
        assert loss.item() > 0.0

    def test_classification_loss_masked(self, consistency_loss: ConsistencyLoss) -> None:
        """Masked classification loss only considers selected samples."""
        student = torch.randn(4, 6)
        teacher = torch.randn(4, 6)
        mask = torch.tensor([True, False, True, False])
        loss_masked = consistency_loss.classification_loss(student, teacher, mask=mask)
        loss_full = consistency_loss.classification_loss(student, teacher)
        assert loss_masked.item() >= 0.0
        assert loss_masked.item() != loss_full.item()  # different due to masking

    def test_classification_loss_empty_mask(self, consistency_loss: ConsistencyLoss) -> None:
        """Empty mask returns zero loss."""
        student = torch.randn(4, 6)
        teacher = torch.randn(4, 6)
        mask = torch.tensor([False, False, False, False])
        loss = consistency_loss.classification_loss(student, teacher, mask=mask)
        assert loss.item() == 0.0

    def test_segmentation_loss_shape(self, consistency_loss: ConsistencyLoss) -> None:
        """Segmentation consistency loss returns a scalar."""
        student = torch.randn(2, 6, 16, 16)
        teacher = torch.randn(2, 6, 16, 16)
        loss = consistency_loss.segmentation_loss(student, teacher)
        assert loss.ndim == 0

    def test_segmentation_loss_identical(self, consistency_loss: ConsistencyLoss) -> None:
        """Identical segmentation logits give zero loss."""
        logits = torch.randn(2, 6, 16, 16)
        loss = consistency_loss.segmentation_loss(logits, logits)
        assert loss.item() == 0.0

    def test_segmentation_loss_masked(self, consistency_loss: ConsistencyLoss) -> None:
        """Masked segmentation loss only considers selected pixels."""
        student = torch.randn(2, 6, 8, 8)
        teacher = torch.randn(2, 6, 8, 8)
        mask = torch.zeros(2, 8, 8, dtype=torch.bool)
        mask[0, :, :] = True  # only first sample
        loss = consistency_loss.segmentation_loss(student, teacher, mask=mask)
        assert loss.item() > 0.0

    def test_forward_dict(self, consistency_loss: ConsistencyLoss) -> None:
        """Forward returns dict with both task losses."""
        student_out = {
            "classification": torch.randn(4, 6),
            "segmentation": torch.randn(4, 6, 16, 16),
        }
        teacher_out = {
            "classification": torch.randn(4, 6),
            "segmentation": torch.randn(4, 6, 16, 16),
        }
        losses = consistency_loss(student_out, teacher_out)
        assert "classification" in losses
        assert "segmentation" in losses
        assert losses["classification"].ndim == 0
        assert losses["segmentation"].ndim == 0

    def test_teacher_detached(self, consistency_loss: ConsistencyLoss) -> None:
        """Teacher logits are detached from computation graph."""
        student = torch.randn(4, 6, requires_grad=True)
        teacher = torch.randn(4, 6, requires_grad=True)
        loss = consistency_loss.classification_loss(student, teacher)
        loss.backward()
        # Student should have grad, teacher should not (detached)
        assert student.grad is not None
        # Verify teacher was detached by checking no grad flows back
        assert teacher.grad is None  # detached before loss


# ── Config Loading ─────────────────────────────────────────────────────────────


def test_config_has_semi_supervised() -> None:
    """Config file contains semi_supervised section."""
    config = Config.from_yaml(CONFIG_PATH)
    ss = config.get("semi_supervised", {})
    assert "enabled" in ss
    assert "ema_decay" in ss
    assert "confidence_threshold" in ss
    assert "consistency_weight" in ss


def test_config_semi_supervised_values() -> None:
    """Config semi_supervised values are reasonable."""
    config = Config.from_yaml(CONFIG_PATH)
    assert config.get("semi_supervised.enabled") is False
    assert 0.0 < config.get("semi_supervised.ema_decay", 0.999) < 1.0
    assert 0.0 < config.get("semi_supervised.confidence_threshold", 0.9) <= 1.0
    assert config.get("semi_supervised.consistency_weight", 0.1) >= 0.0


def test_config_ema_decay_default() -> None:
    """EMA decay defaults to 0.999 if not set."""
    config = Config.from_yaml(CONFIG_PATH)
    ema_decay = config.get("semi_supervised.ema_decay", 0.999)
    assert ema_decay == 0.999