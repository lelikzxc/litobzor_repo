"""Three-stage training schedule manager for semi-supervised learning.

Manages the workflow for the three-stage semi-supervised training pipeline:

Stage 1 — Supervised only:
    Train on labeled data only. No pseudo-labels, no teacher.

Stage 2 — Pseudo-label generation + adaptive thresholding:
    Generate pseudo-labels from teacher, compute statistics,
    apply adaptive thresholding and uncertainty filtering,
    train on labeled + accepted pseudo-labels.

Stage 3 — Refresh and retrain:
    Regenerate pseudo-labels, recompute statistics,
    refresh accepted pseudo-label dataset, train again.
"""

from __future__ import annotations

import torch
from torch import nn

from papers.semiwafernet.training.adaptive_threshold import AdaptiveThreshold
from papers.semiwafernet.training.consistency import ConsistencyLoss
from papers.semiwafernet.training.ema import EMATeacher
from papers.semiwafernet.training.mc_dropout import MonteCarloDropout
from papers.semiwafernet.training.uncertainty import UncertaintyFilter


class StageManager:
    """Three-stage semi-supervised training workflow manager.

    This class manages the training stage transitions and provides
    the logic for each stage. It does not implement dataset loading,
    optimizers, or training loops — those are handled by the Trainer.

    Args:
        student: The student model (SemiWaferNet instance).
        num_classes: Number of output classes.
        ema_decay: EMA decay rate for teacher model.
        base_threshold: Base confidence threshold.
        alpha: Weight for coefficient of variation term.
        beta: Weight for entropy bonus term.
        mc_passes: Number of Monte Carlo Dropout passes.
        entropy_threshold: Maximum allowed predictive entropy.
        mi_threshold: Maximum allowed mutual information.
        consistency_weight: Weight for consistency loss.
    """

    def __init__(
        self,
        student: nn.Module,
        num_classes: int = 9,
        ema_decay: float = 0.999,
        base_threshold: float = 0.94,
        alpha: float = 0.08,
        beta: float = 0.02,
        mc_passes: int = 20,
        entropy_threshold: float = 0.08,
        mi_threshold: float = 0.12,
        consistency_weight: float = 0.1,
    ) -> None:
        self.student = student
        self.num_classes = num_classes
        self.consistency_weight = consistency_weight
        self.current_stage: int = 1

        # Training components
        self.teacher = EMATeacher(student, momentum=ema_decay)
        self.adaptive_threshold = AdaptiveThreshold(
            num_classes=num_classes,
            base_threshold=base_threshold,
            alpha=alpha,
            beta=beta,
        )
        self.mc_dropout = MonteCarloDropout(num_passes=mc_passes)
        self.uncertainty_filter = UncertaintyFilter(
            entropy_threshold=entropy_threshold,
            mi_threshold=mi_threshold,
        )
        self.consistency_loss = ConsistencyLoss()

    def set_stage(self, stage: int) -> None:
        """Set the current training stage.

        Args:
            stage: Stage number (1, 2, or 3).

        Raises:
            ValueError: If stage is not 1, 2, or 3.
        """
        if stage not in (1, 2, 3):
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")
        self.current_stage = stage

    def get_stage(self) -> int:
        """Get the current training stage.

        Returns:
            Current stage number.
        """
        return self.current_stage

    def is_semi_supervised(self) -> bool:
        """Check if the current stage uses semi-supervised learning.

        Returns:
            True if stage is 2 or 3.
        """
        return self.current_stage in (2, 3)

    def generate_pseudo_labels(
        self, unlabeled_x: torch.Tensor
    ) -> dict[str, torch.Tensor | float]:
        """Generate pseudo-labels for unlabeled data using the teacher.

        This method:
            1. Runs teacher forward to get logits
            2. Runs MC Dropout for uncertainty estimation
            3. Generates pseudo-labels with confidence scores
            4. Updates adaptive threshold statistics
            5. Applies uncertainty filtering

        Args:
            unlabeled_x: Unlabeled input tensor [B, C, H, W].

        Returns:
            Dictionary with:
                "pseudo_labels_class": [B] pseudo-label indices.
                "pseudo_labels_seg": [B, H, W] pseudo-label indices.
                "mask_class": [B] boolean acceptance mask.
                "mask_seg": [B, H, W] boolean acceptance mask.
                "confidence_class": [B] confidence scores.
                "confidence_seg": [B, H, W] confidence scores.
                "adaptive_threshold": float threshold value.
        """
        # MC Dropout for uncertainty estimation (Equations 5-6, 11-12).
        # Dropout is activated in the ViT branch during pseudo-label generation,
        # so we run stochastic forward passes on the student and average the
        # predictive probabilities to obtain the mean predictive distribution.
        mc_results = self.mc_dropout(self.student, unlabeled_x)

        # Candidate pseudo-label and confidence from the mean predictive
        # distribution (Equation 6): y_hat = argmax_c p̄_c(x), q(x) = max_c p̄_c(x)
        class_probs = mc_results["mean_probs_class"]
        seg_probs = mc_results["mean_probs_seg"]
        class_confidence, pseudo_class_labels = class_probs.max(dim=1)
        seg_confidence, pseudo_seg_labels = seg_probs.max(dim=1)

        # Update adaptive threshold statistics (Equations 7-9)
        self.adaptive_threshold.update_statistics(
            confidence=class_confidence,
            pseudo_labels=pseudo_class_labels,
        )
        self.adaptive_threshold.update_statistics(
            confidence=seg_confidence,
            pseudo_labels=pseudo_seg_labels,
        )

        # Class-adaptive, sample-wise threshold (Equation 10)
        adaptive_tau_class = self.adaptive_threshold.compute_threshold(
            pseudo_labels=pseudo_class_labels,
            entropy=mc_results["entropy_class"],
        )
        adaptive_tau_seg = self.adaptive_threshold.compute_threshold(
            pseudo_labels=pseudo_seg_labels,
            entropy=mc_results["entropy_seg"],
        )

        # Uncertainty filtering (Equation 13)
        filter_masks = self.uncertainty_filter(
            confidence_class=class_confidence,
            confidence_seg=seg_confidence,
            adaptive_threshold=adaptive_tau_class,
            adaptive_threshold_seg=adaptive_tau_seg,
            entropy_class=mc_results["entropy_class"],
            entropy_seg=mc_results["entropy_seg"],
            mutual_info_class=mc_results["mutual_info_class"],
            mutual_info_seg=mc_results["mutual_info_seg"],
        )

        return {
            "pseudo_labels_class": pseudo_class_labels,
            "pseudo_labels_seg": pseudo_seg_labels,
            "mask_class": filter_masks["classification"],
            "mask_seg": filter_masks["segmentation"],
            "confidence_class": class_confidence,
            "confidence_seg": seg_confidence,
            "adaptive_threshold": adaptive_tau_class.mean().item(),
        }

    def compute_consistency_loss(
        self,
        student_output: dict[str, torch.Tensor],
        teacher_output: dict[str, torch.Tensor],
        class_mask: torch.Tensor | None = None,
        seg_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute consistency loss between student and teacher.

        Args:
            student_output: Student forward output dict.
            teacher_output: Teacher forward output dict.
            class_mask: Optional classification mask [B].
            seg_mask: Optional segmentation mask [B, H, W].

        Returns:
            Dictionary with "classification" and "segmentation" loss values.
        """
        return self.consistency_loss(
            student_output, teacher_output,
            class_mask=class_mask, seg_mask=seg_mask,
        )

    def refresh_teacher(self) -> None:
        """Refresh the teacher model by copying current student."""
        self.teacher = EMATeacher(
            self.student,
            momentum=self.teacher.momentum,
        )

    def reset_statistics(self) -> None:
        """Reset adaptive threshold statistics."""
        self.adaptive_threshold.reset()