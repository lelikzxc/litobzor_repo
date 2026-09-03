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
        consistency_weight: Kept for API compatibility; unused (paper uses CE on pseudo-labels).
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
        consistency_weight: float = 0.0,
    ) -> None:
        self.student = student
        self.num_classes = num_classes
        self.consistency_weight = consistency_weight
        self.current_stage: int = 1

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

    def set_stage(self, stage: int) -> None:
        """Set the current training stage (1, 2, or 3)."""
        if stage not in (1, 2, 3):
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")
        self.current_stage = stage

    def get_stage(self) -> int:
        """Get the current training stage."""
        return self.current_stage

    def is_semi_supervised(self) -> bool:
        """True if stage is 2 or 3."""
        return self.current_stage in (2, 3)

    def generate_pseudo_labels(
        self, unlabeled_x: torch.Tensor
    ) -> dict[str, torch.Tensor | float]:
        """Generate pseudo-labels for unlabeled data via MC Dropout + filters.

        Returns:
            Dictionary with pseudo labels, masks, confidences, and adaptive threshold.
        """
        mc_results = self.mc_dropout(self.student, unlabeled_x)

        class_probs = mc_results["mean_probs_class"]
        class_confidence, pseudo_class_labels = class_probs.max(dim=1)

        self.adaptive_threshold.update_statistics(
            confidence=class_confidence,
            pseudo_labels=pseudo_class_labels,
        )

        adaptive_tau_class = self.adaptive_threshold.compute_threshold(
            pseudo_labels=pseudo_class_labels,
            entropy=mc_results["entropy_class"],
        )

        filter_masks = self.uncertainty_filter(
            confidence_class=class_confidence,
            confidence_seg=class_confidence,
            adaptive_threshold=adaptive_tau_class,
            adaptive_threshold_seg=adaptive_tau_class,
            entropy_class=mc_results["entropy_class"],
            entropy_seg=mc_results["entropy_class"],
            mutual_info_class=mc_results["mutual_info_class"],
            mutual_info_seg=mc_results["mutual_info_class"],
        )

        B = unlabeled_x.shape[0]
        H, W = unlabeled_x.shape[-2], unlabeled_x.shape[-1]
        dummy_seg = torch.zeros(B, H, W, dtype=torch.long, device=unlabeled_x.device)
        dummy_mask = torch.zeros(B, H, W, dtype=torch.bool, device=unlabeled_x.device)

        return {
            "pseudo_labels_class": pseudo_class_labels,
            "pseudo_labels_seg": dummy_seg,
            "mask_class": filter_masks["classification"],
            "mask_seg": dummy_mask,
            "confidence_class": class_confidence,
            "confidence_seg": torch.zeros(B, H, W, device=unlabeled_x.device),
            "adaptive_threshold": adaptive_tau_class.mean().item(),
        }

    def refresh_teacher(self) -> None:
        """Refresh the teacher model by copying current student."""
        self.teacher = EMATeacher(
            self.student,
            momentum=self.teacher.momentum,
        )

    def reset_statistics(self) -> None:
        """Reset adaptive threshold statistics."""
        self.adaptive_threshold.reset()
