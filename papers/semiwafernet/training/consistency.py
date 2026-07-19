"""Consistency regularization loss for semi-supervised learning.

Enforces prediction consistency between student and teacher models
under different augmentations. Supports both classification and
segmentation consistency.
"""

from __future__ import annotations

import torch
from torch import nn


class ConsistencyLoss(nn.Module):
    """Consistency loss between student and teacher predictions.

    Uses Mean Squared Error (MSE) between student and teacher logits
    as the consistency measure. The teacher logits are detached from
    the computation graph to prevent gradient flow through the teacher.

    Args:
        reduction: Loss reduction method. One of "mean", "sum", "none"
            (default: "mean").
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def classification_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute consistency loss for classification.

        Args:
            student_logits: Student classification logits [B, num_classes].
            teacher_logits: Teacher classification logits [B, num_classes].
                Detached internally.
            mask: Optional boolean mask [B] to select samples.

        Returns:
            Scalar consistency loss.
        """
        teacher_logits = teacher_logits.detach()
        if mask is not None:
            student_logits = student_logits[mask]
            teacher_logits = teacher_logits[mask]
            if student_logits.numel() == 0:
                return torch.tensor(0.0, device=student_logits.device)
        return self.mse(student_logits, teacher_logits)

    def segmentation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute consistency loss for segmentation.

        Args:
            student_logits: Student segmentation logits [B, num_classes, H, W].
            teacher_logits: Teacher segmentation logits [B, num_classes, H, W].
                Detached internally.
            mask: Optional boolean mask [B, H, W] to select pixels.

        Returns:
            Scalar consistency loss.
        """
        teacher_logits = teacher_logits.detach()
        if mask is not None:
            # Expand mask to match logits dimensions
            mask = mask.unsqueeze(1).expand_as(student_logits)
            student_logits = student_logits[mask].view(-1, student_logits.size(1))
            teacher_logits = teacher_logits[mask].view(-1, teacher_logits.size(1))
            if student_logits.numel() == 0:
                return torch.tensor(0.0, device=student_logits.device)
        return self.mse(student_logits, teacher_logits)

    def forward(
        self,
        student_outputs: dict[str, torch.Tensor],
        teacher_outputs: dict[str, torch.Tensor],
        class_mask: torch.Tensor | None = None,
        seg_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute consistency losses for both tasks.

        Args:
            student_outputs: Student forward output dict.
            teacher_outputs: Teacher forward output dict.
            class_mask: Optional mask for classification [B].
            seg_mask: Optional mask for segmentation [B, H, W].

        Returns:
            Dictionary with "classification" and "segmentation" loss values.
        """
        return {
            "classification": self.classification_loss(
                student_outputs["classification"],
                teacher_outputs["classification"],
                mask=class_mask,
            ),
            "segmentation": self.segmentation_loss(
                student_outputs["segmentation"],
                teacher_outputs["segmentation"],
                mask=seg_mask,
            ),
        }