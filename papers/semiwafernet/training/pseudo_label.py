"""Pseudo-label generation with confidence thresholding.

Generates pseudo-labels from teacher predictions and filters
low-confidence predictions using a configurable threshold.
"""

from __future__ import annotations

import torch
from torch import nn


class PseudoLabelGenerator(nn.Module):
    """Generate confidence-thresholded pseudo-labels from teacher logits.

    For each pixel (segmentation) or sample (classification):
        1. Compute softmax over class dimension
        2. Take argmax as pseudo-label
        3. Take max probability as confidence score
        4. Mask out predictions below confidence_threshold

    Args:
        confidence_threshold: Minimum confidence to retain a pseudo-label
            (default: 0.9).
    """

    def __init__(self, confidence_threshold: float = 0.9) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def generate_classification(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate pseudo-labels for classification.

        Args:
            logits: Teacher classification logits [B, num_classes].

        Returns:
            pseudo_labels: [B] with argmax class indices.
            mask: [B] boolean mask where confidence >= threshold.
        """
        probs = torch.softmax(logits, dim=1)
        confidence, pseudo_labels = probs.max(dim=1)
        mask = confidence >= self.confidence_threshold
        return pseudo_labels, mask

    def generate_segmentation(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate pseudo-labels for segmentation.

        Args:
            logits: Teacher segmentation logits [B, num_classes, H, W].

        Returns:
            pseudo_labels: [B, H, W] with argmax class indices per pixel.
            mask: [B, H, W] boolean mask where confidence >= threshold.
        """
        probs = torch.softmax(logits, dim=1)
        confidence, pseudo_labels = probs.max(dim=1)
        mask = confidence >= self.confidence_threshold
        return pseudo_labels, mask

    def forward(
        self, class_logits: torch.Tensor, seg_logits: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Generate pseudo-labels for both tasks.

        Args:
            class_logits: Teacher classification logits [B, num_classes].
            seg_logits: Teacher segmentation logits [B, num_classes, H, W].

        Returns:
            Dictionary with:
                "classification": (pseudo_labels [B], mask [B])
                "segmentation": (pseudo_labels [B, H, W], mask [B, H, W])
        """
        return {
            "classification": self.generate_classification(class_logits),
            "segmentation": self.generate_segmentation(seg_logits),
        }