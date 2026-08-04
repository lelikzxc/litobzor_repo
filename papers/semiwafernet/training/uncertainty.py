"""Uncertainty-based filtering for pseudo-label selection.

Accepts pseudo-labels only if they pass all three criteria:
    1. confidence >= adaptive threshold
    2. entropy < entropy_threshold
    3. mutual_information < mi_threshold
"""

from __future__ import annotations

import torch
from torch import nn


class UncertaintyFilter(nn.Module):
    """Uncertainty-based filter for pseudo-label selection.

    A pseudo-label is accepted only if:
        - confidence >= adaptive_threshold
        - entropy < entropy_threshold
        - mutual_information < mi_threshold

    Args:
        entropy_threshold: Maximum allowed predictive entropy (default: 0.5).
        mi_threshold: Maximum allowed mutual information (default: 0.3).
    """

    def __init__(
        self,
        entropy_threshold: float = 0.08,
        mi_threshold: float = 0.12,
    ) -> None:
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.mi_threshold = mi_threshold

    def filter_classification(
        self,
        confidence: torch.Tensor,
        adaptive_threshold: torch.Tensor | float,
        entropy: torch.Tensor,
        mutual_information: torch.Tensor,
    ) -> torch.Tensor:
        """Filter classification pseudo-labels.

        Args:
            confidence: Confidence scores [B].
            adaptive_threshold: Adaptive threshold value (scalar or tensor).
            entropy: Predictive entropy [B].
            mutual_information: Mutual information [B].

        Returns:
            Boolean mask [B] where all criteria are satisfied.
        """
        if isinstance(adaptive_threshold, float):
            adaptive_threshold = torch.tensor(adaptive_threshold, device=confidence.device)

        mask = (
            (confidence >= adaptive_threshold)
            & (entropy < self.entropy_threshold)
            & (mutual_information < self.mi_threshold)
        )
        return mask

    def filter_segmentation(
        self,
        confidence: torch.Tensor,
        adaptive_threshold: torch.Tensor | float,
        entropy: torch.Tensor,
        mutual_information: torch.Tensor,
    ) -> torch.Tensor:
        """Filter segmentation pseudo-labels.

        Args:
            confidence: Confidence scores [B, H, W].
            adaptive_threshold: Adaptive threshold value (scalar or tensor).
            entropy: Predictive entropy [B, H, W].
            mutual_information: Mutual information [B, H, W].

        Returns:
            Boolean mask [B, H, W] where all criteria are satisfied.
        """
        if isinstance(adaptive_threshold, float):
            adaptive_threshold = torch.tensor(adaptive_threshold, device=confidence.device)

        mask = (
            (confidence >= adaptive_threshold)
            & (entropy < self.entropy_threshold)
            & (mutual_information < self.mi_threshold)
        )
        return mask

    def forward(
        self,
        confidence_class: torch.Tensor,
        confidence_seg: torch.Tensor,
        adaptive_threshold: torch.Tensor | float,
        entropy_class: torch.Tensor,
        entropy_seg: torch.Tensor,
        mutual_info_class: torch.Tensor,
        mutual_info_seg: torch.Tensor,
        adaptive_threshold_seg: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Filter pseudo-labels for both tasks.

        Args:
            confidence_class: Classification confidence [B].
            confidence_seg: Segmentation confidence [B, H, W].
            adaptive_threshold: Adaptive threshold for classification [B] or scalar.
            entropy_class: Classification entropy [B].
            entropy_seg: Segmentation entropy [B, H, W].
            mutual_info_class: Classification mutual info [B].
            mutual_info_seg: Segmentation mutual info [B, H, W].
            adaptive_threshold_seg: Optional adaptive threshold for segmentation
                [B, H, W] or scalar. Defaults to ``adaptive_threshold``.

        Returns:
            Dictionary with:
                "classification": boolean mask [B].
                "segmentation": boolean mask [B, H, W].
        """
        if adaptive_threshold_seg is None:
            adaptive_threshold_seg = adaptive_threshold
        return {
            "classification": self.filter_classification(
                confidence_class, adaptive_threshold, entropy_class, mutual_info_class
            ),
            "segmentation": self.filter_segmentation(
                confidence_seg, adaptive_threshold_seg, entropy_seg, mutual_info_seg
            ),
        }