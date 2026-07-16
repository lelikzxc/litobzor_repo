"""Monte Carlo Dropout for uncertainty estimation.

Enables dropout during inference and performs multiple stochastic
forward passes to estimate predictive uncertainty via:
    - mean probabilities
    - predictive entropy
    - mutual information
"""

from __future__ import annotations

import torch
from torch import nn


class MonteCarloDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation.

    Performs multiple stochastic forward passes with dropout enabled
    during inference, then computes uncertainty metrics from the
    distribution of predictions.

    Args:
        num_passes: Number of stochastic forward passes (default: 20).
    """

    def __init__(self, num_passes: int = 20) -> None:
        super().__init__()
        self.num_passes = num_passes

    @torch.no_grad()
    def forward(self, model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run MC Dropout and return uncertainty estimates.

        Args:
            model: A PyTorch model with dropout layers.
            x: Input tensor [B, C, H, W].

        Returns:
            Dictionary with:
                "mean_probs_class": [B, num_classes] mean softmax over passes.
                "mean_probs_seg": [B, num_classes, H, W] mean softmax over passes.
                "entropy_class": [B] predictive entropy for classification.
                "entropy_seg": [B, H, W] predictive entropy for segmentation.
                "mutual_info_class": [B] mutual information for classification.
                "mutual_info_seg": [B, H, W] mutual information for segmentation.
        """
        model.train()  # enable dropout
        B = x.shape[0]

        class_probs_list: list[torch.Tensor] = []
        seg_probs_list: list[torch.Tensor] = []

        for _ in range(self.num_passes):
            output = model(x)
            class_probs_list.append(torch.softmax(output["classification"], dim=1))  # [B, C]
            seg_probs_list.append(torch.softmax(output["segmentation"], dim=1))      # [B, C, H, W]

        model.eval()  # restore eval mode

        # Stack and compute mean
        class_probs_stack = torch.stack(class_probs_list, dim=0)  # [num_passes, B, C]
        seg_probs_stack = torch.stack(seg_probs_list, dim=0)      # [num_passes, B, C, H, W]

        mean_class_probs = class_probs_stack.mean(dim=0)  # [B, C]
        mean_seg_probs = seg_probs_stack.mean(dim=0)      # [B, C, H, W]

        # Predictive entropy: H[p(y|x)] = -sum_c p_c * log(p_c)
        entropy_class = self._entropy(mean_class_probs)   # [B]
        entropy_seg = self._entropy(mean_seg_probs)       # [B, H, W]

        # Mutual information: H[E[p]] - E[H[p]]
        # Expected entropy: E[H[p]] = mean over passes of per-pass entropy
        # Compute per-pass entropy manually (Tensor.map not available in all PyTorch versions)
        per_pass_entropy_class = torch.stack(
            [self._entropy(class_probs_stack[i]) for i in range(self.num_passes)],
            dim=0,
        )  # [num_passes, B]
        expected_entropy_class = per_pass_entropy_class.mean(dim=0)  # [B]

        per_pass_entropy_seg = torch.stack(
            [self._entropy(seg_probs_stack[i]) for i in range(self.num_passes)],
            dim=0,
        )  # [num_passes, B, H, W]
        expected_entropy_seg = per_pass_entropy_seg.mean(dim=0)  # [B, H, W]

        mutual_info_class = entropy_class - expected_entropy_class  # [B]
        mutual_info_seg = entropy_seg - expected_entropy_seg        # [B, H, W]

        return {
            "mean_probs_class": mean_class_probs,
            "mean_probs_seg": mean_seg_probs,
            "entropy_class": entropy_class,
            "entropy_seg": entropy_seg,
            "mutual_info_class": mutual_info_class,
            "mutual_info_seg": mutual_info_seg,
        }

    @staticmethod
    def _entropy(probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy along the class dimension.

        Args:
            probs: Probability tensor with class dimension at dim=1.

        Returns:
            Entropy tensor with class dimension reduced.
        """
        # Clamp to avoid log(0)
        eps = torch.finfo(probs.dtype).eps
        clamped = probs.clamp(min=eps)
        return -(clamped * clamped.log()).sum(dim=1)