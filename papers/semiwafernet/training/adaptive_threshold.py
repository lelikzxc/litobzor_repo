"""Adaptive thresholding for pseudo-label selection.

Maintains per-class statistics (mean confidence, confidence std)
and computes an adaptive threshold that adjusts based on class-wise
prediction confidence distribution and entropy.
"""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveThreshold(nn.Module):
    """Adaptive threshold for pseudo-label selection.

    Maintains per-class running statistics of prediction confidence
    and computes a dynamic threshold:

        tau = base_threshold + alpha * (sigma / mu) + beta * (1 - entropy)

    where:
        - mu: per-class mean confidence
        - sigma: per-class confidence standard deviation
        - entropy: normalised predictive entropy (in [0, 1])

    The threshold is clamped to [0, 1].

    Args:
        num_classes: Number of output classes.
        base_threshold: Base confidence threshold (default: 0.9).
        alpha: Weight for coefficient of variation term (default: 0.1).
        beta: Weight for entropy bonus term (default: 0.05).
    """

    def __init__(
        self,
        num_classes: int,
        base_threshold: float = 0.9,
        alpha: float = 0.1,
        beta: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.beta = beta

        # Per-class running statistics
        self.register_buffer("class_mean", torch.zeros(num_classes))
        self.register_buffer("class_std", torch.ones(num_classes))
        self.register_buffer("class_count", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def update_statistics(
        self,
        confidence: torch.Tensor,
        pseudo_labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Update per-class confidence statistics.

        Args:
            confidence: Confidence scores [N] or [N, H, W].
            pseudo_labels: Predicted class indices [N] or [N, H, W].
            mask: Optional boolean mask to select valid predictions.
        """
        flat_conf = confidence.flatten()
        flat_labels = pseudo_labels.flatten()

        if mask is not None:
            flat_mask = mask.flatten()
            flat_conf = flat_conf[flat_mask]
            flat_labels = flat_labels[flat_mask]

        if flat_conf.numel() == 0:
            return

        # Compute per-class statistics
        for c in range(self.num_classes):
            class_mask_c = flat_labels == c
            class_conf = flat_conf[class_mask_c]
            if class_conf.numel() == 0:
                continue

            # Welford's online update for mean and std
            n_new = class_conf.numel()
            n_old = self.class_count[c].item()
            n_total = n_old + n_new

            if n_total == 0:
                continue

            old_mean = self.class_mean[c].item()
            new_mean = class_conf.mean().item()

            # Update mean
            updated_mean = (n_old * old_mean + n_new * new_mean) / n_total

            # Update std using Welford's combined variance
            if n_old > 0:
                old_var = self.class_std[c].item() ** 2
                new_var = class_conf.var(correction=0).item() if n_new > 1 else 0.0
                delta = new_mean - old_mean
                combined_var = (
                    n_old * old_var
                    + n_new * new_var
                    + n_old * n_new * delta * delta / n_total
                ) / n_total
                updated_std = max(combined_var**0.5, 1e-8)
            else:
                updated_std = max(class_conf.std(correction=0).item(), 1e-8) if n_new > 1 else 1.0

            self.class_mean[c] = updated_mean
            self.class_std[c] = updated_std
            self.class_count[c] = n_total

    @torch.no_grad()
    def compute_threshold(
        self,
        entropy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the adaptive threshold.

        Args:
            entropy: Optional normalised entropy [0, 1] for entropy bonus.
                If None, entropy bonus is zero.

        Returns:
            Adaptive threshold value (scalar tensor).
        """
        # Coefficient of variation term: sigma / mu (mean across classes)
        mu = self.class_mean
        sigma = self.class_std

        # Avoid division by zero
        cv = sigma / mu.clamp(min=1e-8)

        # Mean CV across classes that have been observed
        valid = self.class_count > 0
        if valid.any():
            cv_term = cv[valid].mean().item()
        else:
            cv_term = 0.0

        # Entropy bonus
        if entropy is not None:
            # Normalise entropy to [0, 1] if not already
            if entropy.numel() > 0:
                e_mean = entropy.mean().item()
            else:
                e_mean = 0.0
            entropy_term = 1.0 - e_mean
        else:
            entropy_term = 0.0

        tau = self.base_threshold + self.alpha * cv_term + self.beta * entropy_term
        tau = max(0.0, min(1.0, tau))
        return torch.tensor(tau, dtype=torch.float32)

    def get_threshold_value(self) -> float:
        """Return the current adaptive threshold as a Python float.

        Uses the current statistics without entropy bonus.
        """
        return self.compute_threshold(entropy=None).item()

    def reset(self) -> None:
        """Reset all statistics to initial values."""
        self.class_mean.zero_()
        self.class_std.fill_(1.0)
        self.class_count.zero_()