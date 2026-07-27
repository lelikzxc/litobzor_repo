"""BiFPN (Bidirectional Feature Pyramid Network) module.

Replaces the PAN-FPN neck in YOLOv10 with a weighted bidirectional feature
pyramid, as described in:

    "Wafer Defect Detection Technology Based on CTM-IYOLOv10 Network"
    (Section 2.2.1, Figure 4, Figure 6c)

BiFPN introduces:
    1. Weighted feature fusion with learnable per-channel weights
       (fast normalized fusion via softmax-based normalization).
    2. Additional cross-scale connections between input and output nodes
       at the same resolution.
    3. Repeated top-down and bottom-up bidirectional pathways.

Reference:
    Tan et al., "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BiFPNBlock(nn.Module):
    """Single BiFPN block with weighted bidirectional fusion.

    Operates on a list of feature maps at different scales [P3, P4, P5]
    (or more levels) and performs one top-down + bottom-up fusion pass.

    Args:
        channels: Number of channels for all feature levels.
        num_levels: Number of feature pyramid levels (default 3 for P3-P5).
        epsilon: Small constant for numerical stability in weight normalization.
    """

    def __init__(self, channels: int, num_levels: int = 3, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.channels = channels
        self.num_levels = num_levels
        self.epsilon = epsilon

        # Learnable weights for top-down fusion (one weight per input connection)
        self.td_weights = nn.Parameter(torch.ones(num_levels, 2))
        # Learnable weights for bottom-up fusion (one weight per input connection)
        self.bu_weights = nn.Parameter(torch.ones(num_levels, 3))

        # Convs after fusion (one per level)
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(num_levels)
        ])
        self.bu_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(num_levels)
        ])

    @staticmethod
    def _fast_normalized_fusion(weights: torch.Tensor, features: list[torch.Tensor], epsilon: float) -> torch.Tensor:
        """Apply fast normalized fusion: out = sum(w_i / (sum(w_j) + eps) * feat_i)."""
        w = F.relu(weights)  # ensure non-negative
        w_norm = w / (w.sum(dim=0, keepdim=True) + epsilon)
        out = sum(w_norm[i] * features[i] for i in range(len(features)))
        return out

    @staticmethod
    def _resize_to(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Resize ``source`` to match spatial size of ``target``."""
        if source.shape[2:] == target.shape[2:]:
            return source
        return F.interpolate(source, size=target.shape[2:], mode="nearest")

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            features: List of feature maps [P3, P4, P5] (low to high level).

        Returns:
            List of enhanced feature maps at the same resolutions.
        """
        assert len(features) == self.num_levels, f"Expected {self.num_levels} features, got {len(features)}"

        # ── Top-down pathway ─────────────────────────────────────────────
        td_features = list(features)
        for i in range(self.num_levels - 2, -1, -1):  # from top to bottom
            # Fusion: current level input + upsampled higher-level feature
            higher_up = F.interpolate(td_features[i + 1], size=td_features[i].shape[2:], mode="nearest")
            fused = self._fast_normalized_fusion(
                self.td_weights[i],
                [td_features[i], higher_up],
                self.epsilon,
            )
            td_features[i] = self.td_convs[i](fused)

        # ── Bottom-up pathway ────────────────────────────────────────────
        bu_features = list(td_features)
        for i in range(1, self.num_levels):  # from bottom to top
            # Fusion: current level + downsampled lower-level feature + original input
            lower_down = F.max_pool2d(bu_features[i - 1], kernel_size=2, stride=2)
            # Pad if spatial size mismatch after downsampling
            if lower_down.shape[2:] != bu_features[i].shape[2:]:
                lower_down = F.interpolate(lower_down, size=bu_features[i].shape[2:], mode="nearest")
            fused = self._fast_normalized_fusion(
                self.bu_weights[i],
                [bu_features[i], lower_down, features[i]],  # + original input (residual)
                self.epsilon,
            )
            bu_features[i] = self.bu_convs[i](fused)

        return bu_features


class BiFPN(nn.Module):
    """BiFPN neck for multi-scale feature fusion.

    Repeats the BiFPN block ``num_repeats`` times for deeper fusion.

    Args:
        channels: Number of channels for all feature levels.
        num_levels: Number of feature pyramid levels.
        num_repeats: Number of BiFPN block repeats.
    """

    def __init__(self, channels: int = 256, num_levels: int = 3, num_repeats: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.num_levels = num_levels
        self.num_repeats = num_repeats

        # Input channels for YOLOv10n: P3=64, P4=128, P5=256
        self._in_channels_list = [64, 128, 256]

        # Project all input features to the same channel dimension
        self.input_proj = nn.ModuleList([
            nn.Conv2d(ch, channels, kernel_size=1, bias=False) if ch != channels else nn.Identity()
            for ch in self._in_channels_list
        ])

        # Project back to original channel dimensions after fusion
        self.output_proj = nn.ModuleList([
            nn.Conv2d(channels, ch, kernel_size=1, bias=False) if ch != channels else nn.Identity()
            for ch in self._in_channels_list
        ])

        self.blocks = nn.ModuleList([
            BiFPNBlock(channels, num_levels) for _ in range(num_repeats)
        ])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            features: List of feature maps [P3, P4, P5] from backbone.

        Returns:
            List of fused feature maps at the same resolutions and channel dims.
        """
        # Project all to same channel dim
        projected = [proj(f) for proj, f in zip(self.input_proj, features)]

        # Apply BiFPN blocks
        out = projected
        for block in self.blocks:
            out = block(out)

        # Project back to original channel dimensions
        out = [proj(f) for proj, f in zip(self.output_proj, out)]

        return out