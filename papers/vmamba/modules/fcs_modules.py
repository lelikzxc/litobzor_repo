"""FCS-VMamba specific modules: FA, SFS, CLCA.

Implements the three core modules introduced in the FCS-VMamba paper:

    1. **Frequency Attention (FA)**
        - Converts features to frequency domain via FFT
        - Learns frequency-domain attention weights
        - Applies inverse FFT to return to spatial domain
        - Compresses/selects important frequency components

    2. **Saliency Feature Suppression (SFS)**
        - Learns to suppress non-salient (background) features
        - Uses a gating mechanism with sigmoid activation
        - Preserves defect-salient features while suppressing noise

    3. **Cross-Layer Channel Attention (CLCA)**
        - Connects features across stages via channel attention
        - Aggregates multi-scale context from earlier stages
        - Applies channel-wise recalibration to later stages

All hyperparameters come from ``configs/config.yaml``.
No hardcoded architecture parameters.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.fft as fft
import torch.nn.functional as F
from torch import nn


def _gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Create a GroupNorm layer with a valid number of groups.

    ``nn.GroupNorm`` requires ``num_channels`` to be divisible by
    ``num_groups``. This helper picks the largest divisor of
    ``num_channels`` that is ≤ ``max_groups`` (default 32), falling
    back to ``num_groups=1`` (equivalent to per-channel LayerNorm).

    This makes all modules robust to any batch size, unlike
    ``nn.BatchNorm2d`` which fails with batch size 1 when the
    spatial dimensions are 1×1 (e.g. after GAP with reduced channels).
    """
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


# ── Frequency Attention (FA) ──────────────────────────────────────────────


class FrequencyAttention(nn.Module):
    """Frequency Attention (FA) module.

    Applies attention in the frequency domain to selectively emphasise or
    suppress frequency components. The module:

        1. Applies 2D FFT to convert spatial features to frequency domain.
        2. Shifts low frequencies to centre (fftshift).
        3. Learns a frequency-domain attention mask via a small conv net.
        4. Applies the mask to the frequency components.
        5. Inverse shifts and inverse FFT to return to spatial domain.

    This implements the "frequency-compressed" concept from the FCS-VMamba
    paper, where important frequency bands are selectively amplified and
    irrelevant ones are suppressed.

    Args:
        dim: Input channel dimension.
        reduction: Channel reduction ratio for the attention MLP
            (default: 16, from config ``model.fa.reduction``).
        fft_norm: Normalisation mode for FFT (``"ortho"``, from config
            ``model.fa.fft_norm``).
    """

    def __init__(self, dim: int, reduction: int = 16, fft_norm: str = "ortho") -> None:
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.fft_norm = fft_norm

        # Channel squeeze-excitation style gating for frequency domain
        # Use GroupNorm instead of BatchNorm for batch-size independence
        reduced_dim = max(1, dim // reduction)
        self.freq_gate = nn.Sequential(
            nn.Conv2d(dim, reduced_dim, kernel_size=1, bias=False),
            _gn(reduced_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, dim, kernel_size=1, bias=False),
            _gn(dim),
            nn.Sigmoid(),
        )

        # Learnable scale for the frequency residual
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            Frequency-attended output of shape ``[B, C, H, W]``.
        """
        # 2D FFT: [B, C, H, W] → [B, C, H, W] (complex)
        x_fft = fft.fft2(x, norm=self.fft_norm)
        x_fft_shifted = fft.fftshift(x_fft, dim=(-2, -1))

        # Compute magnitude for gating
        magnitude = torch.abs(x_fft_shifted)

        # Learn frequency-domain attention mask
        # Use log-magnitude for better numerical stability
        log_magnitude = torch.log(magnitude + 1e-8)
        gate = self.freq_gate(log_magnitude)

        # Apply gate to complex spectrum
        x_fft_gated = x_fft_shifted * gate

        # Inverse FFT
        x_fft_unshifted = fft.ifftshift(x_fft_gated, dim=(-2, -1))
        x_out = fft.ifft2(x_fft_unshifted, norm=self.fft_norm).real

        # Residual connection with learnable scale
        out = x + self.scale * x_out

        return out


# ── Saliency Feature Suppression (SFS) ────────────────────────────────────


class SaliencySuppression(nn.Module):
    """Saliency Feature Suppression (SFS) module.

    Suppresses non-salient (background) features while preserving
    defect-salient features. The module:

        1. Computes a spatial saliency map via channel-wise statistics.
        2. Learns a suppression gate via a small conv net.
        3. Suppresses low-saliency regions and preserves high-saliency ones.

    This helps the model focus on defect-relevant regions and ignore
    background noise, which is critical for wafer defect detection where
    defects are often small and subtle.

    Args:
        dim: Input channel dimension.
        reduction: Channel reduction ratio for the gating MLP
            (default: 4, from config ``model.sfs.reduction``).
    """

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = reduction

        # Saliency gating network
        # Uses a bottleneck design: C → C/r → C
        # Use GroupNorm instead of BatchNorm for batch-size independence
        reduced_dim = max(1, dim // reduction)
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim, reduced_dim, kernel_size=3, padding=1, bias=False),
            _gn(reduced_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, dim, kernel_size=3, padding=1, bias=False),
            _gn(dim),
            nn.Sigmoid(),
        )

        # Learnable bias for the suppression threshold
        self.suppression_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            Saliency-suppressed output of shape ``[B, C, H, W]``.
        """
        # Compute spatial saliency: channel-wise mean as a crude saliency map
        # [B, C, H, W] → [B, 1, H, W] via mean across channels
        saliency = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Normalise saliency to [0, 1] per sample
        B, _, H, W = saliency.shape
        saliency_flat = saliency.view(B, -1)
        saliency_min = saliency_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        saliency_max = saliency_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        saliency_norm = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)

        # Compute suppression gate from input features
        gate = self.gate_net(x)  # [B, C, H, W]

        # Combine saliency map with learned gate
        # Regions with low saliency get suppressed more
        suppression = gate * (1.0 - saliency_norm + self.suppression_bias)

        # Apply suppression: x * (1 - suppression) = preserve salient, suppress non-salient
        out = x * (1.0 - suppression)

        return out


# ── Cross-Layer Channel Attention (CLCA) ──────────────────────────────────


class CrossLayerChannelAttention(nn.Module):
    """Cross-Layer Channel Attention (CLCA) module.

    Connects features across stages by computing channel attention from
    one feature map and applying it to another. The module:

        1. Takes a "guide" feature (from an earlier stage) and a "target"
           feature (from a later stage).
        2. Pools the guide feature to match the target spatial resolution.
        3. Computes channel attention weights from the pooled guide.
        4. Applies the attention to recalibrate the target features.

    This enables multi-scale context aggregation across the hierarchical
    backbone, allowing later stages to benefit from fine-grained details
    preserved in earlier stages.

    Args:
        guide_dim: Channel dimension of the guide (earlier stage) features.
        target_dim: Channel dimension of the target (later stage) features.
        reduction: Channel reduction ratio for the attention MLP
            (default: 16, from config ``model.clca.reduction``).
    """

    def __init__(self, guide_dim: int, target_dim: int, reduction: int = 16) -> None:
        super().__init__()
        self.guide_dim = guide_dim
        self.target_dim = target_dim
        self.reduction = reduction

        # Project guide features to target dimension if needed
        if guide_dim != target_dim:
            self.guide_proj = nn.Conv2d(guide_dim, target_dim, kernel_size=1, bias=False)
        else:
            self.guide_proj = nn.Identity()

        # Channel attention MLP (SE-style)
        # Use GroupNorm instead of BatchNorm for batch-size independence
        reduced_dim = max(1, target_dim // reduction)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(target_dim, reduced_dim, kernel_size=1, bias=False),
            _gn(reduced_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, target_dim, kernel_size=1, bias=False),
            _gn(target_dim),
            nn.Sigmoid(),
        )

        # Learnable scale for the CLCA residual
        # Initialised to a small positive value so gradients flow to the
        # guide path from the start (zero would block guide gradients).
        self.scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, guide: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            guide: Guide features from an earlier stage ``[B, C_g, H_g, W_g]``.
            target: Target features from a later stage ``[B, C_t, H_t, W_t]``.

        Returns:
            Recalibrated target features ``[B, C_t, H_t, W_t]``.
        """
        # Project guide to target channel dimension
        guide_proj = self.guide_proj(guide)  # [B, C_t, H_g, W_g]

        # Pool guide to target spatial resolution
        _, _, H_t, W_t = target.shape
        guide_pooled = F.interpolate(guide_proj, size=(H_t, W_t), mode="bilinear", align_corners=False)

        # Compute channel attention from pooled guide features
        attn = self.channel_attn(guide_pooled)  # [B, C_t, 1, 1]

        # Apply attention to target features with residual
        out = target + self.scale * (target * attn)

        return out


# ── Factory / helper ──────────────────────────────────────────────────────


def build_fa(config: Any, dim: int) -> FrequencyAttention:
    """Build FrequencyAttention from config.

    Args:
        config: Config object with ``model.fa.*`` accessible.
        dim: Input channel dimension.

    Returns:
        Configured FrequencyAttention instance.
    """
    return FrequencyAttention(
        dim=dim,
        reduction=config.get("model.fa.reduction", 16),
        fft_norm=config.get("model.fa.fft_norm", "ortho"),
    )


def build_sfs(config: Any, dim: int) -> SaliencySuppression:
    """Build SaliencySuppression from config.

    Args:
        config: Config object with ``model.sfs.*`` accessible.
        dim: Input channel dimension.

    Returns:
        Configured SaliencySuppression instance.
    """
    return SaliencySuppression(
        dim=dim,
        reduction=config.get("model.sfs.reduction", 4),
    )


def build_clca(config: Any, guide_dim: int, target_dim: int) -> CrossLayerChannelAttention:
    """Build CrossLayerChannelAttention from config.

    Args:
        config: Config object with ``model.clca.*`` accessible.
        guide_dim: Channel dimension of guide features.
        target_dim: Channel dimension of target features.

    Returns:
        Configured CrossLayerChannelAttention instance.
    """
    return CrossLayerChannelAttention(
        guide_dim=guide_dim,
        target_dim=target_dim,
        reduction=config.get("model.clca.reduction", 16),
    )


__all__ = [
    "FrequencyAttention",
    "SaliencySuppression",
    "CrossLayerChannelAttention",
    "build_fa",
    "build_sfs",
    "build_clca",
]