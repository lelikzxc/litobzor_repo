"""FCS-VMamba modules: FA, SFS, CLCA.

Faithful to the FCS-VMamba paper (J. Imaging 2026) equations and the
author-linked reference implementation (gggmamba_sys.py), adapted to
channel-first ``[B, C, H, W]`` tensors used by this repo.

    1. Frequency Attention (FA) — Eqs. (1)–(3)
       RFFT amplitude → GAP → bottleneck → sigmoid channel weights → x * w

    2. Saliency Feature Suppression (SFS) — Eqs. (4)–(6)
       Channel-mean |x| saliency → top-k soft mask (0.1) → detached residual

    3. Cross-Layer Cross-Attention (CLCA) — Eqs. (7)–(9)
       Q from deep features; K/V from resized shallow features; MHSA
"""

from __future__ import annotations

from typing import Any

import torch
import torch.fft as fft
import torch.nn.functional as F
from torch import nn


# ── Frequency Attention (FA) ──────────────────────────────────────────────


class FrequencyAttention(nn.Module):
    """Frequency Attention via 2D RFFT amplitude channel gating.

    Matches paper Eqs. (1)–(3) and the reference ``FrequencyAttention``:
        F = RFFT2D(X)
        g = GAP(|F|)
        w = σ(W2 · ReLU(W1 · g))
        Y = X ⊙ w
    """

    def __init__(self, dim: int, reduction: int = 16) -> None:
        super().__init__()
        self.dim = dim
        reduced = max(1, dim // reduction)
        self.reduction = nn.Linear(dim, reduced, bias=False)
        self.expansion = nn.Linear(reduced, dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args/Returns: ``[B, C, H, W]``."""
        # RFFT over spatial dims → amplitude spectrum (shift-invariant)
        x_fft = fft.rfft2(x, dim=(-2, -1), norm="ortho")
        magnitude = torch.abs(x_fft)  # [B, C, H, W']

        # Average frequency energy per channel (Eq. 2)
        gap = magnitude.mean(dim=(-2, -1))  # [B, C]

        # Bottleneck channel attention (Eq. 3)
        weights = self.sigmoid(self.expansion(F.relu(self.reduction(gap))))
        weights = weights.view(-1, self.dim, 1, 1)
        return x * weights


# ── Saliency Feature Suppression (SFS) ────────────────────────────────────


class SaliencySuppression(nn.Module):
    """Soft saliency suppression with detached residual (Eqs. 4–6).

    Reference defaults: suppression_ratio α=0.2, radius R=1, soft value 0.1.
    Parameter-free (deterministic regulariser).
    """

    def __init__(
        self,
        dim: int = 0,  # kept for API compatibility; unused
        reduction: int = 4,  # unused (no learned gate in the paper)
        suppression_ratio: float = 0.2,
        suppression_radius: int = 1,
        soft_value: float = 0.1,
    ) -> None:
        super().__init__()
        self.suppression_ratio = suppression_ratio
        self.suppression_radius = suppression_radius
        self.soft_value = soft_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args/Returns: ``[B, C, H, W]``."""
        identity = x
        B, C, H, W = x.shape

        # Spatial saliency: mean |x| over channels → [B, 1, H, W] (Eq. 4)
        spatial_saliency = torch.mean(torch.abs(x), dim=1, keepdim=True)

        k = max(1, int(H * W * self.suppression_ratio))
        flat = spatial_saliency.view(B, -1)
        _, topk_indices = torch.topk(flat, k, dim=-1)  # [B, k]

        # Soft mask: 0.1 on suppressed neighbourhoods, 1 elsewhere (Eq. 5)
        mask = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        rows = topk_indices // W
        cols = topk_indices % W
        R = self.suppression_radius
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                rr = (rows + dr).clamp(0, H - 1)
                cc = (cols + dc).clamp(0, W - 1)
                # Advanced indexing per batch
                b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand_as(rr)
                mask[b_idx, 0, rr, cc] = self.soft_value

        x_suppressed = x * mask
        # Detached residual identity mapping (Eq. 6)
        return identity + (x_suppressed - identity).detach()


# ── Cross-Layer Cross-Attention (CLCA) ────────────────────────────────────


class CrossLayerChannelAttention(nn.Module):
    """Cross-Layer Cross-Attention (paper CLCA, Eqs. 7–9).

    Despite the historical class name ``CrossLayerChannelAttention``, this
    implements true cross-attention: Q from the deep (target) map, K/V from
    the shallow (guide) map after bilinear spatial alignment.
    """

    def __init__(
        self,
        guide_dim: int,
        target_dim: int,
        reduction: int = 16,  # unused; kept for config API compatibility
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert target_dim % num_heads == 0, "target_dim must be divisible by num_heads"
        self.guide_dim = guide_dim
        self.target_dim = target_dim
        self.num_heads = num_heads
        self.head_dim = target_dim // num_heads
        self.scale = self.head_dim**-0.5

        if guide_dim != target_dim:
            self.guide_proj = nn.Conv2d(guide_dim, target_dim, kernel_size=1, bias=False)
        else:
            self.guide_proj = nn.Identity()

        self.q = nn.Linear(target_dim, target_dim, bias=False)
        self.kv = nn.Linear(target_dim, target_dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(target_dim, target_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, guide: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args: guide/target ``[B, C, H, W]``. Returns recalibrated target."""
        B, C, H, W = target.shape

        guide = self.guide_proj(guide)
        guide = F.interpolate(guide, size=(H, W), mode="bilinear", align_corners=False)

        # Channel-last tokens for attention
        x = target.permute(0, 2, 3, 1).reshape(B, H * W, C)
        ctx = guide.permute(0, 2, 3, 1).reshape(B, H * W, C)

        q = self.q(x).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(ctx).reshape(B, H * W, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj_drop(self.proj(out))
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        # Locked baseline: residual inside CLCA; block also does x + clca(...).
        return target + out


# ── Factories ─────────────────────────────────────────────────────────────


def build_fa(config: Any, dim: int) -> FrequencyAttention:
    return FrequencyAttention(
        dim=dim,
        reduction=config.get("model.fa.reduction", 16),
    )


def build_sfs(config: Any, dim: int) -> SaliencySuppression:
    return SaliencySuppression(
        dim=dim,
        suppression_ratio=config.get("model.sfs.suppression_ratio", 0.2),
        suppression_radius=config.get("model.sfs.suppression_radius", 1),
    )


def build_clca(config: Any, guide_dim: int, target_dim: int) -> CrossLayerChannelAttention:
    return CrossLayerChannelAttention(
        guide_dim=guide_dim,
        target_dim=target_dim,
        num_heads=config.get("model.clca.num_heads", 4),
    )


__all__ = [
    "FrequencyAttention",
    "SaliencySuppression",
    "CrossLayerChannelAttention",
    "build_fa",
    "build_sfs",
    "build_clca",
]
