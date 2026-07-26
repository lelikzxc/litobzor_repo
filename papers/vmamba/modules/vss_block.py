"""VSS Block (VMamba State-Space Block) for FCS-VMamba.

Uses the official VMamba ``SS2D`` and ``Mlp`` implementations from
``papers.vmamba.kernels.vmamba_official`` while integrating the FCS-VMamba
specific modules: Frequency Attention (FA) and Saliency Feature Suppression
(SFS).

The block applies (matching the paper exactly):

    1. LayerNorm
    2. SS2D (2D Selective Scan)
    3. DropPath + residual
    4. FA (Frequency Attention) — frequency-domain attention
    5. SFS (Saliency Feature Suppression) — suppresses non-salient features
    6. LayerNorm
    7. MLP
    8. DropPath + residual

FA and SFS are inserted *between* the SS2D residual and the MLP path, exactly
as described in the FCS-VMamba paper (Sections 3.2, 3.3).
"""

from __future__ import annotations

import torch
from torch import nn

from papers.vmamba.kernels.vmamba_official import (
    Mlp as _OfficialMlp,
    SS2D as _OfficialSS2D,
    DropPath,
)
from papers.vmamba.modules.fcs_modules import FrequencyAttention, SaliencySuppression


# ── SS2D: 2D Selective Scan (official wrapper) ────────────────────────────


class SS2D(nn.Module):
    """2D Selective Scan for VMamba (wraps official implementation).

    Delegates to the official ``SS2D`` from ``vmamba_official.py`` with
    ``forward_type="v2"`` (default), ``d_state=16``, ``d_conv=3``, and
    ``initialize="v0"``.

    The official implementation provides:
        - 4-directional Cross Scan (left→right, right→left, top→bottom,
          bottom→top) via ``cross_scan_fn`` / ``cross_merge_fn``.
        - Depthwise convolution (``d_conv=3``) before SSM.
        - Selective scan with proper SSM recurrence.
        - Official ``mamba_init`` parameter initialisation.

    Args:
        dim: Input channel dimension.
        ssm_ratio: Expansion ratio for the SSM hidden state.
        dt_rank: Rank of the discretisation projection (``None`` =
            auto-compute as ``dim // 16``).
    """

    def __init__(
        self,
        dim: int,
        ssm_ratio: float = 2.0,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ssm_ratio = ssm_ratio
        self.dt_rank = dt_rank

        self.official = _OfficialSS2D(
            d_model=dim,
            d_state=16,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank if dt_rank is not None else "auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            initialize="v0",
            forward_type="v2",
            channel_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]`` (channel-first).

        Returns:
            Output tensor of shape ``[B, C, H, W]``.
        """
        return self.official(x)


# ── FCSVSSBlock: custom block with FA/SFS between SS2D and MLP ────────────


class FCSVSSBlock(nn.Module):
    """FCS-VMamba State-Space Block with FA and SFS between SS2D and MLP.

    Execution order (matches the FCS-VMamba paper exactly):

        1. LayerNorm
        2. SS2D (2D Selective Scan, official implementation)
        3. DropPath + residual
        4. FA (Frequency Attention) — frequency-domain attention
        5. SFS (Saliency Feature Suppression) — suppresses non-salient features
        6. LayerNorm
        7. MLP
        8. DropPath + residual

    FA and SFS are inserted *between* the SS2D residual and the MLP path,
    exactly as described in the FCS-VMamba paper (Sections 3.2, 3.3).

    All sub-modules (SS2D, MLP, DropPath) use the official VMamba
    implementations from ``papers.vmamba.kernels.vmamba_official``.

    The entire block operates in channel-first ``[B, C, H, W]`` format.

    Args:
        dim: Input/output channel dimension.
        num_heads: Number of attention heads (reserved for future use).
        ssm_ratio: SSM expansion ratio.
        mlp_ratio: MLP hidden dimension = ``dim * mlp_ratio``.
        drop_path: Stochastic depth drop rate.
        fa_reduction: Reduction ratio for FA (from config).
        sfs_reduction: Reduction ratio for SFS (from config).
        fa_enabled: Whether to enable FA.
        sfs_enabled: Whether to enable SFS.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        fa_reduction: int = 16,
        sfs_reduction: int = 4,
        fa_enabled: bool = True,
        sfs_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ssm_ratio = ssm_ratio
        self.mlp_ratio = mlp_ratio
        self.fa_enabled = fa_enabled
        self.sfs_enabled = sfs_enabled

        # ── SS2D path ────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(dim)
        self.op = _OfficialSS2D(
            d_model=dim,
            d_state=16,
            ssm_ratio=ssm_ratio,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            initialize="v0",
            forward_type="v2",
            channel_first=True,
        )
        self.drop_path = DropPath(drop_path)

        # ── Frequency Attention (FA) ────────────────────────────────────
        # Inserted after SS2D residual, before MLP path
        self.fa: nn.Module
        if fa_enabled:
            self.fa = FrequencyAttention(dim=dim, reduction=fa_reduction)
        else:
            self.fa = nn.Identity()

        # ── Saliency Feature Suppression (SFS) ──────────────────────────
        # Inserted after FA, before MLP path
        self.sfs: nn.Module
        if sfs_enabled:
            self.sfs = SaliencySuppression(dim=dim, reduction=sfs_reduction)
        else:
            self.sfs = nn.Identity()

        # ── MLP path ────────────────────────────────────────────────────
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _OfficialMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
            channels_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]`` (channel-first).

        Returns:
            Output tensor of shape ``[B, C, H, W]`` (channel-first).
        """
        B, C, H, W = x.shape

        # ── SS2D path ───────────────────────────────────────────────────
        # LayerNorm operates on channel-last, so permute
        identity = x
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] → [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] → [B, C, H, W]
        x = self.op(x)  # SS2D (channel-first)

        # ── FA: Frequency Attention ─────────────────────────────────────
        # Applied INSIDE the SS2D residual path, before DropPath
        # This matches the FCS-VMamba paper: LN → SS2D → FA → SFS → + residual
        x = self.fa(x)

        # ── SFS: Saliency Feature Suppression ───────────────────────────
        # Applied after FA, before DropPath + residual
        x = self.sfs(x)

        # DropPath + residual (FA and SFS are inside this residual path)
        x = identity + self.drop_path(x)

        # ── MLP path ────────────────────────────────────────────────────
        # MLP with channels_first=True uses Linear2d (1×1 conv equivalent)
        identity = x
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] → [B, H, W, C]
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] → [B, C, H, W]
        x = self.mlp(x)  # MLP (channel-first via Linear2d)
        x = identity + self.drop_path(x)

        return x


# ── VSSBlock: alias for backward compatibility ────────────────────────────


class VSSBlock(FCSVSSBlock):
    """Alias for ``FCSVSSBlock``.

    Retained for backward compatibility with existing code that imports
    ``VSSBlock``.  ``FCSVSSBlock`` is the canonical name.
    """
    pass


__all__ = ["SS2D", "VSSBlock", "FCSVSSBlock", "DropPath"]