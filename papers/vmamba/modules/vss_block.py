"""VSS / FSS Block for FCS-VMamba.

Paper FSSLayer workflow (Sec. 3.3) and author reference ``VSSBlock``:

    LN → FA → SS2D → SFS → DropPath + residual → (optional CLCA)

Unlike vanilla VMamba, the FCS FSSLayer does **not** include an MLP branch.
CLCA is applied only on the last block of stages after the first, using the
previous stage's features as context.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from papers.vmamba.kernels.vmamba_official import (
    SS2D as _OfficialSS2D,
    DropPath,
)
from papers.vmamba.modules.fcs_modules import (
    FrequencyAttention,
    SaliencySuppression,
    CrossLayerChannelAttention,
)


class SS2D(nn.Module):
    """Thin wrapper around the vendored official SS2D (channel-first)."""

    def __init__(
        self,
        dim: int,
        ssm_ratio: float = 2.0,
        dt_rank: int | None = None,
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.official = _OfficialSS2D(
            d_model=dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank if dt_rank is not None else "auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            initialize="v0",
            # v03 = oflex + force_fp32 (v3 disables fp32 → unstable on wafer maps)
            forward_type="v03",
            channel_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.official(x)


class FCSVSSBlock(nn.Module):
    """FCS FSSLayer: LN → FA → SS2D → SFS → residual → optional CLCA."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,  # unused; kept for API compat
        drop_path: float = 0.0,
        fa_reduction: int = 16,
        sfs_reduction: int = 4,  # unused (SFS is param-free)
        fa_enabled: bool = True,
        sfs_enabled: bool = True,
        clca_enabled: bool = False,
        clca_num_heads: int = 4,
        d_state: int = 16,
        sfs_ratio: float = 0.2,
        sfs_radius: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.fa_enabled = fa_enabled
        self.sfs_enabled = sfs_enabled
        self.clca_enabled = clca_enabled

        self.norm = nn.LayerNorm(dim)
        self.op = _OfficialSS2D(
            d_model=dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            initialize="v0",
            # v03: oflex + force_fp32. Plain "v3" sets force_fp32=False and
            # explodes LayerNorm grads (~1e18) on poorly encoded wafer maps.
            forward_type="v03",
            channel_first=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.fa: nn.Module = (
            FrequencyAttention(dim=dim, reduction=fa_reduction) if fa_enabled else nn.Identity()
        )
        self.sfs: nn.Module = (
            SaliencySuppression(
                dim=dim,
                suppression_ratio=sfs_ratio,
                suppression_radius=sfs_radius,
            )
            if sfs_enabled
            else nn.Identity()
        )

        self.norm_clca: nn.Module
        self.clca: nn.Module
        if clca_enabled:
            self.norm_clca = nn.LayerNorm(dim)
            self.clca = CrossLayerChannelAttention(
                guide_dim=dim,
                target_dim=dim,
                num_heads=clca_num_heads,
            )
        else:
            self.norm_clca = nn.Identity()
            self.clca = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args: ``x`` / ``context`` as ``[B, C, H, W]``."""
        identity = x

        # LN (channel-last) → FA → SS2D → SFS
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.fa(x)
        x = self.op(x)
        x = self.sfs(x)

        x = identity + self.drop_path(x)

        # Optional CLCA with previous-stage context
        if self.clca_enabled and context is not None:
            x_n = x.permute(0, 2, 3, 1).contiguous()
            x_n = self.norm_clca(x_n)
            x_n = x_n.permute(0, 3, 1, 2).contiguous()
            x = x + self.clca(context, x_n)

        return x


class VSSBlock(FCSVSSBlock):
    """Alias for backward compatibility."""

    pass


__all__ = ["SS2D", "VSSBlock", "FCSVSSBlock", "DropPath"]
