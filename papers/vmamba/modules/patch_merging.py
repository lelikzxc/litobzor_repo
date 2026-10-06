"""Patch merging module for FCS-VMamba.

Paper (Sec. 3.3): keep a fixed channel count during subsampling (rather than
doubling), halving only the spatial resolution — e.g. H/4×W/4×96 → H/32×W/32×96.

Default ``out_dim=dim`` implements this fixed-width merge. Setting
``out_dim=2*dim`` recovers classic VMamba / Swin doubling if needed.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchMerging(nn.Module):
    """2× spatial downsample with optional channel change.

    Unfolds 2×2 neighbourhoods (4·dim channels) and projects to ``out_dim``.

    Args:
        dim: Input channel dimension.
        out_dim: Output channel dimension. Defaults to ``dim`` (paper FCS).
        norm_layer: Normalisation layer (default ``nn.LayerNorm``).
    """

    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``[B, C, H, W]``. Returns ``[B, out_dim, H/2, W/2]``."""
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"Spatial dims ({H}, {W}) must be even"

        # VMamba order: top-left, bottom-left, top-right, bottom-right.
        x = torch.cat([
            x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2],
            x[:, :, 0::2, 1::2], x[:, :, 1::2, 1::2],
        ], dim=1).flatten(2).transpose(1, 2)

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, out_dim]

        return x.transpose(1, 2).reshape(B, self.out_dim, H // 2, W // 2)
