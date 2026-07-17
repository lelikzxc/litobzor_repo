"""FCS-VMamba backbone model.

Hierarchical VMamba backbone with 4 stages of VSSBlocks, patch merging
between stages, Frequency Attention (FA), Saliency Feature Suppression (SFS),
Cross-Layer Channel Attention (CLCA), and a classification head.

Architecture:
    Input [B, 3, H, W]
    → PatchEmbed2D (stride=4)
    → Stage 1: VSSBlock × N1 (+ FA + SFS)  [B, C1, H/4, W/4]
    → PatchMerging
    → Stage 2: VSSBlock × N2 (+ FA + SFS)  [B, C2, H/8, W/8]
    → PatchMerging
    → Stage 3: VSSBlock × N3 (+ FA + SFS)  [B, C3, H/16, W/16]
    → PatchMerging
    → Stage 4: VSSBlock × N4 (+ FA + SFS)  [B, C4, H/32, W/32]
    → CLCA: Cross-Layer Channel Attention (multi-scale aggregation)
    → Global Average Pooling
    → Linear classifier
    → Logits [B, num_classes]

All architectural parameters come from ``configs/config.yaml``.
No hardcoded values.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from papers.vmamba.modules.patch_embed import PatchEmbed2D
from papers.vmamba.modules.patch_merging import PatchMerging
from papers.vmamba.modules.vss_block import VSSBlock
from papers.vmamba.modules.fcs_modules import CrossLayerChannelAttention


class FCSVMamba(nn.Module):
    """FCS-VMamba backbone for wafer defect classification.

    Hierarchical VMamba with configurable depth, embedding dimensions,
    SSM ratios per stage, and FCS-specific modules (FA, SFS, CLCA).

    Args:
        in_channels: Number of input image channels.
        image_size: Input image size (assumed square).
        embed_dim: Base embedding dimension (doubled after each merge).
        depths: Number of VSSBlocks per stage (list of 4 ints).
        num_heads: Number of attention heads per stage (list of 4 ints).
        ssm_ratio: SSM expansion ratio.
        mlp_ratio: MLP hidden dimension ratio.
        drop_path_rate: Stochastic depth drop rate.
        num_classes: Number of output classes.
        fa_enabled: Enable Frequency Attention.
        fa_reduction: Reduction ratio for FA.
        sfs_enabled: Enable Saliency Feature Suppression.
        sfs_reduction: Reduction ratio for SFS.
        clca_enabled: Enable Cross-Layer Channel Attention.
        clca_reduction: Reduction ratio for CLCA.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        embed_dim: int = 96,
        depths: tuple[int, ...] | list[int] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] | list[int] = (3, 6, 12, 24),
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        num_classes: int = 8,
        fa_enabled: bool = True,
        fa_reduction: int = 16,
        sfs_enabled: bool = True,
        sfs_reduction: int = 4,
        clca_enabled: bool = True,
        clca_reduction: int = 16,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.ssm_ratio = ssm_ratio
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self.fa_enabled = fa_enabled
        self.fa_reduction = fa_reduction
        self.sfs_enabled = sfs_enabled
        self.sfs_reduction = sfs_reduction
        self.clca_enabled = clca_enabled
        self.clca_reduction = clca_reduction

        assert len(self.depths) == 4, f"Expected 4 depths, got {len(self.depths)}"
        assert len(self.num_heads) == 4, f"Expected 4 num_heads, got {len(self.num_heads)}"

        # Patch embedding (stride 4)
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=4,
        )

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.stages = nn.ModuleList()
        self.mergings = nn.ModuleList()

        curr_dim = embed_dim
        depth_idx = 0
        stage_dims: list[int] = []

        for stage_idx in range(4):
            stage_dims.append(curr_dim)

            # VSSBlocks for this stage (with FA + SFS integrated)
            stage_blocks: list[nn.Module] = []
            for _ in range(self.depths[stage_idx]):
                stage_blocks.append(
                    VSSBlock(
                        dim=curr_dim,
                        num_heads=self.num_heads[stage_idx],
                        ssm_ratio=ssm_ratio,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[depth_idx],
                        fa_reduction=fa_reduction,
                        sfs_reduction=sfs_reduction,
                        fa_enabled=fa_enabled,
                        sfs_enabled=sfs_enabled,
                    )
                )
                depth_idx += 1
            self.stages.append(nn.Sequential(*stage_blocks))

            # Patch merging (except after the last stage)
            if stage_idx < 3:
                self.mergings.append(PatchMerging(dim=curr_dim))
                curr_dim *= 2

        # Final embedding dimension (after all mergings)
        self.final_dim = curr_dim

        # ── Cross-Layer Channel Attention (CLCA) ────────────────────────
        # Connects earlier stage features to the final stage for multi-scale
        # context aggregation. Applied after all stages, before GAP.
        self.clca_enabled = clca_enabled
        self.clca_modules = nn.ModuleList()
        if clca_enabled:
            # CLCA connects each earlier stage to the final stage
            for stage_idx in range(3):  # stages 0, 1, 2 → stage 3
                self.clca_modules.append(
                    CrossLayerChannelAttention(
                        guide_dim=stage_dims[stage_idx],
                        target_dim=self.final_dim,
                        reduction=clca_reduction,
                    )
                )
        else:
            self.clca_modules = nn.ModuleList([nn.Identity() for _ in range(3)])

        # Store stage dims for CLCA forward
        self._stage_dims = stage_dims

        # Classification head
        self.norm = nn.LayerNorm(self.final_dim)
        self.head = nn.Linear(self.final_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Logits tensor of shape [B, num_classes] (no Softmax).
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/4, W/4]

        # Stages with patch merging, collecting intermediate features for CLCA
        stage_features: list[torch.Tensor] = []

        for stage_idx in range(4):
            x = self.stages[stage_idx](x)
            stage_features.append(x)
            if stage_idx < 3:
                x = self.mergings[stage_idx](x)

        # ── CLCA: Cross-Layer Channel Attention ─────────────────────────
        # Aggregate multi-scale context from earlier stages into final stage
        if self.clca_enabled:
            for stage_idx in range(3):
                x = self.clca_modules[stage_idx](stage_features[stage_idx], x)

        # Global average pooling
        x = x.mean(dim=[-2, -1])  # [B, final_dim]

        # Classification head
        x = self.norm(x)
        x = self.head(x)  # [B, num_classes] — logits only, no Softmax

        return x

    @classmethod
    def from_config(cls, config: Any) -> FCSVMamba:
        """Instantiate model from a config object.

        Args:
            config: Config object with ``model.backbone.*``,
                    ``model.input.*``, ``model.fa.*``, ``model.sfs.*``,
                    ``model.clca.*``, and ``training.num_classes``
                    accessible via ``config.get("...")``.

        Returns:
            Configured FCSVMamba instance.
        """
        return cls(
            in_channels=config.get("model.input.channels", 3),
            image_size=config.get("model.input.image_size", 224),
            embed_dim=config.get("model.backbone.embed_dim", 96),
            depths=config.get("model.backbone.depths", [2, 2, 6, 2]),
            num_heads=config.get("model.backbone.num_heads", [3, 6, 12, 24]),
            ssm_ratio=config.get("model.backbone.ssm_ratio", 2.0),
            mlp_ratio=config.get("model.backbone.mlp_ratio", 4.0),
            drop_path_rate=config.get("model.backbone.drop_path_rate", 0.2),
            num_classes=config.get("training.num_classes", 8),
            fa_enabled=config.get("model.fa.enabled", True),
            fa_reduction=config.get("model.fa.reduction", 16),
            sfs_enabled=config.get("model.sfs.enabled", True),
            sfs_reduction=config.get("model.sfs.reduction", 4),
            clca_enabled=config.get("model.clca.enabled", True),
            clca_reduction=config.get("model.clca.reduction", 16),
        )