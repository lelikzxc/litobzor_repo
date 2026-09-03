"""FCS-VMamba backbone model.

Lightweight hierarchical VMamba adapted for wafer defect classification,
matching the FCS-VMamba paper (J. Imaging 2026):

    - PatchEmbed2D (stride 4) → 96-d tokens
    - 4 stages with fixed channel width (no doubling on merge)
    - FSSLayer: LN → FA → SS2D → SFS → residual → (CLCA on later stages)
    - GAP → LayerNorm → Linear classifier

Default config targets ~1.2M parameters (paper Table 3).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from papers.vmamba.modules.patch_embed import PatchEmbed2D
from papers.vmamba.modules.patch_merging import PatchMerging
from papers.vmamba.modules.vss_block import FCSVSSBlock


class FCSVMamba(nn.Module):
    """FCS-VMamba for wafer defect classification.

    Args:
        in_channels: Input image channels.
        image_size: Assumed square input size.
        embed_dim: Channel width kept fixed across stages (paper: 96).
        depths: Blocks per stage (paper/reference default: [2, 2, 2, 2]).
        num_heads: Legacy per-stage heads list (CLCA uses ``clca_num_heads``).
        ssm_ratio: SS2D expansion ratio.
        mlp_ratio: Unused (FSSLayer has no MLP); kept for API compat.
        drop_path_rate: Stochastic depth.
        num_classes: Output classes (WM-811K: 9).
        fa_enabled / sfs_enabled / clca_enabled: Ablation switches.
        fixed_channels: If True (paper), patch merge keeps ``embed_dim``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        embed_dim: int = 96,
        depths: tuple[int, ...] | list[int] = (2, 2, 2, 2),
        num_heads: tuple[int, ...] | list[int] = (3, 6, 12, 24),
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        num_classes: int = 9,
        fa_enabled: bool = True,
        fa_reduction: int = 16,
        sfs_enabled: bool = True,
        sfs_reduction: int = 4,
        sfs_ratio: float = 0.2,
        sfs_radius: int = 1,
        clca_enabled: bool = True,
        clca_reduction: int = 16,
        clca_num_heads: int = 4,
        d_state: int = 16,
        fixed_channels: bool = True,
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
        self.fixed_channels = fixed_channels

        assert len(self.depths) == 4, f"Expected 4 depths, got {len(self.depths)}"

        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=4,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]

        self.stages = nn.ModuleList()
        self.mergings = nn.ModuleList()

        curr_dim = embed_dim
        depth_idx = 0
        stage_dims: list[int] = []

        for stage_idx in range(4):
            stage_dims.append(curr_dim)
            # CLCA from stage 1 onward, only on the last block of the stage
            stage_use_clca = clca_enabled and stage_idx > 0

            blocks: list[nn.Module] = []
            for block_i in range(self.depths[stage_idx]):
                is_last = block_i == self.depths[stage_idx] - 1
                blocks.append(
                    FCSVSSBlock(
                        dim=curr_dim,
                        num_heads=self.num_heads[stage_idx] if stage_idx < len(self.num_heads) else 4,
                        ssm_ratio=ssm_ratio,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[depth_idx],
                        fa_reduction=fa_reduction,
                        sfs_reduction=sfs_reduction,
                        fa_enabled=fa_enabled,
                        sfs_enabled=sfs_enabled,
                        clca_enabled=stage_use_clca and is_last,
                        clca_num_heads=clca_num_heads,
                        d_state=d_state,
                        sfs_ratio=sfs_ratio,
                        sfs_radius=sfs_radius,
                    )
                )
                depth_idx += 1
            self.stages.append(nn.ModuleList(blocks))

            if stage_idx < 3:
                out_dim = curr_dim if fixed_channels else curr_dim * 2
                self.mergings.append(PatchMerging(dim=curr_dim, out_dim=out_dim))
                curr_dim = out_dim

        self.final_dim = curr_dim
        self._stage_dims = stage_dims

        self.norm = nn.LayerNorm(self.final_dim)
        self.head = nn.Linear(self.final_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits ``[B, num_classes]`` (no Softmax)."""
        x = self.patch_embed(x)

        prev_features: torch.Tensor | None = None
        for stage_idx in range(4):
            for block in self.stages[stage_idx]:
                # Pass previous-stage map into CLCA-enabled (last) blocks
                if getattr(block, "clca_enabled", False) and prev_features is not None:
                    x = block(x, context=prev_features)
                else:
                    x = block(x)

            if stage_idx < 3:
                prev_features = x
                x = self.mergings[stage_idx](x)

        x = x.mean(dim=(-2, -1))  # GAP
        x = self.norm(x)
        return self.head(x)

    @classmethod
    def from_config(cls, config: Any) -> FCSVMamba:
        """Build from YAML / EngineConfig."""
        return cls(
            in_channels=config.get("model.input.channels", 3),
            image_size=config.get("model.input.image_size", 224),
            embed_dim=config.get("model.backbone.embed_dim", 96),
            depths=config.get("model.backbone.depths", [2, 2, 2, 2]),
            num_heads=config.get("model.backbone.num_heads", [3, 6, 12, 24]),
            ssm_ratio=config.get("model.backbone.ssm_ratio", 2.0),
            mlp_ratio=config.get("model.backbone.mlp_ratio", 4.0),
            drop_path_rate=config.get("model.backbone.drop_path_rate", 0.1),
            num_classes=config.get("model.num_classes", config.get("training.num_classes", 9)),
            fa_enabled=config.get("model.fa.enabled", True),
            fa_reduction=config.get("model.fa.reduction", 16),
            sfs_enabled=config.get("model.sfs.enabled", True),
            sfs_reduction=config.get("model.sfs.reduction", 4),
            sfs_ratio=config.get("model.sfs.suppression_ratio", 0.2),
            sfs_radius=config.get("model.sfs.suppression_radius", 1),
            clca_enabled=config.get("model.clca.enabled", True),
            clca_reduction=config.get("model.clca.reduction", 16),
            clca_num_heads=config.get("model.clca.num_heads", 4),
            d_state=config.get("model.backbone.d_state", 16),
            fixed_channels=config.get("model.backbone.fixed_channels", True),
        )
