"""ConvoFormer-UNet for wafer defect segmentation.

Paper (Electronics 2026, 15, 1437, Section 3.1):
    - Convolution-enhanced embedding + ConvoFormer encoder blocks
      (MSA + depthwise 3×3 local enhancement, Eq. 15)
    - Progressive decoder with transposed convolutions
    - 1×1 projections before skip fusion
    - Deep supervision (Eq. 17)

Target size ≈ 7.11M parameters (Table 8).
Input: 1×64×64 wafer maps → binary mask logits.
"""

from __future__ import annotations

import torch
from torch import nn

from papers.semiwafernet.modules.transformer import ConvoFormerBlock


class DoubleConv(nn.Module):
    """Two 3×3 convolutions with BN + GELU (U-Net style)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Stride-2 downsample + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            DoubleConv(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """Transposed-conv upsample + 1×1 gap reduction + skip fusion + DoubleConv.

    Matches Section 3.1.2: transposed convolutions, 1×1 convolutions before
    fusing encoder and decoder features, skip connections.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # 1×1 reduces semantic gap of the skip before fusion
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
        self.fuse = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.skip_proj(skip)
        # Handle odd spatial sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class ConvoFormerUNet(nn.Module):
    """Lightweight hybrid encoder–decoder for binary wafer segmentation.

    Default width (base_channels=48, embed_dim=160, L=4) yields ≈7.11M
    parameters, matching paper Table 8.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 48,
        embed_dim: int = 160,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # ConvEmbed local prior (Section 3.1.1): 3×3 conv + GELU at full resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            DoubleConv(c1, c1),
        )

        # Hierarchical encoder (produces multi-scale skips)
        self.down1 = Down(c1, c2)  # H/2
        self.down2 = Down(c2, c3)  # H/4
        self.down3 = Down(c3, c4)  # H/8  (8×8 for 64×64 input)

        # Bottleneck: project into Transformer space (ConvEmbed stride-8 analogue
        # when already at H/8) + ConvoFormer blocks with LocalConv (Eq. 15)
        self.bottleneck_in = nn.Conv2d(c4, embed_dim, kernel_size=1, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                ConvoFormerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )
        self.bottleneck_norm = nn.LayerNorm(embed_dim)
        self.bottleneck_out = nn.Conv2d(embed_dim, c4, kernel_size=1, bias=False)

        # Progressive decoder with transposed convs + skip fusion
        self.up3 = Up(c4, c3, c3)  # → H/4
        self.up2 = Up(c3, c2, c2)  # → H/2
        self.up1 = Up(c2, c1, c1)  # → H

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)
        # Auxiliary heads for deep supervision (Eq. 17)
        self.aux_head1 = nn.Conv2d(c2, num_classes, kernel_size=1)  # after up2
        self.aux_head2 = nn.Conv2d(c3, num_classes, kernel_size=1)  # after up3

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Args: ``[B, 1, H, W]``. Returns full-res logits or aux dict."""
        # Encoder
        s1 = self.stem(x)       # [B, c1, H, W]
        s2 = self.down1(s1)     # [B, c2, H/2, W/2]
        s3 = self.down2(s2)     # [B, c3, H/4, W/4]
        s4 = self.down3(s3)     # [B, c4, H/8, W/8]

        # ConvoFormer bottleneck
        B, _, H8, W8 = s4.shape
        tokens = self.bottleneck_in(s4).flatten(2).transpose(1, 2)  # [B, N, D]
        n_tokens = tokens.shape[1]
        if self.pos_embed.shape[1] == n_tokens:
            tokens = tokens + self.pos_embed
        else:
            # Interpolate positional embeddings for non-64×64 inputs
            pos = self.pos_embed.transpose(1, 2).reshape(1, self.embed_dim, 8, 8)
            pos = nn.functional.interpolate(pos, size=(H8, W8), mode="bilinear", align_corners=False)
            tokens = tokens + pos.flatten(2).transpose(1, 2)

        for block in self.blocks:
            tokens = block(tokens, spatial_shape=(H8, W8))
        tokens = self.bottleneck_norm(tokens)
        feat = tokens.transpose(1, 2).reshape(B, self.embed_dim, H8, W8)
        feat = self.bottleneck_out(feat)

        # Decoder
        d3 = self.up3(feat, s3)   # H/4
        aux2 = self.aux_head2(d3)
        d2 = self.up2(d3, s2)     # H/2
        aux1 = self.aux_head1(d2)
        d1 = self.up1(d2, s1)     # H
        main = self.head(d1)

        if return_aux:
            return {"main": main, "aux1": aux1, "aux2": aux2}
        return main


__all__ = ["ConvoFormerUNet", "DoubleConv", "Down", "Up"]
