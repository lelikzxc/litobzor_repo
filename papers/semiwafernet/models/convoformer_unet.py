"""ConvoFormer-UNet for wafer defect segmentation.

Paper (Electronics 2026, 15, 1437, Section 3.1):
    - ConvEmbed on the input image (3×3 conv + 8×8 stride-8 conv, §3.1.1)
    - ConvoFormer encoder blocks (MSA + depthwise 3×3, Eq. 15)
    - Parallel lightweight CNN encoder for multi-scale skip connections
    - Progressive decoder with transposed convolutions + 1×1 skip fusion
    - Deep supervision (Eq. 17)

    Target size ≈ 7.11M parameters (Table 8, base_channels=66).
Input: 1×64×64 wafer maps → binary mask logits.
"""

from __future__ import annotations

import torch
from torch import nn

from papers.semiwafernet.modules.transformer import ConvoFormerBlock, ConvEmbed


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
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
        self.fuse = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.skip_proj(skip)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class ConvoFormerUNet(nn.Module):
    """Lightweight hybrid encoder–decoder for binary wafer segmentation.

    Architecture (paper-faithful):
        1. ConvEmbed on raw input → 8×8 token grid (§3.1.1)
        2. ConvoFormer blocks with LocalConv (Eq. 15)
        3. Parallel CNN skip encoder (64 → 32 → 16) for decoder fusion
        4. Progressive transposed-conv decoder + deep supervision (Eq. 17)

    Default width (base_channels=66, embed_dim=160, L=4) yields ≈7.11M
    parameters, matching paper Table 8.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 66,
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

        # Parallel CNN skip encoder (multi-scale features for decoder §3.1.2)
        self.skip_stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            DoubleConv(c1, c1),
        )
        self.skip_down1 = Down(c1, c2)  # H/2
        self.skip_down2 = Down(c2, c3)  # H/4

        # ConvEmbed on raw input (§3.1.1): 3×3 → 8×8 stride-8 → N=64 tokens
        self.conv_embed = ConvEmbed(in_channels=in_channels, embed_dim=embed_dim)

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
        self.up3 = Up(c4, c3, c3)  # 8×8 → 16×16
        self.up2 = Up(c3, c2, c2)  # 16×16 → 32×32
        self.up1 = Up(c2, c1, c1)  # 32×32 → 64×64

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.aux_head1 = nn.Conv2d(c2, num_classes, kernel_size=1)
        self.aux_head2 = nn.Conv2d(c3, num_classes, kernel_size=1)

    def _apply_pos_embed(
        self, tokens: torch.Tensor, spatial_shape: tuple[int, int]
    ) -> torch.Tensor:
        """Add learnable positional embeddings, with interpolation if needed."""
        n_tokens = tokens.shape[1]
        if self.pos_embed.shape[1] == n_tokens:
            return tokens + self.pos_embed
        h, w = spatial_shape
        pos = self.pos_embed.transpose(1, 2).reshape(1, self.embed_dim, 8, 8)
        pos = nn.functional.interpolate(
            pos, size=(h, w), mode="bilinear", align_corners=False
        )
        return tokens + pos.flatten(2).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Args: ``[B, 1, H, W]``. Returns full-res logits or aux dict."""
        # Multi-scale skip features (H, H/2, H/4)
        s1 = self.skip_stem(x)
        s2 = self.skip_down1(s1)
        s3 = self.skip_down2(s2)

        # ConvEmbed + ConvoFormer bottleneck (§3.1.1, Eq. 15)
        tokens, (h8, w8) = self.conv_embed(x)
        tokens = self._apply_pos_embed(tokens, (h8, w8))
        for block in self.blocks:
            tokens = block(tokens, spatial_shape=(h8, w8))
        tokens = self.bottleneck_norm(tokens)

        b = x.shape[0]
        feat = tokens.transpose(1, 2).reshape(b, self.embed_dim, h8, w8)
        feat = self.bottleneck_out(feat)

        # Decoder
        d3 = self.up3(feat, s3)
        aux2 = self.aux_head2(d3)
        d2 = self.up2(d3, s2)
        aux1 = self.aux_head1(d2)
        d1 = self.up1(d2, s1)
        main = self.head(d1)

        if return_aux:
            return {"main": main, "aux1": aux1, "aux2": aux2}
        return main


__all__ = ["ConvoFormerUNet", "DoubleConv", "Down", "Up"]
