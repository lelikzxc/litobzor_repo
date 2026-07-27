"""Transformer encoder for SemiWaferNet.

Implements two variants:

1. **HybridCNN-ViT (classification)**: Standard ViT encoder that processes
   CNN backbone features via 1×1 conv projection → transformer blocks.

2. **ConvoFormer-UNet (segmentation)**: Convolution-enhanced Transformer with:
   - ConvEmbed: 3×3 conv (GELU) → 8×8 conv stride 8 → token grid
   - Each block: self-attention + depthwise 3×3 conv fusion (Equation 15)
"""

from __future__ import annotations

import torch
from torch import nn


class ConvEmbed(nn.Module):
    """Convolution-enhanced patch embedding for ConvoFormer-UNet.

    Matches Section 3.1.1 from the paper:
        1. 3×3 convolution to extract localized features (GELU activation)
        2. 8×8 convolution with stride 8 to project into embedding space

    Produces an 8×8 token grid (N=64 tokens) from a 64×64 input.
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        # 3×3 conv for local feature extraction
        self.conv1 = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(embed_dim // 2)
        self.act = nn.GELU()

        # 8×8 conv stride 8 for patch projection
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=8, stride=8, bias=False)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W] (typically 1×64×64 for segmentation).

        Returns:
            tokens: [B, N, embed_dim] token sequence (N = H/8 * W/8 = 64).
            spatial: (H_out, W_out) spatial dimensions of the token grid.
        """
        # Local feature extraction
        x = self.conv1(x)  # [B, embed_dim/2, H, W]
        x = self.norm1(x)
        x = self.act(x)

        # Patch projection
        x = self.conv2(x)  # [B, embed_dim, H/8, W/8]
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        x = self.norm2(x)

        return x, (H, W)


class PatchProjection(nn.Module):
    """Project CNN feature maps into transformer token sequences.

    Uses a 1×1 conv to project channels to embed_dim, then flattens
    spatial dimensions into a sequence. Used for the classification path
    (HybridCNN-ViT) where CNN features are already at the right resolution.

    For segmentation (ConvoFormer-UNet), use ConvEmbed instead.
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            tokens: [B, H*W, embed_dim] token sequence.
            spatial: (H, W) spatial dimensions.
        """
        x = self.proj(x)  # [B, embed_dim, H, W]
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        x = self.norm(x)
        return x, (H, W)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional dropout."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerMLP(nn.Module):
    """MLP for transformer encoder block: Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, embed_dim: int, mlp_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvoFormerBlock(nn.Module):
    """ConvoFormer encoder block with self-attention + depthwise conv fusion.

    Matches Equation (15) from the paper:
        Z'_l = MSA(LN(Z_l)) + Conv_dw_3×3(Z_l)

    The depthwise 3×3 convolution provides local enhancement alongside
    global self-attention, preserving boundary details for segmentation.

    For the classification path (no spatial tokens), the depthwise conv
    is skipped (operates on 1×1 spatial grid).
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        # Depthwise 3×3 convolution for local enhancement
        # Operates on spatial tokens reshaped to [B, D, H, W]
        self.depthwise_conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1,
            groups=embed_dim,  # depthwise
            bias=False,
        )
        self.norm_conv = nn.BatchNorm2d(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = TransformerMLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, spatial_shape: tuple[int, int] | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Token sequence [B, N, embed_dim].
            spatial_shape: Optional (H, W) for depthwise conv.
                If None, conv is skipped (classification mode).

        Returns:
            Output token sequence [B, N, embed_dim].
        """
        # Self-attention path
        attn_out = self.attn(self.norm1(x))

        # Local enhancement path (depthwise conv)
        if spatial_shape is not None and spatial_shape[0] > 1 and spatial_shape[1] > 1:
            H, W = spatial_shape
            B, N, D = x.shape
            # Reshape tokens to spatial feature map
            x_spatial = x.transpose(1, 2).reshape(B, D, H, W)  # [B, D, H, W]
            conv_out = self.depthwise_conv(x_spatial)
            conv_out = self.norm_conv(conv_out)
            conv_out = conv_out.flatten(2).transpose(1, 2)  # [B, N, D]
        else:
            conv_out = 0.0

        # Fuse: MSA + Conv_dw
        x = x + attn_out + conv_out

        # MLP path
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerEncoderBlock(nn.Module):
    """Standard transformer encoder block: LN → MHA → residual → LN → MLP → residual.

    Used for the classification path (HybridCNN-ViT).
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = TransformerMLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder with patch projection + N encoder blocks.

    Supports two modes:
        - Classification: uses PatchProjection + standard TransformerEncoderBlock
        - Segmentation: uses ConvEmbed + ConvoFormerBlock (with depthwise conv)

    Args:
        in_channels: Input channel dimension from CNN backbone.
        embed_dim: Transformer embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        mlp_ratio: MLP hidden dimension ratio.
        dropout: Dropout rate.
        use_conv_embed: If True, use ConvEmbed (segmentation mode).
            If False, use PatchProjection (classification mode).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_conv_embed: bool = False,
    ) -> None:
        super().__init__()

        # Patch embedding
        if use_conv_embed:
            self.patch_embed = ConvEmbed(in_channels=1, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchProjection(in_channels, embed_dim)

        # Transformer blocks
        if use_conv_embed:
            # ConvoFormer blocks with depthwise conv fusion
            self.blocks = nn.ModuleList([
                ConvoFormerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ])
        else:
            # Standard ViT blocks
            self.blocks = nn.ModuleList([
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ])

        self.norm = nn.LayerNorm(embed_dim)
        self._use_conv_embed = use_conv_embed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Forward pass.

        Args:
            x: Input tensor.
                Classification mode: [B, C, H, W] from CNN backbone stage 2.
                Segmentation mode: [B, 1, H, W] raw input.

        Returns:
            tokens: [B, N, embed_dim] transformer output tokens.
            spatial: (H, W) spatial dimensions of the token grid.
        """
        tokens, (H, W) = self.patch_embed(x)  # [B, N, embed_dim]

        for block in self.blocks:
            if isinstance(block, ConvoFormerBlock):
                tokens = block(tokens, spatial_shape=(H, W))
            else:
                tokens = block(tokens)

        tokens = self.norm(tokens)
        return tokens, (H, W)


__all__ = [
    "ConvEmbed",
    "PatchProjection",
    "MultiHeadSelfAttention",
    "TransformerMLP",
    "ConvoFormerBlock",
    "TransformerEncoderBlock",
    "TransformerEncoder",
]