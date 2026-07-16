"""Transformer encoder for SemiWaferNet.

Projects CNN feature maps into token sequences and applies
transformer encoder blocks with multi-head self-attention.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchProjection(nn.Module):
    """Project CNN feature maps into transformer token sequences.

    Uses a 1×1 conv to project channels to embed_dim, then flattens
    spatial dimensions into a sequence.
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
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


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block: LN → MHA → residual → LN → MLP → residual."""

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

    Takes CNN feature maps and outputs transformed token sequences
    with spatial dimensions preserved for reshaping back.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_proj = PatchProjection(in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        # x: [B, C, H, W] from CNN backbone stage 4
        tokens, (H, W) = self.patch_proj(x)  # [B, H*W, embed_dim]

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return tokens, (H, W)