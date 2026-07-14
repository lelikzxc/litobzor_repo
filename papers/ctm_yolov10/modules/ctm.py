"""Context Transformer Module (CTM) for wafer defect detection.

Implements the CTM module described in:

    "Wafer Defect Detection Technology Based on CTM-IYOLOv10 Network"

The module receives convolutional feature maps [B, C, H, W], applies a
lightweight self-attention + feed-forward block with residual connections,
and returns [B, C_out, H, W] preserving spatial dimensions.

Architecture:
    1. Feature projection: Conv2d(C→C) + BatchNorm + Activation
    2. Flatten spatial dims → tokens [B, N, C] where N = H*W
    3. Self-attention block (ContextAttention)
    4. Feed-forward block (ContextMLP)
    5. Restore spatial dims → [B, C, H, W]
    6. Final projection: Conv2d + BatchNorm
    7. Residual addition: output = input + transformed_features
"""

from __future__ import annotations

import torch
from torch import nn


class ContextAttention(nn.Module):
    """Lightweight multi-head self-attention for context tokens.

    Projects input tokens to Q, K, V, computes scaled dot-product attention,
    and returns the attended representation.

    Args:
        dim: Token embedding dimension.
        num_heads: Number of attention heads (must divide ``dim``).
        dropout: Dropout rate applied after attention.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, N, dim].

        Returns:
            Attended tensor of shape [B, N, dim].
        """
        B, N, D = x.shape

        # Project to Q, K, V and reshape for multi-head attention
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, d]
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, d]
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, d]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # [B, H, N, d]
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)  # [B, N, dim]
        out = self.out_proj(out)

        return out


class ContextMLP(nn.Module):
    """Two-layer feed-forward network with GELU activation.

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, N, dim].

        Returns:
            Output tensor of shape [B, N, dim].
        """
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CTM(nn.Module):
    """Context Transformer Module.

    Enhances convolutional feature maps with contextual self-attention.

    The module:
        1. Projects input features with Conv2d + BN + SiLU
        2. Flattens to token sequence [B, N, C]
        3. Applies self-attention with residual connection
        4. Applies MLP with residual connection
        5. Restores spatial structure [B, C, H, W]
        6. Final Conv2d + BN projection
        7. Residual addition with input

    Args:
        dim: Input/output channel dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension = ``dim * mlp_ratio``.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # 1. Feature projection
        self.feature_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

        # 3. Self-attention block
        self.norm1 = nn.LayerNorm(dim)
        self.attention = ContextAttention(dim, num_heads=num_heads, dropout=dropout)

        # 4. Feed-forward block
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ContextMLP(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

        # 6. Final projection
        self.final_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Output tensor of shape [B, C, H, W] (same spatial resolution).
        """
        identity = x

        # 1. Feature projection
        x = self.feature_proj(x)  # [B, C, H, W]

        # 2. Flatten to tokens
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C] where N = H*W

        # 3. Self-attention with residual
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out

        # 4. MLP with residual
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        # 5. Restore spatial structure
        x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]

        # 6. Final projection
        x = self.final_proj(x)

        # 7. Residual addition with input
        out = identity + x

        return out