"""MixVisionTransformer (MiT) backbone for SegFormer.

Implements the hierarchical MiT encoder from:
    SegFormer: Simple and Efficient Design for Semantic Segmentation
    with Transformers (NeurIPS 2021)

Components:
    - OverlapPatchEmbed: Convolutional patch embedding with overlapping kernels
    - EfficientSelfAttention: Self-attention with spatial reduction of K, V
    - MixFFN: MLP with 3×3 depthwise convolution between two linear layers
    - TransformerBlock: LN → EfficientSelfAttention → residual → LN → MixFFN → residual
    - MiTBackbone: 4 hierarchical stages producing multi-scale feature maps

All modules operate in channel-first format [B, C, H, W] except where noted.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


# ── OverlapPatchEmbed ──────────────────────────────────────────────────────


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding with 7×7 convolution.

    Maps an input image or feature map to patch tokens with overlapping
    receptive fields. Unlike ViT's non-overlapping patch embedding, this
    uses stride < kernel_size to create overlapping patches.

    Args:
        in_channels: Number of input channels.
        embed_dim: Number of output embedding dimensions.
        patch_size: Kernel size for the convolution (default: 7).
        stride: Stride of the convolution (default: 4).
        padding: Padding for the convolution (default: 3, for 7×7 kernel).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 7,
        stride: int = 4,
        padding: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            Patch-embedded tensor of shape ``[B, embed_dim, H_out, W_out]``.
        """
        x = self.proj(x)  # [B, embed_dim, H_out, W_out]
        B, C, H, W = x.shape
        # LayerNorm operates on channel-last
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return x


# ── Efficient Self-Attention ──────────────────────────────────────────────


class EfficientSelfAttention(nn.Module):
    """Self-attention with spatial reduction of keys and values.

    Reduces the spatial resolution of K and V by a factor of ``reduction_ratio``
    before computing attention, making it more efficient than standard
    self-attention for high-resolution feature maps.

    Operates in channel-last format internally (standard for attention).

    Args:
        dim: Input/output channel dimension.
        num_heads: Number of attention heads.
        reduction_ratio: Spatial reduction factor for K and V (default: 8).
        qkv_bias: Whether to use bias in QKV projection (default: False).
        qk_scale: Manual scale for QK (default: ``dim ** -0.5``).
        dropout: Dropout rate for attention weights (default: 0.0).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        reduction_ratio: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (self.head_dim ** -0.5)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Spatial reduction for K and V
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]`` (channel-first).

        Returns:
            Output tensor of shape ``[B, C, H, W]`` (channel-first).
        """
        B, C, H, W = x.shape

        # Channel-last for attention
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_flat = x.view(B, H * W, C)  # [B, N, C]

        # Q projection
        q = self.q(x_flat)  # [B, N, C]
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nH, N, hD]

        # K, V with spatial reduction
        x_cf = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W] (channel-first for conv)
        x_sr = self.sr(x_cf)  # [B, C, H_sr, W_sr]
        _, _, H_sr, W_sr = x_sr.shape
        x_sr = x_sr.permute(0, 2, 3, 1).contiguous()  # [B, H_sr, W_sr, C]
        x_sr = self.norm(x_sr)
        x_sr_flat = x_sr.view(B, H_sr * W_sr, C)  # [B, N_sr, C]

        kv = self.kv(x_sr_flat)  # [B, N_sr, 2*C]
        k, v = kv.chunk(2, dim=-1)  # each [B, N_sr, C]
        k = k.reshape(B, H_sr * W_sr, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, H_sr * W_sr, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, nH, N, N_sr]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x_out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)  # [B, N, C]
        x_out = self.proj(x_out)
        x_out = self.dropout(x_out)

        # Reshape back to channel-first
        x_out = x_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return x_out


# ── Mix-FFN ────────────────────────────────────────────────────────────────


class MixFFN(nn.Module):
    """Mix-FFN: MLP with 3×3 depthwise convolution.

    Unlike standard FFN which operates on flattened tokens, Mix-FFN uses a
    3×3 depthwise convolution between two linear projections to incorporate
    local positional information directly in the feed-forward path.

    Operates in channel-first format.

    Args:
        dim: Input/output channel dimension.
        hidden_dim: Hidden dimension (typically ``dim * mlp_ratio``).
        dropout: Dropout rate (default: 0.0).
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1, groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]`` (channel-first).

        Returns:
            Output tensor of shape ``[B, C, H, W]`` (channel-first).
        """
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ── Transformer Block ──────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """SegFormer transformer encoder block.

    Execution order:
        1. LayerNorm (channel-last)
        2. Efficient Self-Attention
        3. Residual connection
        4. LayerNorm (channel-last)
        5. Mix-FFN
        6. Residual connection

    Args:
        dim: Input/output channel dimension.
        num_heads: Number of attention heads.
        reduction_ratio: Spatial reduction ratio for K, V in attention.
        mlp_ratio: Ratio for MLP hidden dimension (``dim * mlp_ratio``).
        qkv_bias: Whether to use bias in QKV projection.
        qk_scale: Manual scale for QK.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        reduction_ratio: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            reduction_ratio=reduction_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(dim=dim, hidden_dim=mlp_hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]`` (channel-first).

        Returns:
            Output tensor of shape ``[B, C, H, W]`` (channel-first).
        """
        # Self-attention with residual
        B, C, H, W = x.shape
        identity = x
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        x = self.attn(x)
        x = identity + x

        # Mix-FFN with residual
        identity = x
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        x = self.mlp(x)
        x = identity + x

        return x


# ── MiT Stage ──────────────────────────────────────────────────────────────


class MiTStage(nn.Module):
    """A single hierarchical stage of the MiT backbone.

    Each stage consists of:
        1. OverlapPatchEmbed (downsampling + channel projection)
        2. N transformer blocks

    Args:
        in_channels: Input channel dimension.
        embed_dim: Output embedding dimension for this stage.
        depth: Number of transformer blocks in this stage.
        num_heads: Number of attention heads.
        reduction_ratio: Spatial reduction ratio for K, V in attention.
        mlp_ratio: Ratio for MLP hidden dimension.
        patch_size: Kernel size for OverlapPatchEmbed.
        stride: Stride for OverlapPatchEmbed.
        padding: Padding for OverlapPatchEmbed.
        qkv_bias: Whether to use bias in QKV projection.
        qk_scale: Manual scale for QK.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int = 2,
        num_heads: int = 1,
        reduction_ratio: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 7,
        stride: int = 4,
        padding: int = 3,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = OverlapPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            stride=stride,
            padding=padding,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                reduction_ratio=reduction_ratio,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            Output tensor of shape ``[B, embed_dim, H_out, W_out]``.
        """
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return x


# ── MiT Backbone ───────────────────────────────────────────────────────────


# Default MiT variant configurations (from SegFormer paper)
MIT_CONFIGS: dict[str, dict[str, Any]] = {
    "B0": {
        "embed_dims": [32, 64, 160, 256],
        "depths": [2, 2, 2, 2],
        "num_heads": [1, 2, 5, 8],
        "reduction_ratios": [8, 4, 2, 1],
        "mlp_ratios": [4, 4, 4, 4],
        "strides": [4, 2, 2, 2],
        "patch_sizes": [7, 3, 3, 3],
        "paddings": [3, 1, 1, 1],
    },
    "B1": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "num_heads": [1, 2, 5, 8],
        "reduction_ratios": [8, 4, 2, 1],
        "mlp_ratios": [4, 4, 4, 4],
        "strides": [4, 2, 2, 2],
        "patch_sizes": [7, 3, 3, 3],
        "paddings": [3, 1, 1, 1],
    },
    "B2": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 6, 3],
        "num_heads": [1, 2, 5, 8],
        "reduction_ratios": [8, 4, 2, 1],
        "mlp_ratios": [4, 4, 4, 4],
        "strides": [4, 2, 2, 2],
        "patch_sizes": [7, 3, 3, 3],
        "paddings": [3, 1, 1, 1],
    },
    "B3": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 4, 18, 3],
        "num_heads": [1, 2, 5, 8],
        "reduction_ratios": [8, 4, 2, 1],
        "mlp_ratios": [4, 4, 4, 4],
        "strides": [4, 2, 2, 2],
        "patch_sizes": [7, 3, 3, 3],
        "paddings": [3, 1, 1, 1],
    },
    "B4": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 8, 27, 3],
        "num_heads": [1, 2, 5, 8],
        "reduction_ratios": [8, 4, 2, 1],
        "mlp_ratios": [4, 4, 4, 4],
        "strides": [4, 2, 2, 2],
        "patch_sizes": [7, 3, 3, 3],
        "paddings": [3, 1, 1, 1],
    },
    "B5": {
        "embed_dims": [64, 128, 320, 512],
        "depths": [3, 6, 40, 3],
        "num_heads": [1, 2, 5, 8],
        "reduction_ratios": [8, 4, 2, 1],
        "mlp_ratios": [4, 4, 4, 4],
        "strides": [4, 2, 2, 2],
        "patch_sizes": [7, 3, 3, 3],
        "paddings": [3, 1, 1, 1],
    },
}


class MiTBackbone(nn.Module):
    """Hierarchical MiT encoder backbone.

    Produces multi-scale feature maps from 4 stages at resolutions:
        - Stage 1: 1/4 of input
        - Stage 2: 1/8 of input
        - Stage 3: 1/16 of input
        - Stage 4: 1/32 of input

    Args:
        in_channels: Number of input image channels.
        embed_dims: Embedding dimensions for each stage (list of 4 ints).
        depths: Number of transformer blocks per stage (list of 4 ints).
        num_heads: Number of attention heads per stage (list of 4 ints).
        reduction_ratios: Spatial reduction ratios per stage (list of 4 ints).
        mlp_ratios: MLP expansion ratios per stage (list of 4 floats).
        strides: Strides for OverlapPatchEmbed per stage (list of 4 ints).
        patch_sizes: Kernel sizes for OverlapPatchEmbed per stage (list of 4 ints).
        paddings: Paddings for OverlapPatchEmbed per stage (list of 4 ints).
        qkv_bias: Whether to use bias in QKV projection.
        qk_scale: Manual scale for QK.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: list[int] | tuple[int, ...] = (32, 64, 160, 256),
        depths: list[int] | tuple[int, ...] = (2, 2, 2, 2),
        num_heads: list[int] | tuple[int, ...] = (1, 2, 5, 8),
        reduction_ratios: list[int] | tuple[int, ...] = (8, 4, 2, 1),
        mlp_ratios: list[float] | tuple[float, ...] = (4, 4, 4, 4),
        strides: list[int] | tuple[int, ...] = (4, 2, 2, 2),
        patch_sizes: list[int] | tuple[int, ...] = (7, 3, 3, 3),
        paddings: list[int] | tuple[int, ...] = (3, 1, 1, 1),
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dims = list(embed_dims)
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.reduction_ratios = list(reduction_ratios)
        self.mlp_ratios = list(mlp_ratios)
        self.strides = list(strides)
        self.patch_sizes = list(patch_sizes)
        self.paddings = list(paddings)

        assert len(self.embed_dims) == 4, f"Expected 4 embed_dims, got {len(self.embed_dims)}"
        assert len(self.depths) == 4, f"Expected 4 depths, got {len(self.depths)}"

        self.stages = nn.ModuleList()
        prev_dim = in_channels
        for i in range(4):
            stage = MiTStage(
                in_channels=prev_dim,
                embed_dim=self.embed_dims[i],
                depth=self.depths[i],
                num_heads=self.num_heads[i],
                reduction_ratio=self.reduction_ratios[i],
                mlp_ratio=self.mlp_ratios[i],
                patch_size=self.patch_sizes[i],
                stride=self.strides[i],
                padding=self.paddings[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                dropout=dropout,
            )
            self.stages.append(stage)
            prev_dim = self.embed_dims[i]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass producing multi-scale features.

        Args:
            x: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            List of 4 feature maps ``[stage1, stage2, stage3, stage4]``
            at resolutions 1/4, 1/8, 1/16, 1/32 of input.
        """
        features: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


__all__ = [
    "OverlapPatchEmbed",
    "EfficientSelfAttention",
    "MixFFN",
    "TransformerBlock",
    "MiTStage",
    "MiTBackbone",
    "MIT_CONFIGS",
]