"""Lightweight CNN feature extraction backbone for SemiWaferNet.

Implements the HybridCNN-ViT CNN module from the paper (Section 2.1):

    Stage 1: Conv3×3(64) → BN → ReLU → MaxPool(k=2,s=2) → 64×H/2×W/2
    Stage 2: ResBlock(64→128, stride=2) → 128×H/4×W/4

The residual block uses a projection shortcut (1×1 conv + BN) when dimensions
change, and identity mapping otherwise.

For segmentation (ConvoFormer-UNet), the backbone is not used — the encoder
uses convolution-enhanced patch embedding directly on the input.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Basic convolutional block: Conv2d → BN → Activation → (optional MaxPool).

    Matches Equation (1) from the paper:
        F1 = MaxPool(ReLU(BN(Conv3×3_64(X))))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | str = 1,
        norm: str = "bn",
        activation: str = "relu",
        use_pool: bool = False,
        pool_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "ln":
            self.norm = nn.GroupNorm(1, out_channels)
        else:
            raise ValueError(f"Unsupported norm: {norm}")

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel) if use_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with optional projection shortcut.

    Matches Equation (2)-(3) from the paper:
        y = F(x, Wi) + T(x)

    When input and output dimensions differ, T(x) is a 1×1 conv + BN projection.
    Otherwise, T(x) is identity mapping.

    The block performs channel expansion (e.g. 64→128) with stride=2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: str = "bn",
        activation: str = "relu",
    ) -> None:
        super().__init__()

        # Main path: two 3×3 convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if norm == "bn" else nn.GroupNorm(1, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if norm == "bn" else nn.GroupNorm(1, out_channels)

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Shortcut path: projection if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if norm == "bn" else nn.GroupNorm(1, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act(out)

        return out


class CNNBackbone(nn.Module):
    """Lightweight 2-stage CNN feature extractor for HybridCNN-ViT.

    Produces feature maps at 2 stages:
        stage1: [B, 64,  H/2, W/2]  — after Conv3×3 + BN + ReLU + MaxPool
        stage2: [B, 128, H/4, W/4]  — after ResBlock(64→128, stride=2)

    After the backbone, AdaptiveAvgPool is applied to obtain a fixed-size
    feature map (8×8 at 32×32 input), which is then fed to the Transformer.

    For segmentation (ConvoFormer-UNet), this backbone is NOT used.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] | None = None,
        norm: str = "bn",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128]

        # Stage 1: Conv3×3(64) → BN → ReLU → MaxPool(k=2,s=2)
        # Matches Equation (1) from the paper
        self.stage1 = ConvBlock(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            norm=norm,
            activation=activation,
            use_pool=True,
            pool_kernel=2,
        )

        # Stage 2: ResBlock(64→128, stride=2)
        # Matches Equation (3) from the paper
        self.stage2 = ResidualBlock(
            in_channels=channels[0],
            out_channels=channels[1],
            stride=2,
            norm=norm,
            activation=activation,
        )

        # Adaptive average pooling to fixed spatial size
        # At 32×32 input: stage2 output is 8×8, so pooling is identity
        # At other sizes: pools to 8×8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self._out_channels = channels

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            List of 2 feature maps: [stage1, stage2].
            stage1: [B, 64,  H/2, W/2]
            stage2: [B, 128, H/4, W/4]
        """
        features: list[torch.Tensor] = []

        # Stage 1
        x = self.stage1(x)  # [B, 64, H/2, W/2]
        features.append(x)

        # Stage 2
        x = self.stage2(x)  # [B, 128, H/4, W/4]
        features.append(x)

        return features  # [stage1, stage2]


__all__ = [
    "ConvBlock",
    "ResidualBlock",
    "CNNBackbone",
]