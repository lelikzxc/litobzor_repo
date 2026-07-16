"""Lightweight CNN feature extraction backbone for SemiWaferNet.

Produces multi-scale feature maps at 4 stages with progressive
downsampling (×4, ×8, ×16, ×32) and channel expansion.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Basic convolutional block: Conv2d → BN → Activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | str = 1,
        norm: str = "bn",
        activation: str = "relu",
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
            self.norm = nn.GroupNorm(1, out_channels)  # equivalent to LayerNorm for 2D
        else:
            raise ValueError(f"Unsupported norm: {norm}")

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class CNNStage(nn.Module):
    """A single stage of the CNN backbone with downsampling + N ConvBlocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 2,
        norm: str = "bn",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        # First block may downsample
        layers.append(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3 if stride > 1 else 3,
                stride=stride,
                padding=1,
                norm=norm,
                activation=activation,
            )
        )

        # Remaining blocks at same resolution
        for _ in range(1, depth):
            layers.append(
                ConvBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm=norm,
                    activation=activation,
                )
            )

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CNNBackbone(nn.Module):
    """Lightweight multi-scale CNN feature extractor.

    Produces 4 stages of feature maps at resolutions:
        stage1: [B, C1, H/4,  W/4]
        stage2: [B, C2, H/8,  W/8]
        stage3: [B, C3, H/16, W/16]
        stage4: [B, C4, H/32, W/32]

    The first stage uses stride=4 (e.g. 7×7 conv) for aggressive
    early downsampling; subsequent stages use stride=2.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: list[int] | None = None,
        depths: list[int] | None = None,
        norm: str = "bn",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512]
        if depths is None:
            depths = [2, 2, 6, 2]

        self.stages = nn.ModuleList()

        # Stage 1: stride-4 downsampling (e.g. 7×7 conv)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]) if norm == "bn" else nn.GroupNorm(1, channels[0]),
            nn.ReLU(inplace=True) if activation == "relu" else nn.GELU(),
        )

        # Stage 1 refinement blocks
        stage1_blocks: list[nn.Module] = []
        for _ in range(1, depths[0]):
            stage1_blocks.append(
                ConvBlock(channels[0], channels[0], kernel_size=3, stride=1, padding=1, norm=norm, activation=activation)
            )
        self.stage1_refine = nn.Sequential(*stage1_blocks)

        # Stages 2-4: stride-2 downsampling
        prev_ch = channels[0]
        for i in range(1, 4):
            stage = CNNStage(
                in_channels=prev_ch,
                out_channels=channels[i],
                depth=depths[i],
                stride=2,
                norm=norm,
                activation=activation,
            )
            self.stages.append(stage)
            prev_ch = channels[i]

        self._out_channels = channels

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []

        # Stem + stage 1
        x = self.stem(x)  # [B, C1, H/4, W/4]
        x = self.stage1_refine(x)
        features.append(x)

        # Stages 2-4
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features  # [stage1, stage2, stage3, stage4]