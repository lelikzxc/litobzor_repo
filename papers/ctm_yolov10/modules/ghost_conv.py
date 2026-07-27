"""GhostConv module — lightweight convolution from GhostNet.

Replaces standard Conv in the first three stages of the YOLOv10 backbone
to reduce computational redundancy and model size, as described in:

    "Wafer Defect Detection Technology Based on CTM-IYOLOv10 Network"
    (Section 2.2.1, Figure 5)

GhostConv splits the output channels into two halves:
    - Y:   computed via a standard convolution (kernel_size, stride)
    - Y':  computed via cheap linear operations (depthwise conv) from Y

The final output is the concatenation [Y, Y'].

Reference:
    Han et al., "GhostNet: More Features from Cheap Operations", CVPR 2020.
"""

from __future__ import annotations

import torch
from torch import nn


class GhostConv(nn.Module):
    """Ghost convolution module.

    Note:
        ``f = -1`` is set as a class attribute for compatibility with
        YOLOv10's ``_predict_once`` which iterates over model layers
        and accesses ``m.f`` for skip connections.

    Produces ``2 * half_dim`` output channels from ``in_channels`` inputs
    using one standard convolution (for the first half) and cheap depthwise
    operations (for the second half).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (must be even).
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Padding (auto-computed if ``None``).
        groups: Number of groups for the primary convolution.
        act: Activation function (``True`` → SiLU, ``False`` → none, or a module).
    """

    f: int = -1  # YOLO compatibility: no skip connection
    i: int = -1  # YOLO compatibility: layer index

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        assert out_channels % 2 == 0, f"GhostConv requires out_channels ({out_channels}) to be even"

        half_dim = out_channels // 2

        if padding is None:
            padding = kernel_size // 2

        # Primary convolution: produces half the output channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                half_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(half_dim),
            nn.SiLU(inplace=True) if isinstance(act, bool) and act else (act if isinstance(act, nn.Module) else nn.Identity()),
        )

        # Cheap operation: depthwise conv on the primary output to produce the other half
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                half_dim,
                half_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=half_dim,  # depthwise
                bias=False,
            ),
            nn.BatchNorm2d(half_dim),
            nn.SiLU(inplace=True) if isinstance(act, bool) and act else (act if isinstance(act, nn.Module) else nn.Identity()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, in_channels, H, W].

        Returns:
            Output tensor [B, out_channels, H', W'].
        """
        y = self.primary_conv(x)       # [B, half_dim, H', W']
        y_prime = self.cheap_operation(y)  # [B, half_dim, H', W']
        return torch.cat([y, y_prime], dim=1)  # [B, out_channels, H', W']