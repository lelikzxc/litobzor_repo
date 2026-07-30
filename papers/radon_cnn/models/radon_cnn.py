"""RadonCNN: Rotation- and flip-invariant CNN for wafer map classification.

Implements both Baseline and Proposed architectures from:
    Jeong et al., "Wafer map failure pattern classification using geometric
    transformation-invariant convolutional neural network",
    Scientific Reports 2023, 13:8127.

Architecture (Table 2):
    ┌──────────────┬───────────────────────┬──────────────────────────────┐
    │ Layer        │ Baseline              │ Proposed                     │
    ├──────────────┼───────────────────────┼──────────────────────────────┤
    │ Input        │ 64×64×1              │ 64×64×1                      │
    │ Radon        │ -                     │ 64×64×1                      │
    │ Conv1        │ ReLU 32×32×16        │ ReLU 32×32×16                │
    │ BN + MaxPool │                       │                              │
    │ Conv2        │ ReLU 16×16×64        │ ReLU 16×16×64 ×2 (KF)        │
    │ BN + MaxPool │                       │                              │
    │ Max out      │ -                     │ 16×16×64                     │
    │ Conv3        │ ReLU 8×8×128         │ ReLU 8×8×128                  │
    │ BN + MaxPool │                       │                              │
    │ Conv4        │ ReLU 4×4×256         │ ReLU 4×4×256                  │
    │ BN + MaxPool │                       │                              │
    │ FC1          │ ReLU 256             │ ReLU 256                      │
    │ BN           │                       │                              │
    │ FC2          │ ReLU 128             │ ReLU 128                      │
    │ BN           │                       │                              │
    │ FC3          │ 7                     │ 7                            │
    └──────────────┴───────────────────────┴──────────────────────────────┘
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from papers.radon_cnn.modules.kernel_flip import KernelFlip
from papers.radon_cnn.modules.radon_transform import RadonTransformModule


class _ConvBlock(nn.Module):
    """Convolution + ReLU + BatchNorm + MaxPool block.

    Per the paper (Table 2): Conv → ReLU → BatchNorm → MaxPool.
    BatchNorm after ReLU stabilises training by normalising non-negative
    activations, matching the paper's architecture exactly.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (default 3).
        pool_kernel: MaxPool kernel size (default 2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        return x


class _ConvBlockKF(nn.Module):
    """Convolution block with KernelFlip + ReLU + BatchNorm + MaxPool.

    Used in the Proposed model's Conv2 layer (weight-shared kernel flip).
    Order per Table 2: Conv(KF) → ReLU → BatchNorm → MaxPool.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (default 3).
        pool_kernel: MaxPool kernel size (default 2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_flip = KernelFlip(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.kernel_flip(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        return x


class _ClassifierHead(nn.Module):
    """Classification head: GAP → FC(256) → ReLU → BN → FC(128) → ReLU → BN → FC(7).

    Per Table 2:
        - Conv4 output: 4×4×256
        - Global Average Pooling to reduce 4×4 → 1×1
        - FC1: ReLU 256 + BatchNorm
        - FC2: ReLU 128 + BatchNorm
        - FC3: 7 (logits, no activation)
    Order per Table 2: FC → ReLU → BatchNorm (not FC → BN → ReLU).
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # [B, 256, 4, 4] → [B, 256, 1, 1]
        self.fc1 = nn.Linear(256, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x)           # [B, 256, 1, 1]
        x = x.flatten(1)          # [B, 256]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        return x


class BaselineCNN(nn.Module):
    """Baseline CNN model (no Radon transform, no kernel flip).

    Architecture per Table 2 (Baseline column):
        Input(64×64×1) → Conv1(32×32×16) → BN+Pool → Conv2(16×16×64) → BN+Pool
        → Conv3(8×8×128) → BN+Pool → Conv4(4×4×256) → BN+Pool
        → FC(256) → BN → FC(128) → BN → FC(7)

    Args:
        in_channels: Number of input channels (default 1 for grayscale).
        num_classes: Number of output classes (default 7).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 7) -> None:
        super().__init__()
        self.conv1 = _ConvBlock(in_channels, 16)
        self.conv2 = _ConvBlock(16, 64)
        self.conv3 = _ConvBlock(64, 128)
        self.conv4 = _ConvBlock(128, 256)
        self.classifier = _ClassifierHead(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # [B, 16, 32, 32]
        x = self.conv2(x)  # [B, 64, 16, 16]
        x = self.conv3(x)  # [B, 128, 8, 8]
        x = self.conv4(x)  # [B, 256, 4, 4]
        x = self.classifier(x)  # [B, 7]
        return x


class RadonCNN(nn.Module):
    """Proposed RadonCNN model with kernel flip.

    Architecture per Table 2 (Proposed column):
        Input(64×64×1) → Conv1(32×32×16) → BN+Pool
        → Conv2-KF(16×16×64×2) → BN+Pool → MaxOut(16×16×64)
        → Conv3(8×8×128) → BN+Pool → Conv4(4×4×256) → BN+Pool
        → FC(256) → BN → FC(128) → BN → FC(7)

    Note: The Radon transform is applied at the dataset level (see
    ``WaferRadonDataset``), not in the model forward pass. This avoids
    gradient graph breaks caused by numpy/skimage operations and speeds
    up training by ~3x (no redundant Radon computation per epoch).

    Args:
        in_channels: Number of input channels (default 1 for grayscale).
        num_classes: Number of output classes (default 7).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        self.conv1 = _ConvBlock(in_channels, 16)
        # Conv2 with kernel flip (weight-shared, 2 branches)
        self.conv2_kf = _ConvBlockKF(16, 64)
        self.conv3 = _ConvBlock(64, 128)
        self.conv4 = _ConvBlock(128, 256)
        self.classifier = _ClassifierHead(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already Radon-transformed (done in dataset)
        # [B, 1, 64, 64] Radon sinogram

        # Standard conv blocks
        x = self.conv1(x)  # [B, 16, 32, 32]

        # Kernel flip conv block (weight-shared, max-out internally)
        x = self.conv2_kf(x)  # [B, 64, 16, 16]

        # Standard conv blocks
        x = self.conv3(x)  # [B, 128, 8, 8]
        x = self.conv4(x)  # [B, 256, 4, 4]

        # Classifier head
        x = self.classifier(x)  # [B, 7]
        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference and return class probabilities.

        Args:
            x: Input tensor [B, 1, H, W].

        Returns:
            Class probabilities [B, num_classes].
        """
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    @classmethod
    def from_config(cls, config: Any) -> RadonCNN:
        """Create RadonCNN from an EngineConfig or dict-like object.

        Args:
            config: Configuration object with ``get`` method or dict.

        Returns:
            A new ``RadonCNN`` instance.
        """
        if hasattr(config, "_data"):
            in_channels = config.get("model.in_channels", 1)
            num_classes = config.get("model.num_classes", 7)
        else:
            in_channels = config.get("in_channels", 1)
            num_classes = config.get("num_classes", 7)
        return cls(
            in_channels=in_channels,
            num_classes=num_classes,
        )