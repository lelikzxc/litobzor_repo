"""EMA teacher model for semi-supervised learning.

Maintains an exponential moving average of the student model parameters
for stable pseudo-label generation during training.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class EMATeacher(nn.Module):
    """Exponential Moving Average teacher model.

    The teacher maintains a slowly-updated copy of the student model.
    Teacher parameters are updated as:
        θ_teacher = momentum * θ_teacher + (1 - momentum) * θ_student

    The teacher is always in eval mode and does not compute gradients.

    Args:
        student: The student model to track.
        momentum: EMA decay rate (default: 0.999). Higher = slower update.
    """

    def __init__(self, student: nn.Module, momentum: float = 0.999) -> None:
        super().__init__()
        self.momentum = momentum
        self.teacher: nn.Module = copy.deepcopy(student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """Update teacher parameters via EMA.

        Args:
            student: The current student model with updated parameters.
        """
        for t_param, s_param in zip(self.teacher.parameters(), student.parameters()):
            t_param.data.mul_(self.momentum).add_(s_param.data, alpha=1.0 - self.momentum)

        for t_buffer, s_buffer in zip(self.teacher.buffers(), student.buffers()):
            t_buffer.data.copy_(s_buffer.data)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the teacher model.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Dictionary with "classification" and "segmentation" logits.
        """
        return self.teacher(x)