"""Loss factory for classification and segmentation tasks.

Supports:
- Classification: CrossEntropyLoss, BCEWithLogitsLoss, FocalLoss
- Segmentation: DiceLoss, IoULoss, BCE + Dice combination
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Classification losses
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal loss for classification with imbalanced classes.

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Args:
        alpha: Class weighting factor (float or per-class tensor). ``None`` disables.
        gamma: Focusing parameter. ``0`` is equivalent to cross-entropy.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        alpha: float | torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw class scores ``[B, C]``.
            targets: Ground-truth class indices ``[B]``.

        Returns:
            Scalar loss if reduction is ``"mean"`` or ``"sum"``.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        modulating = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * modulating * ce_loss
        else:
            loss = modulating * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Segmentation losses
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """Dice loss for segmentation.

    ``DiceLoss = 1 - (2 * intersection + smooth) / (sum(y_hat) + sum(y) + smooth)``

    Args:
        smooth: Smoothing factor to avoid division by zero.
        reduction: ``"mean"`` (average over batch) or ``"sum"``.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: Raw pixel scores ``[B, C, H, W]``.
            targets: Ground-truth pixel indices ``[B, H, W]``.

        Returns:
            Scalar loss.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)
        targets_one_hot = targets_one_hot.float()

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class IoULoss(nn.Module):
    """IoU (Jaccard) loss for segmentation.

    ``IoULoss = 1 - (intersection + smooth) / (union + smooth)``

    Args:
        smooth: Smoothing factor to avoid division by zero.
        reduction: ``"mean"`` (average over batch) or ``"sum"``.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss.

        Args:
            logits: Raw pixel scores ``[B, C, H, W]``.
            targets: Ground-truth pixel indices ``[B, H, W]``.

        Returns:
            Scalar loss.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)
        targets_one_hot = targets_one_hot.float()

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss for binary segmentation.

    ``Loss = bce_weight * BCE(y_hat, y) + dice_weight * DiceLoss(y_hat, y)``

    Args:
        bce_weight: Weight for the BCE term.
        dice_weight: Weight for the Dice term.
        smooth: Smoothing factor for Dice loss.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined BCE + Dice loss.

        Args:
            logits: Raw pixel scores ``[B, 1, H, W]`` or ``[B, C, H, W]``.
            targets: Ground-truth pixel indices ``[B, H, W]``.

        Returns:
            Scalar loss.
        """
        # BCE: handle multi-class via cross-entropy, binary via BCEWithLogits
        if logits.shape[1] == 1:
            bce = F.binary_cross_entropy_with_logits(
                logits.squeeze(1), targets.float(), reduction="mean"
            )
        else:
            bce = F.cross_entropy(logits, targets, reduction="mean")

        dice = DiceLoss(smooth=self.smooth, reduction="mean")(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    # Classification
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
    "focal": FocalLoss,
    # Segmentation
    "dice": DiceLoss,
    "iou": IoULoss,
    "bce_dice": BCEDiceLoss,
}


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    """Build a loss module by name.

    Args:
        name: Loss name (case-insensitive). One of:
            ``"cross_entropy"``, ``"bce"``, ``"focal"``,
            ``"dice"``, ``"iou"``, ``"bce_dice"``.
        **kwargs: Arguments forwarded to the loss constructor.

    Returns:
        An ``nn.Module`` instance that computes the loss.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    key = name.lower().replace("-", "_")
    if key not in _LOSS_REGISTRY:
        allowed = ", ".join(sorted(_LOSS_REGISTRY))
        raise ValueError(
            f"Unknown loss: '{name}'. Allowed: {allowed}"
        )
    return _LOSS_REGISTRY[key](**kwargs)