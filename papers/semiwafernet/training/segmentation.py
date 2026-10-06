"""Binary logits, deep supervision and metrics for ConvoFormer-UNet."""

import torch
from torch import nn
from torch.nn import functional as F


class DiceFocalLoss(nn.Module):
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    def forward(self, logits, targets):
        if logits.shape[1] != 1:
            raise ValueError("Binary DiceFocalLoss expects one output channel")
        target = targets.unsqueeze(1).to(logits.dtype)
        probs = logits.sigmoid()
        intersection = (probs * target).sum((2, 3))
        dice = 1 - ((2 * intersection + 1) / (probs.sum((2, 3)) + target.sum((2, 3)) + 1)).mean()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.where(target.bool(), probs, 1 - probs)
        alpha = torch.where(target.bool(), self.alpha, 1 - self.alpha)
        focal = (alpha * (1 - pt).pow(self.gamma) * bce).mean()
        return dice + 0.5 * focal


class SegmentationWrapper(nn.Module):
    """Expose tensor logits to the common trainer; retain auxiliary graphs for loss."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.aux_logits = None

    def forward(self, x):
        outputs = self.base_model(x, return_aux=True)["segmentation"]
        self.aux_logits = outputs
        return outputs["main"]


class SegmentationLoss(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        # Do not register the model as a child of the loss.
        object.__setattr__(self, "wrapper", wrapper)
        self.base_loss = DiceFocalLoss()

    def forward(self, logits, targets):
        aux = self.wrapper.aux_logits
        loss = self.base_loss(logits, targets)
        for name, weight in (("aux1", 0.3), ("aux2", 0.2)):
            up = F.interpolate(
                aux[name], size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = loss + weight * self.base_loss(up, targets)
        return loss


def binary_metrics(logits, targets):
    pred = logits[:, 0] >= 0
    truth = targets.bool()
    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()
    union = tp + fp + fn
    return {
        "iou": tp / union if union else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "pixel_accuracy": (pred == truth).float().mean().item(),
    }


def metric_functions():
    return {
        name: (lambda x, y, key=name: binary_metrics(x, y)[key])
        for name in ("iou", "dice", "pixel_accuracy")
    }
