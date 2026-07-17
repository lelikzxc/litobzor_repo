"""Reusable metrics for classification and segmentation evaluation.

Classification:
- accuracy, precision, recall, f1 (torch-based, sklearn-free)

Segmentation:
- iou_score, dice_score, pixel_accuracy

All metrics operate on torch tensors and return Python floats.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        logits: Raw class scores ``[B, C]``.
        targets: Ground-truth class indices ``[B]``.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    preds = logits.argmax(dim=1)
    return float(preds.eq(targets).float().mean().item())


def precision(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
    average: str = "macro",
) -> float:
    """Compute classification precision.

    Args:
        logits: Raw class scores ``[B, C]``.
        targets: Ground-truth class indices ``[B]``.
        num_classes: Number of classes. Inferred from ``logits`` if ``None``.
        average: ``"macro"`` (per-class unweighted mean) or ``"micro"`` (global).

    Returns:
        Precision as a float in ``[0, 1]``.
    """
    preds = logits.argmax(dim=1)
    return _per_class_metric(preds, targets, num_classes, average, "precision")


def recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
    average: str = "macro",
) -> float:
    """Compute classification recall.

    Args:
        logits: Raw class scores ``[B, C]``.
        targets: Ground-truth class indices ``[B]``.
        num_classes: Number of classes. Inferred from ``logits`` if ``None``.
        average: ``"macro"`` (per-class unweighted mean) or ``"micro"`` (global).

    Returns:
        Recall as a float in ``[0, 1]``.
    """
    preds = logits.argmax(dim=1)
    return _per_class_metric(preds, targets, num_classes, average, "recall")


def f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
    average: str = "macro",
) -> float:
    """Compute classification F1 score.

    Args:
        logits: Raw class scores ``[B, C]``.
        targets: Ground-truth class indices ``[B]``.
        num_classes: Number of classes. Inferred from ``logits`` if ``None``.
        average: ``"macro"`` (per-class unweighted mean) or ``"micro"`` (global).

    Returns:
        F1 score as a float in ``[0, 1]``.
    """
    preds = logits.argmax(dim=1)
    return _per_class_metric(preds, targets, num_classes, average, "f1")


def _per_class_metric(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None,
    average: str,
    metric: str,
) -> float:
    """Compute a per-class metric (precision, recall, or f1)."""
    if num_classes is None:
        num_classes = max(preds.max().item(), targets.max().item()) + 1
    num_classes = int(num_classes)

    confusion = _confusion_matrix(preds, targets, num_classes)
    tp = confusion.diag().float()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp

    if metric == "precision":
        scores = tp / (tp + fp + 1e-10)
    elif metric == "recall":
        scores = tp / (tp + fn + 1e-10)
    elif metric == "f1":
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        scores = 2.0 * prec * rec / (prec + rec + 1e-10)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if average == "micro":
        if metric == "precision":
            return float((tp.sum() / (tp.sum() + fp.sum() + 1e-10)).item())
        if metric == "recall":
            return float((tp.sum() / (tp.sum() + fn.sum() + 1e-10)).item())
        # f1 micro
        prec_micro = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
        rec_micro = tp.sum() / (tp.sum() + fn.sum() + 1e-10)
        return float((2.0 * prec_micro * rec_micro / (prec_micro + rec_micro + 1e-10)).item())

    return float(scores.mean().item())


def _confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Build a confusion matrix.

    Args:
        preds: Predicted class indices ``[B]``.
        targets: Ground-truth class indices ``[B]``.
        num_classes: Number of classes.

    Returns:
        ``[num_classes, num_classes]`` confusion matrix.
    """
    indices = num_classes * targets + preds
    return torch.bincount(indices, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
    smooth: float = 1e-6,
) -> float:
    """Compute mean IoU (Jaccard index) for segmentation.

    Args:
        logits: Raw pixel scores ``[B, C, H, W]``.
        targets: Ground-truth pixel indices ``[B, H, W]``.
        num_classes: Number of classes. Inferred from ``logits`` if ``None``.
        smooth: Smoothing factor.

    Returns:
        Mean IoU as a float in ``[0, 1]``.
    """
    if num_classes is None:
        num_classes = logits.shape[1]
    preds = logits.argmax(dim=1)

    ious: list[float] = []
    for c in range(num_classes):
        pred_mask = preds == c
        target_mask = targets == c
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(float(iou.item()))

    return float(sum(ious) / len(ious))


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
    smooth: float = 1e-6,
) -> float:
    """Compute mean Dice coefficient for segmentation.

    Args:
        logits: Raw pixel scores ``[B, C, H, W]``.
        targets: Ground-truth pixel indices ``[B, H, W]``.
        num_classes: Number of classes. Inferred from ``logits`` if ``None``.
        smooth: Smoothing factor.

    Returns:
        Mean Dice as a float in ``[0, 1]``.
    """
    if num_classes is None:
        num_classes = logits.shape[1]
    preds = logits.argmax(dim=1)

    dices: list[float] = []
    for c in range(num_classes):
        pred_mask = preds == c
        target_mask = targets == c
        intersection = (pred_mask & target_mask).sum().float()
        dice = (2.0 * intersection + smooth) / (
            pred_mask.sum().float() + target_mask.sum().float() + smooth
        )
        dices.append(float(dice.item()))

    return float(sum(dices) / len(dices))


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute pixel-wise accuracy for segmentation.

    Args:
        logits: Raw pixel scores ``[B, C, H, W]``.
        targets: Ground-truth pixel indices ``[B, H, W]``.

    Returns:
        Pixel accuracy as a float in ``[0, 1]``.
    """
    preds = logits.argmax(dim=1)
    return float(preds.eq(targets).float().mean().item())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: dict[str, str] = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "iou": "iou_score",
    "dice": "dice_score",
    "pixel_accuracy": "pixel_accuracy",
}


def build_metric(name: str) -> callable:
    """Look up a metric function by name.

    Args:
        name: Metric name (case-insensitive). One of:
            ``"accuracy"``, ``"precision"``, ``"recall"``, ``"f1"``,
            ``"iou"``, ``"dice"``, ``"pixel_accuracy"``.

    Returns:
        The corresponding metric function.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    key = name.lower().replace("-", "_")
    if key not in _METRIC_REGISTRY:
        allowed = ", ".join(sorted(_METRIC_REGISTRY))
        raise ValueError(f"Unknown metric: '{name}'. Allowed: {allowed}")

    func_name = _METRIC_REGISTRY[key]
    module = __import__("common.training.metrics", fromlist=[func_name])
    return getattr(module, func_name)