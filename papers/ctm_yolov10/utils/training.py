"""Runtime utilities for training CTM-YOLOv10 with the canonical common Trainer.

Provides:
- ``YOLOLoss`` — Adapter wrapping Ultralytics ``v8DetectionLoss`` for YOLO's
  dict output format (``one2many``/``one2one``).
- ``patch_eval`` — Monkey-patches ``model.eval()`` to keep the model in train
  mode (YOLOv10's eval mode runs ``_inference`` which fails with synthetic data).
- ``training_collate`` — Converts ``DetectionDataset`` output
  (``{"image": ..., "label": [N, 5]}``) to Trainer format
  (``{"inputs": ..., "targets": {"batch_idx": ..., "cls": ..., "bboxes": ...}}``).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


def patch_eval(model: nn.Module) -> nn.Module:
    """Monkey-patch ``model.eval()`` to keep the model in train mode.

    YOLOv10's eval mode runs ``_inference`` which concatenates predictions
    across detection heads. This fails with synthetic data (mismatched tensor
    sizes). This patch replaces ``eval()`` with a no-op that keeps the model
    in ``train(True)``, so ``forward()`` always returns the training-time
    dict format (``one2many``/``one2one``).

    Unlike a wrapper class, this preserves:
    - ``isinstance(model, YOLOv10Baseline)`` — type identity
    - ``model.state_dict()`` — no extra ``model.`` key prefix
    - ``Engine.summary()["model"]`` — reports ``"YOLOv10Baseline"``

    Args:
        model: The model to patch.

    Returns:
        The same model instance (patched in-place).
    """
    model.eval = lambda: model.train(True) or model  # type: ignore[method-assign]
    return model


class YOLOLoss(nn.Module):
    """Adapter loss that wraps Ultralytics ``v8DetectionLoss``.

    YOLOv10's forward returns a dict with ``one2one`` containing the
    predictions (``boxes``, ``scores``, ``feats``). This adapter:

    1. Extracts the ``one2one`` branch from YOLO's output.
    2. Creates a ``v8DetectionLoss`` criterion lazily.
    3. Computes the loss from the ``one2one`` predictions and targets.

    This makes YOLO compatible with the canonical Trainer's
    ``loss_fn(logits, targets)`` interface.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._criterion: nn.Module | None = None

    def _get_criterion(self) -> nn.Module:
        """Lazily initialize the Ultralytics loss criterion."""
        if self._criterion is not None:
            return self._criterion

        from ultralytics.utils.loss import v8DetectionLoss

        # Get the DetectionModel inside YOLOv10Baseline/CTMYOLOv10
        detection_model = getattr(self.model, "model", self.model)
        detect_head = detection_model.model[-1]

        # v8DetectionLoss needs a model with .model[-1] (Detect module)
        # and .args (hyperparameters). Create a minimal wrapper.
        class _LossModel(nn.Module):
            def __init__(self, head: nn.Module) -> None:
                super().__init__()
                self.model = nn.ModuleList([head])
                self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

        loss_model = _LossModel(detect_head)
        self._criterion = v8DetectionLoss(loss_model)
        return self._criterion

    def forward(
        self, logits: dict, targets: dict | None = None
    ) -> torch.Tensor:
        """Compute loss from YOLO's dict output and target batch.

        Args:
            logits: Dict from YOLO forward with ``one2many`` and ``one2one``.
            targets: Dict with ``"batch_idx"``, ``"cls"``, ``"bboxes"`` keys,
                or ``None`` for inference-only mode.

        Returns:
            Scalar total loss tensor.
        """
        criterion = self._get_criterion()

        # Extract the one2one branch (contains boxes, scores, feats)
        if isinstance(logits, dict) and "one2one" in logits:
            preds = logits["one2one"]
        else:
            preds = logits

        # Build the batch dict expected by v8DetectionLoss
        if targets is not None:
            batch = {
                "batch_idx": targets.get(
                    "batch_idx", torch.zeros(0, dtype=torch.int64)
                ),
                "cls": targets.get("cls", torch.zeros(0)),
                "bboxes": targets.get("bboxes", torch.zeros(0, 4)),
            }
        else:
            batch = {
                "batch_idx": torch.zeros(0, dtype=torch.int64),
                "cls": torch.zeros(0),
                "bboxes": torch.zeros(0, 4),
            }

        loss_val, _ = criterion(preds, batch)
        # loss_val is a 3-element tensor [box_loss, cls_loss, dfl_loss]
        return loss_val.sum()


def training_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Wrap default_collate and remap keys for Trainer compatibility.

    ``DetectionDataset`` returns ``{"image": ..., "label": [N, 5]}`` where
    label is in YOLO format ``[cls, x, y, w, h]``.

    ``Trainer._unpack_batch`` expects ``{"inputs": ..., "targets": ...}``.

    The targets dict is converted to the format expected by
    ``v8DetectionLoss``: ``{"batch_idx": ..., "cls": ..., "bboxes": ...}``.

    Args:
        batch: List of dicts from ``DetectionDataset``.

    Returns:
        Dict with ``"inputs"`` (image tensor) and ``"targets"`` (dict with
        ``"batch_idx"``, ``"cls"``, ``"bboxes"``).
    """
    from torch.utils.data._utils.collate import default_collate

    collated = default_collate(batch)
    images = collated["image"]  # [B, 3, H, W]
    labels = collated["label"]  # [B, N, 5] — cls, x, y, w, h

    # Convert labels to the format expected by v8DetectionLoss
    # labels shape: [B, N, 5] where each row is [cls, x, y, w, h]
    batch_size = images.shape[0]
    batch_idx_list: list[torch.Tensor] = []
    cls_list: list[torch.Tensor] = []
    bboxes_list: list[torch.Tensor] = []

    for b in range(batch_size):
        sample_labels = labels[b]  # [N, 5]
        # Filter out padding rows (all zeros)
        valid_mask = sample_labels.sum(dim=1) != 0
        valid_labels = sample_labels[valid_mask]
        if valid_labels.shape[0] > 0:
            batch_idx_list.append(
                torch.full((valid_labels.shape[0],), b, dtype=torch.int64)
            )
            cls_list.append(valid_labels[:, 0])
            bboxes_list.append(valid_labels[:, 1:5])  # [x, y, w, h]

    if batch_idx_list:
        targets = {
            "batch_idx": torch.cat(batch_idx_list),
            "cls": torch.cat(cls_list),
            "bboxes": torch.cat(bboxes_list),
        }
    else:
        targets = {
            "batch_idx": torch.zeros(0, dtype=torch.int64),
            "cls": torch.zeros(0),
            "bboxes": torch.zeros(0, 4),
        }

    return {
        "inputs": images,
        "targets": targets,
    }