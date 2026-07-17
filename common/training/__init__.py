"""Reusable training infrastructure for the litobzor research repository.

Provides generic, paper-agnostic components for training deep learning models:
- Trainer: generic training loop with fit/validate/predict
- Loss factory: classification and segmentation losses
- Metrics: classification and segmentation evaluation
- Optimizer factory: Adam, AdamW, SGD
- Scheduler factory: CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR
- Checkpoint manager: save/load best and last checkpoints
- Early stopping: patience-based stopping with best weight restoration
- Logger: lightweight metric/loss/lr tracking
- Utils: device transfer, gradient clipping, mixed precision helpers
"""

from __future__ import annotations

from common.training.checkpoint import CheckpointManager
from common.training.early_stopping import EarlyStopping
from common.training.logger import TrainingLogger
from common.training.losses import build_loss
from common.training.metrics import (
    accuracy,
    build_metric,
    dice_score,
    iou_score,
    pixel_accuracy,
    precision,
    recall,
    f1,
)
from common.training.optim import build_optimizer
from common.training.scheduler import build_scheduler
from common.training.trainer import Trainer
from common.training.utils import (
    clip_gradients,
    move_batch_to_device,
    NativeScaler,
)

__all__ = [
    # Trainer
    "Trainer",
    # Factories
    "build_optimizer",
    "build_scheduler",
    "build_loss",
    "build_metric",
    # Classification metrics
    "accuracy",
    "precision",
    "recall",
    "f1",
    # Segmentation metrics
    "iou_score",
    "dice_score",
    "pixel_accuracy",
    # Checkpoint
    "CheckpointManager",
    # Early stopping
    "EarlyStopping",
    # Logger
    "TrainingLogger",
    # Utils
    "move_batch_to_device",
    "clip_gradients",
    "NativeScaler",
]