"""High-level trainer for SemiWaferNet semi-supervised training.

Provides the three-stage training workflow:
    - Stage 1: Supervised training only
    - Stage 2: Pseudo-label generation + adaptive thresholding + consistency
    - Stage 3: Refresh pseudo-labels + retrain

This class exposes the training workflow methods but does NOT implement:
    - Dataset loading / data loaders
    - Optimizer or scheduler logic (beyond placeholders)
    - CLI interface
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from papers.semiwafernet.training.stage_manager import StageManager


class Trainer:
    """High-level trainer for SemiWaferNet semi-supervised training.

    Args:
        student: The student model (SemiWaferNet instance).
        stage_manager: Configured StageManager instance.
        device: Torch device for training.
    """

    def __init__(
        self,
        student: nn.Module,
        stage_manager: StageManager,
        device: torch.device | None = None,
    ) -> None:
        self.student = student
        self.stage_manager = stage_manager
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student.to(self.device)

        # Placeholder optimizer (user should replace with actual optimizer)
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None

        # Placeholder loss function (user should replace with actual loss)
        self.supervised_loss_fn: Callable | None = None

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Set the optimizer.

        Args:
            optimizer: PyTorch optimizer instance.
        """
        self.optimizer = optimizer

    def set_scheduler(self, scheduler: Any) -> None:
        """Set the learning rate scheduler.

        Args:
            scheduler: PyTorch LR scheduler instance.
        """
        self.scheduler = scheduler

    def set_supervised_loss(self, loss_fn: Callable) -> None:
        """Set the supervised loss function.

        Args:
            loss_fn: Callable that takes (student_output, targets) and returns loss dict.
        """
        self.supervised_loss_fn = loss_fn

    def train_stage1(
        self,
        labeled_data: Any,
        num_epochs: int = 1,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Stage 1: Supervised training only.

        Trains on labeled data without pseudo-labels or teacher.

        Args:
            labeled_data: Iterable of (inputs, targets) batches.
                targets should be a dict with "classification" and "segmentation" keys.
            num_epochs: Number of training epochs.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            Dictionary with training metrics (e.g., {"loss": 0.0}).

        Raises:
            RuntimeError: If optimizer or loss function is not set.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")
        if self.supervised_loss_fn is None:
            raise RuntimeError("Loss function not set. Call set_supervised_loss() first.")

        self.stage_manager.set_stage(1)
        self.student.train()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch in labeled_data:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in targets.items()
                }

                self.optimizer.zero_grad()
                student_output = self.student(inputs)
                losses = self.supervised_loss_fn(student_output, targets)

                # Sum all loss components
                loss = sum(losses.values()) if isinstance(losses, dict) else losses
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += epoch_loss
            num_batches += batch_count

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    def train_stage2(
        self,
        labeled_data: Any,
        unlabeled_data: Any,
        num_epochs: int = 1,
        consistency_weight: float | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Stage 2: Semi-supervised training with pseudo-labels.

        Generates pseudo-labels from teacher, applies adaptive thresholding
        and uncertainty filtering, and trains on labeled + accepted pseudo-labels
        with consistency regularization.

        Args:
            labeled_data: Iterable of (inputs, targets) batches.
            unlabeled_data: Iterable of unlabeled input batches.
            num_epochs: Number of training epochs.
            consistency_weight: Weight for consistency loss. If None, uses
                the value from StageManager.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            Dictionary with training metrics.

        Raises:
            RuntimeError: If optimizer or loss function is not set.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")
        if self.supervised_loss_fn is None:
            raise RuntimeError("Loss function not set. Call set_supervised_loss() first.")

        self.stage_manager.set_stage(2)
        cw = consistency_weight if consistency_weight is not None else self.stage_manager.consistency_weight

        total_loss = 0.0
        total_sup_loss = 0.0
        total_cons_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_sup = 0.0
            epoch_cons = 0.0
            batch_count = 0

            # Zip labeled and unlabeled data together
            for labeled_batch, unlabeled_batch in zip(labeled_data, unlabeled_data):
                # Labeled branch
                inputs_l, targets = labeled_batch
                inputs_l = inputs_l.to(self.device)
                targets = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in targets.items()
                }

                # Unlabeled branch
                inputs_u = unlabeled_batch
                if isinstance(inputs_u, (list, tuple)):
                    inputs_u = inputs_u[0]
                inputs_u = inputs_u.to(self.device)

                self.optimizer.zero_grad()

                # Student forward on labeled data
                student_output_l = self.student(inputs_l)
                sup_losses = self.supervised_loss_fn(student_output_l, targets)
                sup_loss = sum(sup_losses.values()) if isinstance(sup_losses, dict) else sup_losses

                # Generate pseudo-labels from teacher on unlabeled data
                pseudo_results = self.stage_manager.generate_pseudo_labels(inputs_u)

                # Student forward on unlabeled data
                student_output_u = self.student(inputs_u)
                teacher_output_u = self.stage_manager.teacher(inputs_u)

                # Consistency loss with uncertainty masks
                cons_losses = self.stage_manager.compute_consistency_loss(
                    student_output_u,
                    teacher_output_u,
                    class_mask=pseudo_results["mask_class"],
                    seg_mask=pseudo_results["mask_seg"],
                )
                cons_loss = cons_losses["classification"] + cons_losses["segmentation"]

                # Total loss
                loss = sup_loss + cw * cons_loss
                loss.backward()
                self.optimizer.step()

                # Update teacher
                self.stage_manager.teacher.update(self.student)

                epoch_loss += loss.item()
                epoch_sup += sup_loss.item()
                epoch_cons += cons_loss.item()
                batch_count += 1

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += epoch_loss
            total_sup_loss += epoch_sup
            total_cons_loss += epoch_cons
            num_batches += batch_count

        avg_loss = total_loss / max(num_batches, 1)
        avg_sup = total_sup_loss / max(num_batches, 1)
        avg_cons = total_cons_loss / max(num_batches, 1)
        return {
            "loss": avg_loss,
            "supervised_loss": avg_sup,
            "consistency_loss": avg_cons,
        }

    def train_stage3(
        self,
        labeled_data: Any,
        unlabeled_data: Any,
        num_epochs: int = 1,
        consistency_weight: float | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Stage 3: Refresh pseudo-labels and retrain.

        Refreshes the teacher model, resets adaptive threshold statistics,
        regenerates pseudo-labels, and trains again.

        Args:
            labeled_data: Iterable of (inputs, targets) batches.
            unlabeled_data: Iterable of unlabeled input batches.
            num_epochs: Number of training epochs.
            consistency_weight: Weight for consistency loss. If None, uses
                the value from StageManager.
            **kwargs: Additional arguments (reserved for future use).

        Returns:
            Dictionary with training metrics.
        """
        # Refresh teacher and reset statistics
        self.refresh_teacher()
        self.stage_manager.reset_statistics()

        # Run stage 2 logic (same training procedure)
        return self.train_stage2(
            labeled_data=labeled_data,
            unlabeled_data=unlabeled_data,
            num_epochs=num_epochs,
            consistency_weight=consistency_weight,
            **kwargs,
        )

    def generate_pseudo_labels(
        self, unlabeled_x: torch.Tensor
    ) -> dict[str, Any]:
        """Generate pseudo-labels for unlabeled data.

        Delegates to StageManager.generate_pseudo_labels().

        Args:
            unlabeled_x: Unlabeled input tensor [B, C, H, W].

        Returns:
            Dictionary with pseudo-labels, masks, and uncertainty metrics.
        """
        return self.stage_manager.generate_pseudo_labels(unlabeled_x)

    def refresh_teacher(self) -> None:
        """Refresh the teacher model by copying current student."""
        self.stage_manager.refresh_teacher()