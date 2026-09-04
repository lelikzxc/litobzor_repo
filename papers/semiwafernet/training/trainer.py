"""High-level trainer for SemiWaferNet semi-supervised training.

Provides the three-stage training workflow:
    - Stage 1: Supervised training only
    - Stage 2: Pseudo-label generation + adaptive thresholding + consistency
    - Stage 3: Refresh pseudo-labels + retrain
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from common.training.utils import clip_gradients
from papers.semiwafernet.training.progress import batch_progress, cycle_loader, epoch_progress
from papers.semiwafernet.training.stage_manager import StageManager


class Trainer:
    """High-level trainer for SemiWaferNet semi-supervised training."""

    def __init__(
        self,
        student: nn.Module,
        stage_manager: StageManager,
        optimizer: torch.optim.Optimizer | None = None,
        supervised_loss_fn: Callable | None = None,
        scheduler: Any = None,
        device: torch.device | None = None,
        grad_max_norm: float | None = 1.0,
        verbose: bool = True,
    ) -> None:
        self.student = student
        self.stage_manager = stage_manager
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student.to(self.device)

        self.optimizer: torch.optim.Optimizer | None = optimizer
        self.scheduler: Any = scheduler
        self.supervised_loss_fn: Callable | None = supervised_loss_fn
        self.grad_max_norm = grad_max_norm
        self.verbose = verbose

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def set_scheduler(self, scheduler: Any) -> None:
        self.scheduler = scheduler

    def set_supervised_loss(self, loss_fn: Callable) -> None:
        self.supervised_loss_fn = loss_fn

    def fit(
        self,
        labeled_data: Any,
        unlabeled_data: Any | None = None,
        num_epochs: int = 1,
        consistency_weight: float | None = None,
    ) -> dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")
        if self.supervised_loss_fn is None:
            raise RuntimeError("Loss function not set. Call set_supervised_loss() first.")

        stage1_metrics = self.train_stage1(labeled_data=labeled_data, num_epochs=num_epochs)

        if unlabeled_data is None:
            print("[SSL] No unlabeled data available — running supervised-only (Stage 1).")
            return {"stage1": stage1_metrics}

        stage2_metrics = self.train_stage2(
            labeled_data=labeled_data,
            unlabeled_data=unlabeled_data,
            num_epochs=num_epochs,
            consistency_weight=consistency_weight,
        )
        stage3_metrics = self.train_stage3(
            labeled_data=labeled_data,
            unlabeled_data=unlabeled_data,
            num_epochs=num_epochs,
            consistency_weight=consistency_weight,
        )

        return {
            "stage1": stage1_metrics,
            "stage2": stage2_metrics,
            "stage3": stage3_metrics,
        }

    def _run_supervised_step(
        self,
        inputs: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        assert self.optimizer is not None
        assert self.supervised_loss_fn is not None

        self.optimizer.zero_grad()
        student_output = self.student(inputs)
        losses = self.supervised_loss_fn(student_output, targets)
        loss = sum(losses.values()) if isinstance(losses, dict) else losses
        loss.backward()
        clip_gradients(self.student, self.grad_max_norm, None)
        self.optimizer.step()
        return loss

    def train_stage1(
        self,
        labeled_data: Any,
        num_epochs: int = 1,
        **kwargs: Any,
    ) -> dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")
        if self.supervised_loss_fn is None:
            raise RuntimeError("Loss function not set. Call set_supervised_loss() first.")

        self.stage_manager.set_stage(1)
        self.student.train()

        total_loss = 0.0
        num_batches = 0
        batch_total = len(labeled_data) if hasattr(labeled_data, "__len__") else None

        for epoch in epoch_progress(num_epochs, stage=1, title="supervised warm-up", disable=not self.verbose):
            epoch_loss = 0.0
            batch_count = 0

            batches = batch_progress(
                labeled_data,
                desc=f"  Epoch {epoch + 1}/{num_epochs}",
                total=batch_total,
                disable=not self.verbose,
            )
            for batch in batches:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in targets.items()
                }

                loss = self._run_supervised_step(inputs, targets)
                epoch_loss += loss.item()
                batch_count += 1
                batches.set_postfix(loss=f"{loss.item():.4f}")

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += epoch_loss
            num_batches += batch_count

        self.stage_manager.install_teacher_from_student()
        avg_loss = total_loss / max(num_batches, 1)
        if self.verbose:
            print(f"[SSL Stage 1] avg loss: {avg_loss:.4f}")
        return {"loss": avg_loss}

    def train_stage2(
        self,
        labeled_data: Any,
        unlabeled_data: Any,
        num_epochs: int = 1,
        consistency_weight: float | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")
        if self.supervised_loss_fn is None:
            raise RuntimeError("Loss function not set. Call set_supervised_loss() first.")

        if self.stage_manager.get_stage() == 1:
            self.stage_manager.set_stage(2)
        _ = consistency_weight

        total_loss = 0.0
        total_sup_loss = 0.0
        total_pseudo_loss = 0.0
        total_accepted = 0
        total_pseudo = 0
        num_batches = 0
        batch_total = len(labeled_data) if hasattr(labeled_data, "__len__") else None
        unlabeled_cycle = cycle_loader(unlabeled_data)

        stage_num = self.stage_manager.get_stage()
        stage_title = "pseudo-labels + train" if stage_num == 2 else "refresh + retrain"

        for epoch in epoch_progress(
            num_epochs, stage=stage_num, title=stage_title, disable=not self.verbose
        ):
            epoch_loss = 0.0
            epoch_sup = 0.0
            epoch_pseudo = 0.0
            batch_count = 0

            batches = batch_progress(
                labeled_data,
                desc=f"  Epoch {epoch + 1}/{num_epochs}",
                total=batch_total,
                disable=not self.verbose,
            )
            for labeled_batch in batches:
                unlabeled_batch = next(unlabeled_cycle)

                inputs_l, targets = labeled_batch
                inputs_l = inputs_l.to(self.device)
                targets = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in targets.items()
                }

                inputs_u = unlabeled_batch
                if isinstance(inputs_u, (list, tuple)):
                    inputs_u = inputs_u[0]
                inputs_u = inputs_u.to(self.device)

                self.optimizer.zero_grad()

                student_output_l = self.student(inputs_l)
                sup_losses = self.supervised_loss_fn(student_output_l, targets)
                sup_loss = sum(sup_losses.values()) if isinstance(sup_losses, dict) else sup_losses

                with torch.no_grad():
                    pseudo_results = self.stage_manager.generate_pseudo_labels(inputs_u)

                student_output_u = self.student(inputs_u)
                mask = pseudo_results["mask_class"]
                pseudo_y = pseudo_results["pseudo_labels_class"]

                total_pseudo += mask.numel()
                total_accepted += int(mask.sum().item())

                if mask.any():
                    pseudo_losses = self.supervised_loss_fn(
                        {"classification": student_output_u["classification"][mask]},
                        {"classification": pseudo_y[mask]},
                    )
                    pseudo_loss = (
                        sum(pseudo_losses.values())
                        if isinstance(pseudo_losses, dict)
                        else pseudo_losses
                    )
                else:
                    pseudo_loss = student_output_u["classification"].sum() * 0.0

                loss = sup_loss + pseudo_loss
                loss.backward()
                clip_gradients(self.student, self.grad_max_norm, None)
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_sup += float(sup_loss.detach())
                epoch_pseudo += float(pseudo_loss.detach())
                batch_count += 1

                accept_rate = 100.0 * mask.float().mean().item()
                batches.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sup=f"{float(sup_loss.detach()):.3f}",
                    pseudo=f"{float(pseudo_loss.detach()):.3f}",
                    acc=f"{accept_rate:.0f}%",
                )

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += epoch_loss
            total_sup_loss += epoch_sup
            total_pseudo_loss += epoch_pseudo
            num_batches += batch_count

        self.stage_manager.install_teacher_from_student()

        n = max(num_batches, 1)
        avg_pseudo = total_pseudo_loss / n
        accept_pct = 100.0 * total_accepted / max(total_pseudo, 1)
        if self.verbose:
            print(
                f"[SSL Stage {stage_num}] avg loss: {total_loss / n:.4f}, "
                f"pseudo accept: {accept_pct:.1f}%"
            )

        return {
            "loss": total_loss / n,
            "supervised_loss": total_sup_loss / n,
            "pseudo_loss": avg_pseudo,
            "consistency_loss": avg_pseudo,
            "pseudo_accept_rate": accept_pct,
        }

    def train_stage3(
        self,
        labeled_data: Any,
        unlabeled_data: Any,
        num_epochs: int = 1,
        consistency_weight: float | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        self.stage_manager.set_stage(3)
        self.refresh_teacher()
        self.stage_manager.reset_statistics()
        return self.train_stage2(
            labeled_data=labeled_data,
            unlabeled_data=unlabeled_data,
            num_epochs=num_epochs,
            consistency_weight=consistency_weight,
            **kwargs,
        )

    def generate_pseudo_labels(self, unlabeled_x: torch.Tensor) -> dict[str, Any]:
        return self.stage_manager.generate_pseudo_labels(unlabeled_x)

    def refresh_teacher(self) -> None:
        self.stage_manager.refresh_teacher()
