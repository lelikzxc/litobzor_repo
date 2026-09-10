"""High-level trainer for SemiWaferNet semi-supervised training.

Stage 1: supervised warm-up on Dl
Stage 2/3: offline MC pseudo-set on Du, then train on Concat(Dl, D_pseudo)
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from common.training.utils import clip_gradients
from papers.semiwafernet.training.progress import batch_progress, epoch_progress
from papers.semiwafernet.training.stage_manager import StageManager


class _PairFromDictDataset(Dataset):
    """Wrap dict samples ``{image,label}`` as ``(image, label)`` pairs."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        if isinstance(item, dict):
            return item["image"], int(item["label"])
        return item


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
        batch_size: int = 256,
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
        self.batch_size = batch_size

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

    def _ce_step(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert self.optimizer is not None
        assert self.supervised_loss_fn is not None
        self.optimizer.zero_grad()
        out = self.student(images)
        losses = self.supervised_loss_fn(
            {"classification": out["classification"]},
            {"classification": labels},
        )
        loss = sum(losses.values()) if isinstance(losses, dict) else losses
        loss.backward()
        clip_gradients(self.student, self.grad_max_norm, None)
        self.optimizer.step()
        return loss

    def _iter_labeled_pairs(self, labeled_data: Any):
        """Yield (images, class_labels) from adapter or loader."""
        for batch in labeled_data:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
                if isinstance(targets, dict):
                    labels = targets["classification"]
                else:
                    labels = targets
                yield images, labels
            else:
                raise TypeError(f"Unexpected labeled batch type: {type(batch)}")

    def train_stage1(
        self,
        labeled_data: Any,
        num_epochs: int = 1,
        val_eval_fn: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Supervised warm-up; installs the *best* teacher (paper Section 2.2).

        If ``val_eval_fn`` is provided, it is called after every epoch and the
        student state with the highest score becomes the Stage-1 teacher.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")
        if self.supervised_loss_fn is None:
            raise RuntimeError("Loss function not set. Call set_supervised_loss() first.")

        self.stage_manager.set_stage(1)
        self.student.train()

        total_loss = 0.0
        num_batches = 0
        batch_total = len(labeled_data) if hasattr(labeled_data, "__len__") else None
        best_metric = float("-inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = -1

        for epoch in epoch_progress(
            num_epochs, stage=1, title="supervised warm-up", disable=not self.verbose
        ):
            self.student.train()
            epoch_loss = 0.0
            batch_count = 0
            batches = batch_progress(
                self._iter_labeled_pairs(labeled_data),
                desc=f"  Epoch {epoch + 1}/{num_epochs}",
                total=batch_total,
                disable=not self.verbose,
            )
            for images, labels in batches:
                images = images.to(self.device)
                labels = labels.to(self.device)
                loss = self._ce_step(images, labels)
                epoch_loss += loss.item()
                batch_count += 1
                batches.set_postfix(loss=f"{loss.item():.4f}")

            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += epoch_loss
            num_batches += batch_count

            if val_eval_fn is not None:
                metric = float(val_eval_fn())
                if self.verbose:
                    print(f"  [SSL Stage 1] epoch {epoch + 1} val_metric={metric:.4f}")
                if metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(self.student.state_dict())

        if best_state is not None:
            self.student.load_state_dict(best_state)
            if self.verbose:
                print(
                    f"[SSL Stage 1] restored best teacher "
                    f"(epoch {best_epoch}, val_metric={best_metric:.4f})"
                )

        self.stage_manager.install_teacher_from_student()
        avg_loss = total_loss / max(num_batches, 1)
        out: dict[str, float] = {"loss": avg_loss}
        if best_state is not None:
            out["best_val_metric"] = best_metric
            out["best_epoch"] = float(best_epoch)
        if self.verbose:
            print(f"[SSL Stage 1] avg loss: {avg_loss:.4f}")
        return out

    def _build_union_loader(
        self,
        labeled_data: Any,
        pseudo_ds: Dataset | None,
    ) -> DataLoader:
        """Shuffled DataLoader over Dl U D_pseudo (paper Section 2.2)."""
        labeled_pairs: list[tuple[torch.Tensor, int]] = []
        for images, labels in self._iter_labeled_pairs(labeled_data):
            for i in range(images.shape[0]):
                lab = labels[i]
                if torch.is_tensor(lab) and lab.numel() > 1:
                    y = int(lab.argmax().item())
                else:
                    y = int(lab.item()) if torch.is_tensor(lab) else int(lab)
                labeled_pairs.append((images[i].detach().cpu(), y))

        if not labeled_pairs and pseudo_ds is None:
            raise RuntimeError("Empty labeled and pseudo sets.")

        xs: list[torch.Tensor] = []
        ys: list[int] = []
        for x, y in labeled_pairs:
            xs.append(x)
            ys.append(y)
        if pseudo_ds is not None and len(pseudo_ds) > 0:
            for i in range(len(pseudo_ds)):
                px, py = pseudo_ds[i]
                xs.append(px if torch.is_tensor(px) else torch.as_tensor(px))
                ys.append(int(py.item()) if torch.is_tensor(py) else int(py))

        train_ds: Dataset = TensorDataset(
            torch.stack(xs), torch.tensor(ys, dtype=torch.long)
        )
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

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

        stage_num = self.stage_manager.get_stage()
        stage_title = "pseudo-labels + train" if stage_num == 2 else "refresh + retrain"

        pseudo_ds, pseudo_stats = self.stage_manager.build_pseudo_dataset(
            unlabeled_loader=unlabeled_data,
            device=self.device,
            verbose=self.verbose,
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in epoch_progress(
            num_epochs, stage=stage_num, title=stage_title, disable=not self.verbose
        ):
            self.student.train()
            # Rebuild union each epoch so labeled geometric augs refresh
            union_loader = self._build_union_loader(labeled_data, pseudo_ds)
            batches = batch_progress(
                union_loader,
                desc=f"  Epoch {epoch + 1}/{num_epochs}",
                total=len(union_loader),
                disable=not self.verbose,
            )
            epoch_loss = 0.0
            batch_count = 0
            for images, labels in batches:
                images = images.to(self.device)
                labels = labels.to(self.device)
                loss = self._ce_step(images, labels)
                epoch_loss += loss.item()
                batch_count += 1
                batches.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{pseudo_stats.get('accept_rate', 0.0):.0f}%",
                    n=f"{int(pseudo_stats.get('n_accepted', 0))}",
                )

            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += epoch_loss
            num_batches += batch_count

        self.stage_manager.install_teacher_from_student()
        n = max(num_batches, 1)
        accept_pct = float(pseudo_stats.get("accept_rate", 0.0))
        if self.verbose:
            print(
                f"[SSL Stage {stage_num}] avg loss: {total_loss / n:.4f}, "
                f"pseudo accept: {accept_pct:.1f}%"
            )
        return {
            "loss": total_loss / n,
            "supervised_loss": total_loss / n,
            "pseudo_loss": 0.0,
            "consistency_loss": 0.0,
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
