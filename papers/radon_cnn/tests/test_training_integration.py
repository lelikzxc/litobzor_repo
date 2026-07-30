"""Training integration tests for RadonCNN with the canonical common Trainer.

Tests cover:
    - Trainer creation with RadonCNN
    - Training step (forward + backward)
    - Validation step
    - Scheduler step
    - Checkpoint save/load
    - Full training loop
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.training.trainer import Trainer
from common.training.checkpoint import CheckpointManager
from common.training.early_stopping import EarlyStopping
from common.training.logger import TrainingLogger
from common.training.utils import NativeScaler
from common.training.metrics import accuracy, f1
from papers.radon_cnn.models.radon_cnn import RadonCNN


# ── Synthetic Dataset ─────────────────────────────────────────────────────

class SyntheticRadonDataset(Dataset):
    """Synthetic dataset for RadonCNN training tests."""

    def __init__(self, num_samples: int = 32, num_classes: int = 7) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.randn(1, 64, 64)
        label = torch.tensor(index % self.num_classes, dtype=torch.long)
        return {"inputs": image, "targets": label}


def _collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    images = torch.stack([b["inputs"] for b in batch])
    labels = torch.tensor([b["targets"] for b in batch], dtype=torch.long)
    return {"inputs": images, "targets": labels}


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def model() -> RadonCNN:
    return RadonCNN(in_channels=1, num_classes=7)


@pytest.fixture
def synthetic_dataset() -> SyntheticRadonDataset:
    return SyntheticRadonDataset(num_samples=32, num_classes=7)


@pytest.fixture
def train_loader(synthetic_dataset: SyntheticRadonDataset) -> DataLoader:
    return DataLoader(
        synthetic_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=_collate_fn,
    )


@pytest.fixture
def val_loader(synthetic_dataset: SyntheticRadonDataset) -> DataLoader:
    return DataLoader(
        synthetic_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=_collate_fn,
    )


@pytest.fixture
def optimizer(model: RadonCNN) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture
def scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


@pytest.fixture
def loss_fn() -> nn.Module:
    return nn.CrossEntropyLoss()


# ── Trainer Creation ──────────────────────────────────────────────────────

class TestTrainerCreation:
    """Verify Trainer can be created with RadonCNN and all components."""

    def test_trainer_creation_minimal(self, model: RadonCNN) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
        )
        assert trainer.model is model
        assert trainer.optimizer is optimizer

    def test_trainer_creation_full(
        self,
        model: RadonCNN,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device="cpu",
            metric_fns={"accuracy": accuracy, "f1": f1},
            early_stopping=EarlyStopping(patience=5),
            checkpoint_manager=CheckpointManager(
                save_dir=tempfile.mkdtemp(),
                metric_name="val_accuracy",
                mode="max",
            ),
            logger=TrainingLogger(),
            scaler=NativeScaler(enabled=False),
            verbose=False,
        )
        assert trainer.model is model

    def test_trainer_device_auto(self, model: RadonCNN) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="auto",
        )
        assert str(trainer.device) in ("cpu", "cuda")

    def test_trainer_model_on_device(self, model: RadonCNN) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
        )
        assert next(trainer.model.parameters()).device == torch.device("cpu")


# ── Training Step ─────────────────────────────────────────────────────────

class TestTrainingStep:
    """Verify a single training step works end-to-end."""

    def test_train_one_epoch(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_train_one_epoch_with_metrics(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            metric_fns={"accuracy": accuracy},
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_backward_updates_parameters(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )

        # Get initial params
        init_params = [p.clone() for p in model.parameters()]

        trainer.train_one_epoch(train_loader)

        # Check params changed
        params_changed = False
        for init_p, p in zip(init_params, model.parameters()):
            if not torch.allclose(init_p, p):
                params_changed = True
                break
        assert params_changed, "Parameters did not change after training step"

    def test_optimizer_step_updates_params(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )

        # Get initial params
        init_params = [p.clone() for p in model.parameters()]

        # Manually do a step
        batch = next(iter(train_loader))
        images = batch["inputs"]
        labels = batch["targets"]

        trainer.optimizer.zero_grad()
        logits = trainer.model(images)
        loss = trainer.loss_fn(logits, labels)
        loss.backward()
        trainer.optimizer.step()

        # Check params changed
        diffs = []
        for init_p, p in zip(init_params, model.parameters()):
            diff = (init_p - p).abs().sum().item()
            diffs.append(diff)
        assert sum(diffs) > 0, "Parameters did not change after optimizer.step()"


# ── Validation Step ───────────────────────────────────────────────────────

class TestValidationStep:
    """Verify validation step works correctly."""

    def test_validate(
        self,
        model: RadonCNN,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.validate(val_loader)
        assert "loss" in metrics

    def test_validate_with_metrics(
        self,
        model: RadonCNN,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            metric_fns={"accuracy": accuracy},
            verbose=False,
        )
        metrics = trainer.validate(val_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_validate_no_grad(
        self,
        model: RadonCNN,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        # Validate should not accumulate gradients
        metrics = trainer.validate(val_loader)
        for p in model.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0


# ── Scheduler Step ────────────────────────────────────────────────────────

class TestSchedulerStep:
    """Verify scheduler steps correctly."""

    def test_scheduler_step_reduces_lr(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.train_one_epoch(train_loader)
        trainer.scheduler.step()
        new_lr = trainer.optimizer.param_groups[0]["lr"]
        assert new_lr < initial_lr

    def test_scheduler_in_fit(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.fit(train_loader, val_loader, epochs=3)
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr


# ── Checkpoint ────────────────────────────────────────────────────────────

class TestCheckpoint:
    """Verify checkpoint save, load, and resume."""

    def test_save_checkpoint(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(
                save_dir=tmpdir,
                metric_name="val_loss",
                mode="min",
            )
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device="cpu",
                checkpoint_manager=checkpoint_manager,
                verbose=False,
            )
            trainer.save_checkpoint(path=Path(tmpdir) / "checkpoint.pt")
            # Check file exists
            saved_files = list(Path(tmpdir).glob("*.pt"))
            assert len(saved_files) > 0

    def test_load_checkpoint(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(
                save_dir=tmpdir,
                metric_name="val_loss",
                mode="min",
            )
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device="cpu",
                checkpoint_manager=checkpoint_manager,
                verbose=False,
            )
            trainer.save_checkpoint(path=Path(tmpdir) / "checkpoint.pt")
            saved_files = list(Path(tmpdir).glob("*.pt"))
            if saved_files:
                trainer.load_checkpoint(str(saved_files[0]))

    def test_resume_training(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(
                save_dir=tmpdir,
                metric_name="val_loss",
                mode="min",
            )
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device="cpu",
                checkpoint_manager=checkpoint_manager,
                verbose=False,
            )
            # Train for 1 epoch
            trainer.fit(train_loader, val_loader, epochs=1)
            # Save checkpoint
            trainer.save_checkpoint(path=Path(tmpdir) / "checkpoint.pt")


# ── Full Training Loop ────────────────────────────────────────────────────

class TestFullTrainingLoop:
    """Verify the complete training loop end-to-end."""

    def test_fit_with_train_only(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        history = trainer.fit(train_loader, epochs=2)
        assert history is not None
        history_dict = history.to_dict()
        assert len(history_dict["train_loss"]) == 2

    def test_fit_with_train_and_val(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        history = trainer.fit(train_loader, val_loader, epochs=2)
        history_dict = history.to_dict()
        assert "train_loss" in history_dict
        assert "val_loss" in history_dict

    def test_fit_with_early_stopping(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        early_stopping = EarlyStopping(patience=2, mode="min")
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            early_stopping=early_stopping,
            verbose=False,
        )
        history = trainer.fit(train_loader, val_loader, epochs=10)
        assert history is not None

    def test_fit_with_all_components(
        self,
        model: RadonCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(
                save_dir=tmpdir,
                metric_name="val_loss",
                mode="min",
            )
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scheduler=scheduler,
                device="cpu",
                metric_fns={"accuracy": accuracy},
                early_stopping=EarlyStopping(patience=5),
                checkpoint_manager=checkpoint_manager,
                logger=TrainingLogger(),
                scaler=NativeScaler(enabled=False),
                verbose=False,
            )
            history = trainer.fit(train_loader, val_loader, epochs=3)
            history_dict = history.to_dict()
            assert "train_loss" in history_dict
            assert "val_loss" in history_dict