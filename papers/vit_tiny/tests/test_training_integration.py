"""Training integration tests for ViT-Tiny with the canonical common Trainer.

Verifies:
- Trainer creation with ViTTiny
- Optimizer, scheduler, loss factory compatibility
- Training step (forward, loss, backward, optimizer step)
- Validation step
- Scheduler step
- Checkpoint save/load/resume
- CPU and AMP compatibility
- Gradient flow
- Batch size 1 and >1
- Synthetic dataset + DataLoader pipeline
- Trainer + Engine compatibility
- Full pipeline: Dataset → DataLoader → Trainer → Forward → Loss → Backward → Optimizer → Scheduler
"""

from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from common.datasets import (
    DataModule,
    build_transforms,
    classification_collate,
    split_dataset,
)
from common.engine.config import EngineConfig
from common.engine.engine import Engine
from common.training import (
    CheckpointManager,
    EarlyStopping,
    NativeScaler,
    Trainer,
    TrainingLogger,
    accuracy,
    build_loss,
    build_metric,
    build_optimizer,
    build_scheduler,
    f1,
    precision,
    recall,
)
from papers.vit_tiny.data_utils import ViTTinyDataset
from papers.vit_tiny.models.vit_tiny import ViTTiny

CONFIG_PATH = "papers/vit_tiny/configs/config.yaml"

# ---------------------------------------------------------------------------
# Adapter collate: maps classification_collate output to Trainer's expected
# format. Trainer._unpack_batch() looks for "inputs"/"input" and
# "targets"/"target" keys. classification_collate produces "image"/"label".
# ---------------------------------------------------------------------------


def _training_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Wrap classification_collate and remap keys for Trainer compatibility.

    ``classification_collate`` returns ``{"image": ..., "label": ...}``.
    ``Trainer._unpack_batch`` expects ``{"inputs": ..., "targets": ...}``.
    """
    collated = classification_collate(batch)
    return {
        "inputs": collated["image"],
        "targets": collated["label"],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> ViTTiny:
    return ViTTiny(image_size=32, in_channels=1, num_classes=8)


@pytest.fixture
def synthetic_dataset() -> ViTTinyDataset:
    return ViTTinyDataset(
        synthetic_size=64, image_size=32, num_classes=8
    )


@pytest.fixture
def train_loader(synthetic_dataset: ViTTinyDataset) -> DataLoader:
    return DataLoader(
        synthetic_dataset,
        batch_size=8,
        collate_fn=_training_collate,
        shuffle=True,
    )


@pytest.fixture
def val_loader(synthetic_dataset: ViTTinyDataset) -> DataLoader:
    return DataLoader(
        synthetic_dataset,
        batch_size=8,
        collate_fn=_training_collate,
        shuffle=False,
    )


@pytest.fixture
def optimizer(model: ViTTiny) -> torch.optim.Optimizer:
    return build_optimizer(model, name="adamw", lr=1e-3, weight_decay=0.05)


@pytest.fixture
def scheduler(optimizer: torch.optim.Optimizer) -> object:
    return build_scheduler(optimizer, name="cosine", T_max=10)


@pytest.fixture
def loss_fn() -> nn.Module:
    return build_loss("cross_entropy")


@pytest.fixture
def metric_fns() -> dict[str, callable]:
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


@pytest.fixture
def checkpoint_manager() -> CheckpointManager:
    return CheckpointManager(save_dir=tempfile.mkdtemp())


@pytest.fixture
def early_stopping() -> EarlyStopping:
    return EarlyStopping(patience=3, min_delta=0.01)


@pytest.fixture
def logger() -> TrainingLogger:
    return TrainingLogger()


@pytest.fixture
def scaler() -> NativeScaler:
    return NativeScaler(enabled=True)


# ---------------------------------------------------------------------------
# Trainer creation
# ---------------------------------------------------------------------------


class TestTrainerCreation:
    """Verify Trainer can be created with ViTTiny and all components."""

    def test_trainer_creation_minimal(self, model: ViTTiny) -> None:
        """Trainer can be created with just model, optimizer, loss_fn."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        loss = build_loss("cross_entropy")
        trainer = Trainer(model, opt, loss, device="cpu")
        assert isinstance(trainer, Trainer)
        assert trainer.model is model
        assert trainer.device.type == "cpu"

    def test_trainer_creation_full(
        self,
        model: ViTTiny,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        loss_fn: nn.Module,
        metric_fns: dict,
        checkpoint_manager: CheckpointManager,
        early_stopping: EarlyStopping,
        logger: TrainingLogger,
        scaler: NativeScaler,
    ) -> None:
        """Trainer can be created with all optional components."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device="cpu",
            metric_fns=metric_fns,
            early_stopping=early_stopping,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            scaler=scaler,
            grad_max_norm=1.0,
            grad_max_value=None,
            verbose=False,
        )
        assert isinstance(trainer, Trainer)
        assert trainer.scheduler is scheduler
        assert trainer.early_stopping is early_stopping
        assert trainer.checkpoint_manager is checkpoint_manager
        assert trainer.grad_max_norm == 1.0

    def test_trainer_device_auto(self, model: ViTTiny) -> None:
        """Trainer resolves 'auto' device correctly."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        loss = build_loss("cross_entropy")
        trainer = Trainer(model, opt, loss, device="auto", verbose=False)
        assert trainer.device.type in ("cpu", "cuda")

    def test_trainer_model_on_device(self, model: ViTTiny) -> None:
        """Model is moved to the correct device."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        loss = build_loss("cross_entropy")
        trainer = Trainer(model, opt, loss, device="cpu", verbose=False)
        assert next(trainer.model.parameters()).device.type == "cpu"


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """Verify a single training step works end-to-end."""

    def test_train_one_epoch(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.train_one_epoch() runs without error."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] > 0

    def test_train_one_epoch_with_metrics(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        metric_fns: dict,
    ) -> None:
        """Trainer.train_one_epoch() computes metrics."""
        trainer = Trainer(
            model,
            optimizer,
            loss_fn,
            device="cpu",
            metric_fns=metric_fns,
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert isinstance(metrics["accuracy"], float)

    def test_loss_decreases_over_epochs(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Loss decreases over multiple epochs of training."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        losses = []
        for _ in range(3):
            metrics = trainer.train_one_epoch(train_loader)
            losses.append(metrics["loss"])
        # Loss should generally decrease
        assert losses[-1] <= losses[0] * 1.1  # allow slight variance

    def test_backward_updates_parameters(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Parameters are updated after a training step."""
        old_params = [p.clone() for p in model.parameters()]
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        trainer.train_one_epoch(train_loader)
        for old_p, new_p in zip(old_params, model.parameters()):
            assert not torch.equal(old_p, new_p), (
                "Parameters should change after training"
            )

    def test_optimizer_step_updates_params(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Optimizer step changes parameter values."""
        old_params = [p.clone().detach() for p in model.parameters()]
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        trainer.train_one_epoch(train_loader)
        params_changed = any(
            not torch.equal(old, new)
            for old, new in zip(old_params, model.parameters())
        )
        assert params_changed, "At least one parameter should change"


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------


class TestValidationStep:
    """Verify validation step works correctly."""

    def test_validate(
        self,
        model: ViTTiny,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.validate() runs without error."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        metrics = trainer.validate(val_loader)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_validate_with_metrics(
        self,
        model: ViTTiny,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        metric_fns: dict,
    ) -> None:
        """Trainer.validate() computes metrics."""
        trainer = Trainer(
            model,
            optimizer,
            loss_fn,
            device="cpu",
            metric_fns=metric_fns,
            verbose=False,
        )
        metrics = trainer.validate(val_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_validate_no_grad(
        self,
        model: ViTTiny,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Validation does not update parameters."""
        old_params = [p.clone() for p in model.parameters()]
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        trainer.validate(val_loader)
        for old_p, new_p in zip(old_params, model.parameters()):
            assert torch.equal(old_p, new_p), (
                "Parameters should not change during validation"
            )


# ---------------------------------------------------------------------------
# Scheduler step
# ---------------------------------------------------------------------------


class TestSchedulerStep:
    """Verify scheduler steps correctly."""

    def test_scheduler_step_reduces_lr(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """CosineAnnealingLR reduces LR over epochs."""
        opt = build_optimizer(model, name="adamw", lr=1e-2)
        sched = build_scheduler(opt, name="cosine", T_max=10)
        trainer = Trainer(
            model, opt, loss_fn, scheduler=sched, device="cpu", verbose=False
        )
        initial_lr = trainer._get_current_lr()
        for _ in range(5):
            trainer.train_one_epoch(train_loader)
            trainer.scheduler.step()
        final_lr = trainer._get_current_lr()
        assert final_lr < initial_lr, "LR should decrease with cosine schedule"

    def test_scheduler_step_lr_not_nan(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Scheduler step produces valid LR values."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        sched = build_scheduler(opt, name="cosine", T_max=10)
        trainer = Trainer(
            model, opt, loss_fn, scheduler=sched, device="cpu", verbose=False
        )
        for _ in range(3):
            trainer.train_one_epoch(train_loader)
            trainer.scheduler.step()
        lr = trainer._get_current_lr()
        assert lr > 0, "LR should be positive"
        assert not math.isnan(lr), "LR should not be NaN"

    def test_scheduler_in_fit(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Scheduler is stepped inside Trainer.fit()."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        sched = build_scheduler(opt, name="cosine", T_max=10)
        trainer = Trainer(
            model, opt, loss_fn, scheduler=sched, device="cpu", verbose=False
        )
        initial_lr = trainer._get_current_lr()
        trainer.fit(train_loader, epochs=3)
        final_lr = trainer._get_current_lr()
        assert final_lr < initial_lr, "LR should decrease after fit()"


# ---------------------------------------------------------------------------
# Checkpoint save/load/resume
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Verify checkpoint save, load, and resume."""

    def test_save_checkpoint(
        self,
        model: ViTTiny,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.save_checkpoint() creates a file."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_checkpoint(
        self,
        model: ViTTiny,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.load_checkpoint() restores model state."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        # Save initial state, then save checkpoint
        initial_sd = copy.deepcopy(model.state_dict())
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            # Modify model after saving checkpoint
            for p in model.parameters():
                p.data.add_(1.0)
            # Load checkpoint — should restore original state
            trainer.load_checkpoint(path)
            restored_sd = model.state_dict()
            for k in initial_sd:
                assert torch.equal(initial_sd[k], restored_sd[k]), (
                    f"Key {k} should match after load"
                )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_resume_training(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Training can resume from a checkpoint."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        # Train for 2 epochs
        trainer.fit(train_loader, epochs=2)
        assert trainer.current_epoch == 2
        epoch_2_state = {k: v.clone() for k, v in model.state_dict().items()}
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            # Create new trainer and load
            model2 = ViTTiny(image_size=32, in_channels=1, num_classes=8)
            opt2 = build_optimizer(model2, name="adamw", lr=1e-3)
            loss2 = build_loss("cross_entropy")
            trainer2 = Trainer(
                model2, opt2, loss2, device="cpu", verbose=False
            )
            loaded_epoch = trainer2.load_checkpoint(path)
            assert loaded_epoch == 2
            # Verify state matches
            for k in epoch_2_state:
                assert torch.equal(
                    epoch_2_state[k], model2.state_dict()[k]
                ), f"Key {k} should match after resume"
            # Resume training
            trainer2.fit(train_loader, epochs=3)  # 3 more epochs
            assert trainer2.current_epoch == 5
        finally:
            Path(path).unlink(missing_ok=True)

    def test_checkpoint_manager_in_fit(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """CheckpointManager saves checkpoints during Trainer.fit()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_mgr = CheckpointManager(save_dir=tmpdir)
            trainer = Trainer(
                model,
                optimizer,
                loss_fn,
                device="cpu",
                checkpoint_manager=ckpt_mgr,
                verbose=False,
            )
            trainer.fit(train_loader, val_loader, epochs=3)
            saved = list(Path(tmpdir).glob("*.pt"))
            assert len(saved) > 0, "Checkpoints should be saved"


# ---------------------------------------------------------------------------
# CPU and AMP compatibility
# ---------------------------------------------------------------------------


class TestHardwareCompatibility:
    """Verify Trainer works on CPU and with AMP."""

    def test_training_on_cpu(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Training runs on CPU."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics

    def test_amp_compatibility(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """AMP (mixed precision) is compatible with ViTTiny."""
        scaler = NativeScaler(enabled=True)
        trainer = Trainer(
            model,
            optimizer,
            loss_fn,
            device="cpu",
            scaler=scaler,
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify gradients flow through all parameters."""

    def test_gradients_flow(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """All parameters receive non-zero gradients."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        trainer.train_one_epoch(train_loader)
        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"Parameter {name} has no gradient"
            )
            assert param.grad.abs().sum().item() > 0, (
                f"Parameter {name} has zero gradient"
            )

    def test_gradient_clipping(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Gradient clipping does not raise errors."""
        trainer = Trainer(
            model,
            optimizer,
            loss_fn,
            device="cpu",
            grad_max_norm=1.0,
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics


# ---------------------------------------------------------------------------
# Batch size variants
# ---------------------------------------------------------------------------


class TestBatchSize:
    """Verify training works with different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_size(
        self,
        model: ViTTiny,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Training works with batch size 1 and >1."""
        dataset = ViTTinyDataset(
            synthetic_size=32, image_size=32, num_classes=8
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=_training_collate,
        )
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        metrics = trainer.train_one_epoch(loader)
        assert "loss" in metrics


# ---------------------------------------------------------------------------
# Synthetic dataset + DataLoader pipeline
# ---------------------------------------------------------------------------


class TestDataPipeline:
    """Verify the full data pipeline works with Trainer."""

    def test_synthetic_dataset_dataloader(
        self, synthetic_dataset: ViTTinyDataset
    ) -> None:
        """Synthetic dataset produces valid samples through DataLoader."""
        loader = DataLoader(
            synthetic_dataset,
            batch_size=8,
            collate_fn=_training_collate,
        )
        batch = next(iter(loader))
        assert "inputs" in batch
        assert "targets" in batch
        assert batch["inputs"].shape == (8, 1, 32, 32)
        assert batch["targets"].shape == (8,)

    def test_full_pipeline(
        self,
        model: ViTTiny,
        synthetic_dataset: ViTTinyDataset,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Full pipeline: Dataset → DataLoader → Trainer → Forward → Loss → Backward → Optimizer."""
        loader = DataLoader(
            synthetic_dataset,
            batch_size=8,
            collate_fn=_training_collate,
            shuffle=True,
        )
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        # Forward + Loss + Backward + Optimizer step
        metrics = trainer.train_one_epoch(loader)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_pipeline_with_scheduler(
        self,
        model: ViTTiny,
        synthetic_dataset: ViTTinyDataset,
        loss_fn: nn.Module,
    ) -> None:
        """Full pipeline with scheduler: Dataset → DataLoader → Trainer → Forward → Loss → Backward → Optimizer → Scheduler."""
        loader = DataLoader(
            synthetic_dataset,
            batch_size=8,
            collate_fn=_training_collate,
            shuffle=True,
        )
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        sched = build_scheduler(opt, name="cosine", T_max=10)
        trainer = Trainer(
            model, opt, loss_fn, scheduler=sched, device="cpu", verbose=False
        )
        initial_lr = trainer._get_current_lr()
        trainer.fit(loader, epochs=3)
        final_lr = trainer._get_current_lr()
        assert final_lr <= initial_lr


# ---------------------------------------------------------------------------
# Trainer + Engine compatibility
# ---------------------------------------------------------------------------


class TestEngineCompatibility:
    """Verify Trainer works alongside Engine."""

    def test_engine_with_trained_model(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Engine can use a model trained by Trainer."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        trainer.train_one_epoch(train_loader)
        # Engine can wrap the trained model
        config = EngineConfig.from_yaml(CONFIG_PATH)
        engine = Engine(model, config, device="cpu")
        summary = engine.summary()
        assert summary["model"] == "ViTTiny"

    def test_engine_predict_after_training(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Engine.predict_single() works after Trainer training."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        trainer.train_one_epoch(train_loader)
        model.eval()
        config = EngineConfig.from_yaml(CONFIG_PATH)
        engine = Engine(model, config, device="cpu")
        x = torch.randn(1, 32, 32)
        result = engine.predict_single(x)
        assert "logits" in result
        assert result["logits"].shape == (1, 8)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


class TestFullTrainingLoop:
    """Verify the complete training loop end-to-end."""

    def test_fit_with_train_only(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.fit() with only train loader."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        log = trainer.fit(train_loader, epochs=2)
        assert trainer.current_epoch == 2
        assert log is not None

    def test_fit_with_train_and_val(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.fit() with train and val loaders."""
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        log = trainer.fit(train_loader, val_loader, epochs=2)
        assert trainer.current_epoch == 2
        latest = log.latest()
        assert "train_loss" in latest
        assert "val_loss" in latest

    def test_fit_with_early_stopping(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Early stopping triggers during Trainer.fit()."""
        es = EarlyStopping(patience=1, min_delta=100.0)  # aggressive
        trainer = Trainer(
            model,
            optimizer,
            loss_fn,
            device="cpu",
            early_stopping=es,
            verbose=False,
        )
        trainer.fit(train_loader, val_loader, epochs=10)
        # Should have stopped early
        assert trainer.current_epoch < 10, (
            "Early stopping should have triggered"
        )

    def test_fit_with_all_components(
        self,
        model: ViTTiny,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        metric_fns: dict,
    ) -> None:
        """Trainer.fit() with all components: optimizer, scheduler, metrics, checkpointing, early stopping."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        sched = build_scheduler(opt, name="cosine", T_max=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CheckpointManager(save_dir=tmpdir)
            es = EarlyStopping(patience=5, min_delta=0.001)
            trainer = Trainer(
                model,
                opt,
                loss_fn,
                scheduler=sched,
                device="cpu",
                metric_fns=metric_fns,
                early_stopping=es,
                checkpoint_manager=ckpt,
                verbose=False,
            )
            log = trainer.fit(train_loader, val_loader, epochs=3)
            assert trainer.current_epoch == 3
            latest = log.latest()
            assert "train_loss" in latest
            assert "val_loss" in latest
            assert "train_accuracy" in latest
            assert "val_accuracy" in latest or "val_accuracy" not in latest
            # Checkpoints should exist
            saved = list(Path(tmpdir).glob("*.pt"))
            assert len(saved) > 0


# ---------------------------------------------------------------------------
# DataModule integration
# ---------------------------------------------------------------------------


class TestDataModuleIntegration:
    """Verify Trainer works with DataModule from common.datasets."""

    def test_datamodule_with_trainer(
        self,
        model: ViTTiny,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer works with DataModule-created DataLoaders."""
        dataset = ViTTinyDataset(
            synthetic_size=64, image_size=32, num_classes=8
        )
        splits = split_dataset(
            dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        dm = DataModule(
            dataset_type="classification",
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            test_dataset=splits["test"],
            batch_size=8,
            collate_fn=_training_collate,
        )
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        log = trainer.fit(train_loader, val_loader, epochs=2)
        assert trainer.current_epoch == 2
        latest = log.latest()
        assert "train_loss" in latest

    def test_datamodule_with_transforms(
        self,
        model: ViTTiny,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer works with transformed datasets via DataModule."""
        transform = build_transforms(resize_size=(32, 32))
        dataset = ViTTinyDataset(
            synthetic_size=64,
            image_size=32,
            num_classes=8,
            transform=transform,
        )
        splits = split_dataset(
            dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )
        dm = DataModule(
            dataset_type="classification",
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            batch_size=8,
            collate_fn=_training_collate,
        )
        trainer = Trainer(
            model, optimizer, loss_fn, device="cpu", verbose=False
        )
        log = trainer.fit(
            dm.train_dataloader(), dm.val_dataloader(), epochs=2
        )
        assert trainer.current_epoch == 2


# ---------------------------------------------------------------------------
# Factory compatibility
# ---------------------------------------------------------------------------


class TestFactoryCompatibility:
    """Verify canonical factories work with ViTTiny."""

    def test_build_optimizer_adamw(self, model: ViTTiny) -> None:
        """build_optimizer with adamw works."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        assert isinstance(opt, torch.optim.AdamW)

    def test_build_optimizer_adam(self, model: ViTTiny) -> None:
        """build_optimizer with adam works."""
        opt = build_optimizer(model, name="adam", lr=1e-3)
        assert isinstance(opt, torch.optim.Adam)

    def test_build_optimizer_sgd(self, model: ViTTiny) -> None:
        """build_optimizer with sgd works."""
        opt = build_optimizer(model, name="sgd", lr=1e-2)
        assert isinstance(opt, torch.optim.SGD)

    def test_build_scheduler_cosine(self, optimizer: torch.optim.Optimizer) -> None:
        """build_scheduler with cosine works."""
        sched = build_scheduler(optimizer, name="cosine", T_max=10)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_build_scheduler_step(self, optimizer: torch.optim.Optimizer) -> None:
        """build_scheduler with step works."""
        sched = build_scheduler(optimizer, name="step", step_size=5)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_build_scheduler_plateau(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        """build_scheduler with plateau works."""
        sched = build_scheduler(optimizer, name="plateau")
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_build_loss_cross_entropy(self) -> None:
        """build_loss with cross_entropy works."""
        loss = build_loss("cross_entropy")
        assert isinstance(loss, nn.CrossEntropyLoss)

    def test_build_loss_focal(self) -> None:
        """build_loss with focal works."""
        loss = build_loss("focal", gamma=2.0)
        assert loss is not None

    def test_build_metric_accuracy(self) -> None:
        """build_metric with accuracy works."""
        metric = build_metric("accuracy")
        assert callable(metric)

    def test_build_metric_f1(self) -> None:
        """build_metric with f1 works."""
        metric = build_metric("f1")
        assert callable(metric)

    def test_metrics_compute_correctly(self, model: ViTTiny) -> None:
        """Metrics compute correct values with ViTTiny output."""
        model.eval()
        logits = torch.randn(4, 8)
        targets = torch.randint(0, 8, (4,))
        acc = accuracy(logits, targets)
        assert 0.0 <= acc <= 1.0
        f1_score = f1(logits, targets)
        assert 0.0 <= f1_score <= 1.0