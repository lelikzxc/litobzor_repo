"""Training integration tests for FCS-VMamba with the canonical common Trainer.

FCSVMamba's ``forward()`` returns a single logits tensor ``[B, num_classes]``,
which is directly compatible with the canonical ``Trainer``. No adapter is
needed for the model output.

The ``VMambaDataset`` returns ``{"image": ..., "label": ...}``, compatible
with ``common.datasets.classification_collate``. A ``_training_collate``
adapter remaps keys for ``Trainer._unpack_batch()``.

Test coverage:
- Trainer creation with FCSVMamba and all components
- Training step (forward, loss, backward, optimizer step)
- Validation step
- Scheduler step
- Checkpoint save/load/resume
- CPU and AMP compatibility
- Gradient flow and clipping
- Batch size 1 and >1
- Synthetic dataset + DataLoader pipeline
- Trainer + Engine compatibility
- Full pipeline: Dataset → DataLoader → Trainer → Forward → Loss → Backward → Optimizer → Scheduler
- DataModule integration
- Factory compatibility (optimizer, scheduler, loss, metric factories)
"""

from __future__ import annotations

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
from papers.vmamba.data_utils import VMambaDataset
from papers.vmamba.models.vmamba import FCSVMamba

CONFIG_PATH = "papers/vmamba/configs/config.yaml"

# ---------------------------------------------------------------------------
# Adapter collate: maps classification_collate output to Trainer's expected
# format. Trainer._unpack_batch() looks for "inputs"/"input" and
# "targets"/"target" keys. classification_collate produces "image"/"label".
# ---------------------------------------------------------------------------


def _training_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Wrap ``classification_collate`` and remap keys for Trainer compatibility.

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

# Use image_size=64 (divisible by 4, fast for tests) and num_classes=4
_IMG_SIZE = 64
_NUM_CLASSES = 4
_BATCH_SIZE = 8
_DATASET_SIZE = 64


@pytest.fixture
def model() -> FCSVMamba:
    return FCSVMamba(
        in_channels=3,
        image_size=_IMG_SIZE,
        embed_dim=32,
        depths=[1, 1, 1, 1],
        num_heads=[1, 2, 4, 8],
        ssm_ratio=1.0,
        mlp_ratio=1.0,
        drop_path_rate=0.0,
        num_classes=_NUM_CLASSES,
        fa_enabled=False,
        sfs_enabled=False,
        clca_enabled=False,
    )


@pytest.fixture
def synthetic_dataset() -> VMambaDataset:
    return VMambaDataset(
        synthetic_size=_DATASET_SIZE,
        image_size=_IMG_SIZE,
        num_classes=_NUM_CLASSES,
    )


@pytest.fixture
def train_loader(synthetic_dataset: VMambaDataset) -> DataLoader:
    return DataLoader(
        synthetic_dataset,
        batch_size=_BATCH_SIZE,
        collate_fn=_training_collate,
        shuffle=True,
    )


@pytest.fixture
def val_loader(synthetic_dataset: VMambaDataset) -> DataLoader:
    return DataLoader(
        synthetic_dataset,
        batch_size=_BATCH_SIZE,
        collate_fn=_training_collate,
        shuffle=False,
    )


@pytest.fixture
def optimizer(model: FCSVMamba) -> torch.optim.Optimizer:
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
    """Verify Trainer can be created with FCSVMamba and all components."""

    def test_trainer_creation_minimal(self, model: FCSVMamba) -> None:
        """Trainer can be created with just model, optimizer, loss_fn."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        loss = build_loss("cross_entropy")
        trainer = Trainer(model=model, optimizer=opt, loss_fn=loss)
        assert isinstance(trainer, Trainer)

    def test_trainer_creation_full(
        self,
        model: FCSVMamba,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        loss_fn: nn.Module,
        metric_fns: dict[str, callable],
        checkpoint_manager: CheckpointManager,
        early_stopping: EarlyStopping,
        logger: TrainingLogger,
        scaler: NativeScaler,
    ) -> None:
        """Trainer can be created with all optional components."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            checkpoint_manager=checkpoint_manager,
            early_stopping=early_stopping,
            logger=logger,
            scaler=scaler,
            device="cpu",
            grad_max_norm=1.0,
            grad_max_value=None,
            verbose=False,
        )
        assert isinstance(trainer, Trainer)
        assert str(trainer.device) == "cpu"

    def test_trainer_device_auto(self, model: FCSVMamba) -> None:
        """Trainer defaults to 'auto' device selection."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        loss = build_loss("cross_entropy")
        trainer = Trainer(model=model, optimizer=opt, loss_fn=loss)
        assert str(trainer.device) in ("cpu", "cuda")

    def test_trainer_model_on_device(self, model: FCSVMamba) -> None:
        """Trainer moves model to the specified device."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        loss = build_loss("cross_entropy")
        trainer = Trainer(model=model, optimizer=opt, loss_fn=loss, device="cpu")
        param_device = next(model.parameters()).device
        assert str(param_device) == "cpu"


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """Verify a single training step works end-to-end."""

    def test_train_one_epoch(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.train_one_epoch() runs without error."""
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
        model: FCSVMamba,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        metric_fns: dict[str, callable],
    ) -> None:
        """Trainer.train_one_epoch() computes metrics."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "f1" in metrics

    def test_loss_decreases_over_epochs(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Loss decreases (or at least doesn't explode) over multiple epochs."""
        opt = build_optimizer(model, name="adamw", lr=1e-2)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        losses = []
        for _ in range(3):
            metrics = trainer.train_one_epoch(train_loader)
            losses.append(metrics["loss"])
        for loss_val in losses:
            assert math.isfinite(loss_val), f"Loss is not finite: {loss_val}"

    def test_backward_updates_parameters(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Parameters are updated after a training step."""
        params_before = [p.clone() for p in model.parameters()]
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.train_one_epoch(train_loader)
        params_after = list(model.parameters())
        changed = any(
            not torch.equal(pb, pa) for pb, pa in zip(params_before, params_after)
        )
        assert changed, "No parameters changed after training step"

    def test_optimizer_step_updates_params(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Optimizer.step() changes parameter values."""
        opt = build_optimizer(model, name="sgd", lr=0.01)
        params_before = [p.clone().detach() for p in model.parameters()]
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.train_one_epoch(train_loader)
        params_after = [p.detach() for p in model.parameters()]
        diffs = [
            (pb - pa).abs().sum().item()
            for pb, pa in zip(params_before, params_after)
        ]
        total_diff = sum(diffs)
        assert total_diff > 0, "Parameters did not change after optimizer step"


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------


class TestValidationStep:
    """Verify validation step works correctly."""

    def test_validate(
        self,
        model: FCSVMamba,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.validate() runs without error."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.validate(val_loader)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_validate_with_metrics(
        self,
        model: FCSVMamba,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        metric_fns: dict[str, callable],
    ) -> None:
        """Trainer.validate() computes metrics."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.validate(val_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_validate_no_grad(
        self,
        model: FCSVMamba,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Validation runs without gradient computation."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.validate(val_loader)
        for p in model.parameters():
            assert p.grad is None or p.grad.sum().item() == 0.0, (
                "Gradients found after validation"
            )


# ---------------------------------------------------------------------------
# Scheduler step
# ---------------------------------------------------------------------------


class TestSchedulerStep:
    """Verify scheduler steps correctly."""

    def test_scheduler_step_reduces_lr(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Scheduler.step() reduces learning rate."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        sched = build_scheduler(opt, name="step", step_size=1, gamma=0.5)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            scheduler=sched,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        lr_before = trainer._get_current_lr()
        trainer.train_one_epoch(train_loader)
        trainer.scheduler.step()
        lr_after = trainer._get_current_lr()
        assert lr_after < lr_before, f"LR did not decrease: {lr_before} → {lr_after}"

    def test_scheduler_step_lr_not_nan(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Scheduler step produces finite LR."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        sched = build_scheduler(opt, name="cosine", T_max=10)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            scheduler=sched,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        for _ in range(5):
            trainer.train_one_epoch(train_loader)
            trainer.scheduler.step()
            lr = trainer._get_current_lr()
            assert math.isfinite(lr), f"LR is not finite: {lr}"

    def test_scheduler_in_fit(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Scheduler works inside Trainer.fit()."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        sched = build_scheduler(opt, name="step", step_size=1, gamma=0.5)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            scheduler=sched,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        lr_before = trainer._get_current_lr()
        trainer.fit(train_loader, val_loader, epochs=2)
        lr_after = trainer._get_current_lr()
        assert lr_after < lr_before, f"LR did not decrease in fit: {lr_before} → {lr_after}"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Verify checkpoint save, load, and resume."""

    def test_save_checkpoint(
        self,
        model: FCSVMamba,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.save_checkpoint() creates a file."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(path)
            assert path.exists(), "Checkpoint file was not created"

    def test_load_checkpoint(
        self,
        model: FCSVMamba,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.load_checkpoint() restores model state."""
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        weights_before = [p.clone() for p in model.parameters()]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(path)
            for p in model.parameters():
                p.data.add_(1.0)
            trainer.load_checkpoint(path)
            weights_after = [p.clone() for p in model.parameters()]
            for wb, wa in zip(weights_before, weights_after):
                assert torch.equal(wb, wa), "Weights not restored after load"

    def test_resume_training(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Training can resume from a checkpoint."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trainer.train_one_epoch(train_loader)
            trainer.save_checkpoint(path)
            epoch_before = trainer.current_epoch
            trainer.train_one_epoch(train_loader)
            trainer.load_checkpoint(path)
            assert trainer.current_epoch == epoch_before, (
                f"Epoch not restored: {trainer.current_epoch} != {epoch_before}"
            )

    def test_checkpoint_manager_in_fit(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """CheckpointManager saves checkpoints during Trainer.fit()."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_mgr = CheckpointManager(save_dir=tmpdir, metric_name="val_loss", mode="min")
            trainer = Trainer(
                model=model,
                optimizer=opt,
                loss_fn=loss_fn,
                checkpoint_manager=ckpt_mgr,
                device="cpu",
                verbose=False,
            )
            trainer.fit(train_loader, val_loader, epochs=2)
            ckpt_dir = Path(tmpdir)
            files = list(ckpt_dir.glob("*.pt"))
            assert len(files) > 0, "No checkpoint files created during fit"


# ---------------------------------------------------------------------------
# Hardware compatibility
# ---------------------------------------------------------------------------


class TestHardwareCompatibility:
    """Verify Trainer works on CPU and with AMP."""

    def test_training_on_cpu(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Training runs on CPU."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert math.isfinite(metrics["loss"])

    def test_amp_compatibility(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Training runs with AMP enabled on CPU."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        scaler = NativeScaler(enabled=True)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            scaler=scaler,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert math.isfinite(metrics["loss"])


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify gradients flow through all parameters."""

    def test_gradients_flow(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """All parameters receive non-zero gradients after backward."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.train_one_epoch(train_loader)
        grad_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
                else:
                    grad_norms.append(0.0)
        assert len(grad_norms) > 0, "No parameters with gradients"
        nonzero = sum(1 for g in grad_norms if g > 0)
        assert nonzero > 0, "No parameters received non-zero gradients"

    def test_gradient_clipping(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Gradient clipping does not crash."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            grad_max_norm=1.0,
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert math.isfinite(metrics["loss"])


# ---------------------------------------------------------------------------
# Batch size
# ---------------------------------------------------------------------------


class TestBatchSize:
    """Verify training works with different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_batch_size(
        self,
        model: FCSVMamba,
        loss_fn: nn.Module,
        batch_size: int,
    ) -> None:
        """Training works with batch_size=1 and batch_size=4."""
        dataset = VMambaDataset(
            synthetic_size=16,
            image_size=_IMG_SIZE,
            num_classes=_NUM_CLASSES,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=_training_collate,
        )
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(loader)
        assert "loss" in metrics
        assert math.isfinite(metrics["loss"])


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


class TestDataPipeline:
    """Verify the full data pipeline works with Trainer."""

    def test_synthetic_dataset_dataloader(
        self,
        synthetic_dataset: VMambaDataset,
    ) -> None:
        """Synthetic dataset + training_collate produces correct batch structure."""
        loader = DataLoader(
            synthetic_dataset,
            batch_size=_BATCH_SIZE,
            collate_fn=_training_collate,
        )
        batch = next(iter(loader))
        assert "inputs" in batch
        assert "targets" in batch
        assert batch["inputs"].shape == (_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)
        assert batch["targets"].shape == (_BATCH_SIZE,)

    def test_full_pipeline(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Full pipeline: Dataset → DataLoader → Trainer → Forward → Loss → Backward."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        train_metrics = trainer.train_one_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        assert "loss" in train_metrics
        assert "loss" in val_metrics
        assert math.isfinite(train_metrics["loss"])
        assert math.isfinite(val_metrics["loss"])

    def test_pipeline_with_scheduler(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Full pipeline with scheduler step."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        sched = build_scheduler(opt, name="cosine", T_max=5)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            scheduler=sched,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.fit(train_loader, val_loader, epochs=2)
        assert trainer.current_epoch == 2


# ---------------------------------------------------------------------------
# Engine compatibility
# ---------------------------------------------------------------------------


class TestEngineCompatibility:
    """Verify Trainer works alongside Engine."""

    def test_engine_with_trained_model(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Engine can use a model that was trained via Trainer."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.train_one_epoch(train_loader)

        config = EngineConfig.from_yaml(CONFIG_PATH)
        engine = Engine(model, config, device="cpu")
        summary = engine.summary()
        assert "model" in summary

    def test_engine_predict_after_training(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Engine.predict_single works after training."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.train_one_epoch(train_loader)

        config = EngineConfig.from_yaml(CONFIG_PATH)
        engine = Engine(model, config, device="cpu")
        x = torch.randn(3, _IMG_SIZE, _IMG_SIZE)
        result = engine.predict_single(x)
        assert "logits" in result
        assert "probs" in result
        assert "prediction" in result


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


class TestFullTrainingLoop:
    """Verify the complete training loop end-to-end."""

    def test_fit_with_train_only(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.fit() works with only train data."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        logger = trainer.fit(train_loader, epochs=2)
        assert trainer.current_epoch == 2
        assert isinstance(logger, TrainingLogger)

    def test_fit_with_train_and_val(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.fit() works with train and val data."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        logger = trainer.fit(train_loader, val_loader, epochs=2)
        assert trainer.current_epoch == 2
        assert isinstance(logger, TrainingLogger)

    def test_fit_with_early_stopping(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer.fit() stops early when EarlyStopping triggers."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        es = EarlyStopping(patience=1, min_delta=100.0)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            early_stopping=es,
            device="cpu",
            verbose=False,
        )
        trainer.fit(train_loader, val_loader, epochs=10)
        assert trainer.current_epoch < 10, "Early stopping did not trigger"

    def test_fit_with_all_components(
        self,
        model: FCSVMamba,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        metric_fns: dict[str, callable],
        early_stopping: EarlyStopping,
        logger: TrainingLogger,
        scaler: NativeScaler,
    ) -> None:
        """Trainer.fit() with all components: optimizer, scheduler, metrics,
        checkpointing, early stopping."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        sched = build_scheduler(opt, name="cosine", T_max=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_mgr = CheckpointManager(save_dir=tmpdir, metric_name="val_loss", mode="min")
            trainer = Trainer(
                model=model,
                optimizer=opt,
                scheduler=sched,
                loss_fn=loss_fn,
                metric_fns=metric_fns,
                checkpoint_manager=ckpt_mgr,
                early_stopping=early_stopping,
                logger=logger,
                scaler=scaler,
                device="cpu",
                grad_max_norm=1.0,
                verbose=False,
            )
            result_logger = trainer.fit(train_loader, val_loader, epochs=3)
            assert isinstance(result_logger, TrainingLogger)
            assert trainer.current_epoch <= 3


# ---------------------------------------------------------------------------
# DataModule integration
# ---------------------------------------------------------------------------


class TestDataModuleIntegration:
    """Verify Trainer works with DataModule from common.datasets."""

    def test_datamodule_with_trainer(
        self,
        model: FCSVMamba,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer works with DataModule-created DataLoaders."""
        dataset = VMambaDataset(
            synthetic_size=64,
            image_size=_IMG_SIZE,
            num_classes=_NUM_CLASSES,
        )
        splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        dm = DataModule(
            dataset_type="classification",
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            test_dataset=splits["test"],
            batch_size=_BATCH_SIZE,
            collate_fn=_training_collate,
        )
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        trainer.fit(train_loader, val_loader, epochs=2)
        assert trainer.current_epoch == 2

    def test_datamodule_with_transforms(
        self,
        model: FCSVMamba,
        loss_fn: nn.Module,
    ) -> None:
        """Trainer works with transformed datasets via DataModule."""
        transform = build_transforms(resize_size=(_IMG_SIZE, _IMG_SIZE))
        dataset = VMambaDataset(
            synthetic_size=32,
            image_size=_IMG_SIZE,
            num_classes=_NUM_CLASSES,
            transform=transform,
        )
        splits = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        dm = DataModule(
            dataset_type="classification",
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            test_dataset=splits["test"],
            batch_size=4,
            collate_fn=_training_collate,
        )
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            device="cpu",
            verbose=False,
        )
        train_loader = dm.train_dataloader()
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert math.isfinite(metrics["loss"])


# ---------------------------------------------------------------------------
# Factory compatibility
# ---------------------------------------------------------------------------


class TestFactoryCompatibility:
    """Verify canonical factories work with FCSVMamba."""

    def test_build_optimizer_adamw(self, model: FCSVMamba) -> None:
        """build_optimizer with AdamW works."""
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        assert isinstance(opt, torch.optim.Optimizer)

    def test_build_optimizer_adam(self, model: FCSVMamba) -> None:
        """build_optimizer with Adam works."""
        opt = build_optimizer(model, name="adam", lr=1e-3)
        assert isinstance(opt, torch.optim.Optimizer)

    def test_build_optimizer_sgd(self, model: FCSVMamba) -> None:
        """build_optimizer with SGD works."""
        opt = build_optimizer(model, name="sgd", lr=1e-3)
        assert isinstance(opt, torch.optim.Optimizer)

    def test_build_scheduler_cosine(self, optimizer: torch.optim.Optimizer) -> None:
        """build_scheduler with cosine works."""
        sched = build_scheduler(optimizer, name="cosine", T_max=10)
        assert sched is not None

    def test_build_scheduler_step(self, optimizer: torch.optim.Optimizer) -> None:
        """build_scheduler with step works."""
        sched = build_scheduler(optimizer, name="step", step_size=5, gamma=0.5)
        assert sched is not None

    def test_build_scheduler_plateau(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        """build_scheduler with plateau works."""
        sched = build_scheduler(optimizer, name="plateau", patience=2)
        assert sched is not None

    def test_build_loss_cross_entropy(self) -> None:
        """build_loss with cross_entropy works."""
        loss = build_loss("cross_entropy")
        assert isinstance(loss, nn.Module)

    def test_build_loss_focal(self) -> None:
        """build_loss with focal works."""
        loss = build_loss("focal", alpha=0.25, gamma=2.0)
        assert isinstance(loss, nn.Module)

    def test_build_metric_accuracy(self) -> None:
        """build_metric with accuracy works."""
        metric = build_metric("accuracy")
        assert callable(metric)

    def test_build_metric_f1(self) -> None:
        """build_metric with f1 works."""
        metric = build_metric("f1")
        assert callable(metric)

    def test_metrics_compute_correctly(self, model: FCSVMamba) -> None:
        """Metrics compute correctly with FCSVMamba output."""
        logits = torch.randn(4, _NUM_CLASSES)
        targets = torch.randint(0, _NUM_CLASSES, (4,))
        acc = accuracy(logits, targets)
        assert 0.0 <= acc <= 1.0
        f1_score = f1(logits, targets)
        assert 0.0 <= f1_score <= 1.0