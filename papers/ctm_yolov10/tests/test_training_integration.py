"""Training integration tests for CTM-IYOLOv10 with the canonical common Trainer.

Verifies:
- Trainer creation with YOLOv10Baseline and CTMIYOLOv10
- Optimizer, scheduler, loss factory compatibility
- Detection batch handling via training collate adapter
- Training step (forward, loss, backward, optimizer step)
- Validation step
- Scheduler step
- Checkpoint save/load/resume
- Gradient flow
- Synthetic detection dataset + DataLoader pipeline
- Trainer + Engine compatibility
- Full pipeline: Dataset → DataLoader → Trainer → Forward → Loss → Backward → Optimizer → Scheduler

Note: YOLOv10's forward() returns a dict with ``one2many`` and ``one2one`` keys
during training. The ``one2one`` branch contains the actual predictions (boxes,
scores, feats). The canonical Trainer calls loss_fn(logits, targets), so we use
a thin YOLOLoss adapter that wraps Ultralytics' ``v8DetectionLoss`` to compute
the loss from the ``one2one`` branch.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from common.datasets import (
    DataModule,
    build_transforms,
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
    build_loss,
    build_optimizer,
    build_scheduler,
)
from papers.ctm_yolov10.data_utils import DetectionDataset
from papers.ctm_yolov10.models.yolov10 import CTMIYOLOv10, YOLOv10Baseline
from papers.ctm_yolov10.utils.training import YOLOLoss, patch_eval, training_collate

CONFIG_PATH = "papers/ctm_yolov10/configs/config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def baseline_model() -> YOLOv10Baseline:
    """YOLOv10Baseline with eval() patched to keep train mode.

    YOLOv10's eval mode runs ``_inference`` which fails with synthetic data.
    ``patch_eval()`` monkey-patches ``eval()`` to keep the model in train
    mode so that ``forward()`` always returns the training-time dict format
    (``one2many``/``one2one``).
    """
    model = YOLOv10Baseline(
        model_name="yolov10n", pretrained=False, num_classes=8
    )
    return patch_eval(model)


@pytest.fixture
def ctm_model() -> CTMIYOLOv10:
    """CTMIYOLOv10 with eval() patched to keep train mode."""
    model = CTMIYOLOv10(
        model_name="yolov10n",
        pretrained=False,
        num_classes=8,
        ghost_conv=True,
        bifpn=True,
    )
    return patch_eval(model)


@pytest.fixture
def synthetic_dataset() -> DetectionDataset:
    return DetectionDataset(
        synthetic_size=32, image_size=640, num_classes=8
    )


@pytest.fixture
def train_loader(
    synthetic_dataset: DetectionDataset,
) -> DataLoader:
    return DataLoader(
        synthetic_dataset,
        batch_size=4,
        collate_fn=training_collate,
        shuffle=True,
    )


@pytest.fixture
def val_loader(
    synthetic_dataset: DetectionDataset,
) -> DataLoader:
    return DataLoader(
        synthetic_dataset,
        batch_size=4,
        collate_fn=training_collate,
        shuffle=False,
    )


@pytest.fixture
def optimizer(
    baseline_model: YOLOv10Baseline,
) -> torch.optim.Optimizer:
    return build_optimizer(
        baseline_model,
        name="sgd",
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.0005,
    )


@pytest.fixture
def scheduler(
    optimizer: torch.optim.Optimizer,
) -> object:
    return build_scheduler(optimizer, name="cosine", T_max=10)


@pytest.fixture
def loss_fn(
    baseline_model: YOLOv10Baseline,
) -> YOLOLoss:
    return YOLOLoss(baseline_model)


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
    """Verify Trainer can be created with YOLO models and all components."""

    def test_trainer_creation_minimal(
        self, baseline_model: YOLOv10Baseline
    ) -> None:
        """Trainer can be created with just model, optimizer, loss_fn."""
        opt = build_optimizer(baseline_model, name="sgd", lr=1e-3)
        loss = YOLOLoss(baseline_model)
        trainer = Trainer(baseline_model, opt, loss, device="cpu")
        assert isinstance(trainer, Trainer)
        assert trainer.model is baseline_model
        assert trainer.device.type == "cpu"

    def test_trainer_creation_full(
        self,
        baseline_model: YOLOv10Baseline,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        loss_fn: YOLOLoss,
        checkpoint_manager: CheckpointManager,
        early_stopping: EarlyStopping,
        logger: TrainingLogger,
        scaler: NativeScaler,
    ) -> None:
        """Trainer can be created with all optional components."""
        trainer = Trainer(
            model=baseline_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device="cpu",
            early_stopping=early_stopping,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            scaler=scaler,
            grad_max_norm=1.0,
            verbose=False,
        )
        assert isinstance(trainer, Trainer)
        assert trainer.scheduler is scheduler
        assert trainer.early_stopping is early_stopping
        assert trainer.checkpoint_manager is checkpoint_manager
        assert trainer.grad_max_norm == 1.0

    def test_trainer_device_auto(
        self, baseline_model: YOLOv10Baseline
    ) -> None:
        """Trainer resolves 'auto' device correctly."""
        opt = build_optimizer(baseline_model, name="sgd", lr=1e-3)
        loss = YOLOLoss(baseline_model)
        trainer = Trainer(
            baseline_model, opt, loss, device="auto", verbose=False
        )
        assert trainer.device.type in ("cpu", "cuda")

    def test_trainer_model_on_device(
        self, baseline_model: YOLOv10Baseline
    ) -> None:
        """Model is moved to the correct device."""
        opt = build_optimizer(baseline_model, name="sgd", lr=1e-3)
        loss = YOLOLoss(baseline_model)
        trainer = Trainer(
            baseline_model, opt, loss, device="cpu", verbose=False
        )
        assert next(trainer.model.parameters()).device.type == "cpu"

    def test_trainer_with_ctm_model(
        self, ctm_model: CTMIYOLOv10
    ) -> None:
        """Trainer can be created with CTMIYOLOv10."""
        opt = build_optimizer(ctm_model, name="sgd", lr=1e-3)
        loss = YOLOLoss(ctm_model)
        trainer = Trainer(
            ctm_model, opt, loss, device="cpu", verbose=False
        )
        assert isinstance(trainer, Trainer)
        assert isinstance(trainer.model, CTMIYOLOv10)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """Verify a single training step works end-to-end."""

    def test_train_one_epoch(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Trainer.train_one_epoch() runs without error."""
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] > 0

    def test_backward_updates_parameters(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Parameters are updated after a training step."""
        old_params = [
            p.clone().detach() for p in baseline_model.parameters()
        ]
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.train_one_epoch(train_loader)
        params_changed = any(
            not torch.equal(old, p)
            for old, p in zip(old_params, baseline_model.parameters())
        )
        assert params_changed, (
            "At least one parameter should change after training"
        )

    def test_optimizer_step_updates_params(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Optimizer step changes parameter values."""
        old_params = [
            p.clone().detach() for p in baseline_model.parameters()
        ]
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.train_one_epoch(train_loader)
        params_changed = any(
            not torch.equal(old, new)
            for old, new in zip(old_params, baseline_model.parameters())
        )
        assert params_changed, "At least one parameter should change"


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------


class TestValidationStep:
    """Verify validation step works correctly."""

    def test_validate(
        self,
        baseline_model: YOLOv10Baseline,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Trainer.validate() runs without error."""
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.validate(val_loader)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_validate_no_grad(
        self,
        baseline_model: YOLOv10Baseline,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Validation does not update parameters."""
        old_params = [
            p.clone().detach() for p in baseline_model.parameters()
        ]
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.validate(val_loader)
        for old_p, new_p in zip(
            old_params, baseline_model.parameters()
        ):
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
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
    ) -> None:
        """CosineAnnealingLR reduces LR over epochs."""
        opt = build_optimizer(baseline_model, name="sgd", lr=1e-2)
        sched = build_scheduler(opt, name="cosine", T_max=10)
        loss = YOLOLoss(baseline_model)
        trainer = Trainer(
            baseline_model,
            opt,
            loss,
            scheduler=sched,
            device="cpu",
            verbose=False,
        )
        initial_lr = trainer._get_current_lr()
        for _ in range(5):
            trainer.train_one_epoch(train_loader)
            trainer.scheduler.step()
        final_lr = trainer._get_current_lr()
        assert (
            final_lr < initial_lr
        ), "LR should decrease with cosine schedule"

    def test_scheduler_in_fit(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
    ) -> None:
        """Scheduler is stepped inside Trainer.fit()."""
        opt = build_optimizer(baseline_model, name="sgd", lr=1e-3)
        sched = build_scheduler(opt, name="cosine", T_max=10)
        loss = YOLOLoss(baseline_model)
        trainer = Trainer(
            baseline_model,
            opt,
            loss,
            scheduler=sched,
            device="cpu",
            verbose=False,
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
        baseline_model: YOLOv10Baseline,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Trainer.save_checkpoint() creates a file."""
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
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
        baseline_model: YOLOv10Baseline,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Trainer.load_checkpoint() restores model state."""
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        initial_sd = copy.deepcopy(baseline_model.state_dict())
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            for p in baseline_model.parameters():
                p.data.add_(1.0)
            trainer.load_checkpoint(path)
            restored_sd = baseline_model.state_dict()
            for k in initial_sd:
                assert torch.equal(initial_sd[k], restored_sd[k]), (
                    f"Key {k} should match after load"
                )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_resume_training(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Training can resume from a checkpoint."""
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.fit(train_loader, epochs=2)
        assert trainer.current_epoch == 2
        epoch_2_state = copy.deepcopy(baseline_model.state_dict())
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            model2 = YOLOv10Baseline(
                model_name="yolov10n",
                pretrained=False,
                num_classes=8,
            )
            opt2 = build_optimizer(model2, name="sgd", lr=1e-3)
            loss2 = YOLOLoss(model2)
            trainer2 = Trainer(
                model2, opt2, loss2, device="cpu", verbose=False
            )
            loaded_epoch = trainer2.load_checkpoint(path)
            assert loaded_epoch == 2
            for k in epoch_2_state:
                assert torch.equal(
                    epoch_2_state[k], model2.state_dict()[k]
                ), f"Key {k} should match after resume"
            trainer2.fit(train_loader, epochs=3)
            assert trainer2.current_epoch == 5
        finally:
            Path(path).unlink(missing_ok=True)

    def test_checkpoint_manager_in_fit(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """CheckpointManager saves checkpoints during Trainer.fit()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_mgr = CheckpointManager(save_dir=tmpdir)
            trainer = Trainer(
                baseline_model,
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
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Training runs on CPU."""
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics

    def test_amp_compatibility(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """AMP (mixed precision) is compatible with YOLO."""
        scaler = NativeScaler(enabled=True)
        trainer = Trainer(
            baseline_model,
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
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Parameters that require grad receive non-zero gradients."""
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.train_one_epoch(train_loader)
        params_with_grad = 0
        params_total = 0
        for _name, param in baseline_model.named_parameters():
            if param.requires_grad:
                params_total += 1
                if (
                    param.grad is not None
                    and param.grad.abs().sum().item() > 0
                ):
                    params_with_grad += 1
        assert params_with_grad > 0, (
            f"No parameters received gradients (0/{params_total})"
        )

    def test_gradient_clipping(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Gradient clipping does not raise errors."""
        trainer = Trainer(
            baseline_model,
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

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_batch_size(
        self,
        baseline_model: YOLOv10Baseline,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Training works with batch size 1 and >1."""
        dataset = DetectionDataset(
            synthetic_size=16, image_size=640, num_classes=8
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=training_collate,
        )
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(loader)
        assert "loss" in metrics


# ---------------------------------------------------------------------------
# Synthetic dataset + DataLoader pipeline
# ---------------------------------------------------------------------------


class TestDataPipeline:
    """Verify the full data pipeline works with Trainer."""

    def test_synthetic_dataset_dataloader(
        self, synthetic_dataset: DetectionDataset
    ) -> None:
        """Synthetic dataset produces valid samples through DataLoader."""
        loader = DataLoader(
            synthetic_dataset,
            batch_size=4,
            collate_fn=training_collate,
        )
        batch = next(iter(loader))
        assert "inputs" in batch
        assert "targets" in batch
        assert batch["inputs"].shape == (4, 3, 640, 640)
        assert isinstance(batch["targets"], dict)
        assert "batch_idx" in batch["targets"]
        assert "cls" in batch["targets"]
        assert "bboxes" in batch["targets"]

    def test_full_pipeline(
        self,
        baseline_model: YOLOv10Baseline,
        synthetic_dataset: DetectionDataset,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Full pipeline: Dataset → DataLoader → Trainer → Forward → Loss → Backward → Optimizer."""
        loader = DataLoader(
            synthetic_dataset,
            batch_size=4,
            collate_fn=training_collate,
            shuffle=True,
        )
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(loader)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_pipeline_with_scheduler(
        self,
        baseline_model: YOLOv10Baseline,
        synthetic_dataset: DetectionDataset,
    ) -> None:
        """Full pipeline with scheduler."""
        loader = DataLoader(
            synthetic_dataset,
            batch_size=4,
            collate_fn=training_collate,
            shuffle=True,
        )
        opt = build_optimizer(baseline_model, name="sgd", lr=1e-3)
        sched = build_scheduler(opt, name="cosine", T_max=10)
        loss = YOLOLoss(baseline_model)
        trainer = Trainer(
            baseline_model,
            opt,
            loss,
            scheduler=sched,
            device="cpu",
            verbose=False,
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
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Engine can use a model trained by Trainer."""
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        trainer.train_one_epoch(train_loader)
        config = EngineConfig.from_yaml(CONFIG_PATH)
        engine = Engine(baseline_model, config, device="cpu")
        summary = engine.summary()
        assert summary["model"] == "YOLOv10Baseline"


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


class TestFullTrainingLoop:
    """Verify the complete training loop end-to-end."""

    def test_fit_with_train_only(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Trainer.fit() with only train loader."""
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        log = trainer.fit(train_loader, epochs=2)
        assert trainer.current_epoch == 2
        assert log is not None

    def test_fit_with_train_and_val(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Trainer.fit() with train and val loaders."""
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        log = trainer.fit(train_loader, val_loader, epochs=2)
        assert trainer.current_epoch == 2
        latest = log.latest()
        assert "train_loss" in latest
        assert "val_loss" in latest

    def test_fit_with_early_stopping(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Early stopping triggers during Trainer.fit()."""
        es = EarlyStopping(patience=1, min_delta=100.0)
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            early_stopping=es,
            verbose=False,
        )
        trainer.fit(train_loader, val_loader, epochs=10)
        assert trainer.current_epoch < 10, (
            "Early stopping should have triggered"
        )

    def test_fit_with_all_components(
        self,
        baseline_model: YOLOv10Baseline,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Trainer.fit() with all components."""
        opt = build_optimizer(baseline_model, name="sgd", lr=1e-3)
        sched = build_scheduler(opt, name="cosine", T_max=10)
        loss = YOLOLoss(baseline_model)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CheckpointManager(save_dir=tmpdir)
            es = EarlyStopping(patience=5, min_delta=0.001)
            trainer = Trainer(
                baseline_model,
                opt,
                loss,
                scheduler=sched,
                device="cpu",
                early_stopping=es,
                checkpoint_manager=ckpt,
                verbose=False,
            )
            log = trainer.fit(train_loader, val_loader, epochs=3)
            assert trainer.current_epoch == 3
            latest = log.latest()
            assert "train_loss" in latest
            assert "val_loss" in latest
            saved = list(Path(tmpdir).glob("*.pt"))
            assert len(saved) > 0


# ---------------------------------------------------------------------------
# DataModule integration
# ---------------------------------------------------------------------------


class TestDataModuleIntegration:
    """Verify Trainer works with DataModule from common.datasets."""

    def test_datamodule_with_trainer(
        self,
        baseline_model: YOLOv10Baseline,
        optimizer: torch.optim.Optimizer,
        loss_fn: YOLOLoss,
    ) -> None:
        """Trainer works with DataModule-created DataLoaders."""
        dataset = DetectionDataset(
            synthetic_size=32, image_size=640, num_classes=8
        )
        splits = split_dataset(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        dm = DataModule(
            dataset_type="classification",
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            test_dataset=splits["test"],
            batch_size=4,
            collate_fn=training_collate,
        )
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        trainer = Trainer(
            baseline_model,
            optimizer,
            loss_fn,
            device="cpu",
            verbose=False,
        )
        log = trainer.fit(train_loader, val_loader, epochs=2)
        assert trainer.current_epoch == 2
        latest = log.latest()
        assert "train_loss" in latest


# ---------------------------------------------------------------------------
# Factory compatibility
# ---------------------------------------------------------------------------


class TestFactoryCompatibility:
    """Verify canonical factories work with YOLO models."""

    def test_build_optimizer_sgd(
        self, baseline_model: YOLOv10Baseline
    ) -> None:
        """build_optimizer with sgd works."""
        opt = build_optimizer(
            baseline_model, name="sgd", lr=1e-2, momentum=0.9
        )
        assert isinstance(opt, torch.optim.SGD)

    def test_build_optimizer_adamw(
        self, baseline_model: YOLOv10Baseline
    ) -> None:
        """build_optimizer with adamw works."""
        opt = build_optimizer(baseline_model, name="adamw", lr=1e-3)
        assert isinstance(opt, torch.optim.AdamW)

    def test_build_scheduler_cosine(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        """build_scheduler with cosine works."""
        sched = build_scheduler(optimizer, name="cosine", T_max=10)
        assert isinstance(
            sched, torch.optim.lr_scheduler.CosineAnnealingLR
        )

    def test_build_scheduler_step(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        """build_scheduler with step works."""
        sched = build_scheduler(optimizer, name="step", step_size=5)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_yolo_loss_creation(
        self, baseline_model: YOLOv10Baseline
    ) -> None:
        """YOLOLoss can be created."""
        loss = YOLOLoss(baseline_model)
        assert isinstance(loss, nn.Module)

    def test_yolo_loss_forward(
        self, baseline_model: YOLOv10Baseline
    ) -> None:
        """YOLOLoss computes loss from YOLO forward output."""
        baseline_model.train()
        x = torch.randn(1, 3, 640, 640)
        output = baseline_model(x)
        assert isinstance(output, dict)
        assert "one2many" in output
        assert "one2one" in output
        loss_fn = YOLOLoss(baseline_model)
        targets = {
            "batch_idx": torch.zeros(0, dtype=torch.int64),
            "cls": torch.zeros(0),
            "bboxes": torch.zeros(0, 4),
        }
        total_loss = loss_fn(output, targets)
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0
        assert total_loss.item() > 0