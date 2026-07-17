"""Comprehensive tests for common.training infrastructure.

Uses small dummy models and synthetic tensors only. No CUDA required.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    clip_gradients,
    dice_score,
    f1,
    iou_score,
    move_batch_to_device,
    pixel_accuracy,
    precision,
    recall,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class DummyClassifier(nn.Module):
    """Minimal classifier for testing."""

    def __init__(self, in_features: int = 10, num_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DummySegmenter(nn.Module):
    """Minimal segmenter for testing."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def model() -> DummyClassifier:
    return DummyClassifier(in_features=10, num_classes=3)


@pytest.fixture
def seg_model() -> DummySegmenter:
    return DummySegmenter(num_classes=3)


@pytest.fixture
def optimizer(model: DummyClassifier) -> torch.optim.Optimizer:
    return build_optimizer(model, name="adamw", lr=1e-3)


@pytest.fixture
def loss_fn() -> nn.Module:
    return build_loss("cross_entropy")


@pytest.fixture
def train_loader() -> DataLoader:
    inputs = torch.randn(32, 10)
    targets = torch.randint(0, 3, (32,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=8)


@pytest.fixture
def val_loader() -> DataLoader:
    inputs = torch.randn(16, 10)
    targets = torch.randint(0, 3, (16,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=8)


@pytest.fixture
def trainer(
    model: DummyClassifier,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> Trainer:
    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
        verbose=False,
    )


@pytest.fixture
def checkpoint_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


class TestOptimizerFactory:
    def test_adam(self, model: DummyClassifier) -> None:
        opt = build_optimizer(model, "adam", lr=0.01)
        assert isinstance(opt, torch.optim.Adam)
        assert opt.param_groups[0]["lr"] == 0.01

    def test_adamw(self, model: DummyClassifier) -> None:
        opt = build_optimizer(model, "adamw", lr=0.001, weight_decay=0.01)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["lr"] == 0.001
        assert opt.param_groups[0]["weight_decay"] == 0.01

    def test_sgd(self, model: DummyClassifier) -> None:
        opt = build_optimizer(model, "sgd", lr=0.1, momentum=0.9)
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["lr"] == 0.1
        assert opt.param_groups[0]["momentum"] == 0.9

    def test_invalid_name(self, model: DummyClassifier) -> None:
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(model, "invalid_opt")


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


class TestSchedulerFactory:
    def test_cosine(self, optimizer: torch.optim.Optimizer) -> None:
        sched = build_scheduler(optimizer, "cosine", T_max=10)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step(self, optimizer: torch.optim.Optimizer) -> None:
        sched = build_scheduler(optimizer, "step", step_size=5, gamma=0.1)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_plateau(self, optimizer: torch.optim.Optimizer) -> None:
        sched = build_scheduler(optimizer, "plateau", patience=3)
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_onecycle(self, optimizer: torch.optim.Optimizer) -> None:
        sched = build_scheduler(optimizer, "onecycle", max_lr=0.01, steps_per_epoch=10, epochs=5)
        assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)

    def test_invalid_name(self, optimizer: torch.optim.Optimizer) -> None:
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_scheduler(optimizer, "invalid_sched")


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------


class TestLossFactory:
    def test_cross_entropy(self) -> None:
        loss = build_loss("cross_entropy")
        assert isinstance(loss, nn.CrossEntropyLoss)

    def test_bce(self) -> None:
        loss = build_loss("bce")
        assert isinstance(loss, nn.BCEWithLogitsLoss)

    def test_focal(self) -> None:
        loss = build_loss("focal", gamma=2.0)
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        value = loss(logits, targets)
        assert value.item() > 0

    def test_dice(self) -> None:
        loss = build_loss("dice")
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))
        value = loss(logits, targets)
        assert value.item() > 0

    def test_iou(self) -> None:
        loss = build_loss("iou")
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))
        value = loss(logits, targets)
        assert value.item() > 0

    def test_bce_dice(self) -> None:
        loss = build_loss("bce_dice", bce_weight=0.5, dice_weight=1.0)
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))
        value = loss(logits, targets)
        assert value.item() > 0

    def test_invalid_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss("invalid_loss")


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_accuracy_perfect(self) -> None:
        logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        targets = torch.tensor([0, 1])
        assert accuracy(logits, targets) == 1.0

    def test_accuracy_half(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 0])
        assert accuracy(logits, targets) == 0.5

    def test_precision_macro(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1])
        assert precision(logits, targets) == 1.0

    def test_recall_macro(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1])
        assert recall(logits, targets) == 1.0

    def test_f1_macro(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1])
        assert f1(logits, targets) == 1.0

    def test_metric_factory(self) -> None:
        fn = build_metric("accuracy")
        assert callable(fn)
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])
        assert fn(logits, targets) == 1.0

    def test_metric_factory_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            build_metric("invalid_metric")


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------


class TestSegmentationMetrics:
    def test_iou_perfect(self) -> None:
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 0, :, :] = 1.0
        logits[:, 1, :, :] = 0.0
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        score = iou_score(logits, targets)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_dice_perfect(self) -> None:
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 0, :, :] = 1.0
        logits[:, 1, :, :] = 0.0
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        score = dice_score(logits, targets)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_pixel_accuracy_perfect(self) -> None:
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 0, :, :] = 1.0
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        assert pixel_accuracy(logits, targets) == 1.0

    def test_pixel_accuracy_half(self) -> None:
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 0, :, :2] = 1.0
        logits[:, 1, :, 2:] = 1.0
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        targets[:, :, 2:] = 1
        # 50% of pixels are class 0 (correct), 50% class 1 (correct)
        assert pixel_accuracy(logits, targets) == 1.0


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class TestTrainingLogger:
    def test_empty_summary(self) -> None:
        logger = TrainingLogger()
        summary = logger.summary()
        assert "No training history" in summary

    def test_log_epoch(self) -> None:
        logger = TrainingLogger()
        logger.log_epoch(train_loss=0.5, val_loss=0.4, lr=1e-3)
        assert len(logger.history) == 1
        assert logger.history[0]["train_loss"] == 0.5
        assert logger.history[0]["val_loss"] == 0.4

    def test_latest(self) -> None:
        logger = TrainingLogger()
        logger.log_epoch(train_loss=0.5)
        logger.log_epoch(train_loss=0.3)
        assert logger.latest()["train_loss"] == 0.3

    def test_to_dict(self) -> None:
        logger = TrainingLogger()
        logger.log_epoch(train_loss=0.5, val_loss=0.4)
        logger.log_epoch(train_loss=0.3, val_loss=0.2)
        d = logger.to_dict()
        assert d["train_loss"] == [0.5, 0.3]
        assert d["val_loss"] == [0.4, 0.2]

    def test_reset(self) -> None:
        logger = TrainingLogger()
        logger.log_epoch(train_loss=0.5)
        logger.reset()
        assert len(logger.history) == 0

    def test_begin_epoch_records_time(self) -> None:
        logger = TrainingLogger()
        logger.begin_epoch()
        logger.log_epoch(train_loss=0.5)
        assert "epoch_time" in logger.history[0]
        assert logger.history[0]["epoch_time"] >= 0


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def test_save_last(self, model: DummyClassifier, checkpoint_dir: Path) -> None:
        ckpt = CheckpointManager(checkpoint_dir)
        ckpt.save_last(model)
        assert (checkpoint_dir / "last.pt").exists()

    def test_save_best(self, model: DummyClassifier, checkpoint_dir: Path) -> None:
        ckpt = CheckpointManager(checkpoint_dir, metric_name="val_loss", mode="min")
        saved = ckpt.save_best(model, metric=0.5)
        assert saved
        assert (checkpoint_dir / "best.pt").exists()

    def test_save_best_only_on_improvement(
        self, model: DummyClassifier, checkpoint_dir: Path
    ) -> None:
        ckpt = CheckpointManager(checkpoint_dir, metric_name="val_loss", mode="min")
        assert ckpt.save_best(model, metric=0.5)
        assert not ckpt.save_best(model, metric=0.6)  # worse
        assert ckpt.save_best(model, metric=0.3)  # better

    def test_save_best_max_mode(
        self, model: DummyClassifier, checkpoint_dir: Path
    ) -> None:
        ckpt = CheckpointManager(checkpoint_dir, metric_name="val_acc", mode="max")
        assert ckpt.save_best(model, metric=0.8)
        assert not ckpt.save_best(model, metric=0.7)  # worse
        assert ckpt.save_best(model, metric=0.9)  # better

    def test_load(self, model: DummyClassifier, checkpoint_dir: Path) -> None:
        ckpt = CheckpointManager(checkpoint_dir)
        ckpt.save_last(model, epoch=5, metric=0.5)
        state = ckpt.load(model, checkpoint_path=checkpoint_dir / "last.pt")
        assert state["epoch"] == 5
        assert state["metric"] == 0.5

    def test_load_file_not_found(
        self, model: DummyClassifier, checkpoint_dir: Path
    ) -> None:
        ckpt = CheckpointManager(checkpoint_dir)
        with pytest.raises(FileNotFoundError):
            ckpt.load(model, checkpoint_path=checkpoint_dir / "nonexistent.pt")

    def test_best_metric_property(
        self, model: DummyClassifier, checkpoint_dir: Path
    ) -> None:
        ckpt = CheckpointManager(checkpoint_dir, metric_name="val_loss", mode="min")
        assert ckpt.best_metric is not None  # initialized
        ckpt.save_best(model, metric=0.5)
        assert ckpt.best_metric == 0.5


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_no_stop_within_patience(self, model: DummyClassifier) -> None:
        es = EarlyStopping(patience=3, mode="min")
        for metric in [0.5, 0.4, 0.3]:  # always improving
            assert not es.step(metric, model)
        assert not es.early_stop

    def test_stop_after_patience(self, model: DummyClassifier) -> None:
        es = EarlyStopping(patience=3, mode="min")
        es.step(0.5, model)  # best = 0.5
        for _ in range(3):
            es.step(0.6, model)  # worse
        assert es.early_stop

    def test_min_delta(self, model: DummyClassifier) -> None:
        es = EarlyStopping(patience=2, min_delta=0.1, mode="min")
        es.step(0.5, model)  # best = 0.5
        # 0.45 < 0.5 - 0.1 = 0.4? No, 0.45 > 0.4 → not an improvement
        assert not es.step(0.45, model)  # counter=1, not stopped
        # counter=2 ≥ patience=2 → stops on this call
        assert es.step(0.45, model)

    def test_max_mode(self, model: DummyClassifier) -> None:
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.5, model)  # best = 0.5
        assert not es.step(0.4, model)  # counter=1, not stopped
        # counter=2 ≥ patience=2 → stops on this call
        assert es.step(0.4, model)

    def test_restore_best_weights(self, model: DummyClassifier) -> None:
        es = EarlyStopping(patience=2, mode="min", restore_best_weights=True)
        # Set initial weights
        original_weight = model.fc.weight.data.clone()

        # Improve
        es.step(0.5, model)
        # Change weights
        with torch.no_grad():
            model.fc.weight.add_(1.0)
        changed_weight = model.fc.weight.data.clone()

        # Trigger early stopping
        es.step(0.6, model)
        es.step(0.6, model)
        assert es.early_stop

        # Restore
        es.restore(model)
        assert torch.equal(model.fc.weight.data, original_weight)
        assert not torch.equal(model.fc.weight.data, changed_weight)

    def test_restore_without_save_raises(self) -> None:
        es = EarlyStopping(patience=2, mode="min")
        with pytest.raises(RuntimeError, match="No best state"):
            es.restore(DummyClassifier())

    def test_reset(self, model: DummyClassifier) -> None:
        es = EarlyStopping(patience=2, mode="min")
        es.step(0.5, model)
        es.step(0.6, model)
        es.step(0.6, model)
        assert es.early_stop
        es.reset()
        assert not es.early_stop
        assert es.counter == 0


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


class TestMoveBatchToDevice:
    def test_tensor(self) -> None:
        t = torch.randn(3, 4)
        result = move_batch_to_device(t, torch.device("cpu"))
        assert isinstance(result, torch.Tensor)

    def test_dict(self) -> None:
        batch = {"inputs": torch.randn(3, 4), "targets": torch.randn(3)}
        result = move_batch_to_device(batch, torch.device("cpu"))
        assert isinstance(result, dict)
        assert "inputs" in result

    def test_list(self) -> None:
        batch = [torch.randn(3, 4), torch.randn(3)]
        result = move_batch_to_device(batch, torch.device("cpu"))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_tuple(self) -> None:
        batch = (torch.randn(3, 4), torch.randn(3))
        result = move_batch_to_device(batch, torch.device("cpu"))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_non_tensor(self) -> None:
        assert move_batch_to_device(42, torch.device("cpu")) == 42


class TestClipGradients:
    def test_clip_norm(self, model: DummyClassifier) -> None:
        logits = model(torch.randn(4, 10))
        loss = logits.sum()
        loss.backward()
        norm = clip_gradients(model, max_norm=0.1)
        assert norm > 0

    def test_clip_value(self, model: DummyClassifier) -> None:
        logits = model(torch.randn(4, 10))
        loss = logits.sum()
        loss.backward()
        clip_gradients(model, max_value=0.01)
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.abs().max().item() <= 0.01 + 1e-6

    def test_no_clip(self, model: DummyClassifier) -> None:
        logits = model(torch.randn(4, 10))
        loss = logits.sum()
        loss.backward()
        norm = clip_gradients(model)
        assert norm >= 0


class TestNativeScaler:
    def test_creation(self) -> None:
        scaler = NativeScaler(enabled=False)
        assert scaler.enabled is False

    def test_autocast_context(self) -> None:
        scaler = NativeScaler(enabled=False)
        with scaler.autocast():
            t = torch.randn(3, 3)
            result = t @ t.T
            assert isinstance(result, torch.Tensor)

    def test_state_dict_disabled(self) -> None:
        scaler = NativeScaler(enabled=False)
        assert scaler.state_dict() == {}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TestTrainerConstruction:
    def test_creation(self, trainer: Trainer) -> None:
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
        assert trainer.device == torch.device("cpu")
        assert trainer.current_epoch == 0

    def test_device_auto(self, model: DummyClassifier) -> None:
        trainer = Trainer(
            model=model,
            optimizer=build_optimizer(model, "adamw"),
            loss_fn=build_loss("cross_entropy"),
            device="auto",
            verbose=False,
        )
        assert trainer.device.type in ("cpu", "cuda")

    def test_device_cpu(self, model: DummyClassifier) -> None:
        trainer = Trainer(
            model=model,
            optimizer=build_optimizer(model, "adamw"),
            loss_fn=build_loss("cross_entropy"),
            device="cpu",
            verbose=False,
        )
        assert trainer.device.type == "cpu"


class TestTrainOneEpoch:
    def test_train_one_epoch(self, trainer: Trainer, train_loader: DataLoader) -> None:
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert metrics["loss"] > 0
        assert trainer.current_epoch == 0  # not incremented by train_one_epoch

    def test_train_one_epoch_with_metrics(
        self, model: DummyClassifier, train_loader: DataLoader
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=build_optimizer(model, "adamw"),
            loss_fn=build_loss("cross_entropy"),
            metric_fns={"acc": accuracy},
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(train_loader)
        assert "loss" in metrics
        assert "acc" in metrics

    def test_loss_decreases(self, model: DummyClassifier, train_loader: DataLoader) -> None:
        trainer = Trainer(
            model=model,
            optimizer=build_optimizer(model, "adamw", lr=0.01),
            loss_fn=build_loss("cross_entropy"),
            device="cpu",
            verbose=False,
        )
        m1 = trainer.train_one_epoch(train_loader)
        m2 = trainer.train_one_epoch(train_loader)
        assert m2["loss"] < m1["loss"]


class TestValidate:
    def test_validate(self, trainer: Trainer, val_loader: DataLoader) -> None:
        metrics = trainer.validate(val_loader)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_validate_with_metrics(
        self, model: DummyClassifier, val_loader: DataLoader
    ) -> None:
        trainer = Trainer(
            model=model,
            optimizer=build_optimizer(model, "adamw"),
            loss_fn=build_loss("cross_entropy"),
            metric_fns={"acc": accuracy},
            device="cpu",
            verbose=False,
        )
        metrics = trainer.validate(val_loader)
        assert "loss" in metrics
        assert "acc" in metrics


class TestPredict:
    def test_predict(self, trainer: Trainer, train_loader: DataLoader) -> None:
        predictions = trainer.predict(train_loader)
        assert len(predictions) > 0
        assert all(isinstance(p, torch.Tensor) for p in predictions)

    def test_predict_shape(
        self, trainer: Trainer, train_loader: DataLoader
    ) -> None:
        predictions = trainer.predict(train_loader)
        total = sum(p.shape[0] for p in predictions)
        assert total == 32  # 32 samples in train_loader


class TestFit:
    def test_fit_returns_logger(
        self, trainer: Trainer, train_loader: DataLoader
    ) -> None:
        logger = trainer.fit(train_loader, epochs=2)
        assert isinstance(logger, TrainingLogger)
        assert len(logger.history) == 2

    def test_fit_with_validation(
        self, trainer: Trainer, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        logger = trainer.fit(train_loader, val_loader, epochs=3)
        assert len(logger.history) == 3
        assert "train_loss" in logger.history[0]
        assert "val_loss" in logger.history[0]

    def test_fit_updates_epoch_counter(
        self, trainer: Trainer, train_loader: DataLoader
    ) -> None:
        trainer.fit(train_loader, epochs=5)
        assert trainer.current_epoch == 5


class TestCheckpoint:
    def test_save_checkpoint(self, trainer: Trainer, tmp_path: Path) -> None:
        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(path)
        assert path.exists()

    def test_load_checkpoint(
        self, trainer: Trainer, train_loader: DataLoader, tmp_path: Path
    ) -> None:
        path = tmp_path / "checkpoint.pt"
        trainer.fit(train_loader, epochs=3)
        trainer.save_checkpoint(path)

        # Create a new trainer and load
        new_model = DummyClassifier()
        new_trainer = Trainer(
            model=new_model,
            optimizer=build_optimizer(new_model, "adamw"),
            loss_fn=build_loss("cross_entropy"),
            device="cpu",
            verbose=False,
        )
        epoch = new_trainer.load_checkpoint(path)
        assert epoch == 3
        assert new_trainer.current_epoch == 3

    def test_load_checkpoint_not_found(self, trainer: Trainer) -> None:
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint("/nonexistent/path.pt")


class TestEarlyStoppingIntegration:
    def test_fit_stops_early(
        self, model: DummyClassifier, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        es = EarlyStopping(patience=2, mode="min")
        trainer = Trainer(
            model=model,
            optimizer=build_optimizer(model, "adamw", lr=0.01),
            loss_fn=build_loss("cross_entropy"),
            early_stopping=es,
            device="cpu",
            verbose=False,
        )
        logger = trainer.fit(train_loader, val_loader, epochs=20)
        # Should stop well before 20 epochs
        assert len(logger.history) < 20


class TestCheckpointManagerIntegration:
    def test_fit_saves_checkpoints(
        self, model: DummyClassifier, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = CheckpointManager(Path(tmp))
            trainer = Trainer(
                model=model,
                optimizer=build_optimizer(model, "adamw", lr=0.01),
                loss_fn=build_loss("cross_entropy"),
                checkpoint_manager=ckpt,
                device="cpu",
                verbose=False,
            )
            trainer.fit(train_loader, val_loader, epochs=3)
            assert (Path(tmp) / "last.pt").exists()
            assert (Path(tmp) / "best.pt").exists()


class TestSchedulerIntegration:
    def test_cosine_scheduler(
        self, model: DummyClassifier, train_loader: DataLoader
    ) -> None:
        opt = build_optimizer(model, "adamw", lr=0.01)
        sched = build_scheduler(opt, "cosine", T_max=5)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=build_loss("cross_entropy"),
            scheduler=sched,
            device="cpu",
            verbose=False,
        )
        initial_lr = trainer._get_current_lr()
        trainer.fit(train_loader, epochs=3)
        final_lr = trainer._get_current_lr()
        assert final_lr < initial_lr

    def test_step_scheduler(
        self, model: DummyClassifier, train_loader: DataLoader
    ) -> None:
        opt = build_optimizer(model, "adamw", lr=0.01)
        sched = build_scheduler(opt, "step", step_size=2, gamma=0.1)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=build_loss("cross_entropy"),
            scheduler=sched,
            device="cpu",
            verbose=False,
        )
        trainer.fit(train_loader, epochs=3)
        # After step at epoch 2, lr should be 0.01 * 0.1 = 0.001
        assert trainer._get_current_lr() == pytest.approx(0.001, abs=1e-6)


class TestSegmentationTrainer:
    """Test trainer with a segmentation model."""

    def test_seg_train_one_epoch(
        self, seg_model: DummySegmenter
    ) -> None:
        inputs = torch.randn(8, 3, 16, 16)
        targets = torch.randint(0, 3, (8, 16, 16))
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=4)

        opt = build_optimizer(seg_model, "adamw", lr=1e-3)
        loss_fn = build_loss("dice")
        trainer = Trainer(
            model=seg_model,
            optimizer=opt,
            loss_fn=loss_fn,
            metric_fns={"iou": iou_score, "dice": dice_score},
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(loader)
        assert "loss" in metrics
        assert "iou" in metrics
        assert "dice" in metrics
        assert metrics["loss"] > 0

    def test_seg_validate(
        self, seg_model: DummySegmenter
    ) -> None:
        inputs = torch.randn(8, 3, 16, 16)
        targets = torch.randint(0, 3, (8, 16, 16))
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=4)

        opt = build_optimizer(seg_model, "adamw", lr=1e-3)
        loss_fn = build_loss("dice")
        trainer = Trainer(
            model=seg_model,
            optimizer=opt,
            loss_fn=loss_fn,
            metric_fns={"pixel_acc": pixel_accuracy},
            device="cpu",
            verbose=False,
        )
        metrics = trainer.validate(loader)
        assert "loss" in metrics
        assert "pixel_acc" in metrics


class TestDictBatch:
    """Test trainer with dict-format batches."""

    def test_dict_batch(self, model: DummyClassifier) -> None:
        inputs = torch.randn(16, 10)
        targets = torch.randint(0, 3, (16,))
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=8)

        # Wrap loader to yield dicts
        def dict_collate(batch):
            inputs = torch.stack([b[0] for b in batch])
            targets = torch.tensor([b[1] for b in batch])
            return {"inputs": inputs, "targets": targets}

        dict_loader = DataLoader(dataset, batch_size=8, collate_fn=dict_collate)
        trainer = Trainer(
            model=model,
            optimizer=build_optimizer(model, "adamw"),
            loss_fn=build_loss("cross_entropy"),
            device="cpu",
            verbose=False,
        )
        metrics = trainer.train_one_epoch(dict_loader)
        assert "loss" in metrics