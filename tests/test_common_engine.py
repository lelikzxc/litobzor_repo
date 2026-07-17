"""Comprehensive tests for common.engine infrastructure.

Uses small dummy models and synthetic datasets only. No CUDA required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.engine import (
    Builder,
    Engine,
    EngineConfig,
    EngineState,
    build_dataset,
    build_loss,
    build_metric,
    build_model,
    build_optimizer,
    build_scheduler,
    is_registered,
    list_registered,
    register_dataset,
    register_loss,
    register_metric,
    register_model,
    register_optimizer,
    register_scheduler,
)


# ---------------------------------------------------------------------------
# Dummy models
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> DummyClassifier:
    return DummyClassifier()


@pytest.fixture
def seg_model() -> DummySegmenter:
    return DummySegmenter()


@pytest.fixture
def sample_config() -> dict:
    return {
        "model": {"name": "dummy_cls", "kwargs": {"in_features": 10, "num_classes": 3}},
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "scheduler": {"name": "cosine", "kwargs": {"T_max": 10}},
        "loss": {"name": "cross_entropy"},
        "metrics": ["accuracy"],
        "checkpoint": {"save_dir": "/tmp/checkpoints", "metric_name": "val_loss", "mode": "min"},
        "early_stopping": {"patience": 10, "mode": "min"},
        "training": {"device": "cpu", "amp": False},
        "inference": {"device": "cpu"},
    }


@pytest.fixture
def train_loader() -> DataLoader:
    data = TensorDataset(torch.randn(20, 10), torch.randint(0, 3, (20,)))
    return DataLoader(data, batch_size=4)


@pytest.fixture
def val_loader() -> DataLoader:
    data = TensorDataset(torch.randn(10, 10), torch.randint(0, 3, (10,)))
    return DataLoader(data, batch_size=4)


# ---------------------------------------------------------------------------
# EngineState
# ---------------------------------------------------------------------------


class TestEngineState:
    def test_default_values(self) -> None:
        state = EngineState()
        assert state.epoch == 0
        assert state.best_metric is None
        assert state.global_step == 0
        assert state.training_finished is False

    def test_update_metric_first_time(self) -> None:
        state = EngineState()
        assert state.update_metric(0.5) is True
        assert state.best_metric == 0.5
        assert state.current_metric == 0.5

    def test_update_metric_min_mode(self) -> None:
        state = EngineState(mode="min")
        state.update_metric(0.5)
        assert state.update_metric(0.3) is True  # better (lower)
        assert state.best_metric == 0.3
        assert state.update_metric(0.4) is False  # worse (higher)
        assert state.best_metric == 0.3

    def test_update_metric_max_mode(self) -> None:
        state = EngineState(mode="max")
        state.update_metric(0.5)
        assert state.update_metric(0.7) is True  # better (higher)
        assert state.best_metric == 0.7
        assert state.update_metric(0.6) is False  # worse (lower)
        assert state.best_metric == 0.7

    def test_reset(self) -> None:
        state = EngineState(epoch=5, best_metric=0.5, global_step=100)
        state.reset()
        assert state.epoch == 0
        assert state.best_metric is None
        assert state.global_step == 0

    def test_to_dict_and_from_dict(self) -> None:
        state = EngineState(
            epoch=3,
            best_metric=0.2,
            best_metric_name="val_loss",
            current_metric=0.25,
            global_step=150,
            training_finished=True,
            checkpoint_path="/tmp/ckpt.pt",
            mode="min",
            extra={"custom": 42},
        )
        data = state.to_dict()
        restored = EngineState.from_dict(data)
        assert restored.epoch == 3
        assert restored.best_metric == 0.2
        assert restored.global_step == 150
        assert restored.training_finished is True
        assert restored.extra["custom"] == 42

    def test_extra_field(self) -> None:
        state = EngineState()
        state.extra["learning_rate"] = 0.001
        assert state.extra["learning_rate"] == 0.001


# ---------------------------------------------------------------------------
# EngineConfig
# ---------------------------------------------------------------------------


class TestEngineConfig:
    def test_from_dict(self) -> None:
        cfg = EngineConfig.from_dict({"training": {"batch_size": 32}})
        assert cfg.get("training.batch_size") == 32

    def test_get_default(self) -> None:
        cfg = EngineConfig.from_dict({})
        assert cfg.get("missing.key", 42) == 42

    def test_getitem(self) -> None:
        cfg = EngineConfig.from_dict({"a": {"b": 1}})
        assert cfg["a.b"] == 1

    def test_getitem_missing_raises(self) -> None:
        cfg = EngineConfig.from_dict({})
        with pytest.raises(KeyError):
            _ = cfg["missing"]

    def test_contains(self) -> None:
        cfg = EngineConfig.from_dict({"a": 1})
        assert "a" in cfg
        assert "b" not in cfg

    def test_merge(self) -> None:
        cfg = EngineConfig.from_dict({"a": 1, "b": 2})
        cfg.merge({"b": 3, "c": 4})
        assert cfg["a"] == 1
        assert cfg["b"] == 3
        assert cfg["c"] == 4

    def test_merge_deep(self) -> None:
        cfg = EngineConfig.from_dict({"optimizer": {"lr": 1e-3, "weight_decay": 0.0}})
        cfg.merge_deep({"optimizer": {"lr": 1e-4}})
        assert cfg["optimizer.lr"] == 1e-4
        assert cfg["optimizer.weight_decay"] == 0.0  # preserved

    def test_to_dict(self) -> None:
        cfg = EngineConfig.from_dict({"a": {"b": [1, 2, 3]}})
        d = cfg.to_dict()
        assert d["a"]["b"] == [1, 2, 3]
        # Ensure it's a copy
        d["a"]["b"].append(4)
        assert cfg["a.b"] == [1, 2, 3]

    def test_from_yaml_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            EngineConfig.from_yaml("/nonexistent/config.yaml")

    def test_from_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("training:\n  batch_size: 32\n")
        cfg = EngineConfig.from_yaml(path)
        assert cfg["training.batch_size"] == 32

    def test_to_yaml(self, tmp_path: Path) -> None:
        cfg = EngineConfig.from_dict({"a": 1, "b": 2})
        path = tmp_path / "out.yaml"
        cfg.to_yaml(path)
        assert path.exists()
        loaded = EngineConfig.from_yaml(path)
        assert loaded["a"] == 1
        assert loaded["b"] == 2

    def test_repr(self) -> None:
        cfg = EngineConfig.from_dict({"key": "val"})
        assert "EngineConfig" in repr(cfg)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_build_model(self) -> None:
        register_model("test_cls_model", DummyClassifier)
        model = build_model("test_cls_model", in_features=10, num_classes=3)
        assert isinstance(model, DummyClassifier)
        assert model.fc.in_features == 10

    def test_register_duplicate_model_raises(self) -> None:
        register_model("test_dup_model", DummyClassifier)
        with pytest.raises(ValueError, match="already registered"):
            register_model("test_dup_model", DummyClassifier)

    def test_build_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent_model_name")

    def test_register_and_build_dataset(self) -> None:
        class DummyDataset:
            def __init__(self, size: int = 10) -> None:
                self.size = size

        register_dataset("test_data_ds", DummyDataset)
        ds = build_dataset("test_data_ds", size=20)
        assert ds.size == 20

    def test_register_optimizer(self) -> None:
        register_optimizer("test_opt_opt", torch.optim.Adam)
        assert is_registered("optimizers", "test_opt_opt")

    def test_register_scheduler(self) -> None:
        register_scheduler("test_sched_sched", torch.optim.lr_scheduler.CosineAnnealingLR)
        assert is_registered("schedulers", "test_sched_sched")

    def test_register_loss(self) -> None:
        register_loss("test_loss_fn", nn.CrossEntropyLoss)
        assert is_registered("losses", "test_loss_fn")

    def test_register_metric(self) -> None:
        def dummy_metric(logits, targets) -> float:
            return 1.0

        register_metric("test_metric_fn", dummy_metric)
        assert is_registered("metrics", "test_metric_fn")

    def test_list_registered(self) -> None:
        names = list_registered("optimizers")
        assert "adamw" in names

    def test_list_registered_unknown_category(self) -> None:
        with pytest.raises(ValueError, match="Unknown category"):
            list_registered("invalid")

    def test_is_registered(self) -> None:
        assert is_registered("losses", "cross_entropy")
        assert not is_registered("losses", "nonexistent")

    def test_build_optimizer_builtin(self) -> None:
        model = DummyClassifier()
        opt = build_optimizer(model, name="adamw", lr=1e-3)
        assert isinstance(opt, torch.optim.AdamW)

    def test_build_scheduler_builtin(self) -> None:
        model = DummyClassifier()
        opt = build_optimizer(model, name="adamw")
        sched = build_scheduler(opt, name="cosine", T_max=10)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_build_loss_builtin(self) -> None:
        loss = build_loss("cross_entropy")
        assert isinstance(loss, nn.Module)

    def test_build_metric_builtin(self) -> None:
        metric = build_metric("accuracy")
        assert callable(metric)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TestBuilder:
    def test_builder_from_config_dict(self, sample_config: dict) -> None:
        builder = Builder(sample_config)
        assert builder.config is not None
        assert builder.config["optimizer.name"] == "adamw"

    def test_builder_from_engine_config(self, sample_config: dict) -> None:
        engine_cfg = EngineConfig.from_dict(sample_config)
        builder = Builder(engine_cfg)
        assert builder.config is not None

    def test_builder_from_yaml(self, tmp_path: Path, sample_config: dict) -> None:
        import yaml
        path = tmp_path / "cfg.yaml"
        with open(path, "w") as f:
            yaml.dump(sample_config, f)
        builder = Builder(str(path))
        assert builder.config is not None

    def test_builder_build_model_from_instance(self) -> None:
        builder = Builder()
        model = DummyClassifier()
        result = builder.build_model(model)
        assert result is model

    def test_builder_build_model_from_config(self, sample_config: dict) -> None:
        register_model("dummy_cls_builder", DummyClassifier)
        cfg = dict(sample_config)
        cfg["model"] = {"name": "dummy_cls_builder", "kwargs": {"in_features": 10, "num_classes": 3}}
        builder = Builder(cfg)
        model = builder.build_model()
        assert isinstance(model, DummyClassifier)

    def test_builder_build_model_no_config_raises(self) -> None:
        builder = Builder()
        with pytest.raises(ValueError, match="No model provided"):
            builder.build_model()

    def test_builder_build_optimizer(self, sample_config: dict) -> None:
        builder = Builder(sample_config)
        model = DummyClassifier()
        opt = builder.build_optimizer(model)
        assert isinstance(opt, torch.optim.Optimizer)

    def test_builder_build_scheduler(self, sample_config: dict) -> None:
        builder = Builder(sample_config)
        model = DummyClassifier()
        opt = builder.build_optimizer(model)
        sched = builder.build_scheduler(opt)
        assert sched is not None

    def test_builder_build_scheduler_none(self) -> None:
        builder = Builder()
        model = DummyClassifier()
        opt = builder.build_optimizer(model)
        sched = builder.build_scheduler(opt)
        assert sched is None

    def test_builder_build_loss(self, sample_config: dict) -> None:
        builder = Builder(sample_config)
        loss = builder.build_loss()
        assert isinstance(loss, nn.Module)

    def test_builder_build_metrics(self, sample_config: dict) -> None:
        builder = Builder(sample_config)
        metrics = builder.build_metrics()
        assert "accuracy" in metrics

    def test_builder_build_metrics_empty(self) -> None:
        builder = Builder()
        metrics = builder.build_metrics()
        assert metrics == {}

    def test_builder_build_checkpoint_manager(self, sample_config: dict) -> None:
        builder = Builder(sample_config)
        ckpt = builder.build_checkpoint_manager()
        assert ckpt is not None
        assert str(ckpt.save_dir) == "/tmp/checkpoints"

    def test_builder_build_checkpoint_manager_none(self) -> None:
        builder = Builder()
        ckpt = builder.build_checkpoint_manager()
        assert ckpt is None

    def test_builder_build_early_stopping(self, sample_config: dict) -> None:
        builder = Builder(sample_config)
        es = builder.build_early_stopping()
        assert es is not None

    def test_builder_build_early_stopping_none(self) -> None:
        builder = Builder()
        es = builder.build_early_stopping()
        assert es is None

    def test_builder_build_logger(self) -> None:
        builder = Builder()
        logger = builder.build_logger()
        from common.training import TrainingLogger
        assert isinstance(logger, TrainingLogger)

    def test_builder_build_scaler(self, sample_config: dict) -> None:
        builder = Builder(sample_config)
        scaler = builder.build_scaler()
        assert scaler is not None

    def test_builder_build_state(self, sample_config: dict) -> None:
        builder = Builder(sample_config)
        state = builder.build_state()
        assert state.best_metric_name == "val_loss"
        assert state.mode == "min"

    def test_builder_build_all(self, sample_config: dict) -> None:
        register_model("dummy_cls_build_all", DummyClassifier)
        cfg = dict(sample_config)
        cfg["model"] = {"name": "dummy_cls_build_all", "kwargs": {"in_features": 10, "num_classes": 3}}
        builder = Builder(cfg)
        components = builder.build_all()
        assert "model" in components
        assert "optimizer" in components
        assert "scheduler" in components
        assert "loss_fn" in components
        assert "metric_fns" in components
        assert "checkpoint_manager" in components
        assert "early_stopping" in components
        assert "logger" in components
        assert "scaler" in components
        assert "trainer" in components
        assert "predictor" in components
        assert "state" in components
        assert isinstance(components["model"], DummyClassifier)

    def test_builder_with_config_chaining(self) -> None:
        builder = Builder()
        builder.with_config({"optimizer": {"name": "sgd", "lr": 0.01}})
        assert builder.config is not None
        assert builder.config["optimizer.name"] == "sgd"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:
    def test_engine_init_with_model_instance(self) -> None:
        engine = Engine(DummyClassifier(), device="cpu")
        assert isinstance(engine.model, DummyClassifier)
        assert engine.state.epoch == 0

    def test_engine_init_with_config(self, sample_config: dict) -> None:
        register_model("dummy_cls_engine_cfg", DummyClassifier)
        cfg = dict(sample_config)
        cfg["model"] = {"name": "dummy_cls_engine_cfg", "kwargs": {"in_features": 10, "num_classes": 3}}
        engine = Engine("dummy_cls_engine_cfg", config=cfg, device="cpu")
        assert isinstance(engine.model, DummyClassifier)
        assert engine.config is not None

    def test_engine_init_with_yaml_config(self, tmp_path: Path, sample_config: dict) -> None:
        import yaml
        register_model("dummy_cls_yaml", DummyClassifier)
        cfg = dict(sample_config)
        cfg["model"] = {"name": "dummy_cls_yaml", "kwargs": {"in_features": 10, "num_classes": 3}}
        path = tmp_path / "cfg.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)
        engine = Engine("dummy_cls_yaml", config=str(path), device="cpu")
        assert isinstance(engine.model, DummyClassifier)

    def test_engine_fit(self, model: nn.Module, train_loader: DataLoader) -> None:
        engine = Engine(model, device="cpu")
        logger = engine.fit(train_loader, epochs=2)
        assert engine.state.training_finished is True
        assert engine.state.epoch == 2
        assert len(logger.history) == 2

    def test_engine_fit_with_validation(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        engine = Engine(model, device="cpu")
        logger = engine.fit(train_loader, val_loader=val_loader, epochs=2)
        assert len(logger.history) == 2
        latest = logger.latest()
        assert "train_loss" in latest
        assert "val_loss" in latest

    def test_engine_validate(
        self, model: nn.Module, val_loader: DataLoader
    ) -> None:
        engine = Engine(model, device="cpu")
        metrics = engine.validate(val_loader)
        assert "loss" in metrics

    def test_engine_test(
        self, model: nn.Module, val_loader: DataLoader
    ) -> None:
        engine = Engine(model, device="cpu")
        metrics = engine.test(val_loader)
        assert "loss" in metrics

    def test_engine_predict(
        self, model: nn.Module, train_loader: DataLoader
    ) -> None:
        engine = Engine(model, device="cpu")
        results = engine.predict(train_loader)
        assert len(results) > 0
        for r in results:
            assert "logits" in r
            assert "probs" in r
            assert "prediction" in r

    def test_engine_predict_single(self, model: nn.Module) -> None:
        engine = Engine(model, device="cpu")
        image = torch.randn(1, 10)
        result = engine.predict_single(image)
        assert "logits" in result
        assert "probs" in result
        assert "prediction" in result

    def test_engine_save_and_load(self, model: nn.Module, tmp_path: Path) -> None:
        engine = Engine(model, device="cpu")
        ckpt_path = tmp_path / "model.pt"
        saved_path = engine.save(ckpt_path)
        assert saved_path.exists()

        # Load into a new engine
        engine2 = Engine(DummyClassifier(), device="cpu")
        epoch = engine2.load(ckpt_path)
        assert epoch >= 0

    def test_engine_save_without_path(self, model: nn.Module, tmp_path: Path) -> None:
        engine = Engine(
            model,
            config={"checkpoint": {"save_dir": str(tmp_path)}},
            device="cpu",
        )
        saved_path = engine.save()
        assert saved_path.exists()

    def test_engine_summary(self, model: nn.Module) -> None:
        engine = Engine(model, device="cpu")
        summary = engine.summary()
        assert "model" in summary
        assert "optimizer" in summary
        assert "state" in summary
        assert summary["model"] == "DummyClassifier"

    def test_engine_reset(self, model: nn.Module, train_loader: DataLoader) -> None:
        engine = Engine(model, device="cpu")
        engine.fit(train_loader, epochs=2)
        assert engine.state.epoch == 2
        engine.reset()
        assert engine.state.epoch == 0
        assert engine.trainer.current_epoch == 0

    def test_engine_set_model(self, model: nn.Module) -> None:
        engine = Engine(model, device="cpu")
        new_model = DummyClassifier(in_features=10, num_classes=5)
        engine.set_model(new_model)
        assert engine.model is new_model

    def test_engine_set_device(self, model: nn.Module) -> None:
        engine = Engine(model, device="cpu")
        engine.set_device("cpu")
        assert engine.device == "cpu"

    def test_engine_with_metrics(self, model: nn.Module, train_loader: DataLoader) -> None:
        engine = Engine(
            model,
            config={"metrics": ["accuracy"]},
            device="cpu",
        )
        logger = engine.fit(train_loader, epochs=1)
        latest = logger.latest()
        assert "train_accuracy" in latest or "train_loss" in latest

    def test_engine_with_checkpoint_and_early_stopping(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, tmp_path: Path
    ) -> None:
        engine = Engine(
            model,
            config={
                "checkpoint": {"save_dir": str(tmp_path / "ckpts")},
                "early_stopping": {"patience": 5},
                "metrics": ["accuracy"],
            },
            device="cpu",
        )
        logger = engine.fit(train_loader, val_loader=val_loader, epochs=2)
        assert engine.checkpoint_manager is not None
        assert engine.early_stopping is not None
        assert len(logger.history) == 2

    def test_engine_segmentation(self, seg_model: nn.Module) -> None:
        engine = Engine(
            seg_model,
            config={"loss": {"name": "dice"}},
            device="cpu",
        )
        data = TensorDataset(torch.randn(10, 3, 8, 8), torch.randint(0, 3, (10, 8, 8)))
        loader = DataLoader(data, batch_size=4)
        metrics = engine.validate(loader)
        assert "loss" in metrics

    def test_engine_state_persists_across_save_load(
        self, model: nn.Module, tmp_path: Path
    ) -> None:
        engine = Engine(model, device="cpu")
        engine.state.epoch = 5
        engine.state.best_metric = 0.25
        engine.state.global_step = 200

        ckpt_path = tmp_path / "state_test.pt"
        engine.save(ckpt_path)

        engine2 = Engine(DummyClassifier(), device="cpu")
        engine2.load(ckpt_path)
        assert engine2.state.epoch == 5
        assert engine2.state.best_metric == 0.25
        assert engine2.state.global_step == 200


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_training_pipeline(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        """End-to-end: train, validate, predict, save, load."""
        engine = Engine(model, device="cpu")

        # Train
        logger = engine.fit(train_loader, val_loader=val_loader, epochs=2)
        assert len(logger.history) == 2

        # Validate
        val_metrics = engine.validate(val_loader)
        assert "loss" in val_metrics

        # Predict
        results = engine.predict(train_loader)
        assert len(results) > 0

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            engine.save(ckpt_path)
            engine2 = Engine(DummyClassifier(), device="cpu")
            engine2.load(ckpt_path)
            assert engine2.state.epoch == 2
        finally:
            Path(ckpt_path).unlink(missing_ok=True)
            state_path = Path(ckpt_path).with_suffix(".state.json")
            state_path.unlink(missing_ok=True)

    def test_engine_with_registered_model(self, sample_config: dict) -> None:
        """Engine with a model registered in the registry."""
        register_model("dummy_cls_registered", DummyClassifier)
        cfg = dict(sample_config)
        cfg["model"] = {"name": "dummy_cls_registered", "kwargs": {"in_features": 10, "num_classes": 3}}
        engine = Engine("dummy_cls_registered", config=cfg, device="cpu")
        assert isinstance(engine.model, DummyClassifier)

        data = TensorDataset(torch.randn(10, 10), torch.randint(0, 3, (10,)))
        loader = DataLoader(data, batch_size=4)
        metrics = engine.validate(loader)
        assert "loss" in metrics

    def test_builder_all_in_one(self, sample_config: dict) -> None:
        """Builder.build_all creates all components correctly."""
        register_model("dummy_cls_in_one", DummyClassifier)
        cfg = dict(sample_config)
        cfg["model"] = {"name": "dummy_cls_in_one", "kwargs": {"in_features": 10, "num_classes": 3}}
        builder = Builder(cfg)
        components = builder.build_all()

        # Verify all components are functional
        model = components["model"]
        optimizer = components["optimizer"]
        loss_fn = components["loss_fn"]
        trainer = components["trainer"]
        predictor = components["predictor"]

        assert isinstance(model, DummyClassifier)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(loss_fn, nn.Module)
        assert trainer.model is model
        assert predictor.model is model

    def test_engine_config_deep_merge(self) -> None:
        """Deep merge preserves nested keys."""
        base = EngineConfig.from_dict({
            "model": {"name": "test", "kwargs": {"a": 1, "b": 2}},
            "optimizer": {"lr": 1e-3},
        })
        base.merge_deep({"model": {"kwargs": {"a": 10}}, "optimizer": {"weight_decay": 0.01}})
        assert base["model.name"] == "test"
        assert base["model.kwargs.a"] == 10
        assert base["model.kwargs.b"] == 2  # preserved
        assert base["optimizer.lr"] == 1e-3
        assert base["optimizer.weight_decay"] == 0.01