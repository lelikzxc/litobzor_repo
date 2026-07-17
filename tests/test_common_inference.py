"""Comprehensive tests for common.inference infrastructure.

Uses small dummy models and synthetic tensors only. No CUDA required.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.inference import (
    Predictor,
    benchmark_model,
    disable_gradients,
    export_onnx,
    export_torchscript,
    get_best_device,
    inference_memory,
    load_checkpoint,
    logits_to_class,
    logits_to_mask,
    logits_to_probs,
    model_size_mb,
    move_to_device,
    parameter_count,
    plot_classification_result,
    plot_segmentation_comparison,
    plot_segmentation_result,
    seed_everything,
    set_eval_mode,
    topk_predictions,
)


# ---------------------------------------------------------------------------
# Dummy models
# ---------------------------------------------------------------------------


class DummyClassifier(nn.Module):
    """Minimal classifier for testing inference."""

    def __init__(self, in_features: int = 10, num_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DummySegmenter(nn.Module):
    """Minimal segmenter for testing inference."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DummyTinyNet(nn.Module):
    """Minimal network for export/benchmark testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).relu()
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def cls_model() -> DummyClassifier:
    return DummyClassifier(in_features=10, num_classes=3)


@pytest.fixture
def seg_model() -> DummySegmenter:
    return DummySegmenter(num_classes=3)


@pytest.fixture
def tiny_model() -> DummyTinyNet:
    return DummyTinyNet()


@pytest.fixture
def cls_logits() -> torch.Tensor:
    return torch.tensor([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]])


@pytest.fixture
def seg_logits() -> torch.Tensor:
    return torch.randn(1, 3, 8, 8)


@pytest.fixture
def sample_image() -> torch.Tensor:
    return torch.randn(3, 32, 32)


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------


class TestDevice:
    def test_get_best_device_returns_device(self) -> None:
        dev = get_best_device()
        assert isinstance(dev, torch.device)

    def test_move_to_device(self, cls_model: nn.Module) -> None:
        model = move_to_device(cls_model, "cpu")
        assert next(model.parameters()).device.type == "cpu"

    def test_move_to_device_none(self, cls_model: nn.Module) -> None:
        model = move_to_device(cls_model, None)
        assert isinstance(model, nn.Module)

    def test_model_size_mb_positive(self, cls_model: nn.Module) -> None:
        size = model_size_mb(cls_model)
        assert size > 0

    def test_parameter_count(self, cls_model: nn.Module) -> None:
        count = parameter_count(cls_model)
        assert count > 0

    def test_parameter_count_trainable_only(self, cls_model: nn.Module) -> None:
        count = parameter_count(cls_model, trainable_only=True)
        assert count > 0

    def test_parameter_count_frozen(self, cls_model: nn.Module) -> None:
        for p in cls_model.parameters():
            p.requires_grad_(False)
        count = parameter_count(cls_model, trainable_only=True)
        assert count == 0

    def test_inference_memory_returns_dict(self, cls_model: nn.Module) -> None:
        mem = inference_memory(cls_model, (10,))
        assert isinstance(mem, dict)
        assert "model_mb" in mem
        assert "input_mb" in mem
        assert "output_mb" in mem
        assert "total_mb" in mem

    def test_inference_memory_3d_shape(self, tiny_model: nn.Module) -> None:
        mem = inference_memory(tiny_model, (3, 8, 8))
        assert mem["total_mb"] >= 0


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


class TestUtils:
    def test_set_eval_mode(self, cls_model: nn.Module) -> None:
        cls_model.train()
        model = set_eval_mode(cls_model)
        assert not model.training

    def test_disable_gradients(self, cls_model: nn.Module) -> None:
        model = disable_gradients(cls_model)
        for p in model.parameters():
            assert not p.requires_grad

    def test_seed_everything(self) -> None:
        seed_everything(42)
        a = torch.randn(5)
        seed_everything(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_load_checkpoint_file_not_found(self, cls_model: nn.Module) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(cls_model, "/nonexistent/path.pt")

    def test_load_checkpoint_model_key(self, cls_model: nn.Module, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        torch.save({"model": cls_model.state_dict()}, path)
        new_model = DummyClassifier()
        state = load_checkpoint(new_model, path)
        assert "model" in state

    def test_load_checkpoint_state_dict_key(self, cls_model: nn.Module, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        torch.save({"state_dict": cls_model.state_dict()}, path)
        new_model = DummyClassifier()
        state = load_checkpoint(new_model, path)
        assert "state_dict" in state

    def test_load_checkpoint_raw_state_dict(self, cls_model: nn.Module, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        torch.save(cls_model.state_dict(), path)
        new_model = DummyClassifier()
        state = load_checkpoint(new_model, path)
        assert isinstance(state, dict)


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


class TestPostprocessing:
    def test_logits_to_probs_shape(self, cls_logits: torch.Tensor) -> None:
        probs = logits_to_probs(cls_logits)
        assert probs.shape == cls_logits.shape

    def test_logits_to_probs_sum_to_one(self, cls_logits: torch.Tensor) -> None:
        probs = logits_to_probs(cls_logits)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2))

    def test_logits_to_probs_temperature(self, cls_logits: torch.Tensor) -> None:
        probs_default = logits_to_probs(cls_logits)
        probs_hot = logits_to_probs(cls_logits, temperature=0.5)
        # Lower temperature -> sharper distribution
        max_default = probs_default.max(dim=1).values
        max_hot = probs_hot.max(dim=1).values
        assert (max_hot >= max_default).all()

    def test_logits_to_class_shape(self, cls_logits: torch.Tensor) -> None:
        preds = logits_to_class(cls_logits)
        assert preds.shape == (2,)

    def test_logits_to_class_values(self, cls_logits: torch.Tensor) -> None:
        preds = logits_to_class(cls_logits)
        assert preds[0] == 0  # argmax of [2.0, 1.0, 0.1]
        assert preds[1] == 1  # argmax of [0.5, 3.0, 0.2]

    def test_topk_predictions_shape(self, cls_logits: torch.Tensor) -> None:
        indices, probs = topk_predictions(cls_logits, k=2)
        assert indices.shape == (2, 2)
        assert probs.shape == (2, 2)

    def test_topk_predictions_k_larger_than_classes(self, cls_logits: torch.Tensor) -> None:
        indices, probs = topk_predictions(cls_logits, k=10)
        assert indices.shape == (2, 3)  # clamped to num_classes

    def test_logits_to_mask_argmax(self, seg_logits: torch.Tensor) -> None:
        mask = logits_to_mask(seg_logits)
        assert mask.shape == (1, 8, 8)
        assert mask.dtype == torch.long

    def test_logits_to_mask_binary_threshold(self) -> None:
        logits = torch.randn(1, 1, 8, 8)
        mask = logits_to_mask(logits, threshold=0.5)
        assert mask.shape == (1, 8, 8)
        assert mask.dtype == torch.long
        assert mask.unique().tolist() in ([0], [1], [0, 1])

    def test_logits_to_mask_multiclass_threshold(self, seg_logits: torch.Tensor) -> None:
        mask = logits_to_mask(seg_logits, threshold=0.3)
        assert mask.shape == (1, 8, 8)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class TestPredictor:
    def test_init_auto_device(self, cls_model: nn.Module) -> None:
        predictor = Predictor(cls_model, device="auto")
        assert isinstance(predictor, Predictor)
        assert not predictor.model.training

    def test_init_cpu(self, cls_model: nn.Module) -> None:
        predictor = Predictor(cls_model, device="cpu")
        assert next(predictor.model.parameters()).device.type == "cpu"

    def test_predict_single_returns_dict(self, cls_model: nn.Module) -> None:
        predictor = Predictor(cls_model, device="cpu")
        image = torch.randn(1, 10)
        result = predictor.predict_single(image)
        assert isinstance(result, dict)
        assert "logits" in result
        assert "probs" in result
        assert "prediction" in result

    def test_predict_single_numpy_input(self, tiny_model: nn.Module) -> None:
        predictor = Predictor(tiny_model, device="cpu")
        image = np.random.randn(3, 8, 8).astype(np.float32)
        result = predictor.predict_single(image)
        assert isinstance(result, dict)

    def test_predict_single_numpy_hwc(self) -> None:
        """Test with HWC numpy array (e.g. from image loading)."""
        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 3),
        )
        predictor = Predictor(model, device="cpu")
        image = np.random.randn(32, 32, 3).astype(np.float32)
        result = predictor.predict_single(image)
        assert isinstance(result, dict)

    def test_predict_batch_tensor(self, cls_model: nn.Module) -> None:
        predictor = Predictor(cls_model, device="cpu")
        images = torch.randn(4, 10)
        result = predictor.predict_batch(images)
        assert result["logits"].shape == (4, 3)

    def test_predict_batch_list(self, cls_model: nn.Module) -> None:
        predictor = Predictor(cls_model, device="cpu")
        images = [torch.randn(10) for _ in range(3)]
        result = predictor.predict_batch(images)
        assert result["logits"].shape == (3, 3)

    def test_predict_dataloader(self, cls_model: nn.Module) -> None:
        predictor = Predictor(cls_model, device="cpu")
        data = TensorDataset(torch.randn(6, 10))
        loader = DataLoader(data, batch_size=2)
        results = predictor.predict(loader)
        assert len(results) == 3  # 6 items / batch_size 2
        for r in results:
            assert "logits" in r
            assert "probs" in r
            assert "prediction" in r

    def test_predict_dataloader_dict_batch(self, cls_model: nn.Module) -> None:
        """Test DataLoader yielding dicts with 'image' key."""
        predictor = Predictor(cls_model, device="cpu")
        images = torch.randn(4, 10)
        loader = DataLoader(
            [({"image": images[i]}, i) for i in range(4)],
            batch_size=2,
            collate_fn=lambda batch: {
                "image": torch.stack([b[0]["image"] for b in batch]),
                "target": torch.tensor([b[1] for b in batch]),
            },
        )
        results = predictor.predict(loader)
        assert len(results) == 2

    def test_predict_single_segmentation(self, seg_model: nn.Module) -> None:
        predictor = Predictor(seg_model, device="cpu")
        image = torch.randn(3, 16, 16)
        result = predictor.predict_single(image)
        assert result["prediction"].shape == (1, 16, 16)

    def test_predict_with_custom_postprocess(self, cls_model: nn.Module) -> None:
        def custom_postprocess(logits: torch.Tensor) -> dict:
            probs = torch.softmax(logits, dim=1)
            return {"logits": logits, "probs": probs, "custom_key": True}

        predictor = Predictor(cls_model, device="cpu", postprocess_fn=custom_postprocess)
        result = predictor.predict_single(torch.randn(1, 10))
        assert result["custom_key"] is True


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_torchscript_trace(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        path = tmp_path / "traced.pt"
        example = torch.randn(1, 3, 8, 8)
        result = export_torchscript(tiny_model, path, example, method="trace", verify=True)
        assert result.exists()
        assert result.suffix == ".pt"

    def test_export_torchscript_script(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        path = tmp_path / "scripted.pt"
        example = torch.randn(1, 3, 8, 8)
        result = export_torchscript(tiny_model, path, example, method="script", verify=True)
        assert result.exists()

    def test_export_torchscript_invalid_method(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        example = torch.randn(1, 3, 8, 8)
        with pytest.raises(ValueError, match="Unknown export method"):
            export_torchscript(tiny_model, path, example, method="invalid")

    def test_export_torchscript_verify_loaded_works(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        path = tmp_path / "verify.pt"
        example = torch.randn(1, 3, 8, 8)
        export_torchscript(tiny_model, path, example, method="trace", verify=True)
        loaded = torch.jit.load(str(path))
        with torch.no_grad():
            out = loaded(example)
        assert out.shape == (1, 5)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available for device test"
    )
    def test_export_torchscript_with_device(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        path = tmp_path / "device.pt"
        example = torch.randn(1, 3, 8, 8, device="cuda")
        result = export_torchscript(
            tiny_model, path, example, method="trace", device="cuda", verify=False
        )
        assert result.exists()

    def test_export_onnx_import_error(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        """Test that ONNX export raises ImportError when onnx is not installed."""
        path = tmp_path / "model.onnx"
        example = torch.randn(1, 3, 8, 8)
        try:
            import onnx  # noqa: F401
            pytest.skip("onnx is installed, skipping import error test")
        except ImportError:
            with pytest.raises(ImportError, match="onnx"):
                export_onnx(tiny_model, path, example)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_export_torchscript_cuda(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        path = tmp_path / "cuda.pt"
        example = torch.randn(1, 3, 8, 8, device="cuda")
        result = export_torchscript(
            tiny_model.to("cuda"), path, example, method="trace", verify=False
        )
        assert result.exists()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_benchmark_model_returns_dict(self, tiny_model: nn.Module) -> None:
        stats = benchmark_model(tiny_model, (1, 3, 8, 8), device="cpu", warmup=2, num_iterations=5)
        assert isinstance(stats, dict)
        assert stats["device"] == "cpu"
        assert stats["input_shape"] == (1, 3, 8, 8)
        assert stats["warmup"] == 2
        assert stats["num_iterations"] == 5
        assert stats["batch_size"] == 1

    def test_benchmark_model_latency_keys(self, tiny_model: nn.Module) -> None:
        stats = benchmark_model(tiny_model, (1, 3, 8, 8), device="cpu", warmup=2, num_iterations=5)
        assert "latency_mean_ms" in stats
        assert "latency_std_ms" in stats
        assert "latency_min_ms" in stats
        assert "latency_max_ms" in stats
        assert stats["latency_mean_ms"] > 0

    def test_benchmark_model_throughput(self, tiny_model: nn.Module) -> None:
        stats = benchmark_model(tiny_model, (1, 3, 8, 8), device="cpu", warmup=2, num_iterations=5)
        assert stats["throughput_items_per_sec"] > 0
        assert stats["fps"] > 0

    def test_benchmark_model_3d_input(self, tiny_model: nn.Module) -> None:
        stats = benchmark_model(tiny_model, (3, 8, 8), device="cpu", warmup=2, num_iterations=5)
        assert stats["input_shape"] == (1, 3, 8, 8)

    def test_benchmark_model_batch_size(self, tiny_model: nn.Module) -> None:
        stats = benchmark_model(
            tiny_model, (4, 3, 8, 8), device="cpu", warmup=2, num_iterations=5
        )
        assert stats["batch_size"] == 4


# ---------------------------------------------------------------------------
# Visualization (smoke tests with Agg backend)
# ---------------------------------------------------------------------------


class TestVisualization:
    @pytest.fixture(autouse=True)
    def _use_agg_backend(self) -> None:
        import matplotlib
        matplotlib.use("Agg")

    def test_plot_classification_result(self) -> None:
        image = torch.randn(3, 32, 32)
        logits = torch.tensor([2.0, 1.0, 0.5, 0.1, 0.05])
        fig, (ax_img, ax_bar) = plt.subplots(1, 2)
        plot_classification_result(image, logits, top_k=3, ax=ax_img)
        plt.close(fig)

    def test_plot_classification_result_with_names(self) -> None:
        image = torch.randn(3, 32, 32)
        logits = torch.tensor([2.0, 1.0, 0.5])
        names = ["cat", "dog", "bird"]
        fig, (ax_img, ax_bar) = plt.subplots(1, 2)
        plot_classification_result(image, logits, class_names=names, top_k=3, ax=ax_img)
        plt.close(fig)

    def test_plot_classification_result_no_ax(self) -> None:
        image = torch.randn(3, 32, 32)
        logits = torch.tensor([2.0, 1.0, 0.5])
        plot_classification_result(image, logits, top_k=3)
        plt.close("all")

    def test_plot_segmentation_result(self) -> None:
        image = torch.randn(3, 16, 16)
        logits = torch.randn(3, 16, 16)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plot_segmentation_result(image, logits, ax=ax1)
        plt.close(fig)

    def test_plot_segmentation_result_no_ax(self) -> None:
        image = torch.randn(3, 16, 16)
        logits = torch.randn(3, 16, 16)
        plot_segmentation_result(image, logits)
        plt.close("all")

    def test_plot_segmentation_comparison(self) -> None:
        image = torch.randn(3, 16, 16)
        logits = torch.randn(3, 16, 16)
        ground_truth = torch.randint(0, 3, (16, 16))
        plot_segmentation_comparison(image, logits, ground_truth)
        plt.close("all")

    def test_plot_segmentation_comparison_numpy_gt(self) -> None:
        image = torch.randn(3, 16, 16)
        logits = torch.randn(3, 16, 16)
        ground_truth = np.random.randint(0, 3, (16, 16))
        plot_segmentation_comparison(image, logits, ground_truth)
        plt.close("all")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_predictor_with_benchmark(self, tiny_model: nn.Module) -> None:
        """Verify Predictor and benchmark work together."""
        predictor = Predictor(tiny_model, device="cpu")
        image = torch.randn(3, 8, 8)
        result = predictor.predict_single(image)
        assert result["logits"].shape == (1, 5)

        stats = benchmark_model(tiny_model, (1, 3, 8, 8), device="cpu", warmup=2, num_iterations=5)
        assert stats["latency_mean_ms"] > 0

    def test_export_then_predict(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        """Export to TorchScript, load, and run prediction."""
        path = tmp_path / "integ.pt"
        example = torch.randn(1, 3, 8, 8)
        export_torchscript(tiny_model, path, example, method="trace", verify=True)

        loaded = torch.jit.load(str(path))
        with torch.no_grad():
            out = loaded(example)
        assert out.shape == (1, 5)

    def test_full_classification_pipeline(self, cls_model: nn.Module) -> None:
        """End-to-end: predict -> postprocess -> metrics."""
        predictor = Predictor(cls_model, device="cpu")
        image = torch.randn(1, 10)
        result = predictor.predict_single(image)

        logits = result["logits"]
        probs = logits_to_probs(logits)
        pred = logits_to_class(logits)

        assert torch.allclose(probs.sum(dim=1), torch.ones(1))
        assert pred.numel() == 1

    def test_full_segmentation_pipeline(self, seg_model: nn.Module) -> None:
        """End-to-end: predict -> mask -> visualization."""
        predictor = Predictor(seg_model, device="cpu")
        image = torch.randn(3, 16, 16)
        result = predictor.predict_single(image)

        mask = logits_to_mask(result["logits"])
        assert mask.shape == (1, 16, 16)

        # Smoke test visualization
        import matplotlib
        matplotlib.use("Agg")
        plot_segmentation_result(image, result["logits"].squeeze(0))
        plt.close("all")

