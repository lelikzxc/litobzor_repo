"""Reusable inference infrastructure for the litobzor research repository.

Provides generic, paper-agnostic components for model inference:
- Predictor: generic inference with single image, batch, or DataLoader
- Postprocessing: logits→probs, logits→class, top-k, segmentation masks
- Visualization: classification and segmentation result display
- Export: TorchScript and ONNX model export with verification
- Benchmark: latency, throughput, FPS measurement
- Device: best device selection, model size, parameter count
- Utils: checkpoint loading, eval mode, gradient disabling, seeding
"""

from __future__ import annotations

from common.inference.benchmark import benchmark_model
from common.inference.device import (
    get_best_device,
    inference_memory,
    model_size_mb,
    move_to_device,
    parameter_count,
)
from common.inference.export import export_onnx, export_torchscript
from common.inference.postprocessing import (
    logits_to_class,
    logits_to_mask,
    logits_to_probs,
    topk_predictions,
)
from common.inference.predictor import Predictor
from common.inference.utils import (
    disable_gradients,
    load_checkpoint,
    seed_everything,
    set_eval_mode,
)
from common.inference.visualization import (
    plot_classification_result,
    plot_segmentation_comparison,
    plot_segmentation_result,
)

__all__ = [
    # Predictor
    "Predictor",
    # Postprocessing
    "logits_to_probs",
    "logits_to_class",
    "topk_predictions",
    "logits_to_mask",
    # Visualization
    "plot_classification_result",
    "plot_segmentation_result",
    "plot_segmentation_comparison",
    # Export
    "export_torchscript",
    "export_onnx",
    # Benchmark
    "benchmark_model",
    # Device
    "get_best_device",
    "move_to_device",
    "model_size_mb",
    "parameter_count",
    "inference_memory",
    # Utils
    "load_checkpoint",
    "set_eval_mode",
    "disable_gradients",
    "seed_everything",
]
