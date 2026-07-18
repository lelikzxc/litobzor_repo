"""Inference benchmarking utilities.

Provides functions to measure model latency, throughput, and FPS
on a given device with configurable warmup and iteration counts.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from common.inference.device import get_best_device, move_to_device


def benchmark_model(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device | str | None = None,
    warmup: int = 10,
    num_iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark model inference latency, throughput, and FPS.

    Args:
        model: The model to benchmark.
        input_shape: Input tensor shape ``(B, C, H, W)``. If 3D ``(C, H, W)``,
            batch size 1 is assumed.
        device: Device to run on. If ``None``, uses ``get_best_device()``.
        warmup: Number of warmup iterations (not measured).
        num_iterations: Number of measured iterations.

    Returns:
        Dict with keys:
            - ``"device"``: Device string used.
            - ``"input_shape"``: Actual input shape used.
            - ``"warmup"``: Number of warmup iterations.
            - ``"num_iterations"``: Number of measured iterations.
            - ``"latency_mean_ms"``: Mean latency per iteration in ms.
            - ``"latency_std_ms"``: Standard deviation of latency in ms.
            - ``"latency_min_ms"``: Minimum latency in ms.
            - ``"latency_max_ms"``: Maximum latency in ms.
            - ``"throughput_items_per_sec"``: Items processed per second.
            - ``"fps"``: Frames (batches) per second.
            - ``"batch_size"``: Batch size used.
    """
    if device is None:
        device = get_best_device()
    device = torch.device(device) if isinstance(device, str) else device

    model = move_to_device(model, device)
    model.eval()

    # Determine batch size and input shape
    if len(input_shape) == 3:
        batch_shape = (1, *input_shape)
    else:
        batch_shape = input_shape

    batch_size = batch_shape[0]

    # Create synthetic input
    dummy_input = torch.randn(batch_shape, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)

    # Synchronize before timing (important for CUDA)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measured iterations
    latencies: list[float] = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    # Compute statistics
    latencies_t = torch.tensor(latencies)
    mean_ms = float(latencies_t.mean().item())
    std_ms = float(latencies_t.std().item())
    min_ms = float(latencies_t.min().item())
    max_ms = float(latencies_t.max().item())

    # Throughput: items per second
    mean_sec = mean_ms / 1000.0
    throughput = batch_size / mean_sec if mean_sec > 0 else 0.0
    fps = 1.0 / mean_sec if mean_sec > 0 else 0.0

    return {
        "device": str(device),
        "input_shape": batch_shape,
        "warmup": warmup,
        "num_iterations": num_iterations,
        "latency_mean_ms": round(mean_ms, 4),
        "latency_std_ms": round(std_ms, 4),
        "latency_min_ms": round(min_ms, 4),
        "latency_max_ms": round(max_ms, 4),
        "throughput_items_per_sec": round(throughput, 2),
        "fps": round(fps, 2),
        "batch_size": batch_size,
    }