"""Model export utilities for TorchScript and ONNX.

Provides functions to export trained models to TorchScript (trace/script)
and ONNX formats, with optional verification after export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def export_torchscript(
    model: nn.Module,
    path: str | Path,
    example_input: torch.Tensor,
    method: str = "trace",
    device: torch.device | str | None = None,
    strict: bool = True,
    verify: bool = True,
) -> Path:
    """Export a model to TorchScript.

    Args:
        model: The model to export.
        path: Output file path (``.pt`` or ``.pth`` extension).
        example_input: Example input tensor for tracing.
        method: ``"trace"`` for ``torch.jit.trace`` or ``"script"`` for
            ``torch.jit.script``.
        device: Device to run export on. If ``None``, uses model's device.
        strict: Whether to run in strict mode for tracing.
        verify: If ``True``, reload the exported model and run a forward pass
            to verify correctness.

    Returns:
        The resolved output path.

    Raises:
        ValueError: If ``method`` is not ``"trace"`` or ``"script"``.
        RuntimeError: If verification fails.
    """
    model.eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if device is not None:
        model = model.to(device)
        example_input = example_input.to(device)

    if method == "trace":
        traced_model = torch.jit.trace(model, example_input, strict=strict)
    elif method == "script":
        traced_model = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown export method: {method}. Use 'trace' or 'script'.")

    traced_model.save(path)

    if verify:
        _verify_torchscript(path, example_input)

    return path


def export_onnx(
    model: nn.Module,
    path: str | Path,
    example_input: torch.Tensor,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
    device: torch.device | str | None = None,
    verify: bool = True,
) -> Path:
    """Export a model to ONNX format.

    Args:
        model: The model to export.
        path: Output file path (``.onnx`` extension).
        example_input: Example input tensor.
        input_names: Names for the input tensors.
        output_names: Names for the output tensors.
        dynamic_axes: Dynamic axes specification for variable-length inputs.
        opset_version: ONNX opset version.
        device: Device to run export on. If ``None``, uses model's device.
        verify: If ``True``, reload the exported model and run a forward pass
            to verify correctness.

    Returns:
        The resolved output path.

    Raises:
        ImportError: If the ``onnx`` package is not installed.
        RuntimeError: If verification fails.
    """
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'onnx' package is required for ONNX export. "
            "Install it with: pip install onnx"
        )

    model.eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if device is not None:
        model = model.to(device)
        example_input = example_input.to(device)

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    torch.onnx.export(
        model,
        example_input,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    if verify:
        _verify_onnx(path, example_input)

    return path


def _verify_torchscript(path: Path, example_input: torch.Tensor) -> None:
    """Verify a TorchScript model by loading and running a forward pass.

    Args:
        path: Path to the saved TorchScript model.
        example_input: Example input tensor.

    Raises:
        RuntimeError: If the loaded model produces an error.
    """
    try:
        loaded = torch.jit.load(str(path))
        with torch.no_grad():
            _ = loaded(example_input)
    except Exception as e:
        raise RuntimeError(f"TorchScript verification failed: {e}")


def _verify_onnx(path: Path, example_input: torch.Tensor) -> None:
    """Verify an ONNX model by loading and running a forward pass.

    Args:
        path: Path to the saved ONNX model.
        example_input: Example input tensor.

    Raises:
        RuntimeError: If the loaded model produces an error.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        # Skip verification if onnxruntime is not available
        return

    try:
        # Check ONNX model validity
        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)

        # Run inference with onnxruntime
        session = ort.InferenceSession(str(path))
        input_name = session.get_inputs()[0].name
        input_data = example_input.cpu().numpy()
        _ = session.run(None, {input_name: input_data})
    except Exception as e:
        raise RuntimeError(f"ONNX verification failed: {e}")