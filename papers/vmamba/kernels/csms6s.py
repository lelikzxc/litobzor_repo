"""Vendored selective scan implementation from official VMamba repo.

Source: https://github.com/MzeroMiko/VMamba (classification/models/csms6s.py)

Provides ``selective_scan_fn`` with automatic fallback:
    1. CUDA kernels (selective_scan_cuda_oflex / selective_scan_cuda_core / selective_scan_cuda)
    2. Checkpointed PyTorch prefix scan on CUDA / MPS without native extensions
    3. Pure-PyTorch recurrence on CPU (also available as an explicit reference)

The CPU reference walks the sequence step-by-step. CUDA fallback and MPS use bounded, checkpointed
affine-prefix blocks instead. Neither retains full ``[B, D, L, N]`` histories
(those OOMed 8GB GPUs). Prefer ``selective_scan_cuda_oflex`` for CUDA training.
"""

from __future__ import annotations

import importlib
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ── CUDA kernel availability ──────────────────────────────────────────────

SCAN_IMPORT_ERRORS: dict[str, str] = {}


def _load_scan_extension(name: str):
    """A missing or incompatible Windows DLL must not break the portable scan."""
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        SCAN_IMPORT_ERRORS[name] = f"{type(error).__name__}: {error}"
        return None


selective_scan_cuda_oflex = _load_scan_extension("selective_scan_cuda_oflex")
selective_scan_cuda_core = _load_scan_extension("selective_scan_cuda_core")
selective_scan_cuda = _load_scan_extension("selective_scan_cuda")
WITH_SELECTIVESCAN_OFLEX = selective_scan_cuda_oflex is not None
WITH_SELECTIVESCAN_CORE = selective_scan_cuda_core is not None
WITH_SELECTIVESCAN_MAMBA = selective_scan_cuda is not None
_SCAN_BACKEND = "auto"


def set_scan_backend(backend: str = "auto") -> None:
    """Select the scan execution method without changing the model recurrence.

    ``compiled`` requires an installed CUDA extension; ``chunked`` is the native
    Windows-compatible PyTorch prefix scan. ``torch`` is the slow reference.
    """
    if backend not in {"auto", "chunked", "compiled", "torch"}:
        raise ValueError(f"Unknown scan backend: {backend}")
    global _SCAN_BACKEND
    _SCAN_BACKEND = backend


def _compiled_backends() -> list[str]:
    return [
        name
        for name, available in (
            ("oflex", WITH_SELECTIVESCAN_OFLEX),
            ("core", WITH_SELECTIVESCAN_CORE),
            ("mamba", WITH_SELECTIVESCAN_MAMBA),
        )
        if available
    ]


def get_scan_backend(device: str | torch.device, backend: str | None = None) -> str:
    """Resolve the actual execution backend, including SS2D's preferred kernel."""
    if backend not in {None, "auto", "chunked", "compiled", "torch", "oflex", "core", "mamba"}:
        raise ValueError(f"Unknown scan backend: {backend}")
    device_type = torch.device(device).type
    # Explicit reference calls remain available for numerical comparisons even
    # when the training entry point has selected a compiled/chunked method.
    if backend in {"torch", "chunked"}:
        return backend
    requested = _SCAN_BACKEND if _SCAN_BACKEND != "auto" else backend
    if requested in {"torch", "chunked"}:
        return requested
    if device_type != "cuda":
        if requested == "compiled":
            raise RuntimeError("Compiled selective scan requires a CUDA device")
        return "chunked" if device_type == "mps" else "torch"
    available = _compiled_backends()
    if requested in available:
        return requested
    if available:
        return available[0]
    if requested == "compiled":
        raise RuntimeError(
            "No compiled selective-scan extension is available. Use --scan-backend "
            "chunked for native Windows CUDA, or install an extension built for "
            "this Python/PyTorch/CUDA environment."
        )
    return "chunked"


def scan_backend_info(
    device: str | torch.device, backend: str | None = None
) -> dict[str, object]:
    """Return truthful startup diagnostics without importing optional packages."""
    return {
        "backend": get_scan_backend(device, backend),
        "compiled_backends": _compiled_backends(),
        "import_errors": dict(SCAN_IMPORT_ERRORS),
    }


# PyTorch 2.1 (the paper environment) uses the old location; newer versions
# require device_type explicitly and otherwise warn on every module import.
if hasattr(torch.amp, "custom_fwd"):
    _custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
    _custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
else:
    _custom_fwd = torch.cuda.amp.custom_fwd
    _custom_bwd = torch.cuda.amp.custom_bwd


# ── Pure-PyTorch fallback ─────────────────────────────────────────────────


def selective_scan_torch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
    oflex: bool = True,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Memory-safe pure-PyTorch selective scan (no full ``[B,D,L,N]`` tensors).

    Keeps only the recurrent state ``[B, K*C, N]`` and walks ``L`` step-by-step.
    Slower than CUDA kernels, but will not OOM an 8GB laptop GPU the way a
    parallel scan over ``L≈3k`` would.
    """
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = KCdim // K

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)

    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()

    # [B,K,Cdim,L] / [K,Cdim,N] — broadcast, never expand over L×N at once
    u_ = u.view(Batch, K, Cdim, L)
    delta_ = delta.view(Batch, K, Cdim, L)
    A_ = A.view(K, Cdim, N)

    x = u.new_zeros(Batch, K, Cdim, N)
    ys: list[torch.Tensor] = []

    for i in range(L):
        di = delta_[:, :, :, i]  # [B,K,Cdim]
        ui = u_[:, :, :, i]
        Bi = B[:, :, :, i]  # [B,K,N]
        Ci = C[:, :, :, i]

        dA = torch.exp(di.unsqueeze(-1) * A_)  # [B,K,Cdim,N]
        dBu = di.unsqueeze(-1) * Bi.unsqueeze(2) * ui.unsqueeze(-1)
        x = dA * x + dBu
        ys.append((x * Ci.unsqueeze(2)).sum(-1).reshape(Batch, KCdim))

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.float().unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


def _scan_chunk(delta, u, A, B, C, state):
    """Stable affine prefix scan: (a, b) compose as (a2*a1, b2+a2*b1)."""
    a = torch.exp(delta.unsqueeze(-1) * A)
    b = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    offset = 1
    while offset < u.shape[-1]:
        right_a = a[..., offset:, :]
        b = torch.cat(
            (b[..., :offset, :], b[..., offset:, :] + right_a * b[..., :-offset, :]),
            dim=-2,
        )
        a = torch.cat((a[..., :offset, :], right_a * a[..., :-offset, :]), dim=-2)
        offset *= 2
    states = b + a * state.unsqueeze(-2)
    values = (states * C.unsqueeze(2)).sum(-1)
    # A view of the final token would keep the entire block's state allocation
    # alive across checkpoints. Save an independent, small boundary state.
    return states[..., -1, :].clone(), values


def selective_scan_chunked(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
    oflex: bool = True,
    *args,
    chunk_size: int = 128,
    **kwargs,
) -> torch.Tensor:
    """Exact recurrence with blockwise parallel prefixes for CUDA / MPS.

    Multiplication/addition use the same affine recurrence as the reference,
    with a different floating-point association. Unlike division by cumulative
    products, this stays finite even when transition products underflow to 0.
    Checkpoint each block during training: the full [B,D,L,N] state history is
    never retained; backward recomputes one block at a time. No compiler, Triton
    or platform-specific extension is required.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    dtype_in = u.dtype
    batch, groups, state_dim, length = B.shape
    channels = u.shape[1] // groups
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    u_float = u.float()
    us = u_float.reshape(batch, groups, channels, length)
    ds = delta.float().reshape(batch, groups, channels, length)
    As = A.float().reshape(groups, channels, 1, state_dim)
    Bs = B.float().permute(0, 1, 3, 2)
    Cs = C.float().permute(0, 1, 3, 2)
    state = u_float.new_zeros(batch, groups, channels, state_dim)
    outputs = []
    use_checkpoint = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (u, delta, A, B, C)
    )
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        inputs = (
            ds[..., start:end],
            us[..., start:end],
            As,
            Bs[..., start:end, :],
            Cs[..., start:end, :],
            state,
        )
        if use_checkpoint:
            state, values = checkpoint(_scan_chunk, *inputs, use_reentrant=False)
        else:
            state, values = _scan_chunk(*inputs)
        outputs.append(values.reshape(batch, groups * channels, end - start))
    y = torch.cat(outputs, dim=-1)
    out = y if D is None else y + u_float * D.float().unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


# ── CUDA autograd wrapper ─────────────────────────────────────────────────


class SelectiveScanCuda(torch.autograd.Function):
    """Autograd wrapper around CUDA selective scan kernels."""

    @staticmethod
    @_custom_fwd
    def forward(
        ctx,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        delta_bias: torch.Tensor | None = None,
        delta_softplus: bool = False,
        oflex: bool = True,
        backend: str | None = None,
    ) -> torch.Tensor:
        ctx.delta_softplus = delta_softplus
        backend = get_scan_backend(u.device, backend)
        if backend not in {"oflex", "core", "mamba"} or not u.is_cuda:
            # Calling a differentiable PyTorch scan inside Function.forward
            # would discard its graph. Portable dispatch belongs outside apply.
            raise RuntimeError("SelectiveScanCuda requires an available compiled CUDA backend")
        ctx.backend = backend
        u = u.contiguous() if u.stride(-1) != 1 else u
        delta = delta.contiguous() if delta.stride(-1) != 1 else delta
        B = B.contiguous() if B.stride(-1) != 1 else B
        C = C.contiguous() if C.stride(-1) != 1 else C
        D = D.contiguous() if D is not None else None
        delta_bias = delta_bias.contiguous() if delta_bias is not None else None
        if backend == "oflex":
            out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)  # type: ignore[attr-defined]
        elif backend == "core":
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)  # type: ignore[attr-defined]
        elif backend == "mamba":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unknown backend: {backend}")

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @_custom_bwd
    def backward(ctx, dout: torch.Tensor, *args) -> tuple:
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        backend = ctx.backend

        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if backend == "oflex":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(  # type: ignore[attr-defined]
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "core":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(  # type: ignore[attr-defined]
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "mamba":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(  # type: ignore[attr-defined]
                u,
                delta,
                A,
                B,
                C,
                D,
                None,
                delta_bias,
                dout,
                x,
                None,
                None,
                ctx.delta_softplus,
                False,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        return (
            du, ddelta, dA, dB, dC,
            dD if D is not None else None,
            ddelta_bias if delta_bias is not None else None,
            None, None, None,
        )


# ── Public API ────────────────────────────────────────────────────────────


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
    oflex: bool = True,
    backend: str | None = None,
) -> torch.Tensor:
    """Selective scan with automatic CUDA / PyTorch fallback.

    Args:
        u: Input tensor ``[B, K*C, L]``.
        delta: Discretisation step ``[B, K*C, L]``.
        A: State transition matrix ``[K*C, N]``.
        B: Input projection ``[B, K, N, L]``.
        C: Output projection ``[B, K, N, L]``.
        D: Skip connection ``[K*C]`` (optional).
        delta_bias: Bias ``[K*C]`` (optional).
        delta_softplus: Apply ``softplus`` to delta.
        oflex: Keep output in fp32.
        backend: Preferred compiled kernel, or explicit ``"torch"`` / ``"chunked"``.

    Returns:
        Scanned output ``[B, K*C, L]``.
    """
    backend = get_scan_backend(u.device, backend)
    if backend == "chunked":
        return selective_scan_chunked(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex)
    fn = selective_scan_torch if backend == "torch" else SelectiveScanCuda.apply  # type: ignore[assignment]
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)
