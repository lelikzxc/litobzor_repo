"""Vendored selective scan implementation from official VMamba repo.

Source: https://github.com/MzeroMiko/VMamba (classification/models/csms6s.py)

Provides ``selective_scan_fn`` with automatic fallback:
    1. CUDA kernels (selective_scan_cuda_oflex / selective_scan_cuda_core / selective_scan_cuda)
    2. Pure-PyTorch fallback (``selective_scan_torch``) when CUDA is unavailable

The pure-PyTorch fallback walks the sequence step-by-step and never materialises
full ``[B, D, L, N]`` tensors (those OOMed 8GB GPUs). Prefer installing
``selective_scan_cuda_oflex`` for real training speed.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

# ── CUDA kernel availability ──────────────────────────────────────────────

WITH_SELECTIVESCAN_OFLEX = True
WITH_SELECTIVESCAN_CORE = False
WITH_SELECTIVESCAN_MAMBA = True

try:
    import selective_scan_cuda_oflex  # type: ignore[import-untyped]
except ImportError:
    WITH_SELECTIVESCAN_OFLEX = False
    warnings.warn("selective_scan_cuda_oflex not available — using pure-PyTorch fallback")

try:
    import selective_scan_cuda_core  # type: ignore[import-untyped]
except ImportError:
    WITH_SELECTIVESCAN_CORE = False

try:
    import selective_scan_cuda  # type: ignore[import-untyped]
except ImportError:
    WITH_SELECTIVESCAN_MAMBA = False


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


# ── CUDA autograd wrapper ─────────────────────────────────────────────────


class SelectiveScanCuda(torch.autograd.Function):
    """Autograd wrapper around CUDA selective scan kernels."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
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
        if backend is None:
            if WITH_SELECTIVESCAN_OFLEX:
                backend = "oflex"
            elif WITH_SELECTIVESCAN_CORE:
                backend = "core"
            elif WITH_SELECTIVESCAN_MAMBA:
                backend = "mamba"
            else:
                backend = "torch"
        # If a specific backend was requested but not installed, fall back.
        if backend == "oflex" and not WITH_SELECTIVESCAN_OFLEX:
            backend = "core" if WITH_SELECTIVESCAN_CORE else ("mamba" if WITH_SELECTIVESCAN_MAMBA else "torch")
        if backend == "core" and not WITH_SELECTIVESCAN_CORE:
            backend = "oflex" if WITH_SELECTIVESCAN_OFLEX else ("mamba" if WITH_SELECTIVESCAN_MAMBA else "torch")
        if backend == "mamba" and not WITH_SELECTIVESCAN_MAMBA:
            backend = "oflex" if WITH_SELECTIVESCAN_OFLEX else ("core" if WITH_SELECTIVESCAN_CORE else "torch")
        ctx.backend = backend

        if backend == "torch":
            return selective_scan_torch(
                u, delta, A, B, C, D, delta_bias, delta_softplus, oflex
            )
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
    @torch.cuda.amp.custom_bwd
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
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus, False
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


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
        backend: Force specific backend (``"torch"``, ``"oflex"``, ``"core"``, ``"mamba"``).

    Returns:
        Scanned output ``[B, K*C, L]``.
    """
    has_cuda = WITH_SELECTIVESCAN_OFLEX or WITH_SELECTIVESCAN_CORE or WITH_SELECTIVESCAN_MAMBA
    fn = selective_scan_torch if backend == "torch" or (not has_cuda) else SelectiveScanCuda.apply  # type: ignore[assignment]
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)