"""Vendored selective scan implementation from official VMamba repo.

Source: https://github.com/MzeroMiko/VMamba (classification/models/csms6s.py)

Provides ``selective_scan_fn`` with automatic fallback:
    1. CUDA kernels (selective_scan_cuda_oflex / selective_scan_cuda_core / selective_scan_cuda)
    2. Pure-PyTorch fallback (``selective_scan_torch``) when CUDA is unavailable

The pure-PyTorch fallback uses a Python for-loop over the sequence length,
which is significantly slower but allows the model to run on CPU or without
compiled CUDA extensions.
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
    """Pure-PyTorch selective scan (for-loop over sequence length).

    Args:
        u: Input tensor of shape ``[B, K*C, L]``.
        delta: Discretisation step of shape ``[B, K*C, L]``.
        A: State transition matrix of shape ``[K*C, N]``.
        B: Input projection of shape ``[B, K, N, L]``.
        C: Output projection of shape ``[B, K, N, L]``.
        D: Skip connection of shape ``[K*C]`` (optional).
        delta_bias: Bias for delta of shape ``[K*C]`` (optional).
        delta_softplus: Apply ``softplus`` to delta.
        oflex: Keep output in fp32.

    Returns:
        Scanned output of shape ``[B, K*C, L]``.
    """
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = KCdim // K

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)

    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)

    x = A.new_zeros((Batch, KCdim, N))
    ys: list[torch.Tensor] = []
    for i in range(L):
        x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
        y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (B, C, L)

    out = y if D is None else y + u * D.unsqueeze(-1)
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
        backend = "oflex" if WITH_SELECTIVESCAN_OFLEX and (backend is None) else backend
        backend = "core" if WITH_SELECTIVESCAN_CORE and (backend is None) else backend
        backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend
        ctx.backend = backend

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