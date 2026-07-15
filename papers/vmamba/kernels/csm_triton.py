"""Vendored cross-scan / cross-merge implementation from official VMamba repo.

Source: https://github.com/MzeroMiko/VMamba (classification/models/csm_triton.py)

Provides ``cross_scan_fn`` and ``cross_merge_fn`` with automatic fallback:
    1. Triton kernels (when ``triton`` is installed and input is on CUDA)
    2. Pure-PyTorch fallback (``CrossScanF`` / ``CrossMergeF``)

The pure-PyTorch fallback uses ``torch.flip``, ``torch.stack``, and
``torch.transpose`` to implement the 4-directional scanning without
compiled Triton kernels.
"""

from __future__ import annotations

import warnings

import torch

# ── Triton availability ───────────────────────────────────────────────────

WITH_TRITON = True
try:
    import triton  # type: ignore[import-untyped]  # noqa: F401
    import triton.language as tl  # type: ignore[import-untyped]  # noqa: F401
except ImportError:
    WITH_TRITON = False
    warnings.warn("triton not installed — using pure-PyTorch cross-scan/merge")


# ── Pure-PyTorch forward helpers ──────────────────────────────────────────


def cross_scan_fwd(
    x: torch.Tensor,
    in_channel_first: bool = True,
    out_channel_first: bool = True,
    scans: int = 0,
) -> torch.Tensor:
    """Forward cross-scan (4-directional) using pure PyTorch.

    Args:
        x: Input ``[B, C, H, W]`` (channel-first) or ``[B, H, W, C]``.
        in_channel_first: Whether input is channel-first.
        out_channel_first: Whether output is channel-first.
        scans: Scan mode (0=cross, 1=unidirectional, 2=bidirectional).

    Returns:
        Scanned tensor ``[B, 4, C, H*W]`` or equivalent.
    """
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
        else:
            raise ValueError(f"Unknown scan mode: {scans}")
    else:
        B, H, W, C = x.shape
        if scans == 0:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = x.transpose(dim0=1, dim1=2).flatten(1, 2)
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
        elif scans == 1:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 4, 1)
        elif scans == 2:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 2, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)
        else:
            raise ValueError(f"Unknown scan mode: {scans}")

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_fwd(
    y: torch.Tensor,
    in_channel_first: bool = True,
    out_channel_first: bool = True,
    scans: int = 0,
) -> torch.Tensor:
    """Forward cross-merge (fuse 4 directions) using pure PyTorch.

    Args:
        y: Input ``[B, 4, C, H, W]`` (channel-first) or ``[B, H, W, 4, C]``.
        in_channel_first: Whether input is channel-first.
        out_channel_first: Whether output is channel-first.
        scans: Scan mode (0=cross, 1=unidirectional, 2=bidirectional).

    Returns:
        Merged tensor ``[B, C, H*W]`` or equivalent.
    """
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y.sum(1)
        else:
            raise ValueError(f"Unknown scan mode: {scans}")
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y[:, :, 0] + y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).contiguous().view(B, -1, D)
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y.sum(2)
        else:
            raise ValueError(f"Unknown scan mode: {scans}")

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()

    return y


# ── Pure-PyTorch autograd Functions ───────────────────────────────────────


class CrossScanF(torch.autograd.Function):
    """Pure-PyTorch cross-scan with autograd."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        in_channel_first: bool = True,
        out_channel_first: bool = True,
        one_by_one: bool = False,
        scans: int = 0,
    ) -> torch.Tensor:
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            B, K, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, K, C = x.shape
        else:
            B, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)

        y = cross_scan_fwd(x, in_channel_first, out_channel_first, scans)
        return y

    @staticmethod
    def backward(ctx, ys: torch.Tensor) -> tuple:
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        scans = ctx.scans
        B, C, H, W = ctx.shape

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        y = cross_merge_fwd(ys, in_channel_first, out_channel_first, scans)

        y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)
        return y, None, None, None, None


class CrossMergeF(torch.autograd.Function):
    """Pure-PyTorch cross-merge with autograd."""

    @staticmethod
    def forward(
        ctx,
        ys: torch.Tensor,
        in_channel_first: bool = True,
        out_channel_first: bool = True,
        one_by_one: bool = False,
        scans: int = 0,
    ) -> torch.Tensor:
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        B, K, C, H, W = ys.shape
        if not out_channel_first:
            B, H, W, K, C = ys.shape
        ctx.shape = (B, C, H, W)

        y = cross_merge_fwd(ys, in_channel_first, out_channel_first, scans)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor) -> tuple:
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        scans = ctx.scans
        B, C, H, W = ctx.shape

        if in_channel_first:
            x = x.view(B, C, H, W)
        else:
            x = x.view(B, H, W, C)

        x = cross_scan_fwd(x, in_channel_first, out_channel_first, scans)
        x = x.view(B, 4, C, H, W) if out_channel_first else x.view(B, H, W, 4, C)
        return x, None, None, None, None


# ── Triton autograd Functions (stubs when triton unavailable) ─────────────

if WITH_TRITON:

    @triton.jit  # type: ignore[attr-defined]
    def triton_cross_scan_flex(
        x: torch.Tensor,
        y: torch.Tensor,
        B: int,
        C: int,
        H: int,
        W: int,
        scans: int,
        ONE_BY_ONE: bool,
        IN_CHANNEL_FIRST: bool,
        OUT_CHANNEL_FIRST: bool,
        BLOCK: int,
    ):
        """Triton kernel for cross-scan (placeholder — uses PyTorch fallback)."""
        raise NotImplementedError("Triton cross-scan kernel not compiled — use force_torch=True")

    class CrossScanTritonF(torch.autograd.Function):
        """Triton cross-scan (falls back to PyTorch when triton unavailable)."""

        @staticmethod
        def forward(ctx, x, in_channel_first, out_channel_first, one_by_one, scans):
            return CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)

        @staticmethod
        def backward(ctx, ys):
            raise NotImplementedError

    class CrossMergeTritonF(torch.autograd.Function):
        """Triton cross-merge (falls back to PyTorch when triton unavailable)."""

        @staticmethod
        def forward(ctx, ys, in_channel_first, out_channel_first, one_by_one, scans):
            return CrossMergeF.apply(ys, in_channel_first, out_channel_first, one_by_one, scans)

        @staticmethod
        def backward(ctx, x):
            raise NotImplementedError

else:

    class CrossScanTritonF:  # type: ignore[no-redef]
        @staticmethod
        def apply(x, in_channel_first, out_channel_first, one_by_one, scans):
            return CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)

    class CrossMergeTritonF:  # type: ignore[no-redef]
        @staticmethod
        def apply(ys, in_channel_first, out_channel_first, one_by_one, scans):
            return CrossMergeF.apply(ys, in_channel_first, out_channel_first, one_by_one, scans)


# ── Public API ────────────────────────────────────────────────────────────


def cross_scan_fn(
    x: torch.Tensor,
    in_channel_first: bool = True,
    out_channel_first: bool = True,
    one_by_one: bool = False,
    scans: int = 0,
    force_torch: bool = False,
) -> torch.Tensor:
    """Cross-scan with automatic Triton / PyTorch fallback.

    Args:
        x: Input ``[B, C, H, W]`` (channel-first) or ``[B, H, W, C]``.
        in_channel_first: Whether input is channel-first.
        out_channel_first: Whether output is channel-first.
        one_by_one: Use one-by-one scanning variant.
        scans: Scan mode (0=cross, 1=unidirectional, 2=bidirectional).
        force_torch: Force pure-PyTorch implementation.

    Returns:
        Scanned tensor ``[B, 4, C, L]`` or equivalent.
    """
    use_triton = WITH_TRITON and x.is_cuda and (not force_torch)
    CSF = CrossScanTritonF if use_triton else CrossScanF
    if x.is_cuda:
        with torch.cuda.device(x.device):
            return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)
    return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)


def cross_merge_fn(
    y: torch.Tensor,
    in_channel_first: bool = True,
    out_channel_first: bool = True,
    one_by_one: bool = False,
    scans: int = 0,
    force_torch: bool = False,
) -> torch.Tensor:
    """Cross-merge with automatic Triton / PyTorch fallback.

    Args:
        y: Input ``[B, 4, C, H, W]`` (channel-first) or ``[B, H, W, 4, C]``.
        in_channel_first: Whether input is channel-first.
        out_channel_first: Whether output is channel-first.
        one_by_one: Use one-by-one merging variant.
        scans: Scan mode (0=cross, 1=unidirectional, 2=bidirectional).
        force_torch: Force pure-PyTorch implementation.

    Returns:
        Merged tensor ``[B, C, H*W]`` or equivalent.
    """
    use_triton = WITH_TRITON and y.is_cuda and (not force_torch)
    CMF = CrossMergeTritonF if use_triton else CrossMergeF
    if y.is_cuda:
        with torch.cuda.device(y.device):
            return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)
    return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)