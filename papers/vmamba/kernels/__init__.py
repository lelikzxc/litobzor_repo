"""Vendored VMamba kernel implementations.

This package contains vendored copies of the official VMamba kernel modules:
    - csms6s.py: Selective scan implementation (CUDA + pure-PyTorch fallback)
    - csm_triton.py: Cross scan/merge implementation (Triton + pure-PyTorch fallback)
    - vmamba_official.py: Official SS2D, VSSBlock, and supporting classes

These are copied from https://github.com/MzeroMiko/VMamba and adapted
to work without compiled CUDA/Triton extensions by using pure-PyTorch
fallbacks when the native kernels are unavailable.
"""

from papers.vmamba.kernels.csm_triton import cross_scan_fn, cross_merge_fn
from papers.vmamba.kernels.csms6s import selective_scan_fn

__all__ = ["cross_scan_fn", "cross_merge_fn", "selective_scan_fn"]