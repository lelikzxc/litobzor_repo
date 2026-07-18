"""SegFormer + Atrous reproduction.

Registers the SegFormer model with the common engine registry so it can be
instantiated by name via ``build_model("segformer_atrous", ...)``.
"""

from __future__ import annotations

from common.engine.registry import register_model
from papers.transformer_segmentation.models.segformer import SegFormer

# Register with the common engine registry.
# After this import, build_model("segformer_atrous", ...) works without manual imports.
# Use try/except to handle re-registration gracefully (e.g. during test discovery).
try:
    register_model("segformer_atrous", SegFormer)
except ValueError:
    pass

__all__ = [
    "SegFormer",
]
