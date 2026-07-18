"""Tiny Vision Transformer paper reproduction.

Registers the ViTTiny model with the common engine registry so it can be
instantiated by name via ``build_model("vit_tiny", ...)``.
"""

from __future__ import annotations

from common.engine.registry import register_model
from papers.vit_tiny.models.vit_tiny import ViTTiny

# Register with the common engine registry.
# After this import, build_model("vit_tiny", ...) works without manual imports.
# Use try/except to handle re-registration gracefully (e.g. during test discovery).
try:
    register_model("vit_tiny", ViTTiny)
except ValueError:
    pass

__all__ = [
    "ViTTiny",
]
