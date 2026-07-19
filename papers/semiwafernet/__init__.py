"""SemiWaferNet reproduction.

Registers the SemiWaferNet model with the common engine registry so it can be
instantiated by name via ``build_model("semiwafernet", ...)``.
"""

from __future__ import annotations

from common.engine.registry import register_model
from papers.semiwafernet.models.semiwafernet import SemiWaferNet

# Register with the common engine registry.
# After this import, build_model("semiwafernet", ...) works without manual imports.
# Use try/except to handle re-registration gracefully (e.g. during test discovery).
try:
    register_model("semiwafernet", SemiWaferNet)
except ValueError:
    pass

__all__ = [
    "SemiWaferNet",
]
