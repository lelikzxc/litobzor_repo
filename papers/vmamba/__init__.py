"""FCS-VMamba reproduction.

Registers the FCSVMamba model with the common engine registry so it can be
instantiated by name via ``build_model("fcs_vmamba", ...)``.
"""

from __future__ import annotations

from common.engine.registry import register_model
from papers.vmamba.models.vmamba import FCSVMamba

# Register with the common engine registry.
# After this import, build_model("fcs_vmamba", ...) works without manual imports.
# Use try/except to handle re-registration gracefully (e.g. during test discovery).
try:
    register_model("fcs_vmamba", FCSVMamba)
except ValueError:
    pass

__all__ = [
    "FCSVMamba",
]
