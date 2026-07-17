"""CTM-YOLOv10 reproduction.

Registers the CTMYOLOv10 model with the common engine registry so it can be
instantiated by name via ``build_model("ctm_yolov10", ...)``.
"""

from __future__ import annotations

from common.engine.registry import register_model
from papers.ctm_yolov10.models.yolov10 import CTMYOLOv10, YOLOv10Baseline

# Register with the common engine registry.
# After this import, build_model("ctm_yolov10", ...) works without manual imports.
# Use try/except to handle re-registration gracefully (e.g. during test discovery).
try:
    register_model("ctm_yolov10", CTMYOLOv10)
except ValueError:
    pass

try:
    register_model("yolov10_baseline", YOLOv10Baseline)
except ValueError:
    pass

__all__ = [
    "CTMYOLOv10",
    "YOLOv10Baseline",
]
