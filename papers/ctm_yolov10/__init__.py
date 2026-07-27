"""CTM-IYOLOv10 reproduction.

Registers the CTMIYOLOv10 model with the common engine registry so it can be
instantiated by name via ``build_model("ctm_iyolov10", ...)``.
"""

from __future__ import annotations

from common.engine.registry import register_model
from papers.ctm_yolov10.models.yolov10 import CTMIYOLOv10, YOLOv10Baseline

# Register with the common engine registry.
try:
    register_model("ctm_iyolov10", CTMIYOLOv10)
except ValueError:
    pass

try:
    register_model("yolov10_baseline", YOLOv10Baseline)
except ValueError:
    pass

__all__ = [
    "CTMIYOLOv10",
    "YOLOv10Baseline",
]
