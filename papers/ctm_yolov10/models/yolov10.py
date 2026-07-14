"""YOLOv10 baseline detector wrapper.

Wraps the official Ultralytics YOLOv10 DetectionModel as a torch.nn.Module,
providing a clean interface for future CTM integration.

Reference:
    - YOLOv10: Real-Time End-to-End Object Detection (arXiv:2405.14458)
    - https://github.com/THU-MIG/yolov10
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class YOLOv10Baseline(nn.Module):
    """YOLOv10 baseline detector.

    Wraps the Ultralytics DetectionModel and provides a forward() that
    returns raw model outputs (no NMS, no postprocessing).

    The class is designed so that CTM modules can be inserted into the
    backbone/neck in a future stage without changing the external interface.

    Args:
        model_name: YOLOv10 variant name (e.g. ``"yolov10n"``, ``"yolov10s"``).
        pretrained: Whether to load pretrained COCO weights.
        num_classes: Number of output classes (default 80 for COCO).
    """

    def __init__(
        self,
        model_name: str = "yolov10n",
        pretrained: bool = False,
        num_classes: int = 80,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes

        # Build the underlying DetectionModel
        self.model = self._build_model(model_name, pretrained, num_classes)

    def _build_model(
        self, model_name: str, pretrained: bool, num_classes: int
    ) -> nn.Module:
        """Construct the YOLOv10 DetectionModel.

        Args:
            model_name: Variant name (e.g. ``yolov10n``).
            pretrained: Load pretrained COCO weights.
            num_classes: Number of output classes.

        Returns:
            Configured DetectionModel.
        """
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel

        if pretrained:
            # Load full YOLO model with pretrained weights
            yolo = YOLO(f"{model_name}.pt")
            model = yolo.model
            # Update number of classes if different from default (80)
            if num_classes != 80:
                self._adjust_num_classes(model, num_classes)
        else:
            # Create model from YAML config without weights
            model = DetectionModel(f"ultralytics/cfg/models/v10/{model_name}.yaml")
            if num_classes != 80:
                self._adjust_num_classes(model, num_classes)

        return model

    def _adjust_num_classes(self, model: nn.Module, num_classes: int) -> None:
        """Adjust the detection head for a custom number of classes."""
        # The v10Detect head stores nc (num_classes) and reorganises its outputs
        if hasattr(model, "model"):
            head = model.model[-1] if hasattr(model.model, "__getitem__") else None
        else:
            head = None

        # Fallback: iterate modules to find Detect head
        if head is None:
            for module in model.modules():
                module_name = type(module).__name__
                if "Detect" in module_name:
                    head = module
                    break

        if head is not None and hasattr(head, "nc"):
            head.nc = num_classes
            # Re-initialise head conv layers for new class count
            if hasattr(head, "cv2"):
                head.cv2 = None  # will be rebuilt on first forward
            if hasattr(head, "cv3"):
                head.cv3 = None

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, 3, H, W].

        Returns:
            Raw YOLO outputs (tuple of [detections, loss_dict] during training,
            or dict during inference).
        """
        return self.model(x)

    @classmethod
    def from_config(cls, config: Any) -> YOLOv10Baseline:
        """Instantiate model from a config object.

        Args:
            config: Config object with ``model.backbone.*`` and
                    ``model.input.*`` attributes accessible via
                    ``config.get("...")``.

        Returns:
            Configured YOLOv10Baseline instance.
        """
        return cls(
            model_name=config.get("model.backbone.name", "yolov10n"),
            pretrained=config.get("model.backbone.pretrained", False),
            num_classes=config.get("training.num_classes", 80),
        )