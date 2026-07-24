"""YOLOv10 baseline and CTM-IYOLOv10 detector wrappers.

Provides:
    - YOLOv10Baseline: wrapper around the official Ultralytics YOLOv10
      DetectionModel.
    - CTMYOLOv10: YOLOv10 with a Context Transformer Module (CTM) inserted
      between the backbone (SPPF → PSA) and the neck, as described in:

          "Wafer Defect Detection Technology Based on CTM-IYOLOv10 Network"

Reference:
    - YOLOv10: Real-Time End-to-End Object Detection (arXiv:2405.14458)
    - https://github.com/THU-MIG/yolov10
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from papers.ctm_yolov10.modules.ctm import CTM


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


# ── CTM-IYOLOv10 ─────────────────────────────────────────────────────────


class CTMYOLOv10(nn.Module):
    """YOLOv10 with optional Context Transformer Module (CTM) integration.

    Inserts a CTM module between the backbone (after SPPF → PSA at layer 10)
    and the PAN-FPN neck. This allows the model to enhance the highest-level
    semantic features with contextual self-attention before passing them to
    the detection head.

    The underlying YOLOv10 ``DetectionModel.model`` is a ``nn.Sequential``
    with 24 layers for the nano variant:

        [0-8]   Backbone (Conv, C2f, SCDown)
        [9]     SPPF
        [10]    PSA
        [11-22] PAN-FPN neck (Upsample, Concat, C2f, SCDown, C2fCIB)
        [23]    v10Detect head

    CTM is inserted **after** layer 10 (PSA) and **before** layer 11
    (first Upsample in the neck), operating at [B, 256, 20, 20] resolution
    for the nano variant.

    The forward pass replicates the YOLOv10 ``_predict_once`` logic but
    intercepts the output of layer 10 (PSA), passes it through CTM, and
    stores the CTM output as the new "layer 10" result so that the neck's
    ``Concat`` layers (which reference ``f=[-1, 6]``, ``f=[-1, 4]``,
    ``f=[-1, 10]``, etc.) receive the CTM-enhanced features.

    When ``ctm_enabled=False``, the forward pass is identical to the
    baseline YOLOv10 (no CTM applied), enabling fair ablation studies.

    Args:
        model_name: YOLOv10 variant name (e.g. ``"yolov10n"``).
        pretrained: Whether to load pretrained COCO weights for the
            YOLOv10 backbone.
        num_classes: Number of output classes.
        ctm_enabled: If ``True``, apply CTM after PSA (layer 10).
            If ``False``, run pure YOLOv10 forward pass (ablation mode).
        ctm_kwargs: Keyword arguments forwarded to the ``CTM`` constructor
            (e.g. ``dim``, ``num_heads``, ``mlp_ratio``, ``dropout``).
    """

    def __init__(
        self,
        model_name: str = "yolov10n",
        pretrained: bool = False,
        num_classes: int = 80,
        ctm_enabled: bool = True,
        **ctm_kwargs: Any,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.ctm_enabled = ctm_enabled

        # Build the base YOLOv10 DetectionModel (keeps .model Sequential intact)
        self.base_model = self._build_base_model(model_name, pretrained, num_classes)
        self.seq: nn.Sequential = self.base_model.model  # type: ignore[assignment]

        # CTM module inserted conceptually after layer 10 (PSA)
        self.ctm = CTM(**ctm_kwargs) if ctm_enabled else None

    @staticmethod
    def _build_base_model(
        model_name: str, pretrained: bool, num_classes: int
    ) -> nn.Module:
        """Construct the underlying YOLOv10 DetectionModel.

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
            yolo = YOLO(f"{model_name}.pt")
            model = yolo.model
            if num_classes != 80:
                _adjust_num_classes(model, num_classes)
        else:
            model = DetectionModel(f"ultralytics/cfg/models/v10/{model_name}.yaml")
            if num_classes != 80:
                _adjust_num_classes(model, num_classes)

        return model

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass with optional CTM insertion after PSA (layer 10).

        Replicates YOLOv10's ``_predict_once`` logic. When ``ctm_enabled``
        is ``True``, intercepts the output of layer 10 (PSA), passes it
        through CTM, and stores the CTM-enhanced features as the new
        ``y[10]`` so that downstream ``Concat`` layers in the neck receive
        the transformed features.

        When ``ctm_enabled`` is ``False``, the forward pass is identical
        to the baseline YOLOv10.

        Args:
            x: Input tensor of shape [B, 3, H, W].

        Returns:
            Raw YOLO outputs (same format as YOLOv10Baseline).
        """
        y: list[torch.Tensor | None] = []  # outputs

        for m in self.seq:
            # Resolve input: if m.f != -1, gather from earlier layers
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]  # type: ignore[assignment]
                else:
                    # m.f is a list like [-1, 6] → [current, layer_6]
                    x = [x if j == -1 else y[j] for j in m.f]  # type: ignore[assignment]

            # Run the layer
            x = m(x)

            # --- CTM insertion: after layer 10 (PSA) ---
            # If CTM is enabled and this is layer 10, pass its output
            # through CTM and use the CTM output as the new result.
            if self.ctm_enabled and m.i == 10:
                x = self.ctm(x)  # type: ignore[misc]

            y.append(x if m.i in self.base_model.save else None)  # type: ignore[attr]

        return x

    @classmethod
    def from_config(cls, config: Any) -> CTMYOLOv10:
        """Instantiate CTM-YOLOv10 from a config object.

        Args:
            config: Config object with ``model.backbone.*``,
                    ``model.ctm.*``, and ``training.num_classes``
                    accessible via ``config.get("...")``.

        Returns:
            Configured CTMYOLOv10 instance.
        """
        return cls(
            model_name=config.get("model.backbone.name", "yolov10n"),
            pretrained=config.get("model.backbone.pretrained", False),
            num_classes=config.get("training.num_classes", 80),
            ctm_enabled=config.get("model.ctm.enabled", True),
            dim=config.get("model.ctm.embed_dim", 256),
            num_heads=config.get("model.ctm.num_heads", 4),
            mlp_ratio=config.get("model.ctm.mlp_ratio", 4.0),
            dropout=config.get("model.ctm.dropout", 0.1),
        )


# ── Shared helpers ───────────────────────────────────────────────────────


def _adjust_num_classes(model: nn.Module, num_classes: int) -> None:
    """Adjust the detection head for a custom number of classes.

    For v10Detect, this fully re-initialises the classification heads
    (cv3, one2one_cv3) with the correct number of output channels.
    The box heads (cv2, one2one_cv2) are independent of nc and are kept.
    """
    if hasattr(model, "model"):
        head = model.model[-1] if hasattr(model.model, "__getitem__") else None
    else:
        head = None

    if head is None:
        for module in model.modules():
            module_name = type(module).__name__
            if "Detect" in module_name:
                head = module
                break

    if head is not None and hasattr(head, "nc"):
        old_nc = head.nc
        head.nc = num_classes

        # For v10Detect, rebuild the classification heads with new nc
        head_type = type(head).__name__
        if head_type == "v10Detect":
            # Rebuild cv3 (cls head) — the last conv layer's out_channels must match nc
            import copy
            from ultralytics.nn.modules import Conv

            c3 = max(head.cv3[0][0][0].conv.out_channels if hasattr(head.cv3[0][0][0], 'conv') else 64, min(num_classes, 100))
            ch = [head.cv3[i][0][0].conv.in_channels if hasattr(head.cv3[i][0][0], 'conv') else 64 for i in range(len(head.cv3))]

            head.cv3 = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                    nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, num_classes, 1),
                )
                for x in ch
            )
            head.one2one_cv3 = copy.deepcopy(head.cv3)
            # Keep cv2 / one2one_cv2 as-is (box head is class-agnostic)
            if hasattr(head, 'cv2') and head.cv2 is not None:
                pass  # box head stays
            if hasattr(head, 'one2one_cv2') and head.one2one_cv2 is not None:
                pass  # box head stays
        else:
            # Original logic for non-v10Detect heads
            if hasattr(head, "cv2"):
                head.cv2 = None
            if hasattr(head, "cv3"):
                head.cv3 = None