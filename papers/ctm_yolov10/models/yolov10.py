"""CTM-IYOLOv10 detector — improved YOLOv10 with GhostConv, BiFPN, and CTM.

Implements the architecture described in:

    "Wafer Defect Detection Technology Based on CTM-IYOLOv10 Network"
    (J. Imaging 2025, 11, 408)

Key improvements over baseline YOLOv10:
    1. GhostConv — lightweight convolution replacing Conv in backbone stages 1-3
       (Section 2.2.1, Figure 4, Figure 5).
    2. BiFPN — weighted bidirectional feature pyramid replacing PAN-FPN in neck
       (Section 2.2.1, Figure 4, Figure 6c).
    3. CTM (Clustering–Template Matching) — preprocessing to segment individual
       dies from multi-die fields of view before detection (Section 2.1, Figure 2).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from papers.ctm_yolov10.modules.ghost_conv import GhostConv
from papers.ctm_yolov10.modules.bifpn import BiFPN


# ── Helpers ────────────────────────────────────────────────────────────────


def _replace_conv_with_ghost(seq: nn.Sequential) -> None:
    """Replace standard Conv layers with GhostConv in backbone stages 1-3.

    In YOLOv10n, the backbone layers are:
        [0]  Conv(k=6, s=2, p=2)  — stem (keep as is)
        [1]  Conv(k=3, s=2)       — stage 1 down → GhostConv
        [2]  C2f                  — stage 1 features
        [3]  Conv(k=3, s=2)       — stage 2 down → GhostConv
        [4]  C2f                  — stage 2 features
        [5]  SCDown               — stage 3 down (keep as is, not Conv)
        [6]  C2f                  — stage 3 features
        [7]  SCDown               — stage 4 down
        [8]  C2f                  — stage 4 features

    Per the paper (Section 2.2.1): "the standard three-by-three convolutional
    layers in the first three stages of the backbone were replaced with
    GhostConv modules." This means layers [1] and [3] (the stride-2 Conv
    layers at the start of stages 1 and 2).

    Args:
        seq: The YOLOv10 ``DetectionModel.model`` (nn.Sequential).
    """
    from ultralytics.nn.modules import Conv

    # Layer indices to replace (Conv with k=3, s=2 in stages 1-2)
    ghost_replacements = {
        1: {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2},
        3: {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 2},
    }

    for idx, params in ghost_replacements.items():
        if idx < len(seq) and isinstance(seq[idx], Conv):
            old_conv = seq[idx]
            new_conv = GhostConv(
                in_channels=params["in_channels"],
                out_channels=params["out_channels"],
                kernel_size=params["kernel_size"],
                stride=params["stride"],
            )
            # Copy the weight shape info for compatibility
            seq[idx] = new_conv


def _build_bifpn() -> BiFPN:
    """Create a BiFPN module for replacing the PAN-FPN neck.

    In YOLOv10n, the neck (layers 11-22) is replaced by BiFPN that
    takes P3 (64ch, 80x80), P4 (128ch, 40x40), P5 (256ch, 20x20)
    features from the backbone and returns enhanced multi-scale features
    at the same resolutions and channel dimensions.

    Returns:
        Configured BiFPN module.
    """
    return BiFPN(channels=256, num_levels=3, num_repeats=2)


# ── Baseline YOLOv10 ──────────────────────────────────────────────────────


class YOLOv10Baseline(nn.Module):
    """YOLOv10 baseline detector (unchanged, for ablation studies).

    Wraps the Ultralytics DetectionModel and provides a forward() that
    returns raw model outputs (no NMS, no postprocessing).
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
        self.model = self._build_model(model_name, pretrained, num_classes)

    def _build_model(self, model_name: str, pretrained: bool, num_classes: int) -> nn.Module:
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
        return self.model(x)

    @classmethod
    def from_config(cls, config: Any) -> YOLOv10Baseline:
        return cls(
            model_name=config.get("model.backbone.name", "yolov10n"),
            pretrained=config.get("model.backbone.pretrained", False),
            num_classes=config.get("training.num_classes", 80),
        )


# ── CTM-IYOLOv10 ──────────────────────────────────────────────────────────


class CTMIYOLOv10(nn.Module):
    """Improved YOLOv10 with GhostConv, BiFPN, and CTM preprocessing.

    Architecture improvements (matching the paper):
        1. GhostConv replaces Conv in backbone stages 1-2 (stride-2 layers).
        2. BiFPN replaces PAN-FPN in the neck for weighted multi-scale fusion.
        3. CTM (Clustering–Template Matching) is applied as a preprocessing
           step to segment individual dies before detection.

    The CTM preprocessing is applied externally (before model.forward()),
    as it operates on raw images (OpenCV). The model itself focuses on
    the GhostConv + BiFPN improvements.

    Args:
        model_name: YOLOv10 variant name (e.g. ``"yolov10n"``).
        pretrained: Whether to load pretrained COCO weights.
        num_classes: Number of output classes.
        ghost_conv: If ``True``, replace Conv with GhostConv in backbone.
        bifpn: If ``True``, replace PAN-FPN with BiFPN in neck.
    """

    def __init__(
        self,
        model_name: str = "yolov10n",
        pretrained: bool = False,
        num_classes: int = 80,
        ghost_conv: bool = True,
        bifpn: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.ghost_conv = ghost_conv
        self.bifpn = bifpn

        # Build base YOLOv10 model
        self.base_model = self._build_base_model(model_name, pretrained, num_classes)
        self.seq: nn.Sequential = self.base_model.model  # type: ignore[assignment]

        # Apply GhostConv replacements
        if ghost_conv:
            _replace_conv_with_ghost(self.seq)

        # Create BiFPN module (used in _predict_once when bifpn=True)
        self.bifpn_module: nn.Module | None = None
        if bifpn:
            self.bifpn_module = _build_bifpn()

    @staticmethod
    def _build_base_model(model_name: str, pretrained: bool, num_classes: int) -> nn.Module:
        from ultralytics import YOLO

        if pretrained:
            yolo = YOLO(f"{model_name}.pt")
        else:
            yolo = YOLO(f"ultralytics/cfg/models/v10/{model_name}.yaml")
        model = yolo.model
        if num_classes != 80:
            _adjust_num_classes(model, num_classes)
        return model

    def forward(self, x: torch.Tensor | dict) -> Any:
        """Forward pass with optional BiFPN enhancement.

        When ``x`` is a dict (training path), delegates to ``self.loss(x)``.
        When ``x`` is a tensor (inference path), runs the forward pass.
        """
        if isinstance(x, dict):
            return self.loss(x)
        return self._predict_once(x)

    def _predict_once(self, x: torch.Tensor) -> Any:
        """Run forward pass with optional BiFPN enhancement.

        When ``bifpn=True``:
            1. Run backbone layers (0-10) manually, collecting P3 (layer 4),
               P4 (layer 6), P5 (layer 10) feature maps.
            2. Pass [P3, P4, P5] through BiFPN for weighted multi-scale fusion.
            3. Feed BiFPN outputs directly to the detection head (layer 23).

        When ``bifpn=False``:
            Delegate to ``base_model._predict_once(x)`` (standard YOLOv10 forward
            with GhostConv already applied in ``self.seq``).
        """
        if not self.bifpn or self.bifpn_module is None:
            # Standard YOLOv10 forward (with GhostConv already in seq)
            return self.base_model._predict_once(x)

        # ── BiFPN forward ────────────────────────────────────────────────
        # Run backbone layers (0-10) manually to collect P3, P4, P5 features.
        # The neck layers (11-22) are skipped entirely — BiFPN replaces them.
        # Detection head (layer 23) receives BiFPN-enhanced features directly.

        y: list[torch.Tensor] = []
        p3, p4, p5 = None, None, None
        head = self.seq[-1]  # v10Detect

        for i, m in enumerate(self.seq):
            # Stop before the detection head — we handle it separately
            if i == len(self.seq) - 1:
                break

            m_f = getattr(m, 'f', -1)
            m_i = getattr(m, 'i', -1)

            # Handle skip connections (standard YOLOv10 _predict_once logic)
            if m_f != -1:
                if isinstance(m_f, int):
                    x = y[m_f]
                else:
                    x = [x if j == -1 else y[j] for j in m_f]

            x = m(x)

            # Collect backbone feature maps at P3, P4, P5
            if m_i == 4:
                p3 = x  # [B, 64, 80, 80]
            elif m_i == 6:
                p4 = x  # [B, 128, 40, 40]
            elif m_i == 10:
                p5 = x  # [B, 256, 20, 20]

            y.append(x)

        # After backbone (layer 10), run BiFPN
        if p3 is None or p4 is None or p5 is None:
            raise RuntimeError(
                "BiFPN requires P3, P4, P5 features from backbone. "
                f"Got p3={'✓' if p3 is not None else '✗'}, "
                f"p4={'✓' if p4 is not None else '✗'}, "
                f"p5={'✓' if p5 is not None else '✗'}"
            )

        # Run BiFPN: [P3, P4, P5] -> enhanced [P3_out, P4_out, P5_out]
        bifpn_out = self.bifpn_module([p3, p4, p5])

        # Feed BiFPN outputs directly to the detection head
        # v10Detect expects a list of 3 feature maps [P3, P4, P5]
        return head(bifpn_out)

    def loss(self, batch: dict) -> torch.Tensor:
        """Compute loss with improved model predictions."""
        if getattr(self.base_model, "criterion", None) is None:
            self.base_model.criterion = self.base_model.init_criterion()

        from types import SimpleNamespace

        def _fix_hyp(crit: object) -> None:
            if hasattr(crit, "hyp"):
                if isinstance(crit.hyp, dict):
                    crit.hyp = SimpleNamespace(**crit.hyp)
                if not hasattr(crit.hyp, "box"):
                    crit.hyp.box = 7.5
                if not hasattr(crit.hyp, "cls"):
                    crit.hyp.cls = 2.0
                if not hasattr(crit.hyp, "dfl"):
                    crit.hyp.dfl = 1.5

        _fix_hyp(self.base_model.criterion)
        if hasattr(self.base_model.criterion, "one2many"):
            _fix_hyp(self.base_model.criterion.one2many)
        if hasattr(self.base_model.criterion, "one2one"):
            _fix_hyp(self.base_model.criterion.one2one)

        preds = self._predict_once(batch["img"])
        return self.base_model.criterion(preds, batch)

    @classmethod
    def from_config(cls, config: Any) -> CTMIYOLOv10:
        """Instantiate from a config object."""
        return cls(
            model_name=config.get("model.backbone.name", "yolov10n"),
            pretrained=config.get("model.backbone.pretrained", False),
            num_classes=config.get("training.num_classes", 80),
            ghost_conv=config.get("model.ghost_conv.enabled", True),
            bifpn=config.get("model.bifpn.enabled", True),
        )


# ── Shared helpers ────────────────────────────────────────────────────────


def _adjust_num_classes(model: nn.Module, num_classes: int) -> None:
    """Adjust the detection head for a custom number of classes."""
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
        head.nc = num_classes
        head_type = type(head).__name__
        if head_type == "v10Detect":
            import copy
            from ultralytics.nn.modules import Conv

            ch = []
            for i in range(len(head.cv3)):
                first_conv = head.cv3[i][0][0]
                ch.append(first_conv.conv.in_channels)

            c3 = max(ch[0], min(num_classes, 100))

            head.cv3 = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                    nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, num_classes, 1),
                )
                for x in ch
            )
            head.one2one_cv3 = copy.deepcopy(head.cv3)
        else:
            if hasattr(head, "cv2"):
                head.cv2 = None
            if hasattr(head, "cv3"):
                head.cv3 = None