"""CNN from Kim & Kim (2024), Table 4, with a recorded input adapter."""

import torch
from torch import nn
from torch.nn import functional as F


class PolarCNN(nn.Module):
    """Accept [B,1,7,7] sector counts and return unnormalised class logits.

    Table 4 specifies valid convolutions on 26x26 but Sec. 3.3 specifies
    7x7 histograms. The undocumented bridge is exposed as ``input_adapter``.
    One input channel matches the stated first-convolution parameter count
    (320); Table 4's printed three-channel input would require 896 instead.
    """

    def __init__(self, num_classes=8, input_adapter="nearest"):
        super().__init__()
        if input_adapter not in {"nearest", "bilinear"}:
            raise ValueError("input_adapter must be nearest or bilinear")
        self.input_adapter = input_adapter
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        # Keras defaults: Glorot-uniform kernels and zero biases.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 7, 7):
            raise ValueError("PolarCNN requires [B,1,7,7] polar sector counts")
        kwargs = {"align_corners": False} if self.input_adapter == "bilinear" else {}
        x = F.interpolate(x, size=(26, 26), mode=self.input_adapter, **kwargs)
        return self.classifier(self.features(x))
