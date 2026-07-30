"""RadonCNN: Rotation- and flip-invariant CNN for wafer map classification.

Based on:
    Jeong et al., "Wafer map failure pattern classification using geometric
    transformation-invariant convolutional neural network",
    Scientific Reports 2023, 13:8127. https://doi.org/10.1038/s41598-023-34147-2
"""

from __future__ import annotations

from common.engine.registry import register_model
from papers.radon_cnn.models.radon_cnn import RadonCNN

try:
    register_model("radon_cnn", RadonCNN)
except ValueError:
    pass  # Already registered