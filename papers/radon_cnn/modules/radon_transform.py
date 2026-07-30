"""Radon transform module for rotation-equivariant feature extraction.

Based on the paper:
    Jeong et al., "Wafer map failure pattern classification using geometric
    transformation-invariant convolutional neural network",
    Scientific Reports 2023, 13:8127.

The Radon transform converts rotation in the original image to translation
in the Radon feature space, serving as a rotation-equivariant bridge for
translation-invariant CNNs.

Uses scikit-image's ``radon`` function per the paper:
    "Radon and inverse Radon transforms were performed with the scikit-image
    library version 0.20.0."
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RadonTransformModule(nn.Module):
    """Radon transform layer that converts rotation to translation.

    Applies the Radon transform to input wafer maps, producing sinogram
    features where rotation in the input becomes translation along the
    angular axis.

    Per the paper (Eq. 11-12):
        r = x*cos(theta) + y*sin(theta)
        P_theta(r) = sum_x sum_y f(x,y) * delta(x*cos(theta) + y*sin(theta) - r)

    The output is resized to ``(image_size, image_size)`` to match the
    paper's stated output shape of 64×64×1 (Table 2).

    Args:
        theta: Number of projection angles (default 64).
        image_size: Target size for the output sinogram (default 64).
                    The sinogram is resized to ``(image_size, image_size)``.
    """

    def __init__(
        self,
        theta: int = 64,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Radon transform to input batch.

        Args:
            x: Input tensor of shape [B, 1, H, W] (grayscale images).

        Returns:
            Radon-transformed tensor of shape [B, 1, image_size, image_size].
        """
        import warnings

        from skimage.transform import radon, resize

        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Radon transform is a deterministic non-learnable operation.
        # We detach from autograd for the numpy conversion, then re-attach
        # the result so that downstream layers receive gradients.
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()

            theta_range = np.linspace(0.0, 180.0, self.theta, endpoint=False)

            outputs: list[np.ndarray] = []
            for i in range(batch_size):
                img = x_np[i, 0]  # [H, W]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sinogram = radon(
                        img,
                        theta=theta_range,
                        circle=False,
                    )
                # sinogram shape: [num_detectors, num_angles]
                # Resize to (image_size, image_size) for consistent CNN input
                sinogram = resize(
                    sinogram,
                    (self.image_size, self.image_size),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                outputs.append(sinogram)

            # Stack and convert back to tensor
            result = np.stack(outputs, axis=0)  # [B, H, W]
            result_tensor = torch.from_numpy(result).to(device=device, dtype=dtype)

        # Add channel dimension: [B, 1, H, W]
        result_tensor = result_tensor.unsqueeze(1)

        # Re-attach to autograd graph so downstream layers receive gradients.
        # The Radon transform itself is not learnable, but we need the
        # computation graph to flow through it for the CNN layers after it.
        result_tensor.requires_grad_(x.requires_grad)

        return result_tensor

    def extra_repr(self) -> str:
        return f"theta={self.theta}, image_size={self.image_size}"