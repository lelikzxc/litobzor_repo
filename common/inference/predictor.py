"""Generic Predictor class for model inference.

Supports single image, batch, tensor, dict input, and DataLoader input.
Returns a unified dictionary format for all prediction types.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from common.inference.device import get_best_device, move_to_device
from common.inference.postprocessing import logits_to_class, logits_to_probs
from common.inference.utils import disable_gradients, set_eval_mode


class Predictor:
    """Generic model predictor for classification and segmentation.

    Wraps a model with device management, eval mode, gradient disabling,
    and optional custom postprocessing. Supports multiple input formats
    and returns a unified dictionary.

    Args:
        model: The PyTorch model to use for inference.
        device: Target device. If ``"auto"``, uses ``get_best_device()``.
        postprocess_fn: Optional callable that transforms raw model outputs.
            If ``None``, default postprocessing (probs + argmax) is applied.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "auto",
        postprocess_fn: Callable[[torch.Tensor], dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.postprocess_fn = postprocess_fn

        if device == "auto":
            device = get_best_device()
        self.device = torch.device(device) if isinstance(device, str) else device

        # Prepare model
        move_to_device(self.model, self.device)
        set_eval_mode(self.model)
        disable_gradients(self.model)

    def predict_single(
        self,
        image: torch.Tensor | np.ndarray,
    ) -> dict[str, Any]:
        """Run inference on a single image.

        Args:
            image: Input image ``[C, H, W]`` or ``[H, W, C]`` (numpy).
                If numpy, it is converted to a tensor and channel-first layout
                is assumed if the last dimension is not 3 or 1.

        Returns:
            Dict with keys ``"logits"``, ``"probs"``, ``"prediction"``.
        """
        # Convert numpy to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Ensure channel-first: [C, H, W]
        if image.dim() == 3 and image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)

        # Add batch dimension: [1, C, H, W]
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        return self._run_inference(image)

    def predict_batch(
        self,
        images: torch.Tensor | list[torch.Tensor],
    ) -> dict[str, Any]:
        """Run inference on a batch of images.

        Args:
            images: Batch tensor ``[B, C, H, W]`` or list of tensors.

        Returns:
            Dict with keys ``"logits"``, ``"probs"``, ``"prediction"``.
        """
        if isinstance(images, list):
            images = torch.stack(images)
        images = images.to(self.device)
        return self._run_inference(images)

    def predict(
        self,
        dataloader: DataLoader,
    ) -> list[dict[str, Any]]:
        """Run inference over an entire DataLoader.

        Args:
            dataloader: A PyTorch DataLoader yielding tensors, tuples, or dicts.

        Returns:
            List of result dicts, one per batch. Each dict has keys
            ``"logits"``, ``"probs"``, ``"prediction"``.
        """
        results: list[dict[str, Any]] = []
        for batch in dataloader:
            if isinstance(batch, dict):
                # Support dict batches with an "image" key
                images = batch.get("image", batch.get("images", batch.get("x")))
                if images is None:
                    # Use the first tensor value found
                    images = next(v for v in batch.values() if isinstance(v, torch.Tensor))
            elif isinstance(batch, (tuple, list)):
                # DataLoader yields tuples (e.g. from TensorDataset)
                images = batch[0]
            else:
                images = batch

            if isinstance(images, (tuple, list)):
                images = torch.stack(images)
            images = images.to(self.device)
            result = self._run_inference(images)
            results.append(result)
        return results

    def _run_inference(self, inputs: torch.Tensor) -> dict[str, Any]:
        """Execute the forward pass and postprocessing.

        Args:
            inputs: Input tensor ``[B, C, H, W]`` on the correct device.

        Returns:
            Dict with keys ``"logits"``, ``"probs"``, ``"prediction"``.
        """
        with torch.no_grad():
            logits: torch.Tensor = self.model(inputs)

        if self.postprocess_fn is not None:
            return self.postprocess_fn(logits)

        # Default postprocessing
        probs = logits_to_probs(logits)
        prediction = logits_to_class(logits)
        return {
            "logits": logits.cpu(),
            "probs": probs.cpu(),
            "prediction": prediction.cpu(),
        }