"""High-level inference wrapper for model prediction.

The ``Predictor`` wraps a ``torch.nn.Module`` and provides a consistent
interface for single-sample and batch inference. It handles device placement,
eval mode, gradient disabling, and basic postprocessing.

Output format::

    {
        "logits":      torch.Tensor,   # raw model outputs
        "probs":       torch.Tensor,   # softmax / sigmoid probabilities
        "prediction":  torch.Tensor,   # argmax class indices (or raw for detection)
    }
"""

from __future__ import annotations

from typing import Any

import torch


class Predictor:
    """Inference wrapper for a PyTorch model.

    Args:
        model: A ``torch.nn.Module`` in eval mode (will be set automatically).
        device: Device string (e.g. ``"cpu"``, ``"cuda:0"``). If ``None``,
            auto-detected.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | None = None,
    ) -> None:
        self.model = model

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()
        self._disable_gradients()

    @staticmethod
    def _disable_gradients() -> None:
        """Disable gradient computation globally for inference."""
        torch.set_grad_enabled(False)

    # ── Public API ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_single(self, image: torch.Tensor) -> dict[str, Any]:
        """Run inference on a single image.

        Args:
            image: Input tensor of shape ``[C, H, W]``.

        Returns:
            Dict with ``"logits"``, ``"probs"``, and ``"prediction"`` keys.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, C, H, W]
        return self._predict(image)

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> dict[str, Any]:
        """Run inference on a batch of images.

        Args:
            images: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            Dict with ``"logits"``, ``"probs"``, and ``"prediction"`` keys.
        """
        return self._predict(images)

    @torch.no_grad()
    def predict(self, dataloader: Any) -> list[dict[str, Any]]:
        """Run inference over an entire DataLoader.

        Args:
            dataloader: An iterable yielding batches (tensors or dicts).

        Returns:
            List of result dicts, one per batch.
        """
        results: list[dict[str, Any]] = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            elif isinstance(batch, dict):
                images = batch.get("image", batch.get("x"))
            else:
                images = batch

            result = self._predict(images)
            results.append(result)
        return results

    # ── Internal ──────────────────────────────────────────────────────────

    def _predict(self, images: torch.Tensor) -> dict[str, Any]:
        """Core prediction logic.

        Args:
            images: Input tensor ``[B, C, H, W]``.

        Returns:
            Dict with ``"logits"``, ``"probs"``, and ``"prediction"`` keys.
        """
        images = images.to(self.device)
        logits = self.model(images)

        # Build output dict with what we have
        result: dict[str, Any] = {
            "logits": logits,
        }

        # Try to compute probabilities
        if isinstance(logits, torch.Tensor):
            if logits.dim() >= 2 and logits.size(-1) > 1:
                result["probs"] = torch.softmax(logits, dim=-1)
                result["prediction"] = torch.argmax(logits, dim=-1)
            elif logits.dim() >= 2 and logits.size(-1) == 1:
                result["probs"] = torch.sigmoid(logits)
                result["prediction"] = (result["probs"] > 0.5).long()
            else:
                result["probs"] = logits
                result["prediction"] = logits
        else:
            # Detection model output (tuple/dict) — pass through
            result["probs"] = logits
            result["prediction"] = logits

        return result