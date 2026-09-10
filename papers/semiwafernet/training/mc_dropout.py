"""Monte Carlo Dropout for uncertainty estimation.
Enables dropout during inference and performs multiple stochastic
forward passes to estimate predictive uncertainty via:
    - mean probabilities
    - predictive entropy (raw nats, paper Eq. 11)
    - mutual information (raw nats, paper Eq. 12)
BatchNorm stays in eval mode; only Dropout modules are stochastic
(standard MC-Dropout practice; ``model.train()`` would corrupt BN stats).
"""
from __future__ import annotations
import torch
from torch import nn

def enable_mc_dropout(model: nn.Module) -> None:
    """Keep model in eval mode but activate Dropout for MC sampling."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

class MonteCarloDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation.
    Args:
        num_passes: Number of stochastic forward passes (default: 20).
    """

    def __init__(self, num_passes: int = 20) -> None:
        super().__init__()
        self.num_passes = num_passes

    @torch.no_grad()
    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        logit_bias: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Run MC Dropout and return uncertainty estimates.
        Args:
            model: Teacher / student returning ``classification`` logits.
            x: Input batch.
            logit_bias: Optional per-class bias added to logits before softmax
                (e.g. ``log π`` for balanced-training → imbalanced pool).
            temperature: Softmax temperature ``T`` (``logits / T``). ``T < 1``
                sharpens the predictive distribution for selection.
        Returns:
            Dictionary with:
                mean_probs_class / mean_probs_seg
                entropy_class / entropy_seg — raw predictive entropy (Eq. 11)
                mutual_info_class / mutual_info_seg — raw MI (Eq. 12)
        """
        enable_mc_dropout(model)
        t = max(float(temperature), 1e-6)
        class_probs_list: list[torch.Tensor] = []
        seg_probs_list: list[torch.Tensor] = []
        for _ in range(self.num_passes):
            output = model(x)
            logits = output["classification"]
            if logit_bias is not None:
                logits = logits + logit_bias.to(device=logits.device, dtype=logits.dtype)
            class_probs_list.append(torch.softmax(logits / t, dim=1))
            seg = output["segmentation"]
            if seg.shape[1] == 1:
                seg_probs_list.append(torch.sigmoid(seg))
            else:
                seg_probs_list.append(torch.softmax(seg, dim=1))
        model.eval()
        class_probs_stack = torch.stack(class_probs_list, dim=0)
        seg_probs_stack = torch.stack(seg_probs_list, dim=0)
        mean_class_probs = class_probs_stack.mean(dim=0)
        mean_seg_probs = seg_probs_stack.mean(dim=0)
        entropy_class = self._entropy(mean_class_probs)
        if mean_seg_probs.shape[1] == 1:
            p = mean_seg_probs.squeeze(1).clamp(1e-7, 1 - 1e-7)
            entropy_seg = -(p * p.log() + (1 - p) * (1 - p).log())
            per_pass_entropy_seg = []
            for i in range(self.num_passes):
                pi = seg_probs_stack[i].squeeze(1).clamp(1e-7, 1 - 1e-7)
                per_pass_entropy_seg.append(
                    -(pi * pi.log() + (1 - pi) * (1 - pi).log())
                )
            expected_entropy_seg = torch.stack(per_pass_entropy_seg, dim=0).mean(dim=0)
        else:
            entropy_seg = self._entropy(mean_seg_probs)
            per_pass_entropy_seg = torch.stack(
                [self._entropy(seg_probs_stack[i]) for i in range(self.num_passes)],
                dim=0,
            )
            expected_entropy_seg = per_pass_entropy_seg.mean(dim=0)
        per_pass_entropy_class = torch.stack(
            [self._entropy(class_probs_stack[i]) for i in range(self.num_passes)],
            dim=0,
        )
        expected_entropy_class = per_pass_entropy_class.mean(dim=0)
        mutual_info_class = (entropy_class - expected_entropy_class).clamp(min=0.0)
        mutual_info_seg = (entropy_seg - expected_entropy_seg).clamp(min=0.0)
        return {
            "mean_probs_class": mean_class_probs,
            "mean_probs_seg": mean_seg_probs,
            "entropy_class": entropy_class,
            "entropy_seg": entropy_seg,
            "mutual_info_class": mutual_info_class,
            "mutual_info_seg": mutual_info_seg,
        }

    @staticmethod
    def _entropy(probs: torch.Tensor) -> torch.Tensor:
        """Shannon entropy along class dim=1 (natural log, paper Eq. 11)."""
        eps = torch.finfo(probs.dtype).eps
        clamped = probs.clamp(min=eps)
        return -(clamped * clamped.log()).sum(dim=1)
