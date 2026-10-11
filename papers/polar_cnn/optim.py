"""Keras 2.15 Adam's uncorrected-second-moment epsilon convention."""

import math

import torch
from torch.optim import Optimizer


class KerasAdam(Optimizer):
    """Keras Adam (not AMSGrad), no weight decay or gradient clipping.

    alpha = lr * sqrt(1-beta2**t)/(1-beta1**t)
    variable -= alpha*m/(sqrt(v)+epsilon)

    PyTorch Adam adds epsilon after correcting v; setting the same epsilon
    constant in its constructor does not implement this update exactly.
    """

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-7):
        if lr <= 0 or eps <= 0 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("Invalid Keras Adam hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            # Keras uses the optimiser-wide iteration, not a per-variable clock.
            group["iterations"] = group.get("iterations", 0) + 1
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Polar CNN KerasAdam requires dense gradients")
                state = self.state[parameter]
                if not state:
                    state.update(
                        step=0,
                        exp_avg=torch.zeros_like(parameter),
                        exp_avg_sq=torch.zeros_like(parameter),
                    )
                step = group["iterations"]
                state["step"] = step
                alpha = group["lr"] * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.add_((grad - m) * (1 - beta1))
                v.add_((grad.square() - v) * (1 - beta2))
                parameter.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-alpha)
        return loss
