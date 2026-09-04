"""Checkpoint key normalization for SemiWaferNet resume compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def normalize_semiwafernet_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap legacy classifier keys to the current ModuleDict layout.

    Older checkpoints stored ``classifier.weight`` / ``classifier.bias``.
    Current models use ``classifier.head.weight`` / ``classifier.head.bias``.
    """
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if ".classifier.head." not in key:
            if key.endswith(".classifier.weight"):
                new_key = key.replace(".classifier.weight", ".classifier.head.weight")
            elif key.endswith(".classifier.bias"):
                new_key = key.replace(".classifier.bias", ".classifier.head.bias")
        normalized[new_key] = value
    return normalized


def resume_semiwafernet_engine(engine: Any, checkpoint_path: Path) -> int:
    """Load a SemiWaferNet checkpoint into ``engine`` with key remapping."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = normalize_semiwafernet_state_dict(state["model"])
    engine.model.load_state_dict(model_state)
    engine.optimizer.load_state_dict(state["optimizer"])

    epoch = int(state.get("epoch", 0))
    engine.trainer.current_epoch = epoch
    engine.trainer._best_val_loss = state.get("best_val_loss", float("inf"))

    if engine.scheduler is not None:
        if "scheduler" in state:
            engine.scheduler.load_state_dict(state["scheduler"])
        elif "scheduler_state" in state:
            sched = engine.scheduler
            sched._last_lr = state["scheduler_state"].get("_last_lr", sched._last_lr)
            sched.best = state["scheduler_state"].get("best", sched.best)
            sched.cooldown_counter = state["scheduler_state"].get(
                "cooldown_counter", sched.cooldown_counter
            )
            sched.num_bad_epochs = state["scheduler_state"].get(
                "num_bad_epochs", sched.num_bad_epochs
            )

    if engine.trainer.scaler is not None and "scaler" in state:
        engine.trainer.scaler.load_state_dict(state["scaler"])

    state_path = checkpoint_path.with_suffix(".state.json")
    if state_path.exists():
        import json

        from common.engine.state import EngineState

        with open(state_path, encoding="utf-8") as f:
            data = json.load(f)
        loaded_state = EngineState.from_dict(data)
        engine.state.epoch = loaded_state.epoch
        engine.state.best_metric = loaded_state.best_metric
        engine.state.current_metric = loaded_state.current_metric
        engine.state.global_step = loaded_state.global_step
        engine.state.training_finished = loaded_state.training_finished
        engine.state.checkpoint_path = checkpoint_path
    else:
        engine.state.epoch = epoch

    return epoch


__all__ = ["normalize_semiwafernet_state_dict", "resume_semiwafernet_engine"]
