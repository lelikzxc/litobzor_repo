"""Checkpoint key normalization for SemiWaferNet resume compatibility."""

from __future__ import annotations

import math
import shutil
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
        if key in ("classifier.weight", "classifier.bias"):
            new_key = key.replace("classifier.", "classifier.head.", 1)
        elif key.endswith((".classifier.weight", ".classifier.bias")):
            new_key = key.replace(".classifier.", ".classifier.head.")
        if new_key in normalized:
            raise ValueError(f"Checkpoint contains conflicting classifier keys: {new_key}")
        normalized[new_key] = value
    return normalized


def _restore_best_metric(engine: Any, checkpoint_path: Path, state: dict[str, Any]) -> None:
    """Keep the saved best score when resuming a checkpoint without a sidecar.

    The shared supervised trainer stores validation *loss* in ``last.pt``'s
    ``metric`` field, even when selection uses ``val_f1``. Only ``best.pt``'s
    selection score is authoritative. A sidecar's historical score cannot
    protect weights that are absent from a fresh destination directory.
    """
    manager = engine.checkpoint_manager
    if manager is None:
        return
    best_path = Path(manager.best_path)
    if (
        not best_path.exists()
        and checkpoint_path.name == "best.pt"
        and state.get("metric_name") == manager.metric_name
        and state.get("metric") is not None
        and math.isfinite(float(state["metric"]))
    ):
        # Explicitly seeding a fresh run from a selected best checkpoint keeps
        # its actual weights available for final evaluation, not just its score.
        best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, best_path)

    saved = None
    if best_path.exists():
        saved = state if best_path == checkpoint_path else torch.load(
            best_path, map_location="cpu", weights_only=False
        )
    if (
        saved is not None
        and saved.get("metric_name") == manager.metric_name
        and saved.get("metric") is not None
        and math.isfinite(float(saved["metric"]))
    ):
        engine.state.best_metric = float(saved["metric"])
        manager._best_metric = engine.state.best_metric
    else:
        # A resumed stage boundary only contains that stage's chosen weights;
        # its historical global-best artifact may be in another directory.
        engine.state.best_metric = None
        manager._best_metric = -float("inf") if manager.mode == "max" else float("inf")


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
        engine.state.best_metric = loaded_state.best_metric
        engine.state.current_metric = loaded_state.current_metric
        engine.state.global_step = loaded_state.global_step
        engine.state.training_finished = loaded_state.training_finished
        engine.state.extra = dict(loaded_state.extra)
    # Tensor weights and optimizer state define the resumed epoch. A JSON
    # sidecar can be stale if writing was interrupted between the two files.
    engine.state.epoch = epoch
    engine.state.checkpoint_path = checkpoint_path
    _restore_best_metric(engine, checkpoint_path, state)
    return epoch


__all__ = ["normalize_semiwafernet_state_dict", "resume_semiwafernet_engine"]
