"""VMamba checkpoint loading without changing the shared training engine."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from common.engine.state import EngineState
from papers.reproduction import _difference_path


def validate_resume_protocol(
    checkpoint_path: str | Path,
    current_config: Any,
    samples: Any,
    splits: Mapping[str, Any],
    seed: int,
) -> Path:
    """Check the source experiment before loading an external optimizer state.

    Moving checkpoints to a new output directory or choosing another logging
    verbosity does not change training. All scientific configuration, sample
    identities/order and exact split indices must match the original run.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    protocol_path = path.parent / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Missing {protocol_path}: resume requires the original training protocol"
        )
    previous = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = current_config.to_dict() if hasattr(current_config, "to_dict") else dict(current_config)

    def scientific_config(value: Mapping[str, Any]) -> dict[str, Any]:
        # Use the same JSON representation as save_protocol (tuples become
        # lists). This also makes the source and destination independent copies.
        value = json.loads(json.dumps(value))
        value.pop("logging", None)
        value.get("checkpoint", {}).pop("save_dir", None)
        return value

    expected = {
        "config": scientific_config(config),
        "sample_digest": hashlib.sha256(json.dumps(samples).encode()).hexdigest(),
        "splits": json.loads(json.dumps(splits)),
        "seed": seed,
    }
    original = {
        "config": scientific_config(previous.get("config", {})),
        "sample_digest": previous.get("sample_digest"),
        "splits": previous.get("splits"),
        "seed": previous.get("seed"),
    }
    differences = [
        difference
        for key, value in expected.items()
        if (difference := _difference_path(original[key], value, key)) is not None
    ]
    if differences:
        raise ValueError(
            f"Resume protocol differs from {protocol_path}: {', '.join(differences)}; "
            "use the original configuration, dataset and split"
        )
    return protocol_path


def checkpoint_model_state(state: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Read the supported checkpoint containers; architecture keys stay strict."""
    model_state = state.get("model", state.get("model_state_dict", state))
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("Checkpoint does not contain a nonempty model state_dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in model_state.items()):
        raise ValueError("Checkpoint model state_dict must contain named tensors")
    return model_state


def _selection_metric(state: Mapping[str, Any], metric_name: str) -> float | None:
    if state.get("metric_name") != metric_name or state.get("metric") is None:
        return None
    try:
        metric = float(state["metric"])
    except (TypeError, ValueError):
        return None
    return metric if math.isfinite(metric) else None


def _restore_available_best(engine: Any, path: Path, state: Mapping[str, Any]) -> None:
    """Preserve scores only when the corresponding selected weights are present.

    The shared trainer writes validation loss into last.pt's metric field,
    including when metric_name is val_accuracy. Only best.pt represents the
    selected metric. A historical sidecar score alone cannot select weights.
    """
    manager = engine.checkpoint_manager
    if manager is None:
        return
    best_path = Path(manager.best_path)
    best_metric = None
    if best_path.is_file():
        best_state = state if best_path.resolve() == path.resolve() else torch.load(
            best_path, map_location="cpu", weights_only=False
        )
        best_metric = _selection_metric(best_state, manager.metric_name)
        if best_metric is None:
            raise ValueError(
                f"Selected checkpoint {best_path} has no finite {manager.metric_name} score; "
                "use a fresh checkpoint directory"
            )

    incoming_metric = _selection_metric(state, manager.metric_name) if path.name == "best.pt" else None
    incoming_better = incoming_metric is not None and (
        best_metric is None
        or (incoming_metric > best_metric if manager.mode == "max" else incoming_metric < best_metric)
    )
    if incoming_better:
        if path.resolve() != best_path.resolve():
            best_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, best_path)
        best_metric = incoming_metric

    engine.state.best_metric = best_metric
    manager._best_metric = best_metric if best_metric is not None else (
        -float("inf") if manager.mode == "max" else float("inf")
    )


def resume_vmamba_engine(engine: Any, checkpoint_path: str | Path) -> int:
    """Restore model, optimizer, scheduler and the authoritative saved epoch.

    A weights-only file is valid for evaluation, but cannot resume AdamW and
    its learning-rate schedule. Incompatible architecture keys fail loudly.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError("Training checkpoint must be a state dictionary")
    if "optimizer" not in state:
        raise ValueError("Cannot resume a weights-only checkpoint: optimizer state is missing")
    if engine.scheduler is not None and not any(key in state for key in ("scheduler", "scheduler_state")):
        raise ValueError("Cannot resume this training checkpoint: scheduler state is missing")
    epoch = state.get("epoch", 0)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"Checkpoint epoch must be a nonnegative integer, got {epoch!r}")

    engine.model.load_state_dict(checkpoint_model_state(state))
    # PyTorch casts Adam's moments to each parameter's device/dtype here;
    # non-capturable Adam intentionally retains its scalar step on CPU.
    engine.optimizer.load_state_dict(state["optimizer"])
    if engine.scheduler is not None:
        if "scheduler" in state:
            engine.scheduler.load_state_dict(state["scheduler"])
        else:
            for key, value in state["scheduler_state"].items():
                if hasattr(engine.scheduler, key):
                    setattr(engine.scheduler, key, value)
    if engine.trainer.scaler is not None and "scaler" in state:
        engine.trainer.scaler.load_state_dict(state["scaler"])
    engine.trainer.current_epoch = epoch
    engine.trainer._best_val_loss = state.get("best_val_loss", float("inf"))

    sidecar = path.with_suffix(".state.json")
    if sidecar.is_file():
        loaded_state = EngineState.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
        engine.state.current_metric = loaded_state.current_metric
        engine.state.global_step = loaded_state.global_step
        engine.state.training_finished = loaded_state.training_finished
        engine.state.extra = dict(loaded_state.extra)
    # A sidecar can be one write behind after interruption. The .pt epoch
    # belongs to the actual restored model and optimizer, so it takes priority.
    engine.state.epoch = epoch
    engine.state.checkpoint_path = path
    _restore_available_best(engine, path, state)
    return epoch


__all__ = ["checkpoint_model_state", "resume_vmamba_engine", "validate_resume_protocol"]
