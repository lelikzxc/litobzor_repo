"""Run provenance for the two paper reproduction entry points."""

import hashlib
import json
from pathlib import Path

import torch


def _difference_path(previous, current, path):
    """Locate the first genuine mismatch without dumping large sample lists."""
    if previous == current:
        return None
    if isinstance(previous, dict) and isinstance(current, dict):
        for key in sorted(previous.keys() | current.keys()):
            child = f"{path}.{key}"
            if key not in previous or key not in current:
                return child
            difference = _difference_path(previous[key], current[key], child)
            if difference is not None:
                return difference
    elif isinstance(previous, list) and isinstance(current, list):
        if len(previous) != len(current):
            return f"{path}.length"
        for index, (old, new) in enumerate(zip(previous, current, strict=True)):
            difference = _difference_path(old, new, f"{path}[{index}]")
            if difference is not None:
                return difference
    return path


def save_protocol(directory, config, samples, splits, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "torch_version": torch.__version__,
        "config": config.to_dict(),
        "sample_digest": hashlib.sha256(json.dumps(samples).encode()).hexdigest(),
        "splits": splits,
    }
    # Compare the same representation on both sides. JSON restores tuples as
    # lists (notably SemiWaferNet's train_samples and pseudo_eval), so comparing
    # a loaded manifest directly to Python tuples rejects an identical restart.
    payload = json.loads(json.dumps(payload))
    path = directory / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        differences = [
            difference
            for key in ("config", "sample_digest", "splits", "seed")
            if (difference := _difference_path(old.get(key), payload[key], key)) is not None
        ]
        if differences:
            raise ValueError(
                f"Run protocol differs from {path}: {', '.join(differences)}; "
                "use a new checkpoint.save_dir for a different experiment"
            )
        # A matching restart must preserve the original experiment provenance.
        return path
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_holdout(checkpoint, samples, split="holdout"):
    path = Path(checkpoint).parent / "protocol.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: evaluation requires the training run's split")
    payload = json.loads(path.read_text())
    digest = hashlib.sha256(json.dumps(samples).encode()).hexdigest()
    if digest != payload["sample_digest"]:
        raise ValueError("Dataset content/order differs from the saved training protocol")
    return payload["splits"][split], payload["config"]
