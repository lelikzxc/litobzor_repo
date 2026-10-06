"""Run provenance for the two paper reproduction entry points."""

import hashlib
import json
from pathlib import Path

import torch


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
    path = directory / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[key] != payload[key] for key in ("config", "sample_digest", "splits", "seed")):
            raise ValueError(f"Run protocol differs from {path}; use a new checkpoint.save_dir")
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
