"""Reproducible, disjoint wafer splits shared by training and evaluation.

The paper omits exact split sizes, image interpolation, projection settings,
and the treatment of rare classes. These choices are recorded, not presented
as recovered author settings. Replacement is permitted in TRAINING only.
"""

import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
from torch.utils.data import Subset


def dataset_options(config):
    return {
        "data_root": config.get("data.data_root", "datasets/wm811k"),
        "image_size": config.get("data.image_size", 64),
        "num_classes": config.get("model.num_classes", 7),
        "balanced": False,
        "radon_theta": config.get("model.radon_theta", 64),
        "cache_radon": config.get("data.cache_radon", True),
    }


def _identity(dataset):
    return hashlib.sha256(json.dumps(dataset._samples, separators=(",", ":")).encode()).hexdigest()


def make_manifest(dataset, config, seed):
    """Partition original wafers before any balancing or oversampling."""
    val_fraction = config.get("data.val_split", 0.1)
    test_fraction = config.get("data.test_split", 0.1)
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to < 1")
    protocol = config.get("data.protocol", "full")
    if protocol not in {"full", "balanced"}:
        raise ValueError("data.protocol must be full or balanced")
    rng = np.random.default_rng(seed)
    buckets = {i: [] for i in range(7)}
    for index, (_, label) in enumerate(dataset._samples):
        buckets[label].append(index)
    pools = {name: [] for name in ("train", "val", "test")}
    for label, bucket in buckets.items():
        if len(bucket) < 3:
            raise ValueError(f"Class {label} needs at least three distinct wafers")
        indices = rng.permutation(bucket).tolist()
        nv = max(1, int(len(bucket) * val_fraction))
        nt = max(1, int(len(bucket) * test_fraction))
        if nv + nt >= len(bucket):
            raise ValueError(f"No training wafers left for class {label}")
        pools["val"].append(indices[:nv])
        pools["test"].append(indices[nv : nv + nt])
        pools["train"].append(indices[nv + nt :])

    split_indices = {}
    train_size = config.get("data.train_size", None)
    allow_replacement = config.get("data.allow_train_replacement", False)
    if protocol == "full" and train_size is not None:
        raise ValueError("data.train_size applies only to the balanced protocol")
    for name, groups in pools.items():
        if protocol == "balanced":
            total = train_size if name == "train" else None
            if total is not None and (not isinstance(total, int) or total < 7):
                raise ValueError("Balanced train_size must be an integer >= 7")
            counts = (
                [total // 7 + int(i < total % 7) for i in range(7)]
                if total is not None
                else [min(map(len, groups))] * 7
            )
            selected = []
            for group, count in zip(groups, counts, strict=True):
                if count > len(group):
                    if name != "train" or not allow_replacement:
                        raise ValueError(
                            "Not enough distinct training wafers for train_size; "
                            "reduce it or explicitly allow training replacement"
                        )
                    # Retain every available wafer, then repeat only the deficit.
                    selected.extend(group)
                    selected.extend(rng.choice(group, count - len(group)).tolist())
                else:
                    selected.extend(rng.choice(group, count, replace=False).tolist())
        else:
            selected = [i for group in groups for i in group]
        split_indices[name] = rng.permutation(selected).tolist()
    return {
        "version": 1,
        "seed": seed,
        "libraries": {name: version(name) for name in ("torch", "numpy", "scikit-image")},
        "protocol": protocol,
        "dataset_sha256": _identity(dataset),
        "preprocessing": {
            "image_size": dataset.image_size,
            "radon_theta": dataset.radon_theta,
            "defect_value": 2,
            "resize": "nearest",
            "circle": False,
        },
        "indices": split_indices,
        "counts": {
            name: dict(Counter(dataset._samples[i][1] for i in indices))
            for name, indices in split_indices.items()
        },
        "unique_counts": {name: len(set(indices)) for name, indices in split_indices.items()},
    }


def validate_manifest(dataset, manifest):
    if manifest.get("version") != 1 or manifest.get("dataset_sha256") != _identity(dataset):
        raise ValueError("Split manifest does not match this WM-811K label inventory")
    expected = {
        "image_size": dataset.image_size,
        "radon_theta": dataset.radon_theta,
        "defect_value": 2,
        "resize": "nearest",
        "circle": False,
    }
    if manifest.get("preprocessing") != expected:
        raise ValueError("Preprocessing differs from the saved experiment")
    sets = {}
    for name in ("train", "val", "test"):
        indices = manifest["indices"][name]
        if not indices or any(
            not isinstance(i, int) or i < 0 or i >= len(dataset) for i in indices
        ):
            raise ValueError(f"Invalid {name} indices in split manifest")
        if name != "train" and len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate evaluation wafers in {name}")
        if {dataset._samples[i][1] for i in indices} != set(range(7)):
            raise ValueError(f"{name} is missing a defect class")
        sets[name] = set(indices)
    if any(sets[a] & sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Wafer leakage between train/validation/test")


def load_manifest(dataset, path):
    manifest = json.loads(Path(path).read_text())
    validate_manifest(dataset, manifest)
    return manifest


def subsets(dataset, manifest):
    validate_manifest(dataset, manifest)
    return tuple(Subset(dataset, manifest["indices"][name]) for name in ("train", "val", "test"))


def model_state(checkpoint):
    """Support the common Trainer, historical wrappers, and raw state_dicts."""
    return checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
