"""Persist splits of original wafer identities before augmentation."""

from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .dataset import CLASSES


def make_manifest(inventory, config, seed, fold=None, max_per_class=None):
    labels = np.array([label for _, label in inventory.samples])
    rng = np.random.default_rng(seed)
    pools = []
    for label in range(8):
        bucket = np.flatnonzero(labels == label)
        if max_per_class is not None:
            if max_per_class < 10:
                raise ValueError("max_per_class must be >=10 for a stratified diagnostic")
            bucket = rng.permutation(bucket)[:max_per_class]
        if len(bucket) < 10:
            raise ValueError(
                f"Class {CLASSES[label]} has only {len(bucket)} original wafers. "
                "Independent train/val/test is impossible here; use all_maps. "
                "Rotations of one wafer cannot replace independent holdout wafers."
            )
        pools.extend(bucket.tolist())
    indices = np.array(pools)
    if fold is None:
        test_fraction = config["data"]["test_fraction"]
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must lie in (0,1)")
        development, test = train_test_split(
            indices,
            test_size=test_fraction,
            stratify=labels[indices],
            random_state=seed,
        )
    else:
        if fold not in {0, 1, 2}:
            raise ValueError("fold must be 0, 1 or 2")
        splits = list(
            StratifiedKFold(3, shuffle=True, random_state=seed).split(indices, labels[indices])
        )
        development, test = (indices[i] for i in splits[fold])
    val_fraction = config["data"]["val_fraction_of_development"]
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction_of_development must lie in [0,1)")
    if val_fraction == 0:
        train, val = development, np.array([], dtype=np.int64)
    else:
        train, val = train_test_split(
            development,
            test_size=val_fraction,
            stratify=labels[development],
            random_state=seed + 1,
        )
    originals = {
        name: ids.tolist() for name, ids in (("train", train), ("val", val), ("test", test))
    }
    target = config["data"].get("train_per_class")
    minimum = config["data"].get("train_min_per_class")
    if target is not None and minimum is not None:
        raise ValueError("Choose train_per_class or train_min_per_class, not both")
    if minimum is not None:
        if not isinstance(minimum, int) or minimum < 1:
            raise ValueError("train_min_per_class must be a positive integer")
        target = minimum
    records = {name: [[int(index), 0] for index in ids] for name, ids in originals.items()}
    if target is not None:
        if not isinstance(target, int) or target < 1:
            raise ValueError("train_per_class must be null or a positive integer")
        records["train"] = []
        for label in range(8):
            bucket = train[labels[train] == label]
            if len(bucket) >= target:
                chosen = (
                    bucket if minimum is not None else rng.choice(bucket, target, replace=False)
                )
                records["train"].extend([[int(i), 0] for i in chosen])
            else:
                candidates = [[int(i), angle] for i in bucket for angle in range(0, 360, 15)]
                if target > len(candidates):
                    raise ValueError(f"Not enough distinct training rotations for {CLASSES[label]}")
                records["train"].extend([[int(i), 0] for i in bucket])
                extras = [record for record in candidates if record[1] != 0]
                selected = rng.choice(len(extras), target - len(bucket), replace=False)
                records["train"].extend([extras[i] for i in selected])
        rng.shuffle(records["train"])
    return {
        "version": 1,
        "dataset_sha256": inventory.identity,
        "seed": seed,
        "fold": fold,
        "diagnostic_max_per_class": max_per_class,
        "class_names": CLASSES,
        "data_config": config["data"],
        "model_config": config["model"],
        "original_indices": originals,
        "records": records,
        "class_counts": {
            name: dict(Counter(str(int(labels[i])) for i, _ in rows))
            for name, rows in records.items()
        },
        "split_policy": "original wafers first; augment training only; validation/test real-only",
    }


def validate_manifest(inventory, manifest, config=None):
    if manifest.get("version") != 1 or manifest.get("dataset_sha256") != inventory.identity:
        raise ValueError("Saved split does not match this wafer inventory")
    if manifest.get("class_names") != CLASSES:
        raise ValueError("Saved class order differs")
    if config is not None:
        # Data location may legitimately change between Windows and macOS.
        old = {
            k: v
            for k, v in manifest["data_config"].items()
            if k not in {"data_root", "reviewed_ids"}
        }
        new = {k: v for k, v in config["data"].items() if k not in {"data_root", "reviewed_ids"}}
        if old != new or manifest["model_config"] != config["model"]:
            raise ValueError("Preprocessing/model protocol differs from saved run")
    sets = {}
    for name in ("train", "val", "test"):
        indices = manifest["original_indices"][name]
        if name == "val" and manifest["data_config"]["val_fraction_of_development"] == 0:
            if indices or manifest["records"][name]:
                raise ValueError("Fixed-epoch protocol must have no validation samples")
            sets[name] = set()
            continue
        if not indices or len(indices) != len(set(indices)):
            raise ValueError(f"Empty or duplicated original {name} indices")
        if any(type(i) is not int or not 0 <= i < len(inventory.samples) for i in indices):
            raise ValueError("Invalid original wafer index")
        sets[name] = set(indices)
        rows = manifest["records"][name]
        if not rows or len(rows) != len({tuple(row) for row in rows}):
            raise ValueError("Empty or duplicated wafer/angle records")
        for index, angle in rows:
            if index not in sets[name] or type(angle) is not int or angle not in range(0, 360, 15):
                raise ValueError("Augmented record does not belong to its original split")
            if name != "train" and angle != 0:
                raise ValueError("Evaluation must contain only unaugmented originals")
        if {inventory.samples[i][1] for i, _ in rows} != set(range(8)):
            raise ValueError(f"Missing class in {name}")
        if name != "train" and {i for i, _ in rows} != sets[name]:
            raise ValueError("Evaluation does not cover the complete holdout")
    if any(sets[a] & sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Original-wafer leakage between splits")
