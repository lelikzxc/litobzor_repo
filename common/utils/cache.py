"""Disk caching utilities for expensive dataset computations.

Caches class counts / class weights and stratified split indices to disk
so they don't need to be recomputed on every training run. The cache is
keyed by a fingerprint of the inputs (labels array, split ratios, seed)
so it stays valid across runs and invalidates automatically when the
underlying data or parameters change.

Cache files are stored as ``.npz`` (NumPy) archives under a configurable
cache directory (default: ``<data_root>/cache``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _fingerprint(*parts: Any) -> str:
    """Build a stable short hash from a set of serializable parts."""
    payload = json.dumps(
        parts,
        sort_keys=True,
        default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_dir: Path, key: str, kind: str) -> Path:
    """Return the file path for a given cache key and kind."""
    return cache_dir / f"{kind}_{key}.npz"


def load_cached(
    cache_dir: str | Path,
    kind: str,
    *fingerprint_parts: Any,
) -> dict[str, np.ndarray] | None:
    """Load a cached artifact if it exists and matches the fingerprint.

    Args:
        cache_dir: Directory where cache files are stored.
        kind: Artifact kind (e.g. ``"class_counts"``, ``"split"``).
        *fingerprint_parts: Values that identify the cache entry. If any of
            these change, the cache is considered stale and ``None`` is
            returned.

    Returns:
        A dict of ``name -> np.ndarray`` if a valid cache entry exists,
        otherwise ``None``.
    """
    cache_dir = Path(cache_dir)
    key = _fingerprint(*fingerprint_parts)
    path = _cache_path(cache_dir, key, kind)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    except Exception:
        # Corrupt or unreadable cache — treat as a miss.
        return None


def save_cached(
    cache_dir: str | Path,
    kind: str,
    arrays: dict[str, np.ndarray],
    *fingerprint_parts: Any,
) -> Path:
    """Save an artifact to the cache.

    Args:
        cache_dir: Directory where cache files are stored.
        kind: Artifact kind (e.g. ``"class_counts"``, ``"split"``).
        arrays: Mapping of ``name -> np.ndarray`` to persist.
        *fingerprint_parts: Values that identify the cache entry.

    Returns:
        The path of the written cache file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _fingerprint(*fingerprint_parts)
    path = _cache_path(cache_dir, key, kind)
    np.savez(path, **arrays)
    return path


def _labels_fingerprint(labels: np.ndarray) -> str:
    """Compact fingerprint for a labels array (fast for large arrays)."""
    arr = np.ascontiguousarray(labels)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def cache_class_counts(
    labels: np.ndarray,
    num_classes: int,
    cache_dir: str | Path,
) -> list[int]:
    """Compute (or load from cache) per-class sample counts.

    Args:
        labels: Array of class labels for every sample.
        num_classes: Number of classes.
        cache_dir: Cache directory.

    Returns:
        List of ``num_classes`` counts (index = class index).
    """
    fp = _labels_fingerprint(labels)
    cached = load_cached(cache_dir, "class_counts", fp, num_classes)
    if cached is not None and "counts" in cached:
        return [int(c) for c in cached["counts"]]

    counts = [0] * num_classes
    for label in labels:
        if 0 <= int(label) < num_classes:
            counts[int(label)] += 1

    save_cached(
        cache_dir,
        "class_counts",
        {"counts": np.asarray(counts, dtype=np.int64)},
        fp,
        num_classes,
    )
    return counts


def cache_stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    cache_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (or load from cache) stratified train/val/test indices.

    Args:
        labels: Array of class labels for every sample.
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        seed: Random seed for reproducibility.
        cache_dir: Cache directory.

    Returns:
        Tuple of ``(train_idx, val_idx, test_idx)`` arrays.
    """
    fp = _labels_fingerprint(labels)
    cached = load_cached(
        cache_dir,
        "split",
        fp,
        train_ratio,
        val_ratio,
        seed,
    )
    if cached is not None and {"train", "val", "test"} <= set(cached):
        return cached["train"], cached["val"], cached["test"]

    from sklearn.model_selection import StratifiedShuffleSplit

    n = len(labels)
    train_len = int(n * train_ratio)
    val_len = int(n * val_ratio)

    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_len, random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(n), labels))

    temp_labels = labels[temp_idx]
    sss2 = StratifiedShuffleSplit(
        n_splits=1, train_size=val_len, random_state=seed + 1
    )
    val_idx, test_idx = next(sss2.split(np.zeros(len(temp_idx)), temp_labels))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    save_cached(
        cache_dir,
        "split",
        {
            "train": np.asarray(train_idx, dtype=np.int64),
            "val": np.asarray(val_idx, dtype=np.int64),
            "test": np.asarray(test_idx, dtype=np.int64),
        },
        fp,
        train_ratio,
        val_ratio,
        seed,
    )
    return train_idx, val_idx, test_idx