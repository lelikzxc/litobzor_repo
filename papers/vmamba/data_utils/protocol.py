"""Deterministic paper subsets shared by training and standalone evaluation."""

from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def balanced_indices(labels, per_class=100, seed=42):
    if per_class <= 0:
        return list(range(len(labels)))
    rng = np.random.RandomState(seed)
    pools = defaultdict(list)
    for i, label in enumerate(labels):
        pools[int(label)].append(i)
    chosen = []
    for label in sorted(pools):
        pool = pools[label]
        chosen.extend(rng.choice(pool, min(per_class, len(pool)), replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def stratified_split(indices, labels, train_ratio=0.8, seed=42):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    y = np.array([labels[i] for i in indices])
    split = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train, holdout = next(split.split(np.zeros(len(indices)), y))
    return [indices[i] for i in train], [indices[i] for i in holdout]
