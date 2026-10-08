"""The quick experiment must preserve disjoint real-wafer holdouts."""

from collections import Counter

import pytest

from papers.radon_cnn.diagnose_mps import reuse_holdout, select_manifest


class LabelInventory:
    image_size = 64
    radon_theta = 64

    def __init__(self):
        self._samples = [(f"class{label}_{i}.png", label)
                         for label in range(7) for i in range(100 + label * 10)]

    def __len__(self):
        return len(self._samples)


def test_reduced_diagnostic_preserves_disjoint_originals_and_all_classes():
    dataset = LabelInventory()
    manifest = select_manifest(dataset, train_per_class=16, eval_per_class=8, seed=42)
    sets = {}
    for name, count in (("train", 16), ("val", 8), ("test", 8)):
        indices = manifest["indices"][name]
        assert len(indices) == 7 * count
        assert len(set(indices)) == len(indices)
        assert Counter(dataset._samples[i][1] for i in indices) == dict.fromkeys(range(7), count)
        sets[name] = set(indices)
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    assert manifest == select_manifest(dataset, 16, 8, 42)


@pytest.mark.parametrize("train_count,eval_count", [(81, 8), (16, 11), (0, 8), (16, 0)])
def test_reduced_diagnostic_rejects_insufficient_distinct_wafers(train_count, eval_count):
    with pytest.raises(ValueError):
        select_manifest(LabelInventory(), train_count, eval_count, seed=42)


def test_larger_training_sample_can_keep_exact_same_holdouts():
    dataset = LabelInventory()
    reference = select_manifest(dataset, 16, 8, seed=42)
    larger = select_manifest(dataset, 32, 8, seed=42)
    reuse_holdout(dataset, larger, reference)
    assert larger["unique_counts"]["train"] == 7 * 32
    for name in ("val", "test"):
        assert larger["indices"][name] == reference["indices"][name]
        assert not set(larger["indices"]["train"]) & set(reference["indices"][name])


def test_holdout_comparison_refuses_reference_from_another_dataset():
    dataset = LabelInventory()
    reference = select_manifest(dataset, 16, 8, seed=42)
    current = select_manifest(dataset, 16, 8, seed=42)
    reference["dataset_sha256"] = "another dataset"
    with pytest.raises(ValueError, match="inventory"):
        reuse_holdout(dataset, current, reference)
