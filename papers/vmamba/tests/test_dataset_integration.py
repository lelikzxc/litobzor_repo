"""Tests for FCS-VMamba dataset integration with common.datasets.

Verifies:
- VMambaDataset creation (synthetic and from directories)
- BaseDataset inheritance and DatasetType
- Transforms via common.datasets.build_transforms
- Collation via common.datasets.classification_collate
- Splitting via common.datasets.split_dataset
- DataModule integration
- DataLoader output shapes
- Synthetic sample structure
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from common.datasets import (
    BaseDataset,
    DataModule,
    DatasetType,
    build_collate_fn,
    build_transforms,
    classification_collate,
    split_dataset,
)
from papers.vmamba.data_utils import VMambaDataset


# ── Dataset creation ─────────────────────────────────────────────────────


def test_synthetic_dataset_creation() -> None:
    """VMambaDataset can be created in synthetic mode."""
    dataset = VMambaDataset(synthetic_size=100, image_size=224, num_classes=8)
    assert len(dataset) == 100
    assert dataset.dataset_type == DatasetType.CLASSIFICATION


def test_synthetic_sample_structure() -> None:
    """Synthetic sample has expected keys and shapes."""
    dataset = VMambaDataset(synthetic_size=10, image_size=224, num_classes=8)
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "image" in sample
    assert "label" in sample
    assert sample["image"].shape == (3, 224, 224)
    assert isinstance(sample["label"], int)
    assert 0 <= sample["label"] < 8


def test_synthetic_sample_custom_size() -> None:
    """Synthetic sample respects custom image_size."""
    dataset = VMambaDataset(synthetic_size=5, image_size=128, num_classes=4)
    sample = dataset[0]
    assert sample["image"].shape == (3, 128, 128)


def test_synthetic_sample_custom_classes() -> None:
    """Synthetic sample respects custom num_classes."""
    dataset = VMambaDataset(synthetic_size=50, image_size=224, num_classes=4)
    labels = {dataset[i]["label"] for i in range(50)}
    assert labels.issubset({0, 1, 2, 3})


def test_dataset_type_enum() -> None:
    """VMambaDataset uses DatasetType.CLASSIFICATION."""
    dataset = VMambaDataset(synthetic_size=10)
    assert dataset.dataset_type == DatasetType.CLASSIFICATION
    assert dataset.dataset_type.value == "classification"


def test_dataset_is_subclass_of_base_dataset() -> None:
    """VMambaDataset extends BaseDataset."""
    assert issubclass(VMambaDataset, BaseDataset)


def test_dataset_repr() -> None:
    """VMambaDataset has a meaningful repr."""
    dataset = VMambaDataset(synthetic_size=10)
    r = repr(dataset)
    assert "VMambaDataset" in r
    assert "classification" in r


def test_synthetic_sample_consistency() -> None:
    """Synthetic samples are different across calls (random)."""
    dataset = VMambaDataset(synthetic_size=10, image_size=224)
    s1 = dataset[0]
    s2 = dataset[0]
    # Each call generates a new random sample
    assert not torch.allclose(s1["image"], s2["image"])


# ── Real file dataset ────────────────────────────────────────────────────


def test_dataset_from_directory(tmp_path: Path) -> None:
    """VMambaDataset can be created from a directory of images."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(image_dir / f"img_{i}.png")

    dataset = VMambaDataset(image_dir=image_dir, image_size=224)
    assert len(dataset) == 3


def test_dataset_from_directory_sample_structure(tmp_path: Path) -> None:
    """Samples from directory have correct structure."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    img = Image.new("RGB", (100, 100))
    img.save(image_dir / "test.png")

    dataset = VMambaDataset(image_dir=image_dir, image_size=224)
    sample = dataset[0]
    assert "image" in sample
    assert "label" in sample
    assert isinstance(sample["label"], int)


def test_dataset_from_directory_empty(tmp_path: Path) -> None:
    """VMambaDataset handles empty directories gracefully."""
    image_dir = tmp_path / "empty_images"
    image_dir.mkdir()

    dataset = VMambaDataset(image_dir=image_dir, image_size=224)
    assert len(dataset) == 0


def test_dataset_from_directory_nonexistent(tmp_path: Path) -> None:
    """VMambaDataset handles nonexistent directories gracefully."""
    dataset = VMambaDataset(image_dir=tmp_path / "nonexistent", image_size=224)
    assert len(dataset) == 0


# ── Transforms ───────────────────────────────────────────────────────────


def test_dataset_with_transforms() -> None:
    """VMambaDataset works with common.datasets.build_transforms."""
    transform = build_transforms(resize_size=(224, 224))
    dataset = VMambaDataset(synthetic_size=10, image_size=224, transform=transform)
    sample = dataset[0]
    assert "image" in sample
    assert "label" in sample


def test_build_transforms_default() -> None:
    """build_transforms returns a valid transform pipeline."""
    transform = build_transforms()
    assert transform is not None


def test_build_transforms_with_resize() -> None:
    """build_transforms with resize works."""
    transform = build_transforms(resize_size=(224, 224))
    assert transform is not None


# ── Collation ────────────────────────────────────────────────────────────


def test_classification_collate_synthetic() -> None:
    """classification_collate works with VMambaDataset samples."""
    dataset = VMambaDataset(synthetic_size=10, image_size=224, num_classes=8)
    batch = [dataset[i] for i in range(4)]
    result = classification_collate(batch)
    assert result["image"].shape == (4, 3, 224, 224)
    assert result["label"].shape == (4,)
    assert result["label"].dtype == torch.long


def test_classification_collate_single_sample() -> None:
    """classification_collate works with a single sample."""
    dataset = VMambaDataset(synthetic_size=10, image_size=224)
    batch = [dataset[0]]
    result = classification_collate(batch)
    assert result["image"].shape == (1, 3, 224, 224)
    assert result["label"].shape == (1,)


def test_dataloader_with_classification_collate() -> None:
    """DataLoader works with VMambaDataset and classification_collate."""
    dataset = VMambaDataset(synthetic_size=20, image_size=224, num_classes=8)
    loader = DataLoader(dataset, batch_size=4, collate_fn=classification_collate)
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["label"].shape == (4,)


def test_build_collate_fn_classification() -> None:
    """build_collate_fn returns classification_collate for 'classification'."""
    fn = build_collate_fn("classification")
    assert fn is classification_collate


# ── Splitting ────────────────────────────────────────────────────────────


def test_split_dataset() -> None:
    """split_dataset works with VMambaDataset."""
    dataset = VMambaDataset(synthetic_size=100, image_size=224, num_classes=8)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15


def test_split_reproducible() -> None:
    """split_dataset is reproducible with the same seed."""
    dataset = VMambaDataset(synthetic_size=100, image_size=224)
    s1 = split_dataset(dataset, seed=42)
    s2 = split_dataset(dataset, seed=42)
    assert list(s1["train"].indices) == list(s2["train"].indices)


def test_split_different_seed() -> None:
    """split_dataset produces different splits with different seeds."""
    dataset = VMambaDataset(synthetic_size=100, image_size=224)
    s1 = split_dataset(dataset, seed=42)
    s2 = split_dataset(dataset, seed=99)
    assert list(s1["train"].indices) != list(s2["train"].indices)


def test_split_invalid_ratios() -> None:
    """split_dataset raises on invalid ratios."""
    dataset = VMambaDataset(synthetic_size=100, image_size=224)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_dataset(dataset, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


# ── DataModule integration ──────────────────────────────────────────────


def test_datamodule_with_vmamba_dataset() -> None:
    """DataModule works with VMambaDataset using classification_collate."""
    dataset = VMambaDataset(synthetic_size=50, image_size=224, num_classes=8)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm = DataModule(
        dataset_type="classification",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=classification_collate,
    )
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert batch["image"].shape[0] <= 8
    assert batch["image"].ndim == 4
    assert batch["label"].ndim == 1


def test_datamodule_train_val_test() -> None:
    """DataModule provides train, val, and test loaders with classification_collate."""
    dataset = VMambaDataset(synthetic_size=30, image_size=224, num_classes=8)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm = DataModule(
        dataset_type="classification",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=classification_collate,
    )
    assert len(dm.train_dataloader()) > 0
    assert len(dm.val_dataloader()) > 0
    assert len(dm.test_dataloader()) > 0


# ── Integration: full pipeline ──────────────────────────────────────────


def test_full_dataset_pipeline() -> None:
    """End-to-end: dataset → split → DataModule → DataLoader."""
    dataset = VMambaDataset(synthetic_size=100, image_size=224, num_classes=8)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm = DataModule(
        dataset_type="classification",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=classification_collate,
    )
    loader = dm.train_dataloader()
    batches = list(loader)
    assert len(batches) > 0
    for batch in batches:
        assert batch["image"].shape[0] <= 8
        assert batch["image"].ndim == 4
        assert batch["label"].ndim == 1


def test_pipeline_with_transforms() -> None:
    """End-to-end pipeline with transforms applied."""
    transform = build_transforms(resize_size=(224, 224))
    dataset = VMambaDataset(
        synthetic_size=50,
        image_size=224,
        num_classes=8,
        transform=transform,
    )
    splits = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    dm = DataModule(
        dataset_type="classification",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=4,
        collate_fn=classification_collate,
    )
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["label"].shape == (4,)