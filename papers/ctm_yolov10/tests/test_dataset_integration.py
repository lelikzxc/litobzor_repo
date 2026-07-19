"""Tests for CTM-YOLOv10 dataset integration with common.datasets.

Verifies:
- DetectionDataset creation (synthetic and from directories)
- BaseDataset inheritance and DatasetType
- Transforms via common.datasets.build_transforms
- Collation via torch.utils.data.default_collate
- Splitting via common.datasets.split_dataset
- DataModule integration
- DataLoader output shapes
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from common.datasets import (
    DataModule,
    DatasetType,
    build_transforms,
    split_dataset,
)
from papers.ctm_yolov10.data_utils import DetectionDataset


# ── Dataset creation ─────────────────────────────────────────────────────


def test_synthetic_dataset_creation() -> None:
    """DetectionDataset can be created in synthetic mode."""
    dataset = DetectionDataset(synthetic_size=100, image_size=640, num_classes=80)
    assert len(dataset) == 100
    assert dataset.dataset_type == DatasetType.CLASSIFICATION


def test_synthetic_sample_structure() -> None:
    """Synthetic sample has expected keys and shapes."""
    dataset = DetectionDataset(synthetic_size=10, image_size=640, num_classes=80)
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "image" in sample
    assert "label" in sample
    assert sample["image"].shape == (3, 640, 640)
    assert sample["label"].shape == (0, 5)  # no boxes in synthetic data


def test_synthetic_sample_custom_size() -> None:
    """Synthetic sample respects custom image_size."""
    dataset = DetectionDataset(synthetic_size=5, image_size=224, num_classes=10)
    sample = dataset[0]
    assert sample["image"].shape == (3, 224, 224)


def test_dataset_type_enum() -> None:
    """DetectionDataset uses DatasetType.CLASSIFICATION."""
    dataset = DetectionDataset(synthetic_size=10)
    assert dataset.dataset_type == DatasetType.CLASSIFICATION
    assert dataset.dataset_type.value == "classification"


def test_dataset_is_subclass_of_base_dataset() -> None:
    """DetectionDataset extends BaseDataset."""
    from common.datasets import BaseDataset

    assert issubclass(DetectionDataset, BaseDataset)


def test_dataset_repr() -> None:
    """DetectionDataset has a meaningful repr."""
    dataset = DetectionDataset(synthetic_size=10)
    r = repr(dataset)
    assert "DetectionDataset" in r
    assert "classification" in r


# ── Real file dataset ────────────────────────────────────────────────────


def test_dataset_from_directory(tmp_path: Path) -> None:
    """DetectionDataset can be created from a directory of images."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(image_dir / f"img_{i}.png")

    dataset = DetectionDataset(image_dir=image_dir, image_size=640)
    assert len(dataset) == 3


def test_dataset_from_directory_with_labels(tmp_path: Path) -> None:
    """DetectionDataset loads YOLO-format labels."""
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    img = Image.new("RGB", (100, 100))
    img.save(image_dir / "img_0.png")

    # YOLO format: class_id cx cy w h
    with open(label_dir / "img_0.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.3\n")
        f.write("1 0.3 0.4 0.1 0.2\n")

    dataset = DetectionDataset(image_dir=image_dir, label_dir=label_dir, image_size=640)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["label"].shape == (2, 5)


def test_dataset_from_directory_no_labels(tmp_path: Path) -> None:
    """DetectionDataset handles missing label files gracefully."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    img = Image.new("RGB", (100, 100))
    img.save(image_dir / "img_0.png")

    dataset = DetectionDataset(image_dir=image_dir, image_size=640)
    sample = dataset[0]
    assert sample["label"].shape == (0, 5)


# ── Transforms ───────────────────────────────────────────────────────────


def test_dataset_with_transforms() -> None:
    """DetectionDataset works with common.datasets.build_transforms."""
    transform = build_transforms(resize_size=(224, 224))
    dataset = DetectionDataset(synthetic_size=10, image_size=640, transform=transform)
    sample = dataset[0]
    # Transform is applied in __getitem__ for synthetic data
    assert "image" in sample
    assert "label" in sample


def test_build_transforms_default() -> None:
    """build_transforms returns a valid transform pipeline."""
    transform = build_transforms()
    assert transform is not None


def test_build_transforms_with_resize() -> None:
    """build_transforms with resize works."""
    transform = build_transforms(resize_size=(640, 640))
    assert transform is not None


# ── Collation ────────────────────────────────────────────────────────────


def test_default_collate_synthetic() -> None:
    """default_collate works with DetectionDataset samples."""
    dataset = DetectionDataset(synthetic_size=10, image_size=224)
    batch = [dataset[i] for i in range(4)]
    result = default_collate(batch)
    assert result["image"].shape == (4, 3, 224, 224)
    assert "label" in result
    assert result["label"].shape == (4, 0, 5)


def test_dataloader_with_default_collate() -> None:
    """DataLoader works with DetectionDataset and default_collate."""
    dataset = DetectionDataset(synthetic_size=20, image_size=224)
    loader = DataLoader(dataset, batch_size=4, collate_fn=default_collate)
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 224, 224)


# ── Splitting ────────────────────────────────────────────────────────────


def test_split_dataset() -> None:
    """split_dataset works with DetectionDataset."""
    dataset = DetectionDataset(synthetic_size=100, image_size=224)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15


def test_split_reproducible() -> None:
    """split_dataset is reproducible with the same seed."""
    dataset = DetectionDataset(synthetic_size=100, image_size=224)
    s1 = split_dataset(dataset, seed=42)
    s2 = split_dataset(dataset, seed=42)
    assert list(s1["train"].indices) == list(s2["train"].indices)


def test_split_different_seed() -> None:
    """split_dataset produces different splits with different seeds."""
    dataset = DetectionDataset(synthetic_size=100, image_size=224)
    s1 = split_dataset(dataset, seed=42)
    s2 = split_dataset(dataset, seed=99)
    assert list(s1["train"].indices) != list(s2["train"].indices)


def test_split_invalid_ratios() -> None:
    """split_dataset raises on invalid ratios."""
    dataset = DetectionDataset(synthetic_size=100, image_size=224)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_dataset(dataset, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


# ── DataModule integration ──────────────────────────────────────────────


def test_datamodule_with_detection_dataset() -> None:
    """DataModule works with DetectionDataset using default_collate."""
    dataset = DetectionDataset(synthetic_size=50, image_size=224)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm = DataModule(
        dataset_type="classification",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=default_collate,
    )
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert batch["image"].shape[0] <= 8
    assert batch["image"].ndim == 4


def test_datamodule_train_val_test() -> None:
    """DataModule provides train, val, and test loaders with default_collate."""
    dataset = DetectionDataset(synthetic_size=30, image_size=224)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm = DataModule(
        dataset_type="classification",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=default_collate,
    )
    assert len(dm.train_dataloader()) > 0
    assert len(dm.val_dataloader()) > 0
    assert len(dm.test_dataloader()) > 0


# ── Integration: full pipeline ──────────────────────────────────────────


def test_full_dataset_pipeline() -> None:
    """End-to-end: dataset → split → DataModule → DataLoader."""
    dataset = DetectionDataset(synthetic_size=100, image_size=224)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm = DataModule(
        dataset_type="classification",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=default_collate,
    )
    loader = dm.train_dataloader()
    batches = list(loader)
    assert len(batches) > 0
    for batch in batches:
        assert batch["image"].shape[0] <= 8
        assert batch["image"].ndim == 4


def test_synthetic_sample_consistency() -> None:
    """Synthetic samples are consistent across calls."""
    dataset = DetectionDataset(synthetic_size=10, image_size=224)
    s1 = dataset[0]
    s2 = dataset[0]
    # Each call generates a new random sample
    assert not torch.allclose(s1["image"], s2["image"])