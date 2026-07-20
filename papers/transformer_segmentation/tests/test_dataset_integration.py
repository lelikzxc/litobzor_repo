"""Tests for SegFormer dataset integration with common.datasets.

Verifies:
- SegFormerDataset creation (synthetic and from directories)
- BaseDataset inheritance and DatasetType (SEGMENTATION)
- RGB (3-channel) image handling
- Mask structure (integer class indices, [H, W] shape)
- Transforms via common.datasets.build_transforms
- Collation via common.datasets.segmentation_collate
- Splitting via common.datasets.split_dataset
- DataModule integration
- DataLoader output shapes
- Synthetic segmentation sample structure
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    segmentation_collate,
    split_dataset,
)
from papers.transformer_segmentation.data_utils import SegFormerDataset


# ── Dataset creation ─────────────────────────────────────────────────────


def test_synthetic_dataset_creation() -> None:
    """SegFormerDataset can be created in synthetic mode."""
    dataset = SegFormerDataset(synthetic_size=50, image_size=512, num_classes=8)
    assert len(dataset) == 50
    assert dataset.dataset_type == DatasetType.SEGMENTATION


def test_synthetic_sample_structure() -> None:
    """Synthetic sample has expected keys and shapes."""
    dataset = SegFormerDataset(synthetic_size=10, image_size=512, num_classes=8)
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "image" in sample
    assert "mask" in sample
    assert sample["image"].shape == (3, 512, 512)  # RGB
    assert sample["mask"].shape == (512, 512)  # segmentation mask
    assert sample["mask"].dtype == torch.long


def test_synthetic_sample_custom_size() -> None:
    """Synthetic sample respects custom image_size."""
    dataset = SegFormerDataset(synthetic_size=5, image_size=256, num_classes=4)
    sample = dataset[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (256, 256)


def test_synthetic_sample_custom_classes() -> None:
    """Synthetic sample respects custom num_classes."""
    dataset = SegFormerDataset(synthetic_size=50, image_size=64, num_classes=4)
    sample = dataset[0]
    unique_classes = torch.unique(sample["mask"])
    assert all(c in {0, 1, 2, 3} for c in unique_classes.tolist())


def test_synthetic_sample_is_rgb() -> None:
    """Synthetic sample has 3 channels (RGB)."""
    dataset = SegFormerDataset(synthetic_size=10, image_size=64)
    sample = dataset[0]
    assert sample["image"].shape[0] == 3, "Expected RGB (3 channels)"


def test_synthetic_mask_has_integer_labels() -> None:
    """Synthetic mask contains integer class indices."""
    dataset = SegFormerDataset(synthetic_size=10, image_size=64, num_classes=8)
    sample = dataset[0]
    assert sample["mask"].dtype == torch.long
    # Values should be in [0, num_classes)
    assert sample["mask"].min() >= 0
    assert sample["mask"].max() < 8


def test_dataset_type_enum() -> None:
    """SegFormerDataset uses DatasetType.SEGMENTATION."""
    dataset = SegFormerDataset(synthetic_size=10)
    assert dataset.dataset_type == DatasetType.SEGMENTATION
    assert dataset.dataset_type.value == "segmentation"


def test_dataset_is_subclass_of_base_dataset() -> None:
    """SegFormerDataset extends BaseDataset."""
    assert issubclass(SegFormerDataset, BaseDataset)


def test_dataset_repr() -> None:
    """SegFormerDataset has a meaningful repr."""
    dataset = SegFormerDataset(synthetic_size=10)
    r = repr(dataset)
    assert "SegFormerDataset" in r
    assert "segmentation" in r


def test_synthetic_sample_consistency() -> None:
    """Synthetic samples are different across calls (random)."""
    dataset = SegFormerDataset(synthetic_size=10, image_size=64)
    s1 = dataset[0]
    s2 = dataset[0]
    # Each call generates a new random sample
    assert not torch.allclose(s1["image"], s2["image"])
    assert not torch.allclose(s1["mask"].float(), s2["mask"].float())


# ── Real file dataset ────────────────────────────────────────────────────


def test_dataset_from_directory(tmp_path: Path) -> None:
    """SegFormerDataset can be created from a directory of images."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(image_dir / f"img_{i}.png")

    dataset = SegFormerDataset(image_dir=image_dir, image_size=512)
    assert len(dataset) == 3


def test_dataset_from_directory_sample_structure(tmp_path: Path) -> None:
    """Samples from directory have correct RGB structure."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    img = Image.new("RGB", (100, 100))
    img.save(image_dir / "test.png")

    dataset = SegFormerDataset(image_dir=image_dir, image_size=512)
    sample = dataset[0]
    assert "image" in sample
    assert "mask" in sample
    # Real images are RGB
    assert sample["image"].shape[0] == 3
    # Mask should be present (fallback zero mask)
    assert sample["mask"].shape == (512, 512)
    assert sample["mask"].dtype == torch.long


def test_dataset_from_directory_with_masks(tmp_path: Path) -> None:
    """SegFormerDataset loads masks from mask_dir."""
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    img = Image.new("RGB", (100, 100))
    img.save(image_dir / "img_0.png")

    # Create a mask with some class labels
    mask_array = np.random.randint(0, 8, (100, 100), dtype=np.uint8)
    mask_img = Image.fromarray(mask_array)
    mask_img.save(mask_dir / "img_0.png")

    dataset = SegFormerDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=512,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert "mask" in sample
    assert sample["mask"].dtype == torch.long


def test_dataset_from_directory_empty(tmp_path: Path) -> None:
    """SegFormerDataset handles empty directories gracefully."""
    image_dir = tmp_path / "empty_images"
    image_dir.mkdir()

    dataset = SegFormerDataset(image_dir=image_dir, image_size=512)
    assert len(dataset) == 0


def test_dataset_from_directory_nonexistent(tmp_path: Path) -> None:
    """SegFormerDataset handles nonexistent directories gracefully."""
    dataset = SegFormerDataset(image_dir=tmp_path / "nonexistent", image_size=512)
    assert len(dataset) == 0


# ── Transforms ───────────────────────────────────────────────────────────


def test_dataset_with_transforms() -> None:
    """SegFormerDataset works with common.datasets.build_transforms."""
    transform = build_transforms(resize_size=(512, 512))
    dataset = SegFormerDataset(synthetic_size=10, image_size=512, transform=transform)
    sample = dataset[0]
    assert "image" in sample
    assert "mask" in sample


def test_build_transforms_default() -> None:
    """build_transforms returns a valid transform pipeline."""
    transform = build_transforms()
    assert transform is not None


def test_build_transforms_with_resize() -> None:
    """build_transforms with resize works."""
    transform = build_transforms(resize_size=(512, 512))
    assert transform is not None


# ── Collation ────────────────────────────────────────────────────────────


def test_segmentation_collate_synthetic() -> None:
    """segmentation_collate works with SegFormerDataset samples."""
    dataset = SegFormerDataset(synthetic_size=10, image_size=64, num_classes=8)
    batch = [dataset[i] for i in range(4)]
    result = segmentation_collate(batch)
    assert result["image"].shape == (4, 3, 64, 64)
    assert result["mask"].shape == (4, 64, 64)
    assert result["mask"].dtype == torch.long


def test_segmentation_collate_single_sample() -> None:
    """segmentation_collate works with a single sample."""
    dataset = SegFormerDataset(synthetic_size=10, image_size=64)
    batch = [dataset[0]]
    result = segmentation_collate(batch)
    assert result["image"].shape == (1, 3, 64, 64)
    assert result["mask"].shape == (1, 64, 64)


def test_dataloader_with_segmentation_collate() -> None:
    """DataLoader works with SegFormerDataset and segmentation_collate."""
    dataset = SegFormerDataset(synthetic_size=20, image_size=64, num_classes=8)
    loader = DataLoader(dataset, batch_size=4, collate_fn=segmentation_collate)
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 64, 64)
    assert batch["mask"].shape == (4, 64, 64)


def test_build_collate_fn_segmentation() -> None:
    """build_collate_fn returns segmentation_collate for 'segmentation'."""
    fn = build_collate_fn("segmentation")
    assert fn is segmentation_collate


# ── Splitting ────────────────────────────────────────────────────────────


def test_split_dataset() -> None:
    """split_dataset works with SegFormerDataset."""
    dataset = SegFormerDataset(synthetic_size=100, image_size=64, num_classes=8)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15


def test_split_reproducible() -> None:
    """split_dataset is reproducible with the same seed."""
    dataset = SegFormerDataset(synthetic_size=100, image_size=64)
    s1 = split_dataset(dataset, seed=42)
    s2 = split_dataset(dataset, seed=42)
    assert list(s1["train"].indices) == list(s2["train"].indices)


def test_split_different_seed() -> None:
    """split_dataset produces different splits with different seeds."""
    dataset = SegFormerDataset(synthetic_size=100, image_size=64)
    s1 = split_dataset(dataset, seed=42)
    s2 = split_dataset(dataset, seed=99)
    assert list(s1["train"].indices) != list(s2["train"].indices)


def test_split_invalid_ratios() -> None:
    """split_dataset raises on invalid ratios."""
    dataset = SegFormerDataset(synthetic_size=100, image_size=64)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_dataset(dataset, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


# ── DataModule integration ──────────────────────────────────────────────


def test_datamodule_with_segformer_dataset() -> None:
    """DataModule works with SegFormerDataset using segmentation_collate."""
    dataset = SegFormerDataset(synthetic_size=50, image_size=64, num_classes=8)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm = DataModule(
        dataset_type="segmentation",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=segmentation_collate,
    )
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert batch["image"].shape[0] <= 8
    assert batch["image"].ndim == 4
    assert batch["mask"].ndim == 3


def test_datamodule_train_val_test() -> None:
    """DataModule provides train, val, and test loaders with segmentation_collate."""
    dataset = SegFormerDataset(synthetic_size=30, image_size=64, num_classes=8)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm = DataModule(
        dataset_type="segmentation",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=segmentation_collate,
    )
    assert len(dm.train_dataloader()) > 0
    assert len(dm.val_dataloader()) > 0
    assert len(dm.test_dataloader()) > 0


# ── Integration: full pipeline ──────────────────────────────────────────


def test_full_dataset_pipeline() -> None:
    """End-to-end: dataset → split → DataModule → DataLoader."""
    dataset = SegFormerDataset(synthetic_size=100, image_size=64, num_classes=8)
    splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm = DataModule(
        dataset_type="segmentation",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=8,
        collate_fn=segmentation_collate,
    )
    loader = dm.train_dataloader()
    batches = list(loader)
    assert len(batches) > 0
    for batch in batches:
        assert batch["image"].shape[0] <= 8
        assert batch["image"].ndim == 4
        assert batch["mask"].ndim == 3


def test_pipeline_with_transforms() -> None:
    """End-to-end pipeline with transforms applied."""
    transform = build_transforms(resize_size=(64, 64))
    dataset = SegFormerDataset(
        synthetic_size=50,
        image_size=64,
        num_classes=8,
        transform=transform,
    )
    splits = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    dm = DataModule(
        dataset_type="segmentation",
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        batch_size=4,
        collate_fn=segmentation_collate,
    )
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 64, 64)
    assert batch["mask"].shape == (4, 64, 64)


# ── Mask-specific tests ─────────────────────────────────────────────────


def test_mask_values_in_range() -> None:
    """Mask values are within [0, num_classes)."""
    dataset = SegFormerDataset(synthetic_size=10, image_size=64, num_classes=8)
    for i in range(10):
        sample = dataset[i]
        mask = sample["mask"]
        assert mask.min() >= 0
        assert mask.max() < 8


def test_mask_is_2d() -> None:
    """Mask is a 2D tensor (no channel dimension)."""
    dataset = SegFormerDataset(synthetic_size=10, image_size=64)
    sample = dataset[0]
    assert sample["mask"].ndim == 2


def test_rgb_image_from_file(tmp_path: Path) -> None:
    """RGB images maintain 3 channels when loaded from disk."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    img = Image.new("RGB", (64, 64), color=(120, 200, 50))
    img.save(image_dir / "rgb_test.png")

    dataset = SegFormerDataset(image_dir=image_dir, image_size=64)
    sample = dataset[0]
    assert sample["image"].shape[0] == 3
    assert sample["image"].dtype == torch.float32