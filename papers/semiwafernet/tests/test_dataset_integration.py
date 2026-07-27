"""Tests for SemiWaferNet dataset integration with common.datasets.

Verifies:
- LabeledWaferDataset creation (synthetic and from directories)
- UnlabeledWaferDataset creation (synthetic and from directories)
- BaseDataset inheritance and DatasetType
- Grayscale (1-channel) image handling
- Mask structure (integer class indices, [H, W] shape)
- Label structure (int)
- Transforms via common.datasets.build_transforms
- Collation via common.datasets.multitask_collate
- Splitting via common.datasets.split_dataset
- DataModule integration
- DataLoader output shapes
- Synthetic multitask batch
- Semi-supervised pipeline (labeled + unlabeled)
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
    multitask_collate,
    split_dataset,
)
from papers.semiwafernet.data_utils import LabeledWaferDataset, UnlabeledWaferDataset


# ═════════════════════════════════════════════════════════════════════════
#  LabeledWaferDataset
# ═════════════════════════════════════════════════════════════════════════


class TestLabeledDataset:
    """Tests for LabeledWaferDataset."""

    def test_synthetic_creation(self) -> None:
        """LabeledWaferDataset can be created in synthetic mode."""
        dataset = LabeledWaferDataset(synthetic_size=50, image_size=32, num_classes=9)
        assert len(dataset) == 50
        assert dataset.dataset_type == DatasetType.MULTITASK

    def test_synthetic_sample_structure(self) -> None:
        """Synthetic sample has expected keys and shapes."""
        dataset = LabeledWaferDataset(synthetic_size=10, image_size=32, num_classes=9)
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "label" in sample
        assert "mask" in sample
        assert sample["image"].shape == (1, 32, 32)  # grayscale
        assert isinstance(sample["label"], int)
        assert 0 <= sample["label"] < 9
        assert sample["mask"].shape == (32, 32)  # segmentation mask
        assert sample["mask"].dtype == torch.long

    def test_synthetic_sample_custom_size(self) -> None:
        """Synthetic sample respects custom image_size."""
        dataset = LabeledWaferDataset(synthetic_size=5, image_size=64, num_classes=4)
        sample = dataset[0]
        assert sample["image"].shape == (1, 64, 64)
        assert sample["mask"].shape == (64, 64)

    def test_synthetic_sample_custom_classes(self) -> None:
        """Synthetic sample respects custom num_classes."""
        dataset = LabeledWaferDataset(synthetic_size=50, image_size=64, num_classes=4)
        sample = dataset[0]
        assert 0 <= sample["label"] < 4
        unique_mask = torch.unique(sample["mask"])
        assert all(c in {0, 1, 2, 3} for c in unique_mask.tolist())

    def test_synthetic_sample_is_grayscale(self) -> None:
        """Synthetic sample has 1 channel (grayscale)."""
        dataset = LabeledWaferDataset(synthetic_size=10, image_size=32)
        sample = dataset[0]
        assert sample["image"].shape[0] == 1

    def test_synthetic_mask_has_integer_labels(self) -> None:
        """Synthetic mask contains integer class indices."""
        dataset = LabeledWaferDataset(synthetic_size=10, image_size=32, num_classes=9)
        sample = dataset[0]
        assert sample["mask"].dtype == torch.long
        assert sample["mask"].min() >= 0
        assert sample["mask"].max() < 9

    def test_dataset_type_enum(self) -> None:
        """LabeledWaferDataset uses DatasetType.MULTITASK."""
        dataset = LabeledWaferDataset(synthetic_size=10)
        assert dataset.dataset_type == DatasetType.MULTITASK
        assert dataset.dataset_type.value == "multitask"

    def test_is_subclass_of_base_dataset(self) -> None:
        """LabeledWaferDataset extends BaseDataset."""
        assert issubclass(LabeledWaferDataset, BaseDataset)

    def test_repr(self) -> None:
        """LabeledWaferDataset has a meaningful repr."""
        dataset = LabeledWaferDataset(synthetic_size=10)
        r = repr(dataset)
        assert "LabeledWaferDataset" in r
        assert "multitask" in r

    def test_synthetic_sample_consistency(self) -> None:
        """Synthetic samples are different across calls (random)."""
        dataset = LabeledWaferDataset(synthetic_size=10, image_size=32)
        s1 = dataset[0]
        s2 = dataset[0]
        assert not torch.allclose(s1["image"], s2["image"])
        assert not torch.allclose(s1["mask"].float(), s2["mask"].float())

    # ── Real file dataset ────────────────────────────────────────────

    def test_from_directory(self, tmp_path: Path) -> None:
        """LabeledWaferDataset can be created from a directory of images."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for i in range(3):
            img = Image.new("L", (100, 100), color=i * 50)  # grayscale
            img.save(image_dir / f"img_{i}.png")

        dataset = LabeledWaferDataset(image_dir=image_dir, image_size=32)
        assert len(dataset) == 3

    def test_from_directory_sample_structure(self, tmp_path: Path) -> None:
        """Samples from directory have correct structure."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        img = Image.new("L", (100, 100))
        img.save(image_dir / "test.png")

        dataset = LabeledWaferDataset(image_dir=image_dir, image_size=32)
        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert "mask" in sample
        assert sample["image"].shape[0] == 1
        assert isinstance(sample["label"], int)
        assert sample["mask"].shape == (32, 32)

    def test_from_directory_with_masks(self, tmp_path: Path) -> None:
        """LabeledWaferDataset loads masks from mask_dir."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        img = Image.new("L", (100, 100))
        img.save(image_dir / "img_0.png")

        mask_array = np.random.randint(0, 9, (100, 100), dtype=np.uint8)
        mask_img = Image.fromarray(mask_array)
        mask_img.save(mask_dir / "img_0.png")

        dataset = LabeledWaferDataset(
            image_dir=image_dir, mask_dir=mask_dir, image_size=32,
        )
        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["mask"].dtype == torch.long

    def test_from_directory_empty(self, tmp_path: Path) -> None:
        """LabeledWaferDataset handles empty directories."""
        image_dir = tmp_path / "empty_images"
        image_dir.mkdir()
        dataset = LabeledWaferDataset(image_dir=image_dir, image_size=32)
        assert len(dataset) == 0

    def test_from_directory_nonexistent(self, tmp_path: Path) -> None:
        """LabeledWaferDataset handles nonexistent directories."""
        dataset = LabeledWaferDataset(image_dir=tmp_path / "nonexistent", image_size=32)
        assert len(dataset) == 0

    # ── Transforms ───────────────────────────────────────────────────

    def test_with_transforms(self) -> None:
        """LabeledWaferDataset works with build_transforms."""
        transform = build_transforms(resize_size=(32, 32))
        dataset = LabeledWaferDataset(
            synthetic_size=10, image_size=32, num_classes=9, transform=transform,
        )
        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert "mask" in sample

    # ── Collation ────────────────────────────────────────────────────

    def test_multitask_collate_synthetic(self) -> None:
        """multitask_collate works with LabeledWaferDataset samples."""
        dataset = LabeledWaferDataset(synthetic_size=10, image_size=32, num_classes=9)
        batch = [dataset[i] for i in range(4)]
        result = multitask_collate(batch)
        assert result["image"].shape == (4, 1, 32, 32)
        assert result["label"].shape == (4,)
        assert result["label"].dtype == torch.long
        assert result["mask"].shape == (4, 32, 32)
        assert result["mask"].dtype == torch.long

    def test_multitask_collate_single_sample(self) -> None:
        """multitask_collate works with a single sample."""
        dataset = LabeledWaferDataset(synthetic_size=10, image_size=32)
        batch = [dataset[0]]
        result = multitask_collate(batch)
        assert result["image"].shape == (1, 1, 32, 32)
        assert result["label"].shape == (1,)
        assert result["mask"].shape == (1, 32, 32)

    def test_dataloader_with_multitask_collate(self) -> None:
        """DataLoader works with LabeledWaferDataset and multitask_collate."""
        dataset = LabeledWaferDataset(synthetic_size=20, image_size=32, num_classes=9)
        loader = DataLoader(dataset, batch_size=4, collate_fn=multitask_collate)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 1, 32, 32)
        assert batch["label"].shape == (4,)
        assert batch["mask"].shape == (4, 32, 32)

    def test_build_collate_fn_multitask(self) -> None:
        """build_collate_fn returns multitask_collate for 'multitask'."""
        fn = build_collate_fn("multitask")
        assert fn is multitask_collate

    # ── Splitting ────────────────────────────────────────────────────

    def test_split_dataset(self) -> None:
        """split_dataset works with LabeledWaferDataset."""
        dataset = LabeledWaferDataset(synthetic_size=100, image_size=32, num_classes=9)
        splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15

    def test_split_reproducible(self) -> None:
        """split_dataset is reproducible with the same seed."""
        dataset = LabeledWaferDataset(synthetic_size=100, image_size=32)
        s1 = split_dataset(dataset, seed=42)
        s2 = split_dataset(dataset, seed=42)
        assert list(s1["train"].indices) == list(s2["train"].indices)

    def test_split_different_seed(self) -> None:
        """split_dataset produces different splits with different seeds."""
        dataset = LabeledWaferDataset(synthetic_size=100, image_size=32)
        s1 = split_dataset(dataset, seed=42)
        s2 = split_dataset(dataset, seed=99)
        assert list(s1["train"].indices) != list(s2["train"].indices)

    def test_split_invalid_ratios(self) -> None:
        """split_dataset raises on invalid ratios."""
        dataset = LabeledWaferDataset(synthetic_size=100, image_size=32)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            split_dataset(dataset, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    # ── DataModule ───────────────────────────────────────────────────

    def test_datamodule_integration(self) -> None:
        """DataModule works with LabeledWaferDataset using multitask_collate."""
        dataset = LabeledWaferDataset(synthetic_size=50, image_size=32, num_classes=9)
        splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        dm = DataModule(
            dataset_type="multitask",
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            test_dataset=splits["test"],
            batch_size=8,
            collate_fn=multitask_collate,
        )
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert batch["image"].shape[0] <= 8
        assert batch["image"].ndim == 4
        assert batch["label"].ndim == 1
        assert batch["mask"].ndim == 3

    def test_datamodule_train_val_test(self) -> None:
        """DataModule provides train, val, and test loaders."""
        dataset = LabeledWaferDataset(synthetic_size=30, image_size=32, num_classes=9)
        splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        dm = DataModule(
            dataset_type="multitask",
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            test_dataset=splits["test"],
            batch_size=8,
            collate_fn=multitask_collate,
        )
        assert len(dm.train_dataloader()) > 0
        assert len(dm.val_dataloader()) > 0
        assert len(dm.test_dataloader()) > 0

    # ── Full pipeline ────────────────────────────────────────────────

    def test_full_pipeline(self) -> None:
        """End-to-end: dataset → split → DataModule → DataLoader."""
        dataset = LabeledWaferDataset(synthetic_size=100, image_size=32, num_classes=9)
        splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        dm = DataModule(
            dataset_type="multitask",
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            test_dataset=splits["test"],
            batch_size=8,
            collate_fn=multitask_collate,
        )
        loader = dm.train_dataloader()
        batches = list(loader)
        assert len(batches) > 0
        for batch in batches:
            assert batch["image"].shape[0] <= 8
            assert batch["image"].ndim == 4
            assert batch["label"].ndim == 1
            assert batch["mask"].ndim == 3

    def test_pipeline_with_transforms(self) -> None:
        """End-to-end pipeline with transforms applied."""
        transform = build_transforms(resize_size=(32, 32))
        dataset = LabeledWaferDataset(
            synthetic_size=50, image_size=32, num_classes=9, transform=transform,
        )
        splits = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        dm = DataModule(
            dataset_type="multitask",
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            test_dataset=splits["test"],
            batch_size=4,
            collate_fn=multitask_collate,
        )
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 1, 32, 32)
        assert batch["label"].shape == (4,)
        assert batch["mask"].shape == (4, 32, 32)


# ═════════════════════════════════════════════════════════════════════════
#  UnlabeledWaferDataset
# ═════════════════════════════════════════════════════════════════════════


class TestUnlabeledDataset:
    """Tests for UnlabeledWaferDataset."""

    def test_synthetic_creation(self) -> None:
        """UnlabeledWaferDataset can be created in synthetic mode."""
        dataset = UnlabeledWaferDataset(synthetic_size=100, image_size=32)
        assert len(dataset) == 100

    def test_synthetic_sample_structure(self) -> None:
        """Synthetic unlabeled sample has only image key."""
        dataset = UnlabeledWaferDataset(synthetic_size=10, image_size=32)
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "label" not in sample
        assert "mask" not in sample
        assert sample["image"].shape == (1, 32, 32)

    def test_synthetic_sample_custom_size(self) -> None:
        """Synthetic unlabeled sample respects custom image_size."""
        dataset = UnlabeledWaferDataset(synthetic_size=5, image_size=64)
        sample = dataset[0]
        assert sample["image"].shape == (1, 64, 64)

    def test_is_subclass_of_base_dataset(self) -> None:
        """UnlabeledWaferDataset extends BaseDataset."""
        assert issubclass(UnlabeledWaferDataset, BaseDataset)

    def test_repr(self) -> None:
        """UnlabeledWaferDataset has a meaningful repr."""
        dataset = UnlabeledWaferDataset(synthetic_size=10)
        r = repr(dataset)
        assert "UnlabeledWaferDataset" in r

    def test_from_directory(self, tmp_path: Path) -> None:
        """UnlabeledWaferDataset can be created from a directory."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for i in range(3):
            img = Image.new("L", (100, 100))
            img.save(image_dir / f"img_{i}.png")
        dataset = UnlabeledWaferDataset(image_dir=image_dir, image_size=32)
        assert len(dataset) == 3

    def test_from_directory_empty(self, tmp_path: Path) -> None:
        """UnlabeledWaferDataset handles empty directories."""
        image_dir = tmp_path / "empty"
        image_dir.mkdir()
        dataset = UnlabeledWaferDataset(image_dir=image_dir, image_size=32)
        assert len(dataset) == 0

    def test_from_directory_nonexistent(self, tmp_path: Path) -> None:
        """UnlabeledWaferDataset handles nonexistent directories."""
        dataset = UnlabeledWaferDataset(image_dir=tmp_path / "nonexistent", image_size=32)
        assert len(dataset) == 0

    def test_dataloader_with_unlabeled(self) -> None:
        """DataLoader works with UnlabeledWaferDataset."""
        dataset = UnlabeledWaferDataset(synthetic_size=20, image_size=32)
        loader = DataLoader(dataset, batch_size=4)
        batch = next(iter(loader))
        # Without custom collate, default_collate stacks the dicts
        assert batch["image"].shape == (4, 1, 32, 32)


# ═════════════════════════════════════════════════════════════════════════
#  Semi-supervised pipeline
# ═════════════════════════════════════════════════════════════════════════


class TestSemiSupervisedPipeline:
    """Tests for semi-supervised pipeline combining labeled and unlabeled data."""

    def test_labeled_and_unlabeled_separate(self) -> None:
        """Labeled and unlabeled datasets can be created independently."""
        labeled = LabeledWaferDataset(synthetic_size=30, image_size=32, num_classes=9)
        unlabeled = UnlabeledWaferDataset(synthetic_size=70, image_size=32)
        assert len(labeled) == 30
        assert len(unlabeled) == 70

    def test_labeled_sample_has_all_keys(self) -> None:
        """Labeled sample has image, label, and mask."""
        labeled = LabeledWaferDataset(synthetic_size=10, image_size=32, num_classes=9)
        sample = labeled[0]
        assert set(sample.keys()) == {"image", "label", "mask"}

    def test_unlabeled_sample_has_image_only(self) -> None:
        """Unlabeled sample has only image."""
        unlabeled = UnlabeledWaferDataset(synthetic_size=10, image_size=32)
        sample = unlabeled[0]
        assert set(sample.keys()) == {"image"}

    def test_separate_dataloaders(self) -> None:
        """Labeled and unlabeled data can be loaded in separate DataLoaders."""
        from torch.utils.data._utils.collate import default_collate

        labeled = LabeledWaferDataset(synthetic_size=30, image_size=32, num_classes=9)
        unlabeled = UnlabeledWaferDataset(synthetic_size=70, image_size=32)

        splits_l = split_dataset(labeled, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        splits_u = split_dataset(unlabeled, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        dm_l = DataModule(
            dataset_type="multitask",
            train_dataset=splits_l["train"],
            val_dataset=splits_l["val"],
            test_dataset=splits_l["test"],
            batch_size=4,
            collate_fn=multitask_collate,
        )
        # Unlabeled data has no "label" key — use default_collate
        dm_u = DataModule(
            dataset_type="classification",
            train_dataset=splits_u["train"],
            val_dataset=splits_u["val"],
            test_dataset=splits_u["test"],
            batch_size=4,
            collate_fn=default_collate,
        )

        labeled_batch = next(iter(dm_l.train_dataloader()))
        unlabeled_batch = next(iter(dm_u.train_dataloader()))

        assert labeled_batch["image"].shape[0] <= 4
        assert labeled_batch["label"].shape[0] <= 4
        assert labeled_batch["mask"].shape[0] <= 4
        assert unlabeled_batch["image"].shape[0] <= 4

    def test_zipped_dataloaders(self) -> None:
        """Labeled and unlabeled DataLoaders can be zipped for training."""
        from torch.utils.data._utils.collate import default_collate

        labeled = LabeledWaferDataset(synthetic_size=20, image_size=32, num_classes=9)
        unlabeled = UnlabeledWaferDataset(synthetic_size=20, image_size=32)

        splits_l = split_dataset(labeled, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
        splits_u = split_dataset(unlabeled, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

        dm_l = DataModule(
            dataset_type="multitask",
            train_dataset=splits_l["train"],
            batch_size=4,
            collate_fn=multitask_collate,
        )
        # Unlabeled data has no "label" key — use default_collate
        dm_u = DataModule(
            dataset_type="classification",
            train_dataset=splits_u["train"],
            batch_size=4,
            collate_fn=default_collate,
        )

        for labeled_batch, unlabeled_batch in zip(dm_l.train_dataloader(),
                                                   dm_u.train_dataloader()):
            assert "image" in labeled_batch
            assert "label" in labeled_batch
            assert "mask" in labeled_batch
            assert "image" in unlabeled_batch
            assert "label" not in unlabeled_batch
            assert "mask" not in unlabeled_batch
            break  # just test one iteration