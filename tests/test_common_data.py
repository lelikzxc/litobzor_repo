"""Comprehensive tests for common.datasets infrastructure.

Uses synthetic tensors and temporary files only. No datasets downloaded. No GPU required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common.datasets import (
    BaseDataset,
    DatasetType,
    DataModule,
    build_transforms,
    classification_collate,
    segmentation_collate,
    multitask_collate,
    split_dataset,
    show_image,
    show_mask,
    overlay_mask,
    visualize_batch,
    read_image,
    read_mask,
    verify_dataset,
    count_classes,
)
from common.datasets.augmentation import build_augmentations
from common.datasets.collate import build_collate_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class DummyClassDataset(BaseDataset):
    """Minimal classification dataset for testing."""

    def __init__(self, size: int = 10, num_classes: int = 3, **kwargs) -> None:
        super().__init__(dataset_type=DatasetType.CLASSIFICATION, **kwargs)
        self.size = size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.randn(3, 32, 32),
            "label": index % self.num_classes,
        }


class DummySegDataset(BaseDataset):
    """Minimal segmentation dataset for testing."""

    def __init__(self, size: int = 10, num_classes: int = 3, **kwargs) -> None:
        super().__init__(dataset_type=DatasetType.SEGMENTATION, **kwargs)
        self.size = size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.randn(3, 32, 32),
            "mask": torch.randint(0, self.num_classes, (32, 32), dtype=torch.long),
        }


class DummyMultiDataset(BaseDataset):
    """Minimal multitask dataset for testing."""

    def __init__(self, size: int = 10, num_classes: int = 3, **kwargs) -> None:
        super().__init__(dataset_type=DatasetType.MULTITASK, **kwargs)
        self.size = size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.randn(3, 32, 32),
            "label": index % self.num_classes,
            "mask": torch.randint(0, self.num_classes, (32, 32), dtype=torch.long),
        }


@pytest.fixture
def class_dataset() -> DummyClassDataset:
    return DummyClassDataset(size=20, num_classes=3)


@pytest.fixture
def seg_dataset() -> DummySegDataset:
    return DummySegDataset(size=20, num_classes=3)


@pytest.fixture
def multi_dataset() -> DummyMultiDataset:
    return DummyMultiDataset(size=20, num_classes=3)


@pytest.fixture
def tmp_image() -> Path:
    """Create a temporary RGB image file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(f.name)
        return Path(f.name)


@pytest.fixture
def tmp_mask() -> Path:
    """Create a temporary mask file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        mask = Image.fromarray(np.random.randint(0, 3, (64, 64), dtype=np.uint8))
        mask.save(f.name)
        return Path(f.name)


# ---------------------------------------------------------------------------
# BaseDataset
# ---------------------------------------------------------------------------


class TestBaseDataset:
    def test_classification_sample(self, class_dataset: DummyClassDataset) -> None:
        sample = class_dataset[0]
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape == (3, 32, 32)
        assert isinstance(sample["label"], int)

    def test_segmentation_sample(self, seg_dataset: DummySegDataset) -> None:
        sample = seg_dataset[0]
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "mask" in sample
        assert sample["mask"].shape == (32, 32)

    def test_multitask_sample(self, multi_dataset: DummyMultiDataset) -> None:
        sample = multi_dataset[0]
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "label" in sample
        assert "mask" in sample

    def test_len(self, class_dataset: DummyClassDataset) -> None:
        assert len(class_dataset) == 20

    def test_repr(self, class_dataset: DummyClassDataset) -> None:
        r = repr(class_dataset)
        assert "DummyClassDataset" in r
        assert "classification" in r

    def test_dataset_type_enum(self) -> None:
        assert DatasetType.CLASSIFICATION.value == "classification"
        assert DatasetType.SEGMENTATION.value == "segmentation"
        assert DatasetType.MULTITASK.value == "multitask"


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class TestDataModule:
    def test_train_dataloader(self, class_dataset: DummyClassDataset) -> None:
        dm = DataModule(
            dataset_type="classification",
            train_dataset=class_dataset,
            batch_size=4,
        )
        loader = dm.train_dataloader()
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].shape[0] == 4

    def test_val_dataloader(self, class_dataset: DummyClassDataset) -> None:
        dm = DataModule(
            dataset_type="classification",
            val_dataset=class_dataset,
            batch_size=4,
        )
        loader = dm.val_dataloader()
        assert isinstance(loader, DataLoader)

    def test_test_dataloader(self, class_dataset: DummyClassDataset) -> None:
        dm = DataModule(
            dataset_type="classification",
            test_dataset=class_dataset,
            batch_size=4,
        )
        loader = dm.test_dataloader()
        assert isinstance(loader, DataLoader)

    def test_predict_dataloader(self, class_dataset: DummyClassDataset) -> None:
        dm = DataModule(
            dataset_type="classification",
            predict_dataset=class_dataset,
            batch_size=4,
        )
        loader = dm.predict_dataloader()
        assert isinstance(loader, DataLoader)

    def test_missing_train_raises(self) -> None:
        dm = DataModule(dataset_type="classification")
        with pytest.raises(ValueError, match="train_dataset"):
            dm.train_dataloader()

    def test_missing_val_raises(self) -> None:
        dm = DataModule(dataset_type="classification")
        with pytest.raises(ValueError, match="val_dataset"):
            dm.val_dataloader()

    def test_segmentation_dataloader(self, seg_dataset: DummySegDataset) -> None:
        dm = DataModule(
            dataset_type="segmentation",
            train_dataset=seg_dataset,
            batch_size=4,
        )
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "image" in batch
        assert "mask" in batch

    def test_multitask_dataloader(self, multi_dataset: DummyMultiDataset) -> None:
        dm = DataModule(
            dataset_type="multitask",
            train_dataset=multi_dataset,
            batch_size=4,
        )
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "image" in batch
        assert "label" in batch
        assert "mask" in batch

    def test_custom_collate(self, class_dataset: DummyClassDataset) -> None:
        def custom_collate(batch):
            return {"custom": True}
        dm = DataModule(
            dataset_type="classification",
            train_dataset=class_dataset,
            batch_size=4,
            collate_fn=custom_collate,
        )
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert batch["custom"] is True

    def test_loader_kwargs(self, class_dataset: DummyClassDataset) -> None:
        dm = DataModule(
            dataset_type="classification",
            train_dataset=class_dataset,
            batch_size=4,
            timeout=10,
        )
        loader = dm.train_dataloader()
        assert loader.timeout == 10


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_build_transforms_default(self) -> None:
        transform = build_transforms()
        assert transform is not None

    def test_build_transforms_with_resize(self) -> None:
        transform = build_transforms(resize_size=(64, 64))
        assert transform is not None

    def test_build_transforms_no_normalize(self) -> None:
        transform = build_transforms(normalize=False)
        assert transform is not None

    def test_build_transforms_no_tensor(self) -> None:
        transform = build_transforms(to_tensor=False)
        assert transform is not None

    def test_build_transforms_custom_mean_std(self) -> None:
        transform = build_transforms(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        assert transform is not None

    def test_build_transforms_extra(self) -> None:
        from torchvision.transforms import RandomHorizontalFlip
        transform = build_transforms(extra=[RandomHorizontalFlip(p=1.0)])
        assert transform is not None


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class TestAugmentation:
    def test_no_augmentation(self) -> None:
        aug = build_augmentations()
        assert aug is None

    def test_h_flip(self) -> None:
        aug = build_augmentations(h_flip_prob=0.5)
        assert aug is not None

    def test_v_flip(self) -> None:
        aug = build_augmentations(v_flip_prob=0.5)
        assert aug is not None

    def test_rotation(self) -> None:
        aug = build_augmentations(rotation_degrees=30)
        assert aug is not None

    def test_random_crop(self) -> None:
        aug = build_augmentations(crop_size=(28, 28))
        assert aug is not None

    def test_center_crop(self) -> None:
        aug = build_augmentations(center_crop_size=(28, 28))
        assert aug is not None

    def test_color_jitter(self) -> None:
        aug = build_augmentations(color_jitter_params={"brightness": 0.2, "contrast": 0.2})
        assert aug is not None

    def test_all_augmentations(self) -> None:
        aug = build_augmentations(
            h_flip_prob=0.5,
            v_flip_prob=0.5,
            rotation_degrees=30,
            crop_size=(28, 28),
            color_jitter_params={"brightness": 0.2},
        )
        assert aug is not None
        # Apply to a tensor
        t = torch.randn(3, 32, 32)
        result = aug(t)
        assert result.shape == (3, 28, 28)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


class TestCollate:
    def test_classification_collate(self) -> None:
        batch = [
            {"image": torch.randn(3, 32, 32), "label": 0},
            {"image": torch.randn(3, 32, 32), "label": 1},
        ]
        result = classification_collate(batch)
        assert result["image"].shape == (2, 3, 32, 32)
        assert result["label"].shape == (2,)
        assert result["label"][0] == 0
        assert result["label"][1] == 1

    def test_segmentation_collate(self) -> None:
        batch = [
            {"image": torch.randn(3, 32, 32), "mask": torch.randint(0, 3, (32, 32))},
            {"image": torch.randn(3, 32, 32), "mask": torch.randint(0, 3, (32, 32))},
        ]
        result = segmentation_collate(batch)
        assert result["image"].shape == (2, 3, 32, 32)
        assert result["mask"].shape == (2, 32, 32)

    def test_multitask_collate(self) -> None:
        batch = [
            {"image": torch.randn(3, 32, 32), "label": 0, "mask": torch.randint(0, 3, (32, 32))},
            {"image": torch.randn(3, 32, 32), "label": 1, "mask": torch.randint(0, 3, (32, 32))},
        ]
        result = multitask_collate(batch)
        assert result["image"].shape == (2, 3, 32, 32)
        assert result["label"].shape == (2,)
        assert result["mask"].shape == (2, 32, 32)

    def test_build_collate_fn(self) -> None:
        fn = build_collate_fn("classification")
        assert callable(fn)
        fn = build_collate_fn("segmentation")
        assert callable(fn)
        fn = build_collate_fn("multitask")
        assert callable(fn)

    def test_build_collate_fn_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset type"):
            build_collate_fn("invalid")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


class TestSplits:
    def test_random_split(self) -> None:
        dataset = DummyClassDataset(size=100)
        splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15

    def test_split_reproducible(self) -> None:
        dataset = DummyClassDataset(size=100)
        s1 = split_dataset(dataset, seed=42)
        s2 = split_dataset(dataset, seed=42)
        assert list(s1["train"].indices) == list(s2["train"].indices)

    def test_split_different_seed(self) -> None:
        dataset = DummyClassDataset(size=100)
        s1 = split_dataset(dataset, seed=42)
        s2 = split_dataset(dataset, seed=99)
        assert list(s1["train"].indices) != list(s2["train"].indices)

    def test_invalid_ratios(self) -> None:
        dataset = DummyClassDataset(size=100)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            split_dataset(dataset, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_stratified_split(self) -> None:
        dataset = DummyClassDataset(size=100, num_classes=3)
        labels = [i % 3 for i in range(100)]
        splits = split_dataset(
            dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            stratify=True, labels=labels,
        )
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15

    def test_stratified_missing_labels(self) -> None:
        dataset = DummyClassDataset(size=100)
        with pytest.raises(ValueError, match="labels must be provided"):
            split_dataset(dataset, stratify=True)


# ---------------------------------------------------------------------------
# Visualization (smoke tests — no display)
# ---------------------------------------------------------------------------


class TestVisualization:
    def test_show_image_creates_figure(self) -> None:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        image = torch.randn(3, 32, 32)
        fig, ax = plt.subplots()
        show_image(image, title="test", ax=ax)
        plt.close(fig)

    def test_show_mask_creates_figure(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mask = torch.randint(0, 3, (32, 32))
        fig, ax = plt.subplots()
        show_mask(mask, title="test", ax=ax)
        plt.close(fig)

    def test_overlay_mask_creates_figure(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        image = torch.randn(3, 32, 32)
        mask = torch.randint(0, 3, (32, 32))
        fig, ax = plt.subplots()
        overlay_mask(image, mask, title="test", ax=ax)
        plt.close(fig)

    def test_visualize_batch_classification(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        batch = {
            "image": torch.randn(4, 3, 32, 32),
            "label": torch.randint(0, 3, (4,)),
        }
        visualize_batch(batch, num_samples=2, task="classification")

    def test_visualize_batch_segmentation(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        batch = {
            "image": torch.randn(4, 3, 32, 32),
            "mask": torch.randint(0, 3, (4, 32, 32)),
        }
        visualize_batch(batch, num_samples=2, task="segmentation")

    def test_visualize_batch_multitask(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        batch = {
            "image": torch.randn(4, 3, 32, 32),
            "label": torch.randint(0, 3, (4,)),
            "mask": torch.randint(0, 3, (4, 32, 32)),
        }
        visualize_batch(batch, num_samples=2, task="multitask")


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


class TestReadImage:
    def test_read_image(self, tmp_image: Path) -> None:
        img = read_image(tmp_image)
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_read_image_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Image not found"):
            read_image("/nonexistent/image.png")


class TestReadMask:
    def test_read_mask(self, tmp_mask: Path) -> None:
        mask = read_mask(tmp_mask)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (64, 64)

    def test_read_mask_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Mask not found"):
            read_mask("/nonexistent/mask.png")


class TestVerifyDataset:
    def test_verify_classification(self, class_dataset: DummyClassDataset) -> None:
        issues = verify_dataset(class_dataset, num_samples=3)
        assert issues == []

    def test_verify_segmentation(self, seg_dataset: DummySegDataset) -> None:
        issues = verify_dataset(seg_dataset, num_samples=3)
        assert issues == []

    def test_verify_multitask(self, multi_dataset: DummyMultiDataset) -> None:
        issues = verify_dataset(multi_dataset, num_samples=3)
        assert issues == []


class TestCountClasses:
    def test_count_classes_classification(self, class_dataset: DummyClassDataset) -> None:
        counts = count_classes(class_dataset)
        assert isinstance(counts, dict)
        assert sum(counts.values()) == len(class_dataset)
        # 20 samples, 3 classes: 0->7, 1->7, 2->6
        assert set(counts.keys()) == {0, 1, 2}

    def test_count_classes_segmentation(self, seg_dataset: DummySegDataset) -> None:
        counts = count_classes(seg_dataset, label_key="mask")
        assert isinstance(counts, dict)
        assert set(counts.keys()).issubset({0, 1, 2})

    def test_count_classes_missing_key(self, class_dataset: DummyClassDataset) -> None:
        with pytest.raises(KeyError, match="has no key"):
            count_classes(class_dataset, label_key="nonexistent")


# ---------------------------------------------------------------------------
# Integration: DataModule + collate + dataset
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_classification_pipeline(self) -> None:
        dataset = DummyClassDataset(size=50)
        dm = DataModule(
            dataset_type="classification",
            train_dataset=dataset,
            batch_size=8,
        )
        loader = dm.train_dataloader()
        batches = list(loader)
        assert len(batches) == 7  # 50 / 8 = 6.25 → 7 batches (drop_last=False)
        for batch in batches:
            assert batch["image"].shape[0] <= 8
            assert batch["image"].ndim == 4
            assert batch["label"].ndim == 1

    def test_segmentation_pipeline(self) -> None:
        dataset = DummySegDataset(size=30)
        dm = DataModule(
            dataset_type="segmentation",
            train_dataset=dataset,
            batch_size=8,
        )
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert batch["image"].shape == (8, 3, 32, 32)
        assert batch["mask"].shape == (8, 32, 32)

    def test_multitask_pipeline(self) -> None:
        dataset = DummyMultiDataset(size=30)
        dm = DataModule(
            dataset_type="multitask",
            train_dataset=dataset,
            batch_size=8,
        )
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert batch["image"].shape == (8, 3, 32, 32)
        assert batch["label"].shape == (8,)
        assert batch["mask"].shape == (8, 32, 32)

    def test_train_val_test_split_pipeline(self) -> None:
        dataset = DummyClassDataset(size=100)
        splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        dm = DataModule(
            dataset_type="classification",
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            test_dataset=splits["test"],
            batch_size=8,
        )
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
