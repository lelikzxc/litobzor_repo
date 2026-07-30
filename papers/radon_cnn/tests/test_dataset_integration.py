"""Tests for RadonCNN dataset integration.

Tests cover:
    - WaferRadonDataset creation (synthetic and from directory)
    - Sample structure
    - Background removal
    - 7-class filtering
    - Balanced sampling
    - DataLoader integration
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from papers.radon_cnn.data_utils.base import BaseDataset, DatasetType
from papers.radon_cnn.data_utils.dataset import (
    RADONCNN_CLASSES,
    WM811K_TO_RADONCNN,
    WaferRadonDataset,
    remove_background,
)


# ── Synthetic Dataset Fixture ─────────────────────────────────────────────

@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> WaferRadonDataset:
    """Create a synthetic WaferRadonDataset with temporary files."""
    # Create directory structure
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create labels.csv with 7-class samples
    labels_path = tmp_path / "labels.csv"
    with open(labels_path, "w") as f:
        f.write("image,label\n")
        # Add samples for each of the 7 classes
        for class_idx in [1, 2, 3, 4, 5, 7, 8]:  # WM-811K indices
            for i in range(5):
                f.write(f"wafer_{class_idx}_{i}.png,{class_idx}\n")

    # Create dummy PNG images
    import numpy as np
    from PIL import Image

    for class_idx in [1, 2, 3, 4, 5, 7, 8]:
        for i in range(5):
            img = np.random.randint(0, 2, size=(64, 64)).astype(np.uint8) * 255
            img_path = images_dir / f"wafer_{class_idx}_{i}.png"
            Image.fromarray(img).save(img_path)

    return WaferRadonDataset(
        data_root=str(tmp_path),
        image_size=64,
        num_classes=7,
        balanced=False,
        apply_radon=False,  # Synthetic images, no Radon needed
    )


# ── Tests ─────────────────────────────────────────────────────────────────

class TestRemoveBackground:
    """Tests for background removal."""

    def test_remove_background_zeros(self) -> None:
        import numpy as np
        arr = np.zeros((10, 10), dtype=np.float32)
        result = remove_background(arr)
        assert (result == 0).all()

    def test_remove_background_ones(self) -> None:
        import numpy as np
        arr = np.ones((10, 10), dtype=np.float32)
        result = remove_background(arr)
        assert (result == 1).all()

    def test_remove_background_mixed(self) -> None:
        import numpy as np
        arr = np.array([[0, 1, 2], [0, 0, 1], [2, 0, 0]], dtype=np.float32)
        result = remove_background(arr)
        assert (result[0] == [0, 1, 1]).all()
        assert (result[1] == [0, 0, 1]).all()
        assert (result[2] == [1, 0, 0]).all()

    def test_remove_background_shape(self) -> None:
        import numpy as np
        arr = np.random.rand(64, 64).astype(np.float32)
        result = remove_background(arr)
        assert result.shape == (64, 64)


class TestWM811KMapping:
    """Tests for WM-811K to RadonCNN label mapping."""

    def test_mapping_center(self) -> None:
        assert WM811K_TO_RADONCNN[1] == 0  # Center

    def test_mapping_donut(self) -> None:
        assert WM811K_TO_RADONCNN[2] == 1  # Donut

    def test_mapping_edge_loc(self) -> None:
        assert WM811K_TO_RADONCNN[3] == 2  # Edge-Loc

    def test_mapping_edge_ring(self) -> None:
        assert WM811K_TO_RADONCNN[4] == 3  # Edge-Ring

    def test_mapping_loc(self) -> None:
        assert WM811K_TO_RADONCNN[5] == 4  # Loc

    def test_mapping_random(self) -> None:
        assert WM811K_TO_RADONCNN[7] == 5  # Random

    def test_mapping_scratch(self) -> None:
        assert WM811K_TO_RADONCNN[8] == 6  # Scratch

    def test_excluded_none(self) -> None:
        assert 0 not in WM811K_TO_RADONCNN  # None excluded

    def test_excluded_near_full(self) -> None:
        assert 6 not in WM811K_TO_RADONCNN  # Near-full excluded

    def test_class_count(self) -> None:
        assert len(WM811K_TO_RADONCNN) == 7


class TestRadonCNNClasses:
    """Tests for RadonCNN class names."""

    def test_seven_classes(self) -> None:
        assert len(RADONCNN_CLASSES) == 7

    def test_class_names(self) -> None:
        expected = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch"]
        assert RADONCNN_CLASSES == expected

    def test_no_none(self) -> None:
        assert "None" not in RADONCNN_CLASSES

    def test_no_near_full(self) -> None:
        assert "Near-full" not in RADONCNN_CLASSES


class TestDatasetCreation:
    """Tests for WaferRadonDataset creation."""

    def test_synthetic_creation(self, synthetic_dataset: WaferRadonDataset) -> None:
        assert isinstance(synthetic_dataset, WaferRadonDataset)
        assert isinstance(synthetic_dataset, BaseDataset)

    def test_synthetic_length(self, synthetic_dataset: WaferRadonDataset) -> None:
        # 7 classes × 5 samples = 35
        assert len(synthetic_dataset) == 35

    def test_synthetic_sample_structure(self, synthetic_dataset: WaferRadonDataset) -> None:
        sample = synthetic_dataset[0]
        assert "inputs" in sample
        assert "targets" in sample
        assert isinstance(sample["inputs"], torch.Tensor)
        assert isinstance(sample["targets"], int)

    def test_synthetic_image_shape(self, synthetic_dataset: WaferRadonDataset) -> None:
        sample = synthetic_dataset[0]
        assert sample["inputs"].shape == (1, 64, 64)

    def test_synthetic_label_range(self, synthetic_dataset: WaferRadonDataset) -> None:
        for i in range(len(synthetic_dataset)):
            sample = synthetic_dataset[i]
            assert 0 <= sample["targets"] < 7

    def test_synthetic_image_is_float(self, synthetic_dataset: WaferRadonDataset) -> None:
        sample = synthetic_dataset[0]
        assert sample["inputs"].dtype == torch.float32

    def test_dataset_type(self, synthetic_dataset: WaferRadonDataset) -> None:
        assert synthetic_dataset.dataset_type == DatasetType.CLASSIFICATION

    def test_repr(self, synthetic_dataset: WaferRadonDataset) -> None:
        repr_str = repr(synthetic_dataset)
        assert "WaferRadonDataset" in repr_str
        assert "classification" in repr_str


class TestBalancedDataset:
    """Tests for balanced sampling."""

    def test_balanced_creation(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        labels_path = tmp_path / "labels.csv"
        with open(labels_path, "w") as f:
            f.write("image,label\n")
            # Imbalanced: 10 samples of class 1, 2 samples of class 2
            for i in range(10):
                f.write(f"center_{i}.png,1\n")
            for i in range(2):
                f.write(f"donut_{i}.png,2\n")

        import numpy as np
        from PIL import Image

        for i in range(10):
            img = np.zeros((64, 64), dtype=np.uint8)
            img_path = images_dir / f"center_{i}.png"
            Image.fromarray(img).save(img_path)
        for i in range(2):
            img = np.zeros((64, 64), dtype=np.uint8)
            img_path = images_dir / f"donut_{i}.png"
            Image.fromarray(img).save(img_path)

        dataset = WaferRadonDataset(
            data_root=str(tmp_path),
            image_size=64,
            num_classes=7,
            balanced=True,
            apply_radon=False,
        )
        # After balancing, each class should have min_count = 2
        assert len(dataset) > 0

    def test_balanced_not_balanced(self, tmp_path: Path) -> None:
        """Without balanced=True, all samples are kept."""
        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        labels_path = tmp_path / "labels.csv"
        with open(labels_path, "w") as f:
            f.write("image,label\n")
            for i in range(10):
                f.write(f"center_{i}.png,1\n")
            for i in range(2):
                f.write(f"donut_{i}.png,2\n")

        import numpy as np
        from PIL import Image

        for i in range(10):
            img = np.zeros((64, 64), dtype=np.uint8)
            img_path = images_dir / f"center_{i}.png"
            Image.fromarray(img).save(img_path)
        for i in range(2):
            img = np.zeros((64, 64), dtype=np.uint8)
            img_path = images_dir / f"donut_{i}.png"
            Image.fromarray(img).save(img_path)

        dataset = WaferRadonDataset(
            data_root=str(tmp_path),
            image_size=64,
            num_classes=7,
            balanced=False,
            apply_radon=False,
        )
        assert len(dataset) == 12  # All samples kept


class TestDataLoader:
    """Tests for DataLoader integration."""

    def test_dataloader_with_dataset(self, synthetic_dataset: WaferRadonDataset) -> None:
        loader = DataLoader(synthetic_dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert "inputs" in batch
        assert "targets" in batch
        assert batch["inputs"].shape[0] == 4
        assert batch["inputs"].shape[1] == 1
        assert batch["inputs"].shape[2] == 64
        assert batch["inputs"].shape[3] == 64

    def test_dataloader_all_batches(self, synthetic_dataset: WaferRadonDataset) -> None:
        loader = DataLoader(synthetic_dataset, batch_size=8, shuffle=False)
        total = 0
        for batch in loader:
            total += batch["inputs"].shape[0]
        assert total == len(synthetic_dataset)

    def test_dataloader_label_types(self, synthetic_dataset: WaferRadonDataset) -> None:
        loader = DataLoader(synthetic_dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch["targets"].dtype == torch.int64 or batch["targets"].dtype == torch.int32


class TestFromDirectory:
    """Tests for loading from directory."""

    def test_from_directory(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        labels_path = tmp_path / "labels.csv"
        with open(labels_path, "w") as f:
            f.write("image,label\n")
            for i in range(3):
                f.write(f"sample_{i}.png,1\n")  # Center class

        import numpy as np
        from PIL import Image

        for i in range(3):
            img = np.random.randint(0, 2, size=(64, 64)).astype(np.uint8) * 255
            img_path = images_dir / f"sample_{i}.png"
            Image.fromarray(img).save(img_path)

        dataset = WaferRadonDataset(
            data_root=str(tmp_path),
            image_size=64,
            num_classes=7,
            balanced=False,
            apply_radon=False,
        )
        assert len(dataset) == 3

    def test_from_directory_nonexistent(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            WaferRadonDataset(
                data_root=str(tmp_path / "nonexistent"),
                image_size=64,
                num_classes=7,
            )

    def test_from_directory_empty(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        labels_path = tmp_path / "labels.csv"
        with open(labels_path, "w") as f:
            f.write("image,label\n")
            f.write("nonexistent.png,0\n")  # None class (excluded)

        with pytest.raises(ValueError, match="No valid samples"):
            WaferRadonDataset(
                data_root=str(tmp_path),
                image_size=64,
                num_classes=7,
            )