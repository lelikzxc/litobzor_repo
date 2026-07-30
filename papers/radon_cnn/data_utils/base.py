"""Base dataset class and dataset type enum for RadonCNN.

Provides ``BaseDataset`` (abstract base for all datasets) and ``DatasetType``
(enum for classification tasks).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from torch.utils.data import Dataset


class DatasetType(Enum):
    """Enumeration of supported dataset types."""

    CLASSIFICATION = "classification"


class BaseDataset(Dataset[dict[str, Any]]):
    """Abstract base dataset for all paper-specific datasets.

    All datasets in the repository should extend this class and set
    ``dataset_type`` via ``super().__init__(dataset_type=...)``.

    Args:
        dataset_type: One of ``DatasetType`` values.
        transform: Optional transform to apply to images.
        target_transform: Optional transform to apply to targets.
    """

    def __init__(
        self,
        dataset_type: DatasetType,
        transform: callable | None = None,
        target_transform: callable | None = None,
    ) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a sample dict at the given index."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"type={self.dataset_type.value}, "
            f"len={len(self) if hasattr(self, '__len__') else '?'})"
        )