"""Abstract base dataset and dataset type enum."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from torch.utils.data import Dataset


class DatasetType(Enum):
    """Enumeration of supported dataset task types."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    MULTITASK = "multitask"


class BaseDataset(Dataset, ABC):
    """Abstract base dataset for all litobzor datasets.

    Parameters
    ----------
    dataset_type : DatasetType | str
        The type of task this dataset serves (classification, segmentation, multitask).
    transform : callable, optional
        Optional transform to apply to samples.
    target_transform : callable, optional
        Optional transform to apply to targets.
    """

    def __init__(
        self,
        dataset_type: DatasetType | str,
        transform: callable | None = None,
        target_transform: callable | None = None,
    ) -> None:
        super().__init__()
        if isinstance(dataset_type, str):
            dataset_type = DatasetType(dataset_type)
        self._dataset_type = dataset_type
        self.transform = transform
        self.target_transform = target_transform

    @property
    def dataset_type(self) -> DatasetType:
        """Return the dataset type enum."""
        return self._dataset_type

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Return a sample dict with keys like 'image', 'label', 'mask'."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self._dataset_type.value}, len={len(self)})"