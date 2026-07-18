"""DataModule: generic DataLoader factory for classification, segmentation, and multitask."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset, Subset

from common.datasets.collate import build_collate_fn


class DataModule:
    """Generic DataLoader factory.

    Wraps train/val/test/predict datasets and provides corresponding DataLoader
    factory methods. Automatically selects the appropriate collate function based
    on ``dataset_type`` unless a custom ``collate_fn`` is provided.

    Parameters
    ----------
    dataset_type : str
        One of ``"classification"``, ``"segmentation"``, ``"multitask"``.
    train_dataset : Dataset | Subset | None
        Training dataset.
    val_dataset : Dataset | Subset | None
        Validation dataset.
    test_dataset : Dataset | Subset | None
        Test dataset.
    predict_dataset : Dataset | Subset | None
        Prediction dataset.
    batch_size : int
        Batch size for all loaders (default 32).
    shuffle : bool
        Whether to shuffle the training loader (default True).
    num_workers : int
        Number of DataLoader worker processes (default 0).
    collate_fn : callable | None
        Custom collate function. If None, auto-selected from ``dataset_type``.
    pin_memory : bool
        Whether to pin memory in DataLoader (default False).
    drop_last : bool
        Whether to drop the last incomplete batch (default False).
    **kwargs : Any
        Additional keyword arguments forwarded to ``DataLoader``.
    """

    def __init__(
        self,
        dataset_type: str,
        train_dataset: Dataset | Subset | None = None,
        val_dataset: Dataset | Subset | None = None,
        test_dataset: Dataset | Subset | None = None,
        predict_dataset: Dataset | Subset | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn: callable | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        **kwargs: Any,
    ) -> None:
        self._dataset_type = dataset_type
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._collate_fn = collate_fn
        self._loader_kwargs = kwargs

    def _get_collate_fn(self) -> callable | None:
        """Return the collate function, auto-selecting if not explicitly set."""
        if self._collate_fn is not None:
            return self._collate_fn
        return build_collate_fn(self._dataset_type)

    def _build_loader(
        self,
        dataset: Dataset | Subset | None,
        shuffle: bool = False,
        dataset_label: str = "dataset",
    ) -> DataLoader:
        """Build a DataLoader from a dataset."""
        if dataset is None:
            raise ValueError(f"{dataset_label} must be provided")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            **self._loader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self._build_loader(self.train_dataset, shuffle=self.shuffle, dataset_label="train_dataset")

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._build_loader(self.val_dataset, dataset_label="val_dataset")

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return self._build_loader(self.test_dataset, dataset_label="test_dataset")

    def predict_dataloader(self) -> DataLoader:
        """Return the prediction DataLoader."""
        return self._build_loader(self.predict_dataset, dataset_label="predict_dataset")