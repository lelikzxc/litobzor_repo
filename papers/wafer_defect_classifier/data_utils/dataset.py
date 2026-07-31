"""Датасеты для обучения моделей сегментации и классификации дефектов пластин.

SegmentationDataset: загружает изображения и маски с диска
    (маски созданы preprocessing.py из datasets/wm811k/)
ClassificationDataset: загружает изображения и метки классов
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# 9 классов WM-811K (все, включая None и Near-full)
WM811K_CLASSES: list[str] = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]

CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(WM811K_CLASSES)}


class SegmentationDataset(Dataset):
    """Датасет для обучения сегментационной модели.

    Загружает пары (изображение, маска) из CSV-файла с колонкой 'image'.
    Маски — бинарные (0 или 255), созданы preprocessing.py.

    Args:
        csv_path: Путь к CSV-файлу с колонкой 'image'.
        img_dir: Директория с изображениями (после этапа step1).
        mask_dir: Директория с масками.
        transform: Трансформация для изображений и масок.
        image_size: Размер изображения (по умолчанию 128).
    """

    def __init__(
        self,
        csv_path: str | Path,
        img_dir: str | Path,
        mask_dir: str | Path,
        transform: transforms.Compose | None = None,
        image_size: int = 128,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fname = self.df.iloc[idx]["image"]
        img = Image.open(self.img_dir / fname).convert("L")
        mask = Image.open(self.mask_dir / fname).convert("L")
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask


class ClassificationDataset(Dataset):
    """Датасет для обучения классификационной модели.

    Загружает изображения и метки классов из CSV-файла с колонками 'image' и 'label'.

    Args:
        csv_path: Путь к CSV-файлу с колонками 'image' и 'label'.
        img_dir: Директория с изображениями.
        transform: Трансформация для изображений.
        image_size: Размер изображения (по умолчанию 128).
    """

    def __init__(
        self,
        csv_path: str | Path,
        img_dir: str | Path,
        transform: transforms.Compose | None = None,
        image_size: int = 128,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["image"]).convert("L")
        label = self.class_to_idx[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label