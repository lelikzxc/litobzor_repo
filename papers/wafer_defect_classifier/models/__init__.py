"""Wafer Defect Classifier — модели сегментации и классификации дефектов пластин.

Архитектура из ноутбука pipeline (3).ipynb:
1. SegmentationModel: энкодер-декодер для выделения дефектов (3 Conv + 3 ConvTranspose)
2. ClassificationModel: 3 Conv + FC(256) + Dropout + FC(9) для классификации
"""

from __future__ import annotations

import torch
from torch import nn


class SegmentationModel(nn.Module):
    """Сегментационная модель для выделения дефектов на карте пластины.

    Encoder-decoder архитектура:
        Encoder: Conv2d(1→32) → ReLU → MaxPool → Conv2d(32→64) → ReLU → MaxPool → Conv2d(64→128) → ReLU → MaxPool
        Decoder: ConvTranspose2d(128→64) → ReLU → ConvTranspose2d(64→32) → ReLU → ConvTranspose2d(32→1) → Sigmoid

    Вход: [B, 1, H, W] — изображение пластины после предобработки (окно 3×3)
    Выход: [B, 1, H, W] — маска дефектов (пиксели 0..1)
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class ClassificationModel(nn.Module):
    """Классификационная модель для определения типа дефекта пластины.

    Архитектура:
        Encoder: Conv2d(1→32) → ReLU → MaxPool → Conv2d(32→64) → ReLU → MaxPool → Conv2d(64→128) → ReLU → MaxPool
        Classifier: Flatten → Linear(128*16*16 → 256) → ReLU → Dropout(0.5) → Linear(256 → 9)

    Вход: [B, 1, H, W] — изображение пластины (оригинал 128×128 или после предобработки)
    Выход: [B, 9] — логиты для 9 классов

    Примечание: входное изображение должно быть 128×128 (после 3× MaxPool2d(2) получаем 16×16).
    """

    def __init__(self, num_classes: int = 9, dropout: float = 0.5) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))