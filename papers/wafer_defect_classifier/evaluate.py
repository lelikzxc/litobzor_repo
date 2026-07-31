"""Evaluation utilities for Wafer Defect Classifier.

Содержит функции для оценки сегментационной и классификационной моделей,
а также метрики из ноутбука pipeline (3).ipynb.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Segmentation metrics ────────────────────────────────────────────────


def mae_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Средняя абсолютная ошибка между предсказанной и истинной маской."""
    return torch.abs(pred - target).mean()


def rmse_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Среднеквадратичная ошибка между предсказанной и истинной маской."""
    return torch.sqrt(((pred - target) ** 2).mean())


def snr_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Отношение сигнала к шуму (Signal-to-Noise Ratio)."""
    signal = (target ** 2).sum()
    noise = ((pred - target) ** 2).sum()
    return 10 * torch.log10(signal / (noise + 1e-8))


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Dice Index — мера перекрытия предсказанной и истинной маски."""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2 * intersection) / (pred.sum() + target.sum() + 1e-8)


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Intersection over Union — отношение пересечения к объединению масок."""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-8)


def _load_model_weights(model: torch.nn.Module, path: str | Path, device: str) -> None:
    """Загрузить веса модели из чекпоинта (поддерживает оба формата).

    Поддерживает:
    - Полный чекпоинт (dict с ключом "model")
    - Простой state_dict (тензоры напрямую)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)


def evaluate_segmentation(
    model: torch.nn.Module,
    loader: DataLoader,
    save_path: str | Path,
    device: str = "cuda",
) -> dict[str, float]:
    """Оценка сегментационной модели на переданном загрузчике.

    Args:
        model: Сегментационная модель.
        loader: DataLoader с данными.
        save_path: Путь к файлу с весами модели.
        device: Устройство.

    Returns:
        Словарь с метриками: mae, rmse, snr, dice, iou.
    """
    _load_model_weights(model, save_path, device)
    model.eval()

    total_dice, total_iou, total_mae, total_rmse, total_snr = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Evaluating segmentation"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)

            total_dice += dice_score(preds, masks).item()
            total_iou += iou_score(preds, masks).item()
            total_mae += mae_score(preds, masks).item()
            total_rmse += rmse_score(preds, masks).item()
            total_snr += snr_score(preds, masks).item()

    n = len(loader)
    metrics = {
        "mae": total_mae / n,
        "rmse": total_rmse / n,
        "snr": total_snr / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
    }

    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"SNR:  {metrics['snr']:.4f}")
    print(f"Dice: {metrics['dice']:.4f}")
    print(f"IoU:  {metrics['iou']:.4f}")

    return metrics


# ── Classification metrics ──────────────────────────────────────────────


def evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    classes_list: list[str],
    save_path: str | Path,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Оценка классификационной модели.

    Args:
        model: Классификационная модель.
        loader: DataLoader с данными.
        classes_list: Список названий классов.
        save_path: Путь к файлу с весами модели.
        device: Устройство.

    Returns:
        (all_labels, all_preds, all_probs) — numpy массивы.
    """
    _load_model_weights(model, save_path, device)
    model.eval()

    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating classifier"):
            inputs = inputs.to(device)
            preds = model(inputs)
            probs = F.softmax(preds, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, average='weighted'):.4f}")
    print(f"F1:        {f1_score(all_labels, all_preds, average='weighted'):.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion matrix:\n{cm}")

    return all_labels, all_preds, all_probs