"""Предобработка данных для Wafer Defect Classifier.

Реализует пайплайн из ноутбука pipeline (3).ipynb:
1. Downsampling класса 'none' до 6000 образцов
2. Окно 3×3 с conditional filling (удаление изолированных дефектных пикселей)
3. Маскирование: выделение дефектных пикселей (значение 254) в бинарную маску

Всё сохраняется внутри datasets/wm811k/:
    datasets/wm811k/
        labels.csv              # исходные метки
        images/                 # исходные изображения
        labels_balanced.csv     # после downsampling none
        images_step1/           # после окна 3×3
        masks/                  # бинарные маски
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import generic_filter
from tqdm import tqdm


def count_defect_neighbors(window: np.ndarray) -> int:
    """Считает количество дефектных пикселей (значение > 1.5) в окне 3×3,
    исключая центральный пиксель."""
    center = window[4]
    return int(np.sum(window > 1.5) - (1 if center > 1.5 else 0))


def apply_conditional_filling(img_array: np.ndarray, threshold: int = 4) -> np.ndarray:
    """Удаляет изолированные дефектные пиксели.

    Проходит окном 3×3 по изображению. Если у дефектного пикселя меньше
    `threshold` дефектных соседей, он считается шумом и заменяется на фон.

    Args:
        img_array: Входное изображение [H, W] uint8 (0, 127, 254).
        threshold: Минимальное количество дефектных соседей (по умолчанию 4).

    Returns:
        Обработанное изображение [H, W] uint8.
    """
    wmap = np.round(img_array / 127).astype(np.float64)
    neighbor_counts = generic_filter(wmap, count_defect_neighbors, size=3, mode="constant", cval=0)
    result = wmap.copy()
    result[(wmap > 1.5) & (neighbor_counts < threshold)] = 1
    return (result * 127).astype(np.uint8)


def create_masks(img_array: np.ndarray) -> np.ndarray:
    """Создаёт бинарную маску: пиксели со значением 254 → 255, остальные → 0.

    Args:
        img_array: Входное изображение [H, W] uint8.

    Returns:
        Бинарная маска [H, W] uint8 (0 или 255).
    """
    return (img_array == 254).astype(np.uint8) * 255


def downsample_none(
    df: pd.DataFrame,
    none_class: str = "none",
    n_none: int = 6000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Даунсэмплинг класса 'none' до n_none образцов.

    Args:
        df: DataFrame с колонкой 'label'.
        none_class: Название класса для даунсэмплинга.
        n_none: Целевое количество образцов класса none.
        random_state: Seed для воспроизводимости.

    Returns:
        Сбалансированный DataFrame.
    """
    df_none = df[df["label"] == none_class].sample(n=n_none, random_state=random_state)
    df_rest = df[df["label"] != none_class]
    return pd.concat([df_none, df_rest]).sample(frac=1, random_state=random_state).reset_index(drop=True)


def run_preprocessing(
    labels_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    n_none: int = 6000,
    threshold: int = 4,
    random_state: int = 42,
) -> None:
    """Запускает полный пайплайн предобработки.

    1. Загружает labels.csv и делает даунсэмплинг none
    2. Применяет окно 3×3 (conditional filling)
    3. Создаёт маски
    4. Сохраняет результаты в output_dir (обычно datasets/wm811k/)

    Args:
        labels_path: Путь к labels.csv.
        images_dir: Директория с исходными изображениями.
        output_dir: Директория для результатов (та же, где лежит labels.csv).
        n_none: Целевое количество образцов none.
        threshold: Порог для conditional filling.
        random_state: Seed для воспроизводимости.
    """
    output_dir = Path(output_dir)
    step1_dir = output_dir / "images_step1"
    mask_dir = output_dir / "masks"
    step1_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 1. Downsampling
    df = pd.read_csv(labels_path)
    print(f"До даунсэмплинга: {len(df)} образцов")
    df_balanced = downsample_none(df, n_none=n_none, random_state=random_state)
    print(f"После даунсэмплинга: {len(df_balanced)} образцов")
    print(df_balanced["label"].value_counts())

    # Сохраняем сбалансированный CSV
    balanced_csv = output_dir / "labels_balanced.csv"
    df_balanced.to_csv(balanced_csv, index=False)
    print(f"Сохранён {balanced_csv}")

    # 2. Окно 3×3
    print("\nПрименяем окно 3×3 (conditional filling)...")
    for fname in tqdm(df_balanced["image"]):
        img_path = Path(images_dir) / fname
        if not img_path.exists():
            img_path = Path(images_dir) / f"{Path(fname).stem}.png"
        img_array = np.array(Image.open(img_path).convert("L"))
        processed = apply_conditional_filling(img_array, threshold=threshold)
        Image.fromarray(processed).save(step1_dir / fname)
    print(f"Сохранено в {step1_dir}")

    # 3. Маски
    print("\nСоздаём маски...")
    for fname in tqdm(df_balanced["image"]):
        img = np.array(Image.open(step1_dir / fname).convert("L"))
        mask = create_masks(img)
        Image.fromarray(mask).save(mask_dir / fname)
    print(f"Сохранено в {mask_dir}")

    print("\nПредобработка завершена!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Предобработка данных Wafer Defect Classifier")
    parser.add_argument("--labels", type=str, default="datasets/wm811k/labels.csv", help="Путь к labels.csv")
    parser.add_argument("--images", type=str, default="datasets/wm811k/images", help="Директория с исходными изображениями")
    parser.add_argument("--output", type=str, default="datasets/wm811k", help="Директория для результатов")
    parser.add_argument("--n-none", type=int, default=6000, help="Целевое количество образцов none")
    parser.add_argument("--threshold", type=int, default=4, help="Порог для conditional filling")
    args = parser.parse_args()

    run_preprocessing(
        labels_path=args.labels,
        images_dir=args.images,
        output_dir=args.output,
        n_none=args.n_none,
        threshold=args.threshold,
    )