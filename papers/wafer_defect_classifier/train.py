"""Training entry point for Wafer Defect Classifier.

Обучает две модели последовательно:
1. SegmentationModel — сегментация дефектов (BCELoss)
2. ClassificationModel — классификация типов дефектов (CrossEntropyLoss + class weights)

Usage:
    # Полный пайплайн: предобработка → сегментация → классификация
    python papers/wafer_defect_classifier/train.py --full

    # Только сегментация
    python papers/wafer_defect_classifier/train.py --segmentation

    # Только классификация
    python papers/wafer_defect_classifier/train.py --classification

    # С предобработкой
    python papers/wafer_defect_classifier/train.py --preprocessing --segmentation --classification
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from papers.wafer_defect_classifier.data_utils.dataset import (
    ClassificationDataset,
    SegmentationDataset,
)
from papers.wafer_defect_classifier.models import ClassificationModel, SegmentationModel
from papers.wafer_defect_classifier.preprocessing import run_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Wafer Defect Classifier")
    parser.add_argument("--config", type=str, default="papers/wafer_defect_classifier/configs/config.yaml", help="Path to config")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--preprocessing", action="store_true", help="Run preprocessing step")
    parser.add_argument("--segmentation", action="store_true", help="Train segmentation model")
    parser.add_argument("--classification", action="store_true", help="Train classification model")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (preprocessing + segmentation + classification)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--resume-seg", type=str, default=None, help="Resume segmentation from checkpoint path")
    parser.add_argument("--resume-clf", type=str, default=None, help="Resume classification from checkpoint path")
    return parser.parse_args()


def train_segmentation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: str | Path,
    epochs: int = 20,
    patience: int = 5,
    device: str = "cuda",
    resume_path: str | Path | None = None,
) -> tuple[SegmentationModel, list[float], list[float], int]:
    """Обучение сегментационной модели с early stopping по val loss.

    Args:
        train_loader: DataLoader для тренировки.
        val_loader: DataLoader для валидации.
        save_path: Путь для сохранения лучшей модели.
        epochs: Максимальное количество эпох.
        patience: Early stopping patience.
        device: Устройство.
        resume_path: Путь к чекпоинту для продолжения обучения.

    Returns:
        (модель, история train_loss, история val_loss, лучшая эпоха)
    """
    model = SegmentationModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    best_loss = float("inf")
    best_epoch = 1
    patience_counter = 0
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    start_epoch = 0

    # Resume from checkpoint
    if resume_path is not None:
        resume_path = Path(resume_path)
        if not resume_path.exists():
            print(f"Предупреждение: чекпоинт {resume_path} не найден, начинаем с нуля")
        else:
            print(f"Загрузка чекпоинта сегментации из {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint.get("epoch", 0)
            best_loss = checkpoint.get("best_loss", float("inf"))
            best_epoch = checkpoint.get("best_epoch", 1)
            train_loss_history = checkpoint.get("train_loss_history", [])
            val_loss_history = checkpoint.get("val_loss_history", [])
            print(f"Возобновление с эпохи {start_epoch}, лучшая val_loss = {best_loss:.4f} (эпоха {best_epoch})")

    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Seg Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}: train_loss = {avg_train_loss:.4f}  val_loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            # Сохраняем полный чекпоинт
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping на эпохе {epoch+1}, лучшая эпоха: {best_epoch}")
                break

    print(f"\nЛучшая эпоха: {best_epoch}, val_loss = {best_loss:.4f}")
    return model, train_loss_history, val_loss_history, best_epoch


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor | None,
    save_path: str | Path,
    epochs: int = 20,
    patience: int = 5,
    device: str = "cuda",
    resume_path: str | Path | None = None,
) -> tuple[ClassificationModel, list[float], list[float], int]:
    """Обучение классификационной модели с early stopping по val loss.

    Args:
        train_loader: DataLoader для тренировки.
        val_loader: DataLoader для валидации.
        class_weights: Веса классов для взвешенной кросс-энтропии.
        save_path: Путь для сохранения лучшей модели.
        epochs: Максимальное количество эпох.
        patience: Early stopping patience.
        device: Устройство.
        resume_path: Путь к чекпоинту для продолжения обучения.

    Returns:
        (модель, история train_loss, история val_loss, лучшая эпоха)
    """
    model = ClassificationModel(num_classes=9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_loss = float("inf")
    best_epoch = 1
    patience_counter = 0
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    start_epoch = 0

    # Resume from checkpoint
    if resume_path is not None:
        resume_path = Path(resume_path)
        if not resume_path.exists():
            print(f"Предупреждение: чекпоинт {resume_path} не найден, начинаем с нуля")
        else:
            print(f"Загрузка чекпоинта классификации из {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint.get("epoch", 0)
            best_loss = checkpoint.get("best_loss", float("inf"))
            best_epoch = checkpoint.get("best_epoch", 1)
            train_loss_history = checkpoint.get("train_loss_history", [])
            val_loss_history = checkpoint.get("val_loss_history", [])
            print(f"Возобновление с эпохи {start_epoch}, лучшая val_loss = {best_loss:.4f} (эпоха {best_epoch})")

    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Clf Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                loss = criterion(preds, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}: train_loss = {avg_train_loss:.4f}  val_loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            # Сохраняем полный чекпоинт
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping на эпохе {epoch+1}, лучшая эпоха: {best_epoch}")
                break

    # Сохраняем финальную модель отдельно (только state_dict для инференса)
    overfit_path = save_path.replace(".pth", "_overfit.pth")
    torch.save(model.state_dict(), overfit_path)

    print(f"\nЛучшая эпоха: {best_epoch}, val_loss = {best_loss:.4f}")
    return model, train_loss_history, val_loss_history, best_epoch


def main() -> None:
    args = parse_args()

    # Device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load config
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    seg_cfg = config["segmentation"]
    clf_cfg = config["classification"]
    prep_cfg = config["preprocessing"]
    ckpt_cfg = config["checkpoint"]

    # Override epochs/batch_size from CLI
    seg_epochs = args.epochs or seg_cfg["num_epochs"]
    clf_epochs = args.epochs or clf_cfg["num_epochs"]
    seg_batch_size = args.batch_size or seg_cfg["batch_size"]
    clf_batch_size = args.batch_size or clf_cfg["batch_size"]

    run_prep = args.preprocessing or args.full
    run_seg = args.segmentation or args.full
    run_clf = args.classification or args.full

    # ── Preprocessing ────────────────────────────────────────────────────
    if run_prep:
        print("\n" + "=" * 60)
        print("Этап 0: Предобработка данных")
        print("=" * 60)
        run_preprocessing(
            labels_path=data_cfg["labels_path"],
            images_dir=data_cfg["images_original"],
            output_dir=Path(data_cfg["labels_path"]).parent,
            n_none=prep_cfg["n_none"],
            threshold=prep_cfg["threshold"],
            random_state=prep_cfg["random_state"],
        )

    # ── Segmentation ─────────────────────────────────────────────────────
    if run_seg:
        print("\n" + "=" * 60)
        print("Этап 1: Обучение сегментационной модели")
        print("=" * 60)

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((config["model"]["image_size"], config["model"]["image_size"])),
            transforms.ToTensor(),
        ])

        dataset = SegmentationDataset(
            csv_path=data_cfg["labels_balanced"],
            img_dir=data_cfg["images_step1"],
            mask_dir=data_cfg["masks"],
            transform=transform,
        )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=seg_batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=seg_batch_size, shuffle=False, num_workers=0)

        model, train_hist, val_hist, best_epoch = train_segmentation(
            train_loader, val_loader,
            save_path=ckpt_cfg["segmentation"],
            epochs=seg_epochs,
            patience=seg_cfg["patience"],
            device=device,
            resume_path=args.resume_seg,
        )

        # Generate predicted masks for all samples
        print("\nГенерируем предсказанные маски для всех образцов...")
        from torchvision import transforms as T
        transform_simple = T.Compose([
            T.Resize((config["model"]["image_size"], config["model"]["image_size"])),
            T.ToTensor(),
        ])

        full_dataset = SegmentationDataset(
            csv_path=data_cfg["labels_balanced"],
            img_dir=data_cfg["images_step1"],
            mask_dir=data_cfg["masks"],
            transform=transform_simple,
        )
        full_loader = DataLoader(full_dataset, batch_size=seg_batch_size, shuffle=False, num_workers=0)

        import pandas as pd
        df_balanced = pd.read_csv(data_cfg["labels_balanced"])
        pred_mask_dir = Path(data_cfg["masks"]) / "predicted"
        pred_mask_dir.mkdir(parents=True, exist_ok=True)

        idx = 0
        model.eval()
        with torch.no_grad():
            for imgs, _ in tqdm(full_loader, desc="Generating masks"):
                imgs = imgs.to(device)
                preds = model(imgs)
                preds_bin = (preds > 0.5).squeeze(1).cpu().numpy().astype(np.uint8) * 255
                for i in range(len(preds_bin)):
                    fname = df_balanced.iloc[idx]["image"]
                    from PIL import Image
                    Image.fromarray(preds_bin[i]).save(pred_mask_dir / fname)
                    idx += 1
        print(f"Предсказанные маски сохранены в {pred_mask_dir}")

    # ── Classification ───────────────────────────────────────────────────
    if run_clf:
        print("\n" + "=" * 60)
        print("Этап 2: Обучение классификационной модели")
        print("=" * 60)

        from torchvision import transforms as T
        transform_clf = T.Compose([
            T.Resize((config["model"]["image_size"], config["model"]["image_size"])),
            T.ToTensor(),
        ])

        # Используем images_step1 (после conditional filling) как вход для классификатора
        img_dir = data_cfg["images_step1"]

        df_balanced = pd.read_csv(data_cfg["labels_balanced"])

        # Stratified split
        labels_all = df_balanced["label"].map(
            {c: i for i, c in enumerate(sorted(df_balanced["label"].unique()))}
        ).values

        train_val_idx, test_idx = train_test_split(
            np.arange(len(df_balanced)),
            test_size=data_cfg["test_split"],
            stratify=labels_all,
            random_state=42,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=data_cfg["val_split"] / (data_cfg["train_split"] + data_cfg["val_split"]),
            stratify=df_balanced.iloc[train_val_idx]["label"],
            random_state=42,
        )

        # Save split CSVs
        out_dir = Path(data_cfg["labels_path"]).parent
        df_balanced.iloc[train_idx].to_csv(out_dir / "labels_train.csv", index=False)
        df_balanced.iloc[val_idx].to_csv(out_dir / "labels_val.csv", index=False)
        df_balanced.iloc[test_idx].to_csv(out_dir / "labels_test.csv", index=False)

        train_dataset = ClassificationDataset(
            out_dir / "labels_train.csv", img_dir, transform=transform_clf
        )
        val_dataset = ClassificationDataset(
            out_dir / "labels_val.csv", img_dir, transform=transform_clf
        )
        test_dataset = ClassificationDataset(
            out_dir / "labels_test.csv", img_dir, transform=transform_clf
        )

        train_loader = DataLoader(train_dataset, batch_size=clf_batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=clf_batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=clf_batch_size, shuffle=False, num_workers=0)

        # Class weights
        if clf_cfg.get("use_class_weights", True):
            labels_train = df_balanced.iloc[train_idx]["label"].map(
                {c: i for i, c in enumerate(sorted(df_balanced["label"].unique()))}
            ).values
            class_weights = torch.tensor(
                compute_class_weight("balanced", classes=np.arange(len(train_dataset.classes)), y=labels_train),
                dtype=torch.float,
            ).to(device)
            print(f"Class weights: {class_weights}")
        else:
            class_weights = None

        model, train_hist, val_hist, best_epoch = train_classifier(
            train_loader, val_loader, class_weights,
            save_path=ckpt_cfg["classification"],
            epochs=clf_epochs,
            patience=clf_cfg["patience"],
            device=device,
            resume_path=args.resume_clf,
        )

        # Evaluate on test set
        print("\n" + "=" * 60)
        print("Оценка на тестовом наборе")
        print("=" * 60)
        from papers.wafer_defect_classifier.evaluate import evaluate_classifier
        evaluate_classifier(
            model, test_loader, test_dataset.classes,
            save_path=ckpt_cfg["classification"],
            device=device,
        )

    print("\nОбучение завершено!")


if __name__ == "__main__":
    main()