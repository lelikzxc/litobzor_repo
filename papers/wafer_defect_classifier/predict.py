"""Inference entry point for Wafer Defect Classifier.

Загружает лучший чекпоинт классификатора (или сегментации) и
выдаёт метрики на тестовом датасете.

Usage:
    # Оценка классификатора на тесте
    python papers/wafer_defect_classifier/predict.py --mode classification

    # Оценка сегментации на тесте
    python papers/wafer_defect_classifier/predict.py --mode segmentation

    # С указанием пути к чекпоинту и конфигу
    python papers/wafer_defect_classifier/predict.py \
        --mode classification \
        --checkpoint path/to/checkpoint.pth \
        --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from papers.wafer_defect_classifier.data_utils.dataset import (
    ClassificationDataset,
    SegmentationDataset,
)
from papers.wafer_defect_classifier.evaluate import (
    evaluate_classifier,
    evaluate_segmentation,
)
from papers.wafer_defect_classifier.models import ClassificationModel, SegmentationModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Wafer Defect Classifier on test set")
    parser.add_argument(
        "--config",
        type=str,
        default="papers/wafer_defect_classifier/configs/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["segmentation", "classification"],
        help="Which model to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth file (default: from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    return parser.parse_args()


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
    ckpt_cfg = config["checkpoint"]
    image_size = config["model"]["image_size"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    if args.mode == "segmentation":
        # ── Segmentation evaluation ────────────────────────────────────────
        seg_cfg = config["segmentation"]
        batch_size = args.batch_size or seg_cfg["batch_size"]
        checkpoint_path = args.checkpoint or ckpt_cfg["segmentation"]

        print(f"\n{'=' * 60}")
        print(f"Оценка сегментационной модели")
        print(f"Чекпоинт: {checkpoint_path}")
        print(f"{'=' * 60}")

        # Build test dataset (80/20 split, use val half as test)
        dataset = SegmentationDataset(
            csv_path=data_cfg["labels_balanced"],
            img_dir=data_cfg["images_step1"],
            mask_dir=data_cfg["masks"],
            transform=transform,
        )
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        _, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=0,
        )

        model = SegmentationModel().to(device)
        metrics = evaluate_segmentation(model, test_loader, checkpoint_path, device=device)

        print(f"\nИтоговые метрики сегментации на тесте:")
        for name, val in metrics.items():
            print(f"  {name}: {val:.4f}")

    elif args.mode == "classification":
        # ── Classification evaluation ──────────────────────────────────────
        clf_cfg = config["classification"]
        batch_size = args.batch_size or clf_cfg["batch_size"]
        checkpoint_path = args.checkpoint or ckpt_cfg["classification"]

        print(f"\n{'=' * 60}")
        print(f"Оценка классификационной модели")
        print(f"Чекпоинт: {checkpoint_path}")
        print(f"{'=' * 60}")

        # Recreate test split the same way as in train.py
        df_balanced = pd.read_csv(data_cfg["labels_balanced"])

        labels_all = df_balanced["label"].map(
            {c: i for i, c in enumerate(sorted(df_balanced["label"].unique()))}
        ).values

        train_val_idx, test_idx = train_test_split(
            torch.arange(len(df_balanced)),
            test_size=data_cfg["test_split"],
            stratify=labels_all,
            random_state=42,
        )

        # Используем images_step1 (после conditional filling) как вход для классификатора
        img_dir = data_cfg["images_step1"]

        out_dir = Path(data_cfg["labels_path"]).parent
        df_balanced.iloc[test_idx].to_csv(out_dir / "labels_test.csv", index=False)

        test_dataset = ClassificationDataset(
            out_dir / "labels_test.csv", img_dir, transform=transform,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        )

        model = ClassificationModel(
            num_classes=config["model"]["num_classes"],
            dropout=clf_cfg["dropout"],
        ).to(device)

        all_labels, all_preds, all_probs = evaluate_classifier(
            model, test_loader, test_dataset.classes, checkpoint_path, device=device,
        )

        # Per-class metrics
        from sklearn.metrics import classification_report
        print(f"\nPer-class report:")
        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=test_dataset.classes,
                digits=4,
            )
        )


if __name__ == "__main__":
    main()