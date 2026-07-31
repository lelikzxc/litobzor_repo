"""Demo для Wafer Defect Classifier.

Загружает обученные модели и показывает примеры сегментации и классификации.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from papers.wafer_defect_classifier.data_utils.dataset import (
    ClassificationDataset,
    SegmentationDataset,
)
from papers.wafer_defect_classifier.evaluate import evaluate_classifier, evaluate_segmentation
from papers.wafer_defect_classifier.models import ClassificationModel, SegmentationModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo Wafer Defect Classifier")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seg-checkpoint", type=str, default="checkpoints/wafer_defect_classifier/segmentation_best.pth")
    parser.add_argument("--clf-checkpoint", type=str, default="checkpoints/wafer_defect_classifier/classification_best.pth")
    parser.add_argument("--data-csv", type=str, default="data/labels_test.csv")
    parser.add_argument("--img-dir", type=str, default="data/images_step1")
    parser.add_argument("--mask-dir", type=str, default="data/masks")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Segmentation
    seg_path = Path(args.seg_checkpoint)
    if seg_path.exists():
        print(f"\nЗагружаем сегментационную модель из {seg_path}")
        seg_model = SegmentationModel().to(device)
        seg_model.load_state_dict(torch.load(seg_path, map_location=device))
        seg_model.eval()

        seg_dataset = SegmentationDataset(args.data_csv, args.img_dir, args.mask_dir, transform=transform)
        seg_loader = DataLoader(seg_dataset, batch_size=32, shuffle=False, num_workers=0)
        evaluate_segmentation(seg_model, seg_loader, seg_path, device=device)
    else:
        print(f"Сегментационная модель не найдена: {seg_path}")

    # Classification
    clf_path = Path(args.clf_checkpoint)
    if clf_path.exists():
        print(f"\nЗагружаем классификационную модель из {clf_path}")
        clf_model = ClassificationModel(num_classes=9).to(device)

        clf_dataset = ClassificationDataset(args.data_csv, args.img_dir, transform=transform)
        clf_loader = DataLoader(clf_dataset, batch_size=16, shuffle=False, num_workers=0)
        evaluate_classifier(clf_model, clf_loader, clf_dataset.classes, clf_path, device=device)
    else:
        print(f"Классификационная модель не найдена: {clf_path}")


if __name__ == "__main__":
    main()