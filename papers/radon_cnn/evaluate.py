"""Evaluate a trained RadonCNN model on WM-811K test set.

Usage:
    python papers/radon_cnn/evaluate.py --checkpoint checkpoints/radon_cnn/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.training.metrics import accuracy, f1, precision, recall
from papers.radon_cnn.data_utils import WaferRadonDataset
from papers.radon_cnn.models.radon_cnn import RadonCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RadonCNN on WM-811K test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/radon_cnn/best.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="papers/radon_cnn/configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load config ─────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    config = EngineConfig.from_yaml(config_path)

    # ── Create dataset ──────────────────────────────────────────────────
    data_root = config.get("data.data_root", "datasets/wm811k")
    image_size = config.get("data.image_size", 64)
    num_classes = config.get("model.num_classes", 7)
    balanced = config.get("data.balanced", True)
    train_split = config.get("data.train_split", 0.8)
    val_split = config.get("data.val_split", 0.1)

    print(f"Loading WM-811K dataset from: {data_root}")
    full_dataset = WaferRadonDataset(
        data_root=data_root,
        image_size=image_size,
        num_classes=num_classes,
        balanced=False,
    )
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Classes: {full_dataset.class_names}")

    total = len(full_dataset)
    train_len = int(total * train_split)
    val_len = int(total * val_split)
    test_len = total - train_len - val_len

    _, _, test_dataset = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("evaluation.batch_size", 64),
        shuffle=False,
        num_workers=0,
    )

    # ── Load model ──────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = RadonCNN(in_channels=1, num_classes=num_classes, radon_theta=64)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Evaluate ────────────────────────────────────────────────────────
    print("\nEvaluating...")
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in test_loader:
        images = batch["inputs"].to(device)
        labels = batch["targets"].to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    preds_tensor = torch.cat(all_preds)
    targets_tensor = torch.cat(all_targets)

    acc = accuracy(preds_tensor, targets_tensor)
    f1_score = f1(preds_tensor, targets_tensor)
    prec = precision(preds_tensor, targets_tensor)
    rec = recall(preds_tensor, targets_tensor)

    print(f"\n{'=' * 40}")
    print(f"Test Results")
    print(f"{'=' * 40}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1_score:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"{'=' * 40}")

    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for class_idx, class_name in enumerate(full_dataset.class_names):
        mask = targets_tensor == class_idx
        if mask.sum() > 0:
            class_acc = (preds_tensor[mask] == targets_tensor[mask]).float().mean()
            print(f"  {class_name:12s}: {class_acc:.4f}  (n={mask.sum().item()})")
        else:
            print(f"  {class_name:12s}: N/A (no samples)")


if __name__ == "__main__":
    main()