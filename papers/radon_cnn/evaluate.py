"""Evaluate a trained RadonCNN model on WM-811K test set.

Usage:
    python papers/radon_cnn/evaluate.py --checkpoint checkpoints/radon_cnn/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.training.metrics import accuracy, f1, precision, recall
from common.training.utils import resolve_device
from papers.radon_cnn.data_utils import WaferRadonDataset
from papers.radon_cnn.data_utils.protocol import (
    dataset_options,
    load_manifest,
    model_state,
    subsets,
)
from papers.radon_cnn.models.radon_cnn import RadonCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RadonCNN on WM-811K test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/radon_cnn_v2/best.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override the experiment.yaml saved beside the checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # ── Load config ─────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else checkpoint_path.parent / "experiment.yaml"
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    config = EngineConfig.from_yaml(config_path)
    if device == "cpu":
        torch.set_num_threads(config.get("training.cpu_threads", 2))

    # ── Create dataset ──────────────────────────────────────────────────
    data_root = config.get("data.data_root", "datasets/wm811k")
    num_classes = config.get("model.num_classes", 7)

    print(f"Loading WM-811K dataset from: {data_root}")
    full_dataset = WaferRadonDataset(**dataset_options(config))
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Classes: {full_dataset.class_names}")

    manifest = load_manifest(full_dataset, checkpoint_path.parent / "split.json")
    _, _, test_dataset = subsets(full_dataset, manifest)
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

    model = RadonCNN(in_channels=1, num_classes=num_classes)
    model.load_state_dict(model_state(checkpoint))
    model.to(device)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Evaluate ────────────────────────────────────────────────────────
    print("\nEvaluating...")
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in test_loader:
        images = batch["inputs"].to(device)
        labels = batch["targets"].to(device)

        logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    logits_tensor = torch.cat(all_logits)
    preds_tensor = logits_tensor.argmax(dim=1)
    targets_tensor = torch.cat(all_targets)

    acc = accuracy(logits_tensor, targets_tensor)
    f1_score = f1(logits_tensor, targets_tensor, num_classes=num_classes)
    prec = precision(logits_tensor, targets_tensor, num_classes=num_classes)
    rec = recall(logits_tensor, targets_tensor, num_classes=num_classes)

    print(f"\n{'=' * 40}")
    print("Test Results")
    print(f"{'=' * 40}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1_score:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"{'=' * 40}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for class_idx, class_name in enumerate(full_dataset.class_names):
        mask = targets_tensor == class_idx
        if mask.sum() > 0:
            class_acc = (preds_tensor[mask] == targets_tensor[mask]).float().mean()
            print(f"  {class_name:12s}: {class_acc:.4f}  (n={mask.sum().item()})")
        else:
            print(f"  {class_name:12s}: N/A (no samples)")


if __name__ == "__main__":
    main()
