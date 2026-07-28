"""Evaluate a trained FCS-VMamba model on WM-811K test set.

Usage:
    python papers/vmamba/evaluate.py --checkpoint checkpoints/vmamba_wm811k/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.training.metrics import accuracy, f1, precision, recall
from papers.vmamba.data_utils import WaferWM811KDataset
from papers.vmamba.models.vmamba import FCSVMamba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FCS-VMamba on WM-811K test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/vmamba_wm811k/best.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="papers/vmamba/configs/config.yaml",
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
    image_size = config.get("data.image_size", 128)
    train_split = config.get("data.train_split", 0.8)
    val_split = config.get("data.val_split", 0.1)

    print(f"Loading WM-811K dataset from: {data_root}")
    full_dataset = WaferWM811KDataset(data_root=data_root, image_size=image_size)
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

    # ── DataLoader ──────────────────────────────────────────────────────
    eval_batch_size = config.get("evaluation.batch_size", 128)

    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return images, labels

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ── Create model ────────────────────────────────────────────────────
    print("Creating FCS-VMamba model...")
    model = FCSVMamba.from_config(config)
    model = model.to(device)

    # ── Load checkpoint ─────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in state:
        model.load_state_dict(state["model"])
        epoch = state.get("epoch", 0)
        metric = state.get("metric", None)
        print(f"  Loaded from epoch {epoch}, best metric: {metric}")
    elif "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(f"  Loaded from epoch {state.get('epoch', 0)}")
    else:
        model.load_state_dict(state)
        print("  Loaded state_dict directly")

    model.eval()

    # ── Evaluate ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")

    all_logits = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        total_loss += loss.item()
        num_batches += 1
        all_logits.append(logits)
        all_targets.append(labels)

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    avg_loss = total_loss / max(num_batches, 1)
    num_classes = config.get("model.num_classes", 9)
    acc = accuracy(logits, targets)
    f1_score = f1(logits, targets, num_classes=num_classes)
    prec = precision(logits, targets, num_classes=num_classes)
    rec = recall(logits, targets, num_classes=num_classes)

    print(f"\nTest Results:")
    print(f"  Loss:      {avg_loss:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1:        {f1_score:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")

    preds = logits.argmax(dim=1)
    print(f"\nPer-class accuracy:")
    for c in range(num_classes):
        mask = targets == c
        if mask.any():
            class_acc = (preds[mask] == targets[mask]).float().mean().item()
            print(f"  Class {c} ({full_dataset.class_names[c]}): {class_acc:.4f}")
        else:
            print(f"  Class {c}: N/A (no samples)")

    print(f"\nDone!")


if __name__ == "__main__":
    main()