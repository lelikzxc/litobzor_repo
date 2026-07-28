"""Evaluate a trained SegFormer + Atrous model on WM-811K segmentation test set.

Usage:
    python papers/transformer_segmentation/evaluate.py --checkpoint checkpoints/transformer_segmentation_wm811k/best.pt
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
from common.training.metrics import iou_score, dice_score, pixel_accuracy
from papers.transformer_segmentation.data_utils import SegFormerDataset
from papers.transformer_segmentation.models.segformer import SegFormer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SegFormer + Atrous on WM-811K segmentation test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/transformer_segmentation_wm811k/best.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="papers/transformer_segmentation/configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    return parser.parse_args()


def _segmentation_collate(batch):
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    return images, masks


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
    data_root = config.get("data.data_root", "datasets/wm811k_seg")
    image_size = config.get("data.image_size", 512)
    num_classes = config.get("model.num_classes", 7)

    test_image_dir = Path(data_root) / "test" / "images"
    test_mask_dir = Path(data_root) / "test" / "masks"

    print(f"Loading WM-811K segmentation dataset from: {data_root}")
    test_dataset = SegFormerDataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        image_size=image_size,
        num_classes=num_classes,
    )
    print(f"  Test samples: {len(test_dataset)}")

    # ── DataLoader ──────────────────────────────────────────────────────
    eval_batch_size = config.get("evaluation.batch_size", 10)

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_segmentation_collate,
    )

    # ── Create model ────────────────────────────────────────────────────
    print("Creating SegFormer model...")
    model = SegFormer.from_config(config)
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

    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device).long()

        logits = model(images)  # [B, C, H, W]
        loss = loss_fn(logits, masks)

        total_loss += loss.item()
        num_batches += 1
        all_logits.append(logits)
        all_targets.append(masks)

    logits = torch.cat(all_logits, dim=0)   # [N, C, H, W]
    targets = torch.cat(all_targets, dim=0)  # [N, H, W]

    avg_loss = total_loss / max(num_batches, 1)

    # Metrics expect logits [B, C, H, W] and targets [B, H, W]
    iou = iou_score(logits, targets, num_classes=num_classes)
    dice = dice_score(logits, targets, num_classes=num_classes)
    pix_acc = pixel_accuracy(logits, targets)

    print(f"\nTest Results:")
    print(f"  Loss:           {avg_loss:.4f}")
    print(f"  Pixel Accuracy: {pix_acc:.4f}")
    print(f"  Mean IoU:       {iou:.4f}")
    print(f"  Mean Dice:      {dice:.4f}")

    # Per-class IoU
    preds = logits.argmax(dim=1)  # [N, H, W]
    print(f"\nPer-class IoU:")
    for c in range(num_classes):
        mask = targets == c
        if mask.any():
            pred_c = preds == c
            target_c = targets == c
            intersection = (pred_c & target_c).float().sum().item()
            union = (pred_c | target_c).float().sum().item()
            class_iou = intersection / max(union, 1)
            print(f"  Class {c}: {class_iou:.4f}")
        else:
            print(f"  Class {c}: N/A (no samples)")

    print(f"\nDone!")


if __name__ == "__main__":
    main()