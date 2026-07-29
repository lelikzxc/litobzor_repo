"""Training entry point for CTM-IYOLOv10 on Magnetic Tile dataset.

Usage:
    python papers/ctm_yolov10/train.py --config papers/ctm_yolov10/configs/config.yaml

Trains CTM-IYOLOv10 (GhostConv + BiFPN) on the Magnetic Tile defect
detection dataset. Shows mAP@0.5 on validation set every epoch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.training.utils import move_batch_to_device
from papers.ctm_yolov10.data_utils import MagneticTileDataset
from papers.ctm_yolov10.models.yolov10 import CTMIYOLOv10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CTM-IYOLOv10 on Magnetic Tile dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="papers/ctm_yolov10/configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training (auto=use CUDA if available)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Override pretrained from config (true/false)",
    )
    return parser.parse_args()


def collate_fn(batch):
    """Custom collate for detection dict-based samples.

    Returns a dict matching YOLO's expected batch format:
        img: (B, C, H, W) stacked images
        bboxes: (N, 4) [x1, y1, x2, y2] concatenated (xyxy format, normalized)
        cls: (N, 1) class indices
        batch_idx: (N,) image indices for each object
    """
    images = torch.stack([item["image"] for item in batch])
    all_bboxes = []
    all_cls = []
    all_batch_idx = []

    for i, item in enumerate(batch):
        labels = item["label"]  # (num_objects, 5) = [cls, cx, cy, w, h] (normalized)
        if labels.numel() == 0:
            continue
        num_objs = labels.shape[0]
        # Convert cxcywh (normalized) -> xyxy (normalized)
        cx = labels[:, 1]
        cy = labels[:, 2]
        w = labels[:, 3]
        h = labels[:, 4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        obj_bboxes = torch.stack([x1, y1, x2, y2], dim=1)
        all_bboxes.append(obj_bboxes)
        all_cls.append(labels[:, 0:1])
        all_batch_idx.append(torch.full((num_objs,), i, dtype=torch.long))

    batch_dict = {
        "img": images,
    }
    if all_bboxes:
        batch_dict["bboxes"] = torch.cat(all_bboxes, dim=0)
        batch_dict["cls"] = torch.cat(all_cls, dim=0)
        batch_dict["batch_idx"] = torch.cat(all_batch_idx, dim=0)
    else:
        batch_dict["bboxes"] = torch.zeros(0, 4)
        batch_dict["cls"] = torch.zeros(0, 1)
        batch_dict["batch_idx"] = torch.zeros(0, dtype=torch.long)

    return batch_dict


def compute_map(
    all_pred_boxes: list[torch.Tensor],
    all_pred_scores: list[torch.Tensor],
    all_pred_classes: list[torch.Tensor],
    all_gt_boxes: list[torch.Tensor],
    all_gt_classes: list[torch.Tensor],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute mAP@0.5 for detection results (11-point interpolation)."""
    device = all_pred_boxes[0].device if all_pred_boxes else torch.device("cpu")
    ap_per_class = []
    for c in range(num_classes):
        pred_boxes_list = []
        pred_scores_list = []
        gt_boxes_list = []
        gt_detected_list = []
        for img_idx in range(len(all_pred_boxes)):
            gt_mask = all_gt_classes[img_idx] == c
            gt_boxes = all_gt_boxes[img_idx][gt_mask]
            gt_boxes_list.append(gt_boxes)
            gt_detected_list.append(torch.zeros(len(gt_boxes), dtype=torch.bool, device=device))
            pred_mask = all_pred_classes[img_idx] == c
            pred_boxes = all_pred_boxes[img_idx][pred_mask]
            pred_scores = all_pred_scores[img_idx][pred_mask]
            pred_boxes_list.append(pred_boxes)
            pred_scores_list.append(pred_scores)
        all_pred_boxes_c = torch.cat(pred_boxes_list) if pred_boxes_list else torch.zeros(0, 4, device=device)
        all_pred_scores_c = torch.cat(pred_scores_list) if pred_scores_list else torch.zeros(0, device=device)
        if len(all_pred_boxes_c) == 0:
            ap_per_class.append(0.0)
            continue
        sorted_idx = torch.argsort(all_pred_scores_c, descending=True)
        all_pred_boxes_c = all_pred_boxes_c[sorted_idx]
        all_pred_scores_c = all_pred_scores_c[sorted_idx]
        total_gt = sum(len(g) for g in gt_boxes_list)
        if total_gt == 0:
            ap_per_class.append(0.0)
            continue
        tp = torch.zeros(len(all_pred_boxes_c), dtype=torch.float32, device=device)
        fp = torch.zeros(len(all_pred_boxes_c), dtype=torch.float32, device=device)
        pred_img_map = []
        for img_idx in range(len(pred_boxes_list)):
            pred_img_map.extend([img_idx] * len(pred_boxes_list[img_idx]))
        pred_img_map = torch.tensor(pred_img_map, device=device)[sorted_idx]
        for i in range(len(all_pred_boxes_c)):
            img_idx = int(pred_img_map[i].item())
            pred_box = all_pred_boxes_c[i].unsqueeze(0)
            gt_boxes = gt_boxes_list[img_idx]
            gt_detected = gt_detected_list[img_idx]
            if len(gt_boxes) == 0:
                fp[i] = 1.0
                continue
            ious = box_iou(pred_box, gt_boxes)[0]
            max_iou, max_idx = ious.max(dim=0)
            if max_iou >= iou_threshold and not gt_detected[max_idx]:
                tp[i] = 1.0
                gt_detected_list[img_idx][max_idx] = True
            else:
                fp[i] = 1.0
        tp_cum = tp.cumsum(dim=0)
        fp_cum = fp.cumsum(dim=0)
        precision = tp_cum / (tp_cum + fp_cum + 1e-16)
        recall = tp_cum / max(total_gt, 1)
        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            mask = recall >= t
            if mask.any():
                ap += precision[mask].max().item()
        ap /= 11.0
        ap_per_class.append(ap)
    map50 = sum(ap_per_class) / max(len(ap_per_class), 1) * 100.0
    return {"mAP@0.5": map50, "AP_per_class": [ap * 100.0 for ap in ap_per_class]}


@torch.no_grad()
def compute_map_on_loader(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute mAP@0.5 over a DataLoader.

    Runs model in eval mode so v10Detect returns decoded predictions.
    """
    model.eval()
    all_pred_boxes: list[torch.Tensor] = []
    all_pred_scores: list[torch.Tensor] = []
    all_pred_classes: list[torch.Tensor] = []
    all_gt_boxes: list[torch.Tensor] = []
    all_gt_classes: list[torch.Tensor] = []

    for batch in loader:
        images = batch["img"].to(device)
        _, _, H, W = images.shape
        norm_factor = torch.tensor([W, H, W, H], device=device).view(1, 1, 4)

        # Forward through base_model in eval mode
        raw_out = model(images)
        if isinstance(raw_out, (tuple, list)):
            preds_tensor = raw_out[0]
        else:
            preds_tensor = raw_out

        for b in range(preds_tensor.shape[0]):
            preds_img = preds_tensor[b]
            conf_mask = preds_img[:, 4] > conf_threshold
            if conf_mask.any():
                boxes_pixel = preds_img[conf_mask, :4]
                boxes_norm = boxes_pixel / norm_factor[0]
                all_pred_boxes.append(boxes_norm.cpu())
                all_pred_scores.append(preds_img[conf_mask, 4].cpu())
                all_pred_classes.append(preds_img[conf_mask, 5].long().cpu())
            else:
                all_pred_boxes.append(torch.zeros(0, 4))
                all_pred_scores.append(torch.zeros(0))
                all_pred_classes.append(torch.zeros(0, dtype=torch.long))

            img_batch_idx = (batch["batch_idx"] == b)
            if img_batch_idx.any():
                all_gt_boxes.append(batch["bboxes"][img_batch_idx].cpu())
                all_gt_classes.append(batch["cls"][img_batch_idx, 0].long().cpu())
            else:
                all_gt_boxes.append(torch.zeros(0, 4))
                all_gt_classes.append(torch.zeros(0, dtype=torch.long))

    if not all_pred_boxes:
        return {"mAP@0.5": 0.0}

    return compute_map(
        all_pred_boxes, all_pred_scores, all_pred_classes,
        all_gt_boxes, all_gt_classes,
        num_classes=num_classes,
        iou_threshold=iou_threshold,
    )


def main() -> None:
    """Run the training loop."""
    args = parse_args()

    # ── Load configuration ──────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = EngineConfig.from_yaml(config_path)

    # Apply CLI overrides
    if args.epochs is not None:
        config._data.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config._data.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config._data.setdefault("training", {})["learning_rate"] = args.lr
        config._data.setdefault("training", {}).setdefault("optimizer", {})["lr"] = args.lr

    # ── Resolve device ──────────────────────────────────────────────────
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Create datasets ─────────────────────────────────────────────────
    data_root = config.get("data.data_root", "datasets/magnetic_tile")
    image_size = config.get("data.image_size", 640)
    num_classes = config.get("model.num_classes", 3)

    print(f"Loading Magnetic Tile dataset from: {data_root}")

    train_dataset = MagneticTileDataset(
        data_root=data_root,
        split="train",
        image_size=image_size,
    )
    val_dataset = MagneticTileDataset(
        data_root=data_root,
        split="valid",
        image_size=image_size,
    )
    test_dataset = MagneticTileDataset(
        data_root=data_root,
        split="test",
        image_size=image_size,
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # ── Create DataLoaders ──────────────────────────────────────────────
    batch_size = config.get("training.batch_size", 16)
    eval_batch_size = config.get("evaluation.batch_size", 32)
    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # ── Create model ────────────────────────────────────────────────────
    print("Creating CTM-IYOLOv10 model...")
    if args.pretrained is not None:
        config._data.setdefault("model", {}).setdefault("backbone", {})["pretrained"] = args.pretrained == "true"
    model = CTMIYOLOv10.from_config(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  GhostConv: {model.ghost_conv}")
    print(f"  BiFPN: {model.bifpn}")

    model = model.to(device)

    # ── Optimizer with differentiated LR for BiFPN ──────────────────────
    lr = config.get("training.learning_rate", 0.001)
    weight_decay = config.get("training.weight_decay", 0.0005)
    momentum = config.get("optimizer.kwargs.momentum", 0.937)
    bifpn_lr_scale = config.get("model.bifpn.lr_scale", 0.1)  # BiFPN gets 10x lower LR

    # Split parameters into groups: BiFPN gets lower LR
    bifpn_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if "bifpn_module" in name:
            bifpn_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": lr, "momentum": momentum,
             "weight_decay": weight_decay, "nesterov": True},
            {"params": bifpn_params, "lr": lr * bifpn_lr_scale, "momentum": momentum,
             "weight_decay": weight_decay, "nesterov": True},
        ],
        lr=lr,  # default LR for any extra params
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    print(f"  Optimizer groups: backbone={len(backbone_params)} params, "
          f"bifpn={len(bifpn_params)} params (LR={lr*bifpn_lr_scale:.6f})")

    # ── Scheduler ───────────────────────────────────────────────────────
    epochs = config.get("training.num_epochs", 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0.00001,
    )

    # ── BiFPN warmup ────────────────────────────────────────────────────
    # First N epochs: freeze backbone, train only BiFPN + detection head
    bifpn_warmup_epochs = config.get("model.bifpn.warmup_epochs", 5)

    def _set_bifpn_warmup(model: CTMIYOLOv10, epoch: int) -> None:
        """Freeze backbone for first N epochs to let BiFPN stabilise."""
        if epoch <= bifpn_warmup_epochs:
            # Freeze backbone (seq layers 0-10), keep BiFPN and head trainable
            for i, layer in enumerate(model.seq):
                if i <= 10:  # backbone layers
                    for p in layer.parameters():
                        p.requires_grad = False
                else:  # neck + head
                    for p in layer.parameters():
                        p.requires_grad = True
            if model.bifpn_module is not None:
                for p in model.bifpn_module.parameters():
                    p.requires_grad = True
            if epoch == 1:
                print(f"  BiFPN warmup: backbone frozen (epochs 1-{bifpn_warmup_epochs})")
        else:
            # Unfreeze everything
            for p in model.parameters():
                p.requires_grad = True
            if epoch == bifpn_warmup_epochs + 1:
                print("  BiFPN warmup complete: all layers unfrozen")

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'='*60}")

    best_map = 0.0
    from tqdm import tqdm

    for epoch in range(1, epochs + 1):
        # Apply BiFPN warmup (freeze backbone for first N epochs)
        _set_bifpn_warmup(model, epoch)

        # ── Train one epoch ─────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        total_box = 0.0
        total_cls = 0.0
        total_dfl = 0.0
        num_batches = 0
        nan_batches = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in iterator:
            images = batch["img"].to(device)

            # Build YOLO-format batch dict
            batch_dict = {
                "img": images,
            }
            if "bboxes" in batch:
                batch_dict["bboxes"] = batch["bboxes"].to(device)
                batch_dict["cls"] = batch["cls"].to(device)
                batch_dict["batch_idx"] = batch["batch_idx"].to(device)

            optimizer.zero_grad()

            # Forward through CTM model in train mode
            # CTMIYOLOv10.forward() accepts dict → calls self.loss() which
            # runs CTM-enhanced forward + criterion
            # loss() now returns a scalar tensor (sum of box+cls+dfl)
            loss = model(batch_dict)

            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                if nan_batches > 10:
                    print(f"\n  WARNING: Too many NaN losses ({nan_batches}), stopping training!")
                    break
                continue

            loss.backward()

            # Clip gradients to prevent explosion from BiFPN
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            iterator.set_postfix({
                "loss": f"{loss.item():.1f}",
            })

        avg_loss = total_loss / max(num_batches, 1)

        # ── Validate ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                images = batch["img"].to(device)
                batch_dict = {
                    "img": images,
                }
                if "bboxes" in batch:
                    batch_dict["bboxes"] = batch["bboxes"].to(device)
                    batch_dict["cls"] = batch["cls"].to(device)
                    batch_dict["batch_idx"] = batch["batch_idx"].to(device)

                # Compute loss via CTMIYOLOv10.loss()
                model.train()
                vloss = model(batch_dict)
                model.eval()

                val_loss += vloss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        # ── Compute mAP@0.5 on validation set ───────────────────────────
        map_results = compute_map_on_loader(
            model, val_loader, num_classes, device,
            conf_threshold=0.001, iou_threshold=0.5,
        )
        current_map = map_results["mAP@0.5"]

        # ── Print epoch summary ─────────────────────────────────────────
        print(f"\nEpoch {epoch}/{epochs} | "
              f"Train Loss: {avg_loss:.2f} | "
              f"Val Loss: {avg_val_loss:.2f} | "
              f"mAP@0.5: {current_map:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        # ── Save best model ─────────────────────────────────────────────
        if current_map > best_map:
            best_map = current_map
            save_dir = Path(config.get("checkpoint.save_dir", "checkpoints/ctm_yolov10"))
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_map": best_map,
                "config": config._data,
            }, save_dir / "best.pt")
            print(f"  -> New best model saved! mAP@0.5: {best_map:.2f}%")

    # ── Final evaluation on test set ────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Best validation mAP@0.5: {best_map:.2f}%")

    print(f"\nEvaluating on test set...")
    test_map = compute_map_on_loader(
        model, test_loader, num_classes, device,
        conf_threshold=0.001, iou_threshold=0.5,
    )
    print(f"Test mAP@0.5: {test_map['mAP@0.5']:.2f}%")
    for c, ap in enumerate(test_map["AP_per_class"]):
        print(f"  Class {c} AP: {ap:.2f}%")

    # ── Save final checkpoint ───────────────────────────────────────────
    save_dir = Path(config.get("checkpoint.save_dir", "checkpoints/ctm_yolov10"))
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_map": best_map,
        "test_map": test_map["mAP@0.5"],
        "config": config._data,
    }, save_dir / "last.pt")
    print(f"\nFinal checkpoint saved to: {save_dir / 'last.pt'}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()