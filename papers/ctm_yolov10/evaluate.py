"""Evaluate a trained CTM-IYOLOv10 model on Magnetic Tile test set.

Usage:
    python papers/ctm_yolov10/evaluate.py --checkpoint checkpoints/ctm_yolov10/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from papers.ctm_yolov10.data_utils import MagneticTileDataset
from papers.ctm_yolov10.models.yolov10 import CTMIYOLOv10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CTM-IYOLOv10 on Magnetic Tile test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ctm_yolov10/best.pt",
        help="Path to checkpoint .pt file",
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
    )
    return parser.parse_args()


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    all_bboxes, all_cls, all_batch_idx = [], [], []
    for i, item in enumerate(batch):
        labels = item["label"]
        if labels.numel() == 0:
            continue
        cx, cy, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
        obj_bboxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        all_bboxes.append(obj_bboxes)
        all_cls.append(labels[:, 0:1])
        all_batch_idx.append(torch.full((labels.shape[0],), i, dtype=torch.long))
    batch_dict = {"img": images}
    if all_bboxes:
        batch_dict["bboxes"] = torch.cat(all_bboxes, dim=0)
        batch_dict["cls"] = torch.cat(all_cls, dim=0)
        batch_dict["batch_idx"] = torch.cat(all_batch_idx, dim=0)
    else:
        batch_dict["bboxes"] = torch.zeros(0, 4)
        batch_dict["cls"] = torch.zeros(0, 1)
        batch_dict["batch_idx"] = torch.zeros(0, dtype=torch.long)
    return batch_dict


@torch.no_grad()
def compute_map(all_pred_boxes, all_pred_scores, all_pred_classes,
                all_gt_boxes, all_gt_classes, num_classes, iou_threshold=0.5):
    device = all_pred_boxes[0].device if all_pred_boxes else torch.device("cpu")
    ap_per_class = []
    for c in range(num_classes):
        pred_boxes_list, pred_scores_list, gt_boxes_list, gt_detected_list = [], [], [], []
        for img_idx in range(len(all_pred_boxes)):
            gt_mask = all_gt_classes[img_idx] == c
            gt_boxes = all_gt_boxes[img_idx][gt_mask]
            gt_boxes_list.append(gt_boxes)
            gt_detected_list.append(torch.zeros(len(gt_boxes), dtype=torch.bool, device=device))
            pred_mask = all_pred_classes[img_idx] == c
            pred_boxes_list.append(all_pred_boxes[img_idx][pred_mask])
            pred_scores_list.append(all_pred_scores[img_idx][pred_mask])
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
        pred_img_map = torch.tensor(
            [img_idx for img_idx in range(len(pred_boxes_list)) for _ in range(len(pred_boxes_list[img_idx]))],
            device=device)[sorted_idx]
        for i in range(len(all_pred_boxes_c)):
            img_idx = int(pred_img_map[i].item())
            gt_boxes = gt_boxes_list[img_idx]
            gt_detected = gt_detected_list[img_idx]
            if len(gt_boxes) == 0:
                fp[i] = 1.0
                continue
            ious = box_iou(all_pred_boxes_c[i].unsqueeze(0), gt_boxes)[0]
            max_iou, max_idx = ious.max(dim=0)
            if max_iou >= iou_threshold and not gt_detected[max_idx]:
                tp[i] = 1.0
                gt_detected_list[img_idx][max_idx] = True
            else:
                fp[i] = 1.0
        tp_cum, fp_cum = tp.cumsum(dim=0), fp.cumsum(dim=0)
        prec = tp_cum / (tp_cum + fp_cum + 1e-16)
        rec = tp_cum / max(total_gt, 1)
        ap = sum(prec[rec >= t].max().item() for t in torch.linspace(0, 1, 11) if (rec >= t).any()) / 11.0
        ap_per_class.append(ap)
    map50 = sum(ap_per_class) / max(len(ap_per_class), 1) * 100.0
    return map50, [ap * 100.0 for ap in ap_per_class]


@torch.no_grad()
def main() -> None:
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    config = EngineConfig.from_yaml(config_path)

    data_root = config.get("data.data_root", "datasets/magnetic_tile")
    image_size = config.get("data.image_size", 640)
    num_classes = config.get("model.num_classes", 3)

    print(f"Loading Magnetic Tile dataset from: {data_root}")
    test_dataset = MagneticTileDataset(data_root=data_root, split="test", image_size=image_size)
    print(f"  Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    print("Creating CTM-IYOLOv10 model...")
    model = CTMIYOLOv10.from_config(config)
    model = model.to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(f"  Loaded from epoch {state.get('epoch', 0)}, best_map={state.get('best_map', 'N/A')}")
    elif "model" in state:
        model.load_state_dict(state["model"])
        print(f"  Loaded from epoch {state.get('epoch', 0)}")
    else:
        model.load_state_dict(state)
        print("  Loaded state_dict directly")

    model.eval()

    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")

    all_pred_boxes, all_pred_scores, all_pred_classes = [], [], []
    all_gt_boxes, all_gt_classes = [], []

    for batch in test_loader:
        images = batch["img"].to(device)
        _, _, H, W = images.shape
        norm_factor = torch.tensor([W, H, W, H], device=device).view(1, 1, 4)

        raw_out = model(images)
        preds_tensor = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

        for b in range(preds_tensor.shape[0]):
            preds_img = preds_tensor[b]
            conf_mask = preds_img[:, 4] > 0.001
            if conf_mask.any():
                boxes_norm = preds_img[conf_mask, :4] / norm_factor[0]
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

    map50, ap_per_class = compute_map(
        all_pred_boxes, all_pred_scores, all_pred_classes,
        all_gt_boxes, all_gt_classes, num_classes,
    )

    print(f"\nTest Results:")
    print(f"  mAP@0.5: {map50:.2f}%")
    for c, ap in enumerate(ap_per_class):
        print(f"  Class {c} AP: {ap:.2f}%")

    print(f"\nDone!")


if __name__ == "__main__":
    main()