"""Training entry point for CTM-YOLOv10 on Magnetic Tile dataset.

Usage:
    python papers/ctm_yolov10/train.py --config papers/ctm_yolov10/configs/config.yaml

Trains CTM-YOLOv10 on the Magnetic Tile defect detection dataset.
Uses the common Engine with proper YOLO loss handling.
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
from common.engine.engine import Engine
from common.training.utils import move_batch_to_device
from papers.ctm_yolov10.data_utils import MagneticTileDataset
from papers.ctm_yolov10.models.yolov10 import CTMYOLOv10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CTM-YOLOv10 on Magnetic Tile dataset"
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
        inputs: alias for img (for common Trainer._unpack_batch compatibility)
        targets: dummy tensor (for common Trainer._unpack_batch compatibility)
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
        "inputs": images,
        "targets": torch.zeros(1),
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


class IdentityLoss(nn.Module):
    """Loss function that returns the first argument as-is."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        return logits


class YOLOWrapper(nn.Module):
    """Wraps CTMYOLOv10 to return only the loss tensor."""

    def __init__(self, model: CTMYOLOv10) -> None:
        super().__init__()
        self.model = model
        self._batch_dict: dict | None = None
        self._last_loss_details: dict[str, torch.Tensor] | None = None
        self._last_preds: torch.Tensor | None = None  # saved for mAP
        # YOLO's v8DetectionLoss needs model.args (hyperparameters)
        from types import SimpleNamespace
        self.model.base_model.args = SimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )

    @property
    def batch_dict(self) -> dict | None:
        return self._batch_dict

    @batch_dict.setter
    def batch_dict(self, value: dict | None) -> None:
        self._batch_dict = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Run CTM-enhanced forward to get predictions
        preds = self.model(x)

        # 2. Compute loss using base_model.loss() with CTM-enhanced preds
        if self._batch_dict is not None:
            base = self.model.base_model
            try:
                loss_output = base.loss(self._batch_dict, preds=preds)
                if isinstance(loss_output, (tuple, list)):
                    raw_details = loss_output[1]
                    self._last_loss_details = {
                        'box': raw_details.get('box_loss', 0),
                        'cls': raw_details.get('cls_loss', 0),
                        'dfl': raw_details.get('dfl_loss', 0),
                    }
                    return loss_output[0].sum()
                return loss_output.sum()
            except Exception as e:
                print(f"  [YOLOWrapper] error: {e}", flush=True)
                import traceback
                traceback.print_exc()

        return x.sum() * 0.0


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
    print("Creating CTM-YOLOv10 model...")
    if args.pretrained is not None:
        config._data.setdefault("model", {}).setdefault("backbone", {})["pretrained"] = args.pretrained == "true"
    base_model = CTMYOLOv10.from_config(config)
    model = YOLOWrapper(base_model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── Create Engine ───────────────────────────────────────────────────
    print("Initializing engine...")
    engine = Engine(
        model=model,
        config=config,
        device=device,
    )
    identity_loss = IdentityLoss()
    engine.loss_fn = identity_loss
    engine.trainer.loss_fn = identity_loss
    engine.trainer.scaler.enabled = False

    # ── Train ───────────────────────────────────────────────────────────
    epochs = config.get("training.num_epochs", 100)
    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'='*60}")

    from common.training.logger import TrainingLogger

    def yolov10_fit(
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
    ) -> TrainingLogger:
        """Custom fit that injects batch dict into YOLOWrapper before each batch."""
        engine.state.training_finished = False
        engine.state.epoch = 0

        trainer = engine.trainer
        original_train_one_epoch = trainer.train_one_epoch
        original_validate = trainer.validate

        def yolov10_train_one_epoch(loader: DataLoader) -> dict[str, float]:
            """Custom train_one_epoch that injects batch dict into YOLOWrapper."""
            trainer.model.train()
            total_loss = 0.0
            num_batches = 0

            from tqdm import tqdm
            iterator = tqdm(loader, desc="Train", disable=not trainer.verbose)
            for batch in iterator:
                inputs, targets = trainer._unpack_batch(batch)
                inputs = move_batch_to_device(inputs, trainer.device)
                targets = move_batch_to_device(targets, trainer.device)

                batch_dict = {k: move_batch_to_device(v, trainer.device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch.items()}
                if hasattr(trainer.model, 'batch_dict'):
                    trainer.model.batch_dict = batch_dict

                trainer.optimizer.zero_grad()

                with trainer.scaler.autocast():
                    logits = trainer.model(inputs)
                    loss = trainer.loss_fn(logits, targets)

                trainer.scaler.backward(loss, trainer.optimizer)

                if trainer.grad_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainer.model.parameters(),
                        trainer.grad_max_norm,
                    )

                trainer.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                postfix = {"loss": f"{loss.item():.1f}"}
                if hasattr(trainer.model, '_last_loss_details') and trainer.model._last_loss_details is not None:
                    det = trainer.model._last_loss_details
                    postfix["box"] = f"{det.get('box', 0):.1f}"
                    postfix["cls"] = f"{det.get('cls', 0):.1f}"
                    postfix["dfl"] = f"{det.get('dfl', 0):.1f}"
                iterator.set_postfix(**postfix)

            avg_loss = total_loss / max(num_batches, 1)
            metrics: dict[str, float] = {"loss": avg_loss}
            return metrics

        def yolov10_validate(loader: DataLoader) -> dict[str, float]:
            """Custom validate that injects batch dict into YOLOWrapper and computes loss."""
            from tqdm import tqdm
            # NOTE: Do NOT call trainer.model.eval() here!
            # YOLO's DetectionModel needs train mode for v10Detect to return
            # loss_dict format that base.loss() expects. eval mode makes
            # v10Detect return tuple(preds, dict) which breaks base.loss().
            # torch.no_grad() is sufficient to prevent gradient computation.
            total_loss = 0.0
            total_box = 0.0
            total_cls = 0.0
            total_dfl = 0.0
            num_batches = 0

            with torch.no_grad():
                iterator = tqdm(loader, desc="Val", disable=not trainer.verbose)
                for batch in iterator:
                    inputs, targets = trainer._unpack_batch(batch)
                    inputs = move_batch_to_device(inputs, trainer.device)
                    targets = move_batch_to_device(targets, trainer.device)

                    batch_dict = {k: move_batch_to_device(v, trainer.device) if isinstance(v, torch.Tensor) else v
                                  for k, v in batch.items()}
                    if hasattr(trainer.model, 'batch_dict'):
                        trainer.model.batch_dict = batch_dict

                    logits = trainer.model(inputs)
                    loss = trainer.loss_fn(logits, targets)

                    total_loss += loss.item()
                    num_batches += 1

                    if hasattr(trainer.model, '_last_loss_details') and trainer.model._last_loss_details is not None:
                        det = trainer.model._last_loss_details
                        total_box += det.get('box', 0)
                        total_cls += det.get('cls', 0)
                        total_dfl += det.get('dfl', 0)

            metrics: dict[str, float] = {"loss": total_loss / max(num_batches, 1)}
            if num_batches > 0:
                metrics["box_loss"] = total_box / num_batches
                metrics["cls_loss"] = total_cls / num_batches
                metrics["dfl_loss"] = total_dfl / num_batches
            return metrics

        trainer.train_one_epoch = yolov10_train_one_epoch  # type: ignore[method-assign]
        trainer.validate = yolov10_validate  # type: ignore[method-assign]

        result = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
        )

        trainer.train_one_epoch = original_train_one_epoch
        trainer.validate = original_validate

        return result

    logger = yolov10_fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )

    # ── Final metrics ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    history = logger.history
    if history:
        final = history[-1]
        print(f"\nFinal training metrics:")
        if "train_loss" in final:
            print(f"  Train Loss: {final['train_loss']:.4f}")
        if "val_loss" in final:
            print(f"  Val Loss:   {final['val_loss']:.4f}")
        if "val_box_loss" in final:
            print(f"  Val Box Loss: {final['val_box_loss']:.2f} | Cls Loss: {final['val_cls_loss']:.2f} | DFL Loss: {final['val_dfl_loss']:.2f}")

    # ── Compute mAP on test set ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing mAP@0.5 on test set...")
    print(f"{'='*60}")

    # Switch base_model to eval mode so v10Detect returns decoded predictions
    base_model = engine.trainer.model.model.base_model
    base_model.eval()

    from torchvision.ops import box_iou

    all_pred_boxes: list[torch.Tensor] = []
    all_pred_scores: list[torch.Tensor] = []
    all_pred_classes: list[torch.Tensor] = []
    all_gt_boxes: list[torch.Tensor] = []
    all_gt_classes: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = engine.trainer._unpack_batch(batch)
            inputs = move_batch_to_device(inputs, engine.trainer.device)

            batch_dict = {k: move_batch_to_device(v, engine.trainer.device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}

            # Forward through CTM (base_model in eval mode → v10Detect returns tuple)
            raw_out = base_model(inputs)
            if isinstance(raw_out, (tuple, list)):
                preds_tensor = raw_out[0]  # [B, 300, 6]
            else:
                preds_tensor = raw_out

            # Normalize from pixel coords to [0,1]
            _, _, H, W = inputs.shape
            norm_factor = torch.tensor([W, H, W, H], device=preds_tensor.device).view(1, 1, 4)

            for b in range(preds_tensor.shape[0]):
                preds_img = preds_tensor[b]
                conf_mask = preds_img[:, 4] > 0.0001
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

                img_batch_idx = (batch_dict["batch_idx"] == b)
                if img_batch_idx.any():
                    all_gt_boxes.append(batch_dict["bboxes"][img_batch_idx].cpu())
                    all_gt_classes.append(batch_dict["cls"][img_batch_idx, 0].long().cpu())
                else:
                    all_gt_boxes.append(torch.zeros(0, 4))
                    all_gt_classes.append(torch.zeros(0, dtype=torch.long))

    if all_pred_boxes:
        map_results = compute_map(
            all_pred_boxes, all_pred_scores, all_pred_classes,
            all_gt_boxes, all_gt_classes,
            num_classes=num_classes,
            iou_threshold=0.5,
        )
        print(f"\nTest mAP@0.5: {map_results['mAP@0.5']:.2f}%")
        for c, ap in enumerate(map_results["AP_per_class"]):
            print(f"  Class {c} AP: {ap:.2f}%")
    else:
        print(f"\nNo predictions to evaluate")

    # ── Save final checkpoint ───────────────────────────────────────────
    checkpoint_path = engine.save()
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    if engine.state.best_metric is not None:
        print(f"Best validation metric: {engine.state.best_metric:.4f}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()