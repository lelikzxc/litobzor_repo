"""Evaluate FCS-VMamba using the checkpoint's saved validation holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.engine.config import EngineConfig
from common.training.metrics import accuracy, f1, precision, recall
from common.training.utils import resolve_device
from papers.reproduction import load_holdout
from papers.vmamba.data_utils.author_dataset import build_dataset, validate_dataset_images
from papers.vmamba.kernels import csms6s
from papers.vmamba.models.vmamba import FCSVMamba
from papers.vmamba.utils.checkpoint import checkpoint_model_state
from papers.vmamba.utils.console import ProgressLoader, training_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FCS-VMamba on its saved holdout")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/vmamba_fcs_reproduction/best.pt"),
        help="Checkpoint with protocol.json in the same directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML; model and evaluation data must match the saved protocol",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--scan-backend", default="auto", choices=["auto", "chunked", "compiled"])
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args(argv)


def checkpoint_config(args) -> EngineConfig:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    protocol_path = args.checkpoint.parent / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing {protocol_path}: evaluation requires the training split")
    saved = EngineConfig(json.loads(protocol_path.read_text(encoding="utf-8"))["config"])
    if args.config is None:
        return saved
    supplied = EngineConfig.from_yaml(args.config)
    if supplied.get("model") != saved.get("model"):
        raise ValueError("Model config differs from the checkpoint training protocol")
    # Training-only overrides do not affect inference; source, pixels and
    # normalization must remain exactly the same.
    for key, default in (
        ("data.source", "raw_maps"),
        ("data.data_root", None),
        ("data.image_size", 224),
        ("data.mean", [0.485, 0.456, 0.406]),
        ("data.std", [0.229, 0.224, 0.225]),
        ("data.verify_author_archive", True),
    ):
        if supplied.get(key, default) != saved.get(key, default):
            raise ValueError(f"{key} differs from the checkpoint training protocol")
    # Saved configuration stays authoritative, including effective CLI overrides.
    return saved


def main() -> None:
    args = parse_args()
    with training_output(args.log_file):
        _run_evaluation(args)


@torch.no_grad()
def _run_evaluation(args) -> dict[str, float]:
    config = checkpoint_config(args)
    batch_size = config.get("evaluation.batch_size", 16)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("evaluation.batch_size must be a positive integer")
    device = resolve_device(args.device)
    csms6s.set_scan_backend(args.scan_backend)
    print(f"Using device: {device}")
    print(f"Selective scan backend: {csms6s.get_scan_backend(device)}")
    print(f"Loading saved dataset: {config.get('data.data_root')}")
    dataset = build_dataset(config, train=False)
    holdout, _ = load_holdout(args.checkpoint, dataset._samples)
    if not holdout or any(type(i) is not int or not 0 <= i < len(dataset) for i in holdout):
        raise ValueError("The saved holdout is empty or contains invalid sample indices")
    if len(set(holdout)) != len(holdout):
        raise ValueError("The saved holdout contains duplicate sample indices")
    validate_dataset_images(dataset, holdout)
    print(f"  Saved holdout samples: {len(holdout)}")

    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return images, labels

    loader = DataLoader(
        Subset(dataset, holdout),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    model = FCSVMamba.from_config(config)
    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_model_state(state))
    if isinstance(state, dict) and "epoch" in state:
        print(f"  Loaded epoch: {state['epoch']}")
    model.to(device).eval()

    all_logits, all_targets = [], []
    total_loss, total_samples = 0.0, 0
    for images, labels in ProgressLoader(loader, desc="Holdout"):
        logits = model(images.to(device))
        loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise RuntimeError("Evaluation produced nonfinite outputs")
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())
    logits, targets = torch.cat(all_logits), torch.cat(all_targets)
    num_classes = config.get("model.num_classes", 9)
    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": accuracy(logits, targets),
        "f1": f1(logits, targets, num_classes=num_classes),
        "precision": precision(logits, targets, num_classes=num_classes),
        "recall": recall(logits, targets, num_classes=num_classes),
    }
    print("\nHoldout Results (used for checkpoint selection during training):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    preds = logits.argmax(dim=1)
    print("\nPer-class recall:")
    for c in range(num_classes):
        mask = targets == c
        if mask.any():
            class_recall = (preds[mask] == targets[mask]).float().mean().item()
            print(f"  Class {c} ({dataset.class_names[c]}): {class_recall:.4f}")
        else:
            print(f"  Class {c}: N/A (no samples)")
    return metrics


if __name__ == "__main__":
    main()
