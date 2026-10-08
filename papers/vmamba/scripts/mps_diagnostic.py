"""Bounded author-data diagnostics, separate from the full paper experiment.

Example (run from repository root)::

    python papers/vmamba/scripts/mps_diagnostic.py --device mps --image-size 64

The default keeps the full FCS architecture but reduces input resolution and
uses only 16 training images per class. Its accuracy is not a reproduction
claim. The full, deterministic author-data holdout stays outside training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.engine.config import EngineConfig
from common.training.utils import resolve_device
from common.utils.seed import set_seed
from papers.reproduction import save_protocol
from papers.vmamba.data_utils.author_dataset import build_dataset
from papers.vmamba.data_utils.protocol import balanced_indices, stratified_split
from papers.vmamba.models.vmamba import FCSVMamba


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def collate(batch):
    return torch.stack([item["image"] for item in batch]), torch.tensor(
        [item["label"] for item in batch], dtype=torch.long
    )


def build_optimizer(model, config):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.get("optimizer.weight_decay", 0.05)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.get("optimizer.lr", 0.001),
    )


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    truth, predicted = [], []
    loss_sum, count = 0.0, 0
    for images, labels in loader:
        logits = model(images.to(device))
        loss_sum += torch.nn.functional.cross_entropy(
            logits, labels.to(device), reduction="sum"
        ).item()
        truth.extend(labels.tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())
        count += len(labels)
    return {
        "loss": loss_sum / count,
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(
            truth, predicted, labels=list(range(9)), average="macro", zero_division=0
        ),
        "class_report": classification_report(
            truth,
            predicted,
            labels=list(range(9)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "samples": count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="papers/vmamba/configs/config.yaml")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-per-class", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional matched-step cap for ablations; 0 means no extra cap",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--variant", choices=["full", "no_sfs", "backbone"], default="full")
    parser.add_argument(
        "--profile-only", action="store_true", help="One real training batch, no accuracy estimate"
    )
    parser.add_argument(
        "--profile-steps", type=int, default=1, help="Batches to time in profile mode"
    )
    parser.add_argument(
        "--scan-chunk-size",
        type=int,
        default=128,
        help="Diagnostic prefix block size; production default stays unchanged",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.image_size < 32 or args.image_size % 32:
        parser.error("--image-size must be a positive multiple of 32, at least 32")
    if (
        args.max_seconds <= 0
        or args.epochs <= 0
        or args.batch_size <= 0
        or args.profile_steps <= 0
        or args.scan_chunk_size <= 0
    ):
        parser.error("time, epochs and batch size must be positive")
    device = resolve_device(args.device)
    if device == "mps" and args.scan_chunk_size != 128:
        from functools import partial
        from papers.vmamba.kernels import csms6s

        csms6s.selective_scan_chunked = partial(
            csms6s.selective_scan_chunked, chunk_size=args.scan_chunk_size
        )
    set_seed(args.seed)
    config = EngineConfig.from_yaml(args.config)
    config._data["seed"] = args.seed
    config._data["data"]["image_size"] = args.image_size
    config._data["model"]["input"]["image_size"] = args.image_size
    config._data["training"].update(
        batch_size=args.batch_size, device=device, num_epochs=args.epochs
    )
    if args.variant == "no_sfs":
        config._data["model"]["sfs"]["enabled"] = False
    elif args.variant == "backbone":
        for module in ("fa", "sfs", "clca"):
            config._data["model"][module]["enabled"] = False
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.variant
    output = Path(args.output or ROOT / "checkpoints/mps_diagnostics/vmamba" / run_name)
    if (output / "report.json").exists():
        raise FileExistsError(f"Diagnostic report already exists: {output}; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    config._data["checkpoint"]["save_dir"] = str(output)
    dataset = build_dataset(config, train=False)
    labels = [label for _, label in dataset._samples]
    train_pool, holdout = stratified_split(list(range(len(dataset))), labels, seed=args.seed)
    relative = balanced_indices(
        [labels[index] for index in train_pool], args.train_per_class, args.seed
    )
    train_indices = [train_pool[index] for index in relative]
    save_protocol(
        output, config, dataset._samples, {"train": train_indices, "test": holdout}, args.seed
    )
    augmented = build_dataset(config, train=not args.no_aug)
    train_loader = DataLoader(
        Subset(augmented, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        Subset(dataset, holdout),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    model = FCSVMamba.from_config(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    report = {
        "diagnostic_only": True,
        "paper_targets": {
            "accuracy": 0.8791,
            "macro_f1": 0.8606,
            "source": "vmamba.pdf Table 3, 902-image Subset A",
        },
        "device": device,
        "torch_version": torch.__version__,
        "variant": args.variant,
        "image_size": args.image_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_samples": len(train_indices),
        "holdout_samples": len(holdout),
        "train_class_counts": dict(
            sorted(Counter(labels[index] for index in train_indices).items())
        ),
        "holdout_class_counts": dict(sorted(Counter(labels[index] for index in holdout).items())),
        "holdout_role": "Author benchmark validation split; used for diagnostic model selection, not an independent test",
        "seed": args.seed,
        "augmentation": not args.no_aug,
        "profile_only": args.profile_only,
        "profile_steps": args.profile_steps,
        "scan_chunk_size": args.scan_chunk_size,
        "max_training_seconds": args.max_seconds,
        "max_training_steps": args.max_steps,
        "steps": [],
        "epochs": [],
    }
    start = time.perf_counter()
    if not args.profile_only:
        report["untrained_holdout"] = evaluate(model, val_loader, device, dataset.class_names)
        print("Untrained:", json.dumps(report["untrained_holdout"]), flush=True)
    synchronize(device)
    train_start = time.perf_counter()
    stopped = False
    for epoch in range(args.epochs):
        model.train()
        losses, correct, samples = [], 0, 0
        for images, labels in train_loader:
            if report["steps"] and (
                time.perf_counter() - train_start >= args.max_seconds
                or (args.max_steps > 0 and len(report["steps"]) >= args.max_steps)
            ):
                stopped = True
                break
            synchronize(device)
            step_start = time.perf_counter()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            synchronize(device)
            forward_seconds = time.perf_counter() - step_start
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            gradients = [
                parameter.grad for parameter in model.parameters() if parameter.grad is not None
            ]
            if (
                not torch.stack([torch.isfinite(gradient).all() for gradient in gradients])
                .all()
                .item()
            ):
                raise FloatingPointError("Nonfinite parameter gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            synchronize(device)
            step = {
                "epoch": epoch + 1,
                "loss": loss.item(),
                "gradient_norm_before_clip": grad_norm,
                "forward_seconds": forward_seconds,
                "seconds": time.perf_counter() - step_start,
            }
            if device == "mps":
                step["allocated_memory_bytes"] = torch.mps.current_allocated_memory()
                step["driver_allocated_memory_bytes"] = torch.mps.driver_allocated_memory()
            report["steps"].append(step)
            losses.append(loss.item())
            correct += (logits.argmax(1) == labels).sum().item()
            samples += len(labels)
            if len(report["steps"]) == 1 or len(report["steps"]) % 10 == 0:
                print("Step", len(report["steps"]), json.dumps(step), flush=True)
            if args.profile_only and len(report["steps"]) >= args.profile_steps:
                stopped = True
                break
        if losses:
            report["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)),
                    "train_accuracy": correct / samples,
                    "samples_seen": samples,
                    "complete": not stopped,
                }
            )
            print("Epoch:", json.dumps(report["epochs"][-1]), flush=True)
        if stopped:
            break
        scheduler.step()
    report["training_seconds"] = time.perf_counter() - train_start
    report["training_budget_reached"] = stopped and not args.profile_only
    if not args.profile_only:
        report["final_holdout"] = evaluate(model, val_loader, device, dataset.class_names)
        print("Final:", json.dumps(report["final_holdout"]), flush=True)
        torch.save(
            {
                "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "config": config.to_dict(),
                "diagnostic_only": True,
            },
            output / "last.pt",
        )
    report["total_seconds"] = time.perf_counter() - start
    report["mean_step_seconds_after_first"] = (
        float(np.mean([step["seconds"] for step in report["steps"][1:]]))
        if len(report["steps"]) > 1
        else report["steps"][0]["seconds"]
    )
    if args.profile_only:
        report["rough_50_epoch_hours_at_profiled_settings"] = (
            report["mean_step_seconds_after_first"]
            * math.ceil(len(train_pool) / args.batch_size)
            * 50
            / 3600
        )
        report["timing_caveat"] = (
            "First batch includes first-use compilation; later batches are used for the rough projection at the recorded resolution and batch size, not a measured full-run duration."
        )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
