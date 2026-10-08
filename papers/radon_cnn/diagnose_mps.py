"""Bounded real-data RadonCNN experiment, separate from paper reproduction.

Keep the full architecture and its preprocessing. Reduce only the number of
distinct wafers and epochs, cache Radon inputs before timing GPU training, and
select the checkpoint by validation loss. Test wafers never select the model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.engine.config import EngineConfig
from common.training.utils import resolve_device
from papers.radon_cnn.data_utils import WaferRadonDataset
from papers.radon_cnn.data_utils.protocol import make_manifest, validate_manifest
from papers.radon_cnn.models.radon_cnn import RadonCNN


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="datasets/wm811k")
    parser.add_argument("--output-dir", default="checkpoints/mps_diagnostics/radon")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--train-per-class", type=int, default=128)
    parser.add_argument("--eval-per-class", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sanity-steps", type=int, default=40)
    parser.add_argument("--sanity-max-seconds", type=float, default=30)
    parser.add_argument("--prepare-only", action="store_true", help="Cache CPU inputs without training")
    parser.add_argument("--holdout-from", help="Reuse exact validation/test indices from split.json")
    return parser.parse_args()


def select_manifest(dataset, train_per_class, eval_per_class, seed):
    """Select disjoint originals from the existing balanced split protocol."""
    if train_per_class < 1 or eval_per_class < 1:
        raise ValueError("Each class needs positive training/evaluation counts")
    config = EngineConfig({"data": {
        "protocol": "balanced", "train_size": 7 * train_per_class,
        "val_split": 0.1, "test_split": 0.1,
        "allow_train_replacement": False,
    }})
    manifest = make_manifest(dataset, config, seed)
    for name in ("val", "test"):
        selected, counts = [], Counter()
        for index in manifest["indices"][name]:
            label = dataset._samples[index][1]
            if counts[label] < eval_per_class:
                selected.append(index)
                counts[label] += 1
        if set(counts.values()) != {eval_per_class} or len(counts) != 7:
            raise ValueError("Not enough distinct evaluation wafers per class")
        manifest["indices"][name] = selected
        manifest["counts"][name] = dict(counts)
        manifest["unique_counts"][name] = len(selected)
    validate_manifest(dataset, manifest)
    return manifest


def materialize(dataset, indices):
    items = [dataset[index] for index in indices]
    images = torch.stack([item["inputs"] for item in items])
    labels = torch.tensor([item["targets"] for item in items], dtype=torch.long)
    if not torch.isfinite(images).all():
        raise ValueError("Nonfinite Radon inputs")
    return TensorDataset(images, labels)


def reuse_holdout(dataset, manifest, reference):
    """Keep model/data-size comparisons on exactly the same held-out wafers."""
    validate_manifest(dataset, reference)
    for name in ("val", "test"):
        manifest["indices"][name] = reference["indices"][name]
        manifest["counts"][name] = reference["counts"][name]
        manifest["unique_counts"][name] = reference["unique_counts"][name]
    validate_manifest(dataset, manifest)


def score(model, loader, device):
    model.eval()
    predictions, targets, loss_sum = [], [], 0.0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            loss_sum += nn.functional.cross_entropy(
                logits, labels.to(device), reduction="sum"
            ).item()
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    report = classification_report(
        targets, predictions, labels=list(range(7)),
        target_names=dataset_class_names(), output_dict=True, zero_division=0,
    )
    return {
        "loss": loss_sum / len(targets),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {name: report[name] for name in dataset_class_names()},
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(7))).tolist(),
    }


def dataset_class_names():
    from papers.radon_cnn.data_utils.dataset import RADONCNN_CLASSES
    return RADONCNN_CLASSES


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def overfit_sanity(train_data, device, steps, max_seconds, seed):
    """Memorization check on 28 TRAINING wafers; never a generalization score."""
    torch.manual_seed(seed + 1)
    images, labels = train_data.tensors
    selected = torch.cat([torch.where(labels == label)[0][:4] for label in range(7)])
    images, labels = images[selected].to(device), labels[selected].to(device)
    model = RadonCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    started = time.monotonic()
    first_loss, final_loss, completed = None, None, 0
    for step in range(steps):
        if time.monotonic() - started >= max_seconds:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        first_loss = final_loss if first_loss is None else first_loss
        completed = step + 1
    model.eval()
    with torch.no_grad():
        eval_accuracy = (model(images).argmax(1) == labels).float().mean().item()
    result = {"wafer_count": len(selected), "steps": completed,
              "first_train_loss": first_loss, "last_train_loss": final_loss,
              "memorization_accuracy_eval_mode": eval_accuracy,
              "seconds": time.monotonic() - started}
    del model, optimizer
    if device == "mps":
        torch.mps.empty_cache()
    return result


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.max_seconds <= 0:
        raise ValueError("Positive epochs/budget and batch_size >= 2 are required")
    if 7 * args.train_per_class < args.batch_size:
        raise ValueError("Need at least one full training batch")
    device = resolve_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() or (output / "report.json").exists():
        raise FileExistsError("Use a fresh --output-dir for every diagnostic experiment")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    started = time.monotonic()
    dataset = WaferRadonDataset(args.data_root, balanced=False, cache_radon=False)
    manifest = select_manifest(dataset, args.train_per_class, args.eval_per_class, args.seed)
    if args.holdout_from:
        reference = json.loads(Path(args.holdout_from).read_text())
        reuse_holdout(dataset, manifest, reference)
    split_path = output / "split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Cached input split differs; use a fresh --output-dir")
    split_path.write_text(json.dumps(manifest, indent=2))
    config = {"model": {"num_classes": 7, "radon_theta": 64},
              "training": {"seed": args.seed, "cpu_threads": 2},
              "evaluation": {"batch_size": args.batch_size},
              "data": {"data_root": args.data_root, "image_size": 64, "cache_radon": True}}
    (output / "experiment.yaml").write_text(yaml.safe_dump(config))
    cache_path = output / "inputs.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        train_data, val_data, test_data = [TensorDataset(*pair) for pair in cached["tensors"]]
        print("Reusing precomputed CPU Radon inputs", flush=True)
    else:
        print(f"Precomputing {sum(manifest['unique_counts'].values())} Radon inputs on CPU...", flush=True)
        train_data, val_data, test_data = [
            materialize(dataset, manifest["indices"][name]) for name in ("train", "val", "test")
        ]
        torch.save({"tensors": [data.tensors for data in (train_data, val_data, test_data)],
                    "preprocessing_seconds": time.monotonic() - started}, cache_path)
    prep_seconds = time.monotonic() - started
    if args.prepare_only:
        print(json.dumps({"prepared": str(cache_path), "seconds": prep_seconds}), flush=True)
        return
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    model = RadonCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    training_started = time.monotonic()
    time_limit = training_started + args.max_seconds
    for epoch in range(1, args.epochs + 1):
        if time.monotonic() >= time_limit:
            break
        epoch_started = time.monotonic()
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            if time.monotonic() >= time_limit and seen:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
        val = score(model, val_loader, device)
        entry = {"epoch": epoch, "train_loss": total_loss / seen,
                 "train_accuracy": correct / seen, "train_samples_seen": seen,
                 "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                 "val_macro_f1": val["macro_f1"],
                 "seconds": time.monotonic() - epoch_started}
        history.append(entry)
        if val["loss"] < best_loss:
            best_loss, best_epoch, best_state = val["loss"], epoch, cpu_state(model)
        scheduler.step()
        print(json.dumps(entry), flush=True)
    training_seconds = time.monotonic() - training_started
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "epoch": best_epoch, "metric": best_loss}, output / "best.pt")
    train_score = score(model, DataLoader(train_data, batch_size=args.batch_size), device)
    validation = score(model, val_loader, device)
    test = score(model, test_loader, device)
    parameter_count = sum(p.numel() for p in model.parameters())
    del model, optimizer
    if device == "mps":
        torch.mps.empty_cache()
    sanity = overfit_sanity(train_data, device, args.sanity_steps, args.sanity_max_seconds, args.seed)
    inputs = train_data.tensors[0]
    report = {"model": "radon_cnn", "device": device, "seed": args.seed,
              "torch_version": torch.__version__, "parameters": parameter_count,
              "diagnostic_only": True,
              "limitations": ["Reduced balanced sample, one seed; not the article protocol.",
                              "Paper result is a mean over 20 runs; no guaranteed full-run metrics."],
              "args": vars(args), "split_counts": manifest["counts"],
              "input_statistics": {"min": inputs.min().item(), "max": inputs.max().item(),
                                   "mean": inputs.mean().item(), "std": inputs.std().item()},
              "preprocessing_seconds": prep_seconds, "training_seconds": training_seconds,
              "total_seconds": time.monotonic() - started, "best_epoch": best_epoch,
              "history": history, "train": train_score, "validation": validation,
              "test": test, "overfit_sanity": sanity}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(output / "report.json"), "best_epoch": best_epoch,
                      "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
                      "sanity": sanity, "total_seconds": report["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
