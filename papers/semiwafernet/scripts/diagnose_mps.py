"""Bounded HybridCNN-ViT learning diagnostics on official Training only.

The full paper architecture is retained. Miniature balanced/stratified data
and explicit optimizer-step/time limits keep this separate from reproduction.
Comparisons isolate the learning rate and encoded-space SMOTE. Validation
is reserved before sampling/SMOTE; official Test is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.engine.config import EngineConfig
from common.training.utils import resolve_device
from common.utils.seed import set_seed
from papers.semiwafernet.data_utils.wafer_dataset import (
    SMOTEDataset,
    WM811K_CLASSES,
    _resolve_image_path,
    encode_wafer_image,
    geometric_augment,
    parse_wm811k_labeled_rows,
)
from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.scripts.diagnose_stage1 import _per_class_report
from papers.semiwafernet.training.stage_manager import StageManager


VARIANT_NAMES = ("paper_smote", "mid_smote", "fast_smote", "fast_real", "fast_real_diverse_none")


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def variant_learning_rates(variants, overrides, paper_lr):
    """Explicit diagnostic rates; no modification of the paper configuration."""
    rates = {
        name: (paper_lr if name == "paper_smote" else 2e-4 if name == "mid_smote" else 5e-4)
        for name in variants
    }
    seen = set()
    for override in overrides or []:
        name, separator, value = override.partition("=")
        if not separator or name not in rates or name in seen:
            raise ValueError("--variant-lr requires one selected NAME=VALUE per variant")
        rate = float(value)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Learning rates must be finite and positive")
        rates[name] = rate
        seen.add(name)
    return rates


def inventory_digest(samples):
    return hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest()


def load_diagnostic_split(source_path, training_samples):
    """Reuse exact roles after validating membership in official Training only."""
    path = Path(source_path)
    if path.is_dir():
        path = path / "report.json"
    source = json.loads(path.read_text())
    inventory = dict(training_samples)
    if len(inventory) != len(training_samples):
        raise ValueError("Official Training inventory has duplicate filenames")
    if source.get("training_inventory_digest") not in (None, inventory_digest(training_samples)):
        raise ValueError("Official Training inventory differs from the source report")
    roles = {
        "train": source["samples"]["train"],
        "validation": source["samples"]["validation"],
        "none_stress": source.get("none_stress_extra_samples", []),
        "diverse_none": source.get("diverse_none_extra_training_samples", []),
    }
    seen = set()
    for role, saved_rows in roles.items():
        rows = []
        for row in saved_rows:
            if len(row) != 2 or not isinstance(row[0], str) or not isinstance(row[1], int):
                raise ValueError(f"Invalid saved {role} sample")
            filename, label = row
            if inventory.get(filename) != label:
                raise ValueError(
                    f"Saved {role} sample is not an unchanged official Training row: {filename}"
                )
            if filename in seen:
                raise ValueError(f"Duplicate or overlapping saved subsets: {filename}")
            if role in {"none_stress", "diverse_none"} and label != 0:
                raise ValueError(f"Saved {role} sample is not None")
            seen.add(filename)
            rows.append((filename, label))
        if role in {"train", "validation"} and {label for _, label in rows} != set(range(9)):
            raise ValueError(f"Saved {role} must include all nine classes")
        roles[role] = rows
    expected = hashlib.sha256(
        json.dumps({"train": roles["train"], "val": roles["validation"]}).encode()
    ).hexdigest()
    if source.get("subset_digest") not in (None, expected):
        raise ValueError("Saved base-subset digest does not match its sample lists")
    roles["data_seed"] = int(source.get("data_seed", source["seed"]))
    roles["source"] = str(path.resolve())
    return roles


def split_subset(samples, train_per_class, val_per_class, seed):
    """Disjoint per-class subsets; at least 20% of each rare class is held out."""
    if train_per_class < 2 or val_per_class < 1:
        raise ValueError("Need >=2 training samples and >=1 validation sample per class")
    rng = np.random.RandomState(seed)
    train, val = [], []
    for label in range(len(WM811K_CLASSES)):
        rows = [row for row in samples if row[1] == label]
        if len(rows) < 3:
            raise ValueError(f"Class {WM811K_CLASSES[label]} has fewer than three samples")
        rng.shuffle(rows)
        n_val = min(val_per_class, max(1, len(rows) // 5), len(rows) - 2)
        val.extend(rows[:n_val])
        train.extend(rows[n_val : n_val + train_per_class])
    return train, val


def report_predictions(logits, targets):
    pred = logits.argmax(dim=1)
    accuracy, macro_f1, per_class = _per_class_report(pred.cpu(), targets.cpu(), 9)
    confusion = torch.bincount(targets.cpu() * 9 + pred.cpu(), minlength=81).reshape(9, 9)
    chance = (
        float((confusion.sum(0).double() * confusion.sum(1).double()).sum()) / len(targets) ** 2
    )
    return {
        "loss": nn.functional.cross_entropy(logits.cpu(), targets.cpu()).item(),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class])),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "cohen_kappa": (accuracy - chance) / (1 - chance) if chance < 1 else None,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_class_order": list(WM811K_CLASSES),
    }


@torch.no_grad()
def evaluate(model, images, targets, batch_size):
    model.eval()
    logits = torch.cat([model(chunk)["classification"].cpu() for chunk in images.split(batch_size)])
    return report_predictions(logits, targets)


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def gate_diagnostics(collected, accepted, tau, entropy_threshold, mi_threshold):
    confidence = collected["conf"].cpu()
    entropy, mi = collected["ent"].cpu(), collected["mi"].cpu()
    predicted, labels = collected["pred"].cpu(), collected["y"].cpu()
    tau, accepted = tau.cpu(), accepted.cpu()
    masks = {
        "confidence": confidence >= tau,
        "entropy": entropy < entropy_threshold,
        "mutual_information": mi < mi_threshold,
    }
    masks["confidence_and_entropy"] = masks["confidence"] & masks["entropy"]
    masks["all_gates"] = masks["confidence_and_entropy"] & masks["mutual_information"]
    if not torch.equal(masks["all_gates"], accepted):
        raise ValueError("Diagnostic gates disagree with the actual SSL selection mask")
    rows = []
    for label, name in enumerate(WM811K_CLASSES):
        subset = labels == label
        count, selected = int(subset.sum()), accepted & subset
        accepted_count = int(selected.sum())
        row = {
            "class": name,
            "support": count,
            "gate_pass_counts": {name: int((mask & subset).sum()) for name, mask in masks.items()},
            "accepted_count": accepted_count,
            "accepted_fraction": accepted_count / count if count else None,
            "accepted_accuracy": (
                float((predicted[selected] == labels[selected]).float().mean())
                if accepted_count
                else None
            ),
            "prediction_accuracy": (
                float((predicted[subset] == labels[subset]).float().mean()) if count else None
            ),
            "threshold_at_one_count": int(((tau >= 1) & subset).sum()),
        }
        if count:
            row["quantiles_p10_p50_p90"] = {
                name: torch.quantile(values[subset].float(), torch.tensor([0.1, 0.5, 0.9])).tolist()
                for name, values in (
                    ("confidence", confidence),
                    ("entropy", entropy),
                    ("threshold", tau),
                )
            }
        rows.append(row)
    return {
        "gate_pass_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "entropy_threshold": entropy_threshold,
        "mi_threshold": mi_threshold,
        "threshold_at_one_fraction": float((tau >= 1).float().mean()),
        "per_true_class": rows,
    }


@torch.no_grad()
def pseudo_diagnostic(model, images, labels, args, config):
    """Measure published selection gates without changing them or training on val."""
    selected = torch.cat(
        [torch.where(labels.cpu() == label)[0][: args.mc_samples_per_class] for label in range(9)]
    )
    images = images[selected.to(images.device)]
    labels = labels[selected]
    synchronize(args.device)
    started = time.perf_counter()
    ssl = config.get("semi_supervised", {})
    manager = StageManager(
        model,
        mc_passes=args.mc_passes,
        base_threshold=ssl.get("confidence_threshold", 0.94),
        alpha=ssl.get("alpha", 0.08),
        beta=ssl.get("beta", 0.02),
        entropy_threshold=ssl.get("entropy_threshold", 0.08),
        mi_threshold=ssl.get("mutual_information_threshold", 0.12),
    )
    collected = manager._mc_collect(
        DataLoader(TensorDataset(images.cpu(), labels.cpu()), batch_size=args.batch_size),
        torch.device(args.device),
        None,
        "validation MC diagnostics",
        False,
        with_labels=True,
    )
    accepted, tau = manager._apply_gates(
        collected["conf"], collected["pred"], collected["ent"], collected["mi"]
    )
    n = int(accepted.sum())
    synchronize(args.device)
    return {
        **gate_diagnostics(
            collected,
            accepted,
            tau,
            ssl.get("entropy_threshold", 0.08),
            ssl.get("mutual_information_threshold", 0.12),
        ),
        "adaptive_statistics_source": "held-out saved validation MC subset, not actual unlabeled Du; acceptance does not extrapolate across class-prior shifts",
        "per_predicted_class_confidence_statistics": [
            {
                "class": name,
                "count": int(manager.adaptive_threshold.class_count[label]),
                "mean": float(manager.adaptive_threshold.class_mean[label]),
                "coefficient_of_variation": (
                    float(
                        manager.adaptive_threshold.class_std[label]
                        / manager.adaptive_threshold.class_mean[label].clamp(min=1e-8)
                    )
                    if manager.adaptive_threshold.class_count[label]
                    else None
                ),
            }
            for label, name in enumerate(WM811K_CLASSES)
        ],
        "mc_passes": args.mc_passes,
        "sample_count": len(labels),
        "accepted_count": n,
        "accepted_fraction": n / len(labels),
        "accepted_accuracy": (
            float((collected["pred"][accepted] == collected["y"][accepted]).float().mean())
            if n
            else None
        ),
        "accepted_class_counts": torch.bincount(collected["pred"][accepted], minlength=9).tolist(),
        "mean_confidence": float(collected["conf"].mean()),
        "mean_entropy": float(collected["ent"].mean()),
        "mean_threshold": float(tau.mean()),
        "elapsed_seconds": time.perf_counter() - started,
        "note": "Validation labels only diagnose gate precision; not used for pseudo-training or calibration.",
    }


def run_variant(
    name,
    lr,
    train_x,
    train_y,
    real_x,
    real_y,
    val_x,
    val_y,
    stress_x,
    stress_y,
    args,
    config,
    progress_callback=None,
):
    set_seed(args.seed)
    model = SemiWaferNet.from_config(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0004)
    cpu_rng = torch.Generator().manual_seed(args.seed)
    aug_rng = np.random.RandomState(args.seed + 7)
    counts = torch.bincount(train_y.cpu(), minlength=9).float()
    sample_weights = counts.reciprocal()[train_y.cpu()]
    history, best_score, best_state, steps = [], -1.0, None, 0
    best_step, best_metrics, initial = None, None, None
    losses = []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    synchronize(args.device)
    started = time.perf_counter()

    def snapshot(status, stop_reason=None):
        elapsed = time.perf_counter() - started
        return {
            "name": name,
            "status": status,
            "stop_reason": stop_reason,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "weight_decay": 0.0004,
            "seed": args.seed,
            "data_seed": getattr(args, "data_seed", args.seed),
            "steps": steps,
            "requested_steps": args.steps,
            "best_step": best_step,
            "checkpoint_selection": "saved validation macro-F1 only",
            "best_checkpoint_available": best_state is not None,
            "best_checkpoint": str(output / f"{name}.pt") if best_state is not None else None,
            "train_and_validation_seconds": elapsed,
            "seconds_per_training_step_including_validation": elapsed / max(steps, 1),
            "initial_validation": initial,
            "best_validation": best_metrics,
            "history": list(history),
        }

    def persist(result):
        atomic_json(output / f"{name}.progress.json", result)
        if progress_callback is not None:
            progress_callback(result)

    def checkpoint_best(candidate_state, candidate_step, candidate_metrics):
        temporary = output / f"{name}.pt.tmp"
        torch.save(
            {
                "model": candidate_state,
                "config": config.to_dict(),
                "diagnostic": True,
                "variant": name,
                "best_step": candidate_step,
                "best_validation": candidate_metrics,
                "checkpoint_selection": "saved validation macro-F1 only",
                "diagnostic_options": {**vars(args), "learning_rate": lr},
            },
            temporary,
        )
        temporary.replace(output / f"{name}.pt")

    def record_validation():
        nonlocal best_score, best_state, best_step, best_metrics
        metrics = evaluate(model, val_x, val_y, args.batch_size)
        entry = {"step": steps, "train_batch_loss": sum(losses) / max(len(losses), 1), **metrics}
        history.append(entry)
        losses.clear()
        if metrics["macro_f1"] > best_score:
            candidate_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            checkpoint_best(candidate_state, steps, metrics)
            best_score, best_step, best_metrics = metrics["macro_f1"], steps, metrics
            best_state = candidate_state
        persist(snapshot("running"))
        print(
            f"{name}: step={steps}, loss={entry['train_batch_loss']:.4f}, val_acc={metrics['accuracy']:.4f}, val_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )

    persist(snapshot("initial_validation"))
    try:
        initial = evaluate(model, val_x, val_y, args.batch_size)
        persist(snapshot("running"))
        stop_reason = "requested_steps_completed"
        for step in range(1, args.steps + 1):
            if time.perf_counter() - started >= args.max_seconds and step > 1:
                stop_reason = "time_budget_reached"
                break
            indices = torch.multinomial(
                sample_weights, args.batch_size, replacement=True, generator=cpu_rng
            )
            batch = torch.stack([geometric_augment(train_x[i], aug_rng) for i in indices])
            x, y = batch.to(args.device), train_y[indices].to(args.device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)["classification"]
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {name} loss at step {step}")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite {name} gradient at step {step}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            steps = step
            if step % args.eval_every == 0 or step == args.steps:
                record_validation()
        if not history or history[-1]["step"] != steps:
            record_validation()
        synchronize(args.device)
        result = snapshot("postprocessing", stop_reason)
        persist(result)
        model.load_state_dict(best_state)
        result["real_training_at_best_validation"] = evaluate(
            model, real_x, real_y, args.batch_size
        )
        result["pseudo_gate_diagnostic"] = pseudo_diagnostic(model, val_x, val_y, args, config)
        if args.device == "mps":
            result["mps_memory_bytes"] = {
                "tensor_allocated": torch.mps.current_allocated_memory(),
                "driver_allocated": torch.mps.driver_allocated_memory(),
            }
        if stress_x is not None:
            result["none_stress_validation"] = evaluate(model, stress_x, stress_y, args.batch_size)
        result["status"] = "complete"
        persist(result)
    except KeyboardInterrupt:
        result = snapshot("interrupted", "keyboard_interrupt")
        persist(result)
    except Exception as error:
        result = snapshot("failed", type(error).__name__)
        result["error"] = str(error)
        persist(result)
        raise
    finally:
        del model, optimizer
        if args.device == "mps":
            torch.mps.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="papers/semiwafernet/configs/config.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output", default="checkpoints/mps_diagnostics/semiwafernet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-from",
        default=None,
        help="Reuse exact subsets/data seed from a prior report or directory; sampling-size flags are ignored",
    )
    parser.add_argument(
        "--variant-lr",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Explicit diagnostic learning-rate override for a selected variant",
    )
    parser.add_argument("--train-per-class", type=int, default=180)
    parser.add_argument("--val-per-class", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--eval-every", type=int, default=30)
    parser.add_argument(
        "--max-seconds", type=float, default=120, help="Training+validation cap per variant"
    )
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--mc-samples-per-class", type=int, default=8)
    parser.add_argument(
        "--none-stress-samples",
        type=int,
        default=3000,
        help="Additional held-out official Training None maps; never select checkpoints with this score",
    )
    parser.add_argument("--diverse-none-per-class", type=int, default=720)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_NAMES,
        default=["paper_smote", "fast_smote", "fast_real"],
    )
    args = parser.parse_args()
    if (
        min(args.steps, args.batch_size, args.eval_every, args.mc_passes, args.mc_samples_per_class)
        <= 0
        or args.max_seconds <= 0
    ):
        raise ValueError("Step/batch/MC/time limits must be positive")
    if args.none_stress_samples < 0:
        raise ValueError("--none-stress-samples must be nonnegative")
    if (
        not args.split_from
        and "fast_real_diverse_none" in args.variants
        and args.diverse_none_per_class <= args.train_per_class
    ):
        raise ValueError("Diversity ablation requires --diverse-none-per-class > --train-per-class")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Each diagnostic variant must be selected only once")
    args.device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError(f"{output}/report.json exists; use a new --output")
    set_seed(args.seed)
    config = EngineConfig.from_yaml(args.config)
    if config.get("model.mode") != "classification":
        raise ValueError("This diagnostic measures HybridCNN-ViT classification only")
    root = Path(config.get("data.data_root"))
    samples = parse_wm811k_labeled_rows(root / "labels.csv", split="training")
    rates = variant_learning_rates(
        args.variants, args.variant_lr, float(config.get("optimizer.lr"))
    )
    reused = load_diagnostic_split(args.split_from, samples) if args.split_from else None
    args.data_seed = reused["data_seed"] if reused else args.seed
    train, val = (
        (reused["train"], reused["validation"])
        if reused
        else split_subset(samples, args.train_per_class, args.val_per_class, args.seed)
    )
    encode_started = time.perf_counter()
    real_x = torch.stack(
        [
            encode_wafer_image(_resolve_image_path(root / "images", filename), 32)
            for filename, _ in train
        ]
    )
    real_y = torch.tensor([label for _, label in train], dtype=torch.long)
    val_x = torch.stack(
        [
            encode_wafer_image(_resolve_image_path(root / "images", filename), 32)
            for filename, _ in val
        ]
    )
    val_y = torch.tensor([label for _, label in val], dtype=torch.long)
    used = {filename for filename, _ in train + val}
    if reused:
        extra_none = reused["none_stress"]
    else:
        extra_none = [row for row in samples if row[1] == 0 and row[0] not in used]
        np.random.RandomState(args.seed + 999).shuffle(extra_none)
        extra_none = extra_none[: args.none_stress_samples]
    stress_x = stress_y = None
    if extra_none:
        extra_x = torch.stack(
            [
                encode_wafer_image(_resolve_image_path(root / "images", filename), 32)
                for filename, _ in extra_none
            ]
        )
        stress_x = torch.cat([val_x, extra_x])
        stress_y = torch.cat([val_y, torch.zeros(len(extra_none), dtype=torch.long)])
    diverse_extra_none = reused["diverse_none"] if reused else []
    diverse_x = diverse_y = None
    if "fast_real_diverse_none" in args.variants:
        if reused:
            n_extra = len(diverse_extra_none)
            if not n_extra:
                raise ValueError("The reused split has no diverse-None training subset")
        else:
            used.update(filename for filename, _ in extra_none)
            candidates = [row for row in samples if row[1] == 0 and row[0] not in used]
            np.random.RandomState(args.seed + 1000).shuffle(candidates)
            n_extra = args.diverse_none_per_class - int((real_y == 0).sum())
            diverse_extra_none = candidates[:n_extra]
        if len(diverse_extra_none) != n_extra:
            raise ValueError("Not enough disjoint None maps for the requested diversity ablation")
        diverse_extra_x = torch.stack(
            [
                encode_wafer_image(_resolve_image_path(root / "images", filename), 32)
                for filename, _ in diverse_extra_none
            ]
        )
        diverse_x = torch.cat([real_x, diverse_extra_x])
        diverse_y = torch.cat([real_y, torch.zeros(n_extra, dtype=torch.long)])
    smote_x = smote_y = None
    if any("smote" in variant for variant in args.variants):
        # Bound BLAS/OpenMP work before initializing Metal. This tiny diagnostic
        # needs no CPU thread pool and avoids competing with MPS on unified RAM.
        with threadpool_limits(limits=1):
            ds = SMOTEDataset(root, train, image_size=32, seed=args.data_seed, augment=False)
        smote_x = torch.from_numpy(ds._X)
        smote_y = torch.from_numpy(ds._y)
    val_x = val_x.to(args.device)
    if stress_x is not None:
        stress_x = stress_x.to(args.device)
    report = {
        "status": "running",
        "model": "semiwafernet_classification",
        "device": args.device,
        "torch_version": torch.__version__,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "split_from": reused["source"] if reused else None,
        "training_inventory_digest": inventory_digest(samples),
        "split_reuse_note": (
            "Exact saved roles/data seed reused; --seed controls model initialization and training randomness"
            if reused
            else None
        ),
        "architecture": config.get("model"),
        "parameter_count": sum(p.numel() for p in SemiWaferNet.from_config(config).parameters()),
        "training_counts": torch.bincount(real_y, minlength=9).tolist(),
        "validation_counts": torch.bincount(val_y, minlength=9).tolist(),
        "smote_counts": (
            torch.bincount(smote_y, minlength=9).tolist() if smote_y is not None else None
        ),
        "subset_digest": hashlib.sha256(
            json.dumps({"train": train, "val": val}).encode()
        ).hexdigest(),
        "samples": {"train": train, "validation": val},
        "none_stress_extra_samples": extra_none,
        "diverse_none_extra_training_samples": diverse_extra_none,
        "diverse_none_extra_disjoint_from_base_train_validation_stress": not (
            {f for f, _ in diverse_extra_none} & {f for f, _ in train + val + extra_none}
        ),
        "none_stress_counts": (
            torch.bincount(stress_y, minlength=9).tolist() if stress_y is not None else None
        ),
        "none_stress_none_fraction": (
            float((stress_y == 0).float().mean()) if stress_y is not None else None
        ),
        "none_stress_disjoint_from_training": not (
            {f for f, _ in train} & {f for f, _ in val + extra_none}
        ),
        "extra_none_disjoint_from_balanced_validation": not (
            {f for f, _ in val} & {f for f, _ in extra_none}
        ),
        "preparation_seconds": time.perf_counter() - encode_started,
        "split": "Stratified balanced subsets of official Training; official Test untouched",
        "limits": vars(args),
        "limitations": [
            "Tiny balanced holdout metrics are not official test metrics or proof of paper reproduction.",
            "Fast variants change learning rate for diagnosis; paper config is unchanged.",
            "SSL gates are diagnosed on a tiny held-out subset; full three-stage SSL is not trained.",
            "Miniature batch/data counts change update count and BatchNorm statistics versus full training.",
            "None stress validation simulates a high majority-class prior on unused Training maps; it is not official Test and is never used for checkpoint selection.",
        ],
        "variants": [],
    }
    print(
        f"Device={args.device}; full model={report['parameter_count']} params; train={report['training_counts']}; val={report['validation_counts']}",
        flush=True,
    )
    atomic_json(output / "report.json", report)
    for variant in args.variants:
        lr = rates[variant]
        if variant == "fast_real_diverse_none":
            x, y = diverse_x, diverse_y
            real_samples = train + diverse_extra_none
            real_variant_x, real_variant_y = diverse_x, diverse_y
        elif variant == "fast_real":
            x, y = real_x, real_y
            real_samples = train
            real_variant_x, real_variant_y = real_x, real_y
        else:
            x, y = smote_x, smote_y
            real_samples = train
            real_variant_x, real_variant_y = real_x, real_y
        details = {
            "training_counts": torch.bincount(y.cpu(), minlength=9).tolist(),
            "real_training_counts": torch.bincount(real_variant_y.cpu(), minlength=9).tolist(),
            "real_training_samples": real_samples,
            "real_training_distinct_filenames": len({f for f, _ in real_samples}),
            "real_training_disjoint_from_validation_and_stress": not (
                {f for f, _ in real_samples} & {f for f, _ in val + extra_none}
            ),
        }

        def progress_callback(result):
            snapshot = {**result, **details}
            existing = next(
                (i for i, entry in enumerate(report["variants"]) if entry["name"] == variant), None
            )
            if existing is None:
                report["variants"].append(snapshot)
            else:
                report["variants"][existing] = snapshot
            if result["status"] in {"failed", "interrupted"}:
                report["status"] = result["status"]
            atomic_json(output / "report.json", report)

        result = run_variant(
            variant,
            lr,
            x,
            y,
            real_variant_x.to(args.device),
            real_variant_y,
            val_x,
            val_y,
            stress_x,
            stress_y,
            args,
            config,
            progress_callback=progress_callback,
        )
        if result["status"] == "interrupted":
            raise SystemExit(130)
    report["status"] = "complete"
    atomic_json(output / "report.json", report)
    print(f"Report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
