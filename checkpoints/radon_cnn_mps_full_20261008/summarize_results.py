"""CPU-only audit of all three finished RadonCNN runs; never select a seed.

Run from the repository root after run_state.json says all runs are complete:

    python checkpoints/radon_cnn_mps_full_20261008/summarize_results.py

This refuses evaluation while training/evaluation is still active. All model
selection is the existing per-run minimum validation-loss best.pt; test labels
are used only for final reporting. Different seeds have different splits.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Set before importing numpy/torch/matplotlib; no GPU operations are performed.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/litobzor-mpl-cache")

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
EXPECTED_SEEDS = [42, 43, 44]


def require_finished_runs(directory):
    state_path = directory / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    runs = {int(run["seed"]): run for run in state.get("runs", [])}
    if state.get("seeds") != EXPECTED_SEEDS:
        raise RuntimeError("Expected declared seeds [42, 43, 44]")
    if (
        state.get("status") != "complete"
        or len(runs) != 3
        or any(runs.get(seed, {}).get("status") != "complete" for seed in EXPECTED_SEEDS)
    ):
        statuses = {
            seed: runs.get(seed, {}).get("status", "not_started") for seed in EXPECTED_SEEDS
        }
        raise RuntimeError(f"Evaluation refused until ALL runs complete: {statuses}")
    if "sample-weighted" not in state.get("validation_loss", "").lower():
        raise RuntimeError(
            "Only the corrected sample-weighted validation runs belong in this summary"
        )
    if any(
        runs[seed].get("training_exit_code") != 0 or runs[seed].get("evaluation_exit_code") != 0
        for seed in EXPECTED_SEEDS
    ):
        raise RuntimeError("A training or standalone evaluation did not finish successfully")
    return state, runs


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def normalized_config(config):
    result = copy.deepcopy(config)
    result["training"].pop("seed", None)
    result["checkpoint"].pop("save_dir", None)
    return result


def exact_metrics(confusion, class_names):
    import numpy as np

    diagonal = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    precision = np.divide(
        diagonal, predicted_support, out=np.zeros_like(diagonal), where=predicted_support > 0
    )
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    f1 = np.divide(
        2 * diagonal,
        support + predicted_support,
        out=np.zeros_like(diagonal),
        where=support + predicted_support > 0,
    )
    return {
        "samples": int(confusion.sum()),
        "correct": int(diagonal.sum()),
        "accuracy": float(diagonal.sum() / confusion.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "support": int(support[index]),
                "predicted_support": int(predicted_support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion.tolist(),
        "confusion_rows": "true class",
        "confusion_columns": "predicted class",
        "class_order": class_names,
    }


def evaluate_cpu(model, dataset, indices, batch_size, prediction_path):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0
    )
    targets, predictions, scores = [], [], []
    summed_loss, batch_mean_losses = 0.0, []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = batch["inputs"].cpu(), batch["targets"].cpu()
            logits = model(inputs)
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("Non-finite logits in finished best checkpoint")
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            summed_loss += losses.sum().item()
            batch_mean_losses.append(losses.mean().item())
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(1).tolist())
            scores.extend(logits.softmax(1).amax(1).tolist())
    size = len(dataset.class_names)
    confusion = np.bincount(
        np.asarray(targets) * size + np.asarray(predictions), minlength=size * size
    ).reshape(size, size)
    result = exact_metrics(confusion, dataset.class_names)
    result.update(
        sample_mean_cross_entropy=summed_loss / len(indices),
        legacy_mean_of_batch_losses=float(np.mean(batch_mean_losses)),
        evaluation_seconds=time.perf_counter() - started,
        batch_size=batch_size,
        batches=len(batch_mean_losses),
        final_batch_samples=len(indices) % batch_size or batch_size,
    )
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_index",
                "filename",
                "true_label",
                "true_class",
                "predicted_label",
                "predicted_class",
                "max_probability",
            ]
        )
        for index, truth, prediction, score in zip(
            indices, targets, predictions, scores, strict=True
        ):
            filename, expected_label = dataset._samples[index]
            if truth != expected_label:
                raise ValueError("Loader target disagrees with the saved sample inventory")
            writer.writerow(
                [
                    index,
                    filename,
                    truth,
                    dataset.class_names[truth],
                    prediction,
                    dataset.class_names[prediction],
                    score,
                ]
            )
    return result


def reconcile_standalone(log_path, computed):
    text = log_path.read_text(encoding="utf-8")
    names = {
        "accuracy": "Accuracy",
        "macro_f1": "F1 Score",
        "macro_precision": "Precision",
        "macro_recall": "Recall",
    }
    comparison = {}
    for metric, label in names.items():
        match = re.search(rf"^\s*{re.escape(label)}:\s*([0-9.]+)\s*$", text, re.MULTILINE)
        if not match:
            comparison[metric] = {"available": False}
            continue
        logged = float(match.group(1))
        delta = computed[metric] - logged
        comparison[metric] = {
            "available": True,
            "logged_rounded": logged,
            "cpu_exact": computed[metric],
            "difference": delta,
            "within_four_decimal_rounding": abs(delta) <= 0.0000502,
        }
    return comparison


def statistics(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sample_standard_deviation": float(values.std(ddof=1)),
        "mean_percent": float(values.mean() * 100),
        "sample_standard_deviation_percentage_points": float(values.std(ddof=1) * 100),
    }


def plot_histories(seed_results, directory):
    try:
        import matplotlib
    except ImportError:
        return {"created": False, "reason": "matplotlib is not installed"}
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    for row, result in enumerate(seed_results):
        history = result["history"]
        epochs = np.arange(1, len(history) + 1)
        loss_axis, accuracy_axis = axes[row]
        loss_axis.plot(
            epochs,
            [entry["train_loss"] for entry in history],
            label="Training loss",
            color="#2878b5",
        )
        loss_axis.plot(
            epochs,
            [entry["val_loss"] for entry in history],
            label="Validation loss",
            color="#db6d28",
        )
        accuracy_axis.plot(
            epochs,
            [entry["val_accuracy"] * 100 for entry in history],
            label="Validation accuracy",
            color="#2878b5",
        )
        for axis in (loss_axis, accuracy_axis):
            axis.axvline(
                result["best_epoch"],
                color="#358b4a",
                linestyle="--",
                label=f"Best validation loss: epoch {result['best_epoch']}",
            )
            axis.set_xlabel("Completed epoch")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        loss_axis.set_ylabel("Cross entropy (sample mean)")
        accuracy_axis.set_ylabel("Validation accuracy (%)")
        accuracy_axis.set_ylim(0, 100)
        loss_axis.set_title(f"Seed {result['seed']} — train and validation loss")
        accuracy_axis.set_title(
            f"Seed {result['seed']} — validation accuracy; stopped at epoch {result['stopped_epoch']}"
        )
    figure.suptitle(
        "RadonCNN: all three fixed-seed runs; checkpoint selection uses validation loss",
        fontsize=14,
    )
    path = directory / "training_curves.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {
        "created": True,
        "path": str(path.relative_to(ROOT)),
        "panels": "Three seed rows: training/validation loss and validation accuracy",
    }


def write_markdown(report, path):
    lines = [
        "# Full RadonCNN fixed-seed experiment",
        "",
        "Each seed uses its own saved, disjoint train/validation/test split. All three seeds are included; no seed is chosen by test performance and no ensemble is evaluated.",
        "",
        "| Seed | Best epoch | Stopped epoch | Unique train wafers | Test accuracy | Test macro-F1 | Training process (min) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        lines.append(
            f"| {run['seed']} | {run['best_epoch']} | {run['stopped_epoch']} | {run['unique_samples']['train']} | {run['test']['accuracy'] * 100:.4f}% | {run['test']['macro_f1'] * 100:.4f}% | {run['training_process_seconds'] / 60:.2f} |"
        )
    lines += [
        "",
        "Across all three seeds (mean ± sample standard deviation, in percentage points):",
        "",
    ]
    for metric, values in report["aggregate"]["test"].items():
        lines.append(
            f"- {metric}: {values['mean_percent']:.4f}% ± {values['sample_standard_deviation_percentage_points']:.4f} pp"
        )
    lines += [
        "",
        "Maximum 500 epochs, early stopping patience 30; 6400 training presentations with training-only replacement. Balanced validation and test remain unique and disjoint from training. Different seeds change both initialization and saved split.",
        "",
        "Before these runs, RadonCNN validation loss was corrected to sample-weighted cross entropy. The prior unweighted mean of batch losses gave a final singleton disproportionate weight. The superseded run was stopped before test evaluation and is excluded. Both loss definitions are retained in the audit, with checkpoint selection verified against the corrected logged sample mean.",
        "",
        "Exact per-class metrics and confusion matrices: `summary.json`; individual predictions: each seed's `cpu_validation_predictions.csv` / `cpu_test_predictions.csv`. The training curves show all completed epochs and validation-based checkpoint selection.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=BASE)
    args = parser.parse_args()
    directory = args.experiment_dir.resolve()
    try:
        state, run_records = require_finished_runs(directory)
    except (RuntimeError, FileNotFoundError) as error:
        parser.exit(2, str(error) + "\n")

    # Imports and substantive evaluation happen ONLY after the finished guard.
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import yaml
    from common.engine.config import EngineConfig
    from papers.radon_cnn.data_utils import WaferRadonDataset
    from papers.radon_cnn.data_utils.protocol import dataset_options, load_manifest, model_state
    from papers.radon_cnn.models.radon_cnn import RadonCNN

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.chdir(ROOT)
    started = time.perf_counter()
    results, reference_config, dataset = [], None, None
    for seed in EXPECTED_SEEDS:
        run_record = run_records[seed]
        run_dir = (ROOT / run_record["directory"]).resolve()
        paths = {
            name: run_dir / name
            for name in (
                "config.yaml",
                "experiment.yaml",
                "split.json",
                "history.json",
                "best.pt",
                "last.pt",
                "train.log",
                "evaluate.log",
            )
        }
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"Missing input artifact for seed {seed}")
        config_dict = yaml.safe_load(paths["experiment.yaml"].read_text(encoding="utf-8"))
        input_config = yaml.safe_load(paths["config.yaml"].read_text(encoding="utf-8"))
        if config_dict != input_config:
            raise ValueError(f"Saved experiment differs from declared config for seed {seed}")
        if config_dict["training"]["seed"] != seed:
            raise ValueError("Config seed mismatch")
        if (
            config_dict["training"]["num_epochs"] != 500
            or config_dict["training"]["early_stopping_patience"] != 30
        ):
            raise ValueError("Expected declared maximum 500 epochs and patience 30")
        if reference_config is None:
            reference_config = normalized_config(config_dict)
        elif normalized_config(config_dict) != reference_config:
            raise ValueError(
                "Model/training/data settings differ across seeds beyond seed and directory"
            )
        config = EngineConfig.from_yaml(paths["experiment.yaml"])
        if dataset is None:
            dataset = WaferRadonDataset(**dataset_options(config))
        manifest = load_manifest(dataset, paths["split.json"])
        if manifest["seed"] != seed or manifest["protocol"] != config_dict["data"]["protocol"]:
            raise ValueError("Saved split seed/protocol mismatch")
        presentations, unique_samples, class_counts, unique_class_counts = {}, {}, {}, {}
        for split in ("train", "val", "test"):
            indices = manifest["indices"][split]
            counts = Counter(dataset._samples[index][1] for index in indices)
            unique_counts = Counter(dataset._samples[index][1] for index in set(indices))
            if {str(key): value for key, value in counts.items()} != manifest["counts"][split]:
                raise ValueError("Stored class counts disagree with actual manifest indices")
            if len(set(indices)) != manifest["unique_counts"][split]:
                raise ValueError("Stored unique count disagrees with manifest")
            presentations[split], unique_samples[split] = len(indices), len(set(indices))
            class_counts[split] = {dataset.class_names[key]: counts[key] for key in range(7)}
            unique_class_counts[split] = {
                dataset.class_names[key]: unique_counts[key] for key in range(7)
            }
        best = torch.load(paths["best.pt"], map_location="cpu", weights_only=False)
        last = torch.load(paths["last.pt"], map_location="cpu", weights_only=False)
        history = json.loads(paths["history.json"].read_text(encoding="utf-8"))
        best_epoch, stopped_epoch = int(best["epoch"]), int(last["epoch"])
        if not history or stopped_epoch != len(history) or not 1 <= best_epoch <= stopped_epoch:
            raise ValueError("Checkpoint epochs do not match persisted history")
        validation_losses = [entry["val_loss"] for entry in history]
        if best_epoch != int(np.argmin(validation_losses)) + 1:
            raise ValueError("best.pt was not selected by minimum logged validation loss")
        if best.get("metric_name") != "val_loss" or not math.isclose(
            best["metric"], min(validation_losses), rel_tol=1e-7, abs_tol=1e-8
        ):
            raise ValueError("Best checkpoint metric does not match validation history")
        model = RadonCNN(in_channels=1, num_classes=7).cpu()
        model.load_state_dict(model_state(best), strict=True)
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise RuntimeError("This summarizer must never evaluate on GPU")
        print(
            f"CPU audit seed {seed}: best epoch {best_epoch}, stopped epoch {stopped_epoch}, validation+test",
            flush=True,
        )
        batch_size = int(config.get("evaluation.batch_size", 64))
        validation = evaluate_cpu(
            model,
            dataset,
            manifest["indices"]["val"],
            batch_size,
            run_dir / "cpu_validation_predictions.csv",
        )
        test = evaluate_cpu(
            model,
            dataset,
            manifest["indices"]["test"],
            batch_size,
            run_dir / "cpu_test_predictions.csv",
        )
        train_log = paths["train.log"].read_text(encoding="utf-8")
        stop_match = re.search(r"Early stopping triggered at epoch (\d+)", train_log)
        if stop_match and int(stop_match.group(1)) != stopped_epoch:
            raise ValueError("Early-stopping log epoch disagrees with last checkpoint")
        record = {
            "seed": seed,
            "directory": str(run_dir.relative_to(ROOT)),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "stop_reason": "early_stopping" if stop_match else "maximum_epochs",
            "maximum_epochs": 500,
            "early_stopping_patience": 30,
            "training_process_seconds": run_record["training_process_seconds"],
            "standalone_evaluation_process_seconds": run_record["evaluation_process_seconds"],
            "summed_epoch_seconds": sum(entry["epoch_time"] for entry in history),
            "training_presentations_per_epoch": presentations["train"],
            "split_sizes": presentations,
            "unique_samples": unique_samples,
            "class_counts": class_counts,
            "unique_class_counts": unique_class_counts,
            "best_logged_validation_loss": best["metric"],
            "cpu_validation_sample_loss_difference": validation["sample_mean_cross_entropy"]
            - best["metric"],
            "cpu_validation_sample_loss_matches_logged": math.isclose(
                validation["sample_mean_cross_entropy"], best["metric"], rel_tol=1e-4, abs_tol=2e-5
            ),
            "best_and_last_weights_identical": all(
                torch.equal(value, model_state(last)[name])
                for name, value in model_state(best).items()
            ),
            "validation": validation,
            "test": test,
            "history": history,
            "standalone_test_log_comparison": reconcile_standalone(paths["evaluate.log"], test),
            "input_sha256": {name: file_sha256(path) for name, path in paths.items()},
        }
        results.append(record)
        print(
            f"seed {seed}: accuracy={test['correct']}/{test['samples']}={test['accuracy']:.10f}, macro-F1={test['macro_f1']:.10f}",
            flush=True,
        )
        del model, best, last
    keys = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
    aggregate = {
        split: {key: statistics([run[split][key] for run in results]) for key in keys}
        for split in ("validation", "test")
    }
    per_class = {
        name: {
            metric: statistics([run["test"]["per_class"][name][metric] for run in results])
            for metric in ("precision", "recall", "f1")
        }
        for name in dataset.class_names
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_device": "cpu",
        "torch_threads": 2,
        "torch_interop_threads": 1,
        "blas_threads": 1,
        "torch_version": torch.__version__,
        "seeds": EXPECTED_SEEDS,
        "runs_included": 3,
        "run_state": state,
        "selection": "All three declared seeds, each own minimum-validation-loss checkpoint; no seed selected by test and no pooled ensemble",
        "split_caveat": "Seeds change both initialization and independent disjoint splits; mean/sample SD are across runs, not a pooled-test or ensemble score",
        "validation_loss_fix": "RadonCNN validation/checkpoint selection uses sample-weighted cross entropy. The prior singleton-overweighted run was stopped before test evaluation and excluded; legacy batch-mean loss is an optional audit statistic only",
        "settings_without_seed_and_directory": reference_config,
        "runs": results,
        "aggregate": aggregate,
        "aggregate_test_per_class": per_class,
        "source_sha256": {
            filename: file_sha256(ROOT / filename)
            for filename in (
                "papers/radon_cnn/models/radon_cnn.py",
                "papers/radon_cnn/modules/kernel_flip.py",
                "papers/radon_cnn/trainer.py",
                "papers/radon_cnn/data_utils/dataset.py",
            )
        },
    }
    report["training_curves"] = plot_histories(results, directory)
    report["summarization_seconds"] = time.perf_counter() - started
    output = directory / "summary.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_markdown(report, directory / "summary.md")
    with (directory / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["seed", "split", "class", "support", "predicted_support", "precision", "recall", "f1"]
        )
        for run in results:
            for split in ("validation", "test"):
                for name in dataset.class_names:
                    values = run[split]["per_class"][name]
                    writer.writerow(
                        [
                            run["seed"],
                            split,
                            name,
                            values["support"],
                            values["predicted_support"],
                            values["precision"],
                            values["recall"],
                            values["f1"],
                        ]
                    )
    print(json.dumps(aggregate["test"], indent=2), flush=True)
    print("Saved", output, directory / "summary.md", directory / "training_curves.png", flush=True)


if __name__ == "__main__":
    main()
