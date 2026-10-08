"""Summarize all four completed paired Training-only diagnostic runs.

Run from any directory. Aggregate-only is the default; --cpu-bin-inference
additionally evaluates the shared None holdouts on CPU, within a wall-time cap.
This artifact never imports the training entry point or reads official Test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/litobzor-mpl-cache")

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SEEDS = (42, 43)
ARMS = ("natural_none", "stratified_none")
CLASSES = (
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
)
BIN_NAMES = ("under700", "700_to2499", "at_least2500")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def confusion_metrics(matrix):
    require(
        len(matrix) == 9 and all(len(row) == 9 for row in matrix),
        "Expected 9-class confusion",
    )
    rows = [sum(row) for row in matrix]
    columns = [sum(row[c] for row in matrix) for c in range(9)]
    total = sum(rows)
    require(
        total > 0 and all(n > 0 for n in rows),
        "Every validation class must have support",
    )
    per_class = []
    for label, name in enumerate(CLASSES):
        tp = matrix[label][label]
        precision = tp / columns[label] if columns[label] else 0.0
        recall = tp / rows[label]
        f1 = 2 * tp / (rows[label] + columns[label])
        per_class.append(
            {
                "class": name,
                "support": rows[label],
                "predicted_count": columns[label],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    accuracy = sum(matrix[c][c] for c in range(9)) / total
    chance = sum(a * b for a, b in zip(rows, columns)) / total**2
    return {
        "sample_count": total,
        "accuracy": accuracy,
        "macro_f1": statistics.mean(r["f1"] for r in per_class),
        "balanced_accuracy": statistics.mean(r["recall"] for r in per_class),
        "macro_precision": statistics.mean(r["precision"] for r in per_class),
        "cohen_kappa": (accuracy - chance) / (1 - chance) if chance < 1 else None,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_class_order": list(CLASSES),
        "none_to_scratch_count": matrix[0][8],
        "none_to_scratch_fpr": matrix[0][8] / rows[0],
    }


def metric_summary(saved):
    exact = confusion_metrics(saved["confusion_matrix"])
    for metric in ("accuracy", "macro_f1"):
        require(
            abs(saved[metric] - exact[metric]) < 1e-6,
            f"Saved {metric} differs from confusion",
        )
    exact["loss"] = saved["loss"]
    return exact


def distribution(values):
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def read_completed_runs():
    state_path = BASE / "paired_run_state.json"
    state = read_json(state_path)
    require(
        state["status"] == "complete",
        "All four MPS runs must be complete before summarization",
    )
    run_states = {(run["seed"], run["arm"]): run for run in state["runs"]}
    require(
        set(run_states) == {(s, a) for s in SEEDS for a in ARMS},
        "Expected exactly seeds 42/43 and both arms",
    )
    require(
        all(
            run["status"] == "complete" and run["exit_code"] == 0
            for run in run_states.values()
        ),
        "Every training process must have completed successfully",
    )
    builder_path = BASE / "paired_splits/builder_summary.json"
    builder = read_json(builder_path)
    require(
        builder["all_rows_verified_official_Training_only"]
        and builder["holdouts_never_used_in_either_training_arm"],
        "Builder does not establish Training-only disjoint holdouts",
    )
    require(
        builder["metadata_dieSize_verified_against_raw_images_for_all_selected_None"],
        "Raw-size metadata not verified",
    )
    splits = {arm: read_json(BASE / f"paired_splits/{arm}.json") for arm in ARMS}
    for arm in ARMS:
        require(
            sha256_file(BASE / f"paired_splits/{arm}.json")
            == builder["arms"][arm]["sha256"],
            f"Changed {arm} split",
        )
    shared_validation = splits[ARMS[0]]["samples"]["validation"]
    shared_extra = splits[ARMS[0]]["none_stress_extra_samples"]
    require(
        splits[ARMS[1]]["samples"]["validation"] == shared_validation
        and splits[ARMS[1]]["none_stress_extra_samples"] == shared_extra,
        "Holdouts differ between arms",
    )
    require(
        [r for r in splits[ARMS[0]]["samples"]["train"] if r[1] != 0]
        == [r for r in splits[ARMS[1]]["samples"]["train"] if r[1] != 0],
        "Non-None training samples differ",
    )
    reports, results, reference = {}, [], None
    # Guard above runs before importing torch or reading any checkpoint.
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for seed in SEEDS:
        for arm in ARMS:
            path = BASE / f"paired_seed{seed}/{arm}/report.json"
            report = read_json(path)
            require(
                report["status"] == "complete" and len(report["variants"]) == 1,
                f"Incomplete report: {path}",
            )
            variant = report["variants"][0]
            require(
                variant["status"] == "complete" and variant["name"] == "mid_smote",
                "Expected completed mid_smote",
            )
            require(
                variant["steps"] == variant["requested_steps"] == 1800,
                "Update budgets are not matched",
            )
            require(
                variant["seed"] == report["seed"] == seed
                and report["data_seed"] == variant["data_seed"] == 42,
                "Initialization/data seeds differ from intended experiment",
            )
            require(
                report["samples"] == splits[arm]["samples"]
                and report["none_stress_extra_samples"] == shared_extra,
                f"Run {seed}/{arm} does not preserve saved roles",
            )
            require(
                variant["real_training_samples"] == report["samples"]["train"],
                "Unexpected real training samples",
            )
            require(
                variant["real_training_distinct_filenames"]
                == len(report["samples"]["train"]),
                "Distinct training-original count differs from sample lists",
            )
            role_sets = [
                {name for name, _ in rows}
                for rows in (
                    report["samples"]["train"],
                    shared_validation,
                    shared_extra,
                )
            ]
            require(
                all(
                    not role_sets[i] & role_sets[j] for i in range(3) for j in range(i)
                ),
                "Training/holdout overlap",
            )
            require(
                all(
                    len(rows) == len({name for name, _ in rows})
                    for rows in (
                        report["samples"]["train"],
                        shared_validation,
                        shared_extra,
                    )
                ),
                "Duplicate originals",
            )
            fixed = {
                "architecture": report["architecture"],
                "parameter_count": report["parameter_count"],
                "learning_rate": variant["learning_rate"],
                "optimizer": variant["optimizer"],
                "weight_decay": variant["weight_decay"],
                "training_counts": report["training_counts"],
                "smote_counts": report["smote_counts"],
                "limits": {
                    k: report["limits"][k]
                    for k in (
                        "batch_size",
                        "steps",
                        "eval_every",
                        "max_seconds",
                        "mc_passes",
                        "mc_samples_per_class",
                    )
                },
            }
            if reference is None:
                reference = fixed
            require(
                fixed == reference and variant["learning_rate"] == 2e-4,
                "Architecture/optimization/data-count controls differ",
            )
            history = variant["history"]
            require(
                history
                and [row["step"] for row in history] == list(range(150, 1801, 150)),
                "Validation schedule differs",
            )
            best = max(history, key=lambda row: row["macro_f1"])
            require(
                best["step"] == variant["best_step"]
                and best["confusion_matrix"]
                == variant["best_validation"]["confusion_matrix"],
                "Best checkpoint is not earliest validation maximum",
            )
            checkpoint_path = path.parent / "mid_smote.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            require(
                checkpoint["diagnostic"]
                and checkpoint["variant"] == variant["name"]
                and checkpoint["best_step"] == variant["best_step"]
                and checkpoint["best_validation"] == variant["best_validation"],
                "Checkpoint metadata differs from report",
            )
            require(
                checkpoint["diagnostic_options"]["seed"] == seed
                and checkpoint["diagnostic_options"]["data_seed"] == 42
                and checkpoint["diagnostic_options"]["learning_rate"]
                == variant["learning_rate"],
                "Checkpoint seed/LR metadata differs",
            )
            require(
                checkpoint["checkpoint_selection"] == "saved validation macro-F1 only",
                "Unexpected checkpoint-selection rule",
            )
            del checkpoint
            item = {
                "seed": seed,
                "arm": arm,
                "report_path": str(path.relative_to(ROOT)),
                "report_sha256": sha256_file(path),
                "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "best_step": variant["best_step"],
                "last_step": variant["steps"],
                "stop_reason": variant["stop_reason"],
                "training_and_validation_seconds": variant[
                    "train_and_validation_seconds"
                ],
                "process_seconds": run_states[seed, arm]["process_seconds"],
                "preparation_seconds": report["preparation_seconds"],
                "training_distinct_originals": variant[
                    "real_training_distinct_filenames"
                ],
                "real_training_class_counts": variant["real_training_counts"],
                "smote_class_counts": variant["training_counts"],
                "training_none_raw_size_counts": builder["arms"][arm]["training"][
                    "None_size_counts"
                ],
                "initial_validation": metric_summary(variant["initial_validation"]),
                "validation": metric_summary(variant["best_validation"]),
                "stress": metric_summary(variant["none_stress_validation"]),
                "real_training": metric_summary(
                    variant["real_training_at_best_validation"]
                ),
                "mc_gates": variant["pseudo_gate_diagnostic"],
                "history": history,
            }
            results.append(item)
            reports[seed, arm] = report
    return state, builder, splits, reports, results, reference


def aggregate(results):
    lookup = {(r["seed"], r["arm"]): r for r in results}
    names = {
        "accuracy": lambda r: r["accuracy"],
        "macro_f1": lambda r: r["macro_f1"],
        "balanced_accuracy": lambda r: r["balanced_accuracy"],
        "none_f1": lambda r: r["per_class"][0]["f1"],
        "scratch_f1": lambda r: r["per_class"][8]["f1"],
        "none_to_scratch_fpr": lambda r: r["none_to_scratch_fpr"],
    }
    means = {
        arm: {
            subset: {
                name: distribution([extract(lookup[s, arm][subset]) for s in SEEDS])
                for name, extract in names.items()
            }
            for subset in ("validation", "stress")
        }
        for arm in ARMS
    }
    deltas = {
        subset: {
            name: distribution(
                [
                    extract(lookup[s, ARMS[1]][subset])
                    - extract(lookup[s, ARMS[0]][subset])
                    for s in SEEDS
                ]
            )
            for name, extract in names.items()
        }
        for subset in ("validation", "stress")
    }
    return means, deltas


def plot_curves(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        3, 2, figsize=(10, 8), sharex=True, constrained_layout=True
    )
    colors = {ARMS[0]: "#3769a0", ARMS[1]: "#d27633"}
    for column, seed in enumerate(SEEDS):
        for item in (r for r in results if r["seed"] == seed):
            history, arm = item["history"], item["arm"]
            x = [row["step"] for row in history]
            for row_index, key in enumerate(("train_batch_loss", "loss", "macro_f1")):
                y = [row[key] for row in history]
                axes[row_index, column].plot(x, y, label=arm, color=colors[arm])
                if key == "macro_f1":
                    axes[row_index, column].scatter(
                        [item["best_step"]],
                        [item["validation"][key]],
                        color=colors[arm],
                        s=25,
                    )
        axes[0, column].set_title(f"Initialization seed {seed}; same saved holdouts")
        axes[2, column].set_xlabel("Optimizer updates")
        for row_index in range(3):
            axes[row_index, column].grid(alpha=0.2)
    for axis, label in zip(
        axes[:, 0],
        ("Mean sampled-batch training CE", "Validation CE", "Validation macro-F1"),
    ):
        axis.set_ylabel(label)
    axes[0, 0].legend(fontsize=8)
    target = BASE / "paired_validation_curves.png"
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target.name


def bin_result(names, predictions, features):
    rows = []
    for size in BIN_NAMES:
        indices = [
            i
            for i, filename in enumerate(names)
            if features[filename]["size_stratum"] == size
        ]
        count = len(indices)
        false_scratch = sum(predictions[i] == 8 for i in indices)
        any_error = sum(predictions[i] != 0 for i in indices)
        rows.append(
            {
                "raw_size_bin": size,
                "none_count": count,
                "none_to_scratch_count": false_scratch,
                "none_to_scratch_fpr": false_scratch / count if count else None,
                "any_wrong_class_count": any_error,
                "any_wrong_class_rate": any_error / count if count else None,
            }
        )
    return rows


def cpu_bin_inference(summary, builder, splits, reports, max_seconds):
    """One shared encoded pack, four CPU models, cached all-class logits."""
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    sys.path.insert(0, str(ROOT))
    from common.engine.config import EngineConfig
    from papers.semiwafernet.data_utils.wafer_dataset import (
        encode_wafer_image,
        _resolve_image_path,
    )
    from papers.semiwafernet.models.semiwafernet import SemiWaferNet

    started = time.monotonic()
    progress_path = BASE / "paired_cpu_bins.progress.json"
    inference = {
        "status": "running",
        "backend": "cpu",
        "torch_threads": 1,
        "max_seconds": max_seconds,
        "runs": [],
        "note": "Raw-size bins are descriptive only; no checkpoint or hyperparameter selection uses them.",
    }
    summary["cpu_none_size_bins"] = inference

    def persist(phase):
        inference["phase"] = phase
        inference["elapsed_seconds"] = time.monotonic() - started
        atomic_json(progress_path, inference)
        atomic_json(BASE / "paired_summary.json", summary)
        print(
            f"CPU bin diagnostic: {phase}; {inference['elapsed_seconds']:.1f}s",
            flush=True,
        )

    def check_budget():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("CPU bin diagnostic reached its wall-time cap")

    shared = splits[ARMS[0]]
    validation_rows = shared["samples"]["validation"]
    all_rows = validation_rows + shared["none_stress_extra_samples"]
    names = [filename for filename, _ in all_rows]
    labels = np.asarray([label for _, label in all_rows], dtype=np.int64)
    validation_count = len(validation_rows)
    validation_none_indices = np.flatnonzero(labels[:validation_count] == 0)
    none_indices = np.flatnonzero(labels == 0)
    validation_names = [names[i] for i in validation_none_indices]
    none_names = [names[i] for i in none_indices]
    features = builder["selected_None_raw_features"]
    require(
        len(names) == len(set(names))
        and all(features[n]["label"] == 0 for n in none_names),
        "Invalid shared holdout pack",
    )
    require(
        all(
            features[n]["size_stratum"]
            == BIN_NAMES[
                (
                    0
                    if features[n]["raw_active_dies"] < 700
                    else 1 if features[n]["raw_active_dies"] < 2500 else 2
                )
            ]
            for n in none_names
        ),
        "Raw bin definitions changed",
    )
    pack_digest = digest_json(
        [[n, int(label), features.get(n)] for n, label in all_rows]
    )
    inference.update(
        shared_holdout_count=len(names),
        shared_validation_count=validation_count,
        shared_none_count=len(none_names),
        shared_validation_none_count=len(validation_names),
        shared_roles_and_raw_features_sha256=pack_digest,
        bin_definitions=builder["bin_definition"],
    )
    persist("encoding_shared_holdout_pack")
    try:
        first = torch.load(
            ROOT / summary["runs"][0]["checkpoint_path"],
            map_location="cpu",
            weights_only=False,
        )
        data_root = Path(EngineConfig(first["config"]).get("data.data_root"))
        if not data_root.is_absolute():
            data_root = ROOT / data_root
        require(
            sha256_file(data_root / "labels.csv") == builder["labels_csv_sha256"],
            "Dataset labels changed",
        )
        del first
        tensors, encoded_hash = [], hashlib.sha256()
        for i, filename in enumerate(names):
            check_budget()
            encoded = encode_wafer_image(
                _resolve_image_path(data_root / "images", filename), 32
            )
            tensors.append(encoded)
            encoded_hash.update(encoded.numpy().tobytes())
            if (i + 1) % 2000 == 0:
                inference["encoded_count"] = i + 1
                persist("encoding_shared_holdout_pack")
        images = torch.stack(tensors)
        del tensors
        inference["encoded_pack_sha256"] = encoded_hash.hexdigest()
        persist("shared_holdout_pack_ready")
        with threadpool_limits(limits=1), torch.inference_mode():
            for item in summary["runs"]:
                check_budget()
                seed, arm = item["seed"], item["arm"]
                identity = {
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    "encoded_pack_sha256": encoded_hash.hexdigest(),
                    "shared_roles_and_raw_features_sha256": pack_digest,
                    "backend": "cpu",
                }
                cache = BASE / f"paired_seed{seed}/{arm}/cpu_holdout_logits.json"
                logits_path = cache.with_suffix(".npz")
                cached = read_json(cache) if cache.exists() else None
                if (
                    cached is not None
                    and cached.get("identity") == identity
                    and logits_path.exists()
                    and cached.get("npz_sha256") == sha256_file(logits_path)
                ):
                    with np.load(logits_path, allow_pickle=False) as saved:
                        require(
                            saved["filenames"].tolist() == names
                            and np.array_equal(saved["true_labels"], labels),
                            "Cached logit holdout roles differ",
                        )
                        logits = saved["logits"].copy()
                    require(
                        logits.shape == (len(names), 9) and np.isfinite(logits).all(),
                        "Incomplete cached logits",
                    )
                    seconds, reused = cached["inference_seconds"], True
                else:
                    checkpoint = torch.load(
                        ROOT / item["checkpoint_path"],
                        map_location="cpu",
                        weights_only=False,
                    )
                    model = SemiWaferNet.from_config(EngineConfig(checkpoint["config"]))
                    model.load_state_dict(checkpoint["model"], strict=True)
                    model.eval()
                    del checkpoint
                    model_started = time.monotonic()
                    logit_chunks = []
                    for batch in images.split(64):
                        check_budget()
                        logit_chunks.append(model(batch)["classification"].numpy())
                    logits = np.concatenate(logit_chunks).astype(np.float32, copy=False)
                    require(np.isfinite(logits).all(), "Non-finite CPU logits")
                    seconds, reused = time.monotonic() - model_started, False
                    temporary = logits_path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            filenames=np.asarray(names),
                            true_labels=labels,
                            logits=logits,
                        )
                    temporary.replace(logits_path)
                    atomic_json(
                        cache,
                        {
                            "identity": identity,
                            "checkpoint_sha256": item["checkpoint_sha256"],
                            "npz_sha256": sha256_file(logits_path),
                            "npz_path": str(logits_path.relative_to(ROOT)),
                            "validation_count": validation_count,
                            "sample_count": len(names),
                            "arrays": {
                                "filenames": "Unicode[N]",
                                "true_labels": "int64[N]",
                                "logits": "float32[N,9]",
                            },
                            "class_order": list(CLASSES),
                            "raw_none_features_path": str(
                                (
                                    BASE / "paired_splits/builder_summary.json"
                                ).relative_to(ROOT)
                            ),
                            "raw_none_features_key": "selected_None_raw_features",
                            "inference_seconds": seconds,
                        },
                    )
                    del model
                predictions = logits.argmax(axis=1)
                none_predictions = predictions[none_indices].tolist()
                counts = [none_predictions.count(label) for label in range(9)]
                expected = item["stress"]["confusion_matrix"][0]
                result = {
                    "seed": seed,
                    "arm": arm,
                    "identity": identity,
                    "cache": str(cache.relative_to(ROOT)),
                    "logits_npz": str(logits_path.relative_to(ROOT)),
                    "logits_npz_sha256": sha256_file(logits_path),
                    "inference_seconds": seconds,
                    "reused_cache": reused,
                    "validation": bin_result(
                        validation_names,
                        predictions[validation_none_indices].tolist(),
                        features,
                    ),
                    "stress": bin_result(none_names, none_predictions, features),
                    "cpu_none_prediction_counts": counts,
                    "saved_mps_none_prediction_counts": expected,
                    "cpu_matches_saved_mps_none_confusion_row": counts == expected,
                }
                inference["runs"].append(result)
                persist(f"finished_seed{seed}_{arm}")
        inference["status"] = "complete"
        lookup = {(r["seed"], r["arm"]): r for r in inference["runs"]}
        inference["paired_stratified_minus_natural_fpr"] = {
            subset: {
                size: distribution(
                    [
                        lookup[s, ARMS[1]][subset][index]["none_to_scratch_fpr"]
                        - lookup[s, ARMS[0]][subset][index]["none_to_scratch_fpr"]
                        for s in SEEDS
                    ]
                )
                for index, size in enumerate(BIN_NAMES)
            }
            for subset in ("validation", "stress")
        }
        persist("complete")
    except (TimeoutError, KeyboardInterrupt) as error:
        inference["status"] = (
            "budget_reached" if isinstance(error, TimeoutError) else "interrupted"
        )
        inference["stop_reason"] = str(error) or "keyboard_interrupt"
        persist(inference["status"])
    except Exception as error:
        inference["status"], inference["error"] = "failed", str(error)
        persist("failed")
        raise


def write_markdown(summary):
    lines = [
        "# Paired SemiWaferNet short diagnostics",
        "",
        "Four matched 1,800-update runs use the full classification architecture, LR 0.0002 and fixed source-data seed 42. "
        "Initialization seeds 42 and 43 each compare natural None sampling with size-stratified None sampling. "
        "Validation and stress filenames are identical across all four runs; non-None training roles are unchanged. "
        "All images come from official Training. No official Test metrics are reported.",
        "",
        "Checkpoints are selected by the earliest maximum of saved validation macro-F1. "
        "The shared stress score and size bins are descriptive. Values below are percentages.",
        "",
        "| Seed | Arm | Best update | Val accuracy | Val macro-F1 | Stress accuracy | Stress macro-F1 | Val None F1 | Val Scratch F1 | Process seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["runs"]:
        v, s = r["validation"], r["stress"]
        lines.append(
            f"| {r['seed']} | {r['arm']} | {r['best_step']} | {v['accuracy']*100:.2f} | {v['macro_f1']*100:.2f} | "
            f"{s['accuracy']*100:.2f} | {s['macro_f1']*100:.2f} | {v['per_class'][0]['f1']*100:.2f} | "
            f"{v['per_class'][8]['f1']*100:.2f} | {r['process_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "| Shared subset | Metric | Mean paired delta, stratified minus natural (pp) | Sample SD (pp) |",
            "|---|---|---:|---:|",
        ]
    )
    for subset, metrics in summary["paired_deltas"].items():
        for name, values in metrics.items():
            lines.append(
                f"| {subset} | {name} | {values['mean']*100:+.2f} | {values['sample_sd']*100:.2f} |"
            )
    lines.extend(
        [
            "",
            "MC gate diagnostics use the first saved validation samples per class, with the published thresholds unchanged. "
            "Their adaptive confidence statistics come from this held-out subset rather than real unlabeled Du, so coverage does not establish full SSL quality.",
            "",
            "| Seed | Arm | MC samples | Confidence pass | Entropy pass | MI pass | All gates | Accepted accuracy |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in summary["runs"]:
        gate = r["mc_gates"]
        counts = gate["gate_pass_counts"]
        precision = (
            "—"
            if gate["accepted_accuracy"] is None
            else f"{gate['accepted_accuracy']*100:.2f}%"
        )
        lines.append(
            f"| {r['seed']} | {r['arm']} | {gate['sample_count']} | {counts['confidence']} | "
            f"{counts['entropy']} | {counts['mutual_information']} | {counts['all_gates']} | {precision} |"
        )
    bins = summary.get("cpu_none_size_bins")
    if bins:
        lines.extend(
            [
                "",
                f"CPU raw-size diagnostic status: **{bins['status']}**. "
                "False-positive rates below mean predicted Scratch among true None within each original occupied-die-count bin.",
                "",
                "| Seed | Arm | Holdout | Raw occupied dies | None count | None→Scratch count | FPR |",
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for r in bins["runs"]:
            for subset in ("validation", "stress"):
                for row in r[subset]:
                    fpr = (
                        "—"
                        if row["none_to_scratch_fpr"] is None
                        else f"{row['none_to_scratch_fpr']*100:.2f}%"
                    )
                    lines.append(
                        f"| {r['seed']} | {r['arm']} | {subset} | {row['raw_size_bin']} | "
                        f"{row['none_count']} | {row['none_to_scratch_count']} | {fpr} |"
                    )
        if any(not r["cpu_matches_saved_mps_none_confusion_row"] for r in bins["runs"]):
            lines.extend(
                [
                    "",
                    "CPU predictions differ from the saved MPS None confusion row for at least one run; "
                    "size-bin numbers use the CPU backend and aggregate metrics above use the saved MPS reports.",
                ]
            )
    else:
        lines.extend(
            [
                "",
                "Raw-size FPR awaits the optional CPU inference pass (`--cpu-bin-inference`).",
            ]
        )
    lines.extend(
        [
            "",
            f"![Matched validation curves]({summary['validation_curves_png']})",
            "",
            "Exact checkpoint hashes, all per-class metrics/confusions, run timings, gate counts and paired means/sample SD are in `paired_summary.json`. "
            "Two initialization seeds provide a diagnostic comparison, not a reliable population uncertainty estimate or paper reproduction.",
        ]
    )
    (BASE / "paired_summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-bin-inference", action="store_true")
    parser.add_argument("--max-inference-seconds", type=float, default=60)
    args = parser.parse_args()
    require(
        args.max_inference_seconds > 0 and math.isfinite(args.max_inference_seconds),
        "Positive finite inference budget required",
    )
    started = time.monotonic()
    state, builder, splits, reports, results, controls = read_completed_runs()
    means, deltas = aggregate(results)
    summary = {
        "status": "complete",
        "scope": "Official Training-only paired miniature diagnostics; official Test untouched",
        "checkpoint_selection": "Earliest maximum saved validation macro-F1; stress and bins are descriptive",
        "matched_controls": controls,
        "initialization_seeds": list(SEEDS),
        "source_data_seed": 42,
        "identical_validation_and_stress_roles": True,
        "identical_non_none_training_roles": True,
        "no_train_holdout_overlap_or_duplicate_originals": True,
        "paired_run_state_sha256": sha256_file(BASE / "paired_run_state.json"),
        "builder_summary_sha256": sha256_file(
            BASE / "paired_splits/builder_summary.json"
        ),
        "summarizer_sha256": sha256_file(Path(__file__)),
        "shared_validation": builder["shared_validation"],
        "shared_stress_extra": builder["shared_stress_extra"],
        "split_sha256": {arm: builder["arms"][arm]["sha256"] for arm in ARMS},
        "runs": results,
        "arm_means_sample_sd": means,
        "paired_deltas": deltas,
        "validation_curves_png": plot_curves(results),
        "limitations": [
            "Two seeds; uncertainty estimates are descriptive only.",
            "Short supervised warm-up only; full three-stage SSL not trained.",
            "Sampling and learning rate are explicit diagnostics, not changes to the paper defaults.",
            "Shared None validation is deliberately size-stratified; stress prior is artificial.",
        ],
    }
    previous = BASE / "paired_summary.json"
    if previous.exists():
        old = read_json(previous)
        if old.get("builder_summary_sha256") == summary["builder_summary_sha256"] and [
            r["checkpoint_sha256"] for r in old.get("runs", [])
        ] == [r["checkpoint_sha256"] for r in results]:
            if "cpu_none_size_bins" in old:
                summary["cpu_none_size_bins"] = old["cpu_none_size_bins"]
    atomic_json(previous, summary)
    write_markdown(summary)
    if args.cpu_bin_inference:
        cpu_bin_inference(summary, builder, splits, reports, args.max_inference_seconds)
        write_markdown(summary)
    summary["summarizer_seconds"] = time.monotonic() - started
    atomic_json(previous, summary)
    print(f"Saved {previous}, paired_summary.md and {summary['validation_curves_png']}")


if __name__ == "__main__":
    main()
