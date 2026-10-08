"""Fixed paper gates with class CV measured on real unlabeled WM-811K.

Artifact-only investigation: no training, no test images, no gate calibration.
The entire supplied validation manifest is descriptive; it never updates the
real-Du adaptive statistics. A second, explicitly labeled self-CV calculation
reuses exactly the same validation MC predictions for comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.engine.config import EngineConfig
from common.training.utils import resolve_device
from common.utils.seed import set_seed
from papers.semiwafernet.data_utils.wafer_dataset import (
    WM811K_CLASSES,
    _resolve_image_path,
    encode_wafer_image,
    parse_wm811k_labeled_rows,
)
from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.training.stage_manager import StageManager
from papers.semiwafernet.utils.checkpoint import normalize_semiwafernet_state_dict


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def statistics(manager):
    adaptive = manager.adaptive_threshold
    rows = []
    for label, name in enumerate(WM811K_CLASSES):
        count = int(adaptive.class_count[label])
        mean, std = float(adaptive.class_mean[label]), float(adaptive.class_std[label])
        rows.append({"class": name, "predicted_candidates": count,
                     "confidence_mean": mean if count else None,
                     "confidence_population_std": std if count else None,
                     "coefficient_of_variation": std / max(mean, 1e-8) if count else 0.0,
                     "unobserved_class_uses_zero_CV": count == 0})
    return rows


def summarize(pack, accepted, tau, manager):
    """Report gate mechanics and labeled precision without altering statistics."""
    conf, pred, ent, mi = (pack[key].cpu() for key in ("conf", "pred", "ent", "mi"))
    accepted, tau = accepted.cpu(), tau.cpu()
    eps_h = float(manager.uncertainty_filter.entropy_threshold)
    eps_mi = float(manager.uncertainty_filter.mi_threshold)
    gates = {"confidence": conf >= tau, "entropy": ent < eps_h, "mutual_information": mi < eps_mi}
    gates["confidence_and_entropy"] = gates["confidence"] & gates["entropy"]
    gates["entropy_and_mutual_information"] = gates["entropy"] & gates["mutual_information"]
    gates["all_gates"] = gates["confidence_and_entropy"] & gates["mutual_information"]
    if not torch.equal(gates["all_gates"], accepted):
        raise AssertionError("Analytical gate decomposition differs from StageManager")

    def counts(subset):
        support = int(subset.sum())
        row = {"support": support,
               "gate_pass_counts": {key: int((value & subset).sum()) for key, value in gates.items()},
               "accepted_count": int((accepted & subset).sum()),
               "threshold_at_one_count": int(((tau >= 1) & subset).sum()),
               "threshold_at_one_fraction": float((tau[subset] >= 1).float().mean()) if support else None}
        if support:
            row["quantiles_p10_p50_p90"] = {
                key: torch.quantile(value[subset].float(), torch.tensor([.1, .5, .9])).tolist()
                for key, value in (("confidence", conf), ("entropy", ent), ("mutual_information", mi), ("threshold", tau))}
        return row

    result = {**counts(torch.ones_like(accepted)),
              "accepted_fraction": float(accepted.float().mean()),
              "candidate_predicted_class_counts": torch.bincount(pred, minlength=9).tolist(),
              "accepted_predicted_class_counts": torch.bincount(pred[accepted], minlength=9).tolist(),
              "class_confidence_statistics": statistics(manager),
              "per_predicted_class": [{"class": name, **counts(pred == label)}
                                      for label, name in enumerate(WM811K_CLASSES)]}
    if "y" not in pack:
        result["accepted_accuracy"] = None
        result["true_labels_available"] = False
        return result

    labels = pack["y"].cpu()
    correct = pred == labels
    selected_count = int(accepted.sum())
    result.update({"true_labels_available": True,
                   "prediction_accuracy": float(correct.float().mean()),
                   "accepted_accuracy": float(correct[accepted].float().mean()) if selected_count else None,
                   "confusion_matrix": torch.bincount(labels * 9 + pred, minlength=81).reshape(9, 9).tolist(),
                   "accepted_confusion_matrix": torch.bincount(labels[accepted] * 9 + pred[accepted], minlength=81).reshape(9, 9).tolist(),
                   "per_true_class": []})
    for label, name in enumerate(WM811K_CLASSES):
        subset, selected = labels == label, accepted & (labels == label)
        support, selected_count = int(subset.sum()), int(selected.sum())
        correct_count = int((selected & correct).sum())
        result["per_true_class"].append({"class": name, **counts(subset),
            "accepted_fraction": selected_count / support if support else None,
            "correct_accepted_coverage": correct_count / support if support else None,
            "accepted_accuracy": correct_count / selected_count if selected_count else None,
            "accepted_wrong_count": selected_count - correct_count,
            "accepted_wrong_predicted_class_counts": torch.bincount(pred[selected & ~correct], minlength=9).tolist(),
            "prediction_accuracy": float(correct[subset].float().mean()) if support else None})
    none = labels == 0
    none_count = int(none.sum())
    wrong_none = none & (pred != 0)
    result["None_false_positives"] = {
        "true_None_support": none_count,
        "ungated_wrong_count": int(wrong_none.sum()),
        "gated_wrong_count": int((wrong_none & accepted).sum()),
        "ungated_wrong_predicted_class_counts": torch.bincount(pred[wrong_none], minlength=9).tolist(),
        "gated_wrong_predicted_class_counts": torch.bincount(pred[wrong_none & accepted], minlength=9).tolist(),
        "ungated_None_to_Scratch_count": int((none & (pred == 8)).sum()),
        "gated_None_to_Scratch_count": int((none & (pred == 8) & accepted).sum()),
        "gated_None_to_Scratch_fraction_of_true_None": int((none & (pred == 8) & accepted).sum()) / none_count if none_count else None,
    }
    return result


def apply_fixed_statistics(manager, pack):
    before = {name: value.clone() for name, value in manager.adaptive_threshold.named_buffers()}
    tau = manager.adaptive_threshold.compute_threshold(pseudo_labels=pack["pred"], entropy=pack["ent"])
    selected = manager.uncertainty_filter.filter_classification(pack["conf"], tau, pack["ent"], pack["mi"])
    for name, value in manager.adaptive_threshold.named_buffers():
        if not torch.equal(value, before[name]):
            raise AssertionError("Holdout evaluation mutated real-Du confidence statistics")
    cv = torch.tensor([row["coefficient_of_variation"] for row in statistics(manager)])
    expected = (manager.base_threshold + manager.adaptive_threshold.alpha * cv[pack["pred"]]
                + manager.adaptive_threshold.beta * (1 - pack["ent"])).clamp(0, 1)
    torch.testing.assert_close(tau, expected, atol=1e-7, rtol=1e-6)
    return selected, tau


def verify_analytical_helpers():
    """CPU-only parity, threshold clipping, and fixed-versus-self-CV safeguards."""
    manager = StageManager(nn.Linear(2, 9))
    du = {"conf": torch.tensor([.11, .12, .99, .995]), "pred": torch.zeros(4, dtype=torch.long),
          "ent": torch.tensor([1., 1., .001, .001]), "mi": torch.zeros(4)}
    selected, tau = manager._apply_gates(du["conf"], du["pred"], du["ent"], du["mi"])
    summarize(du, selected, tau, manager)
    heldout = {"conf": torch.tensor([.999, .998]), "pred": torch.zeros(2, dtype=torch.long),
               "ent": torch.tensor([.001, .001]), "mi": torch.zeros(2), "y": torch.tensor([0, 8])}
    fixed, fixed_tau = apply_fixed_statistics(manager, heldout)
    assert not fixed.any() and (fixed_tau == 1).all()
    summarize(heldout, fixed, fixed_tau, manager)
    self_mask, self_tau = manager._apply_gates(heldout["conf"], heldout["pred"], heldout["ent"], heldout["mi"])
    assert self_mask.all()
    summarize(heldout, self_mask, self_tau, manager)
    boundary = manager.uncertainty_filter.filter_classification(
        torch.tensor([.96, .99, .99]), torch.tensor([.96, .96, .96]),
        torch.tensor([.01, .08, .01]), torch.tensor([0., 0., .12]))
    assert boundary.tolist() == [True, False, False]


def prepare_sources(labels_path, report, requested, seed):
    inventory = dict(parse_wm811k_labeled_rows(labels_path, split="training"))
    roles = {"train": report["samples"]["train"], "validation": report["samples"]["validation"],
             "none_stress": report.get("none_stress_extra_samples", []),
             "diverse_none": report.get("diverse_none_extra_training_samples", [])}
    seen = set()
    for role, rows in roles.items():
        for row in rows:
            if (not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
                    or not isinstance(row[1], int) or row[1] not in range(9)):
                raise ValueError(f"Invalid {role} manifest row")
            filename, label = row
            if inventory.get(filename) != label or filename in seen:
                raise ValueError(f"Changed, overlapping, or non-Training {role} row: {filename}")
            seen.add(filename)
    validation_names = {filename for filename, _ in roles["validation"]}
    if {label for _, label in roles["validation"]} != set(range(9)):
        raise ValueError("Validation must contain all nine true classes")
    for variant in report.get("variants", []):
        for filename, label in variant.get("real_training_samples", []):
            if inventory.get(filename) != label or filename in validation_names:
                raise ValueError("Variant real-training rows overlap validation or violate official Training")

    pool, all_names, test_names = [], set(), set()
    blank_rows, blank_test_excluded = 0, 0
    with labels_path.open(encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source)
        required = {"filename", "failureType", "trianTestLabel"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError("Expected original WM-811K label and official-split columns")
        for row in rows:
            filename = row["filename"].strip()
            if filename in all_names:
                raise ValueError(f"Duplicate source filename: {filename}")
            all_names.add(filename)
            is_test = row["trianTestLabel"].strip().lower() == "test"
            if is_test:
                test_names.add(filename)
            if row["failureType"].strip() != "":
                continue
            blank_rows += 1
            if is_test:
                blank_test_excluded += 1
            else:
                pool.append(filename)
    if len(pool) < requested:
        raise ValueError(f"Only {len(pool)} eligible blank-label non-Test images for {requested} requested")
    rng = np.random.RandomState(seed)
    filenames = [pool[int(index)] for index in rng.choice(len(pool), requested, replace=False)]
    assert not (set(filenames) & (set(inventory) | test_names | seen))
    return filenames, roles["validation"], {
        "blank_failureType_rows_in_csv": blank_rows,
        "blank_failureType_Test_rows_excluded": blank_test_excluded,
        "eligible_blank_label_non_Test_pool": len(pool),
        "unlabeled_sampling_seed": seed,
        "unlabeled_sampled_without_replacement": True,
        "validation_verified_official_Training_only": True,
        "validation_disjoint_from_report_training_and_stress": True,
        "unlabeled_disjoint_from_all_labeled_Training_and_Test": True,
        "official_Test_images_loaded": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unlabeled-samples", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="mps")
    parser.add_argument("--mc-passes", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.unlabeled_samples <= 4096 or not 2 <= args.mc_passes <= 50:
        parser.error("Bounded probe requires 1..4096 unlabeled samples and 2..50 MC passes")
    torch.set_num_threads(1)
    verify_analytical_helpers()
    started = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not checkpoint.get("diagnostic"):
        raise ValueError("This bounded probe expects a diagnostic checkpoint")
    config = EngineConfig.from_dict(checkpoint["config"])
    if config.get("model.mode") != "classification" or config.get("model.num_classes") != 9:
        raise ValueError("Expected nine-class classification architecture")
    report_path = args.report / "report.json" if args.report.is_dir() else args.report
    report = json.loads(report_path.read_text())
    saved_split = checkpoint.get("diagnostic_options", {}).get("split_from")
    if saved_split:
        source_path = Path(saved_split)
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if source_path.is_dir():
            source_path = source_path / "report.json"
        checkpoint_split = json.loads(source_path.read_text())
        for role in ("train", "validation"):
            if checkpoint_split["samples"][role] != report["samples"][role]:
                raise ValueError(f"Checkpoint source split differs from supplied {role} manifest")
    seed = int(report.get("data_seed", report.get("seed", 42)))
    data_root = Path(config.get("data.data_root", "datasets/wm811k"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    labels_path, images_dir = data_root / "labels.csv", data_root / "images"
    du_names, val_rows, source_checks = prepare_sources(labels_path, report, args.unlabeled_samples, seed + 420200)
    if len(val_rows) > 5000:
        raise ValueError("Bounded probe permits at most 5000 validation rows")
    image_size = int(config.get("data.image_size", 32))
    # All file I/O and preprocessing finish before the first Metal allocation.
    du_x = torch.stack([encode_wafer_image(_resolve_image_path(images_dir, name), image_size) for name in du_names])
    val_x = torch.stack([encode_wafer_image(_resolve_image_path(images_dir, name), image_size) for name, _ in val_rows])
    val_y = torch.tensor([label for _, label in val_rows], dtype=torch.long)
    model = SemiWaferNet.from_config(config)
    model.load_state_dict(normalize_semiwafernet_state_dict(checkpoint["model"]), strict=True)
    cpu_preparation_seconds = time.perf_counter() - started
    device = resolve_device(args.device)
    set_seed(seed)
    model = model.to(device).eval()
    ssl = config.get("semi_supervised", {})
    if ssl.get("ssl_prior_scale", 0) != 0 or ssl.get("max_none_to_defect_ratio") is not None:
        raise ValueError("Probe requires the unchanged paper configuration without prior bias or None capping")
    parameters = {"base_threshold": ssl.get("confidence_threshold", .94), "alpha": ssl.get("alpha", .08),
                  "beta": ssl.get("beta", .02), "entropy_threshold": ssl.get("entropy_threshold", .08),
                  "mi_threshold": ssl.get("mutual_information_threshold", .12)}
    manager = StageManager(model, mc_passes=args.mc_passes, **parameters)
    batch_size = 64
    sync(device)
    mc_started = time.perf_counter()
    du_pack = manager._mc_collect(DataLoader(TensorDataset(du_x), batch_size=batch_size),
                                  torch.device(device), None, "real Du MC", True)
    sync(device)
    du_mc_seconds = time.perf_counter() - mc_started
    du_mask, du_tau = manager._apply_gates(du_pack["conf"], du_pack["pred"], du_pack["ent"], du_pack["mi"])
    du_summary = summarize(du_pack, du_mask, du_tau, manager)
    mc_started = time.perf_counter()
    val_pack = manager._mc_collect(DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size),
                                   torch.device(device), None, "full saved validation MC", True, with_labels=True)
    sync(device)
    validation_mc_seconds = time.perf_counter() - mc_started
    fixed_mask, fixed_tau = apply_fixed_statistics(manager, val_pack)
    fixed_summary = summarize(val_pack, fixed_mask, fixed_tau, manager)
    self_mask, self_tau = manager._apply_gates(val_pack["conf"], val_pack["pred"], val_pack["ent"], val_pack["mi"])
    self_summary = summarize(val_pack, self_mask, self_tau, manager)
    payload = {
        "probe": "real-Du class CV versus validation self-CV with unchanged gates",
        "script_sha256": digest(__file__),
        "training_performed": False, "device": device, "torch_version": torch.__version__,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_variant": checkpoint.get("variant"), "checkpoint_options": checkpoint.get("diagnostic_options"),
        "report": str(report_path.resolve()), "report_sha256": digest(report_path),
        "labels_csv": str(labels_path.resolve()), "labels_csv_sha256": digest(labels_path),
        "source_provenance": source_checks,
        "class_order": list(WM811K_CLASSES), "gate_parameters": parameters,
        "gate_entropy_units": "raw Shannon nats", "prior_bias": None, "None_cap": None,
        "mc_passes": args.mc_passes, "batch_size": batch_size, "mc_seed": seed,
        "cpu_threads": torch.get_num_threads(), "unlabeled_samples": du_names, "validation_samples": val_rows,
        "validation_true_class_counts": torch.bincount(val_y, minlength=9).tolist(),
        "real_Du": du_summary, "validation_fixed_real_Du_CV": fixed_summary,
        "validation_self_CV_baseline": self_summary,
        "comparison_uses_identical_validation_MC_predictions": True,
        "validation_did_not_update_real_Du_statistics": True,
        "validation_changes_when_switching_CV_source": {
            "accepted_only_real_Du_CV": int((fixed_mask & ~self_mask).sum()),
            "accepted_only_self_CV": int((self_mask & ~fixed_mask).sum()),
            "tau_absolute_difference_mean": float((fixed_tau - self_tau).abs().mean()),
        },
        "timing_seconds": {"cpu_preparation": cpu_preparation_seconds, "real_Du_MC": du_mc_seconds,
                           "validation_MC": validation_mc_seconds, "total": time.perf_counter() - started},
        "model_forward_calls": args.mc_passes * (math.ceil(len(du_names) / batch_size) + math.ceil(len(val_rows) / batch_size)),
        "limitations": ["Du has no ground truth; its accepted accuracy is unknown.",
                        "The sampled Du gives a noisy estimate of full-Du class CV, especially for rare predictions; unobserved predicted classes use CV=0 as in StageManager.",
                        "Validation was used for earlier checkpoint selection; this is descriptive gate analysis, not independent final evaluation.",
                        "The self-CV baseline uses the entire supplied holdout and can differ from earlier smaller balanced-subset diagnostics.",
                        "No gates were selected, weakened, calibrated, or modified; official Test images were never loaded."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "device": device,
                      "validation_support": len(val_rows), "real_Du_accepted": du_summary["accepted_count"],
                      "fixed_CV_validation_accepted": fixed_summary["accepted_count"],
                      "self_CV_validation_accepted": self_summary["accepted_count"],
                      "timing_seconds": payload["timing_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
