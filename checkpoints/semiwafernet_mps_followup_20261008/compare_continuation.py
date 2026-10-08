"""Bounded matched-update supervised versus real Stage2 continuation.

This artifact starts both arms from the same diagnostic weights, regenerates
source SMOTE on CPU, and keeps the published MC gates. Source diagnostic
checkpoints contain no optimizer state: both arms deliberately start fresh
AdamW, so this is a controlled continuation experiment rather than exact
resumption of the original optimization trajectory.
"""

from __future__ import annotations

import argparse
import gc
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
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from common.engine.config import EngineConfig
from common.training.utils import resolve_device
from common.utils.seed import set_seed
from papers.semiwafernet.data_utils.wafer_dataset import (
    SMOTEDataset, _resolve_image_path, encode_wafer_image, geometric_augment,
)
from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.scripts.diagnose_mps import report_predictions
from papers.semiwafernet.training.stage_manager import StageManager
from papers.semiwafernet.training.trainer import Trainer
from papers.semiwafernet.utils.checkpoint import normalize_semiwafernet_state_dict
from probe_real_du import digest, prepare_sources, summarize, sync


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class AugmentedEncodedDataset(Dataset):
    def __init__(self, images, labels, seed):
        self.images, self.labels = images, labels
        self.rng = np.random.RandomState(seed + 7)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return geometric_augment(self.images[index], self.rng), self.labels[index]


class BudgetStop(RuntimeError):
    pass


class BoundedTrainer(Trainer):
    """Artifact guard around actual _ce_step; production trainer is unchanged."""
    def __init__(self, *args, max_steps, max_seconds, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps, self.max_seconds = max_steps, max_seconds
        self.steps, self.training_started, self.batch_losses = 0, None, []
        self.stop_reason = None

    def _ce_step(self, images, labels):
        if self.training_started is None:
            self.training_started = time.perf_counter()
        if self.steps >= self.max_steps:
            self.stop_reason = "matched_update_budget_reached"
            raise BudgetStop(self.stop_reason)
        if time.perf_counter() - self.training_started >= self.max_seconds:
            self.stop_reason = "training_and_validation_time_budget_reached"
            raise BudgetStop(self.stop_reason)
        loss = super()._ce_step(images, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite continuation loss")
        self.steps += 1
        self.batch_losses.append(float(loss.detach().cpu()))
        return loss


class ObservedStageManager(StageManager):
    """Observe the real pseudo build and reset only training random generators."""
    def __init__(self, *args, training_seed, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_seed = training_seed
        self.pseudo_report = None
        self.collected_pack = None

    def _mc_collect(self, *args, **kwargs):
        self.collected_pack = super()._mc_collect(*args, **kwargs)
        return self.collected_pack

    def build_pseudo_dataset(self, *args, **kwargs):
        started = time.perf_counter()
        dataset, stats = super().build_pseudo_dataset(*args, **kwargs)
        pack = self.collected_pack
        tau = self.adaptive_threshold.compute_threshold(pseudo_labels=pack["pred"], entropy=pack["ent"])
        accepted = self.uncertainty_filter.filter_classification(pack["conf"], tau, pack["ent"], pack["mi"])
        assert int(accepted.sum()) == (len(dataset) if dataset is not None else 0)
        self.pseudo_report = {**summarize(pack, accepted, tau, self),
                              "production_pseudo_statistics": stats,
                              "pseudo_generation_seconds": time.perf_counter() - started,
                              "selection_used_no_validation_labels": True}
        # MC draws must not advance the student's initial training RNG relative
        # to the supervised arm. The frozen teacher/gates/data are untouched.
        set_seed(self.training_seed)
        return dataset, stats


@torch.no_grad()
def evaluate(model, images, labels, device):
    model.eval()
    logits = torch.cat([model(chunk.to(device))["classification"].cpu() for chunk in images.split(64)])
    return report_predictions(logits, labels)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def run_arm(name, state, config, train_x, train_y, val_x, val_y, du_x, args, report, publish):
    set_seed(args.seed)
    model = SemiWaferNet.from_config(config)
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    ssl = config.get("semi_supervised", {})
    parameters = {"base_threshold": ssl.get("confidence_threshold", .94), "alpha": ssl.get("alpha", .08),
                  "beta": ssl.get("beta", .02), "entropy_threshold": ssl.get("entropy_threshold", .08),
                  "mi_threshold": ssl.get("mutual_information_threshold", .12)}
    manager = ObservedStageManager(model, training_seed=args.seed, mc_passes=20, **parameters)
    teacher_initial = cpu_state(manager.teacher.teacher)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=4e-4)
    trainer = BoundedTrainer(model, manager, optimizer,
        supervised_loss_fn=lambda out, target: nn.functional.cross_entropy(out["classification"], target["classification"]),
        device=torch.device(args.device), grad_max_norm=1., verbose=True, batch_size=64,
        max_steps=report["matched_update_budget"], max_seconds=args.max_seconds)
    labeled = DataLoader(AugmentedEncodedDataset(train_x, train_y, args.seed), batch_size=64, shuffle=True)
    unlabeled = DataLoader(TensorDataset(du_x), batch_size=64, shuffle=False)
    initial = evaluate(model, val_x, val_y, args.device)
    best_state, best_metrics, best_step = cpu_state(model), initial, 0
    history = [{"step": 0, "epoch": 0, "kind": "incoming_baseline", **initial}]
    frozen_checks = 0
    arm = {"name": name, "status": "running", "initial_validation": initial,
           "history": history, "best_validation": initial, "best_step": 0,
           "optimizer": "fresh AdamW", "learning_rate": 2e-4, "weight_decay": 4e-4,
           "best_includes_incoming_step0_baseline": True}
    report["arms"].append(arm)

    def persist_best():
        temporary = args.output / f"{name}.pt.tmp"
        torch.save({"model": best_state, "config": config.to_dict(), "diagnostic": True,
                    "variant": name, "best_step": best_step, "best_validation": best_metrics,
                    "checkpoint_selection": "source validation macro-F1 including incoming step0 baseline",
                    "optimizer_state": "fresh optimizer used; optimizer not saved in this artifact",
                    "diagnostic_options": {"seed": args.seed, "data_seed": report["data_seed"],
                                           "source_checkpoint": report["source_checkpoint"],
                                           "learning_rate": 2e-4}}, temporary)
        temporary.replace(args.output / f"{name}.pt")

    def check_teacher():
        nonlocal frozen_checks
        for key, value in manager.teacher.teacher.state_dict().items():
            if not torch.equal(value.detach().cpu(), teacher_initial[key]):
                raise AssertionError(f"Teacher changed during optimization: {key}")
        assert not any(parameter.requires_grad for parameter in manager.teacher.teacher.parameters())
        frozen_checks += 1

    def record(kind="epoch_validation"):
        nonlocal best_state, best_metrics, best_step
        check_teacher()
        metrics = evaluate(model, val_x, val_y, args.device)
        history.append({"step": trainer.steps, "epoch": trainer.current_epoch + 1, "kind": kind,
                        "training_batch_loss_mean": sum(trainer.batch_losses) / max(len(trainer.batch_losses), 1), **metrics})
        trainer.batch_losses.clear()
        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_state, best_metrics, best_step = cpu_state(model), metrics, trainer.steps
            persist_best()
        arm.update({"steps": trainer.steps, "history": history, "best_validation": best_metrics,
                    "best_step": best_step, "pseudo_gate_diagnostic": manager.pseudo_report})
        publish()
        print(f"{name}: step={trainer.steps}, val_f1={metrics['macro_f1']:.4f}, best={best_metrics['macro_f1']:.4f}", flush=True)
        return metrics["macro_f1"]

    persist_best()
    publish()
    set_seed(args.seed)
    sync(args.device)
    arm_started = time.perf_counter()
    raw_return = None
    interrupted = False
    try:
        if name == "supervised_continuation":
            raw_return = trainer.train_stage1(labeled, num_epochs=args.epochs, val_eval_fn=record)
        else:
            raw_return = trainer.train_stage2(labeled, unlabeled, num_epochs=args.epochs, val_eval_fn=record)
    except BudgetStop:
        interrupted = True
        # The real trainer was stopped before the next optimizer step. Capture
        # the partial union epoch without changing production restoration logic.
        record("partial_epoch_budget_stop")
    sync(args.device)
    stage_output = evaluate(model, val_x, val_y, args.device)
    raw_candidates = [row for row in history if row["step"] > 0]
    raw_best = max(raw_candidates, key=lambda row: row["macro_f1"]) if raw_candidates else None
    model.load_state_dict(best_state, strict=True)
    final_best = evaluate(model, val_x, val_y, args.device)
    arm.update({"status": "complete", "steps": trainer.steps, "epochs_completed": trainer.current_epoch,
                "partial_union_epoch": interrupted and name != "supervised_continuation",
                "stop_reason": trainer.stop_reason or "requested_epochs_completed",
                "training_and_validation_seconds": time.perf_counter() - trainer.training_started if trainer.training_started else 0.,
                "wall_seconds_including_pseudo_generation": time.perf_counter() - arm_started,
                "raw_stage_return": raw_return, "raw_stage_best_validation": raw_best,
                "raw_stage_output_validation": stage_output,
                "best_step": best_step, "best_validation": best_metrics,
                "final_reloaded_artifact_validation": final_best,
                "teacher_frozen_during_optimization_checks": frozen_checks,
                "teacher_refresh_at_normal_stage_completion_is_production_behavior": not interrupted,
                "pseudo_gate_diagnostic": manager.pseudo_report,
                "best_checkpoint": str((args.output / f"{name}.pt").resolve())})
    publish()
    del trainer, optimizer, manager, model
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=35.)
    parser.add_argument("--prepare-only", action="store_true", help="Validate and regenerate inputs on CPU without training or GPU allocation")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 3 or not 0 < args.max_seconds <= 35:
        parser.error("Bounded comparison requires 1..3 epochs and 0..35 seconds per arm")
    if (args.output / "report.json").exists():
        raise FileExistsError("Use a fresh artifact --output directory")
    torch.set_num_threads(1)
    started = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not checkpoint.get("diagnostic"):
        raise ValueError("Expected source diagnostic checkpoint without optimizer state")
    config = EngineConfig.from_dict(checkpoint["config"])
    if config.get("model.mode") != "classification" or config.get("model.num_classes") != 9:
        raise ValueError("Expected the original nine-class classification architecture")
    if config.get("data.image_size", 32) != 32:
        raise ValueError("Source diagnostic must use32x32 paper inputs")
    source_path = args.report / "report.json" if args.report.is_dir() else args.report
    source = json.loads(source_path.read_text())
    saved_split = checkpoint.get("diagnostic_options", {}).get("split_from")
    if saved_split:
        path = Path(saved_split)
        if not path.is_absolute():
            path = ROOT / path
        split = json.loads((path / "report.json" if path.is_dir() else path).read_text())
        if split["samples"] != source["samples"]:
            raise ValueError("Supplied report differs from checkpoint source split")
    data_seed = int(source.get("data_seed", source.get("seed", 42)))
    data_root = Path(config.get("data.data_root"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    du_names, val_rows, checks = prepare_sources(data_root / "labels.csv", source, 1024, data_seed + 420200)
    train_rows = [tuple(row) for row in source["samples"]["train"]]
    ssl = config.get("semi_supervised", {})
    expected = {"confidence_threshold": .94, "alpha": .08, "beta": .02,
                "entropy_threshold": .08, "mutual_information_threshold": .12, "mc_passes": 20}
    if any(ssl.get(key) != value for key, value in expected.items()):
        raise ValueError("Comparison requires unchanged published gates and20 MC passes")
    if ssl.get("ssl_prior_scale", 0) != 0 or ssl.get("max_none_to_defect_ratio") is not None:
        raise ValueError("Paper continuation requires no prior bias or None cap")
    # Identical algorithm, sample order, and source data seed to diagnose_mps.
    with threadpool_limits(limits=1):
        smote = SMOTEDataset(data_root, train_rows, image_size=32, seed=data_seed, augment=False)
    train_x, train_y = torch.from_numpy(smote._X), torch.from_numpy(smote._y)
    val_x = torch.stack([encode_wafer_image(_resolve_image_path(data_root / "images", name), 32) for name, _ in val_rows])
    val_y = torch.tensor([label for _, label in val_rows], dtype=torch.long)
    du_x = torch.stack([encode_wafer_image(_resolve_image_path(data_root / "images", name), 32) for name in du_names])
    state = normalize_semiwafernet_state_dict(checkpoint["model"])
    cpu_model = SemiWaferNet.from_config(config)
    cpu_model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in cpu_model.parameters())
    state = cpu_state(cpu_model)
    del cpu_model
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "prepared", "artifact": "matched-update continuation comparator",
              "torch_version": torch.__version__, "parameter_count": parameter_count,
              "source_architecture": config.get("model"),
              "script_sha256": digest(__file__), "source_checkpoint": str(args.checkpoint.resolve()),
              "source_checkpoint_sha256": digest(args.checkpoint), "source_report": str(source_path.resolve()),
              "source_report_sha256": digest(source_path), "data_seed": data_seed, "training_seed": args.seed,
              "labels_csv_sha256": digest(data_root / "labels.csv"),
              "training_samples": train_rows, "validation_samples": val_rows, "unlabeled_samples": du_names,
              "source_checks": checks, "smote_counts": torch.bincount(train_y, minlength=9).tolist(),
              "smote_tensor_sha256": hashlib.sha256(smote._X.tobytes()).hexdigest(),
              "matched_update_budget": args.epochs * math.ceil(len(train_y) / 64),
              "epochs_requested": args.epochs, "training_plus_validation_soft_cap_seconds_per_arm": args.max_seconds,
              "batch_size": 64, "mc_passes": 20, "gate_parameters": expected,
              "optimizer": {"name": "fresh AdamW", "learning_rate": 2e-4, "weight_decay": 4e-4},
              "augmentation": "same geometric_augment, flips and90degree rotations, on labeled and accepted pseudo images",
              "CPU_preparation_seconds": time.perf_counter() - started,
              "arms": [], "limitations": [
                  "Source diagnostic checkpoints omit optimizer moments; both arms deliberately restart AdamW.",
                  "The source learning loop used balanced replacement sampling; this comparator uses the actual trainer's shuffled SMOTE epoch/union loop.",
                  "The same update cap can stop the last Stage2 union epoch partway through.",
                  "The35s cap is checked before each optimizer step; one in-flight batch and final validation can overrun it slightly.",
                  "Training randomness is reset after MC; class mixture still changes batch composition and augmentation/dropout draws.",
                  "Source validation already selected the incoming checkpoint; results are descriptive, not independent final metrics.",
                  "No official Test images are loaded, no gate changes/calibration are performed, and no production code is changed."]}

    def publish():
        atomic_json(args.output / "report.json", report)

    publish()
    if args.prepare_only:
        print(json.dumps({"CPU_only_preparation": "passed", "SMOTE_counts": report["smote_counts"],
                          "matched_update_budget": report["matched_update_budget"],
                          "validation": len(val_rows), "Du": len(du_names)}, indent=2))
        return
    args.device = resolve_device(args.device)
    report["device"], report["status"] = args.device, "running"
    publish()
    for name in ("supervised_continuation", "SSL_stage2_continuation"):
        run_arm(name, state, config, train_x, train_y, val_x, val_y, du_x, args, report, publish)
    report["status"] = "complete"
    publish()
    print(json.dumps({"report": str(args.output / "report.json"), "arms": [
        {"name": arm["name"], "steps": arm["steps"], "incoming_f1": arm["initial_validation"]["macro_f1"],
         "best_f1": arm["best_validation"]["macro_f1"], "best_step": arm["best_step"]} for arm in report["arms"]]}, indent=2))


if __name__ == "__main__":
    main()
