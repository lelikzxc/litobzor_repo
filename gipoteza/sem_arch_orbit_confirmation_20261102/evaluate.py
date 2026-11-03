"""Frozen C4 confirmation: select before data, then one explicitly started test."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = ROOT.parent / "sem_arch_orbit_pool_20261102"
sys.path.insert(0, str(OLD))
import runner as orbit

FAMILIES = ["high_lr", "low_lr"]
BUDGETS = [18, 27]
SEEDS = [5501, 5521, 5557]
ARMS = ["identity", "C4"]
MODES = ["primary_fine", "secondary_seven"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    orbit.save(ROOT / name, value)


def key(row):
    return row["source_lr"], row["budget"], row["seed"], row["arm"]


def configuration():
    c = read(ROOT / "CONFIG.json")
    assert c["group"] == "C4" and c["budgets"] == BUDGETS and c["seeds"] == SEEDS
    assert c["source_lr"] == FAMILIES
    assert c["bootstrap_repeats"] == 10000 and c["bootstrap_seed"] == 202611026
    assert c["CI_quantiles"] == [.0125, .9875]
    assert c["gates"] == dict(minimum_mean_F1_gain=.01, minimum_budget_F1_gain=0., minimum_mean_IoU_delta=-.002)
    assert c["protocol_sha256"] == sha(ROOT / "PROTOCOL.md")
    return c


def verify_source(source):
    for file_key, hash_key in [("checkpoint_file", "checkpoint_sha256"),
                              ("control_result_file", "control_result_sha256"),
                              ("source_audit_file", "source_audit_sha256"),
                              ("config_snapshot_file", "config_snapshot_sha256")]:
        assert sha(REPO / source[file_key]) == source[hash_key], source[file_key]


def freeze_selection():
    c = configuration()
    assert not (ROOT / "SELECTION.json").exists(), "Selection is immutable"
    assert not (ROOT / "EVAL.npz").exists() and not (ROOT / "sources").exists(), "Selection must precede new-image download"
    lock = read(ROOT / "MODEL_LOCK.json")
    assert lock["protocol_sha256"] == sha(ROOT / "PROTOCOL.md")
    assert lock["config_sha256"] == sha(ROOT / "CONFIG.json")
    assert lock["source_plan_sha256"] == sha(ROOT / "SOURCE_PLAN.json")
    fine_path = OLD / "threshold_diagnostic" / "RESULT.json"
    aggregates = list((OLD / "trials").glob("C4_replicate_*/AGGREGATE_RESULT.json"))
    assert len(aggregates) == 1
    aggregate_path = aggregates[0]
    fine, aggregate = read(fine_path), read(aggregate_path)
    assert fine["robust_on_both_grids"] and aggregate["gates"]["A_any"]
    fine_rows = {key(row): row for row in fine["rows"]}
    old_rows = {key(row): row for row in aggregate["rows"]}
    expected = {(family, budget, seed, arm) for family in FAMILIES for budget in BUDGETS for seed in SEEDS for arm in ARMS}
    assert len(lock["rows"]) == 24 and {key(row) for row in lock["rows"]} == expected
    for row in lock["rows"]:
        k = key(row)
        assert row["source"] == old_rows[k]["source"]
        assert row["primary_fine_threshold"] == fine_rows[k]["selected"]["threshold"]
        assert row["secondary_seven_threshold"] == old_rows[k]["selected"]["threshold"]
        verify_source(row["source"])
    assert lock["source_aggregate_sha256"] == sha(aggregate_path)
    assert lock["fine_threshold_result_sha256"] == sha(fine_path)
    assert lock["source_model_code_sha256"] == sha(OLD / "model.py")
    code_hashes = {str((OLD / name).resolve()): value for name, value in orbit.source_hashes().items()}
    code_hashes[str(Path(__file__).resolve())] = sha(Path(__file__))
    selection = dict(status="FROZEN_BEFORE_NEW_IMAGE_DOWNLOAD_AND_PREDICTIONS",
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), group="C4", source_model_count=12,
        rows=lock["rows"], modes=MODES, source_lr=FAMILIES, budgets=BUDGETS, seeds=SEEDS, arms=ARMS,
        protocol_sha256=sha(ROOT / "PROTOCOL.md"), config_sha256=sha(ROOT / "CONFIG.json"),
        source_plan_sha256=sha(ROOT / "SOURCE_PLAN.json"), model_lock_sha256=sha(ROOT / "MODEL_LOCK.json"),
        fine_threshold_result_sha256=sha(fine_path), source_aggregate_sha256=sha(aggregate_path),
        evaluation_script_sha256=sha(Path(__file__)), code_sha256=code_hashes,
        bootstrap_repeats=10000, bootstrap_seed=202611026, CI_quantiles=c["CI_quantiles"], gates=c["gates"],
        fit_count=0, new_parameters=0, CNN_passes_per_image=dict(identity=1, C4=4),
        fresh_images_present=False, primary="Fixed fine-VAL thresholds; secondary seven-grid thresholds cannot replace primary outcome")
    save("SELECTION.json", selection)
    print(json.dumps(dict(status=selection["status"], selection_sha256=sha(ROOT / "SELECTION.json"),
        evaluation_script_sha256=selection["evaluation_script_sha256"], source_model_count=12, threshold_rows=24)), flush=True)


def macro_from_confusion(confusion):
    denominator = confusion.sum(-1) + confusion.sum(-2)
    numerator = 2 * np.diagonal(confusion, axis1=-2, axis2=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0).mean(-1)


def evaluate():
    started = time.monotonic()
    c = configuration();selection = read(ROOT / "SELECTION.json")
    assert selection["evaluation_script_sha256"] == sha(Path(__file__))
    for path, expected in selection["code_sha256"].items():
        assert sha(path) == expected
    assert selection["protocol_sha256"] == sha(ROOT / "PROTOCOL.md") and selection["config_sha256"] == sha(ROOT / "CONFIG.json")
    assert selection["source_plan_sha256"] == sha(ROOT / "SOURCE_PLAN.json")
    assert selection["model_lock_sha256"] == sha(ROOT / "MODEL_LOCK.json")
    assert not any((ROOT / name).exists() for name in ["EVALUATION_FROZEN.json", "PREDICTIONS.npz", "RESULT.json"]), "One immutable evaluation only"
    data_audit = read(ROOT / "DATA_AUDIT.json")
    assert data_audit["status"].startswith("PASS"), "Independent data audit must pass before predictions"
    assert torch.backends.mps.is_available() and os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    for row in selection["rows"]:
        verify_source(row["source"])
    inputs = [ROOT / name for name in ["SELECTION.json", "MODEL_LOCK.json", "PROTOCOL.md", "CONFIG.json", "SOURCE_PLAN.json",
              "DATA_AUDIT.json", "DATA_MANIFEST.json", "EVAL_INDEX.json", "EVAL.npz"]]
    input_hashes = {str(path): sha(path) for path in inputs}
    save("EVALUATION_FROZEN.json", dict(status="FROZEN_BEFORE_FIRST_TEST_PREDICTION", input_sha256=input_hashes,
        evaluation_script_sha256=sha(Path(__file__)), bootstrap_repeats=10000, bootstrap_seed=202611026,
        CI_quantiles=c["CI_quantiles"], selection_sha256=sha(ROOT / "SELECTION.json"), fit_count=0))
    with np.load(ROOT / "EVAL.npz", allow_pickle=False) as archive:
        x = archive["X"].copy();target = archive["Y"][:, 0].astype(bool)
        truth = archive["label"].copy();frames = archive["frame"].copy()
        assert np.all(archive["role"] == "test")
    frame_values, gid = np.unique(frames, return_inverse=True)
    assert x.shape == (2048, 1, 12, 12) and target.shape == (2048, 12, 12)
    assert set(frame_values.tolist()) == set(c["test_frames"]) and len(frame_values) == 8
    assert np.all(np.bincount(gid) == 256) and np.bincount(truth, minlength=9).min() >= 8
    assert min(len(np.unique(frames[truth == klass])) for klass in range(9)) >= 3
    n = len(truth)
    probability = np.empty((2, 2, 3, 2, n, 12, 12), np.float32)
    aligned_C4 = np.empty((2, 2, 3, 4, n, 12, 12), np.float32)
    predictions = np.empty((2, 2, 2, 3, 2, n), np.int64)
    packed_masks = np.empty((2, 2, 2, 3, 2, n, 18), np.uint8)
    ious = np.empty((2, 2, 2, 3, 2, n), np.float64)
    thresholds = np.empty((2, 2, 2, 3, 2), np.float64)
    confusion = np.zeros((2, 2, 2, 3, 2, 8, 9, 9), np.int32)
    by_key = {key(row): row for row in selection["rows"]}
    rows = []
    torch.set_num_threads(2)
    tensor = torch.tensor(5 * (x - .2), device="mps")
    for f, family in enumerate(FAMILIES):
        for b, budget in enumerate(BUDGETS):
            for s, seed in enumerate(SEEDS):
                reference = by_key[family, budget, seed, "identity"]["source"]
                assert reference == by_key[family, budget, seed, "C4"]["source"]
                state = torch.load(REPO / reference["checkpoint_file"], map_location="cpu", weights_only=True)["model"]
                assert orbit.state_digest(state) == reference["backbone_state_sha256"]
                for a, arm in enumerate(ARMS):
                    net = orbit.OrbitPoolSegmenter(state, arm).to("mps").eval()
                    prob, aligned, timing = orbit.evaluate(net, dict(x=tensor), "mps")
                    probability[f, b, s, a] = prob[:, 0]
                    if arm == "C4":
                        aligned_C4[f, b, s] = aligned[:, :, 0]
                    assert orbit.state_digest(net.backbone.state_dict()) == reference["backbone_state_sha256"]
                    metrics = {}
                    for m, mode in enumerate(MODES):
                        chosen = by_key[family, budget, seed, arm]["primary_fine_threshold" if m == 0 else "secondary_seven_threshold"]
                        thresholds[m, f, b, s, a] = chosen
                        mask = prob[:, 0] >= chosen
                        pred = orbit.hard_classes(mask)
                        predictions[m, f, b, s, a] = pred
                        packed_masks[m, f, b, s, a] = np.packbits(mask.reshape(n, 144), axis=1)
                        union = (mask | target).sum((1, 2));intersection = (mask & target).sum((1, 2))
                        iou = np.divide(intersection, union, out=np.ones(n), where=union > 0)
                        ious[m, f, b, s, a] = iou
                        np.add.at(confusion[m, f, b, s, a], (gid, truth, pred), 1)
                        metrics[mode] = dict(threshold=chosen,
                            mask_F1=float(orbit.f1_score(truth, pred, labels=range(9), average="macro", zero_division=0)),
                            IoU=float(iou.mean()))
                    row = dict(source_lr=family, budget=budget, seed=seed, arm=arm, source=reference,
                        metrics=metrics, parameters=10681, fit_count=0, optimizer_updates=0, **timing)
                    rows.append(row);save("PARTIAL_RESULTS.json", rows)
                    save("PROGRESS.json", dict(stage="FROZEN_NEW_FRAME_EVALUATION", completed=len(rows), total=24,
                        seconds=time.monotonic() - started, source_lr=family, budget=budget, seed=seed, arm=arm))
                    print(json.dumps(dict(stage="FROZEN_NEW_FRAME_EVALUATION", completed=len(rows), total=24,
                        source_lr=family, budget=budget, seed=seed, arm=arm, metrics=metrics, **timing)), flush=True)
                    del net
    np.savez_compressed(ROOT / "PREDICTIONS.npz", probability=probability, aligned_C4=aligned_C4,
        mask_predictions=predictions, packed_masks=packed_masks, IoU=ious, thresholds=thresholds,
        truth=truth, target=target, frames=frames, frame_values=frame_values, frame_ids=gid,
        source_indices=np.asarray(read(ROOT / "EVAL_INDEX.json")["local_TEST_indices"]),
        normalized_fixture_x=5 * (x[:16] - .2), modes=np.asarray(MODES), source_lr=np.asarray(FAMILIES),
        budgets=np.asarray(BUDGETS), seeds=np.asarray(SEEDS), arms=np.asarray(ARMS), confusion=confusion)
    points = macro_from_confusion(confusion.sum(5))
    rng = np.random.default_rng(c["bootstrap_seed"])
    seed_ids = rng.integers(0, 3, (10000, 3));frame_ids = rng.integers(0, 8, (10000, 8))
    scores = np.empty((2, 2, 2, 2, 10000), np.float64)
    for m in range(2):
        for f in range(2):
            for b in range(2):
                for a in range(2):
                    table = confusion[m, f, b, :, a]
                    sampled = table[seed_ids[:, :, None], frame_ids[:, None, :]].sum(2)
                    scores[m, f, b, a] = macro_from_confusion(sampled).mean(1)
        print(json.dumps(dict(stage="PAIRED_CROSSED_BOOTSTRAP", completed_modes=m + 1, total_modes=2,
            repeats=10000, seconds=time.monotonic() - started)), flush=True)
    family_scores = scores.mean(2)
    deltas = family_scores[:, :, 1] - family_scores[:, :, 0]
    np.savez_compressed(ROOT / "BOOTSTRAP.npz", scores=scores, family_scores=family_scores, deltas=deltas,
        seed_ids=seed_ids, frame_ids=frame_ids, confusion=confusion, point_per_seed_budget=points,
        modes=np.asarray(MODES), source_lr=np.asarray(FAMILIES), budgets=np.asarray(BUDGETS), seeds=np.asarray(SEEDS),
        arms=np.asarray(ARMS), frame_values=frame_values, CI_quantiles=np.asarray(c["CI_quantiles"]))
    analyses = {}
    for m, mode in enumerate(MODES):
        families = {}
        for f, family in enumerate(FAMILIES):
            identity = float(points[m, f, :, :, 0].mean());pooled = float(points[m, f, :, :, 1].mean())
            delta = pooled - identity
            budget_deltas = (points[m, f, :, :, 1] - points[m, f, :, :, 0]).mean(1).tolist()
            iou_delta = float((ious[m, f, :, :, 1] - ious[m, f, :, :, 0]).mean())
            ci = np.quantile(deltas[m, f], c["CI_quantiles"]).tolist()
            checks = dict(mean_gain_at_least_one_pp=bool(delta >= .01), positive_CI_lower=bool(ci[0] > 0),
                each_budget_nonnegative=bool(min(budget_deltas) >= 0), IoU_guard=bool(iou_delta >= -.002))
            families[family] = dict(identity=dict(mask_F1=identity, IoU=float(ious[m, f, :, :, 0].mean())),
                C4=dict(mask_F1=pooled, IoU=float(ious[m, f, :, :, 1].mean())), delta=delta, CI97_5=ci,
                IoU_delta=iou_delta, budget_deltas=budget_deltas,
                per_seed_deltas=(points[m, f, :, :, 1] - points[m, f, :, :, 0]).mean(0).tolist(),
                checks=checks, passed=all(checks.values()))
        analyses[mode] = dict(by_source_lr=families, passed_both_source_families=all(item["passed"] for item in families.values()))
    assert all(sha(path) == expected for path, expected in input_hashes.items())
    assert sha(Path(__file__)) == selection["evaluation_script_sha256"]
    for row in selection["rows"]:
        verify_source(row["source"])
    result = dict(status="PASS_PILOT" if analyses["primary_fine"]["passed_both_source_families"] else "FAIL_PILOT",
        primary_mode="primary_fine", secondary_mode="secondary_seven", analyses=analyses, rows=rows,
        test_examples=n, test_frames=frame_values.tolist(), test_class_counts=np.bincount(truth, minlength=9).tolist(),
        source_model_count=12, evaluated_routes=24, fit_count=0, optimizer_updates=0,
        historical_unique_training_fits=56, bootstrap_repeats=10000, bootstrap_seed=202611026,
        CI_quantiles=c["CI_quantiles"], selection_sha256=sha(ROOT / "SELECTION.json"),
        evaluation_frozen_sha256=sha(ROOT / "EVALUATION_FROZEN.json"), analysis_sha256=sha(Path(__file__)),
        predictions_sha256=sha(ROOT / "PREDICTIONS.npz"), bootstrap_sha256=sha(ROOT / "BOOTSTRAP.npz"),
        seconds=time.monotonic() - started, scope="Fixed new fields of one IC, frozen VAL thresholds; no external-chip or novelty claim")
    save("RESULT.json", result)
    save("PROGRESS.json", dict(status=result["status"], completed=24, total=24, seconds=result["seconds"]))
    print(json.dumps(dict(status=result["status"], analyses=analyses, seconds=result["seconds"])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-selection", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.freeze_selection:
            freeze_selection()
        else:
            try:
                evaluate()
            except BaseException as error:
                save("EVALUATION_FAILURE.json", dict(status="FAILED_EVALUATION", error_type=type(error).__name__,
                    error=str(error), traceback=traceback.format_exc()))
                raise


if __name__ == "__main__":
    main()
