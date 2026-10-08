"""Collect completed bounded experiments without inference or training."""
import hashlib
import json
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(name):
    return json.loads((BASE / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


paired = read("paired_summary.json")
assert paired["status"] == "complete"
assert paired["cpu_none_size_bins"]["status"] == "complete"
continuations = []
for seed in (42, 43):
    source = read(f"continuation_seed{seed}_natural/report.json")
    assert source["status"] == "complete"
    assert len(source["arms"]) == 2
    arms = []
    for arm in source["arms"]:
        assert arm["status"] == "complete" and arm["steps"] == 306
        arms.append({key: arm[key] for key in (
            "name", "steps", "initial_validation", "best_validation", "best_step",
            "history", "training_and_validation_seconds", "wall_seconds_including_pseudo_generation",
            "raw_stage_output_validation", "pseudo_gate_diagnostic")})
        path = Path(arm["best_checkpoint"])
        arms[-1]["checkpoint"] = str(path.relative_to(ROOT))
        arms[-1]["checkpoint_sha256"] = sha(path)
        arms[-1]["terminal_epoch_validation"] = arm["history"][-1]
    continuations.append({"seed": seed, "arms": arms, "limitations": source["limitations"]})

probes = []
for arm in ("natural", "stratified"):
    source = read(f"real_du_{arm}_seed42.json")
    probes.append({"arm": arm, **{key: source[key] for key in (
        "checkpoint_sha256", "gate_parameters", "mc_passes", "source_provenance",
        "real_Du", "validation_fixed_real_Du_CV", "validation_changes_when_switching_CV_source",
        "timing_seconds", "limitations")}})

files = [
    "paired_summary.json", "paired_summary.md", "paired_validation_curves.png",
    "paired_run_state.json", "paired_splits/natural_none.json", "paired_splits/stratified_none.json",
    "data_audit.json", "warmup_seed42/report.json", "none_bias_calibration.json",
    "real_du_natural_seed42.json", "real_du_stratified_seed42.json",
    "continuation_seed42_natural/report.json", "continuation_seed43_natural/report.json",
    "probe_real_du.py", "compare_continuation.py", "summarize_pairs.py", "calibrate_none_bias.py",
    "run_pairs.py", "paired_splits/build_paired_splits.py", "aggregate_followup.py",
]
evidence = [{"path": str((BASE / file).relative_to(ROOT)), "sha256": sha(BASE / file)}
            for file in files]
warmup = read("warmup_seed42/report.json")
calibration = read("none_bias_calibration.json")
stage_means = {}
for i, name in enumerate(("supervised_continuation", "SSL_stage2_continuation")):
    stage_means[name] = {
        "mean_best_validation_macro_f1": mean(item["arms"][i]["best_validation"]["macro_f1"] for item in continuations),
        "mean_terminal_validation_macro_f1": mean(item["arms"][i]["terminal_epoch_validation"]["macro_f1"] for item in continuations),
    }
payload = {
    "date": "2026-10-08", "status": "completed_bounded_diagnostics",
    "accepted_by_user": False, "paper_metrics_reproduced": False,
    "model": "SemiWaferNet classification", "parameter_count": 788169,
    "device": "Apple MPS", "hardware": "MacBook Air M3, 8 GB",
    "torch_version": warmup["torch_version"],
    "scope": "Reduced official Training splits and real blank-label Du; no official Test image evaluation.",
    "publication_config_changed": False,
    "publication_config_sha256": sha(ROOT / "papers/semiwafernet/configs/config.yaml"),
    "paired_experiment": {key: paired[key] for key in (
        "initialization_seeds", "source_data_seed", "matched_controls",
        "arm_means_sample_sd", "paired_deltas", "shared_validation", "shared_stress_extra")},
    "none_size_fpr_deltas": paired["cpu_none_size_bins"]["paired_stratified_minus_natural_fpr"],
    "real_Du_probes": probes, "continuations": continuations,
    "continuation_means": stage_means,
    "none_bias_calibration": calibration["summary_by_arm_and_scenario"],
    "findings": [
        "Short-training errors correlate strongly with raw None size and sparse failed-die density. Sampled None maps legitimately contain failed dies.",
        "Size-stratified None training improves validation but worsens the None-dominant stress evaluation in both seeds. It trades fewer large-None errors for more small-None errors.",
        "Unchanged gates accept only9/1024 and8/1024 real Du maps in the two seed42 models; neither accepts None, Loc or Scratch.",
        "Two short actual Stage2 continuations do not establish a reliable SSL benefit: seed42 improves selected validation slightly; seed43 retains its incoming baseline in both arms.",
        "None logit bias improves aggregate stress F1 while losing Scratch recall; it is not promoted to the publication configuration.",
    ],
    "decision": "Keep publication defaults and all successful/accepted models unchanged. Improve supervised minority recognition and validate per-class recall before an expensive full SSL experiment.",
    "limitations": [
        "Reduced-data validation and descriptive stress scores are not full official-Test paper metrics.",
        "Two seeds on one split do not provide a reliable population uncertainty estimate.",
        "Paired diagnostics use batch64 and LR2e-4 rather than published batch256/LR5e-5; sampling interventions are explicit ablations.",
        "Validation selected incoming checkpoints and later diagnostic choices, so it is not an independent final evaluation.",
        "Continuation resets AdamW because source diagnostic checkpoints omit moments; the full training path preserves optimizer state.",
        "The pseudo union changes shuffle, augmentation, dropout and final BatchNorm batch sizes; a tiny validation difference cannot be attributed solely to pseudo supervision.",
        "Accepted Du has no ground truth; high accepted validation precision does not establish real-Du precision or problematic-class coverage.",
        "Prior-calibration gains trade minority recall for precision and are separate from the published protocol.",
        "No result here guarantees the article metrics after full CUDA training.",
    ],
    "evidence": evidence,
    "artifact_source_provenance": paired["summarizer_source_provenance"],
}
output = ROOT / "papers/semiwafernet/results/mps_followup_20261008.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
print(json.dumps({"output": str(output), "continuation_means": stage_means}, indent=2))
