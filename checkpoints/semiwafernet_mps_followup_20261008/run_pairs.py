"""Bounded paired None-sampling diagnostics, without official Test access."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
args = parser.parse_args()
env = os.environ.copy()
env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MPLCONFIGDIR="/private/tmp/litobzor-mpl-cache", PYTHONUNBUFFERED="1")
state = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
         "seeds": args.seeds, "runs": [], "device": "mps",
         "learning_rate": 2e-4, "steps": 1800, "max_seconds_per_arm": 130,
         "selection": "Saved representative validation macro-F1 only; shared None-dominant stress is descriptive.",
         "purpose": "Diagnostic None-size resampling ablation; no paper-default changes or official Test use."}


def persist():
    (BASE / "paired_run_state.json").write_text(json.dumps(state, indent=2))


if (BASE / "paired_run_state.json").exists():
    raise FileExistsError("Existing paired experiment; preserve it rather than overwrite")
persist()
for seed in args.seeds:
    for arm in ("natural_none", "stratified_none"):
        output = BASE / f"paired_seed{seed}" / arm
        output.mkdir(parents=True, exist_ok=False)
        source = BASE / "paired_splits" / f"{arm}.json"
        command = [sys.executable, "-u", "papers/semiwafernet/scripts/diagnose_mps.py",
                   "--device", "mps", "--output", str(output),
                   "--split-from", str(source), "--seed", str(seed),
                   "--variants", "mid_smote", "--steps", "1800",
                   "--eval-every", "150", "--max-seconds", "130",
                   "--mc-passes", "20", "--mc-samples-per-class", "20",
                   "--none-stress-samples", "9000"]
        run = {"seed": seed, "arm": arm, "status": "training",
               "output": str(output.relative_to(ROOT)), "command": command}
        state["runs"].append(run)
        persist()
        print(f"Starting seed {seed}, {arm}; matched data/architecture/LR/updates; MPS cap130s", flush=True)
        started = time.monotonic()
        with (output / "train.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        run.update(process_seconds=time.monotonic() - started, exit_code=result.returncode)
        if result.returncode:
            state["status"] = run["status"] = "failed"
            persist()
            raise SystemExit(f"Failed: {output / 'train.log'}")
        report = json.loads((output / "report.json").read_text())
        variant = report["variants"][0]
        if variant["status"] != "complete" or variant["steps"] != 1800:
            state["status"] = run["status"] = "incomplete_budget"
            persist()
            raise SystemExit("A paired arm did not complete the matched update budget")
        run.update(status="complete", best_step=variant["best_step"],
                   validation_macro_f1=variant["best_validation"]["macro_f1"],
                   none_stress_macro_f1=variant["none_stress_validation"]["macro_f1"])
        persist()
        print(json.dumps(run, indent=2), flush=True)
state.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
persist()
print("All matched paired runs complete.", flush=True)
