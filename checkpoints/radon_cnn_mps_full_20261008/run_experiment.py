"""Sequential MPS runs with fixed settings and independent seeds, no test tuning."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
env = os.environ.copy()
env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MPLCONFIGDIR="/private/tmp/litobzor-mpl-cache", PYTHONUNBUFFERED="1")
state = {"started_utc": datetime.now(timezone.utc).isoformat(),
         "device": "mps", "seeds": SEEDS, "status": "running", "runs": [],
         "selection": "Each best.pt minimizes its own validation loss; no run selected by test metrics.",
         "config_source": "papers/radon_cnn/configs/config_balanced.yaml",
         "validation_loss": "Sample-weighted cross entropy; Radon-local Trainer corrects final incomplete batch weighting.",
         "superseded_run": "superseded_before_loss_fix/seed42; interrupted before any test evaluation and excluded from results",
         "protocol_note": "6400 training presentations with training-only replacement; documented interpretation of the paper, not a recovered author split."}


def persist():
    (BASE / "run_state.json").write_text(json.dumps(state, indent=2))


persist()
for seed in SEEDS:
    destination = BASE / f"seed{seed}"
    destination.mkdir(exist_ok=False)
    config = yaml.safe_load((ROOT / state["config_source"]).read_text())
    config["training"]["seed"] = seed
    config["checkpoint"]["save_dir"] = str(destination.relative_to(ROOT))
    config_path = destination / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    run = {"seed": seed, "directory": str(destination.relative_to(ROOT)), "status": "training"}
    state["runs"].append(run)
    persist()
    print(f"Starting seed {seed}: MPS, train_size=6400, maximum=500 epochs, patience=30", flush=True)
    started = time.monotonic()
    command = [sys.executable, "-u", str(BASE / "run_seed.py"), "--config", str(config_path), "--device", "mps"]
    with (destination / "train.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    run.update(training_process_seconds=time.monotonic() - started, training_exit_code=result.returncode)
    if result.returncode:
        run["status"] = state["status"] = "failed"
        persist()
        raise SystemExit(f"Training failed for seed {seed}; inspect {destination / 'train.log'}")
    run["status"] = "evaluating"
    persist()
    print(f"Seed {seed} training complete in {run['training_process_seconds']:.1f}s; evaluating best.pt", flush=True)
    started = time.monotonic()
    command = [sys.executable, "-u", "papers/radon_cnn/evaluate.py", "--checkpoint", str(destination / "best.pt"), "--device", "mps"]
    with (destination / "evaluate.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    run.update(evaluation_process_seconds=time.monotonic() - started, evaluation_exit_code=result.returncode,
               status="complete" if result.returncode == 0 else "failed")
    if result.returncode:
        state["status"] = "failed"
        persist()
        raise SystemExit(f"Evaluation failed for seed {seed}; inspect {destination / 'evaluate.log'}")
    persist()
    print((destination / "evaluate.log").read_text(), flush=True)
state.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
persist()
print("All three runs and standalone evaluations completed.", flush=True)
