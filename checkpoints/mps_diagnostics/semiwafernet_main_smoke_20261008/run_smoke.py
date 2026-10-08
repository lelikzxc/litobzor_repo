"""Bound each mechanics-only hardware check; retain exact commands and logs."""

import json
import subprocess
import sys
import time
from pathlib import Path

output = Path(__file__).resolve().parent
commands = [
    ("train", [sys.executable, "-u", "papers/semiwafernet/train.py", "--config", str(output / "config.yaml"), "--device", "mps", "--data-fraction", "0.001"], 30),
    ("evaluate", [sys.executable, "-u", "papers/semiwafernet/evaluate.py", "--checkpoint", str(output / "run" / "best.pt"), "--device", "mps"], 30),
    ("ssl_hardware_test", [sys.executable, "-m", "pytest", "papers/semiwafernet/tests/test_mps_diagnostic.py::test_full_model_ssl_three_stages_train_on_mps", "-q", "-s"], 15),
]
report = {"purpose": "Mechanics only: synthetic permissive gates and tiny data do not measure quality", "checks": []}
for name, command, timeout in commands:
    started = time.perf_counter()
    logfile = output / f"{name}.log"
    with logfile.open("w") as stream:
        try:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    entry = {"name": name, "command": command, "timeout_seconds": timeout, "elapsed_seconds": time.perf_counter() - started, "exit_code": code, "log": str(logfile)}
    report["checks"].append(entry)
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(entry), flush=True)
    print("\n".join(logfile.read_text().splitlines()[-12:]), flush=True)
    if code != 0:
        raise SystemExit(1)
