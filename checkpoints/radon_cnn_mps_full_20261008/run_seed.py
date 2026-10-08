"""Run the unchanged training entry point while persisting its epoch logger."""
import json
from pathlib import Path
import runpy
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.training import logger as logger_module

config_path = Path(sys.argv[sys.argv.index("--config") + 1])
output = Path(yaml.safe_load(config_path.read_text())["checkpoint"]["save_dir"])
BaseLogger = logger_module.TrainingLogger


class PersistentLogger(BaseLogger):
    def log_epoch(self, **kwargs):
        super().log_epoch(**kwargs)
        (output / "history.json").write_text(json.dumps(self.history, indent=2))


logger_module.TrainingLogger = PersistentLogger
sys.argv[0] = str(ROOT / "papers/radon_cnn/train.py")
runpy.run_path(sys.argv[0], run_name="__main__")
