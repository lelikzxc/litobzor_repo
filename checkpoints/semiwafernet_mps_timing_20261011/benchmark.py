"""Bounded batch256 timing; source weights and production config are read-only."""
import hashlib
import json
import sys
import time
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from common.engine.config import EngineConfig
from common.training.utils import resolve_device
from common.utils.seed import set_seed
from papers.semiwafernet.data_utils.wafer_dataset import _resolve_image_path, encode_wafer_image, geometric_augment
from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.training.stage_manager import StageManager
from papers.semiwafernet.utils.checkpoint import normalize_semiwafernet_state_dict

torch.set_num_threads(1)
set_seed(42)
start = time.perf_counter()
old = ROOT / "checkpoints/semiwafernet_mps_followup_20261008"
source = old / "paired_seed42/natural_none/mid_smote.pt"
saved = torch.load(source, map_location="cpu", weights_only=False)
config = EngineConfig.from_dict(saved["config"])
report = json.loads((source.parent / "report.json").read_text())
probe = json.loads((old / "real_du_natural_seed42.json").read_text())
images = ROOT / config.get("data.data_root") / "images"
rng = np.random.RandomState(42)
rows = report["samples"]["train"]
chosen = [rows[int(index)] for index in rng.choice(len(rows), 512, replace=False)]
train_x = torch.stack([encode_wafer_image(_resolve_image_path(images, name), 32) for name, _ in chosen])
train_y = torch.tensor([label for _, label in chosen], dtype=torch.long)
du_x = torch.stack([encode_wafer_image(_resolve_image_path(images, name), 32) for name in probe["unlabeled_samples"]])
model = SemiWaferNet.from_config(config)
model.load_state_dict(normalize_semiwafernet_state_dict(saved["model"]), strict=True)
manager = StageManager(model, mc_passes=20)
cpu_seconds = time.perf_counter() - start
device = torch.device(resolve_device("mps"))
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=4e-4)

class Augmented(Dataset):
    def __len__(self):
        return len(train_y)
    def __getitem__(self, index):
        return geometric_augment(train_x[index], rng), train_y[index]

loader = DataLoader(Augmented(), batch_size=256, shuffle=True, num_workers=0)
iterator = iter(loader)
def step():
    global iterator
    try:
        x, y = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        x, y = next(iterator)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x.to(device))["classification"]
    loss = torch.nn.functional.cross_entropy(logits, y.to(device))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    torch.mps.synchronize()
    assert torch.isfinite(loss), "Non-finite benchmark loss"

metal_start = time.perf_counter()
for _ in range(5):
    step()
times = []
for _ in range(20):
    if time.perf_counter() - metal_start > 40:
        break
    tick = time.perf_counter()
    step()
    times.append(time.perf_counter() - tick)
assert len(times) >= 5
train_memory = {"allocated": torch.mps.current_allocated_memory(), "driver": torch.mps.driver_allocated_memory()}
manager.teacher.teacher.to(device)
du_loader = DataLoader(TensorDataset(du_x), batch_size=256, shuffle=False, num_workers=0)
torch.mps.synchronize()
tick = time.perf_counter()
pack = manager._mc_collect(du_loader, device, None, "bounded MC timing", False)
torch.mps.synchronize()
mc_seconds = time.perf_counter() - tick
payload = {
    "date": "2026-10-11", "purpose": "throughput only; no recognition results or saved trained weights",
    "device": "mps", "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
    "batch_size": 256, "model_parameters": sum(p.numel() for p in model.parameters()),
    "training_steps_warmup": 5, "training_steps_measured": len(times),
    "training_step_seconds": times, "training_step_mean_seconds": mean(times),
    "training_step_median_seconds": median(times), "training_samples_per_second": 256 / mean(times),
    "training_memory_bytes": train_memory,
    "mc_samples": len(du_x), "mc_passes": 20, "mc_seconds": mc_seconds,
    "mc_samples_per_second": len(du_x) / mc_seconds,
    "estimated_mc150000_seconds_per_refresh": 150000 / len(du_x) * mc_seconds,
    "cpu_preparation_seconds": cpu_seconds, "total_seconds": time.perf_counter() - start,
    "source_checkpoint": str(source.relative_to(ROOT)),
    "source_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "limitations": [
        "Short burst does not measure sustained fanless thermal performance.",
        "CPU images are already encoded; training includes real geometric augmentation but excludes full85k SMOTE preparation and lazy pseudo image I/O.",
        "MC input is already encoded; full150k pool image decoding and I/O add time.",
        "The full encoded training dataset and other applications can cause additional8GB memory pressure.",
        "This timing does not predict training accuracy or guarantee completion without memory pressure.",
    ],
}
(BASE / "timing.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({key: payload[key] for key in (
    "training_step_mean_seconds", "training_samples_per_second", "mc_seconds",
    "estimated_mc150000_seconds_per_refresh", "total_seconds", "training_memory_bytes")}, indent=2))
