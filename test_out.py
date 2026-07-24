"""Diagnostic script v2 — check loss output and mAP readiness."""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from common.engine.config import EngineConfig
from papers.ctm_yolov10.models.yolov10 import CTMYOLOv10
from papers.ctm_yolov10.train import YOLOWrapper, collate_fn
from papers.ctm_yolov10.data_utils import MagneticTileDataset
from torch.utils.data import DataLoader
from types import SimpleNamespace

# Load config
config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")

# Create model
base_model = CTMYOLOv10.from_config(config)
model = YOLOWrapper(base_model)

# Verify model.args is set
print(f"model.model.base_model.args: {model.model.base_model.args}")
print(f"  type: {type(model.model.base_model.args).__name__}")
print(f"  box={model.model.base_model.args.box}, cls={model.model.base_model.args.cls}, dfl={model.model.base_model.args.dfl}")

# Get a batch
dataset = MagneticTileDataset(
    data_root="datasets/magnetic_tile",
    split="train",
    image_size=640,
)
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda b: b)

for batch in loader:
    break

batch_dict = collate_fn(batch)
print(f"\nbatch_dict keys: {list(batch_dict.keys())}")
print(f"img shape: {batch_dict['img'].shape}")
print(f"bboxes: {batch_dict['bboxes'].shape}")
print(f"cls: {batch_dict['cls'].shape}")
print(f"batch_idx: {batch_dict['batch_idx'].shape}")

# Set batch_dict on wrapper
model.batch_dict = batch_dict

# Forward
inputs = batch_dict['img']
logits = model(inputs)
print(f"\nYOLOWrapper output: {logits}")
print(f"  type: {type(logits).__name__}")
print(f"  value: {logits.item():.4f}")

# Check _last_loss_details
if model._last_loss_details is not None:
    print(f"\n_last_loss_details:")
    for k, v in model._last_loss_details.items():
        print(f"  {k}: {v}")
else:
    print(f"\n_last_loss_details: None")

# Now test eval mode predictions
print(f"\n\n=== Testing eval mode predictions ===")
base_model.eval()
with torch.no_grad():
    raw_out = base_model(inputs)
print(f"raw_out type: {type(raw_out).__name__}")
if isinstance(raw_out, (tuple, list)):
    preds_tensor = raw_out[0]
    print(f"preds_tensor shape: {preds_tensor.shape}")
    conf = preds_tensor[0, :, 4]
    print(f"Confidence stats: min={conf.min().item():.6f}, max={conf.max().item():.6f}, mean={conf.mean().item():.6f}")
    print(f"Conf > 0.01: {(conf > 0.01).sum().item()}/{conf.shape[0]}")
    print(f"Conf > 0.0001: {(conf > 0.0001).sum().item()}/{conf.shape[0]}")
    
    # Check class predictions
    cls_preds = preds_tensor[0, :, 5]
    print(f"Class predictions: unique={torch.unique(cls_preds).tolist()}")
    print(f"  class 0: {(cls_preds == 0).sum().item()}")
    print(f"  class 1: {(cls_preds == 1).sum().item()}")
    print(f"  class 2: {(cls_preds == 2).sum().item()}")

# Test with multiple forward passes (simulating epochs)
print(f"\n\n=== Testing multiple forward passes ===")
model.train()
for i in range(5):
    model.batch_dict = batch_dict
    logits = model(inputs)
    if model._last_loss_details is not None:
        print(f"  Pass {i+1}: loss={logits.item():.4f}, box={model._last_loss_details['box']:.4f}, cls={model._last_loss_details['cls']:.4f}, dfl={model._last_loss_details['dfl']:.4f}")
    else:
        print(f"  Pass {i+1}: loss={logits.item():.4f}, _last_loss_details=None")

# Test with empty batch (no objects)
print(f"\n\n=== Testing with empty batch ===")
empty_batch_dict = {
    "img": batch_dict["img"],
    "inputs": batch_dict["inputs"],
    "targets": torch.zeros(1),
    "bboxes": torch.zeros(0, 4),
    "cls": torch.zeros(0, 1),
    "batch_idx": torch.zeros(0, dtype=torch.long),
}
model.batch_dict = empty_batch_dict
logits = model(inputs)
print(f"Empty batch loss: {logits.item():.4f}")
if model._last_loss_details is not None:
    print(f"  box={model._last_loss_details['box']:.4f}, cls={model._last_loss_details['cls']:.4f}, dfl={model._last_loss_details['dfl']:.4f}")
else:
    print(f"  _last_loss_details=None")