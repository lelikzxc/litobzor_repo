"""Diagnostic script for CTM-IYOLOv10 training issues."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

import torch
import numpy as np

print("=" * 60)
print("CTM-IYOLOv10 DIAGNOSTIC")
print("=" * 60)

# 1. Dataset analysis
print("\n1. DATASET ANALYSIS")
print("-" * 40)
data_root = project_root / "datasets" / "magnetic_tile"

for split in ["train", "valid", "test"]:
    label_dir = data_root / split / "labels"
    if not label_dir.exists():
        print(f"{split}: no label dir")
        continue
    total_objs = 0
    class_counts = {}
    files_with_objs = 0
    files_total = 0
    for lbl_path in sorted(label_dir.glob("*.txt")):
        files_total += 1
        with open(lbl_path) as f:
            objs = [line.strip().split() for line in f if line.strip()]
        if objs:
            files_with_objs += 1
        for obj in objs:
            cls_id = int(obj[0])
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            total_objs += 1
    print(f"{split}: {files_total} images, {files_with_objs} with objects, {total_objs} total objects")
    for c in sorted(class_counts):
        print(f"  Class {c}: {class_counts[c]} objects")
    print()

# 2. Model forward pass analysis
print("\n2. MODEL FORWARD PASS (EVAL MODE)")
print("-" * 40)

from common.engine.config import EngineConfig
from papers.ctm_yolov10.models.yolov10 import CTMIYOLOv10

config_path = project_root / "papers" / "ctm_yolov10" / "configs" / "config.yaml"
config = EngineConfig.from_yaml(config_path)
model = CTMIYOLOv10.from_config(config)
model.eval()
print(f"Model: GhostConv={model.ghost_conv}, BiFPN={model.bifpn}, num_classes={model.num_classes}")

x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    out = model(x)

print(f"Output type: {type(out)}")
if isinstance(out, (tuple, list)):
    print(f"  len: {len(out)}")
    for i, o in enumerate(out):
        if isinstance(o, torch.Tensor):
            print(f"  [{i}] shape: {o.shape}, dtype={o.dtype}")
            print(f"      min={o.min():.4f}, max={o.max():.4f}, mean={o.mean():.4f}")
elif isinstance(out, torch.Tensor):
    print(f"  shape: {out.shape}, dtype={out.dtype}")
    print(f"  min: {out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
    if out.dim() == 3:
        print(f"  First image, first 5 preds:\n{out[0, :5]}")

# 3. Check v10Detect head structure
print("\n3. v10Detect HEAD ANALYSIS")
print("-" * 40)
head = model.seq[-1]
print(f"Head type: {type(head).__name__}")
print(f"Head.nc: {head.nc}")
print(f"Head.nl: {head.nl}")
print(f"Head.na: {head.na}")
print(f"Head.reg_max: {head.reg_max}")
print(f"Head.stride: {head.stride}")

# Check what the head returns in eval mode
with torch.no_grad():
    # Run forward through the sequential model
    y = []
    for m in model.seq:
        if m.f != -1:
            if isinstance(m.f, int):
                x_in = y[m.f]
            else:
                x_in = [x if j == -1 else y[j] for j in m.f]
        else:
            x_in = x
        
        x_out = m(x_in)
        
        if model.ctm_enabled and m.i == 10:
            x_out = model.ctm(x_out)
        
        y.append(x_out if m.i in model.base_model.save else None)
    
    print(f"\nHead output type: {type(x_out)}")
    if isinstance(x_out, (tuple, list)):
        print(f"  len: {len(x_out)}")
        for i, o in enumerate(x_out):
            if isinstance(o, torch.Tensor):
                print(f"  [{i}] shape: {o.shape}")
                print(f"      min={o.min():.4f}, max={o.max():.4f}")
    elif isinstance(x_out, torch.Tensor):
        print(f"  shape: {x_out.shape}")

# 4. Compare with original YOLOv10
print("\n4. ORIGINAL YOLOv10 FORWARD")
print("-" * 40)
from ultralytics import YOLO
yolo = YOLO("yolov10n.pt")
yolo.model.eval()
with torch.no_grad():
    orig_out = yolo.model(x)
print(f"Original output type: {type(orig_out)}")
if isinstance(orig_out, (tuple, list)):
    print(f"  len: {len(orig_out)}")
    for i, o in enumerate(orig_out):
        if isinstance(o, torch.Tensor):
            print(f"  [{i}] shape: {o.shape}")
            print(f"      min={o.min():.4f}, max={o.max():.4f}, mean={o.mean():.4f}")

# 5. Check loss computation
print("\n5. LOSS COMPUTATION CHECK")
print("-" * 40)
model.train()
# Create a fake batch
batch = {
    "img": torch.randn(2, 3, 640, 640),
    "bboxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]]),
    "cls": torch.tensor([[0], [1]]),
    "batch_idx": torch.tensor([0, 1]),
}
loss_out = model(batch)
print(f"Loss output type: {type(loss_out)}")
if isinstance(loss_out, (tuple, list)):
    print(f"  len: {len(loss_out)}")
    print(f"  loss[0]: {loss_out[0]}")
    if len(loss_out) > 1:
        print(f"  loss[1]: {loss_out[1]}")
elif isinstance(loss_out, torch.Tensor):
    print(f"  loss: {loss_out}")

# 6. Check criterion structure
print("\n6. CRITERION STRUCTURE")
print("-" * 40)
if hasattr(model.base_model, "criterion") and model.base_model.criterion is not None:
    crit = model.base_model.criterion
    print(f"criterion type: {type(crit).__name__}")
    if hasattr(crit, "hyp"):
        print(f"  hyp: {crit.hyp}")
        print(f"  hyp.box: {crit.hyp.box}")
        print(f"  hyp.cls: {crit.hyp.cls}")
        print(f"  hyp.dfl: {crit.hyp.dfl}")
    if hasattr(crit, "one2many"):
        print(f"  one2many: {type(crit.one2many).__name__}")
        if hasattr(crit.one2many, "hyp"):
            print(f"    hyp.box: {crit.one2many.hyp.box}")
            print(f"    hyp.cls: {crit.one2many.hyp.cls}")
            print(f"    hyp.dfl: {crit.one2many.hyp.dfl}")
    if hasattr(crit, "one2one"):
        print(f"  one2one: {type(crit.one2one).__name__}")
        if hasattr(crit.one2one, "hyp"):
            print(f"    hyp.box: {crit.one2one.hyp.box}")
            print(f"    hyp.cls: {crit.one2one.hyp.cls}")
            print(f"    hyp.dfl: {crit.one2one.hyp.dfl}")
else:
    print("criterion not initialized yet")

print("\nDiagnostic complete!")