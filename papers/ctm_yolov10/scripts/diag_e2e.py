"""Diagnose E2ELoss format expectations."""
import sys, inspect
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from ultralytics import YOLO

yolo = YOLO("yolov10n.yaml")
model = yolo.model
crit = model.criterion

print(f"Criterion type: {type(crit).__name__}")
print(f"hyp type: {type(crit.hyp)}")
print(f"  hyp.box: {crit.hyp.box}")
print(f"  hyp.cls: {crit.hyp.cls}")
print(f"  hyp.dfl: {crit.hyp.dfl}")

# Check if it has one2many/one2one
if hasattr(crit, "one2many"):
    print(f"\none2many: {type(crit.one2many).__name__}")
    print(f"  one2many hyp type: {type(crit.one2many.hyp)}")
if hasattr(crit, "one2one"):
    print(f"\none2one: {type(crit.one2one).__name__}")
    print(f"  one2one hyp type: {type(crit.one2one.hyp)}")

# Find the source file
src_file = inspect.getfile(type(crit))
print(f"\nSource: {src_file}")

# Read the __call__ method
with open(src_file, "r") as f:
    content = f.read()

# Find __call__ or forward
if "__call__" in content:
    idx = content.index("__call__")
    print(f"\n--- __call__ method (around line {content[:idx].count(chr(10)) + 1}) ---")
    # Print 50 lines after
    lines = content.split("\n")
    start = content[:idx].count(chr(10))
    for i in range(start, min(start + 50, len(lines))):
        print(f"  {lines[i]}")