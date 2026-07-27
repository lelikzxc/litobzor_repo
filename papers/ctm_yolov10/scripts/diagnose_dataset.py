"""Diagnose Magnetic Tile dataset statistics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from papers.ctm_yolov10.data_utils.magnetic_dataset import MagneticTileDataset

for split in ["train", "valid", "test"]:
    ds = MagneticTileDataset("datasets/magnetic_tile", split)
    print(f"\n{'='*50}")
    print(f"Split: {split} — {len(ds)} samples")
    print(f"{'='*50}")
    
    class_counts = {0: 0, 1: 0, 2: 0}
    total_objs = 0
    empty_images = 0
    
    for i in range(len(ds)):
        item = ds[i]
        label = item["label"]
        if label.numel() == 0:
            empty_images += 1
            continue
        for c in label[:, 0].long():
            class_counts[int(c)] = class_counts.get(int(c), 0) + 1
        total_objs += label.shape[0]
    
    print(f"  Total objects: {total_objs}")
    print(f"  Empty images:  {empty_images}")
    print(f"  Class distribution:")
    for c in sorted(class_counts.keys()):
        pct = class_counts[c] / max(total_objs, 1) * 100
        print(f"    Class {c}: {class_counts[c]} ({pct:.1f}%)")
    
    # Check label ranges
    if total_objs > 0:
        all_cx, all_cy, all_w, all_h = [], [], [], []
        for i in range(len(ds)):
            item = ds[i]
            label = item["label"]
            if label.numel() == 0:
                continue
            all_cx.extend(label[:, 1].tolist())
            all_cy.extend(label[:, 2].tolist())
            all_w.extend(label[:, 3].tolist())
            all_h.extend(label[:, 4].tolist())
        print(f"  cx range: [{min(all_cx):.4f}, {max(all_cx):.4f}]")
        print(f"  cy range: [{min(all_cy):.4f}, {max(all_cy):.4f}]")
        print(f"  w  range: [{min(all_w):.4f}, {max(all_w):.4f}]")
        print(f"  h  range: [{min(all_h):.4f}, {max(all_h):.4f}]")