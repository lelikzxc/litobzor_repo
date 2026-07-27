"""Stratified split for Magnetic Tile dataset.

The original Roboflow split (228 train / 10 valid / 10 test) is too small
and unbalanced for validation/testing. This script merges all splits and
creates a new stratified split by image-level class presence.

Usage:
    python papers/ctm_yolov10/scripts/stratify_split.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def main() -> None:
    data_root = Path("datasets/magnetic_tile")
    backup_root = data_root / "original_split_backup"

    # Ensure backup exists
    if not backup_root.exists():
        print(f"Backup not found at {backup_root}. Creating backup first...")
        for split in ("train", "valid", "test"):
            for subdir in ("images", "labels"):
                src = data_root / split / subdir
                dst = backup_root / split / subdir
                if src.exists():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
        print("Backup created.")

    # ── 1. Collect all images and labels from BACKUP ──────────────────
    all_stems: list[str] = []
    all_classes: list[int] = []  # primary class for stratification

    for split in ("train", "valid", "test"):
        lbl_dir = backup_root / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            with open(lbl_path) as f:
                classes = [int(line.split()[0]) for line in f if line.strip().split()]
            if not classes:
                continue
            all_stems.append(lbl_path.stem)
            # Use the most frequent class as stratify key
            all_classes.append(max(set(classes), key=classes.count))

    all_stems = np.array(all_stems)
    all_classes = np.array(all_classes)

    print(f"Total samples collected: {len(all_stems)}")
    unique, counts = np.unique(all_classes, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    # ── 2. Stratified split ──────────────────────────────────────────
    # 70% train, 15% val, 15% test
    train_idx, temp_idx = train_test_split(
        np.arange(len(all_stems)),
        test_size=0.30,
        random_state=42,
        stratify=all_classes,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        stratify=all_classes[temp_idx],
    )

    splits = {
        "train": train_idx,
        "valid": val_idx,
        "test": test_idx,
    }

    for name, idx in splits.items():
        print(f"{name}: {len(idx)} samples")

    # ── 3. Write new split (copy from backup) ────────────────────────
    print("\nWriting new stratified split ...")
    for name, idx in splits.items():
        img_dir = data_root / name / "images"
        lbl_dir = data_root / name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Clear existing files
        for f in img_dir.iterdir():
            f.unlink()
        for f in lbl_dir.iterdir():
            f.unlink()

        for i in idx:
            stem = all_stems[i]
            # Find the file in backup (search all splits)
            found_img = False
            found_lbl = False
            for src_split in ("train", "valid", "test"):
                src_img = backup_root / src_split / "images" / f"{stem}.jpg"
                if not src_img.exists():
                    src_img = backup_root / src_split / "images" / f"{stem}.png"
                if src_img.exists() and not found_img:
                    shutil.copy2(src_img, img_dir / src_img.name)
                    found_img = True

                src_lbl = backup_root / src_split / "labels" / f"{stem}.txt"
                if src_lbl.exists() and not found_lbl:
                    shutil.copy2(src_lbl, lbl_dir / src_lbl.name)
                    found_lbl = True

                if found_img and found_lbl:
                    break

        # Verify
        n_img = len(list(img_dir.glob("*")))
        n_lbl = len(list(lbl_dir.glob("*")))
        print(f"  {name}: {n_img} images, {n_lbl} labels")

    # ── 4. Verify class distribution in new splits ───────────────────
    print("\nVerifying class distribution:")
    for name in ("train", "valid", "test"):
        label_dir = data_root / name / "labels"
        class_counts: dict[int, int] = {}
        for lbl_path in sorted(label_dir.glob("*.txt")):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c = int(parts[0])
                        class_counts[c] = class_counts.get(c, 0) + 1
        print(f"  {name}: {class_counts}")

    print("\nDone! Stratified split complete.")


if __name__ == "__main__":
    main()