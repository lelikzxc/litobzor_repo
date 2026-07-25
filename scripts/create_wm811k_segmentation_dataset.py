#!/usr/bin/env python3
"""Generate a semantic segmentation dataset from WM-811K wafer maps.

Converts the WM-811K classification dataset (grayscale wafer maps with
defect labels) into a binary segmentation dataset where defective pixels
(value 254) become 255 and all other pixels become 0.

Usage:
    python scripts/create_wm811k_segmentation_dataset.py

Output:
    datasets/wm811k_seg/
        train/images/  (original wafer maps)
        train/masks/   (binary segmentation masks)
        val/images/
        val/masks/
        test/images/
        test/masks/
        dataset.yaml   (metadata)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

# Paths
SOURCE_DIR = Path("datasets/wm811k")
IMAGES_DIR = SOURCE_DIR / "images"
LABELS_PATH = SOURCE_DIR / "labels.csv"

OUTPUT_DIR = Path("datasets/wm811k_seg")
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

RANDOM_STATE = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Classes to exclude (lowercase, as stored in labels.csv)
EXCLUDED_CLASSES = {"none", "random"}


def load_labels(path: Path) -> pd.DataFrame:
    """Load labels.csv and return a DataFrame with filename and defect_class."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    name_col = None
    class_col = None
    for col in df.columns:
        if col in ("filename", "file", "image", "image_name", "name", "file_name"):
            name_col = col
        if col in ("defect_class", "class", "label", "defect", "type", "failure_type"):
            class_col = col
    if name_col is None or class_col is None:
        raise ValueError(
            f"Cannot identify filename/class columns in {list(df.columns)}. "
            f"Expected columns like 'filename' and 'defect_class'."
        )
    df = df.rename(columns={name_col: "filename", class_col: "defect_class"})
    df["filename"] = df["filename"].astype(str).str.strip()
    df["defect_class"] = df["defect_class"].astype(str).str.strip().str.lower()
    return df


def generate_mask(image_path: Path) -> np.ndarray:
    """Generate a binary segmentation mask from a wafer map image.

    White pixels (value 254) become 255 (defect).
    All other pixels become 0 (background / normal die / outside wafer).
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    mask = np.where(arr == 254, 255, 0).astype(np.uint8)
    return mask


def validate_mask(mask: np.ndarray, image_path: Path, mask_path: Path) -> None:
    """Validate that a mask contains only 0 and 255."""
    unique = np.unique(mask)
    if not set(unique).issubset({0, 255}):
        raise ValueError(
            f"Mask {mask_path} (from {image_path.name}) contains invalid values: "
            f"{unique.tolist()}. Expected only [0, 255]."
        )


def main() -> None:
    print("=" * 60)
    print("  WM-811K -> Segmentation Dataset Generator")
    print("=" * 60)

    # 1. Load labels
    print(f"\n[1] Loading labels from {LABELS_PATH}...")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
    df = load_labels(LABELS_PATH)
    total_samples = len(df)
    print(f"    Total samples: {total_samples}")

    # 2. Filter excluded classes
    print(f"\n[2] Filtering excluded classes: {EXCLUDED_CLASSES}")
    excluded_mask = df["defect_class"].isin(EXCLUDED_CLASSES)
    excluded_count = excluded_mask.sum()
    df_filtered = df[~excluded_mask].reset_index(drop=True)
    print(f"    Excluded: {excluded_count} samples")
    print(f"    Remaining: {len(df_filtered)} samples")

    if excluded_count > 0:
        excluded_dist = df[excluded_mask]["defect_class"].value_counts()
        print("    Excluded class distribution:")
        for cls, cnt in excluded_dist.items():
            print(f"      {cls}: {cnt}")

    # 3. Class distribution before split
    print(f"\n[3] Class distribution before split:")
    class_dist = df_filtered["defect_class"].value_counts().sort_index()
    for cls, cnt in class_dist.items():
        print(f"    {cls}: {cnt}")

    # 4. Stratified split
    print(f"\n[4] Performing stratified split "
          f"(train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO})...")

    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
    )
    train_idx, temp_idx = next(sss1.split(df_filtered, df_filtered["defect_class"]))

    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=val_size,
        random_state=RANDOM_STATE,
    )
    val_idx, test_idx = next(
        sss2.split(df_filtered.iloc[temp_idx], df_filtered.iloc[temp_idx]["defect_class"])
    )
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    train_df = df_filtered.iloc[train_idx].reset_index(drop=True)
    val_df = df_filtered.iloc[val_idx].reset_index(drop=True)
    test_df = df_filtered.iloc[test_idx].reset_index(drop=True)

    print(f"    Train: {len(train_df)}")
    print(f"    Val:   {len(val_df)}")
    print(f"    Test:  {len(test_df)}")

    # 5. Class distribution after split
    print(f"\n[5] Class distribution after split:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split_df["defect_class"].value_counts().sort_index()
        dist_str = ", ".join(f"{cls}={cnt}" for cls, cnt in dist.items())
        print(f"    {split_name}: {dist_str}")

    # 6. Create output directories
    print(f"\n[6] Creating output directories...")
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)
        print(f"    Created: {split_dir.relative_to(OUTPUT_DIR.parent)}/")

    # 7. Generate images and masks
    print(f"\n[7] Generating images and masks...")

    total_copied = 0
    total_masks = 0

    for split_name, split_df, split_dir in [
        ("train", train_df, TRAIN_DIR),
        ("val", val_df, VAL_DIR),
        ("test", test_df, TEST_DIR),
    ]:
        img_out = split_dir / "images"
        mask_out = split_dir / "masks"

        for _, row in split_df.iterrows():
            fname = row["filename"]
            stem = Path(fname).stem
            src_img = IMAGES_DIR / f"{stem}.png"

            if not src_img.exists():
                print(f"    WARNING: Image not found: {src_img}, skipping")
                continue

            dst_img = img_out / f"{stem}.png"
            dst_img.write_bytes(src_img.read_bytes())

            mask = generate_mask(src_img)
            dst_mask = mask_out / f"{stem}.png"
            Image.fromarray(mask).save(dst_mask)

            validate_mask(mask, src_img, dst_mask)

            total_copied += 1
            total_masks += 1

        print(f"    {split_name}: {len(split_df)} images, {len(split_df)} masks")

    # 8. Validation
    print(f"\n[8] Validating dataset...")

    errors: list[str] = []

    for split_name, split_dir in [
        ("train", TRAIN_DIR),
        ("val", VAL_DIR),
        ("test", TEST_DIR),
    ]:
        img_dir = split_dir / "images"
        mask_dir = split_dir / "masks"

        img_files = sorted(img_dir.glob("*.png"))
        mask_files = sorted(mask_dir.glob("*.png"))

        img_stems = {f.stem for f in img_files}
        mask_stems = {f.stem for f in mask_files}
        missing_masks = img_stems - mask_stems
        missing_images = mask_stems - img_stems

        if missing_masks:
            errors.append(
                f"{split_name}: {len(missing_masks)} images without masks: "
                f"{sorted(missing_masks)[:5]}..."
            )
        if missing_images:
            errors.append(
                f"{split_name}: {len(missing_images)} masks without images: "
                f"{sorted(missing_images)[:5]}..."
            )

        if img_files and mask_files:
            ref_img = Image.open(img_files[0])
            ref_size = ref_img.size
            for f in img_files[1:]:
                img = Image.open(f)
                if img.size != ref_size:
                    errors.append(
                        f"{split_name}: Image {f.name} has size {img.size}, "
                        f"expected {ref_size}"
                    )
            for f in mask_files:
                mask = np.array(Image.open(f))
                if mask.shape[:2] != (ref_size[1], ref_size[0]):
                    errors.append(
                        f"{split_name}: Mask {f.name} has shape {mask.shape[:2]}, "
                        f"expected ({ref_size[1]}, {ref_size[0]})"
                    )
                validate_mask(mask, f, f)

    if errors:
        error_msg = "\n".join(errors)
        raise RuntimeError(f"Validation failed:\n{error_msg}")

    print("    [OK] All images have corresponding masks")
    print("    [OK] All masks have identical resolution to their images")
    print("    [OK] All masks contain only values [0, 255]")

    # 9. Create dataset.yaml
    print(f"\n[9] Creating dataset.yaml...")

    yaml_content = f"""# WM-811K Segmentation Dataset
# Auto-generated by scripts/create_wm811k_segmentation_dataset.py

dataset:
    name: wm811k_seg

classes:
    - background
    - defect

image_format: png

mask_format: png

splits:
    train: {len(train_df)}
    val: {len(val_df)}
    test: {len(test_df)}
"""
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"    Created: {yaml_path.relative_to(OUTPUT_DIR.parent)}")

    # 10. Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  Source:           {SOURCE_DIR}")
    print(f"  Output:           {OUTPUT_DIR}")
    print(f"  Total samples:    {total_samples}")
    print(f"  Excluded:         {excluded_count} ({EXCLUDED_CLASSES})")
    print(f"  Used:             {len(df_filtered)}")
    print(f"  Train:            {len(train_df)}")
    print(f"  Val:              {len(val_df)}")
    print(f"  Test:             {len(test_df)}")
    print(f"  Images copied:    {total_copied}")
    print(f"  Masks generated:  {total_masks}")
    print(f"  Validation:       [OK] Passed")
    print(f"{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()