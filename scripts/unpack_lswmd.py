#!/usr/bin/env python3
"""Unpack datasets/LSWMD.pkl (WM-811K wafer maps) into datasets/wm811k/images.

The pickle is a pandas DataFrame (created by an old pandas version), where each
row holds a 2D wafer map array plus metadata (lot name, wafer index, defect
class, train/test split). This script:

  * loads the pickle (handles the legacy ``pandas.indexes.*`` module paths),
  * writes every wafer map as a grayscale PNG into ``datasets/wm811k/images``,
  * writes ``datasets/wm811k/labels.csv`` with every per-wafer metadata column
    (lot name, wafer index, train/test split, failure type, ...),
  * shows a tqdm progress bar.

By default nothing existing is overwritten (already-present images are skipped
and ``labels.csv`` is left untouched). Pass ``--force`` to first **clear** the
output ``images`` folder and delete ``labels.csv``, then write everything fresh.
This is the recommended way to get a clean, de-duplicated dataset.

Usage:
    python scripts/unpack_lswmd.py                 # incremental, safe
    python scripts/unpack_lswmd.py --force         # wipe & regenerate

Notes on the pixel values:
    Wafer map arrays in this dataset commonly use -1 / 0 / 1 where -1 means
    "no die (background)", 0 means "normal die", 1 means "defective die".
    By default the script keeps these raw integer values in the PNG (so
    downstream code sees exactly the same numbers). Pass --normalize to
    stretch them to the full 0..255 range instead.

Dependencies:
    Pillow, tqdm  (pip install Pillow tqdm)
"""

from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import sys
import warnings
from pathlib import Path

# Silence the harmless NumPy deprecation warning emitted while unpickling old
# arrays (their dtype stored an obsolete ``align=0`` argument).
warnings.filterwarnings("ignore", message=r"dtype\(\): align")

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Missing dependency ({exc}). Run: pip install numpy Pillow")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        """Minimal fallback progress reporter when tqdm is unavailable."""
        total = kwargs.get("total")
        for i, item in enumerate(iterable):
            if total:
                print(f"\r  {i + 1}/{total} ({(i + 1) / total * 100:5.1f}%)", end="", flush=True)
            yield item
        if total:
            print()


# Legacy module paths moved into pandas.core.* in modern pandas.
OLD_TO_NEW = {
    "pandas.indexes.base": "pandas.core.indexes.base",
    "pandas.indexes.range": "pandas.core.indexes.range",
}


class RemapUnpickler(pickle.Unpickler):
    def find_class(self, module, name):  # noqa: D102
        module = OLD_TO_NEW.get(module, module)
        return super().find_class(module, name)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return RemapUnpickler(f, encoding="latin1").load()


def pick_column(columns, candidates):
    """Return the first column matching any candidate (case-insensitive)."""
    col_map = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    return None


def _plain(value):
    """Unwrap numpy scalars / arrays into plain Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def safe_name(value) -> str:
    """Render a metadata value as a clean filename fragment."""
    value = _plain(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    s = str(value).replace("/", "_").replace("\\", "_")
    return s.strip() if s else ""


def csv_value(value) -> str:
    """Serialize a (possibly nested array-like) metadata value for CSV output."""
    value = _plain(value)
    if isinstance(value, (list, tuple)):
        return ";".join(csv_value(v) for v in value)
    if hasattr(value, "shape"):  # numpy ndarray
        return ";".join(csv_value(v) for v in value.ravel())
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return "" if value is None else str(value)


def clear_output(out_dir: Path) -> None:
    """Delete the output ``images`` folder and any existing ``labels.csv``."""
    images_dir = out_dir / "images"
    labels_csv = out_dir / "labels.csv"

    removed = 0
    if images_dir.exists():
        for child in images_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed += 1
        print(f"Cleared images folder: {images_dir} ({removed} entries removed)")

    if labels_csv.exists():
        labels_csv.unlink()
        print(f"Removed existing labels: {labels_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unpack LSWMD.pkl wafer maps into a folder of PNG images."
    )
    parser.add_argument("--pkl", type=Path, default=Path("datasets/LSWMD.pkl"))
    parser.add_argument(
        "--out", type=Path, default=Path("datasets/wm811k"), help="Output root folder."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Stretch wafer-map values to the full 0..255 range before saving.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear the output images folder and labels.csv before writing.",
    )
    args = parser.parse_args()

    if args.force:
        clear_output(args.out)

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.pkl} ...")
    df = load_pickle(args.pkl)
    if not hasattr(df, "columns"):
        sys.exit(f"Unexpected object in pickle: {type(df)} (expected pandas.DataFrame)")

    cols = list(df.columns)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {cols}")

    map_col = pick_column(cols, ["wafermap", "wafer_map", "map", "image", "data", "img"])
    fail_col = pick_column(cols, ["failuretype", "defect_class", "class", "label", "defect"])
    lot_col = pick_column(cols, ["lotname", "lot", "batch"])
    idx_col = pick_column(cols, ["waferindex", "wafer_idx", "index", "wafer"])

    if map_col is None:
        sys.exit(f"Could not find the wafer-map column among: {cols}")
    print(f"Using map column: {map_col}")
    print(f"Using failure-type column: {fail_col}")
    print(f"Using lot column: {lot_col}, wafer-index column: {idx_col}")

    # All non-image columns are treated as per-wafer metadata written to CSV.
    meta_cols = [c for c in cols if c != map_col]
    if not meta_cols:
        sys.exit("No metadata columns besides the wafer map found.")

    labels = []
    seen_names: set[str] = set()
    n_existing = 0
    n_written = 0
    n_skipped = 0

    total = len(df)
    for row_i, raw_map in enumerate(tqdm(df[map_col], total=total, desc="Saving images", unit="img")):
        arr = np.asarray(raw_map)
        if arr.ndim != 2:
            n_skipped += 1
            continue

        # Build a stable, human-readable base name from metadata when available.
        lot = safe_name(df.iloc[row_i][lot_col]) if lot_col else ""
        widx = safe_name(df.iloc[row_i][idx_col]) if idx_col else ""
        base = f"{lot}_{widx}" if (lot and widx) else f"wafer_{row_i}"
        name = base
        if name in seen_names:
            name = f"{base}_{row_i}"
        seen_names.add(name)

        rel_path = f"{name}.png"
        out_path = images_dir / rel_path
        if out_path.exists():
            n_existing += 1
        else:
            if args.normalize:
                lo, hi = float(arr.min()), float(arr.max())
                if hi > lo:
                    img_arr = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
                else:
                    img_arr = np.zeros_like(arr, dtype=np.uint8)
            else:
                img_arr = arr.astype(np.uint8)
            Image.fromarray(img_arr, mode="L").save(out_path)
            n_written += 1

        row_meta = [csv_value(df.iloc[row_i][c]) for c in meta_cols]
        labels.append([rel_path] + row_meta)

    print("\nDone.")
    print(f"  images written     : {n_written}")
    print(f"  images already exist: {n_existing}")
    print(f"  rows skipped (bad map): {n_skipped}")

    # labels.csv is a derived file, so it is always regenerated from the
    # current run. (Without --force, the PNG images themselves are not
    # overwritten; with --force the folder was emptied beforehand.)
    csv_path = args.out / "labels.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + [str(c) for c in meta_cols])
        writer.writerows(labels)
    print(f"  wrote labels       : {csv_path} ({len(labels)} rows)")


if __name__ == "__main__":
    main()