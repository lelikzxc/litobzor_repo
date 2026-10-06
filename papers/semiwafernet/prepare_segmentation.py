"""Generate SemiWaferNet masks from categorical states in the raw WM-811K export."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from papers.semiwafernet.data_utils.wafer_dataset import (
    load_wafer_die_map,
    parse_wm811k_labeled_rows,
)


def prepare(root, output, seed=42):
    root, output = Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty to prevent mixing different splits: {output}")
    samples = [(f, y) for f, y in parse_wm811k_labeled_rows(root / "labels.csv") if y not in (0, 7)]
    labels = [y for _, y in samples]
    train, rest = train_test_split(
        np.arange(len(samples)), test_size=0.2, stratify=labels, random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=[labels[i] for i in rest], random_state=seed
    )
    manifest = {"seed": seed, "counts": np.bincount(labels, minlength=9).tolist(), "splits": {}}
    for split, indices in (("train", train), ("val", val), ("test", test)):
        image_dir, mask_dir = output / split / "images", output / split / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        manifest["splits"][split] = []
        for i in tqdm(indices, desc=split):
            filename, label = samples[int(i)]
            die = load_wafer_die_map(root / "images" / filename)
            # Stable visible intensity mapping; masks use original states, before resizing.
            Image.fromarray((die * 127).astype(np.uint8)).save(image_dir / filename)
            Image.fromarray(((die == 2) * 255).astype(np.uint8)).save(mask_dir / filename)
            manifest["splits"][split].append([filename, label])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(samples)} maps; counts={manifest['counts']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/wm811k")
    parser.add_argument("--output", default="datasets/semiwafernet_seg")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.data_root, args.output, args.seed)
