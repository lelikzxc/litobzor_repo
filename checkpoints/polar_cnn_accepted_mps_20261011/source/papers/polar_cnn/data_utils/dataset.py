"""WM-811K PNG inventory and compact, precomputed polar-sector inputs."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .preprocessing import preprocess

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
ALIASES = {name.lower().replace("_", "-"): i for i, name in enumerate(CLASSES)}


class WaferInventory:
    def __init__(self, data_root, selection="all_maps", reviewed_ids=None):
        self.root = Path(data_root).resolve()
        if selection not in {"all_maps", "native26"}:
            raise ValueError("selection must be all_maps or native26")
        self.samples = []
        seen = set()
        with (self.root / "labels.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"filename", "failureType"} <= set(reader.fieldnames or []):
                raise ValueError("labels.csv needs filename and failureType columns")
            for row in reader:
                text = row["failureType"].strip().lower().replace("_", "-")
                if text in {"", "none", "nan", "0", "unlabeled"}:
                    continue
                if text.isdigit() and 1 <= int(text) <= 8:
                    label = int(text) - 1
                elif text in ALIASES:
                    label = ALIASES[text]
                else:
                    raise ValueError(f"Unknown defect label {text!r}")
                filename = row["filename"].strip().replace("\\", "/")
                path = (self.root / "images" / filename).resolve()
                if not path.is_relative_to(self.root / "images"):
                    raise ValueError("Wafer filename escapes images directory")
                if filename in seen:
                    raise ValueError(f"Duplicate wafer {filename}")
                seen.add(filename)
                self.samples.append((filename, label))
        if reviewed_ids:
            ids = json.loads(Path(reviewed_ids).read_text(encoding="utf-8"))
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                raise ValueError("reviewed_ids must contain a JSON list of original filenames")
            if len(ids) != len(set(ids)) or not set(ids) <= seen:
                raise ValueError("reviewed_ids contains duplicates or unknown wafers")
            selected = set(ids)
            self.samples = [sample for sample in self.samples if sample[0] in selected]
        if selection == "native26":
            selected = []
            for sample in self.samples:
                with Image.open(self.path(sample[0])) as image:
                    if image.size == (26, 26):
                        selected.append(sample)
            self.samples = selected
        if not self.samples:
            raise ValueError("No defect wafers in the selected inventory")
        digest = hashlib.sha256(json.dumps(self.samples, separators=(",", ":")).encode())
        # Detect changed PNG contents as well as changed filenames/labels.
        for filename, _ in self.samples:
            digest.update(hashlib.sha256(self.path(filename).read_bytes()).digest())
        self.identity = digest.hexdigest()

    def path(self, filename):
        return self.root / "images" / filename


class PolarDataset(Dataset):
    """One item references an original wafer and a deterministic angle."""

    def __init__(self, inventory, records, threshold=3, progress=False, **preprocessing):
        self.inventory = inventory
        self.records = records
        features = []
        targets = []
        for record in tqdm(
            records,
            desc="Prepare polar counts",
            unit="map",
            disable=not progress,
            mininterval=0.5,
            leave=False,
        ):
            index, angle = record
            filename, label = inventory.samples[index]
            with Image.open(inventory.path(filename)) as image:
                features.append(preprocess(image, threshold, angle, **preprocessing))
            targets.append(label)
        if not features:
            raise ValueError("Empty polar dataset")
        self.inputs = torch.from_numpy(np.stack(features)).unsqueeze(1)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {"inputs": self.inputs[index], "targets": self.targets[index]}
