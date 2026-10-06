"""Evaluate classification or segmentation using the saved training protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from common.engine.config import EngineConfig
from common.training.metrics import accuracy, f1, precision, recall
from papers.reproduction import load_holdout
from papers.semiwafernet.data_utils import WaferSegmentationDataset, WaferWM811KDataset
from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.training.segmentation import (
    SegmentationLoss,
    SegmentationWrapper,
    metric_functions,
)
from papers.semiwafernet.utils.checkpoint import normalize_semiwafernet_state_dict


class ClassificationWrapper(nn.Module):
    def __init__(self, base_model, log_prior=None):
        super().__init__()
        self.base_model = base_model
        if log_prior is not None:
            self.register_buffer("log_prior", log_prior)
        else:
            self.log_prior = None

    def forward(self, x):
        logits = self.base_model(x)["classification"]
        return logits if self.log_prior is None else logits + self.log_prior


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/semiwafernet_reproduction/best.pt")
    parser.add_argument(
        "--config", default=None, help="Optional config: must match saved model and data"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    import json

    protocol = Path(args.checkpoint).parent / "protocol.json"
    if not protocol.exists():
        raise FileNotFoundError(
            f"Missing {protocol}; use checkpoints from the corrected training entry point"
        )
    saved = json.loads(protocol.read_text())["config"]
    config = EngineConfig.from_dict(saved)
    if args.config:
        supplied = EngineConfig.from_yaml(args.config)
        for section in ("model", "data"):
            if supplied.get(section) != config.get(section):
                raise ValueError(
                    f"Supplied {section} config differs from saved training protocol; omit --config"
                )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seg = config.get("model.mode") == "segmentation"
    if seg:
        dataset = WaferSegmentationDataset(
            config.get("data.seg_data_root"),
            split="test",
            image_size=config.get("data.seg_image_size", 64),
            train=False,
        )
        samples = [(p.name, 0) for p in dataset._image_paths]
    else:
        dataset = WaferWM811KDataset(
            config.get("data.data_root"),
            image_size=config.get("data.image_size", 32),
            train=False,
            hybrid_sampling=False,
            split="test" if config.get("data.use_official_split", True) else None,
        )
        samples = dataset._samples
    indices, _ = load_holdout(args.checkpoint, samples, "test")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = normalize_semiwafernet_state_dict(
        state.get("model", state.get("model_state_dict", state))
    )
    base = SemiWaferNet.from_config(config)
    model = (
        SegmentationWrapper(base) if seg else ClassificationWrapper(base, weights.get("log_prior"))
    )
    model.load_state_dict(weights)
    model.to(device).eval()
    loss_fn = SegmentationLoss(model) if seg else nn.CrossEntropyLoss()
    loss_fn.to(device)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=config.get("evaluation.batch_size", 64), shuffle=False
    )
    logits_all, targets_all = [], []
    loss_sum, count = 0.0, 0
    for batch in tqdm(loader, desc="Test"):
        x = batch["image"].to(device)
        y = batch["mask" if seg else "label"].to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * len(y)
        count += len(y)
        logits_all.append(logits.cpu())
        targets_all.append(y.cpu())
    if not count:
        raise ValueError("Saved evaluation split is empty")
    logits, targets = torch.cat(logits_all), torch.cat(targets_all)
    metrics = (
        metric_functions()
        if seg
        else {
            "accuracy": accuracy,
            "f1": lambda x, y: f1(x, y, num_classes=config.get("model.num_classes", 9)),
            "precision": lambda x, y: precision(
                x, y, num_classes=config.get("model.num_classes", 9)
            ),
            "recall": lambda x, y: recall(x, y, num_classes=config.get("model.num_classes", 9)),
        }
    )
    print(f"Samples: {count}; loss: {loss_sum / count:.6f}")
    for name, fn in metrics.items():
        print(f"{name}: {fn(logits, targets):.6f}")


if __name__ == "__main__":
    main()
