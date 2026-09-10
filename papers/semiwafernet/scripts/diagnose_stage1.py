"""Diagnose Stage-1 HybridCNN-ViT on official WM-811K test.

Reports micro-acc / Macro-F1 with and without eval log-pi prior, plus
per-class precision / recall / F1 / support — so we know whether the gap
to the paper is 'none' on the long tail or rare defects.

Example:
  python papers/semiwafernet/scripts/diagnose_stage1.py \\
    --checkpoint checkpoints/semiwafernet/ssl_stage1.pt --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.engine.config import EngineConfig
from papers.semiwafernet.data_utils.wafer_dataset import (
    WM811K_CLASSES,
    WaferWM811KDataset,
)
from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.utils.checkpoint import normalize_semiwafernet_state_dict


def _collate(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return images, labels

def _load_model(ckpt_path: Path, config: EngineConfig, device: torch.device) -> SemiWaferNet:
    model = SemiWaferNet.from_config(config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = normalize_semiwafernet_state_dict(raw)
    clean = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "base_model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        if "log_prior" in nk:
            continue
        clean[nk] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"  warn missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  warn unexpected keys: {unexpected[:8]}")
    return model.to(device).eval()


def _natural_log_prior(data_root: str, num_classes: int) -> torch.Tensor:
    ds = WaferWM811KDataset(
        data_root, 32, train=False, hybrid_sampling=False, split="training"
    )
    counts = np.bincount([y for _, y in ds._samples], minlength=num_classes).astype(
        np.float64
    )
    prior = counts / max(counts.sum(), 1.0)
    print(f"  natural train prior: {prior.round(4).tolist()}")
    return torch.log(torch.tensor(prior, dtype=torch.float32).clamp(min=1e-12))


@torch.no_grad()
def _collect_logits(
    model: SemiWaferNet, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_list: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in loader:
        out = model(x.to(device))["classification"].cpu()
        logits_list.append(out)
        ys.append(y.view(-1).long())
    return torch.cat(logits_list, dim=0), torch.cat(ys, dim=0)


def _per_class_report(
    pred: torch.Tensor, y: torch.Tensor, num_classes: int
) -> tuple[float, float, list[dict]]:
    rows: list[dict] = []
    f1s: list[float] = []
    for c in range(num_classes):
        tp = int(((pred == c) & (y == c)).sum().item())
        fp = int(((pred == c) & (y != c)).sum().item())
        fn = int(((pred != c) & (y == c)).sum().item())
        support = int((y == c).sum().item())
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
        rows.append(
            {
                "class": WM811K_CLASSES[c],
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
                "pred_count": int((pred == c).sum().item()),
            }
        )
    acc = float((pred == y).float().mean().item())
    macro_f1 = float(np.mean(f1s))
    return acc, macro_f1, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/semiwafernet/ssl_stage1.pt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="papers/semiwafernet/configs/config.yaml",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-test",
        type=int,
        default=0,
        help="If >0, subsample this many test images (debug). 0 = full test.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = EngineConfig.from_yaml(args.config)
    data_root = config.get("data.data_root", "datasets/wm811k")
    image_size = int(config.get("data.image_size", 32))
    num_classes = int(config.get("model.num_classes", 9))

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    model = _load_model(Path(args.checkpoint), config, device)
    log_prior = _natural_log_prior(data_root, num_classes)

    test_ds = WaferWM811KDataset(
        data_root, image_size, train=False, hybrid_sampling=False, split="test"
    )
    if args.max_test and args.max_test < len(test_ds):
        rng = np.random.RandomState(0)
        idx = rng.choice(len(test_ds), args.max_test, replace=False)
        from torch.utils.data import Subset

        test_ds = Subset(test_ds, idx.tolist())
        print(f"Test subsample: {len(test_ds)}")
    else:
        print(f"Test full: {len(test_ds)}")

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    print("Running inference...")
    logits, y = _collect_logits(model, loader, device)

    for scale, name in [(0.0, "prior=0"), (1.0, "prior=1")]:
        pred = (logits + scale * log_prior).argmax(dim=1)
        acc, macro_f1, rows = _per_class_report(pred, y, num_classes)
        print(f"\n=== {name} ===")
        print(f"micro-acc={acc:.4f}  Macro-F1={macro_f1:.4f}")
        print(f"{'class':12s} {'P':>7s} {'R':>7s} {'F1':>7s} {'supp':>7s} {'pred':>7s}")
        for r in rows:
            print(
                f"{r['class']:12s} {r['precision']:7.3f} {r['recall']:7.3f} "
                f"{r['f1']:7.3f} {r['support']:7d} {r['pred_count']:7d}"
            )


if __name__ == "__main__":
    main()
