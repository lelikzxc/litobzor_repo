"""Small FP32 trainer with finite-gradient checks and sample-weighted metrics."""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from tqdm import tqdm

from .data_utils import CLASSES
from .optim import KerasAdam


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_state(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def metrics_from_predictions(targets, predictions, loss):
    report = classification_report(
        targets,
        predictions,
        labels=list(range(8)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": loss,
        "accuracy": float(np.mean(np.array(targets) == predictions)),
        "f1": report["macro avg"]["f1-score"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": report,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(8))).tolist(),
    }


class PolarTrainer:
    def __init__(self, model, device, learning_rate=0.01):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = KerasAdam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def epoch(self, loader, training=False, desc="Val"):
        self.model.train(training)
        total_loss, count = 0.0, 0
        targets_all, predictions_all = [], []
        iterator = tqdm(
            loader,
            desc=desc,
            leave=False,
            mininterval=0.5,
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )
        with torch.set_grad_enabled(training):
            for batch in iterator:
                inputs = batch["inputs"].to(self.device, non_blocking=self.device.type == "cuda")
                targets = batch["targets"].to(self.device, non_blocking=self.device.type == "cuda")
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite {desc} loss; training stopped")
                if training:
                    loss.backward()
                    # Fail loudly; don't silently sanitise invalid gradients.
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all().item()
                        for p in self.model.parameters()
                    ):
                        raise FloatingPointError("Nonfinite gradient; training stopped")
                    self.optimizer.step()
                batch_size = len(targets)
                total_loss += loss.item() * batch_size
                count += batch_size
                targets_all.extend(targets.detach().cpu().tolist())
                predictions_all.extend(logits.detach().argmax(1).cpu().tolist())
                iterator.set_postfix(loss=f"{total_loss / count:.4f}")
        if not count:
            raise ValueError("Empty training/evaluation loader")
        return metrics_from_predictions(targets_all, predictions_all, total_loss / count)

    def fit(
        self, train_loader, val_loader, epochs, run_dir, config, manifest, generator, resume=None
    ):
        run_dir = Path(run_dir)
        best, start = float("inf"), 0
        best_score = -float("inf")
        selection = config["training"].get("selection_metric", "loss")
        if selection not in {"loss", "f1"}:
            raise ValueError("selection_metric must be loss or f1")
        history = []
        if resume:
            checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
            if checkpoint["manifest"] != manifest:
                raise ValueError("Checkpoint/split manifest mismatch")
            previous = checkpoint["config"]["training"]
            for key, default in (("balanced_sampling", False), ("selection_metric", "loss")):
                if previous.get(key, default) != config["training"].get(key, default):
                    raise ValueError(f"Resume {key} differs from saved run")
            if any(
                previous[k] != config["training"][k]
                for k in ("batch_size", "learning_rate", "seed")
            ):
                raise ValueError("Resume batch_size/lr/seed differ from saved run")
            if checkpoint.get("optimizer_convention") != "keras_2.15_epsilon_hat":
                raise ValueError("Checkpoint uses a different Adam update convention")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            restore_rng(checkpoint["rng"], generator)
            start, best = checkpoint["epoch"], checkpoint["best_val_loss"]
            best_score = checkpoint.get("best_val_score", -best)
            history = checkpoint["history"]
        if epochs <= start:
            raise ValueError("epochs must exceed the checkpoint's completed epoch")
        for epoch in range(start + 1, epochs + 1):
            started = time.monotonic()
            train = self.epoch(train_loader, training=True, desc=f"Train {epoch}/{epochs}")
            val = self.epoch(val_loader) if val_loader is not None else None
            score = (val["f1"] if selection == "f1" else -val["loss"]) if val else None
            improved = score is not None and score > best_score
            if score is not None:
                best_score = max(best_score, score)
            if val is not None:
                best = min(best, val["loss"])
            row = {
                "epoch": epoch,
                "train_loss": train["loss"],
                "seconds": time.monotonic() - started,
            }
            if val is not None:
                row.update(val_loss=val["loss"], val_accuracy=val["accuracy"], val_f1=val["f1"])
            history.append(row)
            summary = " | ".join(f"{k}={v:.5f}" for k, v in row.items() if k != "epoch")
            print(f"Epoch {epoch}/{epochs} | {summary}", flush=True)
            with (run_dir / "train.log").open("a", encoding="utf-8") as stream:
                stream.write(f"Epoch {epoch}/{epochs} | {summary}\n")
            payload = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best,
                "best_val_score": best_score,
                "config": config,
                "manifest": manifest,
                "history": history,
                "rng": rng_state(generator),
                "optimizer_convention": "keras_2.15_epsilon_hat",
                "checkpoint_selection": f"validation_{selection}"
                if val is not None
                else "final_epoch",
            }
            atomic_save(payload, run_dir / "last.pt")
            if improved:
                atomic_save(payload, run_dir / "best.pt")
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_loader is None:
            atomic_save(payload, run_dir / "final.pt")
        return history
