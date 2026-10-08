"""Check holdout integrity and exact confusion metrics in reduced diagnostics."""

import argparse
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.engine.config import EngineConfig
from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.scripts.diagnose_mps import report_predictions, split_subset
from papers.semiwafernet.scripts import diagnose_mps as diagnostic
from papers.semiwafernet.training.stage_manager import StageManager
from papers.semiwafernet.training.trainer import Trainer


def test_balanced_subset_is_disjoint_and_preserves_rare_class():
    rows = [(f"c{c}_{i}.png", c) for c in range(9) for i in range(7 if c == 6 else 100)]
    train, val = split_subset(rows, 20, 10, 42)
    assert not {f for f, _ in train} & {f for f, _ in val}
    assert len([y for _, y in train if y == 6]) == 6
    assert len([y for _, y in val if y == 6]) == 1
    assert (train, val) == split_subset(rows, 20, 10, 42)
    assert (train, val) != split_subset(rows, 20, 10, 137)


def test_subset_rejects_missing_class():
    with pytest.raises(ValueError, match="fewer than three"):
        split_subset([("a.png", 0)] * 3, 20, 10, 42)


def test_report_counts_all_nine_classes():
    logits = torch.eye(9) * 10
    targets = torch.arange(9)
    report = report_predictions(logits, targets)
    assert report["accuracy"] == 1
    assert report["macro_f1"] == pytest.approx(1)
    assert report["confusion_matrix"] == torch.eye(9, dtype=torch.long).tolist()
    assert report["balanced_accuracy"] == pytest.approx(1)
    assert report["cohen_kappa"] == pytest.approx(1)


def saved_split(tmp_path):
    inventory = [(f"c{label}_{index}.png", label) for label in range(9) for index in range(8)]
    source = {
        "seed": 42,
        "samples": {
            "train": [(f"c{label}_{index}.png", label) for label in range(9) for index in (0, 1)],
            "validation": [(f"c{label}_2.png", label) for label in range(9)],
        },
        "none_stress_extra_samples": [("c0_3.png", 0)],
        "diverse_none_extra_training_samples": [("c0_4.png", 0)],
    }
    path = tmp_path / "source_report.json"
    path.write_text(json.dumps(source))
    return inventory, source, path


def test_saved_split_roundtrip_preserves_every_role_and_data_seed(tmp_path):
    inventory, source, path = saved_split(tmp_path)
    reused = diagnostic.load_diagnostic_split(path, inventory)
    assert reused["train"] == source["samples"]["train"]
    assert reused["validation"] == source["samples"]["validation"]
    assert reused["none_stress"] == source["none_stress_extra_samples"]
    assert reused["diverse_none"] == source["diverse_none_extra_training_samples"]
    assert reused["data_seed"] == 42


@pytest.mark.parametrize(
    "change", ["test_row", "overlap", "label", "inventory_digest", "subset_digest"]
)
def test_reused_splits_reject_leakage_or_changed_source(tmp_path, change):
    inventory, source, path = saved_split(tmp_path)
    if change == "test_row":
        source["samples"]["validation"][0] = ("official_test.png", 0)
    elif change == "overlap":
        source["none_stress_extra_samples"] = [source["samples"]["validation"][0]]
    elif change == "label":
        source["samples"]["validation"][0] = ("c0_2.png", 8)
    elif change == "inventory_digest":
        source["training_inventory_digest"] = "changed"
    elif change == "subset_digest":
        source["subset_digest"] = "changed"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        diagnostic.load_diagnostic_split(path, inventory)


def test_variant_rates_keep_paper_default_and_validate_explicit_overrides():
    variants = ["paper_smote", "mid_smote", "fast_smote"]
    assert diagnostic.variant_learning_rates(variants, [], 5e-5) == {
        "paper_smote": 5e-5,
        "mid_smote": 2e-4,
        "fast_smote": 5e-4,
    }
    assert (
        diagnostic.variant_learning_rates(variants, ["mid_smote=0.00015"], 5e-5)["mid_smote"]
        == 0.00015
    )
    for override in ("fast_real=0.001", "mid_smote=nan", "mid_smote=0", "oops"):
        with pytest.raises(ValueError):
            diagnostic.variant_learning_rates(variants, [override], 5e-5)


def test_gate_counts_use_published_strict_boundaries_and_true_classes():
    collected = {
        "conf": torch.tensor([0.96, 0.99, 0.98, 0.99, 0.99]),
        "ent": torch.tensor([0.01, 0.01, 0.08, 0.05, 0.01]),
        "mi": torch.tensor([0.01, 0.01, 0.01, 0.12, 0.005]),
        "pred": torch.tensor([0, 1, 1, 1, 2]),
        "y": torch.tensor([0, 0, 1, 1, 2]),
    }
    accepted = torch.tensor([True, False, False, False, True])
    report = diagnostic.gate_diagnostics(
        collected, accepted, torch.tensor([0.96, 1.0, 0.98, 0.98, 0.98]), 0.08, 0.12
    )
    assert report["gate_pass_counts"] == {
        "confidence": 4,
        "entropy": 4,
        "mutual_information": 4,
        "confidence_and_entropy": 3,
        "all_gates": 2,
    }
    assert report["per_true_class"][0]["accepted_fraction"] == 0.5
    assert report["per_true_class"][0]["accepted_accuracy"] == 1
    assert report["per_true_class"][1]["accepted_accuracy"] is None
    assert report["threshold_at_one_fraction"] == pytest.approx(0.2)


class TinyDiagnosticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3 * 4 * 4, 9)

    def forward(self, images):
        return {"classification": self.head(images.flatten(1))}


def tiny_variant(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(
        diagnostic.SemiWaferNet, "from_config", lambda config: TinyDiagnosticModel()
    )
    monkeypatch.setattr(diagnostic, "pseudo_diagnostic", lambda *args: {"accepted_count": 0})
    args = argparse.Namespace(
        seed=42,
        data_seed=42,
        device="cpu",
        output=str(tmp_path),
        batch_size=9,
        steps=3,
        eval_every=1,
        max_seconds=100,
        **overrides,
    )
    images, labels = torch.randn(18, 3, 4, 4), torch.arange(9).repeat(2)
    config = EngineConfig.from_yaml("papers/semiwafernet/configs/config.yaml")
    snapshots = []
    result = diagnostic.run_variant(
        "mid_smote",
        2e-4,
        images,
        labels,
        images,
        labels,
        images,
        labels,
        None,
        None,
        args,
        config,
        snapshots.append,
    )
    return result, snapshots


def test_best_checkpoint_and_progress_survive_mid_variant_interrupt(tmp_path, monkeypatch):
    original_augment, calls = diagnostic.geometric_augment, 0

    def interrupt_after_one_validated_step(image, rng):
        nonlocal calls
        calls += 1
        if calls > 9:
            raise KeyboardInterrupt
        return original_augment(image, rng)

    monkeypatch.setattr(diagnostic, "geometric_augment", interrupt_after_one_validated_step)
    result, snapshots = tiny_variant(tmp_path, monkeypatch)
    assert result["status"] == "interrupted" and result["steps"] == result["best_step"] == 1
    saved = torch.load(tmp_path / "mid_smote.pt", weights_only=False)
    assert saved["best_step"] == 1
    assert saved["best_validation"] == result["best_validation"]
    persisted = json.loads((tmp_path / "mid_smote.progress.json").read_text())
    assert persisted["status"] == "interrupted" and persisted["best_step"] == 1
    assert any(
        snapshot["status"] == "running" and snapshot["best_step"] == 1 for snapshot in snapshots
    )


def test_interrupt_during_checkpoint_save_does_not_claim_missing_best(tmp_path, monkeypatch):
    def interrupted_save(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(diagnostic.torch, "save", interrupted_save)
    result, _ = tiny_variant(tmp_path, monkeypatch)
    assert result["status"] == "interrupted"
    assert result["best_step"] is None and not result["best_checkpoint_available"]
    assert not (tmp_path / "mid_smote.pt").exists()


def test_time_budget_still_validates_and_persists_last_completed_step(tmp_path, monkeypatch):
    monkeypatch.setattr(
        diagnostic.SemiWaferNet, "from_config", lambda config: TinyDiagnosticModel()
    )
    monkeypatch.setattr(diagnostic, "pseudo_diagnostic", lambda *args: {"accepted_count": 0})
    args = argparse.Namespace(
        seed=42,
        data_seed=42,
        device="cpu",
        output=str(tmp_path),
        batch_size=9,
        steps=5,
        eval_every=5,
        max_seconds=1e-9,
    )
    images, labels = torch.randn(18, 3, 4, 4), torch.arange(9).repeat(2)
    result = diagnostic.run_variant(
        "paper_smote",
        5e-5,
        images,
        labels,
        images,
        labels,
        images,
        labels,
        None,
        None,
        args,
        EngineConfig.from_yaml("papers/semiwafernet/configs/config.yaml"),
    )
    assert result["status"] == "complete" and result["stop_reason"] == "time_budget_reached"
    assert result["steps"] == result["best_step"] == 1
    assert result["history"][-1]["step"] == 1
    assert (tmp_path / "paper_smote.pt").is_file()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS unavailable")
def test_full_model_ssl_three_stages_train_on_mps():
    """Real SSL trainer and lazy accepted union; permissive gates are smoke-only."""
    torch.manual_seed(42)
    config = EngineConfig.from_yaml("papers/semiwafernet/configs/config.yaml")
    student = SemiWaferNet.from_config(config).to("mps")
    manager = StageManager(
        student,
        mc_passes=2,
        base_threshold=0,
        alpha=0,
        beta=0,
        entropy_threshold=100,
        mi_threshold=100,
    )
    images = nn.functional.one_hot(torch.randint(0, 3, (9, 32, 32)), 3).permute(0, 3, 1, 2).float()
    labeled_source = [{"image": images[c], "label": c} for c in range(9)]

    class LoaderAdapter:
        def __init__(self):
            self.loader = DataLoader(labeled_source, batch_size=9)

        def __len__(self):
            return len(self.loader)

        def __iter__(self):
            for batch in self.loader:
                yield batch["image"], {"classification": batch["label"]}

    def supervised_loss(outputs, targets):
        return nn.functional.cross_entropy(outputs["classification"], targets["classification"])

    trainer = Trainer(
        student,
        manager,
        optimizer=torch.optim.AdamW(student.parameters(), lr=5e-5),
        supervised_loss_fn=supervised_loss,
        device=torch.device("mps"),
        batch_size=9,
        verbose=False,
    )
    before = student.classifier["head"].weight.detach().cpu().clone()
    labeled = LoaderAdapter()
    unlabeled = DataLoader(TensorDataset(images), batch_size=9, shuffle=False)
    metrics = [trainer.train_stage1(labeled, num_epochs=1)]
    metrics.append(trainer.train_stage2(labeled, unlabeled, num_epochs=1))
    metrics.append(trainer.train_stage3(labeled, unlabeled, num_epochs=1))
    assert trainer.current_epoch == 3
    assert manager.current_stage == 3
    assert all(torch.isfinite(torch.tensor(stage["loss"])) for stage in metrics)
    assert metrics[1]["pseudo_accept_rate"] == metrics[2]["pseudo_accept_rate"] == 100
    assert student.classifier["head"].weight.device.type == "mps"
    assert not torch.equal(before, student.classifier["head"].weight.detach().cpu())
    print(
        {
            "device": "mps",
            "stages": metrics,
            "note": "Permissive synthetic gates test mechanics only",
        }
    )
