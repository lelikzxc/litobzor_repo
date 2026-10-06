"""Regression tests for paper fidelity and executable training paths (CPU)."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.engine.config import EngineConfig
from common.training import CheckpointManager
from common.training import Trainer as CommonTrainer
from papers.reproduction import load_holdout, save_protocol
from papers.semiwafernet.data_utils.wafer_dataset import SMOTEDataset
from papers.semiwafernet.models.semiwafernet import SemiWaferNet
from papers.semiwafernet.training.adaptive_threshold import AdaptiveThreshold
from papers.semiwafernet.training.segmentation import (
    DiceFocalLoss,
    SegmentationLoss,
    SegmentationWrapper,
    metric_functions,
)
from papers.semiwafernet.training.stage_manager import IndexedPseudoDataset, StageManager
from papers.semiwafernet.training.trainer import Trainer
from papers.vmamba.data_utils.protocol import balanced_indices, stratified_split
from papers.vmamba.kernels import csm_triton, csms6s
from papers.vmamba.models.vmamba import FCSVMamba
from papers.vmamba.modules.fcs_modules import CrossLayerChannelAttention, SaliencySuppression
from papers.wafer_encoding import decode_die_map


@pytest.mark.parametrize("encoded", [[0, 1, 2], [0, 127, 254], [0, 128, 255]])
def test_die_encoding(encoded):
    assert decode_die_map(np.array([encoded], dtype=np.uint8)).tolist() == [[0, 1, 2]]
    # Missing 'normal' pixels must not turn defective die into normal die.
    assert decode_die_map(np.array([[encoded[0], encoded[2]]], dtype=np.uint8)).tolist() == [[0, 2]]


def test_invalid_encoding_rejected():
    with pytest.raises(ValueError, match="encoding"):
        decode_die_map(np.array([[0, 73, 255]], dtype=np.uint8))


def test_clca_has_one_residual_and_correct_stage_positions():
    clca = CrossLayerChannelAttention(8, 8, num_heads=2)
    with torch.no_grad():
        clca.proj.weight.zero_()
        clca.proj.bias.zero_()
    target = torch.randn(2, 8, 3, 3)
    assert torch.equal(clca(torch.randn(2, 8, 6, 6), target), torch.zeros_like(target))
    model = FCSVMamba(embed_dim=8, depths=[2] * 4, clca_num_heads=2)
    assert [[b.clca_enabled for b in stage] for stage in model.stages] == [
        [False, False],
        [False, True],
        [False, True],
        [False, True],
    ]
    seen = []
    for stage in model.stages[1:]:
        stage[-1].clca.register_forward_pre_hook(
            lambda m, xs: seen.append((xs[0].shape[-1], xs[1].shape[-1]))
        )
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 9)
    assert seen == [(16, 8), (8, 4), (4, 2)]


def test_sfs_straight_through_and_zero_ratio():
    x = torch.arange(16.0).reshape(1, 1, 4, 4).requires_grad_()
    out = SaliencySuppression(suppression_ratio=0.25, suppression_radius=0)(x)
    assert torch.allclose(out.flatten()[-4:], x.flatten()[-4:] * 0.1)
    out.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    assert torch.equal(SaliencySuppression(suppression_ratio=0)(x), x)


@pytest.mark.parametrize("scans", [0, 1, 2])
def test_cross_scan_merge_gradient_with_triton_flag(monkeypatch, scans):
    monkeypatch.setattr(csm_triton, "WITH_TRITON", True)
    # These names must remain differentiable even on a Triton-enabled machine.
    x = torch.randn(2, 3, 4, 6, dtype=torch.double, requires_grad=True)
    scan = csm_triton.CrossScanTritonF.apply(x, True, True, False, scans)
    merged = csm_triton.CrossMergeTritonF.apply(
        scan.reshape(2, 4, 3, 4, 6), True, True, False, scans
    )
    assert torch.allclose(merged, x.flatten(2) * 4)
    merged.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 4))


def test_cpu_scan_with_cuda_extension_flag(monkeypatch):
    monkeypatch.setattr(csms6s, "WITH_SELECTIVESCAN_OFLEX", True)
    u = torch.randn(1, 4, 5, requires_grad=True)
    delta = torch.randn_like(u, requires_grad=True)
    a = -torch.ones(4, 2, requires_grad=True)
    b = torch.randn(1, 2, 2, 5, requires_grad=True)
    c = torch.randn_like(b, requires_grad=True)
    out = csms6s.selective_scan_fn(u, delta, a, b, c, backend="oflex")
    out.square().mean().backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(delta.grad).all()


def test_adaptive_threshold_matches_population_variance():
    gate = AdaptiveThreshold(3, base_threshold=0.94, alpha=0.08, beta=0.02)
    conf = torch.tensor([0.96, 0.98, 0.99])
    labels = torch.tensor([0, 0, 1])
    gate.update_statistics(conf[:1], labels[:1])
    gate.update_statistics(conf[1:], labels[1:])
    assert gate.class_std[1] == 0
    assert torch.allclose(gate.class_std[0], torch.tensor(0.01), atol=1e-6)
    entropy = torch.tensor([0.04, 0.05, 0.01])
    expected = 0.94 + 0.08 * torch.tensor([0.01 / 0.97, 0.01 / 0.97, 0]) + 0.02 * (1 - entropy)
    assert torch.allclose(gate.compute_threshold(labels, entropy), expected)


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        logits = self.head(x.flatten(1))
        return {"classification": logits, "segmentation": x[:, :1] * 0}


def test_pseudo_set_no_none_cap_and_no_image_copy():
    model = ToyClassifier()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([12.0, -12.0]))
    manager = StageManager(model, num_classes=2, mc_passes=2)
    source = TensorDataset(torch.rand(300, 1, 2, 2))
    pseudo, stats = manager.build_pseudo_dataset(
        DataLoader(source, batch_size=50), torch.device("cpu"), verbose=False
    )
    assert isinstance(pseudo, IndexedPseudoDataset)
    assert len(pseudo) == 300 and stats["accept_rate"] == 100
    assert pseudo.source is source
    assert not hasattr(pseudo, "tensors")


def test_ssl_union_preserves_dynamic_labeled_dataset():
    source = TensorDataset(torch.randn(12, 1, 2, 2), torch.zeros(12, dtype=torch.long))
    model = ToyClassifier()
    trainer = Trainer(model, StageManager(model, num_classes=2), batch_size=4, verbose=False)
    union = trainer._build_union_loader(
        DataLoader(source, batch_size=4),
        TensorDataset(torch.rand(3, 1, 2, 2), torch.ones(3, dtype=torch.long)),
    )
    assert len(union.dataset) == 15
    assert union.dataset.datasets[0].base is source


def test_smote_retains_interpolation(tmp_path):
    (tmp_path / "images").mkdir()
    samples = []
    for i in range(7):
        arr = np.zeros((4, 4), dtype=np.uint8)
        arr[i % 4, :] = 2
        name = f"{i}.png"
        Image.fromarray(arr).save(tmp_path / "images" / name)
        samples.append((name, int(i >= 5)))
    ds = SMOTEDataset(tmp_path, samples, image_size=4, num_classes=2, method="smote")
    assert len(ds) == 10
    assert ((ds._X > 0) & (ds._X < 1)).any()
    assert np.allclose(ds._X.sum(axis=1), 1)


def test_binary_loss_and_auxiliary_gradients():
    model = SemiWaferNet(
        mode="segmentation",
        in_channels=1,
        base_channels=4,
        seg_embed_dim=16,
        num_heads=2,
        num_layers=1,
    )
    wrapper = SegmentationWrapper(model)
    target = torch.zeros(2, 64, 64, dtype=torch.long)
    target[:, 10:30, 10:30] = 1
    output = wrapper(torch.rand(2, 1, 64, 64))
    loss = SegmentationLoss(wrapper)(output, target)
    loss.backward()
    assert torch.isfinite(loss)
    for head in (model.seg_model.head, model.seg_model.aux_head1, model.seg_model.aux_head2):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
    perfect = torch.where(target.unsqueeze(1).bool(), 20.0, -20.0)
    assert DiceFocalLoss()(perfect, target) < DiceFocalLoss()(-perfect, target)
    assert metric_functions()["iou"](perfect, target) == 1


def test_checkpoint_uses_configured_f1_not_accuracy(tmp_path):
    model = nn.Linear(2, 2)
    manager = CheckpointManager(tmp_path, metric_name="val_f1", mode="max")
    trainer = CommonTrainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        nn.CrossEntropyLoss(),
        checkpoint_manager=manager,
        verbose=False,
    )
    results = iter(
        [{"loss": 1.0, "accuracy": 0.9, "f1": 0.3}, {"loss": 0.8, "accuracy": 0.8, "f1": 0.6}]
    )
    trainer.train_one_epoch = lambda loader: {"loss": 1.0}
    trainer.validate = lambda loader: next(results)
    trainer.fit([], [], epochs=2)
    state = torch.load(manager.best_path, weights_only=False)
    assert state["epoch"] == 2 and state["metric"] == 0.6


def test_split_persistence_and_dataset_order_check(tmp_path):
    labels = list(range(9)) * 20
    pool = balanced_indices(labels, per_class=5, seed=7)
    train, val = stratified_split(pool, labels, seed=7)
    assert len(pool) == 45 and len(val) == 9 and not set(train) & set(val)
    samples = [(f"{i}.png", y) for i, y in enumerate(labels)]
    save_protocol(tmp_path, EngineConfig({"model": {}}), samples, {"holdout": val}, 7)
    indices, _ = load_holdout(tmp_path / "best.pt", samples)
    assert indices == val
    with pytest.raises(ValueError, match="Dataset"):
        load_holdout(tmp_path / "best.pt", samples[::-1])


def test_vmamba_cpu_optimizer_step_finite():
    torch.manual_seed(3)
    model = FCSVMamba(embed_dim=16, depths=[1] * 4, clca_num_heads=2, d_state=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = torch.randn(2, 3, 32, 32)
    loss = nn.functional.cross_entropy(model(x), torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend parity requires GPU")
def test_cuda_scan_parity_and_backward():
    assert (
        csms6s.WITH_SELECTIVESCAN_OFLEX
        or csms6s.WITH_SELECTIVESCAN_CORE
        or csms6s.WITH_SELECTIVESCAN_MAMBA
    ), "Install compiled selective-scan kernels before GPU training"
    torch.manual_seed(11)
    tensors = [
        torch.randn(2, 8, 12, device="cuda"),
        torch.randn(2, 8, 12, device="cuda"),
        -torch.rand(8, 3, device="cuda"),
        torch.randn(2, 4, 3, 12, device="cuda"),
        torch.randn(2, 4, 3, 12, device="cuda"),
    ]
    actual = [t.detach().requires_grad_() for t in tensors]
    reference = [t.detach().requires_grad_() for t in tensors]
    y = csms6s.selective_scan_fn(*actual)
    yr = csms6s.selective_scan_fn(*reference, backend="torch")
    assert torch.allclose(y, yr, atol=1e-4, rtol=1e-4)
    y.square().mean().backward()
    yr.square().mean().backward()
    for a, r in zip(actual, reference):
        assert torch.allclose(a.grad, r.grad, atol=1e-4, rtol=1e-3)
    model = FCSVMamba(embed_dim=16, depths=[1] * 4, clca_num_heads=2, d_state=2).cuda()
    loss = nn.functional.cross_entropy(
        model(torch.rand(2, 3, 32, 32, device="cuda")), torch.tensor([0, 1], device="cuda")
    )
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_author_archive_uses_rgb_and_folder_labels(tmp_path):
    import io
    import zipfile

    from papers.vmamba.data_utils.author_dataset import FOLDER_LABELS, AuthorWM811KDataset

    path = tmp_path / "maps.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for folder in FOLDER_LABELS:
            stream = io.BytesIO()
            Image.new("RGB", (32, 32), (20, 120, 230)).save(stream, format="JPEG")
            archive.writestr(f"WM811k_Dataset/{folder}/map.jpg", stream.getvalue())
    dataset = AuthorWM811KDataset(path, image_size=64)
    assert len(dataset) == 9
    assert {dataset[i]["label"] for i in range(9)} == set(range(9))
    assert dataset[0]["image"].shape == (3, 64, 64)
    assert torch.isfinite(dataset[0]["image"]).all()
    config = EngineConfig.from_yaml("papers/vmamba/configs/config.yaml")
    assert config.get("data.source") == "author_archive"
    assert config.get("data.per_class") == 0
    assert all(config.get(f"model.{name}.enabled") for name in ("fa", "sfs", "clca"))


def test_segmentation_preparation_masks_raw_state_two(tmp_path):
    import json

    from papers.semiwafernet.prepare_segmentation import prepare

    root = tmp_path / "raw"
    (root / "images").mkdir(parents=True)
    rows = ["filename,failureType"]
    for i in range(42):
        name = f"{i}.png"
        arr = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        Image.fromarray(arr).save(root / "images" / name)
        label = "Center" if i < 20 else "Donut" if i < 40 else "none" if i == 40 else "Random"
        rows.append(f"{name},{label}")
    (root / "labels.csv").write_text("\n".join(rows))
    output = tmp_path / "seg"
    prepare(root, output)
    manifest = json.loads((output / "manifest.json").read_text())
    sets = [
        set(name for name, _ in manifest["splits"][split]) for split in ("train", "val", "test")
    ]
    assert sum(map(len, sets)) == 40
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    mask = np.asarray(Image.open(next((output / "test" / "masks").glob("*.png"))))
    assert mask.tolist() == [[0, 0, 255], [255, 0, 0]]


def test_ssm_adamw_preserves_state_parameter_flags(tmp_path):
    from common.engine.engine import Engine
    from papers.vmamba.train import configure_ssm_optimizer

    model = FCSVMamba(embed_dim=8, depths=[1] * 4, clca_num_heads=2, d_state=2)
    config = EngineConfig(
        {
            "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.05},
            "scheduler": {"name": "cosine", "kwargs": {"T_max": 50}},
        }
    )
    engine = Engine(model, config, device="cpu")
    configure_ssm_optimizer(engine, config)
    assert engine.trainer.optimizer is engine.optimizer
    groups = engine.optimizer.param_groups
    excluded = {id(p) for p in groups[1]["params"]}
    assert excluded == {id(p) for p in model.parameters() if getattr(p, "_no_weight_decay", False)}
    assert groups[1]["weight_decay"] == 0 and groups[0]["weight_decay"] == 0.05
    engine.optimizer.step()
    engine.scheduler.step()
    assert groups[0]["lr"] == groups[1]["lr"] < 0.001
