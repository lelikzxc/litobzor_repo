"""One-batch device preflight on real wafers; no full training/checkpoints."""

# ruff: noqa: E402 -- Standalone entry point bootstraps the repository path.

import argparse
import json
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.training.utils import resolve_device
from common.utils.seed import set_seed
from papers.polar_cnn.data_utils import PolarDataset
from papers.polar_cnn.data_utils.preprocessing import preprocessing_options
from papers.polar_cnn.models import PolarCNN
from papers.polar_cnn.train import inventory_from_config, load_config, project_path
from papers.polar_cnn.trainer import PolarTrainer, atomic_save


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--config", default="papers/polar_cnn/configs/config.yaml")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 8:
        raise ValueError("batch-size must be >=8 to verify all defect classes")
    device = resolve_device(args.device)
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    inventory = inventory_from_config(config)
    buckets = [
        [i for i, (_, y) in enumerate(inventory.samples) if y == label] for label in range(8)
    ]
    if any(not bucket for bucket in buckets):
        raise ValueError("Preflight needs all eight classes")
    records = [[buckets[i % 8][(i // 8) % len(buckets[i % 8])], 0] for i in range(args.batch_size)]
    dataset = PolarDataset(
        inventory,
        records,
        config["data"]["denoise_threshold"],
        **preprocessing_options(config["data"]),
    )
    reference = PolarCNN(input_adapter=config["model"]["input_adapter"])
    model = deepcopy(reference).to(device)
    x, y = dataset.inputs.to(device), dataset.targets.to(device)
    expected = reference(dataset.inputs)
    expected_loss = torch.nn.functional.cross_entropy(expected, dataset.targets)
    expected_loss.backward()
    actual = model(x)
    loss = torch.nn.functional.cross_entropy(actual, y)
    loss.backward()
    torch.testing.assert_close(actual.cpu(), expected.detach(), rtol=1e-3, atol=1e-3)
    differences = []
    gradient_details = {}
    for (name, parameter), (_, cpu_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"Invalid gradient: {name}")
        difference = (parameter.grad.cpu() - cpu_parameter.grad).abs().max().item()
        differences.append(difference)
        gradient_details[name] = {
            "max_absolute_difference": difference,
            "cpu_norm": cpu_parameter.grad.norm().item(),
            "device_norm": parameter.grad.norm().item(),
        }
    # Sparse inputs can produce tiny positive roundoff on CPU where MPS produces
    # exactly zero. ReLU's derivative at that boundary changes bias gradients.
    # Require forward equivalence and finite gradients/updates; record all bias
    # differences rather than claiming bitwise backend equivalence.
    for (name, parameter), (_, cpu_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if name.endswith("weight"):
            torch.testing.assert_close(
                parameter.grad.cpu(), cpu_parameter.grad, rtol=2e-3, atol=2e-3
            )
    trainer = PolarTrainer(model, device, config["training"]["learning_rate"])
    before = model.features[0].weight.detach().clone()
    started = time.monotonic()
    # Fresh forward/backward through the actual training path, exactly one batch.
    result = trainer.epoch(
        [{"inputs": dataset.inputs, "targets": dataset.targets}], training=True, desc="Preflight"
    )
    if device == "mps":
        torch.mps.synchronize()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.monotonic() - started
    if torch.equal(before, model.features[0].weight):
        raise RuntimeError("Optimizer did not update the model")
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise FloatingPointError("Nonfinite parameters after Adam step")
    with tempfile.TemporaryDirectory(prefix="polar_preflight_") as temporary:
        path = Path(temporary) / "roundtrip.pt"
        atomic_save(
            {"model": model.state_dict(), "optimizer": trainer.optimizer.state_dict()}, path
        )
        saved = torch.load(path, map_location="cpu", weights_only=False)
        restored = PolarTrainer(PolarCNN(input_adapter=config["model"]["input_adapter"]), device)
        restored.model.load_state_dict(saved["model"])
        restored.optimizer.load_state_dict(saved["optimizer"])
        with torch.no_grad():
            torch.testing.assert_close(restored.model(x), model(x), rtol=0, atol=0)
    report = {
        "device": device,
        "torch": torch.__version__,
        "original_wafers": len(inventory.samples),
        "dataset_sha256": inventory.identity,
        "batch_size": args.batch_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_adapter": config["model"]["input_adapter"],
        "preprocessing": {
            "denoise_threshold": config["data"]["denoise_threshold"],
            **preprocessing_options(config["data"]),
        },
        "initial_loss": loss.item(),
        "max_logit_difference_from_cpu": (actual.cpu() - expected.detach()).abs().max().item(),
        "max_gradient_difference_from_cpu": max(differences),
        "gradient_details": gradient_details,
        "gradient_note": "Sparse ReLU regions can have different near-zero convolution roundoff across CPU/MPS. Bias-gradient differences are recorded; finite gradients and weight-gradient parity are checked. This is not a claim of exact backend equivalence.",
        "one_training_batch_seconds": seconds,
        "finite_optimizer_step": True,
        "optimizer_convention": "keras_2.15_epsilon_hat",
        "checkpoint_roundtrip": True,
        "full_training_started": False,
        "note": "One-batch execution check, not an accuracy experiment or epoch-time benchmark",
        "preflight_loss": result["loss"],
    }
    if args.output:
        path = project_path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
