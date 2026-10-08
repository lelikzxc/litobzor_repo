"""Every validation wafer must contribute equally to checkpoint selection."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.training.metrics import accuracy, f1
from common.training.utils import NativeScaler
from papers.radon_cnn.trainer import RadonTrainer


@pytest.fixture
def trainer_and_data():
    # Two easy wafers followed by one confidently misclassified wafer. With
    # batch_size=2, averaging batch means would overweight the last wafer.
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]])
    labels = torch.tensor([0, 0, 1])
    trainer = RadonTrainer(
        model, torch.optim.SGD(model.parameters(), lr=0.01), nn.CrossEntropyLoss(),
        device="cpu", scaler=NativeScaler(enabled=False), verbose=False,
        metric_fns={"accuracy": accuracy, "f1": f1},
    )
    return trainer, TensorDataset(logits, labels), nn.functional.cross_entropy(logits, labels).item()


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_validation_loss_is_invariant_to_uneven_batch_partition(trainer_and_data, batch_size):
    trainer, dataset, expected_loss = trainer_and_data
    metrics = trainer.validate(DataLoader(dataset, batch_size=batch_size))
    assert metrics["loss"] == pytest.approx(expected_loss, abs=1e-6)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(0.4)


def test_empty_validation_cannot_become_a_zero_loss_best_checkpoint(trainer_and_data):
    trainer, dataset, _ = trainer_and_data
    empty = TensorDataset(*[tensor[:0] for tensor in dataset.tensors])
    with pytest.raises(ValueError, match="at least one wafer"):
        trainer.validate(DataLoader(empty, batch_size=2))
