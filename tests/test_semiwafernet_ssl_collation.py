"""SSL batches must mix real and accepted pseudo-labels in either order."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from papers.semiwafernet.training.stage_manager import IndexedPseudoDataset, StageManager
from papers.semiwafernet.training.trainer import Trainer


class LabeledMaps(Dataset):
    def __init__(self, sample_format):
        self.sample_format = sample_format
        self.reads = 0

    def __len__(self):
        return 4

    def __getitem__(self, index):
        self.reads += 1
        image = torch.full((1, 2, 2), float(index) / 4)
        label = index % 2
        if self.sample_format.endswith("tensor"):
            label = torch.tensor(label)
        if self.sample_format.startswith("dict"):
            return {"image": image, "label": label}
        return image, label


class ConfidentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.copy_(torch.tensor([12., -12.]))

    def forward(self, images):
        return {"classification": self.head(images.flatten(1))}


def make_trainer(batch_size=4):
    student = ConfidentClassifier()
    manager = StageManager(student, num_classes=2, mc_passes=2)

    def supervised_loss(outputs, targets):
        return nn.functional.cross_entropy(outputs["classification"], targets["classification"])

    return Trainer(
        student, manager, optimizer=torch.optim.SGD(student.parameters(), lr=0.005),
        supervised_loss_fn=supervised_loss, device=torch.device("cpu"),
        batch_size=batch_size, verbose=False,
    )


@pytest.mark.parametrize("sample_format", ["dict_int", "dict_tensor", "pair_int", "pair_tensor"])
@pytest.mark.parametrize("pseudo_first", [True, False])
def test_mixed_union_batches_are_order_independent(sample_format, pseudo_first):
    source = LabeledMaps(sample_format)
    unlabeled = TensorDataset(torch.ones(2, 1, 2, 2))
    pseudo = IndexedPseudoDataset(unlabeled, torch.arange(2), torch.ones(2, dtype=torch.long))
    union = make_trainer()._build_union_loader(DataLoader(source, batch_size=2), pseudo)
    assert source.reads == 0  # Keep augmentation and source loading lazy.
    assert union.dataset.datasets[0].base is source
    indices = [len(source), 0] if pseudo_first else [0, len(source)]
    batches = DataLoader(union.dataset, batch_sampler=[indices])
    images, labels = next(iter(batches))
    assert images.shape == (2, 1, 2, 2)
    assert labels.dtype == torch.long and labels.shape == (2,)
    assert labels.tolist() == ([1, 0] if pseudo_first else [0, 1])
    next(iter(batches))
    assert source.reads == 2


@pytest.mark.parametrize("stage", [2, 3])
def test_ssl_epoch_trains_with_nonempty_lazy_pseudo_set(stage):
    torch.manual_seed(0)
    trainer = make_trainer(batch_size=8)  # One batch contains both sources.
    labeled = DataLoader(LabeledMaps("dict_int"), batch_size=4)
    unlabeled = DataLoader(TensorDataset(torch.ones(4, 1, 2, 2)), batch_size=4)
    before = trainer.student.head.weight.detach().clone()
    completed = []
    metrics = getattr(trainer, f"train_stage{stage}")(
        labeled, unlabeled, num_epochs=1,
        epoch_callback=lambda epoch, which_stage, loss: completed.append((epoch, which_stage, loss)),
    )
    assert metrics["pseudo_accept_rate"] == 100
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, trainer.student.head.weight)
    assert completed[0][:2] == (1, stage)
