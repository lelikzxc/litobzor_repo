"""RadonCNN validation with equal weight for every held-out wafer."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.training.trainer import Trainer
from common.training.utils import move_batch_to_device


class RadonTrainer(Trainer):
    """Keep common training behavior, but average validation loss by samples.

    RadonCNN uses unweighted, mean-reduced cross entropy. Its balanced holdout
    contains 385 maps: at batch size 64, the last batch is a singleton. Averaging
    batch means would give that one map the same weight as 64 other maps and
    distort validation-based checkpoint selection and early stopping.
    """

    def validate(self, loader: DataLoader, desc: str = "Val") -> dict[str, float]:
        self.model.eval()
        total_loss, sample_count = 0.0, 0
        all_logits, all_targets = [], []
        with torch.no_grad():
            iterator = tqdm(loader, desc=desc, disable=not self.verbose)
            for batch in iterator:
                inputs, targets = self._unpack_batch(batch)
                inputs = move_batch_to_device(inputs, self.device)
                targets = move_batch_to_device(targets, self.device)
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)
                batch_size = int(targets.shape[0])
                total_loss += loss.item() * batch_size
                sample_count += batch_size
                metric_logits, metric_targets = self._metric_tensors(logits, targets)
                all_logits.append(metric_logits.detach().cpu())
                all_targets.append(metric_targets.detach().cpu())
        if not sample_count:
            raise ValueError("RadonCNN validation requires at least one wafer")
        metrics = {"loss": total_loss / sample_count}
        if self.metric_fns:
            logits = torch.cat(all_logits, dim=0)
            targets = torch.cat(all_targets, dim=0)
            for name, function in self.metric_fns.items():
                try:
                    metrics[name] = function(logits, targets)
                except Exception:
                    pass
        return metrics
