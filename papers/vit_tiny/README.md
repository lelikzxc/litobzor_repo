# Tiny Vision Transformer

Reproduction of Tiny Vision Transformer for silicon wafer defect detection.

Reference: *Semiconductor Wafer Map Defect Classification with Tiny Vision Transformers*

## Status

**Implemented.** The model returns logits only (no Softmax).

| Component | Status | Verification |
|-----------|--------|-------------|
| Patch Embedding (Conv2d) | ✅ Complete | Shape tests |
| CLS token | ✅ Complete | Shape tests |
| Positional Embedding | ✅ Complete | Shape tests |
| Transformer Encoder blocks | ✅ Complete | Shape + gradient tests |
| Classification head | ✅ Complete | Logit output tests |
| Config-driven creation (from YAML) | ✅ Complete | `from_config()` tests |
| Engine integration | ✅ Complete | Registry, EngineConfig, Predictor, Engine |

## Configuration

See `configs/config.yaml` for experiment hyperparameters. All model parameters
are config-driven via `ViTTiny.from_config()`.

```yaml
model:
  arch:
    image_size: 32
    patch_size: 4
    in_channels: 1
    num_classes: 8
    embed_dim: 192
    num_layers: 4
    num_heads: 3
    mlp_ratio: 4
    dropout: 0.1
    emb_dropout: 0.1
```

## Engine Compatibility

The ViT-Tiny model is fully integrated with the canonical repository engine
infrastructure (`common.engine.*`, `common.inference.*`).

### Model Registration

`ViTTiny` is automatically registered with the engine registry when
`papers.vit_tiny` is imported:

```python
from common.engine.registry import build_model, is_registered, list_registered

# Check registration
assert is_registered("models", "vit_tiny")

# List all registered models
print(list_registered("models"))

# Instantiate by registered name — no manual imports needed
model = build_model("vit_tiny", image_size=32, in_channels=1, num_classes=8)
```

Registration happens in [`papers/vit_tiny/__init__.py`](__init__.py) via
`register_model("vit_tiny", ViTTiny)` with `try/except ValueError` to handle
re-imports gracefully.

### EngineConfig Support

The paper config at [`configs/config.yaml`](configs/config.yaml) is compatible
with `EngineConfig`:

```python
from common.engine.config import EngineConfig

config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")
assert config.get("model.name") == "vit_tiny"
assert config.get("model.num_classes") == 8
assert config.get("model.arch.embed_dim") == 192
```

Engine-compatible fields added to the config:
- `model.num_classes` — number of output classes
- `training.optimizer` — dict with `name`, `lr`, `weight_decay`
- `training.scheduler` — dict with `name`
- `training.loss` — dict with `name`
- `dataset` — section with `name`, `image_size`, `num_classes`

### Builder Compatibility

`ViTTiny` can be instantiated via `common.engine.Builder`:

```python
from common.engine.builder import Builder
from common.engine.config import EngineConfig

config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")
builder = Builder(config)
model = builder.build_model()  # reads model.name → "vit_tiny"
```

### Predictor Compatibility

ViT-Tiny returns a logits tensor `[B, num_classes]` (no Softmax), which is
fully compatible with the canonical `Predictor`'s default postprocessing
(softmax + argmax):

```python
from common.inference.predictor import Predictor

predictor = Predictor(model, device="cpu")
result = predictor.predict_single(image_tensor)
# result = {"logits": ..., "probs": ..., "prediction": ...}
```

No custom postprocessing function is needed.

### Engine Usage

```python
from common.engine.engine import Engine
from common.engine.config import EngineConfig
from papers.vit_tiny.models.vit_tiny import ViTTiny

# Create model and config
model = ViTTiny(num_classes=8)
config = EngineConfig.from_yaml("papers/vit_tiny/configs/config.yaml")

# Create engine
engine = Engine(model, config, device="cpu")
print(engine.summary())

# Single-image inference
x = torch.randn(1, 32, 32)  # grayscale
result = engine.predict_single(x)
# result["logits"].shape → [1, 8]
# result["probs"].shape  → [1, 8]
# result["prediction"]   → argmax class index
```

## Dataset Integration

The ViT-Tiny model is fully integrated with the canonical repository dataset infrastructure (`common.datasets`).

### ViTTinyDataset

[`ViTTinyDataset`](data_utils/dataset.py) extends [`common.datasets.BaseDataset`](../../common/datasets/base_dataset.py) with a classification-specific interface. Each sample is a dict with `"image"` (`torch.Tensor [1, H, W]` grayscale) and `"label"` (`int`), directly compatible with `common.datasets.classification_collate`.

```python
from papers.vit_tiny.data_utils import ViTTinyDataset

# Synthetic data (for testing / demos)
dataset = ViTTinyDataset(synthetic_size=100, image_size=32, num_classes=8)
sample = dataset[0]
# sample["image"].shape → [1, 32, 32]
# sample["label"]       → int (0–7)

# Real data from directories
dataset = ViTTinyDataset(image_dir="path/to/images", image_size=32)
```

### Transforms

Use [`common.datasets.build_transforms`](../../common/datasets/transforms.py) to create torchvision transform pipelines:

```python
from common.datasets import build_transforms

transform = build_transforms(resize_size=(32, 32))
dataset = ViTTinyDataset(synthetic_size=100, image_size=32, transform=transform)
```

### Collation

Use [`common.datasets.classification_collate`](../../common/datasets/collate.py) to batch samples:

```python
from common.datasets import classification_collate

batch = [dataset[i] for i in range(4)]
collated = classification_collate(batch)
# collated["image"].shape → [4, 1, 32, 32]
# collated["label"].shape → [4]
```

### Splitting

Use [`common.datasets.split_dataset`](../../common/datasets/splits.py) for train/val/test splits:

```python
from common.datasets import split_dataset

splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
# splits["train"], splits["val"], splits["test"]
```

### DataModule

Use [`common.datasets.DataModule`](../../common/datasets/datamodule.py) to create DataLoaders:

```python
from common.datasets import DataModule, classification_collate, split_dataset

dataset = ViTTinyDataset(synthetic_size=100, image_size=32, num_classes=8)
splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

dm = DataModule(
    dataset_type="classification",
    train_dataset=splits["train"],
    val_dataset=splits["val"],
    test_dataset=splits["test"],
    batch_size=32,
    collate_fn=classification_collate,
)

train_loader = dm.train_dataloader()
for batch in train_loader:
    # batch["image"].shape → [32, 1, 32, 32]
    # batch["label"].shape → [32]
    ...
```

## Training Integration

The ViT-Tiny model is fully integrated with the canonical repository training infrastructure (`common.training.*`). Training runs through the generic [`Trainer`](../../common/training/trainer.py) without any custom code.

### Trainer

```python
from common.training import (
    Trainer,
    build_optimizer,
    build_scheduler,
    build_loss,
    accuracy,
    f1,
)
from papers.vit_tiny.models.vit_tiny import ViTTiny
from torch.utils.data import DataLoader

# Create model
model = ViTTiny(image_size=32, in_channels=1, num_classes=8)

# Create optimizer, scheduler, loss via canonical factories
optimizer = build_optimizer(model, name="adamw", lr=1e-3, weight_decay=0.05)
scheduler = build_scheduler(optimizer, name="cosine", T_max=100)
loss_fn = build_loss("cross_entropy")

# Create Trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scheduler=scheduler,
    device="cpu",
    metric_fns={"accuracy": accuracy, "f1": f1},
    verbose=True,
)

# Train
train_loader = DataLoader(...)
val_loader = DataLoader(...)
log = trainer.fit(train_loader, val_loader, epochs=100)
```

### Collation Adapter

The canonical [`classification_collate`](../../common/datasets/collate.py) produces `{"image": ..., "label": ...}`, but [`Trainer._unpack_batch`](../../common/training/trainer.py) expects `{"inputs": ..., "targets": ...}`. Use a thin adapter:

```python
from common.datasets import classification_collate


def training_collate(batch):
    collated = classification_collate(batch)
    return {"inputs": collated["image"], "targets": collated["label"]}


loader = DataLoader(dataset, batch_size=32, collate_fn=training_collate)
```

### Checkpointing

Use [`CheckpointManager`](../../common/training/checkpoint.py) for automatic best/last checkpoint saving:

```python
from common.training import CheckpointManager

ckpt_mgr = CheckpointManager(save_dir="runs/vit_tiny/checkpoints")
trainer = Trainer(model, optimizer, loss_fn, checkpoint_manager=ckpt_mgr, ...)
trainer.fit(train_loader, val_loader, epochs=100)
# Checkpoints saved to runs/vit_tiny/checkpoints/{best,last}.pt
```

### Resume Training

```python
# Save
trainer.save_checkpoint("checkpoint.pt")

# Resume in a new session
model = ViTTiny(image_size=32, in_channels=1, num_classes=8)
optimizer = build_optimizer(model, name="adamw", lr=1e-3)
trainer = Trainer(model, optimizer, loss_fn, device="cpu")
loaded_epoch = trainer.load_checkpoint("checkpoint.pt")
print(f"Resuming from epoch {loaded_epoch}")
trainer.fit(train_loader, val_loader, epochs=remaining_epochs)
```

### Optimizer

Supported optimizers via [`build_optimizer`](../../common/training/optim.py):

| Name | Factory Call |
|------|-------------|
| AdamW | `build_optimizer(model, name="adamw", lr=1e-3, weight_decay=0.05)` |
| Adam | `build_optimizer(model, name="adam", lr=1e-3)` |
| SGD | `build_optimizer(model, name="sgd", lr=1e-2, momentum=0.9)` |

### Scheduler

Supported schedulers via [`build_scheduler`](../../common/training/scheduler.py):

| Name | Factory Call |
|------|-------------|
| CosineAnnealingLR | `build_scheduler(optimizer, name="cosine", T_max=100)` |
| StepLR | `build_scheduler(optimizer, name="step", step_size=30)` |
| ReduceLROnPlateau | `build_scheduler(optimizer, name="plateau", patience=5)` |
| OneCycleLR | `build_scheduler(optimizer, name="onecycle", max_lr=1e-3, steps_per_epoch=N, epochs=100)` |

### Mixed Precision

AMP (Automatic Mixed Precision) is supported via [`NativeScaler`](../../common/training/utils.py):

```python
from common.training import NativeScaler

scaler = NativeScaler(enabled=True)  # enabled=False disables AMP
trainer = Trainer(model, optimizer, loss_fn, scaler=scaler, ...)
```

### Early Stopping

```python
from common.training import EarlyStopping

early_stopping = EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)
trainer = Trainer(model, optimizer, loss_fn, early_stopping=early_stopping, ...)
```

### Gradient Clipping

```python
# Norm clipping
trainer = Trainer(model, optimizer, loss_fn, grad_max_norm=1.0, ...)

# Value clipping
trainer = Trainer(model, optimizer, loss_fn, grad_max_value=0.5, ...)
```

### DataModule Integration

The canonical [`DataModule`](../../common/datasets/datamodule.py) works seamlessly with the Trainer:

```python
from common.datasets import DataModule, split_dataset
from papers.vit_tiny.data_utils import ViTTinyDataset

dataset = ViTTinyDataset(synthetic_size=100, image_size=32, num_classes=8)
splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

dm = DataModule(
    dataset_type="classification",
    train_dataset=splits["train"],
    val_dataset=splits["val"],
    test_dataset=splits["test"],
    batch_size=32,
    collate_fn=training_collate,  # adapter from above
)

trainer.fit(dm.train_dataloader(), dm.val_dataloader(), epochs=100)
```

## Structure

- `configs/` — YAML experiment configurations
- `models/` — model architecture and weights
- `data_utils/` — dataset loaders and preprocessing (built on `common.datasets`)
- `utils/` — paper-specific utilities
- `tests/` — unit and integration tests
- `train.py` — training entry point
- `evaluate.py` — evaluation entry point
- `predict.py` — inference entry point
