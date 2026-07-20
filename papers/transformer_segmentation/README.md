# SegFormer + Atrous

Reproduction of **SegFormer** with Atrous convolutions for silicon wafer defect segmentation.

Reference: *SegFormer: Simple and Efficient Design for Semantic Segmentation
with Transformers* (NeurIPS 2021)

## Status

**Baseline + Atrous implemented.** SegFormer baseline (MiT backbone + MLP
decoder) and Atrous Enhancement module are complete.

| Component | Status | Verification |
|-----------|--------|-------------|
| OverlapPatchEmbed (7×7 conv, stride=4) | ✅ Complete | Shape tests |
| Efficient Self-Attention (spatial reduction) | ✅ Complete | Shape tests |
| Mix-FFN (3×3 depthwise conv + MLP) | ✅ Complete | Shape tests |
| Transformer Block (LN → Attn → residual → LN → FFN → residual) | ✅ Complete | Shape tests |
| MiT Stage (PatchEmbed + N blocks) | ✅ Complete | Shape tests |
| MiT Backbone (4 stages, multi-scale features) | ✅ Complete | Feature shape tests |
| Atrous Enhancement (multi-scale atrous conv + fusion) | ✅ Complete | Shape + gradient + rate config tests |
| MLP Decoder (project → upsample → concat → fuse → head) | ✅ Complete | Output shape tests |
| SegFormer model (backbone + decoder) | ✅ Complete | Forward + gradient tests |
| SegFormer + Atrous (backbone + atrous + decoder) | ✅ Complete | Param increase + output shape tests |
| Config-driven creation (from YAML) | ✅ Complete | `from_config()` tests |

## Architecture

```
Input image [B, 3, H, W]
  │
  ├─ OverlapPatchEmbed (stride=4, 7×7 conv)
  │   → [B, C1, H/4, W/4]
  │
  ├─ Stage 1: MiT Block × N1
  │   ├─ Efficient Self-Attention (reduction ratio R1)
  │   ├─ Mix-FFN (3×3 depthwise conv + MLP)
  │   └─ OverlapPatchMerging (stride=2)
  │   → [B, C2, H/8, W/8]
  │
  ├─ Stage 2: MiT Block × N2
  │   ├─ Efficient Self-Attention (reduction ratio R2)
  │   ├─ Mix-FFN (3×3 depthwise conv + MLP)
  │   └─ OverlapPatchMerging (stride=2)
  │   → [B, C3, H/16, W/16]
  │
  ├─ Stage 3: MiT Block × N3
  │   ├─ Efficient Self-Attention (reduction ratio R3)
  │   ├─ Mix-FFN (3×3 depthwise conv + MLP)
  │   └─ OverlapPatchMerging (stride=2)
  │   → [B, C4, H/32, W/32]
  │
  ├─ Stage 4: MiT Block × N4
  │   ├─ Efficient Self-Attention (reduction ratio R4)
  │   └─ Mix-FFN (3×3 depthwise conv + MLP)
  │   → [B, C4, H/32, W/32]
  │
  ├─ Atrous Convolution Enhancement Module (optional)
  │   ├─ Bottleneck: C → C/r (1×1 conv + BN + ReLU)
  │   ├─ Parallel atrous convs (rates: 1, 6, 12, 18)
  │   ├─ Concatenate + fuse (1×1 conv + BN + ReLU)
  │   ├─ Output projection: C/r → C (1×1 conv + BN)
  │   └─ Residual connection with learnable scale
  │   → [B, C4, H/32, W/32]
  │
  ├─ MLP Decoder
  │   ├─ MLP per stage (C_i → C)
  │   ├─ Upsample to 1/4 resolution
  │   └─ Concatenate + fuse
  │   → [B, C, H/4, W/4]
  │
  ├─ Segmentation Head
  │   ├─ MLP (C → num_classes)
  │   └─ Upsample to original resolution
  │   → [B, num_classes, H, W]
  ```

### Atrous Insertion Point

The Atrous Enhancement module is inserted **between the MiT backbone and
the MLP decoder**, operating on the final stage (Stage 4) feature map:

```
MiT Backbone (Stage 4 output)  [B, C4, H/32, W/32]
  │
  ├─ AtrousEnhancement
  │   ├─ 1×1 bottleneck (C → C/r)
  │   ├─ Parallel 3×3 atrous convs (rates: 1, 6, 12, 18)
  │   ├─ Concatenate + 1×1 fuse
  │   ├─ 1×1 output projection (C/r → C)
  │   └─ Residual + learnable scale
  │
  → [B, C4, H/32, W/32]  (same resolution, enhanced features)
  │
  ├─ MLP Decoder (uses all 4 stage features)
```

This placement enhances the highest-level semantic features with multi-scale
context before they are combined with lower-level features in the decoder.

### Key Components

- **OverlapPatchEmbed**: Convolutional patch embedding with overlapping 7×7 kernels
- **MiT Block**: Transformer encoder block with efficient self-attention (spatial reduction) and Mix-FFN (depthwise conv + MLP)
- **Efficient Self-Attention**: Self-attention with spatial reduction (K, V pooled by reduction ratio R)
- **Mix-FFN**: MLP with 3×3 depthwise convolution between two linear layers
- **Atrous Enhancement**: Multi-scale atrous convolution module (rates 1, 6, 12, 18) for expanded receptive field
- **MLP Decoder**: Lightweight decoder aggregating multi-level features via per-stage MLPs and upsampling
- **Segmentation Head**: Final MLP projecting to per-pixel class logits

## Configuration

See `configs/config.yaml` for experiment hyperparameters. All model parameters
are config-driven via `SegFormer.from_config()`.

```yaml
model:
  backbone:
    variant: B0
    qkv_bias: false
  decoder:
    decoder_dim: 256
    dropout: 0.0
  atrous:
    enabled: true
    rates: [1, 6, 12, 18]
    reduction: 4
  input:
    image_size: 512
    channels: 3
```

## Engine Compatibility

The SegFormer model is fully integrated with the canonical repository engine
infrastructure (`common.engine.*`, `common.inference.*`).

### Model Registration

`SegFormer` is automatically registered with the engine registry when
`papers.transformer_segmentation` is imported:

```python
from common.engine.registry import build_model, is_registered, list_registered

# Check registration
assert is_registered("models", "segformer_atrous")

# List all registered models
print(list_registered("models"))

# Instantiate by registered name — no manual imports needed
model = build_model("segformer_atrous", in_channels=3, variant="B0", num_classes=8)
```

Registration happens in [`papers/transformer_segmentation/__init__.py`](__init__.py)
via `register_model("segformer_atrous", SegFormer)` with `try/except ValueError`
to handle re-imports gracefully.

### EngineConfig Support

The paper config at [`configs/config.yaml`](configs/config.yaml) is compatible
with `EngineConfig`:

```python
from common.engine.config import EngineConfig

config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")
assert config.get("model.name") == "segformer_atrous"
assert config.get("model.num_classes") == 8
assert config.get("model.backbone.variant") == "B0"
```

Engine-compatible fields added to the config:
- `model.num_classes` — number of output classes
- `training.optimizer` — dict with `name`, `lr`, `weight_decay`
- `training.scheduler` — dict with `name`
- `training.loss` — dict with `name`
- `dataset` — section with `name`, `image_size`, `num_classes`

### Builder Compatibility

`SegFormer` can be instantiated via `common.engine.Builder`:

```python
from common.engine.builder import Builder
from common.engine.config import EngineConfig

config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")
builder = Builder(config)
model = builder.build_model()  # reads model.name → "segformer_atrous"
```

### Predictor Compatibility

SegFormer returns a segmentation logits tensor `[B, num_classes, H, W]`
(no Softmax), which is fully compatible with the canonical `Predictor`'s
default postprocessing (softmax + argmax):

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
from papers.transformer_segmentation.models.segformer import SegFormer

# Create model and config
model = SegFormer(variant="B0", num_classes=8)
config = EngineConfig.from_yaml("papers/transformer_segmentation/configs/config.yaml")

# Create engine
engine = Engine(model, config, device="cpu")
print(engine.summary())

# Single-image inference
x = torch.randn(3, 512, 512)
result = engine.predict_single(x)
# result["logits"].shape → [1, 8, 512, 512]
# result["probs"].shape  → [1, 8, 512, 512]
# result["prediction"]   → [1, 512, 512] (argmax class per pixel)
```

## Dataset Integration

The SegFormer model is fully integrated with the canonical repository dataset infrastructure (`common.datasets`).

### SegFormerDataset

[`SegFormerDataset`](data_utils/dataset.py) extends [`common.datasets.BaseDataset`](../../common/datasets/base_dataset.py) with a segmentation-specific interface. Each sample is a dict with `"image"` (`torch.Tensor [3, H, W]` RGB) and `"mask"` (`torch.Tensor [H, W]` with integer class indices), directly compatible with `common.datasets.segmentation_collate`.

```python
from papers.transformer_segmentation.data_utils import SegFormerDataset

# Synthetic data (for testing / demos)
dataset = SegFormerDataset(synthetic_size=50, image_size=512, num_classes=8)
sample = dataset[0]
# sample["image"].shape → [3, 512, 512]
# sample["mask"].shape  → [512, 512]  (integer class indices)

# Real data from directories
dataset = SegFormerDataset(
    image_dir="path/to/images",
    mask_dir="path/to/masks",
    image_size=512,
)
```

### Transforms

Use [`common.datasets.build_transforms`](../../common/datasets/transforms.py) to create torchvision transform pipelines:

```python
from common.datasets import build_transforms

transform = build_transforms(resize_size=(512, 512))
dataset = SegFormerDataset(synthetic_size=50, image_size=512, transform=transform)
```

### Collation

Use [`common.datasets.segmentation_collate`](../../common/datasets/collate.py) to batch samples:

```python
from common.datasets import segmentation_collate

batch = [dataset[i] for i in range(4)]
collated = segmentation_collate(batch)
# collated["image"].shape → [4, 3, 512, 512]
# collated["mask"].shape  → [4, 512, 512]
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
from common.datasets import DataModule, segmentation_collate, split_dataset

dataset = SegFormerDataset(synthetic_size=100, image_size=512, num_classes=8)
splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

dm = DataModule(
    dataset_type="segmentation",
    train_dataset=splits["train"],
    val_dataset=splits["val"],
    test_dataset=splits["test"],
    batch_size=8,
    collate_fn=segmentation_collate,
)

train_loader = dm.train_dataloader()
for batch in train_loader:
    # batch["image"].shape → [8, 3, 512, 512]
    # batch["mask"].shape  → [8, 512, 512]
    ...
```

## Training Integration

SegFormer is fully integrated with the canonical ``common.training.Trainer``.
All 44 training integration tests pass.

### Forward Output Format

``SegFormer.forward()`` returns a single logits tensor of shape
``[B, num_classes, H, W]`` (no Softmax), which is directly compatible with
``nn.CrossEntropyLoss`` and other segmentation losses from the canonical
loss factory.

### Loss Functions

SegFormer works with any loss from ``common.training.build_loss``:

- ``cross_entropy`` — standard pixel-wise cross-entropy
- ``dice`` — Dice loss for segmentation
- ``iou`` — IoU (Jaccard) loss
- ``bce_dice`` — combined BCE + Dice

```python
from common.training import build_loss

loss_fn = build_loss("cross_entropy")  # recommended for SegFormer
```

### Training Collate Adapter

``SegFormerDataset`` returns ``{"image": ..., "mask": ...}``.
``Trainer._unpack_batch`` expects ``{"inputs": ..., "targets": ...}``.
Use the ``_training_collate`` adapter to remap keys:

```python
from torch.utils.data import DataLoader
from common.datasets import segmentation_collate
from papers.transformer_segmentation.data_utils import SegFormerDataset

def training_collate(batch):
    collated = segmentation_collate(batch)
    return {"inputs": collated["image"], "targets": collated["mask"]}

dataset = SegFormerDataset(synthetic_size=100, image_size=512, num_classes=8)
loader = DataLoader(dataset, batch_size=4, collate_fn=training_collate)
```

### Full Training Loop

```python
from common.training import Trainer, build_optimizer, build_scheduler
from papers.transformer_segmentation.models.segformer import SegFormer

model = SegFormer(variant="B0", num_classes=8)
optimizer = build_optimizer(model, name="adamw", lr=1e-4, weight_decay=0.01)
loss_fn = build_loss("cross_entropy")
scheduler = build_scheduler(optimizer, name="cosine", T_max=300)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scheduler=scheduler,
    device="cpu",
)

# Dataset → DataLoader → Trainer → Forward → Loss → Backward → Optimizer → Scheduler
trainer.fit(train_loader, epochs=50)
```

### Checkpointing

```python
from common.training import CheckpointManager

ckpt = CheckpointManager(save_dir="runs/segformer/checkpoints")
trainer = Trainer(model, optimizer, loss_fn, checkpoint_manager=ckpt)
trainer.fit(loader, epochs=50)
```

### Supported Optimizers

- AdamW (recommended for SegFormer)
- Adam
- SGD (with momentum)

### Supported Schedulers

- CosineAnnealingLR
- StepLR
- ReduceLROnPlateau

### Mixed Precision

```python
from common.training import NativeScaler

scaler = NativeScaler(enabled=True)
trainer = Trainer(model, optimizer, loss_fn, scaler=scaler)
```

### Early Stopping

```python
from common.training import EarlyStopping

early_stopping = EarlyStopping(patience=10, min_delta=0.001)
trainer = Trainer(model, optimizer, loss_fn, early_stopping=early_stopping)
```

### Gradient Clipping

```python
trainer = Trainer(model, optimizer, loss_fn, grad_max_norm=1.0)
```

### Metrics

SegFormer supports segmentation metrics from ``common.training``:

```python
from common.training import build_metric

metric_fns = {
    "iou": build_metric("iou"),
    "dice": build_metric("dice"),
    "pixel_accuracy": build_metric("pixel_accuracy"),
}
trainer = Trainer(model, optimizer, loss_fn, metric_fns=metric_fns)
```

### DataModule Integration

```python
from common.datasets import DataModule, segmentation_collate, split_dataset
from papers.transformer_segmentation.data_utils import SegFormerDataset

dataset = SegFormerDataset(synthetic_size=100, image_size=512, num_classes=8)
splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
dm = DataModule(
    dataset_type="segmentation",
    train_dataset=splits["train"],
    val_dataset=splits["val"],
    test_dataset=splits["test"],
    batch_size=4,
    collate_fn=training_collate,
)
trainer.fit(dm.train_dataloader(), dm.val_dataloader(), epochs=50)
```

### Test Coverage

Run the training integration tests:

```bash
pytest papers/transformer_segmentation/tests/test_training_integration.py -v
```

The test suite (44 tests) covers:
- Trainer creation with SegFormer (baseline and Atrous-enhanced)
- Segmentation loss (CrossEntropyLoss, DiceLoss)
- Optimizer, scheduler factory compatibility
- Segmentation batch handling via training collate adapter
- Training step (forward, loss, backward, optimizer step)
- Validation step
- Scheduler step (cosine, step, plateau)
- Checkpoint save / load / resume
- Gradient flow through all parameters
- Gradient clipping
- CPU and AMP (mixed precision) compatibility
- Batch size 1 and >1
- Synthetic segmentation dataset + DataLoader pipeline
- DataModule integration
- Trainer + Engine compatibility
- Full pipeline: Dataset → DataLoader → Trainer → Forward → Loss → Backward → Optimizer → Scheduler

## Structure

```
papers/transformer_segmentation/
├── __init__.py              — package marker
├── README.md                — this file
├── config.yaml              — root config template
├── configs/
│   └── config.yaml          — experiment hyperparameters
├── models/
│   ├── __init__.py          — model exports
│   ├── segformer.py         — SegFormer model (backbone + atrous + decoder)
│   └── decoder.py           — MLP decoder
├── modules/
│   ├── __init__.py          — module exports
│   ├── mit.py               — MiT backbone (PatchEmbed, Attention, FFN, Blocks)
│   └── atrous.py            — Atrous Enhancement module
├── data_utils/
│   ├── __init__.py          — dataset exports
│   └── dataset.py           — SegFormerDataset (built on common.datasets)
├── utils/
│   └── __init__.py          — paper-specific utilities (placeholder)
├── tests/
│   ├── __init__.py
│   ├── test_segformer.py    — 18 tests
│   ├── test_atrous.py       — 11 tests
│   └── test_dataset_integration.py — dataset integration tests
└── demo.py                  — comparison demo
```

## References

- SegFormer: Simple and Efficient Design for Semantic Segmentation with
  Transformers. *NeurIPS*, 2021.
- Encoder-Decoder with Atrous Separable Convolution for Semantic Image
  Segmentation. *ECCV*, 2018.
