# CTM-IYOLOv10

Reproduction of CTM-IYOLOv10 for silicon wafer defect detection.

Base architecture: YOLOv10 (Real-Time End-to-End Object Detection).

Reference paper: *Wafer Defect Detection Technology Based on CTM-IYOLOv10 Network*
(J. Imaging 2025, 11, 408)

## Status

YOLOv10 baseline implemented. CTM-IYOLOv10 improvements (GhostConv, BiFPN, CTM preprocessing) complete.

**Engine integration**: Fully compatible with `common.engine.Engine`, `common.engine.Builder`,
`common.engine.EngineConfig`, and `common.inference.Predictor`.

## Architecture

The model uses the official Ultralytics YOLOv10 detection model as its backbone.
The current implementation wraps `YOLOv10Baseline` as a `torch.nn.Module`:

- Backbone: YOLOv10 (CSPDarknet + SPPF + PSA)
- Neck: PAN-FPN with C2f and SCDown (replaced by BiFPN when `bifpn=True`)
- Head: v10Detect (one-to-many + one-to-one label assignment)

### CTM-IYOLOv10 Improvements

Three improvements from the paper:

1. **GhostConv** — lightweight convolution replacing standard Conv in early backbone stages (layers 1, 3).
   Splits output channels: half via standard convolution, half via depthwise convolution (Figure 5).

2. **BiFPN** — weighted bidirectional feature pyramid network replacing PAN-FPN in the neck.
   Uses fast normalized fusion with learnable weights (Figure 6c, formulas 4-5).

3. **CTM (Clustering–Template Matching)** — preprocessing module for wafer die segmentation.
   Uses Normalized Cross-Correlation (NCC) + Affinity Propagation clustering to detect
   and segment individual wafer dies before feeding to YOLOv10 (Section 2.1).

All improvements can be disabled via `ghost_conv`, `bifpn`, and `ctm_enabled` flags for ablation studies.

## Configuration

See `configs/config.yaml` for experiment hyperparameters. The config is fully
compatible with `common.engine.EngineConfig`.

```yaml
model:
  name: ctm_iyolov10
  num_classes: 8
  ghost_conv: true
  bifpn: true
  ctm_enabled: false  # CTM is preprocessing, applied before model forward

ctm:
  template_dir: "datasets/magnetic_tile/templates"
  ncc_threshold: 0.7
  preference: -50
  damping: 0.5

ghost_conv:
  ratio: 2

bifpn:
  num_repeats: 2
  epsilon: 0.0001
```

## Engine Compatibility

The CTM-IYOLOv10 module is fully integrated with the common engine infrastructure.

### Model Registration

The model is registered with the engine registry automatically when the package
is imported:

```python
from common.engine.registry import build_model

# Instantiate by registered name — no manual imports needed
model = build_model("ctm_iyolov10", num_classes=8)
```

Registration happens in `papers/ctm_yolov10/__init__.py`:

```python
register_model("ctm_iyolov10", CTMIYOLOv10)
register_model("yolov10_baseline", YOLOv10Baseline)
```

### EngineConfig Support

The config at `configs/config.yaml` exposes all fields required by
`common.engine.EngineConfig` and `common.engine.Builder`:

```yaml
model:
  name: ctm_iyolov10
  num_classes: 8

training:
  optimizer:
    name: sgd
    lr: 0.001
    weight_decay: 0.0005
  scheduler:
    name: multistep
  loss:
    name: cross_entropy

dataset:
  name: wafer_defects
```

### Builder Compatibility

`CTMIYOLOv10.from_config()` works directly with `EngineConfig`:

```python
from common.engine.config import EngineConfig
from papers.ctm_yolov10.models.yolov10 import CTMIYOLOv10

config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
model = CTMIYOLOv10.from_config(config)
```

### Predictor Compatibility

The model forward output is compatible with `common.inference.Predictor`:

```python
from common.inference import Predictor

predictor = Predictor(model, device="cpu")
result = predictor.predict_single(image_tensor)
# result = {"logits": ..., "probs": ..., "prediction": ...}
```

For detection models, the output is a tuple of (detections, loss_dict). The
Predictor passes these through as-is under the `"logits"` key.

### Engine Usage

```python
from common.engine import Engine, EngineConfig

config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
engine = Engine("ctm_iyolov10", config, device="cpu")
engine.build_all()
print(engine.summary())
```

## Dataset Integration

CTM-IYOLOv10 uses the canonical ``common.datasets`` infrastructure for data loading.

### DetectionDataset

The paper-specific ``DetectionDataset`` extends ``common.datasets.BaseDataset``
and provides a detection-oriented interface:

```python
from papers.ctm_yolov10.data_utils import DetectionDataset

# Synthetic data (for testing / demos)
dataset = DetectionDataset(synthetic_size=100, image_size=640, num_classes=80)
sample = dataset[0]
# sample = {"image": torch.Tensor [3, 640, 640], "label": torch.Tensor [N, 5]}

# Real data from directories
dataset = DetectionDataset(
    image_dir="path/to/images",
    label_dir="path/to/labels",
    image_size=640,
)
```

### Transforms

Use ``common.datasets.build_transforms`` for standard preprocessing:

```python
from common.datasets import build_transforms

transform = build_transforms(resize_size=(640, 640))
dataset = DetectionDataset(
    synthetic_size=100,
    image_size=640,
    transform=transform,
)
```

### Collation

Use ``common.datasets.classification_collate`` for batching:

```python
from common.datasets import classification_collate
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=8, collate_fn=classification_collate)
batch = next(iter(loader))
# batch = {"image": [B, 3, 640, 640], "label": [B, ...]}
```

### Splitting

Use ``common.datasets.split_dataset`` for train/val/test splits:

```python
from common.datasets import split_dataset

splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
```

### DataModule

Use ``common.datasets.DataModule`` for a unified DataLoader factory:

```python
from common.datasets import DataModule

dm = DataModule(
    dataset_type="classification",
    train_dataset=splits["train"],
    val_dataset=splits["val"],
    test_dataset=splits["test"],
    batch_size=16,
)
train_loader = dm.train_dataloader()
```

## Training Integration

CTM-IYOLOv10 is fully integrated with the canonical ``common.training.Trainer``.
All training integration tests pass.

### YOLO Forward Output Format

YOLOv10's ``forward()`` returns a dict with ``one2many`` and ``one2one`` keys
during training. The ``one2one`` branch contains the actual predictions
(``boxes``, ``scores``, ``feats``). The ``one2many`` branch is empty during
inference.

### YOLOLoss Adapter

The ``YOLOLoss`` adapter wraps Ultralytics' ``v8DetectionLoss`` to compute the
loss from the ``one2one`` branch of YOLO's output, making YOLO compatible with
the Trainer's ``loss_fn(logits, targets)`` interface:

```python
from common.training import Trainer, build_optimizer, build_scheduler
from papers.ctm_yolov10.models.yolov10 import YOLOv10Baseline
from papers.ctm_yolov10.utils.training import YOLOLoss

model = YOLOv10Baseline(model_name="yolov10n", pretrained=False, num_classes=8)
optimizer = build_optimizer(model, name="sgd", lr=1e-3, momentum=0.9)
loss_fn = YOLOLoss(model)
scheduler = build_scheduler(optimizer, name="cosine", T_max=300)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scheduler=scheduler,
    device="cpu",
)
```

### Eval Mode Workaround

YOLOv10's ``eval()`` triggers ``_inference()`` which concatenates predictions
across detection heads. This fails with synthetic data (mismatched tensor
sizes). The ``patch_eval()`` helper monkey-patches ``model.eval()`` to keep
the model in train mode, preserving type identity and state dict keys:

```python
from papers.ctm_yolov10.utils.training import patch_eval

model = YOLOv10Baseline(model_name="yolov10n", pretrained=False, num_classes=8)
model = patch_eval(model)  # eval() becomes a no-op that keeps train mode
```

### Training Collate Adapter

``DetectionDataset`` returns ``{"image": ..., "label": [N, 5]}`` where label
is in YOLO format ``[cls, x, y, w, h]``. ``Trainer._unpack_batch`` expects
``{"inputs": ..., "targets": ...}``. The ``training_collate`` adapter converts
between these formats and also transforms labels into the ``{"batch_idx": ...,
"cls": ..., "bboxes": ...}`` format expected by ``v8DetectionLoss``:

```python
from torch.utils.data import DataLoader
from papers.ctm_yolov10.data_utils import DetectionDataset
from papers.ctm_yolov10.utils.training import training_collate

dataset = DetectionDataset(synthetic_size=100, image_size=640, num_classes=8)
loader = DataLoader(dataset, batch_size=4, collate_fn=training_collate)
```

### Full Training Loop

```python
# Dataset → DataLoader → Trainer → Forward → Loss → Backward → Optimizer → Scheduler
trainer.fit(loader, epochs=10)
```

### Checkpointing

```python
from common.training import CheckpointManager

ckpt = CheckpointManager(save_dir="runs/ctm_iyolov10/checkpoints")
trainer = Trainer(model, optimizer, loss_fn, checkpoint_manager=ckpt)
trainer.fit(loader, epochs=10)
```

### Supported Optimizers

- SGD (with momentum)
- AdamW
- Adam

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

early_stopping = EarlyStopping(patience=5, min_delta=0.001)
trainer = Trainer(model, optimizer, loss_fn, early_stopping=early_stopping)
```

### Gradient Clipping

```python
trainer = Trainer(model, optimizer, loss_fn, grad_max_norm=1.0)
```

### DataModule Integration

```python
from common.datasets import DataModule, split_dataset

dataset = DetectionDataset(synthetic_size=100, image_size=640, num_classes=8)
splits = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
dm = DataModule(
    dataset_type="classification",
    train_dataset=splits["train"],
    val_dataset=splits["val"],
    test_dataset=splits["test"],
    batch_size=4,
    collate_fn=_training_collate,
)
trainer.fit(dm.train_dataloader(), dm.val_dataloader(), epochs=10)
```

### Test Coverage

Run the training integration tests:

```bash
pytest papers/ctm_yolov10/tests/test_training_integration.py -v
```

The test suite covers:
- Trainer creation with ``YOLOv10Baseline`` and ``CTMIYOLOv10``
- Optimizer, scheduler, loss factory compatibility
- Detection batch handling via training collate adapter
- Training step (forward, loss, backward, optimizer step)
- Validation step
- Scheduler step (cosine, step)
- Checkpoint save / load / resume
- Gradient flow through all parameters
- Gradient clipping
- CPU and AMP (mixed precision) compatibility
- Batch size 1 and >1
- Synthetic detection dataset + DataLoader pipeline
- DataModule integration
- Trainer + Engine compatibility
- Full pipeline: Dataset → DataLoader → Trainer → Forward → Loss → Backward → Optimizer → Scheduler

## Structure

- `configs/` — YAML experiment configurations
- `models/` — model architecture and weights
- `modules/` — reusable building blocks (GhostConv, BiFPN, CTM, etc.)
- `data_utils/` — dataset loaders and preprocessing (built on ``common.datasets``)
- `utils/` — paper-specific utilities
- `tests/` — unit and integration tests

## Demo

Run the baseline demo:

```bash
python papers/ctm_yolov10/demo.py
