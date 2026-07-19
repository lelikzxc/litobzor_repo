# CTM-YOLOv10

Reproduction of CTM-YOLOv10 for silicon wafer defect detection.

Base architecture: YOLOv10 (Real-Time End-to-End Object Detection).

Reference paper: *Wafer Defect Detection Technology Based on CTM-IYOLOv10 Network*

## Status

YOLOv10 baseline implemented. CTM integration complete.

**Engine integration**: Fully compatible with `common.engine.Engine`, `common.engine.Builder`,
`common.engine.EngineConfig`, and `common.inference.Predictor`.

## Architecture

The model uses the official Ultralytics YOLOv10 detection model as its backbone.
The current implementation wraps `YOLOv10Baseline` as a `torch.nn.Module`:

- Backbone: YOLOv10 (CSPDarknet + SPPF + PSA)
- Neck: PAN-FPN with C2f and SCDown
- Head: v10Detect (one-to-many + one-to-one label assignment)

### CTM Integration

The `CTM` (Context Transformer Module) is located in `modules/ctm.py`. It is
inserted after PSA (layer 10) at [B, 256, 20, 20] resolution to enhance feature
extraction for wafer defect detection. The CTM can be disabled via the
`ctm_enabled` flag for ablation studies.

## Configuration

See `configs/config.yaml` for experiment hyperparameters. The config is fully
compatible with `common.engine.EngineConfig`.

## Engine Compatibility

The CTM-YOLOv10 module is fully integrated with the common engine infrastructure.

### Model Registration

The model is registered with the engine registry automatically when the package
is imported:

```python
from common.engine.registry import build_model

# Instantiate by registered name — no manual imports needed
model = build_model("ctm_yolov10", num_classes=8)
```

Registration happens in `papers/ctm_yolov10/__init__.py`:

```python
register_model("ctm_yolov10", CTMYOLOv10)
register_model("yolov10_baseline", YOLOv10Baseline)
```

### EngineConfig Support

The config at `configs/config.yaml` exposes all fields required by
`common.engine.EngineConfig` and `common.engine.Builder`:

```yaml
model:
  name: ctm_yolov10
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

`CTMYOLOv10.from_config()` works directly with `EngineConfig`:

```python
from common.engine.config import EngineConfig
from papers.ctm_yolov10.models.yolov10 import CTMYOLOv10

config = EngineConfig.from_yaml("papers/ctm_yolov10/configs/config.yaml")
model = CTMYOLOv10.from_config(config)
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
engine = Engine("ctm_yolov10", config, device="cpu")
engine.build_all()
print(engine.summary())
```

## Dataset Integration

CTM-YOLOv10 uses the canonical ``common.datasets`` infrastructure for data loading.

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

## Structure

- `configs/` — YAML experiment configurations
- `models/` — model architecture and weights
- `modules/` — reusable building blocks (CTM, C2f, SCDown, etc.)
- `data_utils/` — dataset loaders and preprocessing (built on ``common.datasets``)
- `utils/` — paper-specific utilities
- `tests/` — unit and integration tests

## Demo

Run the baseline demo:

```bash
python papers/ctm_yolov10/demo.py
