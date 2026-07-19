# SemiWaferNet

**Paper:** *SemiWaferNet: Efficient Semi-Supervised Hybrid CNN–Transformer Models for Wafer Defect Classification and Segmentation*

Reproduction of SemiWaferNet for semi-supervised silicon wafer defect detection.

## Tasks

- **Wafer defect classification** — Classify wafer map defects into known categories (e.g., center, edge, scratch, random).
- **Wafer defect segmentation** — Pixel-level segmentation of defect regions on wafer maps.

## Architecture

SemiWaferNet is a **single multitask network** that produces both classification and segmentation outputs during a single forward pass.

### Inference Architecture (Implemented)

```
Input [B, 3, H, W]
  │
  ├── CNN Backbone (4 stages)
  │     ├── stage1: [B,  64, H/4,  W/4]   (stem stride-4 + refine)
  │     ├── stage2: [B, 128, H/8,  W/8]   (stride-2)
  │     ├── stage3: [B, 256, H/16, W/16]  (stride-2)
  │     └── stage4: [B, 512, H/32, W/32]  (stride-2)
  │
  ├── Transformer Encoder (on stage 4 features)
  │     └── PatchProjection → 4× EncoderBlock → LayerNorm
  │
  ├── Feature Fusion
  │     ├── Align 4 CNN stages + transformer tokens
  │     ├── Upsample to stage 1 resolution
  │     ├── Concatenate (5 × fusion_dim)
  │     └── 3×3 fuse conv → task projections
  │
  ├── Classification Head
  │     └── GAP → LayerNorm → Linear → [B, num_classes]
  │
  └── Segmentation Decoder
        └── Conv → ×2 upsample → Conv → ×2 upsample → 1×1 conv
        └── [B, num_classes, H, W]
```

### Key Components

| Component | Description |
|-----------|-------------|
| **CNN Backbone** | 4-stage hierarchical CNN with configurable channels and depths. Stage 1 uses stride-4 (7×7 conv); stages 2-4 use stride-2. Configurable normalization (BN/LN) and activation (ReLU/GELU). |
| **Transformer Encoder** | Projects stage 4 CNN features into token sequences via 1×1 conv + LayerNorm, then applies N encoder blocks (Pre-LN: LN → MHA → residual → LN → MLP → residual). Configurable embed_dim, heads, layers, mlp_ratio, dropout. |
| **Feature Fusion** | Aligns all 4 CNN stages and transformer tokens to a common fusion dimension, upsamples to stage 1 resolution, concatenates (5 × fusion_dim), and fuses via 3×3 conv + BN + ReLU. Produces separate class and seg feature maps via 1×1 task projections. |
| **Classification Head** | Global Average Pooling → LayerNorm → Linear projection to num_classes. |
| **Segmentation Decoder** | Progressive ×2 bilinear upsample with intermediate 3×3 conv + BN + ReLU, followed by 1×1 conv to num_classes. |

## Current Implementation

The repository implements the **complete SemiWaferNet pipeline**:

- **Inference architecture** — Full forward pass from input image to joint classification and segmentation outputs. All components (CNN backbone, transformer encoder, feature fusion, classification head, segmentation decoder) are implemented and tested.
- **Semi-supervised training framework** — Full three-stage training pipeline with EMA teacher, pseudo-label generation, adaptive confidence thresholding, Monte Carlo Dropout uncertainty estimation, uncertainty filtering, consistency regularization, stage management, and a high-level trainer orchestrator. All training components are implemented and tested.

## Status

| Component | Status |
|-----------|--------|
| Directory structure | ✅ Complete |
| Config placeholders | ✅ Complete |
| CNN backbone | ✅ Implemented |
| Transformer encoder | ✅ Implemented |
| Feature fusion | ✅ Implemented |
| Classification head | ✅ Implemented |
| Segmentation decoder | ✅ Implemented |
| Inference tests | ✅ Implemented |
| EMA teacher | ✅ Implemented |
| Pseudo-label generation | ✅ Implemented |
| Consistency loss | ✅ Implemented |
| Monte Carlo Dropout | ✅ Implemented |
| Adaptive thresholding | ✅ Implemented |
| Uncertainty filtering | ✅ Implemented |
| Stage manager | ✅ Implemented |
| Trainer orchestrator | ✅ Implemented |
| Training pipeline tests | ✅ Implemented |
| Experiment utilities | ✅ Implemented |
| Architecture audit | ✅ Documented |
| Engine integration | ✅ Complete |

## Configuration

See `configs/config.yaml` for experiment hyperparameters.

## Engine Compatibility

SemiWaferNet is fully integrated with the canonical repository engine
infrastructure (`common.engine.*`, `common.inference.*`).

### Model Registration

`SemiWaferNet` is automatically registered with the engine registry when
`papers.semiwafernet` is imported:

```python
from common.engine.registry import build_model, is_registered, list_registered

# Check registration
assert is_registered("models", "semiwafernet")

# List all registered models
print(list_registered("models"))

# Instantiate by registered name — no manual imports needed
model = build_model("semiwafernet", num_classes=6)
```

Registration happens in [`papers/semiwafernet/__init__.py`](__init__.py) via
`register_model("semiwafernet", SemiWaferNet)` with `try/except ValueError` to handle
re-imports gracefully.

### EngineConfig Support

The paper config at [`configs/config.yaml`](configs/config.yaml) is compatible
with `EngineConfig`:

```python
from common.engine.config import EngineConfig

config = EngineConfig.from_yaml("papers/semiwafernet/configs/config.yaml")
assert config.get("model.name") == "semiwafernet"
assert config.get("model.num_classes") == 6
assert config.get("model.backbone.channels") == [64, 128, 256, 512]
assert config.get("model.transformer.embed_dim") == 256
```

Engine-compatible fields added to the config:
- `model.num_classes` — number of output classes
- `training.optimizer` — dict with `name`, `lr`, `weight_decay`
- `training.scheduler` — dict with `name`, `step_size`, `gamma`
- `training.loss` — dict with `name`
- `dataset` — section with `name`, `image_size`, `num_classes`

### Builder Compatibility

`SemiWaferNet` can be instantiated via `common.engine.Builder`:

```python
from common.engine.builder import Builder
from common.engine.config import EngineConfig

config = EngineConfig.from_yaml("papers/semiwafernet/configs/config.yaml")
builder = Builder(config)
model = builder.build_model()  # reads model.name → "semiwafernet"
```

### Predictor Compatibility

SemiWaferNet returns a dict with `"classification"` logits `[B, num_classes]`
and `"segmentation"` logits `[B, num_classes, H, W]`. The canonical Predictor
expects a single tensor output, so a custom postprocessing function is needed
to handle the multitask output:

```python
from common.inference.predictor import Predictor

def semiwafernet_postprocess(logits: dict) -> dict:
    """Extract classification logits for Predictor compatibility."""
    return logits["classification"]

predictor = Predictor(model, device="cpu", postprocess_fn=semiwafernet_postprocess)
result = predictor.predict_single(image_tensor)
# result = {"logits": ..., "probs": ..., "prediction": ...}
```

### Engine Usage

```python
from common.engine.engine import Engine
from common.engine.config import EngineConfig
from papers.semiwafernet.models.semiwafernet import SemiWaferNet

# Create model and config
model = SemiWaferNet(num_classes=6)
config = EngineConfig.from_yaml("papers/semiwafernet/configs/config.yaml")

# Create engine
engine = Engine(model, config, device="cpu")
print(engine.summary())

# Single-image inference
x = torch.randn(3, 512, 512)
result = engine.predict_single(x)
# result["logits"].shape → [1, 6]
# result["probs"].shape  → [1, 6]
# result["prediction"]   → argmax class index
```

## Structure

```
papers/semiwafernet/
├── __init__.py                  # Package init + engine registration
├── README.md                    # This file
├── config.yaml                  # Root config template
├── demo.py                      # Architecture demo
├── configs/
│   └── config.yaml              # Experiment configuration
├── models/
│   ├── __init__.py              # Model exports
│   ├── semiwafernet.py          # Main multitask model
│   ├── classifier.py            # Classification head
│   └── decoder.py               # Segmentation decoder
├── modules/
│   ├── __init__.py              # Module exports
│   ├── cnn_backbone.py          # CNN feature extractor
│   ├── transformer.py           # Transformer encoder
│   └── fusion.py                # Feature fusion
├── training/
│   ├── __init__.py              # Training component exports
│   ├── ema.py                   # EMA teacher model
│   ├── pseudo_label.py          # Pseudo-label generation
│   ├── consistency.py           # Consistency regularization loss
│   ├── mc_dropout.py            # Monte Carlo Dropout uncertainty
│   ├── adaptive_threshold.py    # Adaptive confidence thresholding
│   ├── uncertainty.py           # Uncertainty-based filtering
│   ├── stage_manager.py         # Three-stage training schedule
│   └── trainer.py               # High-level training orchestrator
├── data_utils/
│   └── __init__.py              # Data utility stubs
├── utils/
│   ├── __init__.py              # Utility exports
│   └── experiment.py            # Experiment metadata utilities
├── docs/
│   └── architecture_audit.md    # Architecture audit document
├── tests/
│   ├── __init__.py              # Test stubs
│   ├── test_semiwafernet.py     # Architecture tests (23)
│   ├── test_training.py         # Training component tests (34)
│   ├── test_training_pipeline.py# Pipeline tests (76)
│   ├── test_experiment.py       # Experiment utility tests (24)
│   └── test_engine_integration.py  # Engine integration tests
```

## References

- SemiWaferNet paper (to be linked when available)
