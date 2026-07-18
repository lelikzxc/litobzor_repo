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

## Structure

- `configs/` — YAML experiment configurations
- `models/` — model architecture and weights
- `data_utils/` — dataset loaders and preprocessing
- `utils/` — paper-specific utilities
- `tests/` — unit and integration tests
- `train.py` — training entry point
- `evaluate.py` — evaluation entry point
- `predict.py` — inference entry point
