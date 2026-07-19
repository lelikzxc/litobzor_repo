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
│   └── __init__.py          — dataset loaders (placeholder)
├── utils/
│   └── __init__.py          — paper-specific utilities (placeholder)
├── tests/
│   ├── __init__.py
│   ├── test_segformer.py    — 18 tests
│   └── test_atrous.py       — 11 tests
└── demo.py                  — comparison demo
```

## References

- SegFormer: Simple and Efficient Design for Semantic Segmentation with
  Transformers. *NeurIPS*, 2021.
- Encoder-Decoder with Atrous Separable Convolution for Semantic Image
  Segmentation. *ECCV*, 2018.
