# FCS-VMamba

Reproduction of **FCS-VMamba** for silicon wafer defect detection.

Reference: *FCS-VMamba: A Frequency-Compressed State Space Model for
Silicon Wafer Defect Detection* (IEEE TIM, 2025)

## Status

**Implemented.** Faithful reproduction with implementation assumptions where
the paper does not provide low-level module details.

| Component | Status | Verification |
|-----------|--------|-------------|
| Patch Embedding (Conv2d stride=4 + LayerNorm) | ✅ Complete | Shape tests |
| Patch Merging (2× downsampling, channel doubling) | ✅ Complete | Shape tests |
| SS2D (official VMamba v2, vendored) | ✅ Complete | Vendored from MzeroMiko/VMamba |
| FCSVSSBlock (LN → SS2D → residual → FA → SFS → LN → MLP → residual) | ✅ Complete | Forward hook order verification |
| Frequency Attention (FA) | ✅ Complete | Shape + gradient tests |
| Saliency Feature Suppression (SFS) | ✅ Complete | Shape + gradient tests |
| Cross-Layer Channel Attention (CLCA) | ✅ Complete | Shape + gradient tests |
| Classification head (GAP → LayerNorm → Linear) | ✅ Complete | Logit output tests |
| Ablation support (8 configurations) | ✅ Complete | Param increase + output shape tests |
| Config-driven creation (from YAML) | ✅ Complete | `from_config()` tests |
| Experiment metadata | ✅ Complete | `ExperimentInfo` dataclass |
| Architecture audit | ✅ Complete | `docs/architecture_audit.md` |
| Pretrained weights | ❌ Not available | Training from scratch required |
| Training pipeline | ❌ Not implemented | `data_utils/` is a placeholder |

## Architecture

```
Input image [B, 3, 224, 224]
  │
  ├─ PatchEmbed2D (Conv2d stride=4 + LayerNorm)
  │   → [B, 96, 56, 56]
  │
  ├─ Stage 1: VSSBlock × 2 (+ FA + SFS)
  │   → [B, 96, 56, 56]
  ├─ PatchMerging (2× downsampling)
  │   → [B, 192, 28, 28]
  │
  ├─ Stage 2: VSSBlock × 2 (+ FA + SFS)
  │   → [B, 192, 28, 28]
  ├─ PatchMerging
  │   → [B, 384, 14, 14]
  │
  ├─ Stage 3: VSSBlock × 6 (+ FA + SFS)
  │   → [B, 384, 14, 14]
  ├─ PatchMerging
  │   → [B, 768, 7, 7]
  │
  ├─ Stage 4: VSSBlock × 2 (+ FA + SFS)
  │   → [B, 768, 7, 7]
  │
  ├─ CLCA: Cross-Layer Channel Attention
  │   (Stage 1 → Stage 4, Stage 2 → Stage 4, Stage 3 → Stage 4)
  │   → [B, 768, 7, 7]
  │
  ├─ Global Average Pooling
  │   → [B, 768]
  ├─ LayerNorm
  ├─ Linear classifier
  │   → [B, num_classes] (logits, no Softmax)
```

### Key Components

- **PatchEmbed2D**: Convolutional patch embedding (Conv2d k=4, s=4) + LayerNorm
- **PatchMerging**: 2× spatial downsampling with channel doubling (unfold → Linear → LayerNorm)
- **FCSVSSBlock**: Custom VMamba block with paper-correct execution order:
  `LN → SS2D → DropPath + residual → FA → SFS → LN → MLP → DropPath + residual`
- **SS2D**: 2D Selective Scan (vendored official VMamba v2, 4-directional cross scan)
- **FA (Frequency Attention)**: 2D FFT → frequency-domain gating → inverse FFT
- **SFS (Saliency Feature Suppression)**: Spatial saliency → learned suppression gate
- **CLCA (Cross-Layer Channel Attention)**: SE-style channel attention across stages
- **Classification Head**: Global Average Pooling → LayerNorm → Linear

### FCSVSSBlock Execution Order

The block execution order matches the paper exactly (verified via `register_forward_hook`):

```
Input [B, C, H, W]
  │
  ├─ 1. LayerNorm (channel-last permute)
  ├─ 2. SS2D (official v2, channel-first)
  ├─ 3. DropPath + residual
  ├─ 4. FA (Frequency Attention) — FFT → gate → iFFT
  ├─ 5. SFS (Saliency Feature Suppression)
  ├─ 6. LayerNorm (channel-last permute)
  ├─ 7. MLP (GELU, channels_first=True via Linear2d)
  ├─ 8. DropPath + residual
  │
  Output [B, C, H, W]
```

FA and SFS are inserted *between* the SS2D residual and the MLP path, exactly
as described in the FCS-VMamba paper (Sections 3.2, 3.3).

## Validation Status

See [`docs/architecture_audit.md`](docs/architecture_audit.md) for the full
per-component architecture audit.

### Exact Matches

- ✅ Patch Embedding — Exact match with VMamba paper
- ✅ Patch Merging — Exact match with VMamba paper
- ✅ SS2D (official v2) — Vendored official implementation
- ✅ FCSVSSBlock order — Verified via forward hooks
- ✅ FA placement — Between SS2D residual and MLP
- ✅ SFS placement — Between SS2D residual and MLP
- ✅ CLCA placement — Cross-stage connections
- ✅ Classification head — GAP → LayerNorm → Linear
- ✅ Ablation support — All 8 configurations supported
- ✅ Config-driven — All params from YAML

### Implementation Assumptions

- ⚠️ FA implementation — Approximation (paper lacks low-level detail)
- ⚠️ SFS implementation — Approximation (paper lacks low-level detail)
- ⚠️ CLCA implementation — Approximation (paper lacks low-level detail)

### Missing

- ❌ Pretrained weights — Not available (training from scratch required)
- ❌ Training pipeline — Not implemented (`data_utils/` is a placeholder)

## Configuration

See `configs/config.yaml` for experiment hyperparameters. All model parameters
are config-driven via `FCSVMamba.from_config()`.

```yaml
model:
  backbone:
    embed_dim: 96
    depths: [2, 2, 6, 2]
    num_heads: [3, 6, 12, 24]
    ssm_ratio: 2.0
    mlp_ratio: 4.0
    drop_path_rate: 0.2
  fa:
    enabled: true
    reduction: 16
    fft_norm: ortho
  sfs:
    enabled: true
    reduction: 4
  clca:
    enabled: true
    reduction: 16
```

## Structure

```
papers/vmamba/
├── __init__.py              — package marker
├── README.md                — this file
├── config.yaml              — root config template
├── demo.py                  — research validation demo
├── configs/
│   └── config.yaml          — experiment hyperparameters
├── docs/
│   └── architecture_audit.md — per-component architecture audit
├── models/
│   ├── __init__.py
│   └── vmamba.py            — FCSVMamba backbone model
├── modules/
│   ├── __init__.py          — module exports
│   ├── patch_embed.py       — PatchEmbed2D
│   ├── patch_merging.py     — PatchMerging
│   ├── vss_block.py         — SS2D, FCSVSSBlock, VSSBlock
│   └── fcs_modules.py       — FA, SFS, CLCA
├── kernels/
│   ├── __init__.py
│   ├── vmamba_official.py   — vendored official VMamba (SS2D, Mlp, etc.)
│   ├── csms6s.py            — vendored selective scan kernels
│   └── csm_triton.py        — vendored cross-scan triton kernels
├── data_utils/
│   └── __init__.py          — dataset loaders (placeholder)
├── utils/
│   ├── __init__.py          — utility exports
│   └── experiment.py        — ExperimentInfo dataclass
└── tests/
    ├── __init__.py
    └── test_vmamba.py       — 30+ tests
```

## Engine Compatibility

The FCS-VMamba model is fully integrated with the canonical repository engine infrastructure (`common.engine.*`, `common.inference.*`).

### Model Registration

`FCSVMamba` is automatically registered with the engine registry when `papers.vmamba` is imported:

```python
from common.engine.registry import build_model, is_registered, list_registered

# Check registration
assert is_registered("models", "fcs_vmamba")

# List all registered models
print(list_registered("models"))

# Instantiate by registered name — no manual imports needed
model = build_model("fcs_vmamba", in_channels=3, image_size=224, num_classes=8)
```

Registration happens in [`papers/vmamba/__init__.py`](__init__.py) via `register_model("fcs_vmamba", FCSVMamba)` with `try/except ValueError` to handle re-imports gracefully.

### EngineConfig Support

The paper config at [`configs/config.yaml`](configs/config.yaml) is compatible with `EngineConfig`:

```python
from common.engine.config import EngineConfig

config = EngineConfig.from_yaml("papers/vmamba/configs/config.yaml")
assert config.get("model.name") == "fcs_vmamba"
assert config.get("model.num_classes") == 8
assert config.get("model.backbone.embed_dim") == 96
```

Engine-compatible fields added to the config:
- `model.num_classes` — number of output classes
- `training.optimizer` — dict with `name`, `lr`, `weight_decay`
- `training.scheduler` — dict with `name`
- `training.loss` — dict with `name`
- `dataset` — section with `name`, `image_size`, `num_classes`

### Builder Compatibility

`FCSVMamba` can be instantiated via `common.engine.Builder`:

```python
from common.engine.builder import Builder
from common.engine.config import EngineConfig

config = EngineConfig.from_yaml("papers/vmamba/configs/config.yaml")
builder = Builder(config)
model = builder.build_model()  # reads model.name → "fcs_vmamba"
```

### Predictor Compatibility

FCS-VMamba returns a logits tensor `[B, num_classes]` (no Softmax), which is fully compatible with the canonical `Predictor`'s default postprocessing (softmax + argmax):

```python
from common.inference.predictor import Predictor

predictor = Predictor(model, device="cpu")
result = predictor.predict_single(image_tensor)
# result = {"logits": ..., "probs": ..., "prediction": ...}
```

No custom postprocessing function is needed — unlike detection models (e.g. YOLOv10) which return tuples.

### Engine Usage

```python
from common.engine.engine import Engine
from common.engine.config import EngineConfig
from papers.vmamba.models.vmamba import FCSVMamba

# Create model and config
model = FCSVMamba(num_classes=8)
config = EngineConfig.from_yaml("papers/vmamba/configs/config.yaml")

# Create engine
engine = Engine(model, config, device="cpu")
print(engine.summary())

# Single-image inference
x = torch.randn(3, 224, 224)
result = engine.predict_single(x)
# result["logits"].shape → [1, 8]
# result["probs"].shape  → [1, 8]
# result["prediction"]   → argmax class index
```

## References

- FCS-VMamba: A Frequency-Compressed State Space Model for Silicon Wafer
  Defect Detection. *IEEE Transactions on Instrumentation and Measurement*,
  2025.
- VMamba: Visual State Space Model. arXiv:2401.10166, 2024.
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
  arXiv:2312.00752, 2023.
