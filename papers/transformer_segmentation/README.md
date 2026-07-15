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

See `configs/config.yaml` for experiment hyperparameters.

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
