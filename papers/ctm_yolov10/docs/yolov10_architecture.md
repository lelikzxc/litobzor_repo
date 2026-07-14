# YOLOv10n Architecture Analysis

## Overview

YOLOv10n is the nano variant of the YOLOv10 real-time end-to-end object detector.
It uses a CSPDarknet backbone with PSA (Position-Sensitive Attention), SPPF,
and a PAN-FPN neck with v10Detect head.

**Total parameters:** 2,775,520  
**Total layers:** 224  
**GFLOPs:** 8.7  
**Input:** `[B, 3, 640, 640]`

---

## Layer-by-Layer Structure

### Backbone (Stem + Stages 1–4)

| # | Module | Input → Output Shape | Stride | Description |
|---|--------|---------------------|--------|-------------|
| 0 | `Conv` | `[B,3,640,640]` → `[B,16,320,320]` | 2 | Stem conv |
| 1 | `Conv` | `[B,16,320,320]` → `[B,32,160,160]` | 2 | Stage 1 down |
| 2 | `C2f` | `[B,32,160,160]` → `[B,32,160,160]` | 1 | Stage 1 features |
| 3 | `Conv` | `[B,32,160,160]` → `[B,64,80,80]` | 2 | Stage 2 down |
| 4 | `C2f` | `[B,64,80,80]` → `[B,64,80,80]` | 1 | Stage 2 features |
| 5 | `SCDown` | `[B,64,80,80]` → `[B,128,40,40]` | 2 | Stage 3 down |
| 6 | `C2f` | `[B,128,40,40]` → `[B,128,40,40]` | 1 | Stage 3 features |
| 7 | `SCDown` | `[B,128,40,40]` → `[B,256,20,20]` | 2 | Stage 4 down |
| 8 | `C2f` | `[B,256,20,20]` → `[B,256,20,20]` | 1 | Stage 4 features |
| 9 | `SPPF` | `[B,256,20,20]` → `[B,256,20,20]` | 1 | Spatial Pyramid Pooling |
| 10 | `PSA` | `[B,256,20,20]` → `[B,256,20,20]` | 1 | Position-Sensitive Attention |

### Neck (PAN-FPN)

| # | Module | Input → Output Shape | Description |
|---|--------|---------------------|-------------|
| 11 | `Upsample` | `[B,256,20,20]` → `[B,256,40,40]` | 2× up from P5 |
| 12 | `Concat` | `[B,256,40,40]` + `[B,128,40,40]` → `[B,384,40,40]` | FPN concat (P5↑ + P4) |
| 13 | `C2f` | `[B,384,40,40]` → `[B,128,40,40]` | FPN fusion |
| 14 | `Upsample` | `[B,128,40,40]` → `[B,128,80,80]` | 2× up from P4 |
| 15 | `Concat` | `[B,128,80,80]` + `[B,64,80,80]` → `[B,192,80,80]` | FPN concat (P4↑ + P3) |
| 16 | `C2f` | `[B,192,80,80]` → `[B,64,80,80]` | FPN fusion (P3 out) |
| 17 | `Conv` | `[B,64,80,80]` → `[B,64,40,40]` | PAN down |
| 18 | `Concat` | `[B,64,40,40]` + `[B,128,40,40]` → `[B,192,40,40]` | PAN concat |
| 19 | `C2f` | `[B,192,40,40]` → `[B,128,40,40]` | PAN fusion (P4 out) |
| 20 | `SCDown` | `[B,128,40,40]` → `[B,128,20,20]` | PAN down |
| 21 | `Concat` | `[B,128,20,20]` + `[B,256,20,20]` → `[B,384,20,20]` | PAN concat |
| 22 | `C2fCIB` | `[B,384,20,20]` → `[B,256,20,20]` | PAN fusion (P5 out) |

### Head

| # | Module | Input → Output Shape | Description |
|---|--------|---------------------|-------------|
| 23 | `v10Detect` | `[P3(64), P4(128), P5(256)]` → `[B,300,6]` | One-to-many + one-to-one |

---

## Feature Map Resolutions

| Stage | Resolution | Channels | Stride | Used by Head |
|-------|-----------|----------|--------|-------------|
| P3 (layer 16) | 80×80 | 64 | 8 | ✓ Small objects |
| P4 (layer 19) | 40×40 | 128 | 16 | ✓ Medium objects |
| P5 (layer 22) | 20×20 | 256 | 32 | ✓ Large objects |

---

## Key Module Types

- **`Conv`**: Conv2d + BatchNorm2d + SiLU
- **`C2f`**: CSP bottleneck with 2 convolutions + n Bottleneck blocks
- **`SCDown`**: Spatial-channel downsampler (Conv + DWConv)
- **`SPPF`**: MaxPool pyramid (5×5, 9×9, 13×13)
- **`PSA`**: Position-sensitive self-attention
- **`C2fCIB`**: C2f with Context Integration Block
- **`v10Detect`**: Dual-head (one-to-many + one-to-one)