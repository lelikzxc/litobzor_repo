# CTM-IYOLOv10 Integration Plan

## Paper Reference

*"Wafer Defect Detection Technology Based on CTM-IYOLOv10 Network"*
(J. Imaging 2025, 11, 408)

## Three Improvements from the Paper

The paper introduces three key improvements over baseline YOLOv10:

### 1. GhostConv — Lightweight Convolution (Section 2.2.1, Figure 5)

**What:** Replaces standard convolution in early backbone stages with GhostConv,
which splits output channels into two halves:
- **Y** — standard convolution (half the channels)
- **Y'** — depthwise convolution (half the channels)
- **Output** = concatenation of Y and Y'

**Where:** Layers 1 and 3 of the backbone (first two CSPDarknet stages).

**Why:** Reduces parameters and FLOPs while maintaining accuracy. The paper reports
a 52.3% reduction in model weight.

**Implementation:** `modules/ghost_conv.py` — `GhostConv` class.

### 2. BiFPN — Bidirectional Feature Pyramid Network (Section 2.2.2, Figure 6c)

**What:** Replaces PAN-FPN in the neck with a weighted bidirectional feature
pyramid network using fast normalized fusion:

```
P6_out = Conv(w1 * P6_in + w2 * Resize(P5_out)) / (w1 + w2 + eps)
P5_out = Conv(w1' * P5_in + w2' * P6_out + w3' * Resize(P4_out)) / (w1' + w2' + w3' + eps)
P4_out = Conv(w1'' * P4_in + w2'' * P5_out) / (w1'' + w2'' + eps)
```

**Where:** Replaces layers 11-22 (PAN-FPN neck) in the YOLOv10 model.

**Why:** Improves multi-scale feature fusion with learnable weights, enabling
better detection of defects at different scales.

**Implementation:** `modules/bifpn.py` — `BiFPN` and `BiFPNBlock` classes.

### 3. CTM — Clustering–Template Matching (Section 2.1)

**What:** Preprocessing module for wafer die segmentation that:
1. Uses **Normalized Cross-Correlation (NCC)** via `cv2.matchTemplate` to locate
   wafer dies in the input image
2. Applies **Affinity Propagation (AP) clustering** to group detected die positions
3. Returns bounding boxes for individual wafer dies

**Where:** Applied as preprocessing before YOLOv10 detection (not inserted into
the network itself).

**Why:** Segregates individual wafer dies before defect detection, improving
localization accuracy.

**Implementation:** `modules/ctm.py` — `CTM` class (numpy/cv2-based preprocessing).

---

## Architecture Comparison

### Baseline YOLOv10:
```
Input → Backbone (CSPDarknet + SPPF + PSA) → PAN-FPN Neck → v10Detect Head → Output
```

### CTM-IYOLOv10:
```
Input → [CTM Preprocessing] → Backbone (GhostConv in stages 1,3) → BiFPN Neck → v10Detect Head → Output
```

### Layer Details

| Stage | Baseline YOLOv10 | CTM-IYOLOv10 |
|-------|------------------|--------------|
| Preprocessing | — | CTM (NCC + AP clustering) |
| Backbone layer 1 | Conv | GhostConv |
| Backbone layer 2 | C2f | C2f |
| Backbone layer 3 | Conv | GhostConv |
| Backbone layers 4-10 | C2f, SPPF, PSA | C2f, SPPF, PSA |
| Neck layers 11-22 | PAN-FPN (C2f, Upsample, Concat, SCDown) | BiFPN (weighted bidirectional fusion) |
| Head | v10Detect | v10Detect |

---

## Ablation Studies

All improvements can be independently disabled via config flags:

| `ghost_conv` | `bifpn` | Description |
|:---:|:---:|---|
| false | false | Baseline YOLOv10 |
| true | false | YOLOv10 + GhostConv only |
| false | true | YOLOv10 + BiFPN only |
| true | true | Full CTM-IYOLOv10 |

---

## Implementation Files

| File | Description |
|------|-------------|
| `modules/ghost_conv.py` | GhostConv module (lightweight convolution) |
| `modules/bifpn.py` | BiFPN module (weighted bidirectional feature pyramid) |
| `modules/ctm.py` | CTM preprocessing (NCC + Affinity Propagation clustering) |
| `models/yolov10.py` | `CTMIYOLOv10` and `YOLOv10Baseline` classes |
| `models/__init__.py` | Export `CTMIYOLOv10`, `YOLOv10Baseline` |
| `configs/config.yaml` | Hyperparameters for all improvements |
| `tests/test_ctm.py` | Unit tests for GhostConv, BiFPN, CTM, CTMIYOLOv10 |