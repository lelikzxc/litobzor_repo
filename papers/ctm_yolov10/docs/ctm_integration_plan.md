# CTM Integration Plan

## Paper Reference

*"Wafer Defect Detection Technology Based on CTM-IYOLOv10 Network"*

## What CTM Enhances

The Context Transformer Module (CTM) enhances **spatial feature extraction**
by combining:

1. **Local context** — convolutional feature extraction at the current resolution
2. **Global context** — self-attention across the spatial dimension to capture
   long-range dependencies between wafer defect regions

Wafer defects can be small, sparse, and distributed across the wafer surface.
Standard convolutions have limited receptive fields, making it hard to relate
distant defect patterns. CTM addresses this by adding global attention.

---

## CTM Insertion Point Analysis

### Candidate Locations

| Location | Layer | Shape | Pros | Cons |
|----------|-------|-------|------|------|
| **A. After backbone stage 4 (P5)** | After layer 10 (PSA) | `[B,256,20,20]` | Highest semantic features; small spatial size (cheap attention) | Low resolution may miss small defects |
| **B. After backbone stage 3 (P4)** | After layer 6 (C2f) | `[B,128,40,40]` | Good balance of semantics and resolution | Medium cost |
| **C. Replace C2f blocks in neck** | Layers 13, 16, 19, 22 | Various | Directly affects all detection scales | High cost; multiple insertions |
| **D. After SPPF (layer 9)** | Before PSA | `[B,256,20,20]` | Multi-scale context from SPPF feeds into CTM | PSA already provides attention |

### Recommended: **Location A — After SPPF (layer 9), before PSA (layer 10)**

**Reasoning:**

1. **SPPF output** (`[B,256,20,20]`) contains multi-scale pooled features —
   the ideal input for a context module that needs both local and global information.

2. **20×20 spatial size** keeps the attention computation affordable
   (400 tokens, 256 dim → ~100K params for attention).

3. **Before PSA** — CTM can replace or augment PSA's role, providing
   wafer-specific contextual attention that PSA (designed for natural images)
   may not capture.

4. **Single insertion point** — minimal changes to the backbone, easy to
   ablate in experiments.

### Alternative: **Location B — After backbone stage 3 (layer 6)**

If wafer defects are predominantly small, CTM at 40×40 resolution
(128 dim) provides better spatial detail at the cost of 4× more tokens.

---

## Proposed Architecture

### Current YOLOv10 (layers 8–11):

```
layer_8:  C2f(256→256)     → [B, 256, 20, 20]
layer_9:  SPPF(256→256)    → [B, 256, 20, 20]
layer_10: PSA(256→256)     → [B, 256, 20, 20]
layer_11: Upsample(256→128) → [B, 128, 40, 40]
```

### Proposed CTM-IYOLOv10:

```
layer_8:  C2f(256→256)     → [B, 256, 20, 20]
layer_9:  SPPF(256→256)    → [B, 256, 20, 20]
          │
          ↓
       [CTM]               → [B, 256, 20, 20]   ← NEW
          │
          ↓
layer_10: PSA(256→256)     → [B, 256, 20, 20]
layer_11: Upsample(256→128) → [B, 128, 40, 40]
```

CTM is inserted **between SPPF and PSA**, receiving multi-scale pooled
features and passing context-enhanced features to PSA.

---

## CTM Module Interface

```python
class CTM(nn.Module):
    """Context Transformer Module.

    Args:
        dim: Input/output channel dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        dropout: Dropout rate.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Context-enhanced tensor [B, C, H, W].
        """
```

---

## Implementation Strategy

1. **Subclass `YOLOv10Baseline`** → `CTMYOLOv10`
2. **Override model construction** to insert CTM after SPPF
3. **CTM implementation** (next stage):
   - Conv → LayerNorm → Multi-head Self-Attention → MLP → residual
   - Preserves spatial dimensions: `[B, C, H, W]` → `[B, C, H, W]`
4. **Config-driven** — CTM parameters from `configs/config.yaml`

---

## Files to Modify (Next Stage)

| File | Change |
|------|--------|
| `modules/ctm.py` | Implement real CTM |
| `models/yolov10.py` | Add `CTMYOLOv10` class |
| `models/__init__.py` | Export `CTMYOLOv10` |
| `configs/config.yaml` | Add CTM hyperparameters |
| `tests/test_yolov10.py` | Add CTM tests |