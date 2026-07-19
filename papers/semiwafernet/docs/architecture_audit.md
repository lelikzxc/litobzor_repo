# SemiWaferNet — Architecture Audit

## Overview

This document describes the implemented architecture of SemiWaferNet — a hybrid CNN–Transformer model for joint wafer defect classification and segmentation. The implementation consists of an **inference pipeline** (5 components) and a **semi-supervised training framework** (8 components).

All components are described as implemented. No comparison against any paper is made.

---

## Inference Components

### 1. CNNBackbone

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/modules/cnn_backbone.py`](../modules/cnn_backbone.py) |
| **Purpose** | Multi-scale CNN feature extraction with 4 stages of progressive downsampling |
| **Input shape** | `[B, 3, H, W]` — input RGB image |
| **Output shape** | `list[[B, 64, H/4, W/4], [B, 128, H/8, W/8], [B, 256, H/16, W/16], [B, 512, H/32, W/32]]` |
| **Trainable params** | 7,056,320 |
| **Execution order** | `stem (7×7 conv, stride-4, BN, ReLU)` → `stage1_refine (N-1 ConvBlocks)` → `stage2 (CNNStage, stride-2)` → `stage3 (CNNStage, stride-2)` → `stage4 (CNNStage, stride-2)` |
| **Category** | Inference |

**Sub-components:**

| Module | Params | Description |
|---|---|---|
| `stem` | 9,536 | 7×7 conv, stride-4, BN, ReLU — aggressive early downsampling |
| `stage1_refine` | 36,992 | 1 ConvBlock (3×3, stride-1) refining stage 1 features |
| `stages` (2–4) | 7,009,792 | 3 CNNStage modules, each with stride-2 first block + N-1 stride-1 blocks |

**Architecture details:**
- Default channels: `[64, 128, 256, 512]`
- Default depths: `[2, 2, 6, 2]`
- Each `ConvBlock`: `Conv2d(k=3, bias=False)` → `BatchNorm2d` → `ReLU(inplace=True)`
- Configurable norm (`bn`/`ln`) and activation (`relu`/`gelu`)

---

### 2. TransformerEncoder

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/modules/transformer.py`](../modules/transformer.py) |
| **Purpose** | Project CNN stage-4 features into token sequences and apply global self-attention |
| **Input shape** | `[B, 512, H/32, W/32]` — CNN backbone stage 4 output |
| **Output shape** | `tokens: [B, (H/32)*(W/32), 256]`, `spatial: (H/32, W/32)` |
| **Trainable params** | 3,291,392 |
| **Execution order** | `PatchProjection (1×1 conv → LN → flatten)` → `4× TransformerEncoderBlock (Pre-LN: LN → MHA → residual → LN → MLP → residual)` → `final LayerNorm` |
| **Category** | Inference |

**Sub-components:**

| Module | Params | Description |
|---|---|---|
| `patch_proj` | 131,840 | 1×1 conv (512→256) + LayerNorm + flatten to sequence |
| `blocks` (4×) | 3,159,040 | 4 Pre-LN Transformer encoder blocks |
| `norm` | 512 | Final LayerNorm(256) |

**Per-block breakdown:**
- `MultiHeadSelfAttention`: embed_dim=256, num_heads=8, head_dim=32 — QKV projection (256→768), output projection (256→256)
- `TransformerMLP`: embed_dim=256, mlp_ratio=4 → hidden_dim=1024 — Linear(256→1024) → GELU → Dropout → Linear(1024→256) → Dropout

---

### 3. FeatureFusion

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/modules/fusion.py`](../modules/fusion.py) |
| **Purpose** | Fuse multi-scale CNN features (4 stages) with transformer-enhanced features into a unified representation |
| **Input shape** | `cnn_features: list[4 tensors]` at resolutions H/4, H/8, H/16, H/32; `transformer_tokens: [B, N, 256]`; `transformer_spatial: (H/32, W/32)` |
| **Output shape** | `class_features: [B, 256, H/4, W/4]`, `seg_features: [B, 256, H/4, W/4]` |
| **Trainable params** | 3,395,840 |
| **Execution order** | `4× ChannelAlign (1×1 conv, BN, ReLU)` → `upsample all to H/4` → `reshape transformer tokens to [B, 256, H/32, W/32]` → `transformer_proj (1×1 conv)` → `upsample to H/4` → `concat(5 × 256)` → `fusion_conv (3×3 conv, BN, ReLU)` → `class_proj (1×1 conv)` → `seg_proj (1×1 conv)` |
| **Category** | Inference |

**Sub-components:**

| Module | Params | Description |
|---|---|---|
| `cnn_align` (4×) | 248,832 | 4× ChannelAlign: 1×1 conv (C_i → 256) + BN + ReLU |
| `transformer_proj` | 65,792 | 1×1 conv (256→256) for transformer map |
| `fusion_conv` | 2,949,632 | 3×3 conv (1280→256) + BN + ReLU — main fusion |
| `class_proj` | 65,792 | 1×1 conv (256→256) task-specific projection |
| `seg_proj` | 65,792 | 1×1 conv (256→256) task-specific projection |

---

### 4. ClassifierHead

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/models/classifier.py`](../models/classifier.py) |
| **Purpose** | Global average pooling → LayerNorm → linear projection for wafer defect classification |
| **Input shape** | `[B, 256, H/4, W/4]` — fused classification features |
| **Output shape** | `[B, 6]` — classification logits (6 defect classes) |
| **Trainable params** | 2,054 |
| **Execution order** | `AdaptiveAvgPool2d(1)` → `flatten` → `LayerNorm(256)` → `Linear(256→6)` |
| **Category** | Inference |

---

### 5. SegmentationDecoder

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/models/decoder.py`](../models/decoder.py) |
| **Purpose** | Progressive upsampling of fused features to full-resolution segmentation logits |
| **Input shape** | `[B, 256, H/4, W/4]` — fused segmentation features |
| **Output shape** | `[B, 6, H, W]` — pixel-wise segmentation logits (6 classes) |
| **Trainable params** | 1,182,214 |
| **Execution order** | `3×3 conv + BN + ReLU` → `×2 upsample (bilinear)` → `3×3 conv + BN + ReLU` → `×2 upsample (bilinear)` → `1×1 conv (256→6)` |
| **Category** | Inference |

---

### 6. SemiWaferNet (Main Model)

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/models/semiwafernet.py`](../models/semiwafernet.py) |
| **Purpose** | Orchestrate the full inference pipeline: backbone → transformer → fusion → task heads |
| **Input shape** | `[B, 3, H, W]` |
| **Output shape** | `{"classification": [B, 6], "segmentation": [B, 6, H, W]}` |
| **Trainable params** | 14,927,820 |
| **Execution order** | `CNNBackbone(x)` → `TransformerEncoder(stage4)` → `FeatureFusion(cnn_features, tokens, spatial)` → `ClassifierHead(class_features)` + `SegmentationDecoder(seg_features)` |
| **Category** | Inference |

**Inference execution graph:**

```
Input [B, 3, H, W]
    │
    ▼
CNNBackbone
    │
    ├── stem (7×7, stride-4) ──► stage1_refine ──► [B, 64, H/4, W/4]
    │
    ├── CNNStage (stride-2) ──► [B, 128, H/8, W/8]
    │
    ├── CNNStage (stride-2) ──► [B, 256, H/16, W/16]
    │
    └── CNNStage (stride-2) ──► [B, 512, H/32, W/32]
                                      │
                                      ▼
                              TransformerEncoder
                              PatchProjection (1×1, LN, flatten)
                                      │
                              4× TransformerEncoderBlock (Pre-LN)
                                      │
                              LayerNorm
                                      │
                              tokens [B, N, 256], spatial (H/32, W/32)
                                      │
                                      ▼
                              FeatureFusion
                              ┌─────────────────────────────────────┐
                              │ 4× ChannelAlign + upsample to H/4   │
                              │ transformer_map + upsample to H/4   │
                              │ concat(5 × 256) → 3×3 conv → 256    │
                              └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
            ClassifierHead                      SegmentationDecoder
            GAP → LN → Linear                  3×3 conv → ×2 upsample
                    │                           3×3 conv → ×2 upsample
                    │                           1×1 conv
                    ▼                                   ▼
            [B, 6]                              [B, 6, H, W]
         (classification)                     (segmentation)
```

---

## Semi-Supervised Training Components

### 7. EMATeacher

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/training/ema.py`](../training/ema.py) |
| **Purpose** | Maintain an exponential moving average of student model parameters for stable pseudo-label generation |
| **Input shape** | `[B, 3, H, W]` (same as student model) |
| **Output shape** | `{"classification": [B, 6], "segmentation": [B, 6, H, W]}` (same as student model) |
| **Trainable params** | 0 (frozen by design) |
| **Execution order** | `__init__`: `deepcopy(student)` → `requires_grad_(False)` → `eval()`; `update(student)`: `θ_t = momentum * θ_t + (1 - momentum) * θ_s` for params, `copy_` for buffers; `forward(x)`: `teacher(x)` with `@torch.no_grad()` |
| **Category** | Semi-supervised training |

**Key details:**
- Default momentum: `0.999`
- Buffer copy copies student buffers directly (no EMA for buffers)
- Buffer copy guards against non-floating-point dtypes (e.g., `num_batches_tracked` is `LongTensor`)
- Teacher is always in `eval()` mode regardless of student state

---

### 8. PseudoLabelGenerator

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/training/pseudo_label.py`](../training/pseudo_label.py) |
| **Purpose** | Generate confidence-thresholded pseudo-labels from teacher predictions for both classification and segmentation |
| **Input shape** | `class_logits: [B, 6]`, `seg_logits: [B, 6, H, W]` |
| **Output shape** | `{"classification": (pseudo_labels [B], mask [B]), "segmentation": (pseudo_labels [B, H, W], mask [B, H, W])}` |
| **Trainable params** | 0 (no learned parameters) |
| **Execution order** | `softmax(logits, dim=1)` → `max(dim=1)` → `confidence >= threshold` → `(pseudo_labels, mask)` |
| **Category** | Semi-supervised training |

**Key details:**
- Default confidence threshold: `0.9`
- Classification: per-sample argmax + confidence
- Segmentation: per-pixel argmax + confidence
- Low-confidence predictions are masked out (not used in loss computation)

---

### 9. ConsistencyLoss

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/training/consistency.py`](../training/consistency.py) |
| **Purpose** | Enforce prediction consistency between student and teacher models via MSE on logits |
| **Input shape** | `student_logits: [B, 6]` / `[B, 6, H, W]`, `teacher_logits: [B, 6]` / `[B, 6, H, W]` |
| **Output shape** | Scalar loss (reduction="mean") or per-sample loss (reduction="none") |
| **Trainable params** | 0 (no learned parameters) |
| **Execution order** | `teacher_logits.detach()` → `optional masking` → `MSELoss(student, teacher)` |
| **Category** | Semi-supervised training |

**Key details:**
- Teacher logits are always detached from the computation graph
- Optional boolean mask filters which samples/pixels contribute to the loss
- Empty mask returns `0.0` loss
- Configurable reduction: `"mean"`, `"sum"`, `"none"`
- Segmentation masking expands mask from `[B, H, W]` to `[B, C, H, W]` for proper indexing

---

### 10. MonteCarloDropout

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/training/mc_dropout.py`](../training/mc_dropout.py) |
| **Purpose** | Estimate prediction uncertainty via N stochastic forward passes with dropout enabled |
| **Input shape** | `model: nn.Module`, `x: [B, 3, H, W]` |
| **Output shape** | `{"mean_probs_class": [B, 6], "mean_probs_seg": [B, 6, H, W], "entropy_class": [B], "entropy_seg": [B, H, W], "mutual_info_class": [B], "mutual_info_seg": [B, H, W]}` |
| **Trainable params** | 0 (no learned parameters) |
| **Execution order** | `model.train()` → `N× forward(x)` → `stack(logits)` → `softmax(dim=2)` → `mean_probs = mean(stack, dim=0)` → `entropy = -sum(mean_probs * log(mean_probs), dim=1)` → `expected_entropy = mean(-sum(p * log(p), dim=1), dim=0)` → `mutual_info = entropy - expected_entropy` → `model.eval()` |
| **Category** | Semi-supervised training |

**Key details:**
- Default number of passes: `20`
- Temporarily switches model to `train()` mode to enable dropout, then restores original mode
- All passes are `@torch.no_grad()` — no gradients computed
- Predictive entropy measures total uncertainty (aleatoric + epistemic)
- Mutual information measures epistemic uncertainty (model uncertainty)
- Both entropy and mutual information are non-negative by construction

---

### 11. AdaptiveThreshold

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/training/adaptive_threshold.py`](../training/adaptive_threshold.py) |
| **Purpose** | Compute per-class adaptive confidence thresholds using online statistics (Welford's algorithm) |
| **Input shape** | `confidence: [B]` or `[B, H, W]`, `pseudo_labels: [B]` or `[B, H, W]` |
| **Output shape** | Scalar threshold value (float) |
| **Trainable params** | 0 (no learned parameters) |
| **Execution order** | `update_statistics(confidence, pseudo_labels)` → `Welford online mean/std per class` → `compute_threshold(entropy)` → `tau = base_threshold + alpha * (sigma/mu) + beta * (1 - entropy)` → `clamp(tau, 0, 1)` |
| **Category** | Semi-supervised training |

**Key details:**
- Default: `base_threshold=0.9`, `alpha=0.1`, `beta=0.05`
- Per-class statistics tracked via Welford's online algorithm (numerically stable, single pass)
- `sigma/mu` is the coefficient of variation — higher variance raises the threshold
- Entropy bonus: high entropy (uncertain predictions) raises the threshold
- Result clamped to `[0, 1]`
- `reset()` clears all accumulated statistics
- `get_threshold_value()` returns the last computed threshold (or `base_threshold` if not yet computed)

---

### 12. UncertaintyFilter

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/training/uncertainty.py`](../training/uncertainty.py) |
| **Purpose** | Filter pseudo-labels based on confidence, predictive entropy, and mutual information criteria |
| **Input shape** | `confidence_class: [B]`, `confidence_seg: [B, H, W]`, `adaptive_threshold: float/tensor`, `entropy_class: [B]`, `entropy_seg: [B, H, W]`, `mutual_info_class: [B]`, `mutual_info_seg: [B, H, W]` |
| **Output shape** | `{"classification": [B] bool mask, "segmentation": [B, H, W] bool mask}` |
| **Trainable params** | 0 (no learned parameters) |
| **Execution order** | `confidence >= adaptive_threshold` AND `entropy < entropy_threshold` AND `mutual_info < mi_threshold` → boolean mask per task |
| **Category** | Semi-supervised training |

**Key details:**
- Default: `entropy_threshold=0.5`, `mi_threshold=0.3`
- Three criteria must all pass for a sample/pixel to be accepted:
  1. **Confidence**: `confidence >= adaptive_threshold` (per-class adaptive)
  2. **Entropy**: `entropy < entropy_threshold` (low predictive uncertainty)
  3. **Mutual information**: `mutual_info < mi_threshold` (low epistemic uncertainty)
- Classification: per-sample filtering `[B]`
- Segmentation: per-pixel filtering `[B, H, W]`
- `adaptive_threshold` can be a float (same for all classes) or a tensor `[num_classes]` (per-class)

---

### 13. StageManager

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/training/stage_manager.py`](../training/stage_manager.py) |
| **Purpose** | Orchestrate the three-stage semi-supervised training schedule, coordinating all training components |
| **Input shape** | N/A (orchestrator, not a forward module) |
| **Output shape** | N/A (provides methods for stage management, pseudo-label generation, consistency loss, teacher refresh) |
| **Trainable params** | 0 (orchestrator with references to student/teacher) |
| **Execution order** | `__init__(student, ...)` → `set_stage(1|2|3)` → `is_semi_supervised()` → `generate_pseudo_labels(unlabeled_x)` → `compute_consistency_loss(student_out, teacher_out, masks)` → `refresh_teacher()` → `reset_statistics()` |
| **Category** | Semi-supervised training |

**Key details:**
- **Stage 1** — Supervised only: trains on labeled data. No pseudo-labels, no teacher involvement.
- **Stage 2** — Semi-supervised: generates pseudo-labels from teacher, computes adaptive threshold statistics, applies MC Dropout uncertainty estimation, filters via uncertainty criteria, computes consistency loss.
- **Stage 3** — Refresh + retrain: regenerates pseudo-labels, recomputes statistics, refreshes teacher from current student, retrains.
- Owns instances of: `EMATeacher`, `AdaptiveThreshold`, `MonteCarloDropout`, `UncertaintyFilter`, `ConsistencyLoss`
- `generate_pseudo_labels()` runs the full pipeline: teacher forward → MC Dropout → softmax confidence → adaptive threshold update → uncertainty filtering → return accepted pseudo-labels + masks
- `compute_consistency_loss()` delegates to `ConsistencyLoss` with optional per-task masking
- `refresh_teacher()` creates a fresh `EMATeacher` from the current student (preserving momentum)
- `reset_statistics()` resets adaptive threshold statistics between stages

---

### 14. Trainer

| Property | Value |
|---|---|
| **Location** | [`papers/semiwafernet/training/trainer.py`](../training/trainer.py) |
| **Purpose** | High-level training orchestrator that manages the full three-stage training loop with optimizer, scheduler, and loss functions |
| **Input shape** | N/A (orchestrator) |
| **Output shape** | N/A (provides `train_stage1/2/3` methods returning metric dicts) |
| **Trainable params** | 0 (orchestrator) |
| **Execution order** | `__init__(model, ...)` → `set_optimizer(optim)` → `set_scheduler(sched)` → `set_supervised_loss(loss_fn)` → `train_stage1(train_loader, epochs)` → `train_stage2(label_loader, unlabel_loader, epochs)` → `train_stage3(label_loader, unlabel_loader, epochs)` → `generate_pseudo_labels(unlabeled_x)` → `refresh_teacher()` |
| **Category** | Semi-supervised training |

**Key details:**
- Owns a `StageManager` instance which owns all training sub-components
- `train_stage1`: supervised training loop with optimizer step, loss computation, metric tracking
- `train_stage2`: semi-supervised loop — labeled data uses supervised loss, unlabeled data uses pseudo-labels + consistency loss
- `train_stage3`: refresh phase — regenerates pseudo-labels, resets statistics, retrains with updated teacher
- Placeholder optimizer/scheduler/loss — designed to accept any `torch.optim.Optimizer`, `torch.optim.lr_scheduler`, and callable loss function
- `generate_pseudo_labels()` and `refresh_teacher()` delegate to `StageManager`
- Returns per-epoch metric dictionaries with loss values

---

## Semi-Supervised Training Pipeline

```
Labeled batch [B_l, 3, H, W]    Unlabeled batch [B_u, 3, H, W]
         │                                │
         │                                │
         ▼                                ▼
   Student Model (SemiWaferNet)     Teacher Model (EMATeacher)
         │                                │
         ├── classification [B_l, 6]      ├── classification [B_u, 6]
         └── segmentation [B_l, 6, H, W]  └── segmentation [B_u, 6, H, W]
                                                  │
                    ┌─────────────────────────────┴─────────────────────────────┐
                    │                                                          │
                    ▼                                                          ▼
          MonteCarloDropout (N passes)                              PseudoLabelGenerator
          mean_probs, entropy, mutual_info                          softmax → argmax → confidence
                    │                                                          │
                    │                                                          │
                    └──────────┬───────────────────────────────────────────────┘
                               │
                               ▼
                    AdaptiveThreshold
                    Welford online statistics → tau = base + α·(σ/μ) + β·(1-H)
                               │
                               ▼
                    UncertaintyFilter
                    confidence ≥ tau AND entropy < H_thresh AND mutual_info < MI_thresh
                               │
                               ▼
                    accepted pseudo_labels + masks
                               │
         ┌─────────────────────┘
         │
         ▼
   ConsistencyLoss
   MSE(student_logits, teacher_logits.detach())
   × mask (filter uncertain predictions)
         │
         ▼
   consistency_loss (scalar)
         │
         ▼
   Total loss = supervised_loss + λ * consistency_loss
         │
         ▼
   optimizer.step() → EMATeacher.update(student)
```

---

## Parameter Summary

| Component | Params | % of Total | Category |
|---|---|---|---|
| CNNBackbone | 7,056,320 | 47.3% | Inference |
| TransformerEncoder | 3,291,392 | 22.0% | Inference |
| FeatureFusion | 3,395,840 | 22.7% | Inference |
| ClassifierHead | 2,054 | 0.01% | Inference |
| SegmentationDecoder | 1,182,214 | 7.9% | Inference |
| **Total (inference)** | **14,927,820** | **100%** | |
| EMATeacher | 0 (frozen copy) | — | Training |
| PseudoLabelGenerator | 0 (stateless) | — | Training |
| ConsistencyLoss | 0 (stateless) | — | Training |
| MonteCarloDropout | 0 (stateless) | — | Training |
| AdaptiveThreshold | 0 (stateless) | — | Training |
| UncertaintyFilter | 0 (stateless) | — | Training |
| StageManager | 0 (orchestrator) | — | Training |
| Trainer | 0 (orchestrator) | — | Training |