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

## Configuration

See `configs/config.yaml` for experiment hyperparameters.

## Structure

```
papers/semiwafernet/
├── __init__.py                  # Package init
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
│   └── test_experiment.py       # Experiment utility tests (24)
```

## References

- SemiWaferNet paper (to be linked when available)
