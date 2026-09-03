# SemiWaferNet

**Paper:** *SemiWaferNet: Efficient Semi-Supervised Hybrid CNN–Transformer Models for Wafer Defect Classification and Segmentation*  
https://doi.org/10.3390/electronics15071437

Two **separate** models (not one multitask net):

1. **HybridCNN-ViT** — classification 32×32 / 9 classes + **3-stage SSL** on unlabeled WM-811K  
2. **ConvoFormer-UNet** — binary segmentation 64×64 (~7.11M), fully supervised

## Quick start

```bash
# Classification + SSL (150k unlabeled from empty failureType in labels.csv)
python papers/semiwafernet/train.py --config papers/semiwafernet/configs/config.yaml

# Segmentation
python papers/semiwafernet/train.py --config papers/semiwafernet/configs/config.yaml --mode segmentation
```

`semi_supervised.enabled: true` in `configs/config.yaml`. Unlabeled pool: `datasets/wm811k` (empty `failureType`, max 150000).

## Architecture

### HybridCNN-ViT (classification)

```
Input [B, 1, 32, 32]
  → CNN backbone (64 → 128)
  → AdaptiveAvgPool 8×8
  → Hybrid ViT (D=128, L=4, FFN=256, dropout 0.5/0.2) + class token
  → Linear → [B, 9]
```

SSL Stages 1–3: supervised warm-up → MC-Dropout pseudo-labels + adaptive τ / uncertainty filter → teacher refresh + retrain. Loss on accepted pseudo-labels is supervised CE.

### ConvoFormer-UNet (segmentation)

```
Input [B, 1, 64, 64]
  → hierarchical CNN encoder + ConvoFormer bottleneck
  → transposed-conv decoder + 1×1 skip fusion + deep supervision
  → binary defect logits
```

## Layout

```
papers/semiwafernet/
├── configs/config.yaml
├── models/          # SemiWaferNet, ConvoFormerUNet
├── modules/         # CNN backbone, ViT / ConvoFormer blocks
├── data_utils/      # WM-811K labeled + unlabeled loaders
├── training/        # EMA, MC dropout, StageManager, SSL trainer
├── train.py
├── evaluate.py
└── tests/
```

## Smoke demo

```bash
python papers/semiwafernet/demo.py
```
