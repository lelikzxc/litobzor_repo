# Litobzor

Research repository for reproducing computer vision models from scientific papers focused on semiconductor wafer defect detection.

The goal of the project is to implement, compare, and evaluate deep learning architectures described in research papers using a unified and reproducible codebase.

Each paper implementation is isolated in its own directory with:

- model architecture;
- configuration files;
- datasets and utilities;
- tests.

## Project Structure

```text
litobzor_repo/
│
├── common/                         # Shared utilities
│   ├── utils/                      # Configuration, logging, reproducibility
│   ├── metrics/                    # Evaluation metrics
│   ├── losses/
│   └── visualization/
│
├── papers/                         # Paper-specific implementations
│   ├── vit_tiny/                   # Tiny Vision Transformer
│   ├── ctm_yolov10/                # CTM-YOLOv10
│   ├── vmamba/                     # FCS-VMamba
│   ├── transformer_segmentation/   # Transformer + Atrous Convolution
│   ├── semiwafernet/               # SemiWaferNet
│   ├── radon_cnn/                  # RadonCNN (Jeong et al. 2023)
│   └── wafer_defect_classifier/    # Two-stage pipeline: segmentation + classification
│
├── datasets/                       # Dataset storage
│   ├── wm811k/                     # WM-811K wafer map dataset
│   ├── wm811k_seg/                 # WM-811K segmentation dataset
│   ├── magnetic_tile/
│   └── dagm/
│
├── configs/                        # Global configurations
├── tests/                          # Project tests
├── scripts/                        # Utility scripts
│
├── train.py                        # Training entry point
├── evaluate.py                     # Evaluation entry point
└── predict.py                      # Inference entry point
```

## Implemented Papers

| Paper | Directory | Task | Status |
|-------|-----------|------|--------|
| **RadonCNN** — Jeong et al. (2023) | [`papers/radon_cnn/`](papers/radon_cnn/) | Wafer defect classification (7 classes) | ✅ Implemented |
| **Wafer Defect Classifier** — Two-stage pipeline | [`papers/wafer_defect_classifier/`](papers/wafer_defect_classifier/) | Segmentation + classification (9 classes) | ✅ Implemented |
| **SemiWaferNet** — Semi-supervised wafer defect classification | [`papers/semiwafernet/`](papers/semiwafernet/) | Semi-supervised classification | ✅ Implemented |
| **FCS-VMamba** — VMamba with FCS modules | [`papers/vmamba/`](papers/vmamba/) | Wafer defect classification | ✅ Implemented |
| **CTM-YOLOv10** — YOLOv10 with Coordinate Attention & Triplet Attention | [`papers/ctm_yolov10/`](papers/ctm_yolov10/) | Magnetic tile defect detection | ✅ Implemented |
| **Transformer + Atrous** — SegFormer with Atrous convolution | [`papers/transformer_segmentation/`](papers/transformer_segmentation/) | Semantic segmentation | ✅ Implemented |
| **Tiny ViT** — Lightweight Vision Transformer | [`papers/vit_tiny/`](papers/vit_tiny/) | Wafer defect classification | ✅ Implemented |

## Datasets

- **WM-811K** — 811K wafer bin map images with 9 defect classes (none, Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch). Stored in [`datasets/wm811k/`](datasets/wm811k/).
- **Magnetic Tile** — Surface defect detection dataset for magnetic tile. Used by CTM-YOLOv10.
- **DAGM** — Synthetic surface defect detection dataset.

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run a specific paper
python papers/wafer_defect_classifier/train.py --full
python papers/radon_cnn/train.py --epochs 50
```

## Common Modules

The [`common/`](common/) directory provides reusable components:

- [`common/training/`](common/training/) — Trainer, checkpoint manager, metrics, early stopping, schedulers
- [`common/inference/`](common/inference/) — Predictor, visualization, benchmarking, export
- [`common/utils/`](common/utils/) — Config loader, logger, seed, paths
- [`common/metrics/`](common/metrics/) — Accuracy, precision, recall, F1, IoU, Dice