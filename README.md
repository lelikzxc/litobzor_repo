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
│   └── semiwafernet/               # SemiWaferNet
│
├── configs/                        # Global configurations
├── tests/                          # Project tests
├── scripts/                        # Utility scripts
│
├── train.py                        # Training entry point
├── evaluate.py                     # Evaluation entry point
└── predict.py                      # Inference entry point