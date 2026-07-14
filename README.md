# Litobzor

Research repository for reproducing computer vision models from scientific papers focused on semiconductor wafer defect detection.

The goal of the project is to implement, compare, and evaluate deep learning architectures described in research papers using a unified and reproducible codebase.

Each paper implementation is isolated in its own directory with:
- model architecture;
- configuration files;
- datasets and utilities;
- tests.

## Project Structure
litobzor_repo/
│
├── common/ # Shared utilities
│ ├── utils/ # Configuration, logging, reproducibility
│ ├── metrics/ # Evaluation metrics
│ ├── losses/
│ └── visualization/
│
├── papers/ # Paper-specific implementations
│ ├── vit_tiny/ # Tiny Vision Transformer
│ ├── ctm_yolov10/ # CTM-YOLOv10
│ ├── vmamba/ # FCS-VMamba
│ ├── transformer_segmentation/
│ └── semiwafernet/
│
├── configs/ # Global configurations
├── tests/ # Project tests
├── scripts/ # Utility scripts
│
├── train.py
├── evaluate.py
└── predict.py

## Implemented Papers

| Model | Task | 
|------|------|--------|
| Tiny Vision Transformer | Classification |
| CTM-YOLOv10 | Detection | 
| FCS-VMamba | Classification | 
| Transformer + Atrous Convolution | Segmentation | 
| SemiWaferNet | Semi-supervised learning | 

## Installation

```bash
git clone https://github.com/niime/litobzor_repo.git
cd litobzor_repo

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
