# Litobzor

Research repository for reproducing computer vision models from scientific papers focused on semiconductor wafer defect detection.

Each paper lives in `papers/<name>/` with its own model, config, data utilities, and tests. Shared training/inference code is in `common/`.

## Project Structure

```text
litobzor_repo/
├── common/                         # Shared engine, training, metrics, datasets
├── papers/                         # Paper-specific implementations
│   ├── vit_tiny/
│   ├── ctm_yolov10/
│   ├── vmamba/
│   ├── transformer_segmentation/
│   ├── semiwafernet/
│   ├── radon_cnn/
│   └── wafer_defect_classifier/
├── datasets/                       # Local data (gitignored)
├── configs/                        # Global defaults
├── scripts/                        # Dataset prep utilities
├── tests/                          # Shared smoke tests
├── train.py                        # Dispatcher → papers/<name>/train.py
├── evaluate.py                     # Dispatcher → papers/<name>/evaluate.py
└── predict.py                      # Dispatcher → papers/<name>/predict.py (if present)
```

## Implemented Papers

| Paper | Directory | Task |
|-------|-----------|------|
| RadonCNN | [`papers/radon_cnn/`](papers/radon_cnn/) | Classification |
| Wafer Defect Classifier | [`papers/wafer_defect_classifier/`](papers/wafer_defect_classifier/) | Segmentation + classification |
| SemiWaferNet | [`papers/semiwafernet/`](papers/semiwafernet/) | HybridCNN-ViT SSL + ConvoFormer-UNet |
| FCS-VMamba | [`papers/vmamba/`](papers/vmamba/) | Classification |
| CTM-YOLOv10 | [`papers/ctm_yolov10/`](papers/ctm_yolov10/) | Detection |
| Transformer + Atrous | [`papers/transformer_segmentation/`](papers/transformer_segmentation/) | Segmentation |
| Tiny ViT | [`papers/vit_tiny/`](papers/vit_tiny/) | Classification |

## Getting Started

```bash
pip install -r requirements.txt

# Prefer paper entry points
python papers/semiwafernet/train.py --config papers/semiwafernet/configs/config.yaml
python papers/vmamba/train.py --config papers/vmamba/configs/config.yaml

# Or via root dispatchers
python train.py semiwafernet --config papers/semiwafernet/configs/config.yaml
python evaluate.py semiwafernet --help
```

See each paper's README for dataset layout and hyperparameters.

## Common Modules

- [`common/training/`](common/training/) — Trainer, checkpoints, metrics, losses, schedulers
- [`common/engine/`](common/engine/) — Config-driven Engine
- [`common/inference/`](common/inference/) — Predictor / export helpers
- [`common/datasets/`](common/datasets/) — Shared dataset utilities
- [`common/utils/`](common/utils/) — Config, logger, seed, paths
