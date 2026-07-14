# Tiny Vision Transformer

Reproduction of Tiny Vision Transformer for silicon wafer defect detection.

Reference: *Semiconductor Wafer Map Defect Classification with Tiny Vision Transformers*

## Status

Implementation in progress.

## Configuration

See `configs/config.yaml` for experiment hyperparameters.

## Structure

- `configs/` — YAML experiment configurations
- `models/` — model architecture and weights
- `data_utils/` — dataset loaders and preprocessing
- `utils/` — paper-specific utilities
- `tests/` — unit and integration tests
- `train.py` — training entry point
- `evaluate.py` — evaluation entry point
- `predict.py` — inference entry point
