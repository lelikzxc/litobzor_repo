# CTM-YOLOv10

Reproduction of CTM-YOLOv10 for silicon wafer defect detection.

Base architecture: YOLOv10.

## Status

Implementation in progress.

## Configuration

See `configs/config.yaml` for experiment hyperparameters.

## Structure

- `configs/` — YAML experiment configurations
- `models/` — model architecture and weights
- `modules/` — reusable building blocks (CTM, C2f, SCDown, etc.)
- `data_utils/` — dataset loaders and preprocessing
- `utils/` — paper-specific utilities
- `tests/` — unit and integration tests
