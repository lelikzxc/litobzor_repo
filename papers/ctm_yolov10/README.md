# CTM-YOLOv10

Reproduction of CTM-YOLOv10 for silicon wafer defect detection.

Base architecture: YOLOv10 (Real-Time End-to-End Object Detection).

Reference paper: *Wafer Defect Detection Technology Based on CTM-IYOLOv10 Network*

## Status

YOLOv10 baseline implemented. CTM integration pending.

## Architecture

The model uses the official Ultralytics YOLOv10 detection model as its backbone.
The current implementation wraps `YOLOv10Baseline` as a `torch.nn.Module`:

- Backbone: YOLOv10 (CSPDarknet + SPPF + PSA)
- Neck: PAN-FPN with C2f and SCDown
- Head: v10Detect (one-to-many + one-to-one label assignment)

### Future CTM Integration

The `CTM` (Context Transformer Module) placeholder is located in
`modules/ctm.py`. It will be inserted into the backbone in the next stage
to enhance feature extraction for wafer defect detection.

## Configuration

See `configs/config.yaml` for experiment hyperparameters.

## Structure

- `configs/` — YAML experiment configurations
- `models/` — model architecture and weights
- `modules/` — reusable building blocks (CTM, C2f, SCDown, etc.)
- `data_utils/` — dataset loaders and preprocessing
- `utils/` — paper-specific utilities
- `tests/` — unit and integration tests

## Demo

Run the baseline demo:

```bash
python papers/ctm_yolov10/demo.py
