# Litobzor

Research repository for reproducing Computer Vision models from scientific papers for **silicon wafer defect detection**.

---

## Description

**Litobzor** is a modular research codebase designed to systematically reproduce, compare, and extend state-of-the-art deep learning methods for automated defect inspection on semiconductor wafers. The project provides a unified infrastructure — shared utilities, metrics, datasets, and experiment configs — while keeping each paper's implementation isolated in its own namespace.

The repository supports multiple CV paradigms: classification, object detection, segmentation, and semi-supervised learning.

---

## Goal

Build a **scalable, reproducible research platform** that enables:

- Faithful reproduction of published methods on wafer defect datasets
- Fair comparison across models using shared evaluation protocols
- Incremental addition of new papers without refactoring the core codebase
- Standardized experiment tracking, logging, and configuration management

---

## Project Structure

```
litobzor_repo/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── train.py                  # Global training entry point (stub)
├── evaluate.py               # Global evaluation entry point (stub)
├── predict.py                # Global inference entry point (stub)
│
├── configs/                  # Global default configurations
│   └── default.yaml
│
├── common/                   # Shared modules across all papers
│   ├── utils/                # logger, seed, config, paths
│   ├── metrics/              # evaluation metrics
│   ├── losses/               # shared loss functions
│   ├── datasets/             # base dataset utilities
│   └── visualization/        # plotting and result visualization
│
├── papers/                   # Paper-specific implementations
│   ├── vit_tiny/             # Tiny Vision Transformer
│   ├── ctm_yolov10/          # CTM-YOLOv10
│   ├── vmamba/               # FCS-VMamba
│   ├── transformer_segmentation/  # SegFormer + Atrous
│   └── semiwafernet/         # SemiWaferNet
│
├── scripts/                  # Utility and launch scripts
├── docs/                     # Documentation
└── tests/                    # Project-level tests
```

Each paper directory follows a uniform layout:

```
papers/<model_name>/
├── README.md
├── config.yaml
├── configs/
│   └── config.yaml
├── models/
├── data/
├── utils/
└── tests/
```

---

## Papers

| # | Model | Task | Directory | Status |
|---|-------|------|-----------|--------|
| 1 | Tiny Vision Transformer | Classification | `papers/vit_tiny/` | Planned |
| 2 | CTM-YOLOv10 | Object Detection | `papers/ctm_yolov10/` | Planned |
| 3 | FCS-VMamba | Classification | `papers/vmamba/` | Planned |
| 4 | SegFormer + Atrous | Segmentation | `papers/transformer_segmentation/` | Planned |
| 5 | SemiWaferNet | Semi-supervised Detection | `papers/semiwafernet/` | Planned |

---

## Roadmap

### Phase 1 — Infrastructure (current)

- [x] Project scaffolding and directory structure
- [x] Shared utility modules (`common/`)
- [x] YAML-based configuration system
- [x] Development tooling (pytest, ruff, black, mypy)
- [x] Paper template directories

### Phase 2 — Core Modules

- [ ] Implement `common/utils/logger.py`
- [ ] Implement `common/utils/config.py` (YAML loading)
- [ ] Implement `common/utils/seed.py` (reproducibility)
- [ ] Implement `common/utils/paths.py` (path resolution)
- [ ] Implement `common/metrics/metrics.py`
- [ ] Base dataset and dataloader abstractions
- [ ] Shared visualization utilities

### Phase 3 — Paper Reproductions

- [ ] Tiny Vision Transformer
- [ ] CTM-YOLOv10
- [ ] FCS-VMamba
- [ ] SegFormer + Atrous
- [ ] SemiWaferNet

### Phase 4 — Benchmarking & Documentation

- [ ] Unified evaluation pipeline
- [ ] Cross-model comparison reports
- [ ] Reproduction guides per paper
- [ ] CI/CD integration

---

## Installation

**Requirements:** Python 3.11+

```bash
# Clone the repository
git clone https://github.com/niime/litobzor_repo.git
cd litobzor_repo

# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install from requirements.txt
pip install -r requirements.txt
```

---

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Lint
ruff check .

# Format
black .

# Type checking
mypy common papers
```

### Entry Points

Global scripts are provided as stubs and will be wired to paper-specific implementations:

```bash
python train.py       # Training
python evaluate.py    # Evaluation
python predict.py     # Inference
```

### Adding a New Paper

1. Copy an existing paper directory under `papers/`.
2. Update `README.md` and `configs/config.yaml`.
3. Implement modules in `models/`, `data/`, and `utils/`.
4. Add tests under `tests/`.

### Configuration

Global defaults live in `configs/default.yaml`. Each paper overrides them in `papers/<model_name>/configs/config.yaml`.

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Litobzor Team
