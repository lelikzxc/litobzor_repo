# FCS-VMamba reproduction

Reference: **Wafer Defect Recognition for Industrial Inspection: FCS-VMamba Model and Experimental Validation**, J. Imaging 2026, 12, 142, DOI https://doi.org/10.3390/jimaging12040142 (`vmamba.pdf`).

The default recipe trains the full FCS model, using the **902 RGB JPEGs published by the authors**, rather than a newly sampled collection of categorical WM-811K maps. Full GPU training is still required to measure recognition quality. CPU smoke tests verify execution, gradients, checkpoints and evaluation; they do not establish the paper's accuracy.

## Run on GPU

From the repository root, install project requirements and a CUDA-compatible PyTorch/torchvision pair. The paper used PyTorch 2.1/CUDA 12.1; SemiWaferNet used PyTorch 2.5.1. CUDA extensions must be built against the actual installed PyTorch/CUDA combination.

Install an accelerated selective-scan implementation from the official [VMamba repository](https://github.com/MzeroMiko/VMamba/tree/main/kernels/selective_scan):

```bash
git clone https://github.com/MzeroMiko/VMamba.git /tmp/VMamba
pip install ninja packaging einops
pip install --no-build-isolation /tmp/VMamba/kernels/selective_scan
```

`selective_scan_cuda_oflex`, `selective_scan_cuda_core` or `selective_scan_cuda` is supported. A CUDA toolkit/compiler is needed to build the extension. Cross-scan/merge use working PyTorch autograd on both CPU and GPU; installing Triton is optional and does not select the old incomplete wrappers.

```bash
python papers/vmamba/prepare_data.py
python -m pytest tests/test_paper_reproduction.py -q
python papers/vmamba/train.py --config papers/vmamba/configs/config.yaml --device cuda
python papers/vmamba/evaluate.py --config papers/vmamba/configs/config.yaml --checkpoint checkpoints/vmamba_fcs_reproduction/best.pt --device cuda
```

`prepare_data.py` downloads ~1 MB from a pinned author revision, verifies SHA256, and writes `datasets/vmamba_author/WM811k_Dataset.zip`. This archive has already been prepared on the current machine; copy it to the GPU device or rerun the preparation command. A previously downloaded archive can be supplied with `--archive /path/to/WM811k_Dataset.zip`.

`protocol.json` beside the checkpoints records configuration, seed, sample identity/order and exact train/holdout indices. Evaluation uses these indices and rejects a changed sample list. The 20% holdout is the paper's validation set, used for checkpoint selection and final reporting; it is not an independent test set.

Resume with `--resume` uses `last.pt` and trains the remaining epochs up to `training.num_epochs`. `last.pt` is not overwritten with best weights during final evaluation. Use a new `checkpoint.save_dir` for a different experiment; old checkpoints are not architecture-compatible.

## Ablation and raw-map experiments

`configs/config_backbone_only.yaml` preserves the previous raw-map backbone experiment; it is **not the full FCS reproduction**. For controlled FCS ablations, copy the main configuration, toggle `model.fa/sfs/clca.enabled`, and assign a separate checkpoint directory while keeping the same source and seed. `--seed` controls subset sampling, split and weight initialisation.

For raw categorical maps, set `data.source: raw_maps`, `data.data_root: datasets/wm811k`, and `data.per_class: 100`. This is a distinct 900-map experiment, not the author benchmark. Set `--per-class 0` to use all maps. Subset B (11,000 long-tailed samples) is not silently approximated by the main recipe.

CPU execution needs no CUDA kernel. GPU training without an accelerated scan is refused by default because the recurrent PyTorch fallback is extremely slow; `--allow-slow-scan` opts into that fallback.

See [architecture_audit.md](docs/architecture_audit.md) for the failure analysis, implemented corrections and remaining publication ambiguities.
