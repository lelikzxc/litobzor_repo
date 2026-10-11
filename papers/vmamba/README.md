# FCS-VMamba reproduction

Reference: **Wafer Defect Recognition for Industrial Inspection: FCS-VMamba Model and Experimental Validation**, J. Imaging 2026, 12, 142, DOI https://doi.org/10.3390/jimaging12040142 (`vmamba.pdf`).

The default recipe trains the full FCS model, using the **902 RGB JPEGs published by the authors**, rather than a newly sampled collection of categorical WM-811K maps. Full GPU training is still required to measure recognition quality. CPU smoke tests verify execution, gradients, checkpoints and evaluation; they do not establish the paper's accuracy.

## Native Windows CUDA / PowerShell

Run from the repository root in the environment that already trained SemiWaferNet.
The required packages are in the project requirements. VMamba needs a CUDA-enabled
PyTorch/torchvision pair; native Windows does **not** need a selective-scan extension,
Triton, a C++ compiler or a separately installed CUDA toolkit for the commands below.

```powershell
python papers/vmamba/prepare_data.py
python -u papers/vmamba/train.py --config papers/vmamba/configs/config.yaml --device cuda --scan-backend chunked --check-only
```

The preflight decodes all 902 images, constructs the complete 1,082,697-parameter
FCS model, runs a training update with batch 4 and an evaluation batch of 16 at
224 × 224, and checks losses, all gradients and updated parameters for finite
values. On CUDA it reports update time and peak allocated/reserved GPU memory.
It writes no checkpoints or protocol. Run the full recipe after both batch checks
finish successfully:

```powershell
$runDir = "checkpoints\vmamba_cuda_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
python -u papers/vmamba/train.py --config papers/vmamba/configs/config.yaml --device cuda --scan-backend chunked --save-dir $runDir --log-file "$runDir\train.log"
python -u papers/vmamba/evaluate.py --checkpoint "$runDir\best.pt" --device cuda --scan-backend chunked --log-file "$runDir\metrics.log"
```

This runs all 50 epochs on the author split (721 training images, 181 holdout
images), with the published learning rate 0.001 and batch size 4. The chunked
scan implements the same SSM recurrence with checkpointed 128-token affine-prefix
blocks and ordinary PyTorch CUDA operations. Tests compare outputs and every input
gradient against the sequential reference, including 3136-token sequences and
transition underflow. Actual Windows/CUDA speed and memory must be measured by the
preflight; this Mac has no CUDA device. Execution checks do not guarantee the
paper's recognition metrics.

In an interactive terminal tqdm redraws a single ASCII progress line, with one
metrics line per epoch. Redirected output disables progress redraws. The internal
UTF-8 log saves ordinary output and exception tracebacks without progress updates,
so no PowerShell Tee-Object pipeline is needed.

The default `--scan-backend auto` prefers an installed compiled CUDA kernel and
otherwise chooses the same chunked implementation. `--scan-backend compiled`
requires a working `selective_scan_cuda_oflex`, `selective_scan_cuda_core` or
`selective_scan_cuda` extension built for the exact PyTorch/CUDA environment. For
an optional compiled implementation, follow the official
[VMamba selective-scan build instructions](https://github.com/MzeroMiko/VMamba/tree/main/kernels/selective_scan)
in Linux/WSL2. Cross-scan/merge use PyTorch autograd on all devices.

Local verification on 2026-10-11: 426 tests passed; eight hardware/sandbox tests
were skipped (no accessible CUDA/MPS and restricted shared-memory manager).
The complete author-archive preflight passed at 224 pixels with train/evaluation
batches 4/16. An interrupted short training run resumed correctly and produced
the same classification metrics as standalone evaluation. CUDA execution and
timing remain to be checked on the target Windows computer.

`prepare_data.py` downloads ~1 MB from a pinned author revision, verifies SHA256, and writes `datasets/vmamba_author/WM811k_Dataset.zip`. This archive has already been prepared on the current machine; copy it to the GPU device or rerun the preparation command. A previously downloaded archive can be supplied with `--archive /path/to/WM811k_Dataset.zip`.

`protocol.json` beside the checkpoints records the effective configuration (including CLI overrides), seed, sample identity/order and exact train/holdout indices. Evaluation reads this configuration by default, uses the saved indices and rejects a changed sample list. An optional `--config` must agree with the model, image source and normalization. The author archive checksum and image decoding are checked before model allocation. The 20% holdout is the paper's validation set, used for checkpoint selection and final reporting; it is not an independent test set.

Resume with `--resume` uses `last.pt` and trains the remaining epochs up to `training.num_epochs`. The checkpoint's epoch, optimizer, cosine scheduler and historical best selection are restored. `last.pt` is not overwritten with best weights during final evaluation. Keep the same configuration and flags when resuming:

```powershell
python -u papers/vmamba/train.py --config papers/vmamba/configs/config.yaml --device cuda --scan-backend chunked --save-dir $runDir --log-file "$runDir\train.log" --resume
```

If opening a new terminal, assign `$runDir` to the existing run directory first.
Use a new `--save-dir` for a different experiment. Previous backbone-only
checkpoints are not architecture-compatible, and a changed protocol is rejected.

## Ablation and raw-map experiments

`configs/config_backbone_only.yaml` preserves the previous raw-map backbone experiment; it is **not the full FCS reproduction**. For controlled FCS ablations, copy the main configuration, toggle `model.fa/sfs/clca.enabled`, and assign a separate checkpoint directory while keeping the same source and seed. `--seed` controls subset sampling, split and weight initialisation.

For raw categorical maps, set `data.source: raw_maps`, `data.data_root: datasets/wm811k`, and `data.per_class: 100`. This is a distinct 900-map experiment, not the author benchmark. Set `--per-class 0` to use all maps. Subset B (11,000 long-tailed samples) is not silently approximated by the main recipe.

CPU execution needs no CUDA kernel. `--scan-backend chunked` also works on CPU for
diagnostics. Legacy `--allow-slow-scan` forces the sequential reference on every
device; it is useful for debugging, but is slow on CUDA.

## Short diagnostics on Apple Silicon

Training and evaluation accept `--device mps`; `auto` prefers CUDA, then MPS,
then CPU. MPS uses an affine-prefix selective scan with checkpointed 128-token
blocks, preserving the SSM recurrence while reducing Python/GPU launch overhead
and bounding backward state memory. The same method is the native CUDA fallback;
installed compiled CUDA kernels remain available. CPU tests compare
outputs and all parameter gradients against the recurrence, including 3136-token
sequences and transition underflow. FFT support depends on the installed PyTorch;
the tested environment runs FrequencyAttention directly on MPS.

```bash
python papers/vmamba/scripts/mps_diagnostic.py --device mps --image-size 224 --batch-size 1 --profile-only --output checkpoints/mps_diagnostics/vmamba/profile224
python papers/vmamba/scripts/mps_diagnostic.py --device mps --image-size 64 --max-seconds 180 --output checkpoints/mps_diagnostics/vmamba/full64
```

The second command keeps the full FCS architecture but uses 64-pixel inputs and
16 training images per class selected only from the author benchmark's training
split. It evaluates on the complete, disjoint author holdout, saves explicit
diagnostic metadata, and stops training after a bounded time (checked between
batches). Initial/final evaluation add a little time to that budget. The results
measure early learning; reduced resolution/data and few epochs cannot establish
paper reproduction. `--variant no_sfs` or `backbone` creates explicit ablations;
use a separate `--output` for each. Paper defaults remain in `configs/config.yaml`.

The [measured MPS diagnostic results](docs/mps_diagnostic_results.md) record
the short 64-pixel run, native FFT/scan gradient checks, and warm 224-pixel
timings. They document the limits of interpreting early accuracy as evidence
for eventual paper reproduction.

See [architecture_audit.md](docs/architecture_audit.md) for the failure analysis, implemented corrections and remaining publication ambiguities.
