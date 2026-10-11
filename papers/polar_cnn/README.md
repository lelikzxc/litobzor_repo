# Polar CNN (Kim & Kim, 2024)

Implementation of **Model 4**, the single polar-sector CNN from
[Development of a Wafer Defect Pattern Classifier Using Polar Coordinate System
Transformed Inputs and Convolutional Neural Networks](https://doi.org/10.3390/electronics13071360),
Electronics 13(7), 1360, published 4 April 2024.

**Project status: accepted by the user on 2026-10-11.** The final MPS refit
result is sufficient for this project: **accuracy 88.66%, macro F1 0.83598**
on 7,656 test wafers. Further tuning is not required in the current scope.
This is an accepted experimental result, not a claim of exact author
reproduction or achievement of the paper's rounded all-size macro F1 0.87.

The [acceptance record](results/accepted_mps_20261011.json) stores metrics,
protocol and checksums. A separate archive preserves weights, split, effective
config, training history/log and the implementation source snapshot in
`checkpoints/polar_cnn_accepted_mps_20261011/`. Its checkpoint is `final.pt`.
To evaluate the accepted weights:

```bash
python papers/polar_cnn/evaluate.py --checkpoint checkpoints/polar_cnn_accepted_mps_20261011/final.pt --device mps --output-dir outputs/polar_cnn_accepted_evaluation
```

Use `--device cuda` on Windows and `--data-root` if the dataset is relocated.

## What is implemented

- Eight defect classes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full,
  Random, Scratch. None and unlabelled wafers are excluded.
- Decode categorical PNGs before resizing maps to 26×26 with nearest interpolation.
- Simultaneous 3×3 denoising: a defect with fewer than three defective dies in
  its neighbourhood (including itself) becomes a passed die.
- Defect counts in 49 polar sectors: radial boundaries 0,2,…,14 and angular
  boundaries 0,50,…,300,360. Radius is the row dimension, CCW angle the column.
  These are counts, not an interpolated polar photograph or a density map.
- Table 4 CNN: Conv 32 → MaxPool → Conv 64 → MaxPool → Conv 64 → Flatten
  → Dense 64 → Dense 8, valid 3×3 convolutions and ReLU hidden activations.
  **93,192 parameters**. Cross entropy consumes logits; no extra softmax.
- Keras-like Glorot-uniform/zero-bias initialization, Keras-formula Adam lr=0.01,
  betas=(0.9,0.999), epsilon=1e-7, no weight decay, batch 256, 300 epochs.
- FP32 training on CPU, CUDA and Apple MPS. No custom CUDA extension required.
- Saved original-wafer split and PNG-content fingerprint, resumable optimiser
  and RNG states, sample-weighted loss, macro metrics, per-class report and
  confusion matrix. Invalid loss/gradients stop the run instead of being masked.
- Terminal progress uses tqdm. Redirected output contains one line per epoch;
  `train.log` is written automatically without piping through Tee-Object.

## Limits of an exact paper reproduction

The paper does not supply a complete executable input pipeline or exact split IDs.
See the [source audit](reproduction_audit.md) for the follow-up review and confirmed fixes.
The following choices are **explicit local assumptions**, not recovered author settings:

1. Section 3.3 produces a 7×7 histogram, but Table 4 requires 26×26 inputs.
   Its valid convolutions/pooling cannot operate directly on 7×7. Our configurable
   `model.input_adapter` enlarges counts to 26×26, default `nearest`; `bilinear`
   is available for an explicitly separate experiment. No count normalisation is
   applied. This bridge is not specified by the authors.
2. Table 4 prints a three-channel input, but its first Conv parameter count is
   320, which corresponds to **one** channel. We follow the stated parameter
   counts. Three channels would give 896 parameters in that layer.
3. Polar origin is (12.5,12.5), positive y upwards. The last angular sector is
   60° wide to cover the entire circle. Points outside radius 14 are discarded.
   Exact boundary/origin conventions and interpolation are not published.
4. The principal 2000-example experiment uses manual removal of ambiguous maps,
   **12 synthetic Donut maps**, and rotations. The unmodified native 26×26 subset
   has only **one Donut wafer**. The paper describes augmentation before a 7:3
   split, allowing related rotations to enter both train and test. We split
   original wafers first and augment **only training**, so these high metrics
   cannot be treated as directly comparable independent-test targets.
5. Default `all_maps` uses all 25,519 labelled defect maps, without manual
   cleaning or synthesis. Paper Table 10 reports **89.89% accuracy** for Model 4;
   its small curated experiment reports **91.3%**. The **94%** result belongs to
   Model 5's binary tree, which is not implemented here.
6. The paper specifies 7:3 splitting and three-fold CV without defining how they
   combine. Our default has 30% test, with 10% of the remaining development
   data reserved for validation: **63/7/30 train/val/test**. Best checkpoint is
   selected by validation loss, never test metrics. `last.pt` also preserves the
   fixed final-epoch model. `--fold 0/1/2` instead uses genuine three-fold CV,
   with one third held out and an inner validation split; this is a separately
   documented interpretation. None of these choices guarantees paper metrics.

`config_small.yaml` balances only training to 175 examples/class, using 15°
rotations of training originals when needed. Validation/test stay real and
natural. It is a fast diagnostic, **not** the authors' curated 2000-example set.
`data.selection: native26` and a `reviewed_ids` JSON list of original filenames
are supported for inspecting a curated subset. Split creation refuses classes
with fewer than ten originals; synthetic rotations do not solve this restriction.

## F1 diagnostic study

Completed MPS study (2026-10-11): baseline test macro F1 **0.72707**, selected
validation-checkpoint test macro F1 **0.82540**, and frozen final refit test
macro F1 **0.83598**. The paper's rounded all-size macro F1 **0.87** was **not
reached**. Full results and protocol limitations are in
[`results/f1_study_20261011.json`](results/f1_study_20261011.json).

The final refit uses all 70% development originals, frozen validation-selected
settings and **30 epochs**, not 300. Its config is `configs/config_f1_refit.yaml`;
the saved Mac checkpoint is `checkpoints/polar_cnn_f1_refit_seed42/final.pt`.
It is selected by the predeclared refit procedure, not by comparing test scores.
To reproduce in a fresh directory:

```bash
python -u papers/polar_cnn/train.py --config papers/polar_cnn/configs/config_f1_refit.yaml --device mps --save-dir checkpoints/polar_cnn_f1_refit_repeat
```

Replace `mps` with `cuda` on Windows. More identical epochs are not a supported
remedy: the 300-epoch validation run's best macro F1 occurred at epoch 30.

The validation-selected settings are in `configs/config_f1_optimized.yaml`.
They support the regular train/evaluate CLI on MPS and CUDA, and use a separate
checkpoint directory. They are an experimental improvement, not a recovered
author implementation.

```bash
python -u papers/polar_cnn/train.py --config papers/polar_cnn/configs/config_f1_optimized.yaml --device mps
```

`experiments.py` compares eighteen variants for 40 epochs each using the same
original-wafer 56/14/30 split. The historical 30% test IDs are preserved. Only
train/validation maps are decoded during screening. The winner is selected by
validation macro F1 and retrained for 300 epochs; its best validation-F1
checkpoint is evaluated on test once at the end. Existing baseline results
are retained. The test was previously inspected, so this is not a new blind
benchmark or an exact author-protocol reproduction.

```bash
python -u papers/polar_cnn/experiments.py --device mps --output-dir checkpoints/polar_cnn_f1_study_new
```

CUDA uses the same command with `--device cuda`. Optional config keys:
`data.denoise_order: before_resize|after_resize`,
`data.count_transform: raw|log1p`, `training.balanced_sampling: true|false`,
and `training.selection_metric: loss|f1`. Defaults preserve the baseline.
`data.map_resolution: native_scaled` is a diagnostic alternative that counts
native defects after filtering, maps their coordinates to the 26x26 domain,
and weights counts by 676/(native height * native width). It preserves thin
lines lost by nearest resizing and requires `denoise_order: before_resize`.
The baseline remains `map_resolution: resized26`.
Balanced sampling uses only training labels, keeps the original epoch size,
and never changes evaluation distributions. Saved configs/manifests carry
the preprocessing choices so checkpoint evaluation uses the same transforms.

These settings are experimental changes to improve generalisation, not
assertions that the authors used them.
The first ten variants were screened first; three native-filter combinations
and two native-sector variants were added based on validation evidence, followed
by three variants with training-only rotations to a minimum of 1,500 records per
class. `data.train_min_per_class` retains all majority-class originals, unlike
the separate `train_per_class` balancing option which can subsample them.
Thus the whole search is adaptive, not a preregistered eighteen-way comparison.
`--resume-study` reuses completed comparisons and resumes a partial variant
from `last.pt`; changed settings are rejected rather than overwritten.

## Dataset layout

Use the existing repository dataset, not `datasets/vmamba_author`:

```text
datasets/wm811k/
  labels.csv        # filename,failureType (other metadata columns allowed)
  images/*.png     # original native-size categorical wafer maps
```

Supported levels: 0/1/2, 0/127/254 and 0/128/255. Invalid levels and missing
images fail before model training. Paths in configs are relative to the repo,
so entry points also work when launched from another working directory.
Dependencies are already in the repository requirements.

## Verify preparation without training

```bash
python papers/polar_cnn/train.py --device mps --prepare-only
```

This checks and decodes all selected train/val/test maps, without creating a
checkpoint or fitting a model. A bounded preparation diagnostic can use
`--max-per-class 32`; this option is never enabled by default.

## Training after approval

Interpretation of the **all-size paper experiment** (70% train / 30% test, no
validation selection, fixed 300 epochs, final checkpoint):

The comparison table lists 70/30, while Sec.4.1.3 describes applying the trained
small-data model to all-size maps without further training. Thus this config
is not a uniquely recovered author protocol; see the audit.

```bash
python -u papers/polar_cnn/train.py --config papers/polar_cnn/configs/config_paper_all_maps.yaml --device mps
python papers/polar_cnn/evaluate.py --checkpoint checkpoints/polar_cnn_paper_all_maps_seed42/final.pt --device mps
```

The same commands support `--device cuda` on Windows. Resume with the same
`--config` and `--resume`. The undocumented input bridge still prevents claiming
an exact recovered author implementation. In this mode test is accessed after
fitting, and no `best.pt` is produced. The commands below use the separately
documented validation-based diagnostic protocol.

Apple GPU:

```bash
python -u papers/polar_cnn/train.py --device mps
```

Windows PowerShell, CUDA:

```powershell
$runDir = "checkpoints/polar_cnn_cuda_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
python -u papers/polar_cnn/train.py --device cuda --save-dir $runDir
python -u papers/polar_cnn/evaluate.py --checkpoint "$runDir/best.pt" --device cuda
```

The default is 300 epochs. Use `--epochs` to override the **total** target,
`--batch-size` for the batch size, and `--num-workers` for workers (default 0).
No GPU request silently falls back to CPU. Root dispatchers work too:
`python train.py polar_cnn --device mps`.

Resume an interrupted run with the same config/seed/batch size:

```bash
python papers/polar_cnn/train.py --device mps --resume
```

For a custom CUDA directory add `--save-dir $runDir --resume`. For a small or
otherwise custom config supply the same `--config` used initially. Existing
nonempty directories are never overwritten by a fresh run. A changed model,
preprocessing, seed, batch size, LR or dataset is rejected on resume.

Moving checkpoints between operating systems:

```powershell
python papers/polar_cnn/evaluate.py --checkpoint "$runDir/best.pt" --device cuda --data-root datasets/wm811k
```

Evaluation reads configuration and test IDs from the checkpoint. Results include
accuracy, **macro** F1/precision/recall, weighted F1, per-class supports and an
8×8 confusion matrix. Metrics are stored in JSON and the matrix in CSV.
When comparing adapters or hyperparameters, choose them with validation only.

## Tests

Real-wafer one-batch preflight (does not save a training run):

```bash
python papers/polar_cnn/verify.py --device mps
```

On the Windows CUDA machine use `--device cuda`. It checks forward agreement
with CPU, finite backward/Adam updates, and a checkpoint roundtrip with a real
256-map batch. It does not estimate final metrics or full-training duration.

Local verification on 2026-10-11: all 25,519 real defect maps decoded; 26 paper
tests passed on CPU/MPS and two CUDA tests skipped; 52 shared tests passed.
The real MPS batch/checkpoint report is saved in
[`results/mps_preflight_20261011.json`](results/mps_preflight_20261011.json).
CPU and MPS logits agree to about 4.2e-6. Weight gradients agree to about 3.8e-7;
bias gradients can differ at ReLU boundaries because sparse input regions yield
small positive CPU convolution residuals (up to about 1.1e-5) versus exact MPS
zeros. This is recorded rather than presented as exact numerical equivalence.

```bash
python -m pytest papers/polar_cnn/tests -q
```

Hardware tests run real forward/backward/Adam operations on each available
backend. Missing GPU backends are explicitly skipped. Spawn-worker tests cover
the multiprocessing mode used on Windows. CUDA hardware cannot be validated on
this Mac. A CPU/MPS test pass proves execution, not publication-level accuracy.
