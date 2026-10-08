# Radon-CNN on WM-811K

Reference: [Jeong et al., Scientific Reports 2023](https://www.nature.com/articles/s41598-023-34147-2).

## Run

Install the repository dependencies (`pip install -r requirements.txt`).
Run from the repository root:

```bash
python papers/radon_cnn/train.py --device cuda
python papers/radon_cnn/evaluate.py --checkpoint checkpoints/radon_cnn_v2/best.pt --device cuda
```

On Apple Silicon, use `--device mps`. `auto` selects CUDA, then MPS, then CPU.
MPS training uses FP32; CUDA AMP remains optional.

For a short learning diagnostic on the unchanged architecture:

```bash
python papers/radon_cnn/diagnose_mps.py --device mps
```

It uses 128 distinct training maps and 32 validation/test maps per defect class,
with a 180-second training budget and a separate 30-second memorization check.
CPU Radon preprocessing is cached once. Checkpoints, exact wafer splits and
per-class metrics go into `checkpoints/mps_diagnostics/radon`. Give another
run a fresh `--output-dir`; `--epochs`, `--max-seconds`, and sample caps are
configurable. This reduced, single-seed diagnostic estimates whether learning
works; its metrics do not establish reproduction of the article's results.

Measured diagnostic results on the local M3 Mac (8 GiB, MPS, FP32), seed 42:

| Distinct training wafers | Validation accuracy | Test accuracy | Test macro-F1 | Training time |
| --- | --- | --- | --- | --- |
| 896 (128 per class) | 79.91% | 74.55% | 74.06% | 39.33 s |
| 1792 (256 per class) | 82.59% | 80.36% | 79.86% | 68.89 s |

Both runs kept the original 1,462,327-parameter architecture, Adam 3e-4 and
gamma=0.99. Each ran 30 epochs; the minimum validation loss selected epoch 9
and epoch 8 respectively. Validation and test contain the **same 224 distinct
wafers each** in both runs (32 per class), without train/validation/test leakage.
The larger run reused the held-out indices explicitly:

```bash
python papers/radon_cnn/diagnose_mps.py --device mps --train-per-class 256 --output-dir checkpoints/mps_diagnostics/radon_1792 --holdout-from checkpoints/mps_diagnostics/radon/split.json --max-seconds 120
```

Initial CPU Radon preparation took 6.64 s for the smaller run. Its cached
training/evaluation/sanity run took 44.22 s; the larger run took 82.63 s including
10.77 s of preprocessing. The selected checkpoints achieve 100% accuracy on
their training sets, so extra epochs alone do not resolve the generalization
gap. More distinct maps helped, while Loc remains weak (43.75% test recall in
the larger run).

[Article Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10199043/) reports
85.83 +/- 0.82% with 800 training maps, 87.97 +/- 1.01% with 1600, and
90.84 +/- 0.81% with 6400, averaged over 20 seeds. These short local runs do
not reproduce those values; their small holdouts and interpreted split protocol
also prevent a direct equivalence. Full training settings were not changed
based on this diagnostic. Exact splits, per-class metrics and epoch histories
are saved in each diagnostic directory's `split.json` and `report.json`.

The actual `train.py` and standalone `evaluate.py` entry points were also
checked with `--device mps`, using one epoch, 224 training maps, and 56 maps
in each holdout. Training took 4.40 s including process startup, and standalone
evaluation took 2.08 s. Both reported the same test accuracy (21.43%) after
only three optimizer steps. This is an execution/checkpoint check, not a quality
experiment. Logs and the effective configuration are saved in
`checkpoints/mps_diagnostics/radon_entrypoint/{train.log,evaluate.log,experiment.yaml}`.

## Completed full balanced MPS runs

**Project status: accepted by the user on 2026-10-08.** These three runs are
the recorded final RadonCNN result for the current project scope. Further
RadonCNN tuning is not required for that scope. Exact values and checkpoint
checksums are preserved in the
[acceptance record](results/accepted_mps_20261008.json); the comparison
limitations below remain part of the record.

On 2026-10-08 the unchanged `config_balanced.yaml` recipe completed on the
local M3 / 8 GB with seeds 42, 43 and 44, using the corrected sample-weighted
validation loss. Each run used 6400 training presentations from 5710 distinct
wafers, plus 385 distinct validation and 385 distinct test wafers. Rare-class
repeats occur only in training. The seed changes both initialization and split.

| Seed | Best epoch | Stopped epoch | Test accuracy | Test macro-F1 | Training process |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 32 | 62 | 86.49% | 86.37% | 10.46 min |
| 43 | 6 | 36 | 87.01% | 86.85% | 5.99 min |
| 44 | 6 | 36 | 87.79% | 87.92% | 6.33 min |

All three seeds are included: mean accuracy **87.10% +/- 0.65 percentage
points**, mean macro-F1 **87.05% +/- 0.80 percentage points** (sample standard
deviation, not a confidence interval). Checkpoints minimize validation loss;
test metrics did not select epochs, hyperparameters or seeds. The maximum was
500 epochs with patience 30; all runs stopped early. Independent CPU evaluation
matched the rounded standalone MPS metrics and each checkpoint's logged
sample-weighted validation loss.

Mean test recall remains weakest for Loc (68.48%) and Scratch (76.97%). These
runs establish that full RadonCNN training is practical on this Mac. They do
not reproduce Table 3's 90.84% mean accuracy: only three seeds were run, and
the author split/projection/balancing choices remain unrecovered.

Reports, all per-class metrics, prediction CSVs, exact splits, checkpoints and
logs are in `checkpoints/radon_cnn_mps_full_20261008/`. See the
[full report](../../checkpoints/radon_cnn_mps_full_20261008/summary.md),
[machine-readable audit](../../checkpoints/radon_cnn_mps_full_20261008/summary.json)
and [training curves](../../checkpoints/radon_cnn_mps_full_20261008/training_curves.png).
The interrupted run preceding the validation-loss correction was stopped
before test evaluation and is preserved under `superseded_before_loss_fix/`;
it is excluded from all reported results.

The default config is a **full-data diagnostic**, using all 25,370 maps in
the seven defect classes (excluding None and Near-full). It uses stratified
80/10/10 splitting and a balanced training sampler. Its accuracy is not
directly comparable to the paper's small-data Table 3.

For a balanced small-data experiment:

```bash
python papers/radon_cnn/train.py --config papers/radon_cnn/configs/config_balanced.yaml --device cuda
python papers/radon_cnn/evaluate.py --checkpoint checkpoints/radon_cnn_balanced_seed42/best.pt --device cuda
```

This configuration requests **6400 training presentations** (class counts
differ by at most one). Original wafers are split first. Rare training classes
are explicitly oversampled; validation and test are balanced without duplicates,
55 originals per class for this export and these fractions. No original wafer
crosses a split boundary. Set `allow_train_replacement: false` to forbid repeats;
reduce `train_size` accordingly. The paper does not specify exact split sizes or
how it achieves large balanced subsets with only 555 Donut maps. This configuration
is a documented interpretation, not the authors' recovered protocol. Their
90.84 ± 0.81% is a mean and standard deviation over 20 runs, not a guaranteed
single-run result.

Give each seed its own `checkpoint.save_dir`, and pass `--seed N` for independent
runs. Training refuses to overwrite an existing checkpoint unless `--resume`
is supplied. `split.json` saves indices, class counts, unique counts, seed,
preprocessing settings, library versions, and a hash of the label inventory.
`experiment.yaml` saves the effective config. Keep both files beside checkpoints
when moving to another machine; evaluation checks them rather than inventing a
new test split. The dataset can be relocated using an explicit evaluation config.

## Corrections and assumptions

- CSV columns are read by name: `filename`/`failureType` or `image`/`label`.
  Numeric labels use the repository's nine-class convention (None=0).
- Categorical PNGs in 0/1/2, 0/127/254, and 0/128/255 encodings are decoded.
  Only value 2 (failed dies) survives. Nearest-neighbour resizing to 64×64
  prevents good dies from turning into defects through interpolation.
- Radon inputs use 64 angles over [0, 180), `circle=False`, then resize the
  sinogram to 64×64. These choices and image interpolation are not fully
  specified by the paper. It mentions scikit-image 0.20.0; the installed version
  is recorded for each experiment.
- Kernel Flip uses one shared convolution, shared BatchNorm across the two
  branches, then max-out **after** ReLU, BatchNorm and pooling, following Table 2.
- The 4×4×256 feature map is flattened into FC(256), rather than adding an
  undocumented Global Average Pooling layer. The article does not provide
  author code; kernel sizes (3×3), padding and shared BatchNorm are local choices.
- Adam 3e-4, ExponentialLR gamma=0.99, cross-entropy, and validation-loss early
  stopping with patience 30 follow the article. Batch size 64 and a maximum
  of 500 epochs are local choices. FP32 is the default; AMP is optional.
- `best.pt` and final test evaluation use the minimum validation loss.
  RadonCNN's local `RadonTrainer` averages this loss over individual wafers.
  With 385 validation maps and batch size 64, the final singleton must have
  weight 1/385; averaging seven batch means incorrectly gave it weight 1/7.
  This correction applies to checkpoint selection and early stopping. Four
  regression tests check batch sizes 1, 2+1 and 3 on the same three maps, plus
  rejection of an empty validation loader.
  `--resume` preserves optimizer/scheduler state; patience restarts on resume.
  It does not promise a bitwise-identical continuation of RNG/AMP state.
- AMP gradients are unscaled before clipping in the common Trainer. CPU/FP32
  behavior is unchanged by that correction.
- Radon features are cached lazily in RAM (about 400 MiB for all seven classes),
  and are not recomputed every epoch. Training drops a final incomplete batch
  to avoid singleton batches in the fully-connected BatchNorm layers.

Start a **fresh run**: old weights learned from the incorrect silhouette inputs,
and checkpoints using the old GAP classifier are incompatible with this head.
Successful training and reproduction of paper accuracy still require a full run;
CPU regression tests and a small overfit check establish implementation behavior.
