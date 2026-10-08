# FCS-VMamba MPS diagnostics, 2026-10-08

Measured locally on Apple M3 / 8 GB with PyTorch 2.13.0. These short runs are
learning and execution diagnostics, not paper reproduction. Full recipe
`configs/config.yaml` remains unchanged.

## Implementation and validation

Train/evaluate now accept MPS. Selective scan on MPS uses checkpointed affine
prefix blocks instead of one Python iteration per token. Multiplication/addition
implement the reference recurrence without divisions by tiny transition
products. Each boundary state is cloned into a small independent allocation;
a view would otherwise keep the whole block state alive. CUDA dispatch remains
unchanged. FrequencyAttention runs its FFT directly on MPS in this environment.

All 15 scan tests passed in the MPS-enabled environment: forward and all seven input gradients
versus the CPU recurrence, 3136-token sequences, extreme +/-200 deltas, 128/512
prefix blocks, retained-storage bounds and FFT gradients. Existing model,
training and engine tests: 98 passed. A real `train.py` run (36 train / 9 holdout,
one epoch, full FCS architecture at 64 pixels) wrote best/last checkpoints and
protocol; standalone `evaluate.py` reproduced its loss/metrics. That nine-image
smoke is not a quality estimate.

## Learning diagnostic

Author archive: 902 RGB JPEGs, each originally 32x32. Seed 42 creates the
original 721/181 split. Only 16 examples per class (144 total), selected from
the training split, were used; all 181 holdout images stayed outside training.
Inputs were resized to 64 rather than the paper's 224 pixels. This changes
SSM sequence lengths and network spatial scales, although the source image
contains no additional pixels at 224. All FA/SFS/CLCA modules, eight blocks,
width 96 and 1,082,697 parameters were retained. Paper-style augmentation,
AdamW lr 0.001, A/D no-decay groups and the first part of a 50-epoch cosine
schedule were retained.

| Full FCS at 64 pixels | Untrained | After short training |
|---|---:|---:|
| Holdout cross entropy | 2.2240 | 1.5312 |
| Holdout accuracy | 11.05% | 43.65% |
| Holdout macro-F1 | 3.24% | 33.67% |

Training ran for 170.45 seconds / 156 batches of four: four complete epochs
plus 48 examples in epoch five. Initial/final holdout evaluation increased
total measured runtime to 197.11 seconds. Warm steps averaged 1.064 seconds.
Gradients remained finite. None, Donut, Edge-Ring, Near-full and Random began
learning; Center, Edge-Loc, Local and Scratch still had zero final recall.
Observed driver allocation after steps was approximately 1.28 GB, not a
measurement of peak memory.

## Original-resolution timing and limits

Full FCS at 224 pixels, batch one, 128-token blocks: the initial cold step took
8.82 seconds. A subsequent three-step profile measured 3.778 / 3.304 / 3.286
seconds, with a post-first mean of 3.295 seconds and approximately 0.679 GB
observed driver allocation. This does not predict the paper's batch-four
throughput or a complete run. A separate two-step profile at the paper's batch
size of four measured 14.954 / 12.970 seconds, finite gradients, and approximately
3.44 GB observed driver allocation. The warm second batch projects to about
32.6 training hours for 50 epochs at 721 training examples; this is a rough
two-step projection, excluding evaluation/checkpoint/data overhead, not a
measured complete run. Full-resolution MPS training is therefore substantially
slower than these reduced diagnostics. Increasing blocks to 512 was slower (3.853 seconds
warm) and used approximately 2.38 GB driver allocation; default blocks remain
128.

Paper Subset A targets are 87.91% accuracy and 86.06% macro-F1 (Table 3), with
85.91 +/- 1.65% macro-F1 across five runs (Table 4). The diagnostic shows
nontrivial early learning and removes MPS execution blockers. It does not yet
show proximity to those targets. Full training on 721 examples for 50 epochs,
with the original 224-pixel inputs and multiple seeds, is needed. Unpublished
normalization, augmentation and optimizer choices and the reference-code
limitations in `architecture_audit.md` remain reproduction uncertainties.

Raw reports and checkpoints are under
`checkpoints/mps_diagnostics/vmamba/{full64,profile224,profile224_warm128,profile224_warm512,profile224_batch4,entrypoint}`.
