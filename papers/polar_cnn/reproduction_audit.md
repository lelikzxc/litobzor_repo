# Audit of exact reproduction, 2026-10-11

Sources examined: the publisher PDF (`paper.pdf`), publisher JATS XML,
[article page](https://www.mdpi.com/2079-9292/13/7/1360), Figure 7, Tables 1–4,
Section 4.1.3, and data availability statement. Exact-title/DOI/author searches
did not identify an author implementation or processed dataset. The publisher
XML has no `supplementary-material` entries. This does not prove that no code
exists anywhere, but none was recovered in this audit.

## Confirmed settings and changes

The model channels, kernels, strides, pooling, dense dimensions, denoising,
polar-sector counts, batch size, learning rate and 300-epoch schedule match
the published descriptions. Default Keras Glorot initialization is used.
The dense Adam update now follows the **epsilon-hat** convention in the
[Keras 2.15 implementation](https://github.com/keras-team/keras/blob/v2.15.0/keras/optimizers/adam.py).
The earlier PyTorch Adam with epsilon=1e-7 was not algebraically identical.
This fixes that difference; it does not establish identical floating-point
arithmetic, RNG streams or unpublished author optimizer overrides.

`configs/config_paper_all_maps.yaml` follows the **70/30 train/test** ratio
listed in the comparison table, 300 fixed epochs and natural class distribution.
However, Sec.4.1.3 describes applying the small-data classifier to all-size maps
without additional training. These descriptions do not establish a unique
all-size training/evaluation protocol. Our all-size training is a documented
interpretation, not a recovered author experiment.
It has **no validation partition and no best-epoch selection**;
the final model is written to `final.pt`, test is evaluated after fitting only.
The earlier 63/7/30 validation-based experiment is retained as a separate
diagnostic config, not silently presented as the published protocol.

## Information still unavailable

| Detail | Evidence | Why it prevents an exact claim |
|---|---|---|
| Input geometry | Sec.3.3: 7×7 counts; Table 4 / Fig.7: valid CNN for 26×26 | Resampling/rendering/padding into CNN input is unstated. Nearest enlargement remains a local hypothesis. |
| Input channels | Table 4 says 3 but first Conv has 320 parameters | One channel reproduces parameter counts; three reproduces the printed input shape. Both cannot be true for that Conv. |
| Small-data curation | Tables 1–2: manually removed maps; 12 synthetic Donut maps added | Exact retained IDs and synthetic maps/generation parameters are not supplied. Matching counts would not recover the same data. |
| Train/test identities | Paper gives ratios, not IDs/seeds | Different splits can change reported accuracy; seed 42 is a local choice. |
| Three-fold CV | Paper also specifies 7:3 splitting | Exact relation between the two procedures is unstated. True 3-fold test partitions are 1/3, not 30%. |
| Coordinates / resampling | Radial/angle intervals are described | Origin, boundary inclusion, resizing interpolation and count-to-CNN bridge are not fully specified. |

The data availability statement points to the original public WM-811K dataset,
not the manually curated and augmented experimental dataset. Choosing maps
after looking at test predictions, inventing synthetic maps or searching test
splits until a target score appears would not recover the authors' experiment.

## What would resolve this

An author notebook/script including count-to-input conversion, curated wafer
IDs, 12 synthetic Donut maps or their generator, random seeds/split IDs and CV
procedure. With these inputs, the unknown steps can be replaced and verified.
No messages have been sent to the authors.

**Current conclusion:** specified components and an all-size 70/30 experiment
are executable on CPU/MPS/CUDA. Exact implementation equivalence and exact
91.3%/94%/89.89% metric equality cannot be established from the published
information currently available. The user subsequently approved full MPS
training: the first 300-epoch run obtained accuracy 0.82158 and macro F1 0.72707.

## Validation-only F1 study

The subsequent diagnostic study uses the same original 30% test identities,
but takes 20% of the remaining development data for validation (56/14/30).
Variants are selected by validation **macro F1**, never test F1. The historical
test result has already been inspected, so the final comparison is not a new
blind benchmark. Threshold/order changes, log-count scaling and balanced
sampling are explicitly experimental departures from the baseline paper
implementation. The 7x7-to-26x26 bridge remains unpublished.

The completed study screened 18 adaptive variants for 40 epochs each. The
winner preserves native defects, scales coordinates into the 26x26 domain,
weights counts by native die area, adds training-only rotations of minority
classes, uses balanced sampling and lr=0.001. Its 300-epoch run's best validation
macro F1 was 0.85999 at epoch 30; test macro F1 was 0.82540. A subsequent refit
on all 70% development originals with the frozen settings and 30 epochs reached
test macro F1 0.83598 and accuracy 0.88662. Neither result reaches the rounded
paper all-size macro F1 0.87. The refit was performed after the historical test
had been reported, so no new blind benchmark is claimed. The baseline and all
study artifacts are preserved; `results/f1_study_20261011.json` records details.
