# Radon-CNN on WM-811K

Reference: [Jeong et al., Scientific Reports 2023](https://www.nature.com/articles/s41598-023-34147-2).

## Run

Install the repository dependencies (`pip install -r requirements.txt`).
Run from the repository root:

```bash
python papers/radon_cnn/train.py --device cuda
python papers/radon_cnn/evaluate.py --checkpoint checkpoints/radon_cnn_v2/best.pt --device cuda
```

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
