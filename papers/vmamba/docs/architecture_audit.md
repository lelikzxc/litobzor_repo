# FCS-VMamba reproduction audit

Audited 2026-10-06 against the repository's `vmamba.pdf`, particularly Figure 2, Sections 3.3-3.6 and 4.1-4.3, and [author code](https://github.com/yijiazhang666/VMamba-for-semiconductor/blob/c2f14503979b32ac7b4f0bd1a128ddec1eaaf6ef/gggmamba_sys.py).

## Why the previous runs were not the paper experiment

| Finding | Consequence | Correction |
|---|---|---|
| Main config disabled FA, SFS and CLCA | It trained a backbone ablation | All three modules enabled in the main recipe |
| Only stage 4 contained CLCA | Lost the stage 2/3 interactions from Figure 2 | CLCA in the last block of stages 2, 3, 4 |
| CLCA returned `target + attention`, then the block added another residual | The normalised feature was added twice | Attention module returns only projected attention; one residual at the caller |
| GAP preceded final LayerNorm | Averages and normalisation do not commute | Token LayerNorm, then GAP, then the nine-class head |
| Random 100/class categorical maps, one-hot encoding | Different input representation and different 900-image collection | Default uses the published 902 RGB JPEGs |
| Evaluation independently split all raw maps at 80/20/0 | Empty evaluation set, or overlap with a different training run under altered ratios | Persisted run protocol with exact holdout indices |
| Triton wrappers called an autograd Function inside another Function's forward and had no backward | GPU backpropagation would fail when Triton was installed | Direct complete PyTorch scan/merge autograd, also on CUDA |
| CPU dispatch could select a CUDA extension merely because it was installed | CPU failed in CUDA-equipped environments | Dispatch checks the input device |
| Final reporting loaded best and then saved last | Destroyed the resumable final state | Best loaded for reporting only; last preserved |
| No global RNG seed, learning rate differed from the paper | Split seed alone did not make runs reproducible; optimiser recipe differed | Seed Python/NumPy/PyTorch and use paper AdamW lr=0.001, batch=4, 50 epochs |

The main architecture is 4 stages with 2 blocks each, fixed width 96, resolutions 56/28/14/7, and `LN -> FA -> SS2D -> SFS -> residual`, without an MLP branch. FA follows RFFT amplitude gating; SFS follows the detached soft-mask equation, including zero suppression when k=0. Cross-layer context is taken before merging to preserve the higher-resolution features specified in Eq. 7, detached as in the author implementation. Linear weights use truncated-normal initialisation; SSM dt/A/D retain their specialised initialisation. AdamW respects the SS2D A/D no-weight-decay flags; both groups receive the configured scheduler.

## Limits of exact reproduction

The published source is not a runnable, complete training recipe: its `PatchMerging2D.forward` only executes merging inside an odd-size branch; SS2D registers large unused convolutional layers; there is no training/data-transform implementation. Copying these defects would not reproduce the paper's described four-stage network. This implementation follows the paper's equations and functioning SS2D instead.

The usable model has **1,082,697 parameters**, not exactly the paper's rounded 1.2 M. Unused layers are not added to manufacture a parameter-count match. The operational SS2D is VMamba v03 with fp32 scan and the usual z gate. The published reference uses an older SS2D; exact low-level parity is not claimed.

The archive is pinned to revision `c2f14503979b32ac7b4f0bd1a128ddec1eaaf6ef`, SHA256 `28ed870da08261ecf305b8459f97d36d91579e065eb4179718779a0d5f43fb88`. Counts in repository class order are `[100,100,102,103,102,100,95,100,100]`. Images are already 32x32 colour JPEGs and are resized to the paper's 224x224 input.

The authors do not publish exact split seeds, normalisation values, augmentation magnitudes, weight decay, learning-rate schedule or checkpoint-selection rule. Explicit choices here are ImageNet mean/std, +/-15-degree rotations, 10% translations, 0.2 colour jitter, weight decay 0.05, cosine schedule and maximum validation accuracy. These are assumptions, not values proved to be used in the paper. Compare multiple seeds before interpreting a difference of a few percentage points.

The paper's 86.06% is macro-precision in the abstract and also macro-F1 in Table 3; report accuracy, macro-precision, macro-recall and macro-F1 separately. Table 4 gives multi-run macro-F1 85.91% +/- 1.65%. Eq. 9 is ordinary spatial dot-product attention and is quadratic in token count, despite the text's linear-complexity claim.

CPU verification covers full-resolution forward, short real-image training/evaluation, finite gradients, residual/CLCA placement, scan/merge gradients and checkpoint/split consistency. A GPU-only test compares accelerated selective scan with the recurrent reference in forward and backward. That test is skipped here because no GPU is available. Full training accuracy remains unmeasured.
