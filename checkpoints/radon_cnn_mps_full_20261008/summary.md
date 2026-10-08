# Full RadonCNN fixed-seed experiment

Each seed uses its own saved, disjoint train/validation/test split. All three seeds are included; no seed is chosen by test performance and no ensemble is evaluated.

| Seed | Best epoch | Stopped epoch | Unique train wafers | Test accuracy | Test macro-F1 | Training process (min) |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 32 | 62 | 5710 | 86.4935% | 86.3691% | 10.46 |
| 43 | 6 | 36 | 5710 | 87.0130% | 86.8511% | 5.99 |
| 44 | 6 | 36 | 5710 | 87.7922% | 87.9226% | 6.33 |

Across all three seeds (mean ± sample standard deviation, in percentage points):

- accuracy: 87.0996% ± 0.6537 pp
- macro_f1: 87.0476% ± 0.7952 pp
- macro_precision: 87.2947% ± 0.8025 pp
- macro_recall: 87.0996% ± 0.6537 pp

Maximum 500 epochs, early stopping patience 30; 6400 training presentations with training-only replacement. Balanced validation and test remain unique and disjoint from training. Different seeds change both initialization and saved split.

Before these runs, RadonCNN validation loss was corrected to sample-weighted cross entropy. The prior unweighted mean of batch losses gave a final singleton disproportionate weight. The superseded run was stopped before test evaluation and is excluded. Both loss definitions are retained in the audit, with checkpoint selection verified against the corrected logged sample mean.

Exact per-class metrics and confusion matrices: `summary.json`; individual predictions: each seed's `cpu_validation_predictions.csv` / `cpu_test_predictions.csv`. The training curves show all completed epochs and validation-based checkpoint selection.
