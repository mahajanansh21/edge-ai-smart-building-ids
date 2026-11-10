# Experimental Results

## Final held-out test results

| Metric | Binary Random Forest | Multiclass Decision Tree |
|---|---:|---:|
| Accuracy | 93.64% | 83.51% |
| Macro precision | 85.99% | 86.55% |
| Macro recall | 93.04% | 83.03% |
| Macro-F1 | 88.96% | 82.69% |
| Weighted F1 | 93.92% | 84.77% |
| Balanced accuracy | 93.04% | 83.03% |

Test support was 23,548 records for each task.

## Binary classification observations

The binary confusion matrix was:

| Actual | Predicted Normal | Predicted Attack |
|---|---:|---:|
| Normal | 3,360 | 285 |
| Attack | 1,213 | 18,690 |

Attack recall was 93.91%. The model missed 1,213 attack records in this test partition.

## Multiclass observations

Selected per-class test results:

| Category | Precision | Recall | F1 |
|---|---:|---:|---:|
| DDoS_HTTP | 38.02% | 96.59% | 54.56% |
| Fingerprinting | 41.42% | 66.00% | 50.90% |
| Password | 100.00% | 60.37% | 75.29% |
| Uploading | 90.99% | 48.48% | 63.25% |
| XSS | 95.26% | 66.58% | 78.38% |

DDoS_HTTP has high recall but low precision. Uploading and Password have relatively low recall. These weaknesses should be considered when interpreting the system.

## Local inference demonstration

One recorded offline replay processed 150 records. The screenshot showed 142 predicted attacks and 8 predicted normal records. The observed model-processing throughput was approximately 413 records/second for that run.

This is a run-specific measurement, not a hardware-independent benchmark or an accuracy estimate.

## Interpretation

The scores describe performance on this held-out random record split. They do not establish generalisation to unseen devices, sessions or real buildings.
