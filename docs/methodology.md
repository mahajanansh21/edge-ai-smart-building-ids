# Methodology

## Dataset inspection

The inspected Edge-IIoTset CSV contained 157,800 records and 63 columns. The binary target was `Attack_label`; the multiclass target was `Attack_type`.

The binary distribution was 24,301 Normal and 133,499 Attack records.

## Duplicate handling and partitioning

Complete-row duplicate removal removed 814 records. All removed duplicates belonged to MITM. The retained dataset contained 156,986 records.

A stratified record-level partition was created:

| Partition | Records |
|---|---:|
| Training | 109,890 |
| Validation | 23,548 |
| Test | 23,548 |

The partitions do not represent verified independent devices or capture sessions.

## Preprocessing

The expanded configuration began with 36 source features and produced 43 transformed columns. Imputation, scaling, categorical encoding and constant-feature filtering were fitted using training data only. The saved Phase 3 transformer is reused during inference.

## Model comparison

Phase 4 compared Logistic Regression, Decision Tree, Random Forest and Extra Trees across two feature configurations and two targets. Macro-F1 was the primary selection metric; balanced accuracy was secondary.

Broad and targeted TCP feature-ablation experiments were conducted using validation data.

## Frozen candidates

- Binary: Random Forest without `numeric__tcp.len`.
- Multiclass: Decision Tree without `numeric__tcp.checksum`.

The models were retrained on the full training partition, checked on validation data and frozen before the final test evaluation.

## Final test protocol

The held-out test partition was evaluated once after model freezing. No tuning or model selection was performed using the final test results.

## Dashboard inference

The application loads the frozen preprocessor and estimators. It validates source feature inputs, transforms them, selects the target-specific transformed feature columns and generates predictions locally on CPU.

Offline simulation replays validation records through the frozen models. It is not live network capture.
