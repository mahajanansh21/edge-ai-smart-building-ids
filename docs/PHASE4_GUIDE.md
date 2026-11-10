# Phase 4 — Model Training and Validation

## Goal

Train candidate CPU classifiers using the sparse matrices produced in Phase 3. Compare them on the **validation** partition and save trained estimators, per-class reports, confusion matrices, feature-importance summaries, and a single leaderboard.

**The test split is not loaded or evaluated in Phase 4.** Keep it sealed until the final evaluation stage.

## Candidate experiments

Two feature configurations are compared:

- `core9`: conservative nine-feature baseline.
- `expanded36_normalized`: 36 candidate raw features transformed to 43 numeric/one-hot columns. It is a research comparator, not automatically a production-safe feature set.

Two prediction targets are trained:

- `Attack_label`: binary classification, where `0 = Normal` and `1 = Attack`.
- `Attack_type`: 15-class attack-category classification.

Four CPU-friendly scikit-learn classifiers are compared:

- Logistic Regression
- Decision Tree
- Random Forest (75 trees)
- Extra Trees (75 trees)

The full run attempts up to 16 experiments (2 feature sets × 2 targets × 4 classifiers). Run `--quick` first to check the pipeline with two classifiers for all feature-set/target combinations; quick mode uses a deterministic stratified sample of at most 25,000 training rows per target, while retaining the full validation partition. Quick-mode scores are preliminary and are overwritten by the full run.

## Metrics

The leaderboard includes accuracy, balanced accuracy, macro precision, macro recall, macro-F1, weighted-F1, fitting duration, and validation inference speed. For the binary target it also reports Normal recall and Attack recall.

Use **macro-F1 first**, with balanced accuracy as a secondary metric, because the class distribution is imbalanced and rare attack types should not be hidden by a high overall accuracy. Review the confusion matrix and per-class report; do not select a model from one scalar metric alone.

## Outputs

- `models/phase4/<feature_set>/<target>/<model>.joblib`
- `outputs/metrics/phase4/validation_model_comparison.csv`
- `outputs/metrics/phase4/phase4_run_summary.json`
- `outputs/metrics/phase4/<feature_set>/<target>/*_classification_report.csv`
- `outputs/metrics/phase4/<feature_set>/<target>/*_confusion_matrix.csv`
- `outputs/metrics/phase4/<feature_set>/<target>/*_top_features.csv` (where supported by the estimator)

## Methodological cautions

1. Validation is used for candidate selection; test data remains untouched.
2. Earlier exploratory dataset audits and feature-set decisions used the full dataset before the train/validation/test split was established. Therefore, the validation scores are useful for comparing candidates but are not a fully untouched confirmatory estimate.
3. Exact full-row duplicates were removed before splitting; all 814 removed duplicates were MITM records. This reduces the MITM representation from 1,214 to 400 unique records and must be disclosed in the report.
4. A random stratified row split is not the same as a device-, capture-session-, or time-based split. The available work did not establish reliable group IDs for such a split.
5. `expanded36_normalized` includes protocol and packet metadata that showed dataset-specific label skew in earlier audits. Very high scores, particularly near-perfect results, must trigger leakage/artifact investigation rather than celebration.
6. The nine-feature `core9` configuration previously showed limited multiclass separation. Poor multiclass performance is a useful baseline finding, not a code failure.
7. These are offline labelled dataset experiments. They do not establish live network monitoring, resistance to new attacks, or production building-security readiness.

## Reproducibility

- Random seed: 42.
- Classifiers run on CPU; no cloud API or GPU required.
- Preprocessing matrices and all preprocessing statistics/categories were produced by Phase 3 using the training partition only.
- Do not modify the original CSV or the saved Phase 3 matrices while a run is in progress.
