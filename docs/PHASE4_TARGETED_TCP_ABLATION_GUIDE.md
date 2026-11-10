# Phase 4 — Targeted TCP Feature Ablation

## Purpose

This validation-only experiment removes six high-importance transformed TCP features one at a time and then together:

- `numeric__tcp.seq`
- `numeric__tcp.ack`
- `numeric__tcp.ack_raw`
- `numeric__tcp.checksum`
- `numeric__tcp.flags`
- `numeric__tcp.len`

The binary task reuses the parameter configuration from the saved expanded43 Random Forest. The multiclass task reuses the saved expanded43 Decision Tree. The script clones these estimators and retrains them on the existing training matrix after selecting the reduced columns.

## Safety and methodology

- Reads only `X_train.npz`, `X_validation.npz`, `y_train.csv`, `y_validation.csv`, and `feature_names.json`.
- Does not load or inspect `X_test.npz` or `y_test.csv`.
- Does not overwrite the original Phase 4 models or reports.
- Does not refit preprocessing. It reuses Phase 3 train-fitted transformed matrices.
- Produces per-class reports and confusion matrices for every run.
- Feature dependence is not proof of leakage. These are diagnostic comparisons, not evidence of generalization to unseen devices, capture sessions, or environments.

## Run from project root

```powershell
Set-Location "C:\Users\ASUS\Documents\Projects\edge-ai-smart-building-ids"

Test-Path ".\src\phase4_targeted_tcp_ablation.py"
Test-Path ".\models\phase4\expanded36_normalized\Attack_label\random_forest.joblib"
Test-Path ".\models\phase4\expanded36_normalized\Attack_type\decision_tree.joblib"

.\.venv\Scripts\python.exe -m py_compile .\src\phase4_targeted_tcp_ablation.py
.\.venv\Scripts\python.exe -m pytest tests\test_phase4_targeted_tcp_ablation.py -q
.\.venv\Scripts\python.exe -m src.phase4_targeted_tcp_ablation
```

## Results

Comparison table:

```powershell
Import-Csv ".\outputs\metrics\phase4\targeted_tcp_ablation\targeted_tcp_ablation_comparison.csv" |
Format-Table target, ablation, removed_features, retained_features, macro_f1, balanced_accuracy, accuracy, macro_f1_change_vs_baseline -AutoSize
```

Run summary:

```powershell
Get-Content ".\outputs\metrics\phase4\targeted_tcp_ablation\targeted_tcp_ablation_run_summary.json"
```

Per-class reports and confusion matrices are under:

```text
outputs\metrics\phase4\targeted_tcp_ablation\details\
```
