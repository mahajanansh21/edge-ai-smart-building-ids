# Phase 4 Feature Ablation Guide

This is an additional reliability experiment; it does not overwrite the original Phase 4 models, reports, or Phase 3 matrices.

## What it tests

The script uses the existing `expanded36_normalized` sparse matrices and the exact train-fitted Phase 3 transformation. It removes transformed columns after preprocessing, then trains the Phase 4 candidate model for each target:

- `Attack_label`: Random Forest
- `Attack_type`: Decision Tree

Ablations:

- `full_expanded43`: baseline using all 43 transformed features.
- `no_tcp_sequence_ack_checksum`: removes TCP sequence, acknowledgement, raw acknowledgement, and checksum columns.
- `no_tcp_family`: removes all transformed numeric TCP columns.
- `no_http_family`: removes numeric HTTP columns and one-hot HTTP method/version columns.
- `no_mqtt_dns_families`: removes numeric MQTT/DNS columns and one-hot MQTT columns.

The script does not open `X_test.npz` or `y_test.csv`. It does not refit preprocessing. All results are validation-only.

## Run from project root (Windows PowerShell)

```powershell
Set-Location "C:\Users\ASUS\Documents\Projects\edge-ai-smart-building-ids"

# Test the script's imports and syntax
.\.venv\Scripts\python.exe -m py_compile .\src\phase4_ablation.py

# Run both candidate tasks and all ablations
.\.venv\Scripts\python.exe -m src.phase4_ablation

# Review the result table
Import-Csv ".\outputs\metrics\phase4\ablation\ablation_comparison.csv" |
    Format-Table target, ablation, removed_features, retained_features, macro_f1, balanced_accuracy, accuracy, macro_f1_change_vs_baseline -AutoSize

# Review the run summary
Get-Content ".\outputs\metrics\phase4\ablation\ablation_run_summary.json"

# Verify all saved ablation artifacts
Get-ChildItem ".\models\phase4_ablation" -Recurse -File |
    Select-Object FullName, Length
```

## Expected run count

Five ablation configurations × two targets = ten runs. A successful run summary should show 10 successful runs and 0 failed runs.

## Interpretation

Compare each ablation to `full_expanded43` for the same target. A large Macro-F1 drop after removing a feature family indicates the candidate relies heavily on that family. A similar score after removing a family suggests that the remaining features still carry substantial predictive information. Neither result alone proves or disproves leakage; it is a diagnostic experiment, not a production validation.
