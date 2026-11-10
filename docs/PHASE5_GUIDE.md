# Phase 5 — Candidate Freeze and One-Time Final Test

## Purpose
Phase 5 retrains the two provisional Phase 4 candidates on the complete training partition, records validation checks, freezes both models together with the Phase 3 preprocessor and their feature-index contracts, and then performs one explicitly authorized test evaluation.

## Selected candidates
- Binary: Random Forest, based on the saved Phase 4 Random Forest parameters, removing `numeric__tcp.len` from the 43-column expanded transformed matrix (42 retained columns).
- Multiclass: Decision Tree, based on the saved Phase 4 Decision Tree parameters, removing `numeric__tcp.checksum` (42 retained columns).

These are validation-selected candidates. Their small Macro-F1 improvements over the full 43-feature models should not be overstated.

## Exact PowerShell commands
Run from:

```powershell
Set-Location "C:\Users\ASUS\Documents\Projects\edge-ai-smart-building-ids"
```

Extract the package (ZIP has `src`, `tests`, `docs` at root):

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\phase5_model_freeze_pack.zip" -DestinationPath "." -Force
```

Verify paths:

```powershell
Test-Path ".\src\phase5_utils.py"
Test-Path ".\src\phase5_prepare_candidates.py"
Test-Path ".\src\phase5_freeze_candidates.py"
Test-Path ".\src\phase5_evaluate_test.py"
Test-Path ".\models\preprocessors\expanded36_normalized_preprocessor.joblib"
```

Compile and test:

```powershell
.\.venv\Scripts\python.exe -m py_compile .\src\phase5_utils.py .\src\phase5_prepare_candidates.py .\src\phase5_freeze_candidates.py .\src\phase5_evaluate_test.py
.\.venv\Scripts\python.exe -m pytest .\tests\test_phase5_utils.py -q
```

### Phase 5A — Retrain provisional candidates using training data only

```powershell
.\.venv\Scripts\python.exe -m src.phase5_prepare_candidates
```

This reads only `X_train.npz`, `X_validation.npz`, `y_train.csv`, `y_validation.csv`, feature names, and the Phase 4 model parameter templates. It must not open `X_test.npz` or `y_test.csv`.

Review:

```powershell
Get-Content ".\outputs\metrics\phase5\phase5_candidate_summary.json"
```

Expected: two successful candidates, zero failures, test_split_used false. Check validation Macro-F1 against the targeted ablation run: binary approximately 0.8886; multiclass approximately 0.8249. Tiny deterministic differences may occur across library/runtime versions; investigate material differences before freezing.

### Phase 5B — Freeze the models

Only after Phase 5A succeeds and its validation summary is reviewed:

```powershell
.\.venv\Scripts\python.exe -m src.phase5_freeze_candidates
```

This copies both selected estimators, both target-specific feature specifications, and the Phase 3 train-fitted preprocessor into `models/final/`. It writes `models/final/phase5_model_freeze_manifest.json`. It does not read any dataset matrices or labels.

Verify:

```powershell
Get-ChildItem ".\models\final" -File | Select-Object Name, Length
Get-Content ".\models\final\phase5_model_freeze_manifest.json"
```

### Phase 5C — One-time test evaluation (do not run until candidate freeze is reviewed)

This is the first script in Phase 5 that opens the held-out test matrix and labels. It requires an explicit confirmation switch:

```powershell
.\.venv\Scripts\python.exe -m src.phase5_evaluate_test --confirm-one-time-test-evaluation
```

Do not run this command until both frozen models and the manifest have been checked. After test results are generated, do not tune, select, or retrain models using those test results.

Outputs:
- `outputs/metrics/phase5/final_test/phase5_final_test_summary.json`
- `outputs/metrics/phase5/final_test/Attack_label/test_classification_report.csv`
- `outputs/metrics/phase5/final_test/Attack_label/test_confusion_matrix.csv`
- `outputs/metrics/phase5/final_test/Attack_type/test_classification_report.csv`
- `outputs/metrics/phase5/final_test/Attack_type/test_confusion_matrix.csv`

## Interpretation limits
- Earlier exploratory dataset/feature audits were run before the split and used the full dataset. Treat test metrics as a useful final check on the prepared random record split, not as an entirely pristine confirmatory estimate.
- The partition is record-level. There are no verified device or capture-session identifiers used to create a group holdout.
- Results do not establish generalization to unseen devices, new buildings, different traffic collection setups, or live environments.
- The 60-record MITM validation support and class imbalance require careful discussion.
- Feature dependence, including strong reliance on `tcp.seq`, does not by itself prove leakage.

## After Phase 5
Phase 6 can integrate the frozen preprocessor, feature-selection masks, binary model, and multiclass model into the Streamlit dashboard. The test set should not be used for iterative dashboard tuning.
