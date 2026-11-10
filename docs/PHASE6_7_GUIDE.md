# Phase 6–7 — Manual inference, traffic replay, alerts and CPU telemetry

## Scope

This upgrade builds on the existing Phase 6 dashboard and frozen Phase 5 artifacts. It does not retrain, modify, or reselect models, and it does not rerun the Phase 5 test evaluation.

### Phase 6 capabilities

- Overview with recorded Phase 5 final test metrics.
- Dataset Explorer with full-dataset binary and multiclass distributions plus a separate limited row preview.
- Model Evaluation using saved Phase 5 test-report CSV files only.
- Traffic Analyzer with three inputs: one manually entered feature vector, an uploaded CSV, or a labelled validation example.
- Uses the Phase 3 train-fitted `expanded36_normalized` preprocessor.
- Uses the frozen binary Random Forest (drop `numeric__tcp.len`) and multiclass Decision Tree (drop `numeric__tcp.checksum`).
- Optional per-model `predict_proba` outputs are explicitly labelled uncalibrated model scores.
- Feature range/category diagnostics compare inputs with the training split; they are simple univariate warnings, not a validated OOD detector.

### Phase 7 capabilities

- Batched offline replay from the validation partition only.
- Progress reporting and reproducible random or stratified sampling.
- Session alert feed containing model-predicted attacks with a review-required status.
- Export of predictions, alerts and per-run performance JSON to `outputs/phase7/runs/`.
- CPU/memory telemetry, processing latency, throughput and session run history.

## Install in the Windows project

Extract into a staging folder under Downloads (not into the project). This avoids a duplicate `tests` directory inside the project root.

```powershell
Set-Location "C:\Users\ASUS\Documents\Projects\edge-ai-smart-building-ids"

$stage = Join-Path $env:USERPROFILE "Downloads\phase67_dashboard_pack_staging"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\phase67_dashboard_pack.zip" -DestinationPath $stage -Force

# Back up the currently working Phase 6 app once.
if (-not (Test-Path ".\app_phase6_backup.py")) { Copy-Item ".\app.py" ".\app_phase6_backup.py" }

# Copy the new dashboard and modules into the project.
Copy-Item "$stage\app.py" ".\app.py" -Force
Copy-Item "$stage\src\edge_inference.py" ".\src\edge_inference.py" -Force
Copy-Item "$stage\src\phase7_ops.py" ".\src\phase7_ops.py" -Force
Copy-Item "$stage\src\phase6_utils.py" ".\src\phase6_utils.py" -Force
Copy-Item "$stage\src\phase67_smoke_test.py" ".\src\phase67_smoke_test.py" -Force
Copy-Item "$stage\src\__init__.py" ".\src\__init__.py" -Force
Copy-Item "$stage\tests\test_phase7_ops.py" ".\tests\test_phase7_ops.py" -Force
Copy-Item "$stage\docs\PHASE6_7_GUIDE.md" ".\docs\PHASE6_7_GUIDE.md" -Force
Copy-Item "$stage\docs\UNIVERSITY_DEMO_SCRIPT.md" ".\docs\UNIVERSITY_DEMO_SCRIPT.md" -Force
```

The existing `src/phase6_utils.py` implementation is retained (the pack includes the matching helper file). No dependencies need to be added; the project already has Streamlit, Plotly, pandas, scikit-learn, Joblib, psutil and SciPy installed.

## Verify files, syntax and tests

```powershell
Test-Path ".\app.py"
Test-Path ".\src\edge_inference.py"
Test-Path ".\src\phase7_ops.py"
Test-Path ".\tests\test_phase7_ops.py"

.\.venv\Scripts\python.exe -m py_compile .\app.py .\src\edge_inference.py .\src\phase7_ops.py
.\.venv\Scripts\python.exe -m pytest .\tests\test_phase7_ops.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

## Local artifact smoke test and launch

The smoke test creates one feature vector from independent training-split medians/modes, passes it through the saved preprocessor and frozen models, and prints predictions. It does not use labels, train models, or access the test split.

```powershell
.\.venv\Scripts\python.exe -m src.phase67_smoke_test
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Open `http://localhost:8501`.

## Suggested demonstration

1. **Dataset Explorer:** show full label distribution. The preview row count is intentionally separate from the full dataset distribution.
2. **Model Evaluation:** show saved Phase 5 test results; do not use them for tuning.
3. **Traffic Analyzer → Manual single record:** start with training median/mode defaults, edit selected features, submit one feature vector. The defaults are derived from marginal training statistics and do not represent a validated packet. A manually entered vector is not necessarily a physically valid or realistic flow.
4. **Traffic Analyzer → Upload CSV:** use a CSV containing all 36 Phase 3 source feature columns. Maximum 5,000 rows per run. `Attack_label` and `Attack_type`, if present for a demonstration, are reference-only and excluded from model inputs.
5. **Live Simulation:** replay stratified or random records from `data/processed/splits/validation.csv`, in batches. The test partition is not opened.
6. **Threat Alerts:** show model-predicted attack records. The status is `review required`, not a confirmed incident or calibrated severity rating.
7. **Edge Performance:** show local CPU/memory usage and latest inference timing/throughput.
8. Save a run from the latest result panel. Files appear under `outputs/phase7/runs/<run_id>/`:
   - `predictions.csv`
   - `threat_alerts.csv`
   - `run_summary.json`

## Important limitations

- This is an offline educational defensive prototype, not a packet sniffer, live network sensor, or production building control system.
- It does not capture live packets, actively scan networks, execute attacks, or control building devices.
- Model scores from `predict_proba` are not calibrated probabilities of real-world correctness.
- Training-profile range/category checks are univariate diagnostics, not a multivariate OOD detector.
- A classifier always predicts among its known classes. It does not provide a validated `Unknown` class.
- Validation replay rows have dataset labels for reference only. Neither label is a model feature.
- Dataset split is record-level, not verified device/session-level; earlier exploratory audits used the full dataset. Results do not establish generalization to unseen buildings, devices or sessions.
- Do not tune or retrain based on the final test set. The Phase 5 test evaluation is final.
