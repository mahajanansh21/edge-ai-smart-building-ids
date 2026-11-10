# Phase 6 — Dashboard Integration and Edge Inference

## Scope

This phase connects the frozen Phase 5 binary Random Forest and multiclass Decision Tree to a local Streamlit dashboard. It uses the saved Phase 3 `expanded36_normalized` preprocessing artifact and the saved transformed feature order.

The dashboard includes Overview, Dataset Explorer, Model Evaluation, Traffic Analyzer, Live Simulation, Threat Alerts, Edge Performance, and About.

## Important behavior

- The application does not train or modify models.
- It loads only the frozen model files under `models/final`.
- It loads saved Phase 5 test-report CSVs for the Model Evaluation page; it does not open the test matrices or labels.
- CSV inference requires all 36 Phase 3 source feature columns.
- The binary model excludes transformed feature `numeric__tcp.len`.
- The multiclass model excludes transformed feature `numeric__tcp.checksum`.
- Labels such as `Attack_label` and `Attack_type` are not passed to either model. If they happen to be present in a supplied CSV, they are kept only as reference columns in the output.
- The Live Simulation page replays rows from a local dataset preview. It is not live packet capture, a network sensor, or an active scanner.
- The alert page reports model predictions, not confirmed incidents or calibrated severity levels.

## Install and run

From the project root in PowerShell:

```powershell
Set-Location "C:\Users\ASUS\Documents\Projects\edge-ai-smart-building-ids"
if (-not (Test-Path ".\app_phase2_backup.py")) { Copy-Item ".\app.py" ".\app_phase2_backup.py" }
Copy-Item ".\phase6_dashboard_pack\app_phase6.py" ".\app.py" -Force
Copy-Item ".\phase6_dashboard_pack\src\edge_inference.py" ".\src\edge_inference.py" -Force
Copy-Item ".\phase6_dashboard_pack\src\phase6_utils.py" ".\src\phase6_utils.py" -Force
Copy-Item ".\phase6_dashboard_pack\tests\test_phase6_utils.py" ".\tests\test_phase6_utils.py" -Force
Copy-Item ".\phase6_dashboard_pack\docs\PHASE6_GUIDE.md" ".\docs\PHASE6_GUIDE.md" -Force

.\.venv\Scripts\python.exe -m py_compile .\app.py .\src\edge_inference.py .\src\phase6_utils.py
.\.venv\Scripts\python.exe -m pytest .\tests\test_phase6_utils.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

## Troubleshooting

1. If model loading reports a missing file, compare `models/final` with the Phase 5 frozen bundle.
2. If the preprocessing output dimension differs from 43, do not bypass the check. Confirm the matching Phase 3 preprocessor and `feature_names.json` are present.
3. If uploaded CSV inference reports missing columns, use an Edge-IIoTset-derived CSV containing the complete 36 source features. Extra columns are permitted.
4. Keep each inference request to 5,000 records or fewer.
5. This project was developed in the user's Python 3.13.3 environment; the app should be tested in that same `.venv`.

## Known limitations

- The dataset split is record-level, not verified device/session-level.
- Earlier exploratory audits used the full dataset before splitting.
- The public dataset's capture and formatting patterns may not represent unseen smart-building environments.
- High test performance does not establish deployment readiness.
- The dashboard is a defensive educational prototype; do not connect it to production building controls or treat predictions as incident confirmation.
