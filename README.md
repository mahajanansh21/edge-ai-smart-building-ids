# Edge AI-Based Intrusion Detection for Smart Buildings

An educational Edge AI intrusion detection prototype for analysing IoT/IIoT network traffic using machine-learning models and local CPU inference.

## Project overview

This project uses the Edge-IIoTset dataset to identify potentially malicious network traffic. It combines binary intrusion detection, multiclass attack identification, offline traffic replay, threat-alert reporting and a Streamlit dashboard.

**Project scope:** Local, CPU-based defensive research prototype. It does not capture live network packets, perform active scanning, or control production network devices.

## Main capabilities

- Explore the original dataset and its attack categories.
- Detect Normal versus Attack traffic using a Random Forest.
- Identify one of 15 traffic categories using a Decision Tree.
- Analyse a manually entered feature vector or uploaded CSV.
- Replay validation traffic through the frozen models.
- Display model-predicted threat alerts.
- Measure local CPU inference time, throughput and system memory.
- Export prediction and alert reports.

## Dashboard

Run the application locally:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Open `http://localhost:8501`.

The dashboard contains Overview, Dataset Explorer, Model Evaluation, Traffic Analyzer, Live Simulation, Threat Alerts, Edge Performance and About pages.

## Dataset

Dataset file expected locally:

`data/raw/ML-EdgeIIoT-dataset.csv`

The inspected dataset contains:

| Property | Value |
|---|---:|
| Records | 157,800 |
| Original columns | 63 |
| Provisional source features | 36 |
| Transformed expanded features | 43 |
| Binary classes | 2 |
| Attack categories | 15 |

Binary distribution:

| Class | Records |
|---|---:|
| Normal | 24,301 |
| Attack | 133,499 |

The raw dataset and processed split files are intentionally not committed. Obtain the dataset from its original distributor and follow its applicable licence and citation requirements. See `data/raw/README.md` and `data/processed/README.md`.

## Final frozen models

Both models use the saved Phase 3 `expanded36_normalized` preprocessor.

| Task | Model | Input features | Excluded feature |
|---|---|---:|---|
| Binary detection | Random Forest | 42 | `numeric__tcp.len` |
| Attack identification | Decision Tree | 42 | `numeric__tcp.checksum` |

The six required frozen artifacts are stored in `models/final/`.

## Final held-out test results

These are the recorded Phase 5 results. The test set was evaluated once after model freezing.

| Metric | Binary Random Forest | Multiclass Decision Tree |
|---|---:|---:|
| Accuracy | 93.64% | 83.51% |
| Macro precision | 85.99% | 86.55% |
| Macro recall | 93.04% | 83.03% |
| Macro-F1 | 88.96% | 82.69% |
| Weighted F1 | 93.92% | 84.77% |
| Balanced accuracy | 93.04% | 83.03% |

See `outputs/metrics/phase5/final_test/` for classification reports and confusion matrices.

## Technology stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- psutil
- pytest

## Local installation

Use Python 3.13.3 or another compatible Python environment.

```powershell
git clone https://github.com/mahajanansh21/edge-ai-smart-building-ids.git
cd edge-ai-smart-building-ids

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the raw dataset at `data/raw/ML-EdgeIIoT-dataset.csv`.

For the complete dataset explorer and offline replay, prepare the local data artifacts:

```powershell
python -m src.prepare_splits
python -m src.prepare_model_inputs
```

These preparation commands do not train classifiers. The frozen models in `models/final/` are used for inference.

Run tests:

```powershell
python -m pytest -q
```

Run the inference smoke test after the required local training split is prepared:

```powershell
python -m src.phase67_smoke_test
```

Launch the dashboard:

```powershell
python -m streamlit run app.py
```

## Repository structure

```text
app.py
config.py
requirements.txt
src/
tests/
models/final/
docs/
outputs/figures/
outputs/metrics/
data/raw/          # local dataset, excluded from Git
data/processed/    # local prepared data, excluded from Git
```

## Dataset processing and methodology

The inspected source dataset had 157,800 records and 63 columns. Complete-row duplicate removal removed 814 records, all belonging to MITM. The resulting 156,986 records were split into:

- Training: 109,890
- Validation: 23,548
- Test: 23,548

The preprocessing transformer was fitted on training data only. Two feature configurations and four classifiers were compared during Phase 4. Feature-ablation experiments informed the final candidate configurations.

See `docs/methodology.md` and `docs/RESULTS.md`.

## Limitations

- The split is record-level; verified device/session-level separation was unavailable.
- Earlier exploratory audits used the full dataset before splitting. Validation is therefore not an independent confirmatory assessment.
- TCP sequence information strongly affects model performance.
- DDoS_HTTP has low precision, while Fingerprinting and Uploading have comparatively weak F1/recall.
- MITM has limited support after duplicate removal.
- Model scores displayed by the application are not calibrated probabilities.
- Offline replay is not live packet capture.
- Results on this dataset do not establish generalisation to unseen devices, sessions or buildings.

See `docs/LIMITATIONS.md`.

## Safety and intended use

This is an educational defensive prototype. Model predictions are alerts requiring human review, not confirmed security incidents. Do not connect it to production network controls or treat it as a production security appliance.

## Project documentation

- `docs/architecture.md`
- `docs/methodology.md`
- `docs/RESULTS.md`
- `docs/LIMITATIONS.md`
- `docs/UNIVERSITY_DEMO_SCRIPT.md`
- `docs/PHASE6_7_GUIDE.md`

## Streamlit Community Cloud demo

The public dashboard supports frozen-model inference from manual feature entry or an uploaded CSV, saved Phase 5 model evaluation, session threat alerts, and runtime telemetry. The raw dataset and prepared train/validation splits are intentionally excluded from Git. Therefore, Dataset Explorer and validation-split replay are local-only features; they are not enabled in the hosted demo.

For deployment, select Python 3.13 in Streamlit Community Cloud Advanced settings, use `app.py` as the app entrypoint, and deploy the `main` branch. The small `deployment/training_profile.json` contains TRAIN-only aggregate feature defaults/ranges for the manual-input form; it does not contain raw traffic records.
