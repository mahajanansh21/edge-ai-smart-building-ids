# Edge AI-Based Intrusion Detection for Smart Buildings

An undergraduate defensive AI/ML prototype that classifies labelled IoT/IIoT network-flow records and performs local CPU inference. The first implementation will use a public offline dataset; it does **not** monitor or attack a real network.

## Current status

**Phase 2 starter workspace.** The Streamlit app is currently a setup/status screen. It does not yet train a model or claim to detect attacks. Dataset-specific preprocessing will be implemented only after inspecting the downloaded CSV schema.

## Planned stack

- Python 3.11
- pandas / NumPy for data handling
- scikit-learn for classical machine learning
- Streamlit for the dashboard
- Plotly / Matplotlib for visualizations
- Joblib for model persistence
- pytest for tests

## Quick start (Windows PowerShell)

```powershell
# From the repository root
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, see the activation-policy note in the setup instructions provided with this project. You can also bypass activation by invoking `.venv\Scripts\python.exe` directly.

## Dataset

Primary candidate: **Edge-IIoTset**.

- IEEE DataPort: https://doi.org/10.21227/mbc1-1h68
- Kaggle distribution: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot
- Research paper: Ferrag et al., “Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning,” *IEEE Access*, vol. 10, pp. 40281–40306, 2022. DOI: https://doi.org/10.1109/ACCESS.2022.3165809

Read and comply with the dataset host's current terms/licence before use or redistribution. Download the dataset yourself and place the extracted CSV under `data/raw/`. Do not commit raw dataset files to Git.

The expected filename `DNN-EdgeIIoT-dataset.csv` is only a suggested filename used by some public distributions. Confirm the exact downloaded filename and schema before renaming or preprocessing anything.

## Repository layout

```text
app.py                 Streamlit starter/status app
config.py              Shared paths and random seed
requirements.txt       Python dependencies
src/                   Data and ML modules (added in later phases)
data/raw/              Local downloaded source data (not committed)
data/processed/        Generated processed data (not committed)
models/                Local model artifacts (not committed by default)
outputs/                Experiment figures, metrics and benchmarks
tests/                  Automated tests
docs/                   Architecture and methodology notes
```

## Reproducibility and responsible use

- Random seed: `42` (where applicable).
- Split data before fitting preprocessing transformations.
- Keep the test set untouched until final evaluation.
- Never report unexecuted or illustrative metrics as experiment results.
- Clearly distinguish offline dataset replay from real-time network monitoring.
- This is an educational prototype, not production building-security software.
