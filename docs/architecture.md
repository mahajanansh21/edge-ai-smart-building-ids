# System Architecture

## High-level workflow

```text
Edge-IIoTset CSV
      |
      v
Dataset inspection and audit
      |
      v
Duplicate handling and stratified split
      |
      v
Training-only preprocessing
      |
      v
Model experiments and validation
      |
      v
Frozen Phase 5 model bundle
      |
      v
Streamlit dashboard
      |
      +--> Manual feature input
      +--> CSV traffic analysis
      +--> Offline validation replay
      |
      v
Binary Random Forest + Multiclass Decision Tree
      |
      v
Predictions, alerts and edge telemetry
```

## Main components

| Component | Responsibility |
|---|---|
| `config.py` | Central project paths and settings |
| `src/preprocessing.py` | Data preprocessing |
| `src/prepare_splits.py` | Dataset partition preparation |
| `src/prepare_model_inputs.py` | Train-fitted input transformation |
| `src/train_models.py` | Phase 4 classifier experiments |
| `src/phase5_prepare_candidates.py` | Final candidate retraining |
| `src/phase5_freeze_candidates.py` | Frozen artifact creation |
| `src/phase5_evaluate_test.py` | One-time final test evaluation |
| `src/edge_inference.py` | Saved-model loading and inference |
| `src/phase7_ops.py` | Replay, alert and output operations |
| `app.py` | Streamlit user interface |

## Frozen model bundle

`models/final/` contains two model artifacts, their input specifications, the shared Phase 3 preprocessor and the freeze manifest.

The binary model excludes `numeric__tcp.len`. The multiclass model excludes `numeric__tcp.checksum`.

Both model input specifications must be respected during inference.

## Operational boundary

The application performs local CPU inference and offline replay. It does not sniff packets, scan networks, execute attacks or apply network controls.
