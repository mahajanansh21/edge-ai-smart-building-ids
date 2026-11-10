# Phase 3 completion pack

This pack completes the remaining Phase 3 implementation. It does **not** train a classifier.

## Included files

- `src/__init__.py`
- `src/preprocessing.py`
- `src/prepare_model_inputs.py`
- `tests/test_preprocessing.py`

## Feature configurations

1. `core9`: conservative numeric baseline. The earlier diagnostics show that this set has limited multiclass discrimination; retain it as a simple baseline, not as the assumed final feature set.
2. `expanded36_normalized`: a normalized research comparator. It includes protocol/packet fields that may carry dataset-specific patterns. It is an experimental comparison set, not automatically production-safe.

Both configurations exclude the two targets from model inputs. The expanded set canonicalizes zero-like categorical placeholders, limits HTTP method/version fields to known values, parses the mixed `dns.qry.name.len` field numerically, imputes missing values, one-hot encodes categories, scales numeric fields, and removes features constant in the training partition.

## Install / verify

Run these commands from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests\test_preprocessing.py -q
```

## Prepare model inputs (Phase 3 only)

The raw split files created earlier must exist:

- `data/processed/splits/train.csv`
- `data/processed/splits/validation.csv`
- `data/processed/splits/test.csv`

Run:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_model_inputs
```

The script calls `fit_transform` only on the training partition. It calls `transform` on validation and test. It does not use labels to fit transformations, perform resampling, or train a classifier.

## Output layout

```text
data/processed/model_inputs/
  core9/
    X_train.npz
    X_validation.npz
    X_test.npz
    y_train.csv
    y_validation.csv
    y_test.csv
    feature_names.json
    preprocessing_summary.json
  expanded36_normalized/
    ... same files ...
models/preprocessors/
  core9_preprocessor.joblib
  expanded36_normalized_preprocessor.joblib
outputs/metrics/
  phase3_preprocessing_manifest.json
```

Matrices are saved as SciPy CSR sparse `.npz` files; labels are saved separately. Model artifacts and processed matrices should not be committed to a public Git repository.

## Important research limitations to document

- Earlier full-dataset EDA and feature investigations informed feature selection before the current split was created. Therefore, the current test partition is held out from fitting but is not a completely untouched confirmatory test set. Treat its Phase 4 results as exploratory holdout results, and seek independent validation for stronger claims.
- The deduplication procedure removed 814 exact repeated rows, all from MITM (814 of 1,214 original MITM records). Report the resulting MITM count (400) and this limitation.
- The split is record-level and stratified by `Attack_type`; it is not a device-level or capture-level independent split.
- `expanded36_normalized` may exploit protocol/capture distribution differences. Compare it with `core9`; report per-class metrics and false-positive/false-negative rates, not accuracy alone.
- Do not apply SMOTE or any resampling to validation or test data. If used later, apply it only inside training folds/pipelines.
