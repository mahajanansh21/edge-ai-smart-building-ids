# Processed data

Processed split CSVs and transformed model matrices are generated locally and intentionally excluded from Git.

To regenerate from the raw dataset, run from the project root:

```powershell
python -m src.prepare_splits
python -m src.prepare_model_inputs
```

These commands prepare data and preprocessors; they do not train the Phase 5 frozen classifiers.
