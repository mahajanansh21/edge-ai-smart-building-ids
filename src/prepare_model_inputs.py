"""Fit preprocessing on train only and transform train/validation/test.

No classifier is trained by this script. It completes Phase 3 by producing
reproducible matrices and saving fitted preprocessing pipelines for Phase 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.preprocessing import (
    FEATURE_SETS,
    TARGET_COLUMNS,
    assert_finite_matrix,
    build_preprocessor,
    get_feature_columns,
    get_transformed_feature_names,
    normalize_features,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
MODEL_DIR = ROOT / "models" / "preprocessors"
INPUT_DIR = ROOT / "data" / "processed" / "model_inputs"
METRICS_DIR = ROOT / "outputs" / "metrics"

for directory in (MODEL_DIR, INPUT_DIR, METRICS_DIR):
    directory.mkdir(parents=True, exist_ok=True)



def _load_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing_targets = [c for c in TARGET_COLUMNS if c not in frame.columns]
    if missing_targets:
        raise ValueError(f"{name} split is missing targets: {missing_targets}")
    if frame[TARGET_COLUMNS].isna().any().any():
        raise ValueError(f"Missing target values found in {name} split")
    return frame


def _save_matrix(matrix, path: Path) -> None:
    csr = sparse.csr_matrix(matrix, dtype=np.float32)
    assert_finite_matrix(csr, str(path.name))
    sparse.save_npz(path, csr, compressed=True)


def _save_labels(frame: pd.DataFrame, path: Path) -> None:
    frame.loc[:, TARGET_COLUMNS].to_csv(path, index=False)


def prepare_feature_set(feature_set: str, train_df: pd.DataFrame,
                         validation_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    columns = get_feature_columns(feature_set)
    for split_name, frame in (("train", train_df), ("validation", validation_df), ("test", test_df)):
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"{split_name} split is missing features for {feature_set}: {missing}")

    # Deterministic formatting normalization is row-wise. No statistics or
    # category vocabulary are learned here.
    X_train_raw = normalize_features(train_df, feature_set)
    X_validation_raw = normalize_features(validation_df, feature_set)
    X_test_raw = normalize_features(test_df, feature_set)

    # CRITICAL: fit_transform is called only on training data.
    preprocessor = build_preprocessor(feature_set)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_validation = preprocessor.transform(X_validation_raw)
    X_test = preprocessor.transform(X_test_raw)

    for name, matrix in (("train", X_train), ("validation", X_validation), ("test", X_test)):
        assert_finite_matrix(matrix, f"{feature_set} {name}")

    output_dir = INPUT_DIR / feature_set
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_matrix(X_train, output_dir / "X_train.npz")
    _save_matrix(X_validation, output_dir / "X_validation.npz")
    _save_matrix(X_test, output_dir / "X_test.npz")

    _save_labels(train_df, output_dir / "y_train.csv")
    _save_labels(validation_df, output_dir / "y_validation.csv")
    _save_labels(test_df, output_dir / "y_test.csv")

    model_path = MODEL_DIR / f"{feature_set}_preprocessor.joblib"
    joblib.dump(preprocessor, model_path)

    feature_names = get_transformed_feature_names(preprocessor)
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=2)

    summary = {
        "feature_set": feature_set,
        "source_feature_count": len(columns),
        "source_features": columns,
        "transformed_feature_count": len(feature_names),
        "transformed_feature_names_file": "feature_names.json",
        "fit_partition": "train only",
        "validation_used_for_fit": False,
        "test_used_for_fit": False,
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_validation.shape),
        "test_shape": list(X_test.shape),
        "matrix_format": "scipy CSR sparse NPZ",
        "preprocessor_artifact": str(model_path.relative_to(ROOT)),
        "experimental_warning": (
            "expanded36_normalized is a research comparator. Some protocol and packet metadata "
            "may reflect dataset-specific collection patterns; compare against core9 in Phase 4."
            if feature_set == "expanded36_normalized" else
            "core9 is a conservative baseline but previous diagnostics showed limited multiclass separation."
        ),
    }

    with open(output_dir / "preprocessing_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def main() -> None:
    print("=" * 72)
    print("PHASE 3: TRAIN-ONLY PREPROCESSING")
    print("=" * 72)

    train_df = _load_split("train")
    validation_df = _load_split("validation")
    test_df = _load_split("test")

    print(f"Train rows:      {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows:       {len(test_df):,}")

    expected_classes = set(train_df["Attack_type"].astype(str))
    if len(expected_classes) != 15:
        raise ValueError(f"Expected 15 attack classes in training, found {len(expected_classes)}")

    all_summaries = []
    for feature_set in FEATURE_SETS:
        print("\n" + "-" * 72)
        print(f"Preparing feature set: {feature_set}")
        summary = prepare_feature_set(feature_set, train_df, validation_df, test_df)
        all_summaries.append(summary)
        print(f"Transformed dimensions: {summary['train_shape'][1]}")
        print(f"Saved matrices to: {INPUT_DIR / feature_set}")

    combined_path = METRICS_DIR / "phase3_preprocessing_manifest.json"
    with open(combined_path, "w", encoding="utf-8") as file:
        json.dump(all_summaries, file, indent=2)

    print("\n" + "=" * 72)
    print("PHASE 3 PREPROCESSING COMPLETED")
    print("=" * 72)
    print("Both feature configurations were transformed.")
    print("All imputers, scalers, one-hot categories and constant-feature filters were fitted on TRAIN only.")
    print("No machine-learning classifier was trained.")
    print(f"Manifest: {combined_path}")


if __name__ == "__main__":
    main()
