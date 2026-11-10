"""Frozen Phase 5 model loading and CPU inference for the Edge AI IDS dashboard."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse

from src.phase6_utils import combine_detection, retained_feature_indices, validate_source_columns

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "models" / "final"
INPUT_DIR = ROOT / "data" / "processed" / "model_inputs" / "expanded36_normalized"
TRAIN_SPLIT = ROOT / "data" / "processed" / "splits" / "train.csv"
VALIDATION_SPLIT = ROOT / "data" / "processed" / "splits" / "validation.csv"
TEST_REPORT_DIR = ROOT / "outputs" / "metrics" / "phase5" / "final_test"
RAW_DATASET = ROOT / "data" / "raw" / "ML-EdgeIIoT-dataset.csv"
CATEGORICAL_SOURCE_FEATURES = {
    "http.request.method", "http.request.version", "mqtt.protoname", "mqtt.topic"
}
BINARY_REMOVED_FEATURE = "numeric__tcp.len"
MULTICLASS_REMOVED_FEATURE = "numeric__tcp.checksum"


@dataclass
class ModelBundle:
    preprocessor: Any
    binary_model: Any
    multiclass_model: Any
    source_features: list[str]
    transformed_features: list[str]
    binary_indices: list[int]
    multiclass_indices: list[int]
    binary_spec: dict[str, Any]
    multiclass_spec: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required project file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_model_bundle() -> ModelBundle:
    """Load frozen estimators and validate their saved feature contract."""
    paths = {
        "preprocessor": FINAL_DIR / "expanded36_normalized_preprocessor.joblib",
        "binary_model": FINAL_DIR / "binary_random_forest_without_tcp_len.joblib",
        "multiclass_model": FINAL_DIR / "multiclass_decision_tree_without_tcp_checksum.joblib",
        "binary_spec": FINAL_DIR / "binary_random_forest_without_tcp_len_input_spec.json",
        "multiclass_spec": FINAL_DIR / "multiclass_decision_tree_without_tcp_checksum_input_spec.json",
        "feature_names": INPUT_DIR / "feature_names.json",
        "preprocessing_summary": INPUT_DIR / "preprocessing_summary.json",
        "manifest": FINAL_DIR / "phase5_model_freeze_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Phase 5 artifact(s):\n" + "\n".join(missing))

    manifest = _read_json(paths["manifest"])
    if manifest.get("status") != "frozen_before_test_evaluation":
        raise ValueError("The Phase 5 freeze manifest has an unexpected status.")

    preprocessing_summary = _read_json(paths["preprocessing_summary"])
    with paths["feature_names"].open("r", encoding="utf-8") as handle:
        transformed_features = json.load(handle)
    source_features = preprocessing_summary.get("source_features")
    if not isinstance(source_features, list) or len(source_features) != 36:
        raise ValueError("Expected exactly 36 Phase 3 source features.")
    if not isinstance(transformed_features, list) or len(transformed_features) != 43:
        raise ValueError("Expected exactly 43 transformed Phase 3 features.")

    binary_indices = retained_feature_indices(transformed_features, BINARY_REMOVED_FEATURE)
    multiclass_indices = retained_feature_indices(transformed_features, MULTICLASS_REMOVED_FEATURE)
    if len(binary_indices) != 42 or len(multiclass_indices) != 42:
        raise ValueError("Each frozen estimator must receive exactly 42 transformed features.")

    return ModelBundle(
        preprocessor=joblib.load(paths["preprocessor"]),
        binary_model=joblib.load(paths["binary_model"]),
        multiclass_model=joblib.load(paths["multiclass_model"]),
        source_features=list(source_features),
        transformed_features=list(transformed_features),
        binary_indices=binary_indices,
        multiclass_indices=multiclass_indices,
        binary_spec=_read_json(paths["binary_spec"]),
        multiclass_spec=_read_json(paths["multiclass_spec"]),
    )


def prepare_source_frame(frame: pd.DataFrame, source_features: list[str]) -> pd.DataFrame:
    """Validate input schema and cast columns to Phase 3 conventions."""
    missing = validate_source_columns(frame.columns, source_features)
    if missing:
        preview = ", ".join(missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise ValueError(f"Uploaded data is missing {len(missing)} required feature column(s): {preview}{suffix}")

    prepared = frame.loc[:, source_features].copy()
    for column in source_features:
        if column in CATEGORICAL_SOURCE_FEATURES:
            prepared[column] = prepared[column].where(prepared[column].notna(), np.nan).astype(object)
            non_missing = prepared[column].notna()
            prepared.loc[non_missing, column] = prepared.loc[non_missing, column].astype(str)
        else:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def _slice_columns(matrix: Any, indices: list[int]) -> Any:
    if sparse.issparse(matrix):
        return matrix[:, indices]
    return np.asarray(matrix)[:, indices]


def _class_probability(estimator: Any, matrix: Any, wanted_class: Any | None = None) -> np.ndarray | None:
    """Return an estimator probability column when available; never call it calibrated confidence."""
    if not hasattr(estimator, "predict_proba"):
        return None
    probabilities = np.asarray(estimator.predict_proba(matrix))
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        return None
    classes = list(getattr(estimator, "classes_", range(probabilities.shape[1])))
    if wanted_class is not None and wanted_class in classes:
        return probabilities[:, classes.index(wanted_class)]
    return probabilities.max(axis=1)


def predict_dataframe(frame: pd.DataFrame, bundle: ModelBundle) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run both frozen classifiers. Labels are excluded from the model input frame."""
    if frame.empty:
        raise ValueError("There are no records to analyze.")
    if len(frame) > 5000:
        raise ValueError("Please analyze at most 5,000 records at a time.")

    source_frame = prepare_source_frame(frame, bundle.source_features)
    process = psutil.Process()
    memory_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()
    transformed = bundle.preprocessor.transform(source_frame)
    transform_seconds = time.perf_counter() - start
    if transformed.shape[1] != len(bundle.transformed_features):
        raise ValueError(
            "Preprocessor output dimension does not match Phase 3 feature_names.json: "
            f"{transformed.shape[1]} vs {len(bundle.transformed_features)}."
        )

    binary_matrix = _slice_columns(transformed, bundle.binary_indices)
    multiclass_matrix = _slice_columns(transformed, bundle.multiclass_indices)

    start = time.perf_counter()
    binary_predictions = bundle.binary_model.predict(binary_matrix)
    binary_predict_seconds = time.perf_counter() - start
    binary_attack_score = _class_probability(bundle.binary_model, binary_matrix, wanted_class=1)

    start = time.perf_counter()
    multiclass_predictions = bundle.multiclass_model.predict(multiclass_matrix)
    multiclass_predict_seconds = time.perf_counter() - start
    multiclass_top_score = _class_probability(bundle.multiclass_model, multiclass_matrix)

    result = frame.copy().reset_index(drop=True)
    result["Predicted_Binary"] = ["Attack" if int(value) == 1 else "Normal" for value in binary_predictions]
    result["Predicted_Attack_Type"] = [str(value) for value in multiclass_predictions]
    result["Final_Detection"] = [
        combine_detection(binary, attack_type)
        for binary, attack_type in zip(binary_predictions, multiclass_predictions)
    ]
    if binary_attack_score is not None:
        result["Binary_Attack_Score_Uncalibrated"] = binary_attack_score
    if multiclass_top_score is not None:
        result["Multiclass_Top_Score_Uncalibrated"] = multiclass_top_score

    # Existing labels, if supplied for a validation demonstration, remain reference-only.
    if "Attack_label" in frame.columns:
        result["Dataset_Attack_label_reference"] = frame["Attack_label"].reset_index(drop=True)
    if "Attack_type" in frame.columns:
        result["Dataset_Attack_type_reference"] = frame["Attack_type"].reset_index(drop=True)

    elapsed = transform_seconds + binary_predict_seconds + multiclass_predict_seconds
    memory_after = process.memory_info().rss / (1024 * 1024)
    rows = len(frame)
    performance = {
        "rows": float(rows),
        "transform_seconds": transform_seconds,
        "binary_predict_seconds": binary_predict_seconds,
        "multiclass_predict_seconds": multiclass_predict_seconds,
        "total_seconds": elapsed,
        "records_per_second": rows / max(elapsed, 1e-9),
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "memory_delta_mb": memory_after - memory_before,
    }
    return result, performance


def load_test_report(target: str) -> pd.DataFrame:
    path = TEST_REPORT_DIR / target / "test_classification_report.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Final Phase 5 report not found: {path}")
    return pd.read_csv(path)


def load_test_confusion_matrix(target: str) -> pd.DataFrame:
    path = TEST_REPORT_DIR / target / "test_confusion_matrix.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Final Phase 5 confusion matrix not found: {path}")
    return pd.read_csv(path, index_col=0)


def read_dataset_preview(nrows: int = 1500) -> pd.DataFrame:
    if not RAW_DATASET.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATASET}")
    return pd.read_csv(RAW_DATASET, nrows=nrows, low_memory=False)


def read_full_label_distribution() -> tuple[pd.Series, pd.Series, int]:
    """Read only labels from the original CSV for full-dataset descriptive charts."""
    if not RAW_DATASET.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATASET}")
    labels = pd.read_csv(RAW_DATASET, usecols=["Attack_label", "Attack_type"])
    return labels["Attack_label"].value_counts().sort_index(), labels["Attack_type"].value_counts(), len(labels)


def read_validation_for_replay() -> pd.DataFrame:
    """Load validation rows only for the offline demo; do not read test data."""
    if not VALIDATION_SPLIT.is_file():
        raise FileNotFoundError(f"Validation split not found: {VALIDATION_SPLIT}")
    summary_path = INPUT_DIR / "preprocessing_summary.json"
    summary = _read_json(summary_path)
    features = summary["source_features"]
    optional_labels = ["Attack_label", "Attack_type"]
    return pd.read_csv(VALIDATION_SPLIT, usecols=features + optional_labels, low_memory=False)
