"""Utilities for Phase 4 model training and validation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42
FEATURE_SETS = ("core9", "expanded36_normalized")
TARGETS = ("Attack_label", "Attack_type")
MODEL_NAMES = ("logistic_regression", "decision_tree", "random_forest", "extra_trees")


def load_sparse_matrix(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required preprocessed matrix was not found: {path}\n"
            "Run Phase 3 first: .\\.venv\\Scripts\\python.exe -m src.prepare_model_inputs"
        )
    matrix = sparse.load_npz(path).tocsr()
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Matrix is empty: {path}")
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"Matrix contains NaN or infinite values: {path}")
    return matrix


def load_target(path: Path, target: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required target file was not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if target not in frame.columns:
        raise ValueError(
            f"Target column {target!r} is missing from {path}. "
            f"Available columns: {list(frame.columns)}"
        )
    values = frame[target]
    if values.isna().any():
        raise ValueError(f"Missing labels found in {path} for target {target}")
    if target == "Attack_label":
        numeric = pd.to_numeric(values, errors="raise").astype("int8")
        observed = set(numeric.unique().tolist())
        if not observed.issubset({0, 1}) or observed != {0, 1}:
            raise ValueError(
                f"Attack_label must contain both 0 (Normal) and 1 (Attack); got {sorted(observed)}"
            )
        return numeric.to_numpy()
    return values.astype(str).to_numpy()


def make_estimator(model_name: str):
    """Create a reproducible CPU-compatible scikit-learn classifier."""
    if model_name == "logistic_regression":
        return LogisticRegression(
            solver="saga",
            max_iter=350,
            tol=1e-3,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=24,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=75,
            max_depth=24,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=75,
            max_depth=24,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model {model_name!r}. Choose from {MODEL_NAMES}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if target == "Attack_label":
        # Label 0 = Normal; label 1 = Attack.
        result["normal_recall"] = float(recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0))
        result["attack_recall"] = float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0))
    return result


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, target: str):
    metrics = calculate_metrics(y_true, y_pred, target)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=lambda item: str(item))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose().reset_index(names="class_or_average")
    confusion_frame = pd.DataFrame(
        matrix,
        index=[str(label) for label in labels],
        columns=[str(label) for label in labels],
    )
    confusion_frame.index.name = "actual\\predicted"
    return metrics, report_frame, confusion_frame


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_feature_importance(estimator, feature_names: list[str], path: Path, top_n: int = 25) -> None:
    """Save a compact, model-agnostic ranking where coefficients/importances exist."""
    values = None
    method = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        method = "tree_feature_importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        values = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim == 2 else np.abs(coefficients)
        method = "mean_absolute_logistic_coefficient"

    if values is None or len(values) != len(feature_names):
        return

    order = np.argsort(values)[::-1][:top_n]
    frame = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "feature": [feature_names[index] for index in order],
            "importance": [float(values[index]) for index in order],
            "method": method,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
