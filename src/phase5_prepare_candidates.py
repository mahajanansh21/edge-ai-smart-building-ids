"""Phase 5A: retrain the two selected reduced-feature candidates on train only.

This script reads train and validation matrices/labels only. It intentionally
never opens X_test.npz or y_test.csv.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix

from src.phase4_utils import calculate_metrics, load_target
from src.phase5_utils import build_keep_indices, validate_matrix_and_features

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "processed" / "model_inputs" / "expanded36_normalized"
PHASE4_MODEL_DIR = ROOT / "models" / "phase4" / "expanded36_normalized"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessors" / "expanded36_normalized_preprocessor.joblib"
CANDIDATE_DIR = ROOT / "models" / "phase5" / "candidates"
METRICS_DIR = ROOT / "outputs" / "metrics" / "phase5" / "candidate_validation"

CANDIDATES = {
    "Attack_label": {
        "model_name": "random_forest",
        "source_model": PHASE4_MODEL_DIR / "Attack_label" / "random_forest.joblib",
        "removed_features": ["numeric__tcp.len"],
        "candidate_name": "binary_random_forest_without_tcp_len",
    },
    "Attack_type": {
        "model_name": "decision_tree",
        "source_model": PHASE4_MODEL_DIR / "Attack_type" / "decision_tree.joblib",
        "removed_features": ["numeric__tcp.checksum"],
        "candidate_name": "multiclass_decision_tree_without_tcp_checksum",
    },
}


def _load_matrix(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required matrix missing: {path}")
    matrix = sparse.load_npz(path).tocsr()
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Matrix is empty: {path}")
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"Matrix contains NaN or infinite values: {path}")
    return matrix


def _load_feature_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature manifest missing: {path}")
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError("feature_names.json must contain a non-empty JSON list of strings")
    return names


def _save_reports(target: str, y_true, y_pred) -> tuple[Path, Path]:
    out_dir = METRICS_DIR / target
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "validation_classification_report.csv"
    confusion_path = out_dir / "validation_confusion_matrix.csv"

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(report_path, index_label="class_or_average")
    labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()), key=lambda x: str(x))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        confusion_path, index_label="actual\\predicted"
    )
    return report_path, confusion_path


def main() -> int:
    print("=" * 78)
    print("PHASE 5A: FINAL CANDIDATE RETRAINING AND VALIDATION CHECK")
    print("Only train and validation partitions will be read. Test files will NOT be opened.")
    print("=" * 78)

    required = [
        FEATURE_DIR / "X_train.npz",
        FEATURE_DIR / "X_validation.npz",
        FEATURE_DIR / "y_train.csv",
        FEATURE_DIR / "y_validation.csv",
        FEATURE_DIR / "feature_names.json",
        PREPROCESSOR_PATH,
    ] + [item["source_model"] for item in CANDIDATES.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("ERROR: Required Phase 3/4 artifacts are missing:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    names = _load_feature_names(FEATURE_DIR / "feature_names.json")
    X_train = _load_matrix(FEATURE_DIR / "X_train.npz")
    X_val = _load_matrix(FEATURE_DIR / "X_validation.npz")
    validate_matrix_and_features(X_train, names, "X_train")
    validate_matrix_and_features(X_val, names, "X_validation")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Train and validation feature dimensions do not match")

    results = {}
    failures = []
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    for target, spec in CANDIDATES.items():
        print(f"\n[{target} | {spec['candidate_name']}]")
        try:
            y_train = load_target(FEATURE_DIR / "y_train.csv", target)
            y_val = load_target(FEATURE_DIR / "y_validation.csv", target)
            if len(y_train) != X_train.shape[0] or len(y_val) != X_val.shape[0]:
                raise ValueError(f"Label/matrix row count mismatch for {target}")
            if set(np.asarray(y_val).tolist()) - set(np.asarray(y_train).tolist()):
                raise ValueError(f"Validation contains labels absent from training for {target}")

            keep_indices = build_keep_indices(names, spec["removed_features"])
            Xtr = X_train[:, keep_indices]
            Xv = X_val[:, keep_indices]

            template = joblib.load(spec["source_model"])
            estimator = clone(template)
            start = time.perf_counter()
            estimator.fit(Xtr, y_train)
            fit_seconds = time.perf_counter() - start

            start = time.perf_counter()
            prediction = estimator.predict(Xv)
            predict_seconds = time.perf_counter() - start

            metrics = calculate_metrics(y_val, prediction, target)
            report_path, confusion_path = _save_reports(target, y_val, prediction)

            model_path = CANDIDATE_DIR / f"{spec['candidate_name']}.joblib"
            metadata_path = CANDIDATE_DIR / f"{spec['candidate_name']}_input_spec.json"
            joblib.dump(estimator, model_path, compress=3)
            input_spec = {
                "target": target,
                "model_name": spec["model_name"],
                "candidate_name": spec["candidate_name"],
                "source_feature_set": "expanded36_normalized",
                "source_transformed_feature_count": len(names),
                "removed_features": spec["removed_features"],
                "retained_feature_indices": keep_indices,
                "retained_feature_names": [names[i] for i in keep_indices],
                "preprocessor_path": "models/preprocessors/expanded36_normalized_preprocessor.joblib",
                "input_matrix_expected_columns_before_selection": len(names),
                "input_matrix_expected_columns_after_selection": len(keep_indices),
            }
            metadata_path.write_text(json.dumps(input_spec, indent=2), encoding="utf-8")

            results[target] = {
                "status": "success",
                "candidate_name": spec["candidate_name"],
                "model_name": spec["model_name"],
                "model_path": str(model_path.relative_to(ROOT)),
                "input_spec_path": str(metadata_path.relative_to(ROOT)),
                "validation_classification_report": str(report_path.relative_to(ROOT)),
                "validation_confusion_matrix": str(confusion_path.relative_to(ROOT)),
                "train_rows": int(Xtr.shape[0]),
                "validation_rows": int(Xv.shape[0]),
                "retained_features": int(len(keep_indices)),
                "removed_features": spec["removed_features"],
                "metrics": metrics,
                "fit_seconds": round(fit_seconds, 4),
                "validation_predict_seconds": round(predict_seconds, 4),
                "validation_records_per_second": round(len(y_val) / max(predict_seconds, 1e-12), 2),
            }
            print(
                f"  Validation Macro-F1={metrics['macro_f1']:.4f}; "
                f"accuracy={metrics['accuracy']:.4f}; balanced accuracy={metrics['balanced_accuracy']:.4f}"
            )
            print(f"  Saved candidate: {model_path}")
        except Exception as exc:
            failures.append({"target": target, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    summary = {
        "phase": 5,
        "stage": "5A_candidate_retraining",
        "purpose": "retrain provisional selected reduced-feature candidates on full training partition; validate; do not read test data",
        "test_split_used": False,
        "preprocessor_fitted_on": "train partition only in Phase 3",
        "candidate_results": results,
        "successful_candidates": len(results),
        "failed_candidates": len(failures),
        "failures": failures,
        "next_action": "Review candidate validation metrics and files. Only then run src.phase5_freeze_candidates. Do not evaluate test yet.",
        "limitations": [
            "Feature and model selection used the validation partition repeatedly; validation scores are not independent confirmation.",
            "Earlier exploratory audits used the full dataset before splitting.",
            "The split is record-level, not verified device/session-level.",
            "The eventual test result will estimate performance on this dataset split, not prove generalization to unseen devices or environments.",
        ],
    }
    summary_path = ROOT / "outputs" / "metrics" / "phase5" / "phase5_candidate_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"PHASE 5A FINISHED: {len(results)} successful; {len(failures)} failed")
    print(f"Candidate summary: {summary_path}")
    print("TEST PARTITION NOT OPENED.")
    print("=" * 78)
    return 0 if len(results) == len(CANDIDATES) and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
