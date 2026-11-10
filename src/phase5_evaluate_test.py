"""Phase 5C: one-time final holdout evaluation.

This is the ONLY Phase 5 script that opens X_test.npz and y_test.csv. It
requires an explicit confirmation flag and a frozen model manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import classification_report, confusion_matrix

from src.phase4_utils import calculate_metrics, load_target
from src.phase5_utils import validate_matrix_and_features

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "processed" / "model_inputs" / "expanded36_normalized"
MANIFEST_PATH = ROOT / "models" / "final" / "phase5_model_freeze_manifest.json"
OUTPUT_DIR = ROOT / "outputs" / "metrics" / "phase5" / "final_test"


def parse_args():
    parser = argparse.ArgumentParser(description="One-time Phase 5 final test evaluation")
    parser.add_argument(
        "--confirm-one-time-test-evaluation",
        action="store_true",
        help="Explicitly authorize opening the held-out test partition once.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_one_time_test_evaluation:
        print("STOPPED: test evaluation requires --confirm-one-time-test-evaluation")
        return 2
    if not MANIFEST_PATH.is_file():
        print(f"ERROR: frozen model manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        return 2

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_test_evaluation" or manifest.get("test_split_used") is not False:
        print("ERROR: model bundle is not in the expected frozen-before-test state.", file=sys.stderr)
        return 2
    if manifest.get("no_further_model_or_feature_tuning_after_test_evaluation") is not True:
        print("ERROR: freeze manifest does not prohibit post-test tuning.", file=sys.stderr)
        return 2

    feature_names_path = FEATURE_DIR / "feature_names.json"
    X_test_path = FEATURE_DIR / "X_test.npz"
    y_test_path = FEATURE_DIR / "y_test.csv"
    for path in (feature_names_path, X_test_path, y_test_path):
        if not path.is_file():
            print(f"ERROR: required test artifact missing: {path}", file=sys.stderr)
            return 2

    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    print("PHASE 5C: ONE-TIME FINAL TEST EVALUATION")
    print("The frozen models are fixed. Do not tune based on these results.")
    # This is the first point in Phase 5 at which the test matrix is opened.
    X_test = sparse.load_npz(X_test_path).tocsr()
    validate_matrix_and_features(X_test, feature_names, "X_test")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for target in ("Attack_label", "Attack_type"):
        model_info = manifest["models"].get(target)
        if not model_info:
            raise ValueError(f"Frozen model for target {target} is missing")
        model_path = ROOT / model_info["model_path"]
        spec_path = ROOT / model_info["input_spec_path"]
        if not model_path.is_file() or not spec_path.is_file():
            raise FileNotFoundError(f"Frozen model bundle is incomplete for {target}")

        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        kept_indices = spec.get("retained_feature_indices")
        if not isinstance(kept_indices, list) or not kept_indices:
            raise ValueError(f"Invalid retained feature indices for {target}")
        if max(kept_indices) >= X_test.shape[1] or min(kept_indices) < 0:
            raise ValueError(f"Retained feature index out of bounds for {target}")
        if len(kept_indices) != spec.get("input_matrix_expected_columns_after_selection"):
            raise ValueError(f"Feature index count mismatch for {target}")
        if spec.get("source_transformed_feature_count") != X_test.shape[1]:
            raise ValueError(f"Frozen feature contract does not match X_test for {target}")

        y_test = load_target(y_test_path, target)
        if len(y_test) != X_test.shape[0]:
            raise ValueError(f"Test label/matrix row count mismatch for {target}")

        model = joblib.load(model_path)
        X_selected = X_test[:, kept_indices]
        start = time.perf_counter()
        y_pred = model.predict(X_selected)
        predict_seconds = time.perf_counter() - start
        metrics = calculate_metrics(y_test, y_pred, target)

        detail_dir = OUTPUT_DIR / target
        detail_dir.mkdir(parents=True, exist_ok=True)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_path = detail_dir / "test_classification_report.csv"
        pd.DataFrame(report).transpose().to_csv(report_path, index_label="class_or_average")
        labels = sorted(set(np.asarray(y_test).tolist()) | set(np.asarray(y_pred).tolist()), key=lambda x: str(x))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        confusion_path = detail_dir / "test_confusion_matrix.csv"
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            confusion_path, index_label="actual\\predicted"
        )

        all_results[target] = {
            "candidate_name": model_info["candidate_name"],
            "model_name": model_info["model_name"],
            "test_rows": int(len(y_test)),
            "retained_features": int(len(kept_indices)),
            "metrics": metrics,
            "prediction_seconds": round(predict_seconds, 6),
            "test_records_per_second": round(len(y_test) / max(predict_seconds, 1e-12), 2),
            "classification_report": str(report_path.relative_to(ROOT)),
            "confusion_matrix": str(confusion_path.relative_to(ROOT)),
        }
        print(
            f"{target}: accuracy={metrics['accuracy']:.4f}; macro-F1={metrics['macro_f1']:.4f}; "
            f"balanced accuracy={metrics['balanced_accuracy']:.4f}"
        )

    summary = {
        "phase": 5,
        "stage": "5C_one_time_final_test_evaluation",
        "test_split_used": True,
        "models_were_frozen_before_test": True,
        "no_test_based_tuning_performed": True,
        "test_rows": int(X_test.shape[0]),
        "results": all_results,
        "limitations": manifest.get("limitations", []),
        "interpretation_warning": "This is a final evaluation on the held-out random record split. Because earlier exploratory audits used the full dataset and no verified device/session split exists, results do not establish generalization to unseen devices, sessions, or real buildings. Do not tune or choose models using these test results.",
    }
    summary_path = OUTPUT_DIR / "phase5_final_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Final test summary: {summary_path}")
    print("FINAL TEST EVALUATION COMPLETE. Do not tune or retrain based on this test set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
