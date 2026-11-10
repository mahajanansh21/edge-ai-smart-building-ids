"""Targeted, validation-only ablation of the six dominant TCP features.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m src.phase4_targeted_tcp_ablation

The script loads Phase 3 TRAIN and VALIDATION matrices only. It never opens
X_test.npz or y_test.csv. Estimator hyperparameters are cloned from the
previously saved Phase 4 candidate models so the comparison stays aligned.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "processed" / "model_inputs" / "expanded36_normalized"
MODEL_DIR = ROOT / "models" / "phase4" / "expanded36_normalized"
OUTPUT_DIR = ROOT / "outputs" / "metrics" / "phase4" / "targeted_tcp_ablation"

# Six highest-ranked features across the Phase 4 binary/multiclass feature
# importance reports. Names must match feature_names.json exactly.
TCP_FEATURES = [
    "numeric__tcp.seq",
    "numeric__tcp.ack",
    "numeric__tcp.ack_raw",
    "numeric__tcp.checksum",
    "numeric__tcp.flags",
    "numeric__tcp.len",
]

ABLATIONS: Dict[str, List[str]] = {
    "full_expanded43": [],
    "without_tcp_seq": ["numeric__tcp.seq"],
    "without_tcp_ack": ["numeric__tcp.ack"],
    "without_tcp_ack_raw": ["numeric__tcp.ack_raw"],
    "without_tcp_checksum": ["numeric__tcp.checksum"],
    "without_tcp_flags": ["numeric__tcp.flags"],
    "without_tcp_len": ["numeric__tcp.len"],
    "without_top6_tcp_features": TCP_FEATURES.copy(),
}


def build_keep_indices(feature_names: List[str], removed_features: List[str]) -> List[int]:
    """Return column indices to retain; fail clearly on misspelled features."""
    missing = sorted(set(removed_features) - set(feature_names))
    if missing:
        raise ValueError(f"Requested feature(s) not found in feature_names.json: {missing}")
    return [i for i, name in enumerate(feature_names) if name not in set(removed_features)]


def _load_targets(path: Path, expected_rows: int, required: List[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 3 label file not found: {path}")
    frame = pd.read_csv(path)
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing target column(s) {missing}. "
            f"Available columns: {list(frame.columns)}"
        )
    if len(frame) != expected_rows:
        raise ValueError(
            f"Row count mismatch for {path}: labels={len(frame)}, matrix={expected_rows}"
        )
    return frame


def _write_classification_outputs(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target: str,
    ablation: str,
) -> None:
    detail_dir = OUTPUT_DIR / "details" / target / ablation
    detail_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        detail_dir / "classification_report.csv", index_label="class_or_average"
    )

    labels = sorted(pd.unique(pd.concat([y_true, pd.Series(y_pred)], ignore_index=True)).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        detail_dir / "confusion_matrix.csv", index_label="actual\\predicted"
    )


def main() -> int:
    print("=" * 78)
    print("PHASE 4: TARGETED TCP FEATURE ABLATION")
    print("Validation only; test matrices and labels will NOT be opened.")
    print("Saved Phase 4 estimator parameters will be reused via sklearn.clone().")
    print("=" * 78)

    required_paths = [
        FEATURE_DIR / "X_train.npz",
        FEATURE_DIR / "X_validation.npz",
        FEATURE_DIR / "y_train.csv",
        FEATURE_DIR / "y_validation.csv",
        FEATURE_DIR / "feature_names.json",
        MODEL_DIR / "Attack_label" / "random_forest.joblib",
        MODEL_DIR / "Attack_type" / "decision_tree.joblib",
    ]
    missing_paths = [str(p) for p in required_paths if not p.is_file()]
    if missing_paths:
        print("ERROR: Required Phase 3/Phase 4 files are missing:", file=sys.stderr)
        for item in missing_paths:
            print(f"  - {item}", file=sys.stderr)
        return 2

    with (FEATURE_DIR / "feature_names.json").open("r", encoding="utf-8") as f:
        feature_names = json.load(f)
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("feature_names.json must contain a non-empty JSON list")

    # Intentionally load only train and validation matrices.
    X_train = load_npz(FEATURE_DIR / "X_train.npz")
    X_val = load_npz(FEATURE_DIR / "X_validation.npz")
    if X_train.shape[1] != len(feature_names) or X_val.shape[1] != len(feature_names):
        raise ValueError(
            "Feature dimension does not match feature_names.json: "
            f"train={X_train.shape}, validation={X_val.shape}, names={len(feature_names)}"
        )

    targets = ["Attack_label", "Attack_type"]
    y_train_frame = _load_targets(FEATURE_DIR / "y_train.csv", X_train.shape[0], targets)
    y_val_frame = _load_targets(FEATURE_DIR / "y_validation.csv", X_val.shape[0], targets)

    templates = {
        "Attack_label": joblib.load(MODEL_DIR / "Attack_label" / "random_forest.joblib"),
        "Attack_type": joblib.load(MODEL_DIR / "Attack_type" / "decision_tree.joblib"),
    }
    model_names = {"Attack_label": "random_forest", "Attack_type": "decision_tree"}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    start_all = time.perf_counter()
    expected_runs = len(targets) * len(ABLATIONS)
    print(f"Train matrix: {X_train.shape}; validation matrix: {X_val.shape}")
    print(f"Ablations: {len(ABLATIONS)}; target/model combinations: {expected_runs}")

    for target in targets:
        y_train = y_train_frame[target]
        y_val = y_val_frame[target]
        for ablation, removed in ABLATIONS.items():
            try:
                keep = build_keep_indices(feature_names, removed)
                Xtr = X_train[:, keep]
                Xv = X_val[:, keep]
                estimator = clone(templates[target])

                print(f"\n[{target} | {ablation} | {model_names[target]}]")
                print(f"  Removed {len(removed)} columns; retaining {len(keep)} / {len(feature_names)}.")
                fit_start = time.perf_counter()
                estimator.fit(Xtr, y_train)
                fit_seconds = time.perf_counter() - fit_start

                pred_start = time.perf_counter()
                predictions = estimator.predict(Xv)
                predict_seconds = time.perf_counter() - pred_start

                macro = f1_score(y_val, predictions, average="macro", zero_division=0)
                weighted = f1_score(y_val, predictions, average="weighted", zero_division=0)
                balanced = balanced_accuracy_score(y_val, predictions)
                accuracy = accuracy_score(y_val, predictions)
                _write_classification_outputs(y_val, predictions, target, ablation)

                row = {
                    "target": target,
                    "model": model_names[target],
                    "ablation": ablation,
                    "removed_features": len(removed),
                    "removed_feature_names": ";".join(removed),
                    "retained_features": len(keep),
                    "macro_f1": macro,
                    "weighted_f1": weighted,
                    "balanced_accuracy": balanced,
                    "accuracy": accuracy,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                }
                rows.append(row)
                print(
                    f"  Validation: macro-F1={macro:.4f}, balanced-accuracy={balanced:.4f}, "
                    f"accuracy={accuracy:.4f}"
                )
                print(f"  Fit: {fit_seconds:.2f}s; predict: {predict_seconds:.2f}s")
            except Exception as exc:  # retain other runs if one experiment fails
                failures.append({"target": target, "ablation": ablation, "error": repr(exc)})
                print(f"  ERROR: {exc}", file=sys.stderr)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["macro_f1_change_vs_baseline"] = np.nan
        for target in targets:
            baseline_rows = result[(result.target == target) & (result.ablation == "full_expanded43")]
            if not baseline_rows.empty:
                baseline = float(baseline_rows.iloc[0]["macro_f1"])
                mask = result.target == target
                result.loc[mask, "macro_f1_change_vs_baseline"] = result.loc[mask, "macro_f1"] - baseline
        result = result.sort_values(["target", "macro_f1"], ascending=[True, False])
        result.to_csv(OUTPUT_DIR / "targeted_tcp_ablation_comparison.csv", index=False)

    summary = {
        "phase": 4,
        "experiment": "targeted_single_feature_tcp_ablation",
        "purpose": "measure validation sensitivity to six individual high-importance TCP features and their combined removal",
        "test_split_used": False,
        "matrices_loaded": ["X_train.npz", "X_validation.npz"],
        "labels_loaded": ["y_train.csv", "y_validation.csv"],
        "models_reused_for_hyperparameters": model_names,
        "ablations_requested": list(ABLATIONS.keys()),
        "successful_runs": len(rows),
        "failed_runs": len(failures),
        "elapsed_seconds": round(time.perf_counter() - start_all, 3),
        "comparison_csv": "outputs/metrics/phase4/targeted_tcp_ablation/targeted_tcp_ablation_comparison.csv",
        "detail_reports_directory": "outputs/metrics/phase4/targeted_tcp_ablation/details",
        "failures": failures,
        "warning": "Validation-only diagnostic. Feature dependence does not prove leakage or generalization. Prior exploratory audits used the full dataset; keep the test split sealed until final evaluation.",
    }
    with (OUTPUT_DIR / "targeted_tcp_ablation_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("TARGETED TCP ABLATION FINISHED")
    print(f"Successful runs: {len(rows)} | Failed runs: {len(failures)}")
    print(f"Comparison CSV: {OUTPUT_DIR / 'targeted_tcp_ablation_comparison.csv'}")
    print(f"Run summary: {OUTPUT_DIR / 'targeted_tcp_ablation_run_summary.json'}")
    print("The test split was not read or evaluated.")
    print("=" * 78)
    return 0 if rows and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
