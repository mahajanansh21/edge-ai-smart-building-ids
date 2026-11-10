"""Phase 4: fit candidate classifiers and compare them on validation only.

The test split is intentionally NOT loaded or evaluated in this phase.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from src.phase4_utils import (
    FEATURE_SETS,
    MODEL_NAMES,
    RANDOM_SEED,
    TARGETS,
    evaluate_predictions,
    load_sparse_matrix,
    load_target,
    make_estimator,
    save_feature_importance,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "data" / "processed" / "model_inputs"
MODEL_ROOT = ROOT / "models" / "phase4"
METRICS_ROOT = ROOT / "outputs" / "metrics" / "phase4"
LOG_ROOT = ROOT / "outputs" / "logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 4 classifiers and rank using validation data.")
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller smoke experiment: logistic regression and decision tree only.",
    )
    return parser.parse_args()


def train_one(feature_set: str, target: str, model_name: str,
              X_train, y_train: np.ndarray, X_val, y_val: np.ndarray,
              feature_names: list[str]) -> dict:
    output_dir = MODEL_ROOT / feature_set / target
    report_dir = METRICS_ROOT / feature_set / target
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.joblib"
    print(f"\n[{feature_set} | {target} | {model_name}]", flush=True)
    print(f"  Fitting on {X_train.shape[0]:,} rows x {X_train.shape[1]} features", flush=True)

    estimator = make_estimator(model_name)
    start_fit = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start_fit

    convergence_messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    other_warning_messages = [
        f"{item.category.__name__}: {item.message}"
        for item in caught
        if not issubclass(item.category, ConvergenceWarning)
    ]

    start_predict = time.perf_counter()
    y_pred = estimator.predict(X_val)
    predict_seconds = time.perf_counter() - start_predict

    metrics, report_frame, confusion_frame = evaluate_predictions(y_val, y_pred, target)
    report_path = report_dir / f"{model_name}_classification_report.csv"
    confusion_path = report_dir / f"{model_name}_confusion_matrix.csv"
    report_frame.to_csv(report_path, index=False)
    confusion_frame.to_csv(confusion_path)

    # Save the trained estimator only after fitting and validation prediction succeed.
    joblib.dump(estimator, model_path, compress=3)
    importance_path = report_dir / f"{model_name}_top_features.csv"
    save_feature_importance(estimator, feature_names, importance_path)

    row = {
        "feature_set": feature_set,
        "target": target,
        "model": model_name,
        **metrics,
        "fit_seconds": round(fit_seconds, 3),
        "validation_predict_seconds": round(predict_seconds, 3),
        "validation_records_per_second": round(len(y_val) / max(predict_seconds, 1e-9), 2),
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_val.shape[0]),
        "transformed_features": int(X_train.shape[1]),
        "model_path": str(model_path.relative_to(ROOT)),
        "classification_report": str(report_path.relative_to(ROOT)),
        "confusion_matrix": str(confusion_path.relative_to(ROOT)),
        "feature_importance": str(importance_path.relative_to(ROOT)) if importance_path.exists() else "",
        "convergence_warning": " | ".join(convergence_messages),
        "other_warnings": " | ".join(other_warning_messages),
        "status": "success",
    }
    print(
        "  Validation: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"macro-F1={metrics['macro_f1']:.4f}, "
        f"weighted-F1={metrics['weighted_f1']:.4f}, "
        f"balanced-accuracy={metrics['balanced_accuracy']:.4f}",
        flush=True,
    )
    print(f"  Fit time: {fit_seconds:.1f}s | Model saved: {model_path}", flush=True)
    if convergence_messages:
        print("  WARNING: Logistic Regression reached its iteration limit; inspect the warning in the metrics CSV.", flush=True)
    return row


def main():
    args = parse_args()
    models = ["logistic_regression", "decision_tree"] if args.quick else args.models

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PHASE 4: MODEL TRAINING + VALIDATION COMPARISON")
    print("=" * 78)
    print("The test split will NOT be loaded, scored, or used for model selection.")
    print(f"Feature sets: {', '.join(args.feature_sets)}")
    print(f"Targets:      {', '.join(args.targets)}")
    print(f"Models:       {', '.join(models)}")
    print(f"Random seed:  {RANDOM_SEED}")
    if args.quick:
        print("QUICK MODE: each model uses a stratified 25,000-row training sample (or all rows if fewer).")

    result_rows = []
    failures = []
    start_all = time.perf_counter()

    for feature_set in args.feature_sets:
        input_dir = INPUT_ROOT / feature_set
        X_train = load_sparse_matrix(input_dir / "X_train.npz")
        X_val = load_sparse_matrix(input_dir / "X_validation.npz")
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError(f"Feature dimension mismatch in {feature_set}: train={X_train.shape}, validation={X_val.shape}")
        if X_train.shape[1] == 0:
            raise ValueError(f"No features available for {feature_set}")

        feature_names_path = input_dir / "feature_names.json"
        if not feature_names_path.exists():
            raise FileNotFoundError(f"Feature-name file is missing: {feature_names_path}")
        with feature_names_path.open("r", encoding="utf-8") as handle:
            feature_names = json.load(handle)
        if len(feature_names) != X_train.shape[1]:
            raise ValueError(
                f"Feature-name count ({len(feature_names)}) does not match matrix dimension ({X_train.shape[1]})"
            )

        for target in args.targets:
            y_train = load_target(input_dir / "y_train.csv", target)
            y_val = load_target(input_dir / "y_validation.csv", target)
            if len(y_train) != X_train.shape[0] or len(y_val) != X_val.shape[0]:
                raise ValueError(
                    f"Label/matrix row mismatch for {feature_set}/{target}: "
                    f"train {len(y_train)} vs {X_train.shape[0]}, validation {len(y_val)} vs {X_val.shape[0]}"
                )
            if set(y_val.tolist()) - set(y_train.tolist()):
                raise ValueError(f"Validation contains classes absent from training for {feature_set}/{target}")

            X_train_run = X_train
            y_train_run = y_train
            if args.quick and len(y_train) > 25_000:
                sampled_indices, _ = train_test_split(
                    np.arange(len(y_train)),
                    train_size=25_000,
                    random_state=RANDOM_SEED,
                    stratify=y_train,
                )
                X_train_run = X_train[sampled_indices]
                y_train_run = y_train[sampled_indices]

            for model_name in models:
                try:
                    row = train_one(
                        feature_set, target, model_name,
                        X_train_run, y_train_run, X_val, y_val, feature_names,
                    )
                    result_rows.append(row)
                except Exception as exc:  # Continue other candidates; failures are recorded.
                    failure = {
                        "feature_set": feature_set,
                        "target": target,
                        "model": model_name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(f"  FAILED: {failure['error']}", flush=True)
                finally:
                    gc.collect()

        del X_train, X_val
        gc.collect()

    comparison_path = METRICS_ROOT / "validation_model_comparison.csv"
    if result_rows:
        results = pd.DataFrame(result_rows)
        sort_cols = ["target", "feature_set", "macro_f1", "balanced_accuracy"]
        results = results.sort_values(sort_cols, ascending=[True, True, False, False])
        results.to_csv(comparison_path, index=False)
    else:
        results = pd.DataFrame()

    elapsed = time.perf_counter() - start_all
    summary = {
        "phase": 4,
        "purpose": "train candidate classifiers; compare on validation only",
        "test_split_used": False,
        "feature_sets_requested": args.feature_sets,
        "targets_requested": args.targets,
        "models_requested": models,
        "quick_mode": bool(args.quick),
        "quick_training_sample_limit": 25_000 if args.quick else None,
        "random_seed": RANDOM_SEED,
        "successful_runs": len(result_rows),
        "failed_runs": len(failures),
        "elapsed_seconds": round(elapsed, 3),
        "comparison_csv": str(comparison_path.relative_to(ROOT)),
        "failures": failures,
        "selection_metric": "macro_f1, with balanced_accuracy as a secondary metric",
        "warning": (
            "Validation scores are for candidate comparison, not proof of generalization. "
            "The dataset has known formatting/collection-pattern risks; expanded36_normalized is experimental. "
            "Earlier exploratory audits used the full dataset, so the validation split is not a fully untouched confirmatory set. "
            "Keep the test split sealed until the final evaluation stage."
        ),
    }
    write_json(METRICS_ROOT / "phase4_run_summary.json", summary)

    print("\n" + "=" * 78)
    print("PHASE 4 TRAINING RUN FINISHED")
    print("=" * 78)
    print(f"Successful runs: {len(result_rows)} | Failed runs: {len(failures)}")
    print(f"Elapsed time:    {elapsed / 60:.1f} minutes")
    print(f"Comparison CSV:  {comparison_path}")
    if not results.empty:
        print("\nValidation leaders by target and feature set (macro-F1):")
        leaders = results.sort_values("macro_f1", ascending=False).groupby(
            ["target", "feature_set"], as_index=False
        ).first()
        for _, row in leaders.iterrows():
            print(
                f"  {row['target']} | {row['feature_set']}: {row['model']} "
                f"(macro-F1={row['macro_f1']:.4f}, balanced accuracy={row['balanced_accuracy']:.4f})"
            )
    if failures:
        print("\nSome runs failed. Read failures in phase4_run_summary.json before proceeding.")
    print("\nDo not evaluate the test split during Phase 4.")


if __name__ == "__main__":
    main()
