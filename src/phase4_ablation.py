"""Phase 4 reliability check: controlled feature-ablation experiments.

Uses the existing train-fitted expanded36_normalized matrices. It only removes
transformed columns; it does not refit preprocessing, read the test split, or
change the original Phase 4 models/reports.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.phase4_utils import (
    RANDOM_SEED,
    evaluate_predictions,
    load_sparse_matrix,
    load_target,
    make_estimator,
    save_feature_importance,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "processed" / "model_inputs" / "expanded36_normalized"
OUTPUT_DIR = ROOT / "outputs" / "metrics" / "phase4" / "ablation"
MODEL_DIR = ROOT / "models" / "phase4_ablation"

# The names are derived from the feature_names.json supplied by Phase 3.
ABLATIONS = {
    "full_expanded43": set(),
    "no_tcp_sequence_ack_checksum": {
        "numeric__tcp.seq",
        "numeric__tcp.ack",
        "numeric__tcp.ack_raw",
        "numeric__tcp.checksum",
    },
    "no_tcp_family": "PREFIX:numeric__tcp.",
    "no_http_family": (
        "PREFIX:numeric__http.",
        "PREFIX:categorical__http.",
    ),
    "no_mqtt_dns_families": (
        "PREFIX:numeric__mqtt.",
        "PREFIX:numeric__dns.",
        "PREFIX:categorical__mqtt.",
    ),
}

CANDIDATE_MODELS = {
    "Attack_label": "random_forest",
    "Attack_type": "decision_tree",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run validation-only feature ablations on expanded36_normalized."
    )
    parser.add_argument(
        "--targets", nargs="+", choices=["Attack_label", "Attack_type"],
        default=["Attack_label", "Attack_type"],
    )
    parser.add_argument(
        "--ablations", nargs="+", choices=list(ABLATIONS), default=list(ABLATIONS),
    )
    return parser.parse_args()


def columns_to_drop(feature_names: list[str], rule) -> set[str]:
    if isinstance(rule, set):
        return set(rule)
    rules = [rule] if isinstance(rule, str) else list(rule)
    dropped: set[str] = set()
    for item in rules:
        if item.startswith("PREFIX:"):
            prefix = item.split(":", 1)[1]
            dropped.update(name for name in feature_names if name.startswith(prefix))
    return dropped


def validate_feature_names(feature_names: list[str], n_features: int) -> None:
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names.json has {len(feature_names)} names, but matrix has {n_features} columns."
        )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Duplicate transformed feature names found.")


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PHASE 4: FEATURE ABLATION / RELIABILITY CHECK")
    print("=" * 78)
    print("Validation only. X_test.npz and y_test.csv will NOT be opened.")
    print("Preprocessing remains the Phase 3 train-fitted preprocessing.")
    print(f"Targets: {', '.join(args.targets)}")
    print(f"Ablations: {', '.join(args.ablations)}")

    X_train = load_sparse_matrix(INPUT_DIR / "X_train.npz")
    X_val = load_sparse_matrix(INPUT_DIR / "X_validation.npz")
    with (INPUT_DIR / "feature_names.json").open("r", encoding="utf-8") as f:
        feature_names = json.load(f)
    validate_feature_names(feature_names, X_train.shape[1])
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Training and validation feature dimensions do not match.")

    result_rows = []
    failures = []
    total_start = time.perf_counter()

    for target in args.targets:
        y_train = load_target(INPUT_DIR / "y_train.csv", target)
        y_val = load_target(INPUT_DIR / "y_validation.csv", target)
        if len(y_train) != X_train.shape[0] or len(y_val) != X_val.shape[0]:
            raise ValueError(f"Target/matrix row mismatch for {target}.")

        model_name = CANDIDATE_MODELS[target]
        for ablation_name in args.ablations:
            dropped = columns_to_drop(feature_names, ABLATIONS[ablation_name])
            unknown = dropped.difference(feature_names)
            if unknown:
                raise ValueError(f"Requested drop features absent from matrix: {sorted(unknown)}")
            kept_indices = [i for i, name in enumerate(feature_names) if name not in dropped]
            kept_names = [feature_names[i] for i in kept_indices]
            if not kept_indices:
                raise ValueError(f"Ablation {ablation_name} removed every feature.")

            print(f"\n[{target} | {ablation_name} | {model_name}]")
            print(
                f"  Removed {len(dropped)} columns; retaining {len(kept_indices)} / "
                f"{len(feature_names)} transformed columns."
            )
            start = time.perf_counter()
            X_train_part = X_val_part = estimator = y_pred = None
            try:
                X_train_part = X_train[:, kept_indices].tocsr()
                X_val_part = X_val[:, kept_indices].tocsr()
                estimator = make_estimator(model_name)
                estimator.fit(X_train_part, y_train)
                y_pred = estimator.predict(X_val_part)
                fit_predict_seconds = time.perf_counter() - start

                metrics, report, confusion = evaluate_predictions(y_val, y_pred, target)
                report_dir = OUTPUT_DIR / target / ablation_name
                report_dir.mkdir(parents=True, exist_ok=True)
                report.to_csv(report_dir / "classification_report.csv", index=False)
                confusion.to_csv(report_dir / "confusion_matrix.csv")
                pd.DataFrame({"feature": kept_names}).to_csv(
                    report_dir / "retained_features.csv", index=False
                )
                pd.DataFrame({"dropped_feature": sorted(dropped)}).to_csv(
                    report_dir / "dropped_features.csv", index=False
                )
                importance_path = report_dir / "top_features.csv"
                save_feature_importance(estimator, kept_names, importance_path)

                model_path = MODEL_DIR / target / f"{ablation_name}_{model_name}.joblib"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(
                    {"estimator": estimator, "retained_feature_names": kept_names,
                     "target": target, "ablation": ablation_name},
                    model_path,
                    compress=3,
                )

                row = {
                    "target": target,
                    "ablation": ablation_name,
                    "model": model_name,
                    "removed_features": len(dropped),
                    "retained_features": len(kept_indices),
                    **metrics,
                    "fit_predict_seconds": round(fit_predict_seconds, 3),
                    "model_path": str(model_path.relative_to(ROOT)),
                    "classification_report": str((report_dir / "classification_report.csv").relative_to(ROOT)),
                    "confusion_matrix": str((report_dir / "confusion_matrix.csv").relative_to(ROOT)),
                    "status": "success",
                }
                result_rows.append(row)
                print(
                    f"  Validation: accuracy={metrics['accuracy']:.4f}, "
                    f"macro-F1={metrics['macro_f1']:.4f}, "
                    f"balanced-accuracy={metrics['balanced_accuracy']:.4f}"
                )
                print(f"  Saved model: {model_path}")
            except Exception as exc:
                failures.append({
                    "target": target,
                    "ablation": ablation_name,
                    "model": model_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"  FAILED: {type(exc).__name__}: {exc}")
            finally:
                del X_train_part, X_val_part, estimator, y_pred
                gc.collect()

    results_path = OUTPUT_DIR / "ablation_comparison.csv"
    if result_rows:
        results = pd.DataFrame(result_rows)
        baseline = results[results["ablation"] == "full_expanded43"][
            ["target", "macro_f1", "balanced_accuracy"]
        ].rename(columns={
            "macro_f1": "baseline_macro_f1",
            "balanced_accuracy": "baseline_balanced_accuracy",
        })
        results = results.merge(baseline, on="target", how="left")
        results["macro_f1_change_vs_baseline"] = (
            results["macro_f1"] - results["baseline_macro_f1"]
        )
        results["balanced_accuracy_change_vs_baseline"] = (
            results["balanced_accuracy"] - results["baseline_balanced_accuracy"]
        )
        results = results.sort_values(["target", "macro_f1"], ascending=[True, False])
        results.to_csv(results_path, index=False)
    else:
        results = pd.DataFrame()

    summary = {
        "phase": 4,
        "experiment": "feature_ablation_reliability_check",
        "purpose": "measure validation performance sensitivity to selected feature groups",
        "test_split_used": False,
        "preprocessing": "existing Phase 3 preprocessor fitted on training partition only",
        "models": CANDIDATE_MODELS,
        "ablations_requested": args.ablations,
        "targets_requested": args.targets,
        "successful_runs": len(result_rows),
        "failed_runs": len(failures),
        "elapsed_seconds": round(time.perf_counter() - total_start, 3),
        "comparison_csv": str(results_path.relative_to(ROOT)),
        "failures": failures,
        "warning": (
            "Ablation results are validation-only and remain subject to the known limitation "
            "that earlier exploratory audits used the full dataset. They do not establish "
            "generalization to unseen devices, sessions, or environments."
        ),
    }
    write_json(OUTPUT_DIR / "ablation_run_summary.json", summary)

    print("\n" + "=" * 78)
    print("FEATURE ABLATION RUN FINISHED")
    print("=" * 78)
    print(f"Successful runs: {len(result_rows)} | Failed runs: {len(failures)}")
    print(f"Comparison CSV: {results_path}")
    if not results.empty:
        display_cols = [
            "target", "ablation", "removed_features", "retained_features",
            "macro_f1", "balanced_accuracy", "accuracy",
            "macro_f1_change_vs_baseline",
        ]
        print(results[display_cols].to_string(index=False))
    print("The test split was not read or evaluated.")


if __name__ == "__main__":
    main()
