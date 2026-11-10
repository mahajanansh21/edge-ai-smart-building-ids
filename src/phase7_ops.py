"""Operational helpers for Phase 7 offline replay, alerts, and run exports.

This module performs inference only. It never trains models or opens the held-out
Phase 5 test matrices/labels.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd

from src.edge_inference import predict_dataframe

CATEGORICAL_SOURCE_FEATURES = {
    "http.request.method",
    "http.request.version",
    "mqtt.protoname",
    "mqtt.topic",
}


def load_training_profile(train_path: Path, source_features: list[str]) -> dict[str, Any]:
    """Build manual-input defaults and simple univariate checks from TRAIN only."""
    if not train_path.is_file():
        raise FileNotFoundError(f"Training split not found: {train_path}")
    frame = pd.read_csv(train_path, usecols=source_features, low_memory=False)
    profile: dict[str, Any] = {"numeric": {}, "categorical": {}, "source": str(train_path)}
    for column in source_features:
        series = frame[column]
        if column in CATEGORICAL_SOURCE_FEATURES:
            values = series.dropna().astype(str)
            if values.empty:
                profile["categorical"][column] = {"default": "", "known_values": []}
            else:
                counts = values.value_counts()
                profile["categorical"][column] = {
                    "default": str(counts.index[0]),
                    "known_values": [str(value) for value in counts.index[:30]],
                }
        else:
            numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if numeric.empty:
                profile["numeric"][column] = {
                    "default": 0.0, "q01": None, "q99": None, "minimum": None, "maximum": None
                }
            else:
                profile["numeric"][column] = {
                    "default": float(numeric.median()),
                    "q01": float(numeric.quantile(0.01)),
                    "q99": float(numeric.quantile(0.99)),
                    "minimum": float(numeric.min()),
                    "maximum": float(numeric.max()),
                }
    return profile


def build_manual_record(values: dict[str, Any], source_features: list[str]) -> pd.DataFrame:
    """Create a one-row input frame in the exact Phase 3 source feature order."""
    missing = [feature for feature in source_features if feature not in values]
    if missing:
        raise ValueError("Manual record is missing fields: " + ", ".join(missing))
    row = {feature: values[feature] for feature in source_features}
    return pd.DataFrame([row], columns=source_features)


def diagnose_input(frame: pd.DataFrame, source_features: list[str], profile: dict[str, Any]) -> dict[str, Any]:
    """Simple per-column range/category diagnostics; not a full OOD detector."""
    missing_columns = [name for name in source_features if name not in frame.columns]
    missing_cells = 0
    numeric_outside = 0
    numeric_checked = 0
    categorical_unseen = 0
    categorical_checked = 0
    invalid_numeric = 0

    for column in source_features:
        if column not in frame.columns:
            continue
        values = frame[column]
        missing_cells += int(values.isna().sum())
        if column in CATEGORICAL_SOURCE_FEATURES:
            known = set(profile.get("categorical", {}).get(column, {}).get("known_values", []))
            nonmissing = values.dropna().astype(str)
            categorical_checked += len(nonmissing)
            categorical_unseen += int((~nonmissing.isin(known)).sum()) if known else len(nonmissing)
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            invalid_numeric += int((values.notna() & numeric.isna()).sum())
            stats = profile.get("numeric", {}).get(column, {})
            lower, upper = stats.get("q01"), stats.get("q99")
            valid = numeric.dropna()
            numeric_checked += len(valid)
            if lower is not None and upper is not None:
                numeric_outside += int(((valid < lower) | (valid > upper)).sum())

    return {
        "rows": int(len(frame)),
        "missing_required_columns": missing_columns,
        "missing_cells": missing_cells,
        "invalid_numeric_cells": invalid_numeric,
        "numeric_values_outside_training_q01_q99": numeric_outside,
        "numeric_values_checked": numeric_checked,
        "unseen_categorical_values": categorical_unseen,
        "categorical_values_checked": categorical_checked,
        "note": "Univariate checks only; this is not a calibrated out-of-distribution detector.",
    }


def build_alert_feed(predictions: pd.DataFrame, run_id: str, created_at: str | None = None) -> pd.DataFrame:
    """Turn model-predicted attacks into review alerts; never claim confirmation."""
    if predictions.empty or "Predicted_Binary" not in predictions.columns:
        return pd.DataFrame(columns=[
            "Alert_ID", "Run_ID", "Created_At_UTC", "Predicted_Attack_Type", "Review_Status"
        ])
    attacks = predictions.loc[predictions["Predicted_Binary"].eq("Attack")].copy().reset_index(drop=True)
    if attacks.empty:
        return pd.DataFrame(columns=[
            "Alert_ID", "Run_ID", "Created_At_UTC", "Predicted_Attack_Type", "Review_Status"
        ])
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    alerts = pd.DataFrame({
        "Alert_ID": [f"{run_id}-A{i:05d}" for i in range(1, len(attacks) + 1)],
        "Run_ID": run_id,
        "Created_At_UTC": timestamp,
        "Predicted_Attack_Type": attacks.get("Predicted_Attack_Type", "Unknown").astype(str),
        "Review_Status": "Model-predicted attack — review required",
    })
    for optional in ("Binary_Attack_Score_Uncalibrated", "Multiclass_Top_Score_Uncalibrated"):
        if optional in attacks.columns:
            alerts[optional] = attacks[optional].to_numpy()
    if "Input_Source" in attacks.columns:
        alerts["Input_Source"] = attacks["Input_Source"].to_numpy()
    return alerts


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Produce counts for a run without interpreting predictions as incidents."""
    total = int(len(predictions))
    attack_mask = predictions.get("Predicted_Binary", pd.Series(dtype=str)).eq("Attack")
    attacks = int(attack_mask.sum())
    normal = total - attacks
    type_counts = (
        predictions.loc[attack_mask, "Predicted_Attack_Type"].astype(str).value_counts().to_dict()
        if attacks and "Predicted_Attack_Type" in predictions.columns else {}
    )
    return {
        "records_processed": total,
        "predicted_attacks": attacks,
        "predicted_normal": normal,
        "predicted_attack_types": {str(k): int(v) for k, v in type_counts.items()},
        "warning": "Counts are model predictions, not independently verified security incidents.",
    }


def run_offline_replay(
    frame: pd.DataFrame,
    bundle: Any,
    batch_size: int = 50,
    pause_seconds: float = 0.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replay rows in bounded batches through both frozen models."""
    if frame.empty:
        raise ValueError("No records were selected for replay.")
    if len(frame) > 5000:
        raise ValueError("A single replay is limited to 5,000 records.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if pause_seconds < 0 or pause_seconds > 1:
        raise ValueError("pause_seconds must be between 0 and 1.")

    wall_start = time.perf_counter()
    results: list[pd.DataFrame] = []
    transform_s = binary_s = multiclass_s = 0.0
    total = len(frame)
    completed = 0
    for start in range(0, total, batch_size):
        chunk = frame.iloc[start:start + batch_size].copy()
        predicted, perf = predict_dataframe(chunk, bundle)
        predicted.insert(0, "Replay_Sequence", range(start + 1, start + len(chunk) + 1))
        results.append(predicted)
        transform_s += perf["transform_seconds"]
        binary_s += perf["binary_predict_seconds"]
        multiclass_s += perf["multiclass_predict_seconds"]
        completed += len(chunk)
        if progress_callback:
            progress_callback(completed, total)
        if pause_seconds:
            time.sleep(pause_seconds)

    wall_seconds = time.perf_counter() - wall_start
    combined = pd.concat(results, ignore_index=True)
    processing_seconds = transform_s + binary_s + multiclass_s
    performance = {
        "rows": float(total),
        "batches": float(len(results)),
        "batch_size": float(batch_size),
        "transform_seconds": transform_s,
        "binary_predict_seconds": binary_s,
        "multiclass_predict_seconds": multiclass_s,
        "model_processing_seconds": processing_seconds,
        "wall_seconds_including_optional_pacing": wall_seconds,
        "records_per_second_model_processing": total / max(processing_seconds, 1e-9),
        "records_per_second_wall_clock": total / max(wall_seconds, 1e-9),
    }
    return combined, performance


def save_run_artifacts(
    predictions: pd.DataFrame,
    performance: dict[str, Any],
    output_root: Path,
    run_id: str,
    source: str,
) -> Path:
    """Persist prediction, alert, and performance outputs for project evidence."""
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    alerts = build_alert_feed(predictions, run_id)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    alerts.to_csv(run_dir / "threat_alerts.csv", index=False)
    payload = {
        "run_id": run_id,
        "source": source,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summarize_predictions(predictions),
        "performance": performance,
        "alert_count": int(len(alerts)),
        "scope": "Offline/CSV/manual inference only; no live packet capture.",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_dir


def sample_validation_records(
    frame: pd.DataFrame, n_records: int, random_state: int = 42, strategy: str = "stratified"
) -> pd.DataFrame:
    """Select reproducible validation rows for replay; labels remain reference-only."""
    if n_records < 1 or n_records > len(frame):
        raise ValueError(f"n_records must be between 1 and {len(frame)}.")
    if strategy == "random":
        return frame.sample(n=n_records, random_state=random_state).reset_index(drop=True)
    if strategy != "stratified":
        raise ValueError("strategy must be 'random' or 'stratified'.")
    if "Attack_type" not in frame.columns:
        raise ValueError("Stratified replay requires the Attack_type reference column.")

    groups = {str(label): group for label, group in frame.groupby("Attack_type", sort=True)}
    labels = list(groups)
    base, remainder = divmod(n_records, len(labels))
    pieces: list[pd.DataFrame] = []
    remaining = n_records
    for index, label in enumerate(labels):
        requested = base + (1 if index < remainder else 0)
        take = min(requested, len(groups[label]))
        if take:
            pieces.append(groups[label].sample(n=take, random_state=random_state + index))
            remaining -= take
    if remaining > 0:
        selected_indices = set(pd.concat(pieces).index) if pieces else set()
        leftovers = frame.loc[~frame.index.isin(selected_indices)]
        pieces.append(leftovers.sample(n=remaining, random_state=random_state + 999))
    return pd.concat(pieces).sample(frac=1, random_state=random_state).reset_index(drop=True)
