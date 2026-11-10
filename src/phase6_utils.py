"""Small, independently testable helpers for the Phase 6 dashboard."""
from __future__ import annotations

from typing import Iterable, Sequence


def retained_feature_indices(
    all_feature_names: Sequence[str], removed_feature: str
) -> list[int]:
    """Return original column indices except one exact transformed feature."""
    names = list(all_feature_names)
    if not names:
        raise ValueError("The transformed feature list is empty.")
    if len(names) != len(set(names)):
        raise ValueError("Transformed feature names must be unique.")
    if removed_feature not in names:
        raise ValueError(f"Required feature to remove was not found: {removed_feature}")
    indices = [i for i, name in enumerate(names) if name != removed_feature]
    if not indices:
        raise ValueError("Feature selection removed every feature.")
    return indices


def validate_source_columns(
    available_columns: Iterable[str], required_columns: Sequence[str]
) -> list[str]:
    """Return required columns absent from a supplied dataframe."""
    available = set(available_columns)
    return [name for name in required_columns if name not in available]


def classify_binary_label(value: object) -> str:
    """Convert the binary model's encoded prediction into a readable label."""
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unexpected binary prediction: {value!r}") from exc
    if numeric == 0:
        return "Normal"
    if numeric == 1:
        return "Attack"
    raise ValueError(f"Unexpected binary prediction: {value!r}")


def combine_detection(binary_prediction: object, attack_type_prediction: object) -> str:
    """Use the multiclass model's class only when the binary model says Attack."""
    binary_label = classify_binary_label(binary_prediction)
    if binary_label == "Normal":
        return "Normal"
    return str(attack_type_prediction)
