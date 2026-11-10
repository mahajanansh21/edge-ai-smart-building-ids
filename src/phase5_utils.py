"""Helpers for Phase 5 candidate freezing and one-time holdout evaluation."""
from __future__ import annotations

from typing import Sequence


def build_keep_indices(feature_names: Sequence[str], removed_features: Sequence[str]) -> list[int]:
    """Return the ordered indices retained after removing named transformed features."""
    names = list(feature_names)
    if len(names) != len(set(names)):
        raise ValueError("feature_names contains duplicate entries")
    missing = sorted(set(removed_features) - set(names))
    if missing:
        raise ValueError(f"Requested feature(s) not found: {missing}")
    removed = set(removed_features)
    keep = [i for i, name in enumerate(names) if name not in removed]
    if not keep:
        raise ValueError("Feature selection removed every feature")
    return keep


def validate_matrix_and_features(matrix, feature_names: Sequence[str], matrix_name: str) -> None:
    """Fail early if the transformed matrix is not aligned with its feature manifest."""
    if getattr(matrix, "ndim", None) != 2:
        raise ValueError(f"{matrix_name} must be a two-dimensional matrix")
    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"{matrix_name} has {matrix.shape[1]} columns but feature_names has {len(feature_names)} entries"
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{matrix_name} is empty")
