"""Leakage-aware preprocessing for Edge-IIoTset.

Two configurations are provided:
- core9: a conservative, all-numeric baseline.
- expanded36: a normalized research-comparator feature set.

The expanded36 set intentionally contains some features that may encode
collection/protocol artifacts. It is an experimental comparator, not a claim
that all 36 columns are production-safe. Phase 4 should compare both sets.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMNS = ["Attack_label", "Attack_type"]

CORE_FEATURES: List[str] = [
    "arp.opcode",
    "arp.hw.size",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",
]

EXPANDED_FEATURES: List[str] = [
    "arp.opcode",
    "arp.hw.size",
    "icmp.checksum",
    "icmp.seq_le",
    "http.content_length",
    "http.request.method",
    "http.request.version",
    "http.response",
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.checksum",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",
    "tcp.seq",
    "udp.stream",
    "udp.time_delta",
    "dns.qry.name",
    "dns.qry.name.len",
    "dns.qry.qu",
    "dns.retransmission",
    "dns.retransmit_request",
    "mqtt.conack.flags",
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msgtype",
    "mqtt.protoname",
    "mqtt.topic",
    "mqtt.topic_len",
    "mqtt.ver",
]

FEATURE_SETS: Dict[str, List[str]] = {
    "core9": CORE_FEATURES,
    "expanded36_normalized": EXPANDED_FEATURES,
}

# These columns are categorical in the expanded feature set. All other
# expanded columns (including dns.qry.name.len) are parsed numerically.
CATEGORICAL_COLUMNS: List[str] = [
    "http.request.method",
    "http.request.version",
    "mqtt.conack.flags",
    "mqtt.protoname",
    "mqtt.topic",
]

HTTP_METHODS = {
    "GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS",
    "TRACE", "CONNECT", "PATCH",
}
HTTP_VERSIONS = {"HTTP/1.0", "HTTP/1.1"}

ZERO_PLACEHOLDERS = {
    "", "0", "0.0", "0.00", "-0", "-0.0", "0x00000000"
}


def get_feature_columns(feature_set: str) -> List[str]:
    """Return a copy of the feature list for a named configuration."""
    if feature_set not in FEATURE_SETS:
        choices = ", ".join(FEATURE_SETS)
        raise ValueError(f"Unknown feature set {feature_set!r}. Choose: {choices}")
    return list(FEATURE_SETS[feature_set])


def normalize_features(X: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Apply deterministic, row-wise normalization without fitting statistics.

    This function does not inspect labels and does not learn anything from the
    data. Numeric imputation, scaling and category vocabulary are learned later
    by the sklearn Pipeline using X_train only.
    """
    columns = get_feature_columns(feature_set)
    missing = [column for column in columns if column not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    result = X.loc[:, columns].copy()

    if feature_set == "core9":
        for column in columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        result = result.replace([np.inf, -np.inf], np.nan)
        return result

    for column in CATEGORICAL_COLUMNS:
        values = result[column].astype("string").str.strip()
        upper = values.str.upper()

        # Canonicalize all known zero/empty placeholders. This prevents
        # "0" versus "0.0" formatting from becoming separate categories.
        zero_mask = values.str.lower().isin(ZERO_PLACEHOLDERS)
        values = values.mask(zero_mask, "__NO_VALUE__")
        upper = values.str.upper()

        if column == "http.request.method":
            # Only retain recognized HTTP methods; malformed/payload-like
            # strings are treated as unavailable rather than attack signatures.
            values = upper.where(upper.isin(HTTP_METHODS) | upper.eq("__NO_VALUE__"), "__NO_VALUE__")
        elif column == "http.request.version":
            # Retain valid versions only. Unusual strings in this field can
            # contain attack payload fragments in this dataset.
            values = upper.where(upper.isin(HTTP_VERSIONS) | upper.eq("__NO_VALUE__"), "__NO_VALUE__")
        else:
            # For MQTT text metadata, normalize formatting but retain semantic
            # values such as MQTT and Temperature_and_Humidity. These fields
            # remain an experimental feature group due observed label skew.
            values = values.replace({"0X00000000": "__NO_VALUE__"})

        # sklearn imputers expect ordinary NaN rather than pandas.NA.
        result[column] = values.astype(object).where(values.notna(), np.nan)

    numeric_columns = [c for c in columns if c not in CATEGORICAL_COLUMNS]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def build_preprocessor(feature_set: str) -> Pipeline:
    """Build an unfitted preprocessing pipeline for a feature set."""
    columns = get_feature_columns(feature_set)
    categorical = [c for c in CATEGORICAL_COLUMNS if c in columns]
    numeric = [c for c in columns if c not in categorical]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
        ]
    )

    transformers = [("numeric", numeric_pipeline, numeric)]
    if categorical:
        transformers.append(("categorical", categorical_pipeline, categorical))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Remove columns that become constant after preprocessing, based only on
    # the training partition on which this pipeline is fitted.
    return Pipeline(
        steps=[
            ("columns", column_transformer),
            ("remove_constant", VarianceThreshold(threshold=0.0)),
        ]
    )


def get_transformed_feature_names(fitted_pipeline: Pipeline) -> List[str]:
    """Return output feature names after the fitted variance filter."""
    column_names = fitted_pipeline.named_steps["columns"].get_feature_names_out()
    support = fitted_pipeline.named_steps["remove_constant"].get_support()
    return [str(name) for name, keep in zip(column_names, support) if keep]


def assert_finite_matrix(matrix, label: str = "transformed matrix") -> None:
    """Raise a clear error if a transformed dense or sparse matrix is non-finite."""
    if hasattr(matrix, "data") and hasattr(matrix, "tocsr"):
        values = matrix.data
    else:
        values = np.asarray(matrix)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN or infinite values after preprocessing")
