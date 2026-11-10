import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.preprocessing import (
    CORE_FEATURES,
    EXPANDED_FEATURES,
    build_preprocessor,
    get_feature_columns,
    normalize_features,
)


def make_core_frame(rows=12):
    data = {}
    for i, column in enumerate(CORE_FEATURES):
        data[column] = np.array([(j + i) % 3 for j in range(rows)], dtype=float)
    return pd.DataFrame(data)


def make_expanded_frame(rows=12):
    data = {}
    categorical = {
        "http.request.method": ["0.0", "0", "GET", "POST"],
        "http.request.version": ["0.0", "HTTP/1.1", "HTTP/1.0", "bad payload"],
        "mqtt.conack.flags": ["0.0", "0", "0x00000000", "0.0"],
        "mqtt.protoname": ["0.0", "0", "MQTT", "0.0"],
        "mqtt.topic": ["0.0", "0", "Temperature_and_Humidity", "0.0"],
    }
    for i, column in enumerate(EXPANDED_FEATURES):
        if column in categorical:
            values = categorical[column]
            data[column] = [values[j % len(values)] for j in range(rows)]
        elif column == "dns.qry.name.len":
            values = ["0", "1", "3.debian.pool.ntp.org", "2"]
            data[column] = [values[j % len(values)] for j in range(rows)]
        else:
            data[column] = np.array([(j + i) % 5 for j in range(rows)], dtype=float)
    return pd.DataFrame(data)


def test_feature_sets_exclude_targets():
    for name in ("core9", "expanded36_normalized"):
        columns = get_feature_columns(name)
        assert "Attack_label" not in columns
        assert "Attack_type" not in columns


def test_core9_pipeline_produces_finite_matrix():
    X = make_core_frame()
    normalized = normalize_features(X, "core9")
    pipeline = build_preprocessor("core9")
    transformed = pipeline.fit_transform(normalized)
    assert transformed.shape[0] == len(X)
    values = transformed.data if sparse.issparse(transformed) else np.asarray(transformed)
    assert np.isfinite(values).all()


def test_zero_placeholders_are_normalized_consistently():
    X = make_expanded_frame(4)
    normalized = normalize_features(X, "expanded36_normalized")
    assert normalized.loc[0, "mqtt.protoname"] == "__NO_VALUE__"
    assert normalized.loc[1, "mqtt.protoname"] == "__NO_VALUE__"
    assert normalized.loc[0, "mqtt.conack.flags"] == "__NO_VALUE__"
    assert normalized.loc[1, "mqtt.conack.flags"] == "__NO_VALUE__"


def test_malformed_http_version_is_not_preserved_as_payload_text():
    X = make_expanded_frame(4)
    normalized = normalize_features(X, "expanded36_normalized")
    assert normalized.loc[3, "http.request.version"] == "__NO_VALUE__"


def test_unseen_categories_do_not_crash_transform():
    train = make_expanded_frame(8)
    test = make_expanded_frame(4)
    train["http.request.method"] = "GET"
    test["http.request.method"] = ["PATCH", "TRACE", "OPTIONS", "DELETE"]

    pipeline = build_preprocessor("expanded36_normalized")
    train_normalized = normalize_features(train, "expanded36_normalized")
    test_normalized = normalize_features(test, "expanded36_normalized")
    pipeline.fit(train_normalized)
    transformed = pipeline.transform(test_normalized)
    assert transformed.shape[0] == len(test)


def test_missing_feature_column_raises_clear_error():
    X = make_core_frame().drop(columns=[CORE_FEATURES[0]])
    with pytest.raises(ValueError, match="Missing required feature"):
        normalize_features(X, "core9")
