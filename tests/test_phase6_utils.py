import pytest

from src.phase6_utils import (
    classify_binary_label,
    combine_detection,
    retained_feature_indices,
    validate_source_columns,
)


def test_retained_feature_indices_removes_exact_name():
    names = ["numeric__a", "numeric__tcp.len", "numeric__b"]
    assert retained_feature_indices(names, "numeric__tcp.len") == [0, 2]


def test_retained_feature_indices_requires_feature():
    with pytest.raises(ValueError):
        retained_feature_indices(["a", "b"], "missing")


def test_validate_source_columns():
    assert validate_source_columns(["a", "b"], ["a", "b", "c"]) == ["c"]


def test_binary_labels_and_combined_detection():
    assert classify_binary_label(0) == "Normal"
    assert classify_binary_label(1) == "Attack"
    assert combine_detection(0, "DDoS_HTTP") == "Normal"
    assert combine_detection(1, "DDoS_HTTP") == "DDoS_HTTP"


def test_invalid_binary_label_fails():
    with pytest.raises(ValueError):
        classify_binary_label(2)
