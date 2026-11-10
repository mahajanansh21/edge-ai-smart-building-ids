import pytest
from scipy import sparse

from src.phase5_utils import build_keep_indices, validate_matrix_and_features


def test_build_keep_indices_removes_requested_columns_in_order():
    names = ["numeric__tcp.seq", "numeric__tcp.len", "numeric__tcp.ack"]
    assert build_keep_indices(names, ["numeric__tcp.len"]) == [0, 2]


def test_build_keep_indices_missing_feature_fails():
    with pytest.raises(ValueError, match="not found"):
        build_keep_indices(["a", "b"], ["missing"])


def test_build_keep_indices_all_removed_fails():
    with pytest.raises(ValueError, match="every feature"):
        build_keep_indices(["a"], ["a"])


def test_validate_matrix_and_features_accepts_aligned_csr():
    matrix = sparse.csr_matrix([[1, 0], [0, 1]])
    validate_matrix_and_features(matrix, ["a", "b"], "test_matrix")


def test_validate_matrix_and_features_rejects_mismatch():
    matrix = sparse.csr_matrix([[1, 0]])
    with pytest.raises(ValueError, match="columns"):
        validate_matrix_and_features(matrix, ["a"], "test_matrix")
