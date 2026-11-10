import numpy as np
import pytest
from scipy import sparse

from src.phase4_utils import calculate_metrics, make_estimator


def test_binary_metrics_respect_label_mapping():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = calculate_metrics(y_true, y_pred, "Attack_label")
    assert metrics["normal_recall"] == pytest.approx(0.5)
    assert metrics["attack_recall"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_multiclass_metrics_are_finite():
    y_true = np.array(["Normal", "DDoS", "MITM", "Normal"])
    y_pred = np.array(["Normal", "DDoS", "Normal", "Normal"])
    metrics = calculate_metrics(y_true, y_pred, "Attack_type")
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


def test_model_factory_returns_expected_classifier_families():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

    expected = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
    }
    for name, cls in expected.items():
        assert isinstance(make_estimator(name), cls)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        make_estimator("not_a_model")
