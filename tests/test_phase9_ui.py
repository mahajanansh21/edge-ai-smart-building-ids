from src.phase9_ui import PIPELINE_STEPS


def test_pipeline_has_four_stages() -> None:
    assert len(PIPELINE_STEPS) == 4


def test_pipeline_starts_with_36_source_features() -> None:
    assert "36 source features" in PIPELINE_STEPS[0][2]


def test_pipeline_documents_36_to_43_preprocessing() -> None:
    detail = PIPELINE_STEPS[1][2]
    assert "36 source features" in detail
    assert "43 transformed features" in detail


def test_pipeline_documents_both_model_feature_exclusions() -> None:
    detail = PIPELINE_STEPS[2][2]
    assert "numeric__tcp.len" in detail
    assert "numeric__tcp.checksum" in detail
    assert "Random Forest" in detail
    assert "Decision Tree" in detail


def test_pipeline_explains_binary_gate_for_final_detection() -> None:
    detail = PIPELINE_STEPS[3][2]
    assert "Binary Normal" in detail
    assert "Binary Attack" in detail
