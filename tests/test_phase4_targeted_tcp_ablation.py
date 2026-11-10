from src.phase4_targeted_tcp_ablation import ABLATIONS, TCP_FEATURES, build_keep_indices


def test_single_feature_ablation_removes_only_requested_feature():
    names = ["numeric__tcp.seq", "numeric__tcp.ack", "numeric__tcp.len", "numeric__mqtt.len"]
    keep = build_keep_indices(names, ["numeric__tcp.seq"])
    assert keep == [1, 2, 3]


def test_combined_top_six_ablation_removes_six_features():
    names = TCP_FEATURES + ["numeric__mqtt.len", "numeric__http.response"]
    keep = build_keep_indices(names, TCP_FEATURES)
    assert keep == [6, 7]


def test_unknown_feature_fails_clearly():
    try:
        build_keep_indices(["numeric__tcp.seq"], ["numeric__tcp.missing"])
    except ValueError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a missing feature")


def test_expected_ablation_configurations():
    assert len(ABLATIONS) == 8
    assert len(ABLATIONS["without_top6_tcp_features"]) == 6
