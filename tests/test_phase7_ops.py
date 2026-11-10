from pathlib import Path

import pandas as pd
import pytest

import src.phase7_ops as ops


def test_build_manual_record_preserves_feature_order():
    features = ["tcp.seq", "http.request.method"]
    frame = ops.build_manual_record({"http.request.method": "GET", "tcp.seq": 12}, features)
    assert list(frame.columns) == features
    assert frame.iloc[0]["tcp.seq"] == 12
    assert frame.iloc[0]["http.request.method"] == "GET"


def test_build_manual_record_rejects_missing_field():
    with pytest.raises(ValueError, match="missing fields"):
        ops.build_manual_record({"tcp.seq": 1}, ["tcp.seq", "tcp.ack"])


def test_diagnose_input_counts_unseen_and_out_of_range():
    frame = pd.DataFrame({"tcp.seq": [1, 100], "http.request.method": ["GET", "ALIEN"]})
    profile = {
        "numeric": {"tcp.seq": {"q01": 0, "q99": 10}},
        "categorical": {"http.request.method": {"known_values": ["GET", "POST"]}},
    }
    result = ops.diagnose_input(frame, ["tcp.seq", "http.request.method"], profile)
    assert result["numeric_values_outside_training_q01_q99"] == 1
    assert result["unseen_categorical_values"] == 1
    assert result["missing_required_columns"] == []


def test_build_alert_feed_includes_only_predicted_attacks():
    predictions = pd.DataFrame({
        "Predicted_Binary": ["Normal", "Attack", "Attack"],
        "Predicted_Attack_Type": ["Normal", "DDoS_HTTP", "XSS"],
    })
    alerts = ops.build_alert_feed(predictions, "run_test")
    assert len(alerts) == 2
    assert alerts["Alert_ID"].tolist() == ["run_test-A00001", "run_test-A00002"]
    assert set(alerts["Review_Status"]) == {"Model-predicted attack — review required"}


def test_summarize_predictions_counts_attacks_and_types():
    predictions = pd.DataFrame({
        "Predicted_Binary": ["Normal", "Attack", "Attack", "Attack"],
        "Predicted_Attack_Type": ["Normal", "XSS", "XSS", "Backdoor"],
    })
    summary = ops.summarize_predictions(predictions)
    assert summary["records_processed"] == 4
    assert summary["predicted_attacks"] == 3
    assert summary["predicted_normal"] == 1
    assert summary["predicted_attack_types"] == {"XSS": 2, "Backdoor": 1}


def test_stratified_sample_is_reproducible_and_balanced():
    frame = pd.DataFrame({
        "Attack_type": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        "row": list(range(30)),
    })
    sample = ops.sample_validation_records(frame, 9, random_state=7, strategy="stratified")
    assert len(sample) == 9
    assert sample["Attack_type"].value_counts().to_dict() == {"A": 3, "B": 3, "C": 3}
    again = ops.sample_validation_records(frame, 9, random_state=7, strategy="stratified")
    assert sample["row"].tolist() == again["row"].tolist()


def test_run_offline_replay_batches_and_calls_progress(monkeypatch):
    calls = []

    def fake_predict(frame, bundle):
        calls.append(len(frame))
        result = frame.copy().reset_index(drop=True)
        result["Predicted_Binary"] = ["Attack"] * len(frame)
        result["Predicted_Attack_Type"] = ["XSS"] * len(frame)
        result["Final_Detection"] = ["XSS"] * len(frame)
        return result, {
            "transform_seconds": 0.01,
            "binary_predict_seconds": 0.02,
            "multiclass_predict_seconds": 0.01,
        }

    monkeypatch.setattr(ops, "predict_dataframe", fake_predict)
    progress = []
    frame = pd.DataFrame({"x": range(7)})
    result, performance = ops.run_offline_replay(
        frame, bundle=object(), batch_size=3,
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    assert calls == [3, 3, 1]
    assert len(result) == 7
    assert result["Replay_Sequence"].tolist() == list(range(1, 8))
    assert progress[-1] == (7, 7)
    assert performance["batches"] == 3
    assert performance["records_per_second_model_processing"] > 0


def test_save_run_artifacts_writes_predictions_alerts_and_summary(tmp_path: Path):
    predictions = pd.DataFrame({
        "Predicted_Binary": ["Normal", "Attack"],
        "Predicted_Attack_Type": ["Normal", "DDoS_HTTP"],
        "Final_Detection": ["Normal", "DDoS_HTTP"],
    })
    run_dir = ops.save_run_artifacts(
        predictions,
        {"total_seconds": 0.1, "records_per_second": 20},
        tmp_path,
        "run_demo",
        "manual test",
    )
    assert (run_dir / "predictions.csv").is_file()
    assert (run_dir / "threat_alerts.csv").is_file()
    assert (run_dir / "run_summary.json").is_file()
