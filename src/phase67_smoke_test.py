"""Load final artifacts and execute one training-profile-based manual vector."""
from __future__ import annotations

from src.edge_inference import TRAIN_SPLIT, load_model_bundle, predict_dataframe
from src.phase7_ops import build_manual_record, load_training_profile


def main() -> int:
    print("PHASE 6/7 MODEL INFERENCE SMOKE TEST")
    print("Loading frozen Phase 5 artifacts...")
    bundle = load_model_bundle()
    profile = load_training_profile(TRAIN_SPLIT, bundle.source_features)

    values: dict[str, object] = {}
    for feature in bundle.source_features:
        if feature in profile["numeric"]:
            values[feature] = profile["numeric"][feature]["default"]
        else:
            values[feature] = profile["categorical"][feature]["default"]

    one_record = build_manual_record(values, bundle.source_features)
    predictions, performance = predict_dataframe(one_record, bundle)
    row = predictions.iloc[0]

    print("\nFrozen models loaded successfully.")
    print(f"Source features: {len(bundle.source_features)}")
    print(f"Transformed features: {len(bundle.transformed_features)}")
    print("\nOne synthetic median/mode feature vector was processed:")
    print(f"Binary prediction: {row['Predicted_Binary']}")
    print(f"Multiclass prediction: {row['Predicted_Attack_Type']}")
    print(f"Final detection: {row['Final_Detection']}")
    if "Binary_Attack_Score_Uncalibrated" in predictions:
        print(f"Binary attack score (uncalibrated): {row['Binary_Attack_Score_Uncalibrated']:.4f}")
    if "Multiclass_Top_Score_Uncalibrated" in predictions:
        print(f"Multiclass top score (uncalibrated): {row['Multiclass_Top_Score_Uncalibrated']:.4f}")
    print(f"Inference processing time: {performance['total_seconds'] * 1000:.3f} ms")
    print("\nNOTE: This is a schema/model smoke test, not a validated network flow or a performance benchmark.")
    print("No model training and no test-set evaluation were performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
