"""Phase 5B: freeze the retrained candidates and their preprocessing contract.

No dataset matrices or labels are opened here. This creates a frozen model
bundle and an explicit manifest before the one-time test evaluation.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "models" / "phase5" / "candidates"
FINAL_DIR = ROOT / "models" / "final"
SUMMARY_PATH = ROOT / "outputs" / "metrics" / "phase5" / "phase5_candidate_summary.json"
PREPROCESSOR = ROOT / "models" / "preprocessors" / "expanded36_normalized_preprocessor.joblib"


def main() -> int:
    if not SUMMARY_PATH.is_file():
        print(f"ERROR: Candidate summary not found: {SUMMARY_PATH}", file=sys.stderr)
        return 2
    if not PREPROCESSOR.is_file():
        print(f"ERROR: Phase 3 preprocessor not found: {PREPROCESSOR}", file=sys.stderr)
        return 2

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    results = summary.get("candidate_results", {})
    if summary.get("test_split_used") is not False:
        print("ERROR: Candidate summary does not confirm that test data was untouched.", file=sys.stderr)
        return 2
    if summary.get("failed_candidates") != 0 or set(results) != {"Attack_label", "Attack_type"}:
        print("ERROR: Both candidates must be successfully prepared before freezing.", file=sys.stderr)
        return 2

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for target, result in results.items():
        source_model = ROOT / result["model_path"]
        source_spec = ROOT / result["input_spec_path"]
        if not source_model.is_file() or not source_spec.is_file():
            print(f"ERROR: Missing candidate model or input specification for {target}.", file=sys.stderr)
            return 2
        destination_model = FINAL_DIR / source_model.name
        destination_spec = FINAL_DIR / source_spec.name
        shutil.copy2(source_model, destination_model)
        shutil.copy2(source_spec, destination_spec)
        frozen[target] = {
            "model_path": str(destination_model.relative_to(ROOT)),
            "input_spec_path": str(destination_spec.relative_to(ROOT)),
            "candidate_name": result["candidate_name"],
            "model_name": result["model_name"],
            "validation_metrics": result["metrics"],
        }

    final_preprocessor = FINAL_DIR / PREPROCESSOR.name
    shutil.copy2(PREPROCESSOR, final_preprocessor)
    manifest = {
        "phase": 5,
        "status": "frozen_before_test_evaluation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_split_used": False,
        "test_evaluation_authorized": True,
        "preprocessor_path": str(final_preprocessor.relative_to(ROOT)),
        "preprocessor_feature_set": "expanded36_normalized",
        "transformed_feature_count_before_target_specific_selection": 43,
        "models": frozen,
        "no_further_model_or_feature_tuning_after_test_evaluation": True,
        "limitations": summary.get("limitations", []),
        "evaluation_instruction": "Run the one-time test evaluation only after reviewing this manifest. Do not tune or select another model based on the test results.",
    }
    manifest_path = FINAL_DIR / "phase5_model_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("PHASE 5B: MODEL BUNDLE FROZEN")
    print(f"Frozen manifest: {manifest_path}")
    print("Models: binary Random Forest without numeric__tcp.len; multiclass Decision Tree without numeric__tcp.checksum")
    print("No dataset matrices or labels were opened. Test set remains untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
