"""Print a readable validation-only leaderboard from Phase 4 results."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "metrics" / "phase4" / "validation_model_comparison.csv"


def main():
    if not RESULTS.exists():
        raise FileNotFoundError(
            f"Could not find {RESULTS}. Run .\\.venv\\Scripts\\python.exe -m src.train_models first."
        )
    frame = pd.read_csv(RESULTS)
    if frame.empty:
        print("No successful training runs are recorded yet.")
        return
    columns = [
        "target", "feature_set", "model", "macro_f1", "weighted_f1",
        "balanced_accuracy", "accuracy", "fit_seconds", "validation_records_per_second",
    ]
    print("PHASE 4 VALIDATION LEADERBOARD")
    print("Rank by macro-F1; use balanced accuracy as a secondary metric.")
    for (target, feature_set), group in frame.groupby(["target", "feature_set"], sort=True):
        ranked = group.sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
        print(f"\n{target} | {feature_set}")
        print(ranked[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        winner = ranked.iloc[0]
        print(
            f"Candidate leader: {winner['model']} "
            f"(macro-F1={winner['macro_f1']:.4f}; balanced accuracy={winner['balanced_accuracy']:.4f})"
        )
    print("\nThese are validation results, not final test results or production guarantees.")


if __name__ == "__main__":
    main()
