
from pathlib import Path

import pandas as pd


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "ML-EdgeIIoT-dataset.csv"

OUTPUT_DIR = ROOT / "outputs" / "metrics"

SPLIT_REPORT = OUTPUT_DIR / "split_class_distribution.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD ORIGINAL DATASET
# ==========================================

print("=" * 65)
print("DUPLICATE IMPACT BY ATTACK CATEGORY")
print("=" * 65)

df = pd.read_csv(DATA_PATH, low_memory=False)

targets = ["Attack_label", "Attack_type"]

features = [
    column for column in df.columns
    if column not in targets
]

print("Original records:", len(df))


# ==========================================
# IDENTIFY EXACT DUPLICATES
# ==========================================

duplicate_mask = df.duplicated(
    subset=features,
    keep="first"
)

removed = df.loc[duplicate_mask].copy()

remaining = df.loc[~duplicate_mask].copy()

print("Duplicate records removed:", len(removed))
print("Remaining records:", len(remaining))


# ==========================================
# CLASS DISTRIBUTION COMPARISON
# ==========================================

original_counts = df["Attack_type"].value_counts()

removed_counts = removed["Attack_type"].value_counts()

remaining_counts = remaining["Attack_type"].value_counts()

comparison = pd.DataFrame({
    "original_count": original_counts,
    "removed_count": removed_counts,
    "remaining_count": remaining_counts
}).fillna(0).astype(int)

comparison["original_percentage"] = (
    comparison["original_count"] / len(df) * 100
)

comparison["remaining_percentage"] = (
    comparison["remaining_count"] / len(remaining) * 100
)

comparison["percentage_of_class_removed"] = (
    comparison["removed_count"] /
    comparison["original_count"] * 100
)

comparison = comparison.sort_values(
    "removed_count",
    ascending=False
)

print("\n" + "=" * 65)
print("CLASS-WISE DUPLICATE IMPACT")
print("=" * 65)

print(comparison.round(3).to_string())


# ==========================================
# DUPLICATE GROUP DETAILS
# ==========================================

print("\n" + "=" * 65)
print("REPEATED GROUP DETAILS")
print("=" * 65)

all_duplicate_mask = df.duplicated(
    subset=features,
    keep=False
)

repeated_rows = df.loc[all_duplicate_mask].copy()

repeated_rows["_original_row"] = (
    repeated_rows.index
)

hashes = pd.util.hash_pandas_object(
    repeated_rows[features],
    index=False
)

repeated_rows["_group_id"] = hashes.astype(str)

group_details = (
    repeated_rows
    .groupby("_group_id", sort=False)
    .agg(
        total_records=("Attack_type", "size"),
        distinct_labels=("Attack_label", "nunique"),
        distinct_attack_types=("Attack_type", "nunique"),
        attack_types=(
            "Attack_type",
            lambda x: ", ".join(sorted(x.unique()))
        )
    )
    .sort_values("total_records", ascending=False)
)

print(group_details.to_string())


# ==========================================
# COMPARE SAVED SPLITS
# ==========================================

print("\n" + "=" * 65)
print("CURRENT SAVED SPLIT COUNTS")
print("=" * 65)

if SPLIT_REPORT.exists():

    split_df = pd.read_csv(SPLIT_REPORT)

    pivot = split_df.pivot(
        index="attack_type",
        columns="split",
        values="count"
    )

    print(pivot.to_string())

    print("\nCurrent MITM records:")
    print(pivot.loc["MITM"].to_string())

else:
    print("Previous split report not found.")


# ==========================================
# SAVE REPORTS
# ==========================================

comparison.to_csv(
    OUTPUT_DIR / "duplicate_class_impact.csv",
    index_label="attack_type"
)

group_details.to_csv(
    OUTPUT_DIR / "duplicate_group_details.csv"
)

print("\n" + "=" * 65)
print("DUPLICATE IMPACT AUDIT COMPLETED")
print("=" * 65)

print("\nGenerated files:")
print(OUTPUT_DIR / "duplicate_class_impact.csv")
print(OUTPUT_DIR / "duplicate_group_details.csv")

print("\nOriginal dataset was not modified.")
