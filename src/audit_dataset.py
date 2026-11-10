
from pathlib import Path

import pandas as pd


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "ML-EdgeIIoT-dataset.csv"

OUTPUT_DIR = ROOT / "outputs" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Loaded {len(df):,} records.")


# ==========================================
# 1. VERIFY LABEL MAPPING
# ==========================================

print("\n" + "=" * 65)
print("1. BINARY LABEL VERIFICATION")
print("=" * 65)

cross_tab = pd.crosstab(
    df["Attack_label"],
    df["Attack_type"],
    margins=True
)

print(cross_tab)

cross_tab.to_csv(
    OUTPUT_DIR / "label_cross_tabulation.csv"
)


# ==========================================
# 2. CLASS DISTRIBUTION AND PERCENTAGES
# ==========================================

print("\n" + "=" * 65)
print("2. CLASS DISTRIBUTION")
print("=" * 65)

counts = df["Attack_type"].value_counts()

distribution = pd.DataFrame({
    "count": counts,
    "percentage": counts / len(df) * 100
})

distribution["percentage"] = distribution[
    "percentage"
].round(3)

print(distribution)

distribution.to_csv(
    OUTPUT_DIR / "class_distribution.csv"
)


# ==========================================
# 3. DUPLICATE FEATURE AUDIT
# ==========================================

print("\n" + "=" * 65)
print("3. DUPLICATE FEATURE AUDIT")
print("=" * 65)

target_columns = ["Attack_label", "Attack_type"]

features = df.drop(columns=target_columns)

feature_hash = pd.util.hash_pandas_object(
    features,
    index=False
)

duplicate_mask = feature_hash.duplicated(keep=False)

duplicate_rows = df.loc[duplicate_mask].copy()

duplicate_rows["_feature_hash"] = (
    feature_hash.loc[duplicate_mask].to_numpy()
)

print(
    "Rows with repeated feature patterns:",
    f"{len(duplicate_rows):,}"
)

if not duplicate_rows.empty:

    grouped_labels = (
        duplicate_rows
        .groupby("_feature_hash", sort=False)
        .agg(
            distinct_binary_labels=("Attack_label", "nunique"),
            distinct_attack_types=("Attack_type", "nunique")
        )
    )

    conflicting = grouped_labels[
        (grouped_labels["distinct_binary_labels"] > 1)
        |
        (grouped_labels["distinct_attack_types"] > 1)
    ]

    print(
        "Repeated feature patterns with conflicting labels:",
        len(conflicting)
    )

    conflicting.to_csv(
        OUTPUT_DIR / "conflicting_duplicate_labels.csv"
    )

else:
    print("No repeated feature patterns found.")


# ==========================================
# 4. CONSTANT FEATURES
# ==========================================

print("\n" + "=" * 65)
print("4. CONSTANT FEATURES")
print("=" * 65)

unique_counts = features.nunique(dropna=False)

constant_columns = unique_counts[
    unique_counts <= 1
]

if constant_columns.empty:
    print("No constant features found.")
else:
    print(constant_columns.to_string())


# ==========================================
# 5. HIGH-CARDINALITY FEATURES
# ==========================================

print("\n" + "=" * 65)
print("5. HIGH-CARDINALITY FEATURES")
print("=" * 65)

cardinality = pd.DataFrame({
    "unique_values": unique_counts,
    "unique_percentage": (
        unique_counts / len(df) * 100
    )
})

high_cardinality = cardinality[
    cardinality["unique_percentage"] > 10
].sort_values(
    "unique_percentage",
    ascending=False
)

print(high_cardinality.to_string())


# ==========================================
# 6. SAVE AUDIT RESULTS
# ==========================================

cardinality.to_csv(
    OUTPUT_DIR / "feature_cardinality.csv"
)

print("\n" + "=" * 65)
print("DATASET AUDIT COMPLETED")
print("=" * 65)

print("\nSaved results inside:")
print(OUTPUT_DIR)
