
from pathlib import Path

import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "raw"
    / "ML-EdgeIIoT-dataset.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. NINE INITIAL FEATURES
# ==========================================

FEATURES = [
    "arp.opcode",
    "arp.hw.size",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len"
]


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("=" * 65)
print("FEATURE COMBINATION FEASIBILITY AUDIT")
print("=" * 65)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset shape:", df.shape)


# ==========================================
# 4. CREATE FEATURE GROUPS
# ==========================================

X = df[FEATURES].copy()

# Group records with exactly identical values
# across the nine selected features.

group_ids = X.groupby(
    FEATURES,
    dropna=False,
    sort=False
).ngroup()

work = pd.DataFrame({
    "group_id": group_ids,
    "Attack_label": df["Attack_label"],
    "Attack_type": df["Attack_type"]
})


# ==========================================
# 5. GROUP STATISTICS
# ==========================================

group_stats = (
    work.groupby("group_id")
    .agg(
        records=("Attack_label", "size"),
        binary_classes=("Attack_label", "nunique"),
        attack_classes=("Attack_type", "nunique")
    )
)

print("\nTotal unique feature combinations:")
print(len(group_stats))

print("\nGroups containing both Normal and Attack:")
print(
    (group_stats["binary_classes"] > 1).sum()
)

print("\nGroups containing multiple attack categories:")
print(
    (group_stats["attack_classes"] > 1).sum()
)

# Number of records affected by ambiguity.

binary_mixed_ids = group_stats.index[
    group_stats["binary_classes"] > 1
]

multi_mixed_ids = group_stats.index[
    group_stats["attack_classes"] > 1
]

print("\nRecords in binary-ambiguous groups:")
print(work["group_id"].isin(binary_mixed_ids).sum())

print("\nRecords in multiclass-ambiguous groups:")
print(work["group_id"].isin(multi_mixed_ids).sum())


# ==========================================
# 6. UNIQUE COMBINATIONS PER BINARY CLASS
# ==========================================

print("\n" + "=" * 65)
print("BINARY CLASS GROUP COUNTS")
print("=" * 65)

binary_group_counts = (
    work.drop_duplicates(
        ["group_id", "Attack_label"]
    )
    .groupby("Attack_label")["group_id"]
    .nunique()
)

print(binary_group_counts.to_string())


# ==========================================
# 7. UNIQUE COMBINATIONS PER ATTACK TYPE
# ==========================================

print("\n" + "=" * 65)
print("MULTICLASS GROUP COUNTS")
print("=" * 65)

multiclass_group_counts = (
    work.drop_duplicates(
        ["group_id", "Attack_type"]
    )
    .groupby("Attack_type")["group_id"]
    .nunique()
    .sort_values()
)

print(multiclass_group_counts.to_string())


# ==========================================
# 8. COMMON PATTERN ANALYSIS
# ==========================================

print("\n" + "=" * 65)
print("TOP 20 MOST COMMON FEATURE PATTERNS")
print("=" * 65)

group_stats = group_stats.sort_values(
    "records",
    ascending=False
)

top_groups = group_stats.head(20).index

pattern_counts = (
    work[
        work["group_id"].isin(top_groups)
    ]
    .groupby(["group_id", "Attack_type"])
    .size()
    .unstack(fill_value=0)
)

pattern_counts.to_csv(
    OUTPUT_DIR / "top_feature_pattern_labels.csv"
)

print(
    pattern_counts.to_string()
)


# ==========================================
# 9. SAVE SUMMARY REPORT
# ==========================================

group_stats.to_csv(
    OUTPUT_DIR / "feature_combination_group_stats.csv"
)

binary_group_counts.to_csv(
    OUTPUT_DIR / "binary_feature_group_counts.csv",
    header=["unique_feature_groups"]
)

multiclass_group_counts.to_csv(
    OUTPUT_DIR / "multiclass_feature_group_counts.csv",
    header=["unique_feature_groups"]
)


# ==========================================
# 10. COMPLETION
# ==========================================

print("\n" + "=" * 65)
print("FEASIBILITY AUDIT COMPLETED")
print("=" * 65)

print("\nGenerated files:")

print(OUTPUT_DIR / "top_feature_pattern_labels.csv")
print(OUTPUT_DIR / "feature_combination_group_stats.csv")
print(OUTPUT_DIR / "binary_feature_group_counts.csv")
print(OUTPUT_DIR / "multiclass_feature_group_counts.csv")

print("\nOriginal dataset was not modified.")
