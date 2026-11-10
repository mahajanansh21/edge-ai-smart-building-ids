
from pathlib import Path
import json

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
# 2. INITIAL BASELINE FEATURES
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

TARGETS = [
    "Attack_label",
    "Attack_type"
]


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("=" * 65)
print("DUPLICATE-AWARE SPLIT DIAGNOSTICS")
print("=" * 65)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset shape:", df.shape)


# ==========================================
# 4. VALIDATE FEATURES
# ==========================================

required = FEATURES + TARGETS

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

X = df[FEATURES].copy()

y_multiclass = df["Attack_type"].copy()

y_binary = df["Attack_label"].copy()

print("\nSelected baseline features:", len(FEATURES))

for feature in FEATURES:
    print("-", feature)


# ==========================================
# 5. CREATE FEATURE GROUP IDENTIFIERS
# ==========================================

print("\nCalculating identical feature groups...")

group_ids = (
    pd.util.hash_pandas_object(
        X,
        index=False
    )
    .astype("uint64")
    .astype(str)
)

group_data = pd.DataFrame({
    "group_id": group_ids,
    "Attack_type": y_multiclass,
    "Attack_label": y_binary
})


# ==========================================
# 6. GROUP STATISTICS
# ==========================================

group_stats = (
    group_data
    .groupby("group_id", sort=False)
    .agg(
        row_count=("Attack_type", "size"),
        attack_type_count=("Attack_type", "nunique"),
        binary_label_count=("Attack_label", "nunique")
    )
)

total_groups = len(group_stats)

repeated_groups = group_stats[
    group_stats["row_count"] > 1
]

mixed_label_groups = group_stats[
    group_stats["attack_type_count"] > 1
]

duplicate_rows = int(
    repeated_groups["row_count"].sum()
)

duplicate_excess = int(
    (repeated_groups["row_count"] - 1).sum()
)

print("\n" + "=" * 65)
print("DUPLICATE GROUP RESULTS")
print("=" * 65)

print("Total rows:", len(df))

print("Unique feature groups:", total_groups)

print("Repeated feature groups:", len(repeated_groups))

print("Rows participating in repeated groups:", duplicate_rows)

print("Excess repeated rows:", duplicate_excess)

print("Groups with different attack types:", len(mixed_label_groups))


# ==========================================
# 7. CLASS REPRESENTATION IN GROUPS
# ==========================================

print("\n" + "=" * 65)
print("GROUP COUNTS BY ATTACK TYPE")
print("=" * 65)

groups_per_class = (
    group_data
    .drop_duplicates(["group_id", "Attack_type"])
    .groupby("Attack_type")["group_id"]
    .nunique()
    .sort_values()
)

print(groups_per_class.to_string())


# ==========================================
# 8. SAVE REPEATED GROUP REPORT
# ==========================================

group_report = group_stats.loc[
    repeated_groups.index
].reset_index()

group_report.to_csv(
    OUTPUT_DIR / "selected_feature_duplicate_groups.csv",
    index=False
)

mixed_report = group_stats.loc[
    mixed_label_groups.index
].reset_index()

mixed_report.to_csv(
    OUTPUT_DIR / "selected_feature_mixed_label_groups.csv",
    index=False
)


# ==========================================
# 9. SAVE JSON SUMMARY
# ==========================================

summary = {
    "dataset_rows": len(df),
    "selected_feature_count": len(FEATURES),
    "selected_features": FEATURES,
    "unique_feature_groups": int(total_groups),
    "repeated_feature_groups": int(len(repeated_groups)),
    "rows_in_repeated_groups": duplicate_rows,
    "excess_repeated_rows": duplicate_excess,
    "mixed_attack_type_groups": int(len(mixed_label_groups)),
    "groups_per_attack_type": {
        str(k): int(v)
        for k, v in groups_per_class.items()
    }
}

with open(
    OUTPUT_DIR / "split_diagnostics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(summary, file, indent=4)


# ==========================================
# 10. FINAL RESULT
# ==========================================

print("\n" + "=" * 65)
print("SPLIT DIAGNOSTICS COMPLETED")
print("=" * 65)

print("\nSaved files:")
print(OUTPUT_DIR / "split_diagnostics.json")
print(OUTPUT_DIR / "selected_feature_duplicate_groups.csv")
print(OUTPUT_DIR / "selected_feature_mixed_label_groups.csv")

print("\nNo dataset rows were modified.")
