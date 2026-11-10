
from pathlib import Path
import json

import pandas as pd

from sklearn.model_selection import train_test_split


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "raw"
    / "ML-EdgeIIoT-dataset.csv"
)

SPLIT_DIR = (
    ROOT / "data" / "processed"
    / "splits"
)

METRICS_DIR = (
    ROOT / "outputs" / "metrics"
)

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# 2. CONFIGURATION
# ==================================================

TARGETS = [
    "Attack_label",
    "Attack_type"
]

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ==================================================
# 3. LOAD DATASET
# ==================================================

print("=" * 70)
print("TRAIN / VALIDATION / TEST PREPARATION")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("\nOriginal dataset shape:", df.shape)


# ==================================================
# 4. VALIDATE TARGETS
# ==================================================

missing_targets = [
    col for col in TARGETS
    if col not in df.columns
]

if missing_targets:
    raise ValueError(
        f"Missing target columns: {missing_targets}"
    )

if df[TARGETS].isnull().any().any():
    raise ValueError(
        "Missing target values found."
    )

# Verify the known binary label mapping.

expected_mapping = (
    df.groupby("Attack_type")["Attack_label"]
    .nunique()
)

if (expected_mapping != 1).any():
    raise ValueError(
        "An attack type maps to multiple binary labels."
    )


# ==================================================
# 5. REMOVE EXACT DUPLICATES
# ==================================================

print("\n" + "=" * 70)
print("EXACT DUPLICATE REMOVAL")
print("=" * 70)

# All original input columns are used.
# Target columns are excluded from the duplicate key.

FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in TARGETS
]

# Locate every row that belongs to a repeated group.

duplicate_member_mask = df.duplicated(
    subset=FEATURE_COLUMNS,
    keep=False
)

duplicate_members = df.loc[
    duplicate_member_mask
]

# Check whether identical feature records
# have conflicting target labels.

if not duplicate_members.empty:

    duplicate_label_groups = (
        duplicate_members
        .groupby(
            FEATURE_COLUMNS,
            dropna=False,
            sort=False
        )[TARGETS]
        .nunique()
    )

    conflicting_groups = (
        duplicate_label_groups > 1
    ).any(axis=1)

    if conflicting_groups.any():

        raise ValueError(
            "Conflicting labels found among "
            "exact duplicate feature records. "
            "Manual investigation is required."
        )

# Remove excess exact duplicates.

before = len(df)

df_clean = df.drop_duplicates(
    subset=FEATURE_COLUMNS,
    keep="first"
).copy()

removed = before - len(df_clean)

print("Original records:", before)
print("Duplicates removed:", removed)
print("Remaining records:", len(df_clean))


# ==================================================
# 6. KEEP ORIGINAL ROW IDENTIFIERS FOR AUDIT
# ==================================================

df_clean = df_clean.reset_index()

df_clean = df_clean.rename(
    columns={"index": "_source_row_id"}
)


# ==================================================
# 7. FIRST SPLIT: 70% TRAIN / 30% TEMP
# ==================================================

print("\nCreating training and temporary datasets...")

train_df, temporary_df = train_test_split(
    df_clean,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df_clean["Attack_type"]
)


# ==================================================
# 8. SECOND SPLIT: 15% VALIDATION / 15% TEST
# ==================================================

print("Separating validation and test datasets...")

validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temporary_df["Attack_type"]
)


# ==================================================
# 9. VERIFY SPLIT SIZES
# ==================================================

splits = {
    "train": train_df,
    "validation": validation_df,
    "test": test_df
}

print("\n" + "=" * 70)
print("SPLIT SIZES")
print("=" * 70)

for name, part in splits.items():

    print(
        f"{name}: {len(part):,} "
        f"({len(part) / len(df_clean) * 100:.2f}%)"
    )

if sum(len(part) for part in splits.values()) != len(df_clean):
    raise RuntimeError(
        "Split row count mismatch."
    )


# ==================================================
# 10. CHECK PARTITION OVERLAP
# ==================================================

train_ids = set(train_df["_source_row_id"])
validation_ids = set(validation_df["_source_row_id"])
test_ids = set(test_df["_source_row_id"])

if train_ids & validation_ids:
    raise RuntimeError("Train/validation overlap found.")

if train_ids & test_ids:
    raise RuntimeError("Train/test overlap found.")

if validation_ids & test_ids:
    raise RuntimeError("Validation/test overlap found.")

print("\nPartition overlap check: PASSED")


# ==================================================
# 11. VERIFY ALL CLASSES ARE PRESENT
# ==================================================

expected_classes = set(
    df_clean["Attack_type"].unique()
)

for name, part in splits.items():

    actual_classes = set(
        part["Attack_type"].unique()
    )

    missing = expected_classes - actual_classes

    if missing:
        raise RuntimeError(
            f"Missing classes in {name}: {missing}"
        )

print("All 15 classes represented in all splits.")


# ==================================================
# 12. SAVE SPLIT DATA
# ==================================================

print("\nSaving split datasets...")

for name, part in splits.items():

    output_path = SPLIT_DIR / f"{name}.csv"

    # Remove helper column before saving.
    saved_data = part.drop(
        columns=["_source_row_id"]
    )

    saved_data.to_csv(
        output_path,
        index=False
    )

    print(f"{name}: {output_path}")


# ==================================================
# 13. CREATE SPLIT MANIFEST
# ==================================================

manifest_parts = []

for name, part in splits.items():

    manifest = part[
        [
            "_source_row_id",
            "Attack_label",
            "Attack_type"
        ]
    ].copy()

    manifest["split"] = name

    manifest_parts.append(manifest)

manifest_df = pd.concat(
    manifest_parts,
    ignore_index=True
)

manifest_path = (
    METRICS_DIR / "split_manifest.csv"
)

manifest_df.to_csv(
    manifest_path,
    index=False
)


# ==================================================
# 14. CLASS DISTRIBUTION REPORT
# ==================================================

distribution_records = []

for name, part in splits.items():

    counts = part["Attack_type"].value_counts()

    for attack_type, count in counts.items():

        distribution_records.append({
            "split": name,
            "attack_type": attack_type,
            "count": int(count),
            "percentage": round(
                count / len(part) * 100,
                3
            )
        })

distribution_df = pd.DataFrame(
    distribution_records
)

distribution_path = (
    METRICS_DIR / "split_class_distribution.csv"
)

distribution_df.to_csv(
    distribution_path,
    index=False
)


# ==================================================
# 15. SAVE REPRODUCIBILITY REPORT
# ==================================================

report = {
    "dataset": DATA_PATH.name,
    "original_records": before,
    "duplicates_removed": removed,
    "remaining_records": len(df_clean),
    "random_state": RANDOM_STATE,
    "stratification_column": "Attack_type",
    "train_records": len(train_df),
    "validation_records": len(validation_df),
    "test_records": len(test_df),
    "classes": sorted(
        str(x) for x in expected_classes
    ),
    "split_method": "Two-stage stratified random split",
    "split_proportions": {
        "train": TRAIN_SIZE,
        "validation": VALIDATION_SIZE,
        "test": TEST_SIZE
    },
    "limitation": (
        "Record-level split; independent device-level "
        "or capture-level generalization has not been established."
    )
}

report_path = (
    METRICS_DIR / "split_summary.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ==================================================
# 16. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 70)
print("DATA SPLITTING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated split files:")
print(SPLIT_DIR / "train.csv")
print(SPLIT_DIR / "validation.csv")
print(SPLIT_DIR / "test.csv")

print("\nGenerated reports:")
print(manifest_path)
print(distribution_path)
print(report_path)

print("\nOriginal dataset was not modified.")

print(
    "\nIMPORTANT: Preprocessing transformations "
    "must be fitted using training data only."
)
