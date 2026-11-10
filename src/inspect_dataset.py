
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ML-EdgeIIoT-dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CHECK DATASET
# ==========================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("=" * 60)
print("EDGE-IIOTSET DATASET INSPECTION")
print("=" * 60)

print("Dataset:", DATA_PATH.name)
print(
    "File size:",
    round(DATA_PATH.stat().st_size / (1024**2), 2),
    "MiB"
)


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

try:
    df = pd.read_csv(DATA_PATH, low_memory=False)

except UnicodeDecodeError:
    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
        encoding="latin-1"
    )

print("Dataset loaded successfully!")


# ==========================================
# 4. BASIC INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("BASIC INFORMATION")
print("=" * 60)

rows, columns = df.shape

print(f"Total records: {rows:,}")
print(f"Total columns: {columns:,}")

print(
    "Memory usage:",
    round(df.memory_usage(deep=True).sum() / (1024**2), 2),
    "MiB"
)

print("\nFirst five rows:")
print(df.head().to_string(index=False))


# ==========================================
# 5. COLUMN INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("COLUMN NAMES AND DATA TYPES")
print("=" * 60)

for column in df.columns:
    print(f"{column:45} {df[column].dtype}")


numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical columns:", len(numeric_columns))
print("Categorical columns:", len(categorical_columns))


# ==========================================
# 6. MISSING VALUES
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(
    ascending=False
)

if missing.empty:
    print("No missing values found.")
else:
    print(missing.to_string())


# ==========================================
# 7. INFINITE VALUES
# ==========================================

print("\n" + "=" * 60)
print("INFINITE VALUE ANALYSIS")
print("=" * 60)

infinite_counts = {}

for column in numeric_columns:
    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).to_numpy(dtype="float64")

    count = int(np.isinf(values).sum())

    if count > 0:
        infinite_counts[column] = count

if infinite_counts:
    for column, count in infinite_counts.items():
        print(f"{column}: {count}")
else:
    print("No infinite values found.")


# ==========================================
# 8. DUPLICATE RECORDS
# ==========================================

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

duplicate_count = int(df.duplicated().sum())

print(f"Duplicate records: {duplicate_count:,}")


# ==========================================
# 9. IDENTIFY TARGET COLUMNS
# ==========================================

print("\n" + "=" * 60)
print("TARGET COLUMN ANALYSIS")
print("=" * 60)

def normalize_name(name):
    return "".join(
        char.lower()
        for char in str(name)
        if char.isalnum()
    )


target_candidates = []

for column in df.columns:

    normalized = normalize_name(column)

    if any(
        keyword in normalized
        for keyword in [
            "attacklabel",
            "attacktype",
            "label",
            "target",
            "class"
        ]
    ):
        target_candidates.append(column)


target_distributions = {}

if not target_candidates:
    print("No obvious target columns identified.")

for column in target_candidates:

    print(f"\nDistribution: {column}")

    counts = (
        df[column]
        .astype("string")
        .fillna("<MISSING>")
        .value_counts(dropna=False)
    )

    print(counts.head(50).to_string())

    target_file = (
        OUTPUT_DIR
        / f"distribution_{normalize_name(column)}.csv"
    )

    counts.rename_axis("class").reset_index(
        name="count"
    ).to_csv(target_file, index=False)

    target_distributions[column] = {
        str(label): int(count)
        for label, count in counts.items()
    }


# ==========================================
# 10. SAVE COLUMN PROFILE
# ==========================================

profile = pd.DataFrame({
    "column": df.columns,
    "data_type": df.dtypes.astype(str).values,
    "missing_count": df.isnull().sum().values,
    "missing_percentage": (
        df.isnull().mean().values * 100
    ),
    "unique_values": [
        int(df[column].nunique(dropna=False))
        for column in df.columns
    ]
})

profile.to_csv(
    OUTPUT_DIR / "column_profile.csv",
    index=False
)


# ==========================================
# 11. SAVE DATASET SUMMARY
# ==========================================

report = {
    "dataset": DATA_PATH.name,
    "file_size_mib": round(
        DATA_PATH.stat().st_size / (1024**2), 2
    ),
    "total_records": int(rows),
    "total_columns": int(columns),
    "memory_usage_mib": round(
        df.memory_usage(deep=True).sum() / (1024**2), 2
    ),
    "numerical_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "missing_values": {
        str(k): int(v)
        for k, v in missing.items()
    },
    "infinite_values": infinite_counts,
    "duplicate_records": duplicate_count,
    "target_candidates": target_candidates,
    "target_distributions": target_distributions,
    "column_names": df.columns.tolist()
}

with open(
    OUTPUT_DIR / "dataset_profile.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(report, file, indent=4)


# ==========================================
# 12. COMPLETION
# ==========================================

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)

print("\nGenerated files:")

print(OUTPUT_DIR / "dataset_profile.json")
print(OUTPUT_DIR / "column_profile.csv")

print("\nNo original dataset records were modified.")
