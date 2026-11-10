
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ML-EdgeIIoT-dataset.csv"
)

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD DATASET
# =====================================================

print("=" * 65)
print("EDGE-IIOTSET EXPLORATORY DATA ANALYSIS")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print(f"Loaded {len(df):,} records.")
print(f"Columns: {len(df.columns)}")


# =====================================================
# 3. VALIDATE TARGET COLUMNS
# =====================================================

required_columns = [
    "Attack_label",
    "Attack_type"
]

missing_targets = [
    col for col in required_columns
    if col not in df.columns
]

if missing_targets:
    raise ValueError(
        f"Missing target columns: {missing_targets}"
    )


# =====================================================
# 4. BINARY CLASS DISTRIBUTION
# =====================================================

print("\nGenerating binary class distribution...")

binary_labels = df["Attack_label"].map({
    0: "Normal",
    1: "Attack"
})

if binary_labels.isna().any():
    raise ValueError(
        "Unexpected or missing values in Attack_label."
    )

binary_counts = binary_labels.value_counts()

binary_percentages = (
    binary_counts / len(df) * 100
)

binary_summary = pd.DataFrame({
    "count": binary_counts,
    "percentage": binary_percentages.round(3)
})

binary_summary.to_csv(
    METRICS_DIR / "binary_class_distribution.csv",
    index_label="traffic_class"
)
plt.figure(figsize=(8, 5))

plt.bar(
    binary_counts.index,
    binary_counts.values
)

plt.title("Normal Traffic vs Attack Traffic")
plt.xlabel("Traffic Class")
plt.ylabel("Number of Records")

for i, value in enumerate(binary_counts.values):
    plt.text(
        i,
        value,
        f"{value:,}\n({binary_percentages.iloc[i]:.2f}%)",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_binary_distribution.png",
    dpi=150
)

plt.close()


# =====================================================
# 5. MULTICLASS ATTACK DISTRIBUTION
# =====================================================

print("Generating attack category distribution...")

attack_counts = df["Attack_type"].value_counts()

attack_percentages = (
    attack_counts / len(df) * 100
)

attack_summary = pd.DataFrame({
    "count": attack_counts,
    "percentage": attack_percentages.round(3)
})

attack_summary.to_csv(
    METRICS_DIR / "attack_type_distribution.csv",
    index_label="attack_type"
)
plot_data = attack_counts.sort_values(
    ascending=True
)

plt.figure(figsize=(11, 8))

bars = plt.barh(
    plot_data.index,
    plot_data.values
)

plt.title("Distribution of Normal and Malicious Traffic")
plt.xlabel("Number of Records")
plt.ylabel("Traffic Category")

for bar in bars:

    width = bar.get_width()

    plt.text(
        width + max(plot_data.values) * 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{int(width):,}",
        va="center"
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_attack_type_distribution.png",
    dpi=150
)

plt.close()


# =====================================================
# 6. FEATURE CARDINALITY
# =====================================================

print("Analyzing feature cardinality...")

target_columns = [
    "Attack_label",
    "Attack_type"
]

features = df.drop(
    columns=target_columns
)

cardinality = features.nunique(
    dropna=False
).sort_values(ascending=False)

cardinality_df = pd.DataFrame({
    "unique_values": cardinality,
    "unique_percentage": (
        cardinality / len(df) * 100
    ).round(3)
})

cardinality_df.to_csv(
    METRICS_DIR / "feature_cardinality_summary.csv",
    index_label="feature"
)
top_features = cardinality.head(15).sort_values()

plt.figure(figsize=(11, 8))

plt.barh(
    top_features.index,
    top_features.values
)

plt.title("Top 15 Features by Number of Unique Values")
plt.xlabel("Unique Values")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_feature_cardinality.png",
    dpi=150
)

plt.close()


# =====================================================
# 7. NUMERICAL FEATURE DISTRIBUTIONS
# =====================================================

print("Generating numerical feature distributions...")

numeric_df = df.select_dtypes(
    include=[np.number]
).drop(
    columns=target_columns,
    errors="ignore"
)

variable_numeric = [
    col for col in numeric_df.columns
    if numeric_df[col].nunique(dropna=False) > 1
]

preferred_features = [
    "tcp.len",
    "tcp.dstport",
    "tcp.flags",
    "tcp.connection.syn",
    "tcp.connection.fin",
    "udp.time_delta",
    "mqtt.len"
]

selected_features = [
    col for col in preferred_features
    if col in variable_numeric
][:6]

if len(selected_features) < 6:

    for column in variable_numeric:

        if column not in selected_features:
            selected_features.append(column)

        if len(selected_features) == 6:
            break

if selected_features:

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    axes = axes.flatten()

    for i, column in enumerate(selected_features):

        values = pd.to_numeric(
            numeric_df[column],
            errors="coerce"
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        axes[i].hist(
            values,
            bins=30
        )

        axes[i].set_title(column)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    for i in range(len(selected_features), 6):
        axes[i].set_visible(False)

    fig.suptitle(
        "Distributions of Selected Numerical Features"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "04_numeric_distributions.png",
        dpi=150
    )

    plt.close()

else:
    print("No variable numerical features available.")


# =====================================================
# 8. NUMERICAL FEATURE CORRELATION WITH BINARY LABEL
# =====================================================

print("Analyzing numeric feature associations...")

correlation_features = numeric_df.loc[
    :,
    variable_numeric
].replace(
    [np.inf, -np.inf],
    np.nan
)

correlations = correlation_features.corrwith(
    df["Attack_label"]
).dropna()

correlations = correlations.sort_values(
    key=lambda values: values.abs(),
    ascending=False
)

correlations.rename_axis("feature").rename(
    "correlation_with_Attack_label"
).reset_index().to_csv(
    METRICS_DIR / "numeric_label_associations.csv",
    index=False
)
top_correlations = correlations.head(15).sort_values(
    key=lambda values: values.abs()
)

if not top_correlations.empty:

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_correlations.index,
        top_correlations.values
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.title(
        "Numerical Feature Correlation with Attack_label"
    )

    plt.xlabel("Pearson Correlation")
    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "05_numeric_label_correlation.png",
        dpi=150
    )

    plt.close()


# =====================================================
# 9. NUMERICAL SUMMARY
# =====================================================

print("Generating numerical statistics...")

numeric_summary = numeric_df.describe().T

numeric_summary["skewness"] = (
    numeric_df.skew(numeric_only=True)
)

numeric_summary.to_csv(
    METRICS_DIR / "numeric_feature_summary.csv",
    index_label="feature"
)
# =====================================================
# 10. FINAL REPORT
# =====================================================

print("\n" + "=" * 65)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nSelected numerical features:")
for feature in selected_features:
    print("-", feature)

print("\nGenerated visualizations:")

for file in sorted(FIGURE_DIR.glob("*.png")):
    print("-", file.name)

print("\nGenerated metric files:")

for file in sorted(METRICS_DIR.glob("*.csv")):
    print("-", file.name)

print("\nFigure directory:")
print(FIGURE_DIR)

print("\nMetrics directory:")
print(METRICS_DIR)

print("\nOriginal dataset was not modified.")
