
from pathlib import Path
import json

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
# 2. VERIFY DATASET
# ==========================================

print("=" * 65)
print("EDGE-IIOTSET FEATURE RISK AUDIT")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("\nDataset found:", DATA_PATH)


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset loaded:", df.shape)


# ==========================================
# 4. FEATURES TO INVESTIGATE
# ==========================================

risk_groups = {
    "timestamp_and_identity": [
        "frame.time",
        "ip.src_host",
        "ip.dst_host",
        "arp.dst.proto_ipv4",
        "arp.src.proto_ipv4"
    ],

    "packet_identifiers": [
        "tcp.ack_raw",
        "tcp.ack",
        "tcp.seq",
        "tcp.checksum",
        "udp.stream",
        "icmp.checksum",
        "icmp.seq_le"
    ],

    "raw_content": [
        "tcp.payload",
        "tcp.options",
        "http.file_data",
        "http.request.full_uri",
        "http.request.uri.query",
        "http.referer",
        "mqtt.msg",
        "mqtt.topic",
        "dns.qry.name"
    ],

    "network_ports": [
        "tcp.srcport",
        "tcp.dstport",
        "udp.port"
    ]
}


# ==========================================
# 5. INSPECT FEATURE VALUES
# ==========================================

print("\n" + "=" * 65)
print("FEATURE VALUE INSPECTION")
print("=" * 65)

sample_report = []

for group, columns in risk_groups.items():

    print(f"\nGROUP: {group}")

    for column in columns:

        if column not in df.columns:
            print(f"Column not found: {column}")
            continue

        values = df[column]

        unique_count = values.nunique(
            dropna=False
        )

        samples = (
            values.dropna()
            .astype(str)
            .unique()[:5]
        )

        sample_values = [
            value[:100]
            for value in samples
        ]

        print("\nFeature:", column)
        print("Data type:", values.dtype)
        print("Unique values:", unique_count)
        print("Sample values:", sample_values)

        sample_report.append({
            "feature": column,
            "risk_group": group,
            "data_type": str(values.dtype),
            "unique_values": int(unique_count),
            "sample_values": " | ".join(sample_values)
        })


# ==========================================
# 6. IDENTICAL COLUMN CHECK
# ==========================================

print("\n" + "=" * 65)
print("IDENTICAL FEATURE COLUMN CHECK")
print("=" * 65)

target_columns = [
    "Attack_label",
    "Attack_type"
]

features = df.drop(
    columns=target_columns,
    errors="ignore"
)

identical_pairs = []

columns = list(features.columns)

total_comparisons = (
    len(columns) * (len(columns) - 1) // 2
)

print("Comparing", total_comparisons, "feature pairs...")

for i in range(len(columns)):

    for j in range(i + 1, len(columns)):

        first = columns[i]
        second = columns[j]

        if features[first].equals(features[second]):

            identical_pairs.append({
                "feature_1": first,
                "feature_2": second
            })

            print(
                f"Identical: {first} == {second}"
            )

if not identical_pairs:
    print("No identical columns found.")


# ==========================================
# 7. CONSTANT FEATURE CHECK
# ==========================================

print("\n" + "=" * 65)
print("CONSTANT FEATURE CHECK")
print("=" * 65)

constant_features = [
    column
    for column in features.columns
    if features[column].nunique(dropna=False) <= 1
]

if constant_features:
    for column in constant_features:
        print("-", column)
else:
    print("No constant features found.")


# ==========================================
# 8. SAVE AUDIT REPORTS
# ==========================================

print("\nSaving audit reports...")

pd.DataFrame(sample_report).to_csv(
    OUTPUT_DIR / "feature_risk_samples.csv",
    index=False
)

pd.DataFrame(identical_pairs).to_csv(
    OUTPUT_DIR / "identical_feature_pairs.csv",
    index=False
)

report = {
    "dataset_shape": list(df.shape),
    "risk_groups": risk_groups,
    "constant_features": constant_features,
    "identical_feature_pairs": identical_pairs
}

with open(
    OUTPUT_DIR / "feature_risk_audit.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(report, file, indent=4)


# ==========================================
# 9. COMPLETION
# ==========================================

print("\n" + "=" * 65)
print("FEATURE RISK AUDIT COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nGenerated files:")

print(OUTPUT_DIR / "feature_risk_samples.csv")
print(OUTPUT_DIR / "identical_feature_pairs.csv")
print(OUTPUT_DIR / "feature_risk_audit.json")

print("\nOriginal dataset was not modified.")
