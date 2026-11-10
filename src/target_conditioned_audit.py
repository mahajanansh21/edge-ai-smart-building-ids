
from pathlib import Path
import json

import numpy as np
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
# 2. FEATURES TO INVESTIGATE
# ==========================================

AUDIT_FEATURES = [
    "mqtt.protoname",
    "mqtt.topic",
    "mqtt.conack.flags",
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msgtype",
    "mqtt.topic_len",
    "mqtt.ver",

    "http.request.method",
    "http.request.version",
    "http.response",
    "http.content_length",

    "dns.qry.name",
    "dns.qry.name.len",
    "dns.qry.qu",
    "dns.retransmission",
    "dns.retransmit_request",

    "tcp.flags",
    "tcp.flags.ack",
    "tcp.connection.syn",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.synack",

    "arp.opcode",
    "arp.hw.size"
]


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("=" * 65)
print("TARGET-CONDITIONED FEATURE AUDIT")
print("=" * 65)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset:", df.shape)

required = [
    "Attack_label",
    "Attack_type"
]

if not set(required).issubset(df.columns):
    raise ValueError("Target columns are missing.")


# ==========================================
# 4. FEATURE / LABEL ASSOCIATIONS
# ==========================================

all_records = []
summary_records = []

for feature in AUDIT_FEATURES:

    if feature not in df.columns:
        print("\nSkipping missing column:", feature)
        continue

    print("\n" + "=" * 65)
    print("FEATURE:", feature)
    print("=" * 65)

    raw = (
        df[feature]
        .astype("string")
        .str.strip()
        .fillna("<MISSING>")
    )

    value_counts = raw.value_counts().head(15)

    cross_tab = pd.crosstab(
        raw,
        df["Attack_label"]
    )

    print("\nMost common values by binary label:")

    top_table = cross_tab.loc[
        cross_tab.sum(axis=1)
        .nlargest(15)
        .index
    ]

    print(top_table.to_string())

    type_tab = pd.crosstab(
        raw,
        df["Attack_type"]
    )

    for value in value_counts.index:

        total = int((raw == value).sum())

        normal_count = int(
            ((raw == value) & (df["Attack_label"] == 0)).sum()
        )

        attack_count = int(
            ((raw == value) & (df["Attack_label"] == 1)).sum()
        )

        attack_types = type_tab.loc[value].to_dict()

        all_records.append({
            "feature": feature,
            "raw_value": str(value)[:150],
            "total_count": total,
            "normal_count": normal_count,
            "attack_count": attack_count,
            "attack_types": json.dumps({
                str(k): int(v)
                for k, v in attack_types.items()
            })
        })

    # ======================================
    # 5. CHECK STRING ZERO REPRESENTATIONS
    # ======================================

    if (
        feature in [
            "mqtt.protoname",
            "mqtt.topic",
            "mqtt.conack.flags"
        ]
    ):

        normalized = raw.replace({
            "0.0": "<ZERO>",
            "0": "<ZERO>",
            "0x00000000": "<ZERO>"
        })

        print("\nRaw zero-like values:")
        print(
            raw[
                raw.isin(["0.0", "0", "0x00000000"])
            ].value_counts()
        )

        print("\nAfter provisional zero normalization:")

        print(
            pd.crosstab(
                normalized,
                df["Attack_label"]
            ).to_string()
        )

    # ======================================
    # 6. SUMMARY
    # ======================================

    summary_records.append({
        "feature": feature,
        "data_type": str(df[feature].dtype),
        "unique_values": int(raw.nunique()),
        "top_value": str(value_counts.index[0]),
        "top_value_count": int(value_counts.iloc[0])
    })


# ==========================================
# 7. NUMERICAL ZERO RATE BY LABEL
# ==========================================

print("\n" + "=" * 65)
print("NUMERICAL ZERO-RATE ANALYSIS")
print("=" * 65)

numeric_features = [
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msgtype",
    "mqtt.topic_len",
    "mqtt.ver",
    "http.response",
    "http.content_length",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.connection.syn",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.synack",
    "arp.opcode",
    "arp.hw.size",
    "dns.retransmission",
    "dns.retransmit_request"
]

zero_records = []

for feature in numeric_features:

    if feature not in df.columns:
        continue

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    for label in sorted(df["Attack_label"].unique()):

        subset = values[df["Attack_label"] == label]

        valid = subset.notna()

        zero_percentage = (
            subset[valid].eq(0).mean() * 100
            if valid.any()
            else np.nan
        )

        record = {
            "feature": feature,
            "attack_label": int(label),
            "zero_percentage": round(
                float(zero_percentage), 3
            )
        }

        zero_records.append(record)

        print(
            feature,
            "| Attack_label =", label,
            "| Zero % =", record["zero_percentage"]
        )


# ==========================================
# 8. SAVE RESULTS
# ==========================================

pd.DataFrame(all_records).to_csv(
    OUTPUT_DIR / "target_conditioned_feature_values.csv",
    index=False
)

pd.DataFrame(summary_records).to_csv(
    OUTPUT_DIR / "target_conditioned_summary.csv",
    index=False
)

pd.DataFrame(zero_records).to_csv(
    OUTPUT_DIR / "feature_zero_rates_by_label.csv",
    index=False
)


# ==========================================
# 9. COMPLETION
# ==========================================

print("\n" + "=" * 65)
print("TARGET-CONDITIONED AUDIT COMPLETED")
print("=" * 65)

print("\nGenerated files:")

print(
    OUTPUT_DIR
    / "target_conditioned_feature_values.csv"
)

print(
    OUTPUT_DIR
    / "target_conditioned_summary.csv"
)

print(
    OUTPUT_DIR
    / "feature_zero_rates_by_label.csv"
)

print("\nOriginal dataset was not modified.")
