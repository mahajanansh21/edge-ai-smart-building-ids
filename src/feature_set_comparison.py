
from pathlib import Path
import json

import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "raw"
    / "ML-EdgeIIoT-dataset.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "metrics"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# 2. TARGET COLUMNS
# ==================================================

TARGETS = [
    "Attack_label",
    "Attack_type"
]


# ==================================================
# 3. FEATURE SET A: ORIGINAL 9 FEATURES
# ==================================================

CORE_FEATURES = [
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


# ==================================================
# 4. FEATURE SET B: EXPANDED 36 FEATURES
# ==================================================

EXPANDED_FEATURES = [
    "arp.opcode",
    "arp.hw.size",
    "icmp.checksum",
    "icmp.seq_le",
    "http.content_length",
    "http.request.method",
    "http.request.version",
    "http.response",
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.checksum",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",
    "tcp.seq",
    "udp.stream",
    "udp.time_delta",
    "dns.qry.name",
    "dns.qry.name.len",
    "dns.qry.qu",
    "dns.retransmission",
    "dns.retransmit_request",
    "mqtt.conack.flags",
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msgtype",
    "mqtt.protoname",
    "mqtt.topic",
    "mqtt.topic_len",
    "mqtt.ver"
]


# ==================================================
# 5. LOAD DATASET
# ==================================================

print("=" * 70)
print("FEATURE SET COMPARISON")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset:", df.shape)


# ==================================================
# 6. VALIDATE FEATURE COLUMNS
# ==================================================

all_features = (
    CORE_FEATURES
    + EXPANDED_FEATURES
    + TARGETS
)

missing = sorted(
    set(all_features) - set(df.columns)
)

if missing:
    raise ValueError(
        f"Missing feature columns: {missing}"
    )


# ==================================================
# 7. ANALYZE FEATURE SETS
# ==================================================

summary_records = []
class_records = []


def analyze_feature_set(name, feature_list):

    print("\n" + "=" * 70)
    print("ANALYZING:", name)
    print("=" * 70)

    X = df[feature_list]

    # Identical feature combinations receive
    # the same group ID.

    group_ids = (
        X.groupby(
            feature_list,
            dropna=False,
            sort=False
        )
        .ngroup()
    )

    work = pd.DataFrame({
        "group_id": group_ids.to_numpy(),
        "Attack_label": df["Attack_label"],
        "Attack_type": df["Attack_type"]
    })

    group_stats = (
        work.groupby("group_id", sort=False)
        .agg(
            records=("Attack_type", "size"),
            binary_classes=("Attack_label", "nunique"),
            attack_classes=("Attack_type", "nunique")
        )
    )

    binary_mixed = group_stats[
        group_stats["binary_classes"] > 1
    ]

    multiclass_mixed = group_stats[
        group_stats["attack_classes"] > 1
    ]

    binary_rows = int(
        binary_mixed["records"].sum()
    )

    multiclass_rows = int(
        multiclass_mixed["records"].sum()
    )

    print("Number of features:", len(feature_list))
    print("Unique combinations:", len(group_stats))

    print(
        "Binary-mixed groups:",
        len(binary_mixed)
    )

    print(
        "Rows in binary-mixed groups:",
        binary_rows
    )

    print(
        "Multiclass-mixed groups:",
        len(multiclass_mixed)
    )

    print(
        "Rows in multiclass-mixed groups:",
        multiclass_rows
    )

    # Count distinct feature groups per class.

    class_groups = (
        work.drop_duplicates(
            ["group_id", "Attack_type"]
        )
        .groupby("Attack_type")
        .agg(
            unique_feature_groups=("group_id", "nunique"),
            records=("Attack_type", "size")
        )
        .sort_values("unique_feature_groups")
    )

    print("\nFeature combinations per class:")
    print(class_groups.to_string())

    for attack_type, row in class_groups.iterrows():

        class_records.append({
            "feature_set": name,
            "attack_type": attack_type,
            "unique_feature_groups": int(
                row["unique_feature_groups"]
            ),
            "records": int(row["records"])
        })

    summary_records.append({
        "feature_set": name,
        "feature_count": len(feature_list),
        "unique_combinations": len(group_stats),
        "binary_mixed_groups": len(binary_mixed),
        "binary_mixed_rows": binary_rows,
        "binary_mixed_percentage": round(
            binary_rows / len(df) * 100,
            3
        ),
        "multiclass_mixed_groups": len(multiclass_mixed),
        "multiclass_mixed_rows": multiclass_rows,
        "multiclass_mixed_percentage": round(
            multiclass_rows / len(df) * 100,
            3
        )
    })


# ==================================================
# 8. RUN BOTH COMPARISONS
# ==================================================

analyze_feature_set(
    "core_9",
    CORE_FEATURES
)

analyze_feature_set(
    "expanded_36",
    EXPANDED_FEATURES
)


# ==================================================
# 9. AUDIT EXACT COMPLETE-ROW DUPLICATES
# ==================================================

print("\n" + "=" * 70)
print("COMPLETE FEATURE RECORD DUPLICATE AUDIT")
print("=" * 70)

complete_features = df.drop(
    columns=TARGETS
)

hashes = pd.util.hash_pandas_object(
    complete_features,
    index=False
)

hash_groups = pd.DataFrame({
    "feature_hash": hashes.to_numpy(),
    "Attack_label": df["Attack_label"],
    "Attack_type": df["Attack_type"]
})

full_group_stats = (
    hash_groups.groupby("feature_hash")
    .agg(
        records=("Attack_type", "size"),
        binary_classes=("Attack_label", "nunique"),
        attack_classes=("Attack_type", "nunique")
    )
)

repeated = full_group_stats[
    full_group_stats["records"] > 1
]

mixed = full_group_stats[
    (
        full_group_stats["binary_classes"] > 1
    )
    |
    (
        full_group_stats["attack_classes"] > 1
    )
]

duplicate_summary = {
    "total_records": len(df),
    "unique_complete_feature_groups": len(full_group_stats),
    "repeated_complete_groups": len(repeated),
    "rows_in_repeated_groups": int(
        repeated["records"].sum()
    ),
    "excess_duplicate_records": int(
        (repeated["records"] - 1).sum()
    ),
    "mixed_label_duplicate_groups": len(mixed)
}

print("\nComplete-record duplicate results:")

for key, value in duplicate_summary.items():
    print(f"{key}: {value}")


# ==================================================
# 10. SAVE RESULTS
# ==================================================

pd.DataFrame(summary_records).to_csv(
    OUTPUT_DIR / "feature_set_comparison.csv",
    index=False
)

pd.DataFrame(class_records).to_csv(
    OUTPUT_DIR / "feature_set_class_comparison.csv",
    index=False
)

with open(
    OUTPUT_DIR / "complete_record_duplicate_audit.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        duplicate_summary,
        file,
        indent=4
    )


# ==================================================
# 11. COMPLETION
# ==================================================

print("\n" + "=" * 70)
print("FEATURE SET COMPARISON COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(OUTPUT_DIR / "feature_set_comparison.csv")
print(OUTPUT_DIR / "feature_set_class_comparison.csv")
print(OUTPUT_DIR / "complete_record_duplicate_audit.json")

print("\nOriginal dataset was not modified.")
