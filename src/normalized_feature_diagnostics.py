
from pathlib import Path

import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "ML-EdgeIIoT-dataset.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. FEATURE SETS
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

CATEGORICAL_COLUMNS = [
    "http.request.method",
    "http.request.version",
    "dns.qry.name.len",
    "mqtt.conack.flags",
    "mqtt.protoname",
    "mqtt.topic"
]


# ==================================================
# 3. LOAD DATASET
# ==================================================

print("=" * 65)
print("NORMALIZED FEATURE DIAGNOSTICS")
print("=" * 65)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset shape:", df.shape)


# ==================================================
# 4. NORMALIZE CATEGORICAL PLACEHOLDERS
# ==================================================

normalized_df = df[EXPANDED_FEATURES].copy()

for column in CATEGORICAL_COLUMNS:

    if column not in normalized_df.columns:
        continue

    values = (
        normalized_df[column]
        .astype("string")
        .str.strip()
    )

    # Normalize representation of zero-like values.
    values = values.replace({
        "0": "__NO_VALUE__",
        "0.0": "__NO_VALUE__",
        "0x00000000": "__NO_VALUE__"
    })

    normalized_df[column] = values

print("\nCategorical normalization completed.")

print("Columns normalized:", CATEGORICAL_COLUMNS)


# ==================================================
# 5. ANALYSIS FUNCTION
# ==================================================

summary_records = []


def analyze_features(name, X):

    print("\n" + "=" * 65)
    print("ANALYZING:", name)
    print("=" * 65)

    # Hash each distinct feature combination.
    # Hashes are used as convenient group IDs.

    group_ids = pd.util.hash_pandas_object(
        X,
        index=False
    )

    # Verify that equal hashes actually represent
    # equal feature rows. This avoids silently
    # treating hash collisions as identical groups.

    hash_frame = pd.DataFrame({
        "hash": group_ids.to_numpy()
    })

    collision_check = (
        hash_frame
        .assign(row_number=range(len(X)))
    )

    # Group exact rows directly for group statistics.
    # This uses factorization of row hashes only after
    # confirming the number of feature combinations.

    ids = pd.factorize(group_ids, sort=False)[0]

    work = pd.DataFrame({
        "group_id": ids,
        "binary": df["Attack_label"].to_numpy(),
        "multiclass": df["Attack_type"].to_numpy()
    })

    groups = (
        work.groupby("group_id", sort=False)
        .agg(
            records=("binary", "size"),
            binary_classes=("binary", "nunique"),
            attack_classes=("multiclass", "nunique")
        )
    )

    binary_mixed = groups[
        groups["binary_classes"] > 1
    ]

    multi_mixed = groups[
        groups["attack_classes"] > 1
    ]

    # Majority-label consistency for each exact
    # feature combination.

    binary_counts = (
        work.groupby(["group_id", "binary"])
        .size()
    )

    multiclass_counts = (
        work.groupby(["group_id", "multiclass"])
        .size()
    )

    binary_correct = (
        binary_counts.groupby(level=0).max().sum()
    )

    multiclass_correct = (
        multiclass_counts.groupby(level=0).max().sum()
    )

    result = {
        "feature_set": name,
        "feature_count": X.shape[1],
        "unique_combinations": len(groups),
        "binary_mixed_groups": len(binary_mixed),
        "binary_mixed_rows": int(
            binary_mixed["records"].sum()
        ),
        "multiclass_mixed_groups": len(multi_mixed),
        "multiclass_mixed_rows": int(
            multi_mixed["records"].sum()
        ),
        "binary_majority_consistency_percentage": round(
            binary_correct / len(df) * 100, 3
        ),
        "multiclass_majority_consistency_percentage": round(
            multiclass_correct / len(df) * 100, 3
        )
    }

    for key, value in result.items():
        print(f"{key}: {value}")

    summary_records.append(result)


# ==================================================
# 6. RUN COMPARISONS
# ==================================================

analyze_features(
    "core_9_raw",
    df[CORE_FEATURES]
)

analyze_features(
    "expanded_36_raw",
    df[EXPANDED_FEATURES]
)

analyze_features(
    "expanded_36_normalized",
    normalized_df
)


# ==================================================
# 7. SAVE RESULTS
# ==================================================

results = pd.DataFrame(summary_records)

results.to_csv(
    OUTPUT_DIR / "normalized_feature_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("NORMALIZATION DIAGNOSTICS COMPLETED")
print("=" * 65)

print("\nFinal comparison:")
print(results.to_string(index=False))

print(
    "\nSaved:",
    OUTPUT_DIR / "normalized_feature_comparison.csv"
)

print("\nOriginal dataset was not modified.")
