
from pathlib import Path
import json

import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "ML-EdgeIIoT-dataset.csv"

OUTPUT_DIR = ROOT / "outputs" / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. PROVISIONAL EXCLUSION POLICY
# ==========================================

TARGET_COLUMNS = [
    "Attack_label",
    "Attack_type"
]

RAW_OR_IDENTITY_FEATURES = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.dst.proto_ipv4",
    "arp.src.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "http.request.uri.query",
    "http.referer",
    "icmp.transmit_timestamp",
    "tcp.options",
    "tcp.payload",
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "mqtt.msg"
]

CONSTANT_FEATURES = [
    "icmp.unused",
    "http.tls_port",
    "dns.qry.type",
    "dns.retransmit_request_in",
    "mqtt.msg_decoded_as",
    "mbtcp.len",
    "mbtcp.trans_id",
    "mbtcp.unit_id"
]

REDUNDANT_FEATURES = [
    "mqtt.proto_len"
]

EXCLUDE_COLUMNS = set(
    TARGET_COLUMNS
    + RAW_OR_IDENTITY_FEATURES
    + CONSTANT_FEATURES
    + REDUNDANT_FEATURES
)


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("=" * 65)
print("RETAINED FEATURE AUDIT")
print("=" * 65)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset shape:", df.shape)


# ==========================================
# 4. SELECT CANDIDATE FEATURES
# ==========================================

candidate_columns = [
    col
    for col in df.columns
    if col not in EXCLUDE_COLUMNS
]

print("\nOriginal feature count:", len(df.columns) - 2)
print("Excluded feature count:", len(
    [c for c in df.columns if c in EXCLUDE_COLUMNS]
))
print("Candidate feature count:", len(candidate_columns))

print("\nCandidate features:")

for col in candidate_columns:
    print("-", col)


# ==========================================
# 5. PROFILE EACH CANDIDATE FEATURE
# ==========================================

records = []

print("\n" + "=" * 65)
print("FEATURE VALUE ANALYSIS")
print("=" * 65)

for column in candidate_columns:

    series = df[column]

    unique_count = int(
        series.nunique(dropna=False)
    )

    missing_percentage = (
        series.isna().mean() * 100
    )

    zero_percentage = None
    numeric_parse_percentage = None

    if (
        is_object_dtype(series.dtype)
        or is_string_dtype(series.dtype)
    ):

        converted = pd.to_numeric(
            series,
            errors="coerce"
        )

        numeric_parse_percentage = (
            converted.notna().mean() * 100
        )

        print(f"\nCategorical/string: {column}")

    else:

        converted = pd.to_numeric(
            series,
            errors="coerce"
        )

        zero_percentage = (
            converted.eq(0).mean() * 100
        )

        numeric_parse_percentage = (
            converted.notna().mean() * 100
        )

        print(f"\nNumerical: {column}")

    top_values = (
        series.astype("string")
        .fillna("<MISSING>")
        .value_counts()
        .head(8)
    )

    top_values_dict = {
        str(k): int(v)
        for k, v in top_values.items()
    }

    print("Data type:", series.dtype)
    print("Unique values:", unique_count)
    print("Missing %:", round(missing_percentage, 3))
    print(
        "Numeric conversion %:",
        round(numeric_parse_percentage, 3)
    )

    if zero_percentage is not None:
        print(
            "Zero percentage:",
            round(zero_percentage, 3)
        )

    print("Most common values:")
    print(top_values.to_string())

    records.append({
        "feature": column,
        "data_type": str(series.dtype),
        "unique_values": unique_count,
        "missing_percentage": round(missing_percentage, 3),
        "numeric_parse_percentage": round(
            numeric_parse_percentage, 3
        ),
        "zero_percentage": (
            round(zero_percentage, 3)
            if zero_percentage is not None
            else None
        ),
        "top_values": json.dumps(top_values_dict)
    })


# ==========================================
# 6. VERIFY REDUNDANT MQTT FEATURES
# ==========================================

print("\n" + "=" * 65)
print("MQTT REDUNDANCY VERIFICATION")
print("=" * 65)

if {
    "mqtt.proto_len",
    "mqtt.ver"
}.issubset(df.columns):

    identical = df["mqtt.proto_len"].equals(
        df["mqtt.ver"]
    )

    print("mqtt.proto_len == mqtt.ver:", identical)

else:
    identical = False
    print("Required MQTT columns unavailable.")


# ==========================================
# 7. SAVE REPORT
# ==========================================

profile = pd.DataFrame(records)

profile.to_csv(
    OUTPUT_DIR / "retained_feature_profile.csv",
    index=False
)

report = {
    "dataset_shape": list(df.shape),
    "candidate_feature_count": len(candidate_columns),
    "candidate_features": candidate_columns,
    "raw_identity_exclusions": RAW_OR_IDENTITY_FEATURES,
    "constant_feature_exclusions": CONSTANT_FEATURES,
    "redundant_feature_exclusions": REDUNDANT_FEATURES,
    "mqtt_columns_identical": bool(identical)
}

with open(
    OUTPUT_DIR / "retained_feature_audit.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(report, file, indent=4)


# ==========================================
# 8. COMPLETION
# ==========================================

print("\n" + "=" * 65)
print("RETAINED FEATURE AUDIT COMPLETED")
print("=" * 65)

print("\nGenerated reports:")

print(OUTPUT_DIR / "retained_feature_profile.csv")
print(OUTPUT_DIR / "retained_feature_audit.json")

print("\nOriginal dataset was not modified.")
