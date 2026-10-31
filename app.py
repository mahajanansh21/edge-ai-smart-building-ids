"""Phase 2 starter app: verifies the project workspace before ML is added."""
from pathlib import Path

import streamlit as st

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODEL_DIR, PROJECT_ROOT

st.set_page_config(
    page_title="Edge AI Smart Building IDS",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Edge AI-Based Intrusion Detection for Smart Buildings")
st.caption("Phase 2 workspace check — model training and detection are not implemented yet.")

st.info(
    "This is the initial project shell. It does not classify traffic yet. "
    "We will connect it to a real trained model after dataset inspection and evaluation."
)


def folder_status(path: Path) -> str:
    return "Ready" if path.exists() and path.is_dir() else "Missing"


def csv_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.csv")) if path.exists() else []

col1, col2, col3 = st.columns(3)
col1.metric("Project workspace", "Ready" if PROJECT_ROOT.exists() else "Missing")
col2.metric("Raw data folder", folder_status(DATA_RAW_DIR))
col3.metric("Model folder", folder_status(MODEL_DIR))

st.subheader("Workspace paths")
for label, path in [
    ("Raw dataset folder", DATA_RAW_DIR),
    ("Processed data folder", DATA_PROCESSED_DIR),
    ("Model artifacts folder", MODEL_DIR),
]:
    st.write(f"**{label}:** `{path}`")

st.subheader("Dataset file check")
raw_csvs = csv_files(DATA_RAW_DIR)
if raw_csvs:
    st.success(f"Found {len(raw_csvs)} CSV file(s) in data/raw.")
    st.dataframe(
        [
            {"File": file.name, "Size (MB)": round(file.stat().st_size / (1024**2), 2)}
            for file in raw_csvs
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning(
        "No CSV file found in data/raw yet. In Phase 2, download the dataset and "
        "place the extracted CSV here. We will inspect its real schema in Phase 3."
    )

st.subheader("Build status")
st.write("- [x] Project structure created")
st.write("- [x] Python environment and dependency manifest prepared")
st.write("- [ ] Dataset downloaded and verified")
st.write("- [ ] Dataset schema inspected")
st.write("- [ ] Preprocessing and model training implemented")
st.write("- [ ] Edge inference and full dashboard implemented")

st.caption("Educational defensive prototype. Do not connect this starter app to live production networks.")
