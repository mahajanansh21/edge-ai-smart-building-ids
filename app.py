"""Phase 6/7 application: Edge AI IDS inference, offline replay, alerts and telemetry."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st

from src.edge_inference import (
    RAW_DATASET,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    load_model_bundle,
    load_test_confusion_matrix,
    load_test_report,
    predict_dataframe,
    read_dataset_preview,
    read_full_label_distribution,
    read_validation_for_replay,
)
from src.phase7_ops import (
    build_alert_feed,
    build_manual_record,
    diagnose_input,
    load_training_profile,
    run_offline_replay,
    sample_validation_records,
    save_run_artifacts,
    summarize_predictions,
)

from src.phase9_uploads import CSVUploadError, read_limited_csv
from src.phase9_ui import (
    apply_branding,
    render_input_quick_guide,
    render_page_banner,
    render_pipeline_diagram,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_RUNS = ROOT / "outputs" / "phase7" / "runs"
TRAINING_PROFILE_PATH = ROOT / "deployment" / "training_profile.json"
logger = logging.getLogger(__name__)
MAX_CSV_UPLOAD_MB = 10
MAX_CSV_ROWS = 5000

st.set_page_config(
    page_title="Edge AI Smart Building IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_branding()

@st.cache_resource(show_spinner="Loading frozen Phase 5 models...")
def get_bundle():
    return load_model_bundle()


@st.cache_data(show_spinner=False, ttl=300)
def get_preview(nrows: int = 1500):
    return read_dataset_preview(nrows=nrows)


@st.cache_data(show_spinner="Reading full dataset label counts...", ttl=600)
def get_full_distribution():
    return read_full_label_distribution()


@st.cache_data(show_spinner="Loading validation records for offline replay...", ttl=300)
def get_validation():
    return read_validation_for_replay()


@st.cache_resource(show_spinner="Building manual-input defaults from the training split...")
def get_training_profile(source_features: tuple[str, ...]):
    return load_training_profile(TRAIN_SPLIT, list(source_features), TRAINING_PROFILE_PATH)


def initialize_state():
    defaults = {
        "last_predictions": None,
        "last_performance": None,
        "last_diagnostics": None,
        "last_source": None,
        "last_run_id": None,
        "last_created_at": None,
        "alert_history": pd.DataFrame(),
        "run_history": [],
        "last_saved_run": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def register_run(predictions: pd.DataFrame, performance: dict, source: str, diagnostics: dict | None = None):
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    created_at = datetime.now(timezone.utc).isoformat()
    results = predictions.copy().reset_index(drop=True)
    results.insert(0, "Run_ID", run_id)
    results.insert(1, "Input_Source", source)
    alerts = build_alert_feed(results, run_id, created_at)
    prior_alerts = st.session_state.alert_history
    st.session_state.alert_history = pd.concat([prior_alerts, alerts], ignore_index=True).tail(5000)
    st.session_state.last_predictions = results
    st.session_state.last_performance = dict(performance)
    st.session_state.last_diagnostics = diagnostics
    st.session_state.last_source = source
    st.session_state.last_run_id = run_id
    st.session_state.last_created_at = created_at
    st.session_state.last_saved_run = None
    st.session_state.run_history.append({
        "run_id": run_id,
        "source": source,
        "created_at_utc": created_at,
        "records": int(len(results)),
        "predicted_attacks": int((results["Predicted_Binary"] == "Attack").sum()),
        "alerts": int(len(alerts)),
        "model_processing_seconds": float(performance.get("total_seconds", performance.get("model_processing_seconds", 0.0))),
        "records_per_second": float(performance.get("records_per_second", performance.get("records_per_second_model_processing", 0.0))),
    })
    st.session_state.run_history = st.session_state.run_history[-100:]


def show_result_summary():
    predictions = st.session_state.last_predictions
    if predictions is None:
        st.info("No inference has been performed in this browser session yet.")
        return

    st.markdown("#### Latest inference result")
    summary = summarize_predictions(predictions)
    a, b, c = st.columns(3)
    a.metric("Records processed", f"{summary['records_processed']:,}")
    b.metric("Predicted attacks", f"{summary['predicted_attacks']:,}")
    c.metric("Predicted normal", f"{summary['predicted_normal']:,}")

    if summary["predicted_attack_types"]:
        counts = pd.Series(summary["predicted_attack_types"]).sort_values(ascending=False)
        fig = px.bar(
            x=counts.index, y=counts.values,
            labels={"x": "Model-predicted attack category", "y": "Records"},
            title="Predicted attack categories (attack predictions only)",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    columns = [
        name for name in [
            "Replay_Sequence", "Predicted_Binary", "Predicted_Attack_Type", "Final_Detection",
            "Binary_Attack_Score_Uncalibrated", "Multiclass_Top_Score_Uncalibrated",
            "Dataset_Attack_label_reference", "Dataset_Attack_type_reference",
        ] if name in predictions.columns
    ]
    st.dataframe(predictions[columns].head(250), use_container_width=True, hide_index=True)
    if "Binary_Attack_Score_Uncalibrated" in predictions or "Multiclass_Top_Score_Uncalibrated" in predictions:
        st.caption("Model probability outputs, when shown, are uncalibrated scores. They are not verified confidence levels.")
    if "Dataset_Attack_type_reference" in predictions.columns:
        st.caption("Dataset reference labels are shown only for validation/demo rows. They are not passed to either model.")

    st.download_button(
        "Download latest predictions CSV",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"{st.session_state.last_run_id}_predictions.csv",
        mime="text/csv",
        key="download_latest_predictions",
    )
    if st.button("Save predictions, alerts and metrics under outputs/phase7/runs", key="save_latest_run"):
        try:
            run_dir = save_run_artifacts(
                predictions,
                st.session_state.last_performance or {},
                OUTPUT_RUNS,
                st.session_state.last_run_id,
                st.session_state.last_source or "Unknown",
            )
            st.session_state.last_saved_run = str(run_dir.relative_to(ROOT))
            st.success(f"Run reports saved to: {st.session_state.last_saved_run}")
        except FileExistsError:
            st.warning("This run was already saved. Use the downloaded CSV or create a new inference run before saving another bundle.")
        except Exception:
            logger.exception("Could not save run reports.")
            st.error("Could not save the run reports. Download the prediction CSV for a durable copy; hosted files may be temporary.")
    if st.session_state.last_saved_run:
        st.caption(f"Saved report folder: {st.session_state.last_saved_run}")


def show_input_diagnostics(frame: pd.DataFrame, profile: dict):
    diagnostics = diagnose_input(frame, bundle.source_features, profile)
    with st.expander("Input quality checks", expanded=False):
        st.json(diagnostics)
        if diagnostics["missing_required_columns"]:
            st.error("Required source features are missing. Inference cannot proceed.")
        elif diagnostics["invalid_numeric_cells"] or diagnostics["unseen_categorical_values"] or diagnostics["numeric_values_outside_training_q01_q99"]:
            st.warning("Some inputs differ from values commonly observed in training. The range check is a simple diagnostic, not a complete out-of-distribution detector.")
        else:
            st.success("No issues were found by the simple feature-level checks.")
    return diagnostics


def run_single_or_batch(frame: pd.DataFrame, source: str, profile: dict):
    if frame.empty:
        st.error("The input contains no rows.")
        return
    if len(frame) > 5000:
        st.error("Please submit at most 5,000 records per analysis.")
        return
    diagnostics = show_input_diagnostics(frame, profile)
    if diagnostics["missing_required_columns"]:
        return
    try:
        with st.spinner("Applying saved Phase 3 preprocessing and running both frozen CPU models..."):
            predictions, performance = predict_dataframe(frame, bundle)
        register_run(predictions, performance, source, diagnostics)
        st.success(f"Inference completed for {len(predictions):,} record(s).")
    except Exception:
        logger.exception("Inference failed for source=%s", source)
        st.error("Inference could not be completed. Check the required 36 feature columns and input values, then try again. Technical details are recorded in the app logs.")


initialize_state()
try:
    bundle = get_bundle()
    bundle_error = None
except Exception:
    logger.exception("The frozen Phase 5 model bundle could not be loaded.")
    bundle = None
    bundle_error = True

with st.sidebar:
    st.title("🛡️ Edge AI IDS")
    st.caption("Smart Building Network Security")
    page = st.radio(
        "Navigation",
        [
            "Overview", "Dataset Explorer", "Model Evaluation", "Traffic Analyzer",
            "Live Simulation", "Threat Alerts", "Edge Performance", "About",
        ],
    )
    st.divider()
    st.caption("Frozen CPU models · Manual/CSV inference · Local offline validation replay")

render_page_banner(page, model_loaded=bundle is not None)

if bundle_error:
    st.error("The frozen Phase 5 model bundle could not be loaded. The app maintainer should review the deployment logs.")
    st.info("The hosted app needs all frozen model files and the two saved preprocessing metadata files in the repository.")
    st.stop()

try:
    profile = get_training_profile(tuple(bundle.source_features))
except Exception:
    logger.exception("The saved TRAIN-only feature profile could not be loaded.")
    st.error("The feature profile could not be loaded. Please try again later or contact the app maintainer.")
    st.stop()

if page == "Overview":
    st.subheader("Security operations overview")
    st.write("The application loads the frozen Phase 5 models. It does not train, modify, or select models.")
    render_pipeline_diagram()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Binary test accuracy", "93.64%")
    c2.metric("Binary test Macro-F1", "88.96%")
    c3.metric("Multiclass test accuracy", "83.51%")
    c4.metric("Multiclass test Macro-F1", "82.69%")
    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Frozen inference models")
        st.success("Binary: Random Forest · 42 features · excludes tcp.len")
        st.success("Multiclass: Decision Tree · 42 features · excludes tcp.checksum")
        st.caption("Both use the saved Phase 3 expanded36_normalized preprocessor.")
    with right:
        st.markdown("#### Runtime status")
        st.metric("Model bundle", "Loaded")
        st.metric("Inference device", "CPU")
        st.metric("Raw dataset", "Available" if RAW_DATASET.is_file() else "Not found")
    st.divider()
    latest = st.session_state.last_predictions
    if latest is not None:
        show_result_summary()
    else:
        st.info("Start with Traffic Analyzer for a manual record or CSV upload. Offline validation replay is available in the local setup when the validation split is present.")
    st.warning("Final test scores are the recorded Phase 5 results on a random record-level split. They do not establish performance on unseen buildings or devices.")

elif page == "Dataset Explorer":
    st.subheader("Edge-IIoTset dataset explorer")
    if not RAW_DATASET.is_file():
        st.info("Dataset Explorer is available in the local setup only. The raw dataset is intentionally not included in this public repository.")
    else:
        with st.spinner("Reading full-dataset label counts (only two label columns)..."):
            binary_counts, type_counts, total_rows = get_full_distribution()
        st.caption("The distribution charts below use all original dataset labels. The row preview is a separate, limited sample.")
        a, b, c, d = st.columns(4)
        a.metric("Original records", f"{total_rows:,}")
        b.metric("Original columns", "63")
        c.metric("Phase 3 source features", "36")
        d.metric("Transformed features", "43")
        binary_labels = binary_counts.copy()
        binary_labels.index = ["Normal" if int(label) == 0 else "Attack" for label in binary_labels.index]
        left, right = st.columns(2)
        with left:
            fig = px.bar(x=binary_labels.index, y=binary_labels.values, labels={"x": "Binary class", "y": "Records"}, title="Full dataset: Normal vs Attack")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(binary_labels.rename("Records").rename_axis("Class").reset_index(), use_container_width=True, hide_index=True)
        with right:
            fig = px.bar(x=type_counts.index, y=type_counts.values, labels={"x": "Attack category", "y": "Records"}, title="Full dataset: 15 traffic categories")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(type_counts.rename("Records").rename_axis("Attack type").reset_index(), use_container_width=True, hide_index=True)
        st.divider()
        nrows = st.slider("Preview rows to load", min_value=100, max_value=5000, value=1500, step=100)
        data = get_preview(nrows)
        st.caption(f"Previewing the first {len(data):,} rows only; full dataset distribution above is calculated separately.")
        st.dataframe(data.head(100), use_container_width=True, hide_index=True)
        st.download_button("Download current preview", data.head(5000).to_csv(index=False).encode("utf-8"), "edge_iiotset_preview.csv", "text/csv", key="download_dataset_preview")

elif page == "Model Evaluation":
    st.subheader("Final held-out test evaluation")
    st.warning("These are the recorded Phase 5 results. This page reads saved reports only; do not use test results for model tuning.")
    metric_rows = [
        {"Task": "Binary detection", "Model": "Random Forest", "Accuracy": 0.9363852556, "Macro-F1": 0.8895932088, "Balanced accuracy": 0.9304325567},
        {"Task": "Attack identification", "Model": "Decision Tree", "Accuracy": 0.8350603024, "Macro-F1": 0.8268810743, "Balanced accuracy": 0.8302945351},
    ]
    st.dataframe(pd.DataFrame(metric_rows).style.format({"Accuracy": "{:.2%}", "Macro-F1": "{:.2%}", "Balanced accuracy": "{:.2%}"}), use_container_width=True, hide_index=True)
    target = st.selectbox("Choose evaluation task", ["Attack_label", "Attack_type"])
    report = load_test_report(target)
    matrix = load_test_confusion_matrix(target)
    st.markdown("#### Per-class classification report")
    st.dataframe(report, use_container_width=True, hide_index=True)
    st.markdown("#### Final test confusion matrix")
    fig = go.Figure(data=go.Heatmap(z=matrix.values, x=[str(x) for x in matrix.columns], y=[str(y) for y in matrix.index], colorscale="Blues", text=matrix.values, texttemplate="%{text}"))
    fig.update_layout(xaxis_title="Predicted class", yaxis_title="Actual class", height=650)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Traffic Analyzer":
    st.subheader("Traffic Analyzer — analyze one new record or a CSV")
    st.write("Enter a new feature vector manually or upload a CSV containing all 36 source features.")
    render_input_quick_guide(
        source_feature_count=len(bundle.source_features),
        max_upload_mb=MAX_CSV_UPLOAD_MB,
        max_rows=MAX_CSV_ROWS,
    )
    input_methods = ["Manual single record", "Upload CSV"]
    if VALIDATION_SPLIT.is_file():
        input_methods.append("Choose validation example")
    mode = st.radio("Input method", input_methods, horizontal=True)

    if mode == "Manual single record":
        st.info("The form starts with per-feature median/mode defaults calculated from the training split. Edit the values to create a new input vector. This is a feature-vector experiment, not a captured packet or a verified synthetic network flow.")
        values: dict[str, object] = {}
        with st.form("manual_traffic_record_form"):
            st.markdown("**Numeric fields** — edit the values you want the models to analyse.")
            categorical_features = {"http.request.method", "http.request.version", "mqtt.protoname", "mqtt.topic"}
            numeric_features = [f for f in bundle.source_features if f not in categorical_features]
            left, right = st.columns(2)
            for index, feature in enumerate(numeric_features):
                target_col = left if index % 2 == 0 else right
                with target_col:
                    default = profile["numeric"].get(feature, {}).get("default", 0.0)
                    if not pd.notna(default):
                        default = 0.0
                    values[feature] = st.number_input(feature, value=float(default), format="%.6g", key=f"manual_{feature}")
            st.markdown("**Categorical fields** — enter values such as GET, HTTP/1.1, MQTT or Temperature_and_Humidity where appropriate.")
            cat_left, cat_right = st.columns(2)
            for index, feature in enumerate([f for f in bundle.source_features if f in categorical_features]):
                target_col = cat_left if index % 2 == 0 else cat_right
                with target_col:
                    info = profile["categorical"].get(feature, {})
                    default = info.get("default", "")
                    examples = ", ".join(info.get("known_values", [])[:8])
                    values[feature] = st.text_input(feature, value=default, help=f"Common training values: {examples}", key=f"manual_{feature}")
            submitted = st.form_submit_button("Run both frozen models on this record", type="primary")
        if submitted:
            try:
                one_row = build_manual_record(values, bundle.source_features)
                st.markdown("#### Submitted feature vector")
                st.dataframe(one_row.T.rename(columns={0: "Input value"}), use_container_width=True)
                run_single_or_batch(one_row, "Manual single record", profile)
            except Exception:
                logger.exception("Could not create the manual feature record.")
                st.error("Could not create the feature record. Check the entered values and try again.")

    elif mode == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload traffic CSV (all 36 source feature columns required)",
            type=["csv"],
            max_upload_size=MAX_CSV_UPLOAD_MB,
            key="traffic_csv_upload",
            help=f"Maximum file size: {MAX_CSV_UPLOAD_MB} MB. Maximum rows per analysis: {MAX_CSV_ROWS:,}.",
        )
        if uploaded is not None:
            try:
                frame = read_limited_csv(
                    uploaded,
                    max_rows=MAX_CSV_ROWS,
                    max_bytes=MAX_CSV_UPLOAD_MB * 1024 * 1024,
                )
                st.caption(f"Uploaded {len(frame):,} records and {frame.shape[1]} columns.")
                st.dataframe(frame.head(8), use_container_width=True, hide_index=True)
                if st.button("Analyze uploaded traffic", type="primary", key="analyze_uploaded_csv"):
                    run_single_or_batch(frame, "Uploaded CSV", profile)
            except CSVUploadError as exc:
                st.warning(str(exc))
            except Exception:
                logger.exception("Unexpected error while reading an uploaded CSV.")
                st.error("Could not read this CSV. Check that it is a valid UTF-8 CSV file and try again. Technical details are recorded in the app logs.")

    else:
        st.caption("This option selects a known validation record. Its ground-truth labels are retained only as a separate reference for learning; they are not model inputs.")
        try:
            validation = get_validation()
            attack_types = sorted(validation["Attack_type"].astype(str).unique())
            selected_type = st.selectbox("Choose a validation traffic category", attack_types)
            seed = st.number_input("Random seed for record selection", min_value=0, max_value=999999, value=2026)
            if st.button("Load one validation record", type="secondary", key="load_validation_record"):
                choices = validation.loc[validation["Attack_type"].astype(str).eq(selected_type)]
                st.session_state.validation_single = choices.sample(n=1, random_state=int(seed)).reset_index(drop=True)
            chosen = st.session_state.get("validation_single")
            if chosen is not None:
                st.write("Selected input preview:")
                st.dataframe(chosen[bundle.source_features].T.rename(columns={0: "Input value"}), use_container_width=True)
                if st.button("Classify selected validation record", type="primary", key="classify_validation_record"):
                    run_single_or_batch(chosen, "Validation example", profile)
        except Exception:
            logger.exception("Could not load a validation example.")
            st.warning("Could not load a validation example. Check that the local validation split is available.")

    if st.session_state.last_source in {"Manual single record", "Uploaded CSV", "Validation example"}:
        show_result_summary()

elif page == "Live Simulation":
    st.subheader("Offline IoT traffic replay")
    if not VALIDATION_SPLIT.is_file():
        st.info("Offline validation replay is available in the local setup only because the validation split is intentionally excluded from the public repository. Manual and CSV inference remain available online.")
        st.stop()
    st.warning("This replays rows from the validation split through frozen models. It does not capture packets, connect to a live network, or perform active scanning.")
    st.markdown("#### Configure replay")
    a, b, c = st.columns(3)
    with a:
        sample_size = st.select_slider("Number of records", options=[15, 30, 75, 150, 300, 500], value=150)
    with b:
        batch_size = st.select_slider("Records per batch", options=[5, 10, 25, 50, 100], value=25)
    with c:
        pacing_ms = st.select_slider("Optional replay pacing per batch", options=[0, 10, 25, 50, 100], value=10)
    strategy = st.radio("Sampling strategy", ["stratified", "random"], horizontal=True, format_func=lambda x: "Balanced across attack types" if x == "stratified" else "Random validation sample")
    seed = st.number_input("Replay random seed", min_value=0, max_value=999999, value=42, key="replay_seed")

    if st.button("Start offline simulation", type="primary"):
        try:
            validation = get_validation()
            selected = sample_validation_records(validation, int(sample_size), int(seed), strategy)
            progress = st.progress(0.0, text="Preparing replay...")
            status = st.empty()

            def update_progress(done: int, total: int):
                progress.progress(done / total, text=f"Replayed {done:,} / {total:,} records")
                status.caption(f"Processed {done:,} of {total:,} validation records")

            with st.spinner("Running batched CPU inference..."):
                results, performance = run_offline_replay(
                    selected, bundle, batch_size=int(batch_size),
                    pause_seconds=float(pacing_ms) / 1000.0,
                    progress_callback=update_progress,
                )
            diagnostics = diagnose_input(selected, bundle.source_features, profile)
            register_run(results, performance, "Offline validation replay", diagnostics)
            st.success(f"Offline replay finished: {len(results):,} records across {int(performance['batches'])} batches.")
            progress.progress(1.0, text="Replay complete")
        except Exception:
            logger.exception("Offline simulation failed.")
            st.error("Offline simulation could not be completed. Check the validation data and review the app logs for technical details.")

    if st.session_state.last_source == "Offline validation replay":
        show_result_summary()
        perf = st.session_state.last_performance or {}
        p1, p2, p3 = st.columns(3)
        p1.metric("Model processing time", f"{perf.get('model_processing_seconds', 0) * 1000:.2f} ms")
        p2.metric("Model processing throughput", f"{perf.get('records_per_second_model_processing', 0):,.0f} records/sec")
        p3.metric("Replay wall time", f"{perf.get('wall_seconds_including_optional_pacing', 0):.3f} sec")
        st.caption("Model-processing throughput excludes optional presentation pacing. Wall-clock throughput includes it.")

elif page == "Threat Alerts":
    st.subheader("Threat alert feed")
    st.warning("These are model-predicted attacks that require review. They are not confirmed security incidents and do not have calibrated severity levels.")
    st.caption("Alert and run history is stored only in this Streamlit session. Download important results before leaving; session history is not a permanent incident database.")
    alerts = st.session_state.alert_history
    if alerts.empty:
        st.info("No alerts yet. Run a manual/CSV analysis or offline simulation first.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Alerts in this session", f"{len(alerts):,}")
        c2.metric("Runs recorded", f"{len(st.session_state.run_history):,}")
        if "Predicted_Attack_Type" in alerts.columns:
            options = ["All"] + sorted(alerts["Predicted_Attack_Type"].astype(str).unique().tolist())
            selected = st.selectbox("Filter by predicted attack type", options)
            shown = alerts if selected == "All" else alerts.loc[alerts["Predicted_Attack_Type"].astype(str).eq(selected)]
        else:
            shown = alerts
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button("Download alert history CSV", alerts.to_csv(index=False).encode("utf-8"), "edge_ids_alert_history.csv", "text/csv", key="download_alert_history")
        if st.button("Clear alert history for this browser session", type="secondary"):
            st.session_state.alert_history = pd.DataFrame()
            st.rerun()
    if st.session_state.last_predictions is not None:
        st.caption(f"Latest run: {st.session_state.last_run_id} · {st.session_state.last_source}")

elif page == "Edge Performance":
    st.subheader("CPU inference and system performance")
    cpu = psutil.cpu_percent(interval=0.2)
    memory = psutil.virtual_memory()
    a, b, c, d = st.columns(4)
    a.metric("CPU utilization", f"{cpu:.1f}%")
    b.metric("System memory used", f"{memory.percent:.1f}%")
    c.metric("Available memory", f"{memory.available / (1024**3):.2f} GB")
    d.metric("Inference device", "CPU")
    performance = st.session_state.last_performance
    if performance:
        st.markdown("#### Latest measured inference")
        p1, p2, p3 = st.columns(3)
        p1.metric("Records processed", f"{int(performance.get('rows', 0)):,}")
        total_time = performance.get("total_seconds", performance.get("model_processing_seconds", 0.0))
        p2.metric("Model processing time", f"{total_time * 1000:.2f} ms")
        throughput = performance.get("records_per_second", performance.get("records_per_second_model_processing", 0.0))
        p3.metric("Model processing throughput", f"{throughput:,.0f} records/sec")
        st.json(performance)
        st.caption("These measurements apply to this local machine and the latest batch size. They are not a hardware-independent benchmark.")
    else:
        st.info("Run Traffic Analyzer or Live Simulation to measure inference time and throughput.")
    if st.session_state.run_history:
        history = pd.DataFrame(st.session_state.run_history)
        st.markdown("#### Run history for this browser session")
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button("Download run history CSV", history.to_csv(index=False).encode("utf-8"), "edge_ids_run_history.csv", "text/csv", key="download_run_history")

elif page == "About":
    st.subheader("Project information")
    st.markdown("#### End-to-end detection architecture")
    render_pipeline_diagram()
    st.markdown(
        """
        **Project:** Edge AI-Based Intrusion Detection for Smart Buildings  
        **Dataset:** Edge-IIoTset / `ML-EdgeIIoT-dataset.csv`  
        **Inference:** Local CPU using frozen scikit-learn estimators  
        **Interface:** Streamlit and Plotly  
        **Phase 6:** Manual single-record entry, CSV inference, dataset explorer and model evaluation  
        **Phase 7:** Batched offline validation replay, session alert feed, run exports and CPU telemetry

        **Frozen models**
        - Binary detection: Random Forest, excluding `tcp.len`.
        - Attack identification: Decision Tree, excluding `tcp.checksum`.
        - Shared preprocessing: saved Phase 3 `expanded36_normalized` preprocessor.

        **Limitations and safety**
        - Manual records are feature vectors, not captured packets; statistical defaults are drawn from the training split.
        - Offline simulation replays validation rows; it is not live packet capture or a network sensor.
        - Model-generated alerts require human review; they are not confirmed incidents.
        - Model score outputs are uncalibrated and must not be described as calibrated confidence.
        - The record-level split and earlier full-dataset exploratory audits limit claims about generalization to unseen devices, sessions and buildings.
        - No active scanning, attack execution or production building control is performed.
        """
    )
    freeze_manifest = ROOT / "models" / "final" / "phase5_model_freeze_manifest.json"
    if freeze_manifest.is_file():
        st.download_button("Download frozen-model manifest", freeze_manifest.read_bytes(), "phase5_model_freeze_manifest.json", "application/json", key="download_model_manifest")

st.divider()
st.caption("Edge AI IDS · Local educational prototype · Frozen Phase 5 models · Offline simulation only · No live packet capture")
