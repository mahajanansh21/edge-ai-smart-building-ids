"""Phase 9B presentation helpers for the Edge AI Smart Building IDS dashboard."""
from __future__ import annotations

from html import escape

import streamlit as st


PIPELINE_STEPS = (
    (
        "01",
        "Traffic input",
        "Manual feature vector or uploaded CSV. The model contract expects 36 source features.",
    ),
    (
        "02",
        "Saved preprocessing",
        "The frozen Phase 3 expanded36_normalized transformer converts 36 source features into 43 transformed features.",
    ),
    (
        "03",
        "Dual-model inference",
        "Random Forest: 42 inputs, excluding numeric__tcp.len. Decision Tree: 42 inputs, excluding numeric__tcp.checksum.",
    ),
    (
        "04",
        "Decision and outputs",
        "Binary Normal → Normal. Binary Attack → use the multiclass attack label. Results feed session alerts and downloadable reports.",
    ),
)


def apply_branding() -> None:
    """Apply a consistent dark security-operations theme using Streamlit CSS."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 88% 0%, rgba(22, 163, 174, 0.13), transparent 28%),
                linear-gradient(145deg, #071321 0%, #0a1726 52%, #0b1b2b 100%);
            color: #e8f1f8;
        }
        [data-testid="stAppViewContainer"] { background: transparent; }
        [data-testid="stHeader"] { background: rgba(7, 19, 33, 0.72); }
        .block-container { max-width: 1480px; padding-top: 1.45rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1a2a 0%, #091523 100%);
            border-right: 1px solid rgba(132, 166, 190, 0.16);
        }
        [data-testid="stSidebar"] section { border-color: rgba(132, 166, 190, 0.18); }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(18, 39, 59, 0.92), rgba(12, 29, 46, 0.92));
            border: 1px solid rgba(105, 159, 184, 0.22);
            border-radius: 15px;
            padding: 14px 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }
        [data-testid="stMetricLabel"] { color: #a9bdcc; }
        [data-testid="stMetricValue"] { color: #f2f8fc; font-weight: 700; }
        [data-testid="stDataFrame"], [data-testid="stFileUploader"] section {
            border: 1px solid rgba(105, 159, 184, 0.18);
            border-radius: 13px;
            overflow: hidden;
        }
        [data-testid="stAlert"] { border-radius: 12px; }
        div.stButton > button, div.stDownloadButton > button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 10px;
            font-weight: 650;
            min-height: 2.65rem;
        }
        div.stButton > button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            box-shadow: 0 5px 18px rgba(20, 184, 166, 0.18);
        }
        .edge-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.4rem;
            flex-wrap: wrap;
            padding: 1.45rem 1.55rem;
            margin: 0.2rem 0 0.65rem;
            border: 1px solid rgba(91, 176, 190, 0.28);
            border-radius: 22px;
            background: linear-gradient(112deg, rgba(16, 49, 66, 0.96), rgba(13, 31, 51, 0.96) 64%, rgba(17, 54, 67, 0.92));
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.20);
        }
        .edge-kicker, .edge-view-tag, .edge-pipeline-kicker {
            color: #77d7d0;
            font-size: 0.70rem;
            font-weight: 800;
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }
        .edge-hero h1 {
            color: #f4f9fc;
            font-size: clamp(1.55rem, 3vw, 2.25rem);
            font-weight: 800;
            line-height: 1.12;
            margin: 0.45rem 0 0.45rem;
        }
        .edge-hero h1 span { color: #63d4ca; }
        .edge-hero p { color: #b6cbd7; font-size: 0.94rem; margin: 0; max-width: 58rem; }
        .edge-status {
            min-width: 205px;
            padding: 0.85rem 1rem;
            border: 1px solid rgba(107, 209, 185, 0.3);
            border-radius: 14px;
            background: rgba(5, 22, 35, 0.52);
        }
        .edge-status-label { color: #a9c1cf; font-size: 0.68rem; letter-spacing: 0.10em; font-weight: 800; }
        .edge-status-value { color: #75e0bd; font-size: 0.90rem; font-weight: 800; margin-top: 0.28rem; }
        .edge-status-value.is-down { color: #ffbd89; }
        .edge-status-note { color: #9bb2c2; font-size: 0.75rem; margin-top: 0.16rem; }
        .edge-view-tag { color: #93acbd; margin: 0.9rem 0 1.25rem; }
        .edge-pipeline-head { margin: 1.05rem 0 0.75rem; }
        .edge-pipeline-head h3 { color: #edf6fb; font-size: 1.12rem; margin: 0.18rem 0 0.15rem; }
        .edge-pipeline-head p { color: #a9bdcc; font-size: 0.86rem; margin: 0; }
        .edge-step {
            min-height: 190px;
            height: 100%;
            padding: 1rem 0.95rem;
            border: 1px solid rgba(105, 159, 184, 0.22);
            border-radius: 15px;
            background: linear-gradient(155deg, rgba(20, 43, 63, 0.88), rgba(12, 29, 46, 0.95));
        }
        .edge-step-number { color: #64d5cb; font-size: 0.72rem; font-weight: 850; letter-spacing: 0.12em; }
        .edge-step-title { color: #f0f6fa; font-size: 0.98rem; font-weight: 800; margin: 0.5rem 0 0.48rem; }
        .edge-step-detail { color: #b6c8d4; font-size: 0.80rem; line-height: 1.48; }
        .edge-guide-card {
            height: 100%;
            padding: 0.95rem 1rem;
            border: 1px solid rgba(105, 159, 184, 0.19);
            border-radius: 14px;
            background: rgba(14, 34, 52, 0.78);
        }
        .edge-guide-title { color: #edf6fb; font-size: 0.92rem; font-weight: 800; margin-bottom: 0.35rem; }
        .edge-guide-body { color: #afc2cf; font-size: 0.81rem; line-height: 1.48; }
        @media (max-width: 800px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .edge-hero { padding: 1.05rem; border-radius: 16px; }
            .edge-step { min-height: 150px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_banner(page: str, *, model_loaded: bool) -> None:
    """Render the branded application heading and current page indicator."""
    status_text = "FROZEN MODEL BUNDLE READY" if model_loaded else "MODEL BUNDLE UNAVAILABLE"
    status_class = "edge-status-value" if model_loaded else "edge-status-value is-down"
    page_label = escape(str(page).upper())
    st.markdown(
        f"""
        <section class="edge-hero">
          <div>
            <div class="edge-kicker">EDGE AI · SMART BUILDING SECURITY</div>
            <h1>Intrusion Detection <span>Operations</span></h1>
            <p>Two-stage machine-learning analysis for IoT/IIoT traffic, using frozen CPU models and saved preprocessing.</p>
          </div>
          <div class="edge-status">
            <div class="edge-status-label">RUNTIME STATUS</div>
            <div class="{status_class}">{status_text}</div>
            <div class="edge-status-note">CPU inference · Educational prototype</div>
          </div>
        </section>
        <div class="edge-view-tag">WORKSPACE / {page_label}</div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_diagram() -> None:
    """Show the verified input-to-detection workflow as four responsive cards."""
    st.markdown(
        """
        <div class="edge-pipeline-head">
          <div class="edge-pipeline-kicker">DETECTION WORKFLOW</div>
          <h3>From traffic features to a model-generated detection</h3>
          <p>The binary model gates the final label; the multiclass model supplies the attack category when an attack is predicted.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(4, gap="small")
    for column, (number, title, detail) in zip(columns, PIPELINE_STEPS):
        safe_title = escape(title)
        safe_detail = escape(detail).replace("\n", "<br>")
        with column:
            st.markdown(
                f"""
                <div class="edge-step">
                  <div class="edge-step-number">STEP {number}</div>
                  <div class="edge-step-title">{safe_title}</div>
                  <div class="edge-step-detail">{safe_detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_input_quick_guide(
    *, source_feature_count: int, max_upload_mb: int, max_rows: int
) -> None:
    """Display concise, accurate guidance for the analyzer's two input paths."""
    st.markdown("#### Before you run an analysis")
    cards = (
        (
            "Manual feature vector",
            f"Enter one record using the {source_feature_count} required source features. Training-derived median/mode values are starting defaults; edit them for an experiment.",
        ),
        (
            "CSV upload",
            f"Provide all {source_feature_count} source-feature columns. Maximum file size: {max_upload_mb} MB. Maximum records per run: {max_rows:,} data rows (header excluded).",
        ),
        (
            "Interpretation",
            "Inputs are feature vectors, not captured packets. Predictions and alerts are model outputs requiring human review, not confirmed incidents.",
        ),
    )
    columns = st.columns(3, gap="small")
    for column, (title, body) in zip(columns, cards):
        with column:
            st.markdown(
                f'<div class="edge-guide-card"><div class="edge-guide-title">{escape(title)}</div>'
                f'<div class="edge-guide-body">{escape(body)}</div></div>',
                unsafe_allow_html=True,
            )
