"""
FCM Project Visualizer — Main landing page.

Launch with:
    causal-mm-gui                  # after pip install causal-mm[gui]
    streamlit run gui/app.py       # from repo root (dev mode)
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

# Dev mode: add repo src/ to path; installed mode: causal_mm already importable
_src = _THIS_DIR.parent.parent / "src"
if not _src.is_dir():
    _src = _THIS_DIR.parent.parent.parent.parent / "src"
if _src.is_dir():
    sys.path.insert(0, str(_src))

# Ensure utils.py from the same directory is importable
sys.path.insert(0, str(_THIS_DIR))

import streamlit as st
import pandas as pd
from utils import (
    load_project_raw,
    set_project,
    get_project,
    list_available_models,
    PROJECT_ROOT,
)

st.set_page_config(
    page_title="FCM Visualizer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1a1d23 0%, #252830 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4F8BF9;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #999;
        margin-top: 4px;
    }
    /* Header gradient */
    .main-header {
        background: linear-gradient(90deg, #4F8BF9 0%, #CC5DE8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-top: -8px;
        margin-bottom: 24px;
    }
    /* Sidebar file list */
    div[data-testid="stSidebar"] .stRadio > label {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: File loader ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Load Project")

    tab_upload, tab_browse = st.tabs(["Upload", "Browse"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Drag & drop a .json file",
            type=["json"],
            help="Upload an fcm_project.json file",
        )
        if uploaded is not None:
            import json as _json
            try:
                raw_bytes = uploaded.getvalue()
                raw_dict = _json.loads(raw_bytes)
                # Write to temp then parse
                import tempfile, os
                tmp = Path(tempfile.mkdtemp()) / uploaded.name
                tmp.write_bytes(raw_bytes)
                project = load_project_raw(tmp)
                set_project(project, source=uploaded.name)
                st.success(f"Loaded **{uploaded.name}**")
            except Exception as e:
                st.error(f"Failed to parse: {e}")

    with tab_browse:
        models = list_available_models()
        if models:
            labels = [f.name for f in models]
            choice = st.selectbox("Select a model file", labels, index=None,
                                  placeholder="Choose a file...")
            if choice:
                fpath = models[labels.index(choice)]
                if st.button("Load selected", use_container_width=True):
                    try:
                        project = load_project_raw(fpath)
                        set_project(project, source=str(fpath.relative_to(PROJECT_ROOT)))
                        st.success(f"Loaded **{choice}**")
                    except Exception as e:
                        st.error(f"Failed: {e}")
        else:
            st.info("No model files found in data/models/ or examples/")

    st.divider()

    if get_project():
        st.markdown("### 📑 Pages")
        st.page_link("pages/1_Network.py", label="🕸️ Network Graph", use_container_width=True)
        st.page_link("pages/2_Weights.py", label="⚖️ Weights & Uncertainty", use_container_width=True)
        st.page_link("pages/3_TimeSeries.py", label="📈 Time Series", use_container_width=True)
        st.page_link("pages/4_Metrics.py", label="📊 Metrics Dashboard", use_container_width=True)
        st.page_link("pages/5_Editor.py", label="✏️ Editor", use_container_width=True)


# ── Main content ───────────────────────────────────────────────────────────
st.markdown('<p class="main-header">FCM Project Visualizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explore, analyze, and create Fuzzy Cognitive Map project files</p>',
            unsafe_allow_html=True)

project = get_project()

if project is None:
    # Welcome screen
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🕸️</div>
            <div class="metric-label">Interactive Network<br>Visualization</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">📊</div>
            <div class="metric-label">Weights, Uncertainty<br>& Metrics</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">✏️</div>
            <div class="metric-label">Create & Edit<br>FCM Projects</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info("👈 **Get started** by uploading or selecting a JSON file from the sidebar.")

else:
    # Project overview
    source = st.session_state.get("project_source", "Unknown")
    meta = project.get("meta", {})

    st.markdown(f"**Loaded:** `{source}`")

    # Summary cards
    n_concepts = len(project["concepts"])
    n_edges = len(project["edges"])
    n_estimates = len(project["estimates"])
    n_timepoints = len(project["timeseries_index"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_concepts}</div>
            <div class="metric-label">Concepts</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_edges}</div>
            <div class="metric-label">Edges</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_estimates}</div>
            <div class="metric-label">Estimates</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_timepoints}</div>
            <div class="metric-label">Time Points</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Meta info
    if meta:
        with st.expander("📋 Project Metadata", expanded=False):
            mcols = st.columns(2)
            with mcols[0]:
                st.markdown(f"**Project ID:** {meta.get('project_id', '—')}")
                st.markdown(f"**Creator:** {meta.get('creator', '—')}")
                st.markdown(f"**Status:** {meta.get('status', '—')}")
            with mcols[1]:
                st.markdown(f"**Started:** {meta.get('started_at', '—')}")
                st.markdown(f"**Completed:** {meta.get('completed_at', '—')}")
                tags = meta.get("tags", [])
                if tags:
                    st.markdown(f"**Tags:** {', '.join(tags)}")
            notes = meta.get("notes", "")
            if notes:
                st.markdown(f"**Notes:** {notes}")

    # Concepts table
    with st.expander("🧩 Concepts", expanded=True):
        concept_rows = []
        for c in project["concepts"]:
            color = c.get("color") or c.get("metadata", {}).get("color", "")
            swatch = f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:{color};margin-right:6px;vertical-align:middle;"></span>' if color else ""
            concept_rows.append({
                "ID": c["id"],
                "Label": c.get("label", ""),
                "Color": swatch + (color or "—"),
                "Unit": c.get("unit") or c.get("metadata", {}).get("unit", ""),
            })
        st.markdown(
            pd.DataFrame(concept_rows).to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )

    # Quick navigation
    st.markdown("---")
    st.markdown("### Explore this project")
    nav_cols = st.columns(5)
    with nav_cols[0]:
        st.page_link("pages/1_Network.py", label="🕸️ Network", use_container_width=True)
    with nav_cols[1]:
        st.page_link("pages/2_Weights.py", label="⚖️ Weights", use_container_width=True)
    with nav_cols[2]:
        st.page_link("pages/3_TimeSeries.py", label="📈 Series", use_container_width=True)
    with nav_cols[3]:
        st.page_link("pages/4_Metrics.py", label="📊 Metrics", use_container_width=True)
    with nav_cols[4]:
        st.page_link("pages/5_Editor.py", label="✏️ Editor", use_container_width=True)
