"""
Metrics Dashboard — Graph complexity and concept centrality metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import (
    get_project, compute_metrics_standalone, compute_centrality_standalone,
    build_adjacency_df, get_id_to_label,
)

st.set_page_config(page_title="Metrics Dashboard", page_icon="📊", layout="wide")
st.markdown('<h1 style="margin-bottom:0;">📊 Metrics Dashboard</h1>', unsafe_allow_html=True)

project = get_project()
if project is None:
    st.warning("No project loaded. Go to the main page to load a file.")
    st.stop()

# ── Custom card CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1a1d23, #252830);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 18px 14px;
        text-align: center;
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #4F8BF9; }
    .kpi-label { font-size: 0.82rem; color: #999; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Graph complexity metrics ───────────────────────────────────────────────
st.markdown("### Graph Complexity")

metrics = compute_metrics_standalone(project)

if not metrics:
    st.info("Not enough data to compute metrics.")
    st.stop()

# KPI cards row
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi_items = [
    (k1, metrics["num_nodes"], "Nodes (N)"),
    (k2, metrics["num_connections"], "Connections (C)"),
    (k3, f"{metrics['density']:.3f}", "Density"),
    (k4, metrics["num_transmitters"], "Transmitters"),
    (k5, metrics["num_receivers"], "Receivers"),
    (k6, metrics["num_ordinary"], "Ordinary"),
]
for col, val, label in kpi_items:
    with col:
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown("")

# Additional metrics
if metrics.get("prop_significant") is not None or metrics.get("avg_sign_stability") is not None:
    st.markdown("#### Estimation Quality")
    eq1, eq2 = st.columns(2)
    with eq1:
        ps = metrics.get("prop_significant")
        val = f"{ps:.1%}" if ps is not None else "—"
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">Proportion Significant</div>
        </div>
        ''', unsafe_allow_html=True)
    with eq2:
        ss = metrics.get("avg_sign_stability")
        val = f"{ss:.1%}" if ss is not None else "—"
        color = "#51CF66" if ss and ss >= 0.9 else "#FCC419" if ss and ss >= 0.7 else "#FF6B6B" if ss else "#4F8BF9"
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-value" style="color:{color}">{val}</div>
            <div class="kpi-label">Avg Sign Stability</div>
        </div>
        ''', unsafe_allow_html=True)

# ── Complexity detail table ─────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Full Metrics Table", expanded=False):
    metrics_df = pd.DataFrame([
        {"Metric": "Number of Nodes (N)", "Value": metrics["num_nodes"]},
        {"Metric": "Number of Connections (C)", "Value": metrics["num_connections"]},
        {"Metric": "Density (C / N(N-1))", "Value": f"{metrics['density']:.4f}"},
        {"Metric": "Transmitters (out only)", "Value": metrics["num_transmitters"]},
        {"Metric": "Receivers (in only)", "Value": metrics["num_receivers"]},
        {"Metric": "Ordinary (both in & out)", "Value": metrics["num_ordinary"]},
        {"Metric": "Connections per Node (C/N)", "Value": f"{metrics['num_connections']/metrics['num_nodes']:.2f}" if metrics['num_nodes'] > 0 else "—"},
    ])
    if metrics.get("prop_significant") is not None:
        metrics_df = pd.concat([metrics_df, pd.DataFrame([
            {"Metric": "Proportion Significant", "Value": f"{metrics['prop_significant']:.3f}"},
        ])], ignore_index=True)
    if metrics.get("avg_sign_stability") is not None:
        metrics_df = pd.concat([metrics_df, pd.DataFrame([
            {"Metric": "Avg Sign Stability", "Value": f"{metrics['avg_sign_stability']:.3f}"},
        ])], ignore_index=True)

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ── Concept centrality ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Concept Centrality")

centrality_df = compute_centrality_standalone(project)

if centrality_df.empty:
    st.info("No data for centrality computation.")
else:
    # Bar chart
    cent_sorted = centrality_df.sort_values("Centrality", ascending=True)

    fig_cent = go.Figure()
    fig_cent.add_trace(go.Bar(
        y=cent_sorted.index,
        x=cent_sorted["Out-degree"],
        name="Out-degree",
        orientation="h",
        marker_color="rgba(79, 139, 249, 0.7)",
    ))
    fig_cent.add_trace(go.Bar(
        y=cent_sorted.index,
        x=cent_sorted["In-degree"],
        name="In-degree",
        orientation="h",
        marker_color="rgba(204, 93, 232, 0.7)",
    ))
    fig_cent.update_layout(
        barmode="stack",
        height=max(350, len(cent_sorted) * 35 + 60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Weighted Degree (|w|)",
            gridcolor="rgba(100,100,100,0.2)",
            tickfont=dict(color="#ccc"),
            titlefont=dict(color="#aaa"),
        ),
        yaxis=dict(tickfont=dict(color="#ccc", size=11)),
        legend=dict(font=dict(color="#ccc")),
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_cent, use_container_width=True)

    # Centrality table
    with st.expander("📋 Centrality Table", expanded=False):
        st.dataframe(
            centrality_df.sort_values("Centrality", ascending=False)
            .style.format(precision=4)
            .background_gradient(cmap="Blues", subset=["Centrality"]),
            use_container_width=True,
        )

# ── Node role distribution ────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Node Role Distribution")

roles = {
    "Transmitters": metrics["num_transmitters"],
    "Receivers": metrics["num_receivers"],
    "Ordinary": metrics["num_ordinary"],
}
roles = {k: v for k, v in roles.items() if v > 0}

if roles:
    fig_pie = go.Figure(go.Pie(
        labels=list(roles.keys()),
        values=list(roles.values()),
        hole=0.45,
        marker=dict(
            colors=["#4F8BF9", "#CC5DE8", "#51CF66"],
            line=dict(color="#1a1d23", width=2),
        ),
        textfont=dict(color="#eee", size=13),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig_pie.update_layout(
        height=350,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#ccc")),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("ℹ️ What are node roles?"):
        st.markdown("""
        - **Transmitters**: Concepts with only outgoing edges (drivers / forcing variables)
        - **Receivers**: Concepts with only incoming edges (outcomes / dependent variables)
        - **Ordinary**: Concepts with both incoming and outgoing edges (mediators)
        """)
