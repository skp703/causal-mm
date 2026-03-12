"""
Time Series — Interactive visualization of concept time-series data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils import get_project, build_timeseries_df, get_id_to_label, get_concept_color

st.set_page_config(page_title="Time Series", page_icon="📈", layout="wide")
st.markdown('<h1 style="margin-bottom:0;">📈 Time Series</h1>', unsafe_allow_html=True)

project = get_project()
if project is None:
    st.warning("No project loaded. Go to the main page to load a file.")
    st.stop()

ts_df = build_timeseries_df(project)
if ts_df.empty:
    st.info("No time-series data in this project.")
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Time Series Settings")

    all_concepts = list(ts_df.columns)
    selected = st.multiselect(
        "Select concepts",
        all_concepts,
        default=all_concepts,
        help="Choose which concepts to display",
    )

    normalize = st.toggle("Z-score normalize", value=False,
                          help="Standardize each series to zero mean, unit variance")

    show_points = st.toggle("Show data points", value=True)
    show_area = st.toggle("Fill area under curves", value=False)

# ── Prepare data ──────────────────────────────────────────────────────────
if not selected:
    st.info("Select at least one concept from the sidebar.")
    st.stop()

plot_df = ts_df[selected].copy()

if normalize:
    for col in plot_df.columns:
        mean = plot_df[col].mean()
        std = plot_df[col].std()
        if std > 0:
            plot_df[col] = (plot_df[col] - mean) / std
        else:
            plot_df[col] = 0.0

# ── Build concept colors map ─────────────────────────────────────────────
id_to_label = get_id_to_label(project["concepts"])
label_to_color = {}
for i, c in enumerate(project["concepts"]):
    label = c.get("label", c["id"])
    label_to_color[label] = get_concept_color(c, i)

# ── Main chart ────────────────────────────────────────────────────────────
st.markdown("### Concept Trajectories")

fig = go.Figure()

for col in plot_df.columns:
    color = label_to_color.get(col, "#4F8BF9")
    mode = "lines+markers" if show_points else "lines"
    fill = "tozeroy" if show_area else None

    fig.add_trace(go.Scatter(
        x=plot_df.index.astype(str),
        y=plot_df[col],
        name=col,
        mode=mode,
        line=dict(color=color, width=2.5),
        marker=dict(size=5, color=color),
        fill=fill,
        fillcolor=color.replace(")", ", 0.1)").replace("rgb", "rgba") if fill and "rgb" in color else None,
        hovertemplate=f"<b>{col}</b><br>Time: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
    ))

yaxis_title = "Z-score" if normalize else "Value"
fig.update_layout(
    height=500,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        title="Time",
        gridcolor="rgba(100,100,100,0.15)",
        tickfont=dict(color="#ccc"),
        titlefont=dict(color="#aaa"),
    ),
    yaxis=dict(
        title=yaxis_title,
        gridcolor="rgba(100,100,100,0.15)",
        tickfont=dict(color="#ccc"),
        titlefont=dict(color="#aaa"),
    ),
    legend=dict(
        font=dict(color="#ccc", size=11),
        bgcolor="rgba(26,29,35,0.8)",
        bordercolor="rgba(100,100,100,0.3)",
        borderwidth=1,
    ),
    margin=dict(l=20, r=20, t=20, b=40),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1a1d23", font_size=12),
)

st.plotly_chart(fig, use_container_width=True)

# ── Summary statistics ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Summary Statistics")

stats = ts_df[selected].describe().T
stats.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
st.dataframe(
    stats.style.format(precision=3).background_gradient(
        cmap="Blues", subset=["Mean", "Std"]
    ),
    use_container_width=True,
)

# ── Correlation matrix ─────────────────────────────────────────────────────
if len(selected) > 1:
    st.markdown("---")
    st.markdown("### Pairwise Correlation")

    corr = ts_df[selected].corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale=[
            [0.0, "#FF6B6B"],
            [0.5, "#1a1d23"],
            [1.0, "#4F8BF9"],
        ],
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="%{y} × %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(title="r", tickfont=dict(color="#aaa"), titlefont=dict(color="#aaa")),
    ))
    fig_corr.update_layout(
        height=max(350, len(selected) * 45),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-45, tickfont=dict(color="#ccc")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#ccc")),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ── Individual sparklines ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Individual Series")

cols_per_row = 3
for i in range(0, len(selected), cols_per_row):
    row_cols = st.columns(cols_per_row)
    for j, col_name in enumerate(selected[i:i+cols_per_row]):
        with row_cols[j]:
            color = label_to_color.get(col_name, "#4F8BF9")
            spark = go.Figure(go.Scatter(
                x=ts_df.index.astype(str),
                y=ts_df[col_name],
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba(79,139,249,0.1)",
            ))
            spark.update_layout(
                height=150,
                margin=dict(l=5, r=5, t=25, b=5),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                title=dict(text=col_name, font=dict(size=12, color="#ccc"), x=0.5),
                showlegend=False,
            )
            st.plotly_chart(spark, use_container_width=True, key=f"spark_{col_name}")
