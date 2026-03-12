"""
Weights & Uncertainty — Edge weight tables and forest plots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import get_project, build_edges_table, get_id_to_label

st.set_page_config(page_title="Weights & Uncertainty", page_icon="⚖️", layout="wide")
st.markdown('<h1 style="margin-bottom:0;">⚖️ Weights & Uncertainty</h1>', unsafe_allow_html=True)

project = get_project()
if project is None:
    st.warning("No project loaded. Go to the main page to load a file.")
    st.stop()

id_to_label = get_id_to_label(project["concepts"])

# ── Edge weights table ────────────────────────────────────────────────────
st.markdown("### Edge Weight Summary")

edges_df = build_edges_table(project)
if edges_df.empty:
    st.info("No edges defined in this project.")
    st.stop()

# Style the table
def color_weight(val):
    if pd.isna(val) or val is None:
        return ""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v > 0:
        return "color: #51CF66"
    elif v < 0:
        return "color: #FF6B6B"
    return "color: #888"

styled = edges_df.style.applymap(
    color_weight,
    subset=["Stakeholder Weight", "Scaled Weight", "Tau (raw)"],
).format(precision=4, na_rep="—")

st.dataframe(styled, use_container_width=True, height=min(600, 40 + len(edges_df) * 35))

# ── Forest plot (confidence intervals) ────────────────────────────────────
estimates_with_ci = []
for edge in project["edges"]:
    key = f"{edge['source']}->{edge['target']}"
    est = project["estimates"].get(key, {})
    ci_lo = est.get("ci_low")
    ci_hi = est.get("ci_high")
    tau = est.get("tau_raw")
    scaled = est.get("scaled_weight")
    if tau is not None:
        src_label = id_to_label.get(edge["source"], edge["source"])
        tgt_label = id_to_label.get(edge["target"], edge["target"])
        estimates_with_ci.append({
            "edge": f"{src_label} → {tgt_label}",
            "tau_raw": tau,
            "scaled_weight": scaled,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "sign_stability": est.get("sign_stability"),
        })

if estimates_with_ci:
    st.markdown("---")
    st.markdown("### Forest Plot — Estimated Effects")
    st.caption("Horizontal bars show point estimates with 95% confidence intervals. "
               "Red dashed line = zero (no effect).")

    est_df = pd.DataFrame(estimates_with_ci)
    est_df = est_df.sort_values("tau_raw", ascending=True).reset_index(drop=True)

    fig = go.Figure()

    # CI whiskers
    for i, row in est_df.iterrows():
        color = "#51CF66" if row["tau_raw"] > 0 else "#FF6B6B"
        ci_lo = row["ci_low"] if row["ci_low"] is not None else row["tau_raw"]
        ci_hi = row["ci_high"] if row["ci_high"] is not None else row["tau_raw"]

        # Determine if significant (CI excludes zero)
        significant = (ci_lo > 0 or ci_hi < 0) if (row["ci_low"] is not None) else False
        marker_symbol = "diamond" if significant else "circle"

        # CI line
        fig.add_trace(go.Scatter(
            x=[ci_lo, ci_hi],
            y=[row["edge"], row["edge"]],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Point estimate
        hover_text = (
            f"<b>{row['edge']}</b><br>"
            f"τ_raw: {row['tau_raw']:.4f}<br>"
            f"Scaled: {row['scaled_weight']:.4f}<br>" if row["scaled_weight"] is not None else ""
        )
        if row["ci_low"] is not None:
            hover_text += f"95% CI: [{row['ci_low']:.4f}, {row['ci_high']:.4f}]<br>"
        if row["sign_stability"] is not None:
            hover_text += f"Sign stability: {row['sign_stability']:.2%}"

        fig.add_trace(go.Scatter(
            x=[row["tau_raw"]],
            y=[row["edge"]],
            mode="markers",
            marker=dict(size=10, color=color, symbol=marker_symbol,
                       line=dict(width=1, color="#fff")),
            showlegend=False,
            hovertext=hover_text,
            hoverinfo="text",
        ))

    # Zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="#FF6B6B", opacity=0.5)

    fig.update_layout(
        height=max(350, len(est_df) * 40 + 80),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Causal Effect (τ_raw)",
            gridcolor="rgba(100,100,100,0.2)",
            zeroline=True,
            zerolinecolor="rgba(255,107,107,0.3)",
            tickfont=dict(color="#ccc"),
            titlefont=dict(color="#aaa"),
        ),
        yaxis=dict(
            tickfont=dict(color="#ccc", size=11),
            gridcolor="rgba(100,100,100,0.1)",
        ),
        margin=dict(l=20, r=40, t=20, b=40),
        hoverlabel=dict(bgcolor="#1a1d23", font_size=12),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend for markers
    lcol1, lcol2 = st.columns(2)
    with lcol1:
        st.markdown("◆ **Diamond** = Significant (CI excludes zero)")
    with lcol2:
        st.markdown("● **Circle** = Not significant")

    # ── Sign stability bar chart ───────────────────────────────────────────
    stab_data = est_df[est_df["sign_stability"].notna()]
    if not stab_data.empty:
        st.markdown("---")
        st.markdown("### Sign Stability")
        st.caption("Fraction of bootstrap samples where the estimated sign matches "
                   "the point estimate. Higher = more robust.")

        colors = ["#51CF66" if s >= 0.9 else "#FCC419" if s >= 0.7 else "#FF6B6B"
                  for s in stab_data["sign_stability"]]

        fig_stab = go.Figure(go.Bar(
            x=stab_data["sign_stability"],
            y=stab_data["edge"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}<br>Sign Stability: %{x:.2%}<extra></extra>",
        ))
        fig_stab.add_vline(x=0.9, line_dash="dot", line_color="#51CF66", opacity=0.5,
                           annotation_text="90%", annotation_position="top right")
        fig_stab.update_layout(
            height=max(300, len(stab_data) * 35 + 60),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Sign Stability", range=[0, 1.05],
                       gridcolor="rgba(100,100,100,0.2)",
                       tickfont=dict(color="#ccc"), titlefont=dict(color="#aaa"),
                       tickformat=".0%"),
            yaxis=dict(tickfont=dict(color="#ccc", size=11)),
            margin=dict(l=20, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_stab, use_container_width=True)

else:
    st.markdown("---")
    st.info("No DML estimates available. Run `causal-mm-run` on this project to generate estimates with uncertainty.")

# ── Stakeholder vs. Estimated comparison ──────────────────────────────────
if estimates_with_ci:
    st.markdown("---")
    st.markdown("### Stakeholder vs. Estimated Weights")

    compare_rows = []
    for edge in project["edges"]:
        key = f"{edge['source']}->{edge['target']}"
        est = project["estimates"].get(key, {})
        src_l = id_to_label.get(edge["source"], edge["source"])
        tgt_l = id_to_label.get(edge["target"], edge["target"])
        sw = edge.get("stakeholder_weight")
        ew = est.get("scaled_weight")
        if sw is not None and ew is not None:
            compare_rows.append({
                "edge": f"{src_l} → {tgt_l}",
                "stakeholder": sw,
                "estimated": ew,
            })

    if compare_rows:
        cdf = pd.DataFrame(compare_rows)

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="Stakeholder",
            x=cdf["edge"], y=cdf["stakeholder"],
            marker_color="rgba(79, 139, 249, 0.7)",
        ))
        fig_cmp.add_trace(go.Bar(
            name="Estimated",
            x=cdf["edge"], y=cdf["estimated"],
            marker_color="rgba(204, 93, 232, 0.7)",
        ))
        fig_cmp.update_layout(
            barmode="group",
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-45, tickfont=dict(color="#ccc")),
            yaxis=dict(title="Weight", gridcolor="rgba(100,100,100,0.2)",
                       tickfont=dict(color="#ccc"), titlefont=dict(color="#aaa")),
            legend=dict(font=dict(color="#ccc")),
            margin=dict(l=20, r=20, t=20, b=80),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
