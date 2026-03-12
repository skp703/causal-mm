"""
Network Graph — Interactive visualization of the FCM causal network.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils import get_project, get_id_to_label, get_weight_for_edge, get_concept_color, build_adjacency_df

st.set_page_config(page_title="Network Graph", page_icon="🕸️", layout="wide")
st.markdown('<h1 style="margin-bottom:0;">🕸️ Network Graph</h1>', unsafe_allow_html=True)

project = get_project()
if project is None:
    st.warning("No project loaded. Go to the main page to load a file.")
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Graph Settings")

    has_estimates = bool(project["estimates"])
    use_estimated = st.toggle(
        "Use estimated weights",
        value=has_estimates,
        disabled=not has_estimates,
        help="Toggle between stakeholder weights and DML-estimated scaled weights",
    )

    layout_choice = st.selectbox(
        "Layout",
        ["Circular", "Spring (force-directed)", "Shell", "Kamada-Kawai"],
        index=0,
    )

    edge_scale = st.slider("Edge thickness scale", 1.0, 10.0, 4.0, 0.5)
    node_scale = st.slider("Node size scale", 10.0, 60.0, 30.0, 5.0)
    show_edge_labels = st.toggle("Show edge weights", value=True)
    show_self_loops = st.toggle("Show self-loops", value=False)

# ── Build graph with networkx ────────────────────────────────────────────
import networkx as nx

id_to_label = get_id_to_label(project["concepts"])
G = nx.DiGraph()

# Add nodes
for i, c in enumerate(project["concepts"]):
    G.add_node(
        c["id"],
        label=c.get("label", c["id"]),
        color=get_concept_color(c, i),
    )

# Add edges
for edge in project["edges"]:
    if not show_self_loops and edge["source"] == edge["target"]:
        continue
    w = get_weight_for_edge(edge, project["estimates"], use_estimated)
    G.add_edge(edge["source"], edge["target"], weight=w)

# Layout
layout_funcs = {
    "Circular": nx.circular_layout,
    "Spring (force-directed)": lambda g: nx.spring_layout(g, seed=42, k=2.5),
    "Shell": nx.shell_layout,
    "Kamada-Kawai": nx.kamada_kawai_layout,
}
pos = layout_funcs[layout_choice](G)

# ── Plotly interactive graph ─────────────────────────────────────────────

def make_graph_figure(G, pos, edge_scale, node_scale, show_edge_labels):
    fig = go.Figure()

    # Draw edges as annotations (arrows)
    for src, tgt, data in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        w = data.get("weight", 0)
        abs_w = abs(w)

        # Edge color: green positive, red negative, gray zero
        if w > 0:
            color = "rgba(81, 207, 102, 0.7)"
        elif w < 0:
            color = "rgba(255, 107, 107, 0.7)"
        else:
            color = "rgba(150, 150, 150, 0.4)"

        width = max(1, abs_w * edge_scale)

        # Shorten line to not overlap nodes
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            offset = 0.06
            ux, uy = dx / length, dy / length
            x0a = x0 + ux * offset
            y0a = y0 + uy * offset
            x1a = x1 - ux * offset
            y1a = y1 - uy * offset
        else:
            x0a, y0a, x1a, y1a = x0, y0, x1, y1

        # Edge line
        fig.add_trace(go.Scatter(
            x=[x0a, x1a, None], y=[y0a, y1a, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="text",
            text=f"{id_to_label.get(src, src)} → {id_to_label.get(tgt, tgt)}<br>Weight: {w:.4f}",
            showlegend=False,
        ))

        # Arrowhead annotation
        fig.add_annotation(
            ax=x0a, ay=y0a,
            x=x1a, y=y1a,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=max(1, width * 0.6),
            arrowcolor=color,
            opacity=0.8,
        )

        # Edge label
        if show_edge_labels and abs_w > 0.001:
            mx, my = (x0a + x1a) / 2, (y0a + y1a) / 2
            fig.add_annotation(
                x=mx, y=my,
                text=f"{w:.2f}",
                showarrow=False,
                font=dict(size=10, color="#ccc"),
                bgcolor="rgba(14,17,23,0.7)",
                borderpad=2,
            )

    # Draw nodes
    node_x, node_y, node_colors, node_text, node_labels = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(G.nodes[node].get("color", "#4F8BF9"))
        label = G.nodes[node].get("label", node)
        node_labels.append(label)

        # Hover text
        in_edges = [(s, d.get("weight", 0)) for s, _, d in G.in_edges(node, data=True)]
        out_edges = [(t, d.get("weight", 0)) for _, t, d in G.out_edges(node, data=True)]
        hover = f"<b>{label}</b> (ID: {node})<br>"
        hover += f"In-degree: {len(in_edges)} | Out-degree: {len(out_edges)}<br>"
        if in_edges:
            hover += "<br><b>Incoming:</b><br>"
            for s, w in in_edges:
                hover += f"  {id_to_label.get(s, s)}: {w:.3f}<br>"
        if out_edges:
            hover += "<b>Outgoing:</b><br>"
            for t, w in out_edges:
                hover += f"  {id_to_label.get(t, t)}: {w:.3f}<br>"
        node_text.append(hover)

    # Size by centrality
    sizes = []
    for node in G.nodes():
        deg = G.in_degree(node, weight="weight") + G.out_degree(node, weight="weight")
        sizes.append(max(node_scale, abs(deg) * node_scale * 0.8 + node_scale))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=node_colors,
            line=dict(width=2, color="#222"),
            opacity=0.95,
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=12, color="#eee"),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
        height=600,
        hoverlabel=dict(bgcolor="#1a1d23", font_size=12),
    )
    return fig


fig = make_graph_figure(G, pos, edge_scale, node_scale, show_edge_labels)
st.plotly_chart(fig, use_container_width=True)

# ── Legend ──────────────────────────────────────────────────────────────────
leg_cols = st.columns(4)
with leg_cols[0]:
    st.markdown("🟢 **Positive** weight")
with leg_cols[1]:
    st.markdown("🔴 **Negative** weight")
with leg_cols[2]:
    st.markdown(f"**Nodes:** {len(G.nodes())}")
with leg_cols[3]:
    st.markdown(f"**Edges:** {len(G.edges())}")

# ── Adjacency heatmap ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Adjacency Matrix Heatmap")

adj_df = build_adjacency_df(project, use_estimated)

fig_heat = go.Figure(data=go.Heatmap(
    z=adj_df.values,
    x=list(adj_df.columns),
    y=list(adj_df.index),
    colorscale=[
        [0.0, "#FF6B6B"],
        [0.5, "#1a1d23"],
        [1.0, "#51CF66"],
    ],
    zmid=0,
    text=np.round(adj_df.values, 3),
    texttemplate="%{text}",
    textfont=dict(size=10),
    hovertemplate="From: %{y}<br>To: %{x}<br>Weight: %{z:.4f}<extra></extra>",
    colorbar=dict(title="Weight", tickfont=dict(color="#aaa"), titlefont=dict(color="#aaa")),
))
fig_heat.update_layout(
    height=max(400, len(adj_df) * 45),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(tickangle=-45, tickfont=dict(color="#ccc")),
    yaxis=dict(autorange="reversed", tickfont=dict(color="#ccc")),
    margin=dict(l=20, r=20, t=20, b=20),
)
st.plotly_chart(fig_heat, use_container_width=True)
