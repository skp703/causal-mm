"""
Utility functions for loading FCM project JSON files and building graph data
for the Streamlit GUI. Works both with causal_mm installed and standalone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project root — works from gui/ (dev) or src/causal_mm/gui/ (installed)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent

# Heuristic: if we're inside src/causal_mm/gui, go up 3 levels; else 1
if (_THIS_DIR.parent.parent / "src").is_dir():
    PROJECT_ROOT = _THIS_DIR.parent.parent  # repo root (dev layout: gui/)
elif (_THIS_DIR.parent / "io.py").exists():
    PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent  # src/causal_mm/gui → repo root
else:
    PROJECT_ROOT = _THIS_DIR.parent  # fallback

DATA_MODELS_DIR = PROJECT_ROOT / "data" / "models"
EXAMPLES_DIR = PROJECT_ROOT / "examples"


# ---------------------------------------------------------------------------
# Try importing causal_mm; fall back to standalone parsing
# ---------------------------------------------------------------------------
try:
    from causal_mm.io import load_project as _cm_load
    from causal_mm.fcm import FCMGraph, Concept, Edge, EdgeEstimate
    from causal_mm.metrics import (
        graph_complexity_metrics,
        concept_centrality_metrics,
        proportion_edges_significant,
        average_sign_stability,
        generalized_distance_ratio,
        weight_distance,
    )
    HAS_CAUSAL_MM = True
except ImportError:
    HAS_CAUSAL_MM = False


# ---------------------------------------------------------------------------
# Raw JSON loader (always available, no causal_mm dependency)
# ---------------------------------------------------------------------------

def load_json_raw(path: Path) -> Dict[str, Any]:
    """Load a fcm_project.json and return the raw dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_project_raw(path: Path) -> Dict[str, Any]:
    """
    Parse a fcm_project.json into a normalized dict with keys:
    meta, concepts, edges, timeseries_index, timeseries_data, estimates, settings.
    """
    raw = load_json_raw(path)
    model = raw.get("model", {})

    concepts = []
    for c in model.get("concepts", []):
        concepts.append({
            "id": str(c["id"]),
            "label": c.get("label", str(c["id"])),
            "color": c.get("metadata", {}).get("color", None),
            "unit": c.get("metadata", {}).get("unit", ""),
            "description": c.get("metadata", {}).get("description", ""),
            "metadata": c.get("metadata", {}),
        })

    edges = []
    for e in model.get("edges", []):
        edges.append({
            "source": str(e["source"]),
            "target": str(e["target"]),
            "stakeholder_weight": e.get("stakeholder_weight"),
            "metadata": e.get("metadata", {}),
        })

    ts_raw = raw.get("timeseries", {})
    ts_index = ts_raw.get("index", [])
    ts_data = {str(k): v for k, v in ts_raw.get("data", {}).items()}

    estimates = {}
    for key, val in raw.get("estimates", {}).items():
        if "->" in key:
            estimates[key] = val

    return {
        "meta": raw.get("meta", {}),
        "concepts": concepts,
        "edges": edges,
        "timeseries_index": ts_index,
        "timeseries_data": ts_data,
        "estimates": estimates,
        "settings": raw.get("settings", {}),
        "results": raw.get("results", {}),
    }


# ---------------------------------------------------------------------------
# Graph helper functions (standalone, no causal_mm needed)
# ---------------------------------------------------------------------------

def get_id_to_label(concepts: List[Dict]) -> Dict[str, str]:
    """Map concept id -> label."""
    return {c["id"]: c.get("label", c["id"]) for c in concepts}


def get_weight_for_edge(edge: Dict, estimates: Dict, use_estimated: bool = False) -> float:
    """Return the weight to display for an edge."""
    key = f"{edge['source']}->{edge['target']}"
    if use_estimated and key in estimates:
        est = estimates[key]
        sw = est.get("scaled_weight")
        if sw is not None:
            return float(sw)
    w = edge.get("stakeholder_weight")
    return float(w) if w is not None else 0.0


def build_adjacency_df(project: Dict, use_estimated: bool = False) -> pd.DataFrame:
    """Build a labeled adjacency matrix DataFrame."""
    id_to_label = get_id_to_label(project["concepts"])
    ids = [c["id"] for c in project["concepts"]]
    labels = [id_to_label[i] for i in ids]
    n = len(ids)
    idx_map = {cid: i for i, cid in enumerate(ids)}
    A = np.zeros((n, n))

    for edge in project["edges"]:
        s, t = edge["source"], edge["target"]
        if s in idx_map and t in idx_map:
            A[idx_map[s], idx_map[t]] = get_weight_for_edge(
                edge, project["estimates"], use_estimated
            )

    return pd.DataFrame(A, index=labels, columns=labels)


def build_edges_table(project: Dict) -> pd.DataFrame:
    """Build a table of all edges with their properties."""
    id_to_label = get_id_to_label(project["concepts"])
    rows = []
    for edge in project["edges"]:
        src_label = id_to_label.get(edge["source"], edge["source"])
        tgt_label = id_to_label.get(edge["target"], edge["target"])
        key = f"{edge['source']}->{edge['target']}"
        est = project["estimates"].get(key, {})

        rows.append({
            "Source": src_label,
            "Target": tgt_label,
            "Stakeholder Weight": edge.get("stakeholder_weight"),
            "Scaled Weight": est.get("scaled_weight"),
            "Tau (raw)": est.get("tau_raw"),
            "Tau SE": est.get("tau_se"),
            "CI Low": est.get("ci_low"),
            "CI High": est.get("ci_high"),
            "Sign Stability": est.get("sign_stability"),
            "Status": est.get("status", "—"),
        })

    return pd.DataFrame(rows)


def build_timeseries_df(project: Dict) -> pd.DataFrame:
    """Build a DataFrame of time-series data with labeled columns."""
    id_to_label = get_id_to_label(project["concepts"])
    ts = project["timeseries_data"]
    if not ts:
        return pd.DataFrame()
    df = pd.DataFrame(ts)
    df.index = project["timeseries_index"]
    df.columns = [id_to_label.get(c, c) for c in df.columns]
    return df


def compute_metrics_standalone(project: Dict) -> Dict[str, Any]:
    """Compute graph metrics without causal_mm."""
    adj = build_adjacency_df(project, use_estimated=bool(project["estimates"]))
    A = adj.values
    n = A.shape[0]
    if n == 0:
        return {}

    B = (A != 0).astype(int)
    np.fill_diagonal(B, 0)
    C = int(B.sum())
    density = C / (n * (n - 1)) if n > 1 else 0.0

    out_deg = B.sum(axis=1)
    in_deg = B.sum(axis=0)

    n_transmitters = int(np.sum((out_deg > 0) & (in_deg == 0)))
    n_receivers = int(np.sum((in_deg > 0) & (out_deg == 0)))
    n_ordinary = int(np.sum((in_deg > 0) & (out_deg > 0)))

    # Estimate significance from CI
    n_significant = 0
    n_total_est = 0
    sign_stabs = []
    for est in project["estimates"].values():
        ci_lo = est.get("ci_low")
        ci_hi = est.get("ci_high")
        if ci_lo is not None and ci_hi is not None:
            n_total_est += 1
            if ci_lo > 0 or ci_hi < 0:
                n_significant += 1
        ss = est.get("sign_stability")
        if ss is not None:
            sign_stabs.append(ss)

    return {
        "num_nodes": n,
        "num_connections": C,
        "density": round(density, 3),
        "num_transmitters": n_transmitters,
        "num_receivers": n_receivers,
        "num_ordinary": n_ordinary,
        "prop_significant": round(n_significant / n_total_est, 3) if n_total_est else None,
        "avg_sign_stability": round(float(np.mean(sign_stabs)), 3) if sign_stabs else None,
    }


def compute_centrality_standalone(project: Dict) -> pd.DataFrame:
    """Compute concept centrality without causal_mm."""
    adj = build_adjacency_df(project, use_estimated=bool(project["estimates"]))
    A = np.abs(adj.values)
    labels = list(adj.columns)
    out_deg = A.sum(axis=1)
    in_deg = A.sum(axis=0)
    return pd.DataFrame({
        "Concept": labels,
        "Out-degree": np.round(out_deg, 4),
        "In-degree": np.round(in_deg, 4),
        "Centrality": np.round(out_deg + in_deg, 4),
    }).set_index("Concept")


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def get_project() -> Optional[Dict[str, Any]]:
    """Get the currently loaded project from session state."""
    return st.session_state.get("project")


def set_project(project: Dict[str, Any], source: str = "unknown"):
    """Store a project in session state."""
    st.session_state["project"] = project
    st.session_state["project_source"] = source


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

DEFAULT_COLORS = [
    "#4F8BF9", "#FF6B6B", "#51CF66", "#FCC419", "#CC5DE8",
    "#20C997", "#FF922B", "#748FFC", "#F06595", "#22B8CF",
    "#94D82D", "#E64980", "#5C7CFA", "#FFA94D", "#845EF7",
]


def get_concept_color(concept: Dict, idx: int = 0) -> str:
    """Get color for a concept, falling back to default palette."""
    color = concept.get("color") or concept.get("metadata", {}).get("color")
    if color:
        return color
    return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]


def export_project_json(project: Dict) -> str:
    """Convert a project dict back to a fcm_project.json string."""
    def _maybe_int(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return v

    out = {
        "meta": project.get("meta", {}),
        "model": {
            "concepts": [
                {
                    "id": _maybe_int(c["id"]),
                    "label": c.get("label", ""),
                    "metadata": c.get("metadata", {}),
                }
                for c in project["concepts"]
            ],
            "edges": [
                {
                    "source": _maybe_int(e["source"]),
                    "target": _maybe_int(e["target"]),
                    "stakeholder_weight": e.get("stakeholder_weight"),
                    "metadata": e.get("metadata", {}),
                }
                for e in project["edges"]
            ],
        },
        "timeseries": {
            "index": project.get("timeseries_index", []),
            "data": project.get("timeseries_data", {}),
        },
        "settings": project.get("settings", {}),
        "estimates": project.get("estimates", {}),
    }
    if project.get("results"):
        out["results"] = project["results"]

    return json.dumps(out, indent=2)


# ---------------------------------------------------------------------------
# File listing helpers
# ---------------------------------------------------------------------------

def list_available_models() -> List[Path]:
    """List all .json files in data/models/ and examples/."""
    files = []
    for d in [DATA_MODELS_DIR, EXAMPLES_DIR]:
        if d.exists():
            files.extend(sorted(d.glob("*.json")))
    return files
