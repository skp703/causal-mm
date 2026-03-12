"""
Editor — Create and modify FCM project JSON files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import streamlit as st
import numpy as np
import pandas as pd

from utils import (
    get_project, set_project, export_project_json,
    DEFAULT_COLORS, get_id_to_label,
)

st.set_page_config(page_title="Editor", page_icon="✏️", layout="wide")
st.markdown('<h1 style="margin-bottom:0;">✏️ FCM Project Editor</h1>', unsafe_allow_html=True)


# ── Initialize from loaded project or blank ────────────────────────────────
def init_editor_state():
    """Initialize editor state from loaded project or blank."""
    project = get_project()
    if project and "editor_concepts" not in st.session_state:
        st.session_state["editor_concepts"] = [
            {
                "id": c["id"],
                "label": c.get("label", ""),
                "color": c.get("color") or c.get("metadata", {}).get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
                "unit": c.get("unit") or c.get("metadata", {}).get("unit", ""),
            }
            for i, c in enumerate(project["concepts"])
        ]
        st.session_state["editor_edges"] = [
            {
                "source": e["source"],
                "target": e["target"],
                "stakeholder_weight": e.get("stakeholder_weight", 0.0),
            }
            for e in project["edges"]
        ]
        st.session_state["editor_meta"] = project.get("meta", {})
        st.session_state["editor_ts_index"] = project.get("timeseries_index", [])
        st.session_state["editor_ts_data"] = project.get("timeseries_data", {})
    elif "editor_concepts" not in st.session_state:
        st.session_state["editor_concepts"] = []
        st.session_state["editor_edges"] = []
        st.session_state["editor_meta"] = {
            "project_id": "new_project",
            "creator": "",
            "status": "in_progress",
            "tags": [],
            "notes": "",
        }
        st.session_state["editor_ts_index"] = []
        st.session_state["editor_ts_data"] = {}

init_editor_state()


# ── Sidebar: mode selection ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Editor Mode")
    mode = st.radio(
        "Choose action",
        ["Edit current project", "Create new project"],
        index=0 if get_project() else 1,
    )
    if mode == "Create new project":
        if st.button("🗑️ Clear all", use_container_width=True):
            for key in ["editor_concepts", "editor_edges", "editor_meta",
                        "editor_ts_index", "editor_ts_data"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────
tab_meta, tab_concepts, tab_edges, tab_ts, tab_export = st.tabs(
    ["📋 Metadata", "🧩 Concepts", "🔗 Edges", "📈 Time Series", "💾 Export"]
)


# ── Metadata tab ──────────────────────────────────────────────────────────
with tab_meta:
    st.markdown("### Project Metadata")
    meta = st.session_state["editor_meta"]

    col1, col2 = st.columns(2)
    with col1:
        meta["project_id"] = st.text_input("Project ID", value=meta.get("project_id", ""))
        meta["creator"] = st.text_input("Creator", value=meta.get("creator", ""))
    with col2:
        meta["status"] = st.selectbox(
            "Status",
            ["in_progress", "completed", "draft"],
            index=["in_progress", "completed", "draft"].index(meta.get("status", "in_progress")),
        )
        tags_str = st.text_input("Tags (comma-separated)", value=", ".join(meta.get("tags", [])))
        meta["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]

    meta["notes"] = st.text_area("Notes", value=meta.get("notes", ""), height=100)
    st.session_state["editor_meta"] = meta


# ── Concepts tab ──────────────────────────────────────────────────────────
with tab_concepts:
    st.markdown("### Concepts")
    st.caption("Define the variables (nodes) in your causal map.")

    concepts = st.session_state["editor_concepts"]

    # Add concept form
    with st.expander("➕ Add Concept", expanded=len(concepts) == 0):
        ac1, ac2, ac3, ac4 = st.columns([1, 3, 2, 2])
        with ac1:
            new_id = st.text_input("ID", value=str(len(concepts) + 1), key="new_concept_id")
        with ac2:
            new_label = st.text_input("Label", value="", key="new_concept_label",
                                      placeholder="e.g. Economic Growth")
        with ac3:
            new_color = st.color_picker("Color", value=DEFAULT_COLORS[len(concepts) % len(DEFAULT_COLORS)],
                                        key="new_concept_color")
        with ac4:
            new_unit = st.text_input("Unit", value="", key="new_concept_unit",
                                     placeholder="e.g. USD, acres")

        if st.button("Add concept", use_container_width=True):
            if new_id and new_label:
                if any(c["id"] == new_id for c in concepts):
                    st.error(f"Concept ID '{new_id}' already exists.")
                else:
                    concepts.append({
                        "id": new_id,
                        "label": new_label,
                        "color": new_color,
                        "unit": new_unit,
                    })
                    st.session_state["editor_concepts"] = concepts
                    st.rerun()
            else:
                st.warning("Both ID and Label are required.")

    # Existing concepts
    if concepts:
        st.markdown("#### Current Concepts")
        for i, c in enumerate(concepts):
            with st.container():
                ec1, ec2, ec3, ec4, ec5 = st.columns([1, 3, 2, 2, 1])
                with ec1:
                    st.markdown(f"**{c['id']}**")
                with ec2:
                    new_val = st.text_input("Label", value=c["label"], key=f"cl_{i}", label_visibility="collapsed")
                    concepts[i]["label"] = new_val
                with ec3:
                    new_col = st.color_picker("Color", value=c.get("color", "#4F8BF9"), key=f"cc_{i}",
                                              label_visibility="collapsed")
                    concepts[i]["color"] = new_col
                with ec4:
                    new_unit = st.text_input("Unit", value=c.get("unit", ""), key=f"cu_{i}",
                                             label_visibility="collapsed", placeholder="unit")
                    concepts[i]["unit"] = new_unit
                with ec5:
                    if st.button("🗑️", key=f"cd_{i}", help="Delete this concept"):
                        cid = concepts[i]["id"]
                        concepts.pop(i)
                        # Also remove related edges
                        st.session_state["editor_edges"] = [
                            e for e in st.session_state["editor_edges"]
                            if e["source"] != cid and e["target"] != cid
                        ]
                        st.session_state["editor_concepts"] = concepts
                        st.rerun()

        st.session_state["editor_concepts"] = concepts
    else:
        st.info("No concepts defined yet. Add one above.")


# ── Edges tab ────────────────────────────────────────────────────────────
with tab_edges:
    st.markdown("### Edges")
    st.caption("Define causal connections between concepts.")

    concepts = st.session_state["editor_concepts"]
    edges = st.session_state["editor_edges"]

    if len(concepts) < 2:
        st.info("Add at least 2 concepts before creating edges.")
    else:
        concept_options = {c["label"]: c["id"] for c in concepts}
        label_list = list(concept_options.keys())
        id_to_lbl = {c["id"]: c["label"] for c in concepts}

        # Add edge form
        with st.expander("➕ Add Edge", expanded=len(edges) == 0):
            ae1, ae2, ae3 = st.columns([3, 3, 2])
            with ae1:
                src_label = st.selectbox("Source", label_list, key="new_edge_src")
            with ae2:
                tgt_label = st.selectbox("Target", label_list, key="new_edge_tgt")
            with ae3:
                edge_weight = st.number_input("Weight", value=0.5, min_value=-10.0,
                                              max_value=10.0, step=0.1, key="new_edge_w")

            if st.button("Add edge", use_container_width=True):
                src_id = concept_options[src_label]
                tgt_id = concept_options[tgt_label]
                if any(e["source"] == src_id and e["target"] == tgt_id for e in edges):
                    st.error(f"Edge {src_label} → {tgt_label} already exists.")
                else:
                    edges.append({
                        "source": src_id,
                        "target": tgt_id,
                        "stakeholder_weight": edge_weight,
                    })
                    st.session_state["editor_edges"] = edges
                    st.rerun()

        # Existing edges
        if edges:
            st.markdown("#### Current Edges")
            for i, e in enumerate(edges):
                with st.container():
                    ee1, ee2, ee3, ee4 = st.columns([3, 3, 3, 1])
                    with ee1:
                        st.markdown(f"**{id_to_lbl.get(e['source'], e['source'])}**")
                    with ee2:
                        st.markdown(f"→ **{id_to_lbl.get(e['target'], e['target'])}**")
                    with ee3:
                        new_w = st.number_input(
                            "Weight", value=float(e.get("stakeholder_weight", 0)),
                            min_value=-10.0, max_value=10.0, step=0.1,
                            key=f"ew_{i}", label_visibility="collapsed",
                        )
                        edges[i]["stakeholder_weight"] = new_w
                    with ee4:
                        if st.button("🗑️", key=f"ed_{i}", help="Delete this edge"):
                            edges.pop(i)
                            st.session_state["editor_edges"] = edges
                            st.rerun()

            st.session_state["editor_edges"] = edges
        else:
            st.info("No edges defined yet. Add one above.")


# ── Time Series tab ──────────────────────────────────────────────────────
with tab_ts:
    st.markdown("### Time Series Data")
    st.caption("Import or edit time-series observations for each concept.")

    concepts = st.session_state["editor_concepts"]
    ts_index = st.session_state["editor_ts_index"]
    ts_data = st.session_state["editor_ts_data"]

    if not concepts:
        st.info("Add concepts first before importing time-series data.")
    else:
        # CSV upload
        st.markdown("#### Import from CSV")
        st.markdown("Upload a CSV where the first column is the time index and "
                     "remaining columns are concept values. Column headers should be "
                     "concept IDs or labels.")
        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="ts_csv_upload")

        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
                # First column as index
                index_col = df.columns[0]
                ts_index_new = df[index_col].tolist()
                ts_data_new = {}

                id_to_lbl = {c["id"]: c["label"] for c in concepts}
                lbl_to_id = {c["label"]: c["id"] for c in concepts}

                for col in df.columns[1:]:
                    col_str = str(col)
                    # Try to match by ID or label
                    if col_str in [c["id"] for c in concepts]:
                        ts_data_new[col_str] = df[col].tolist()
                    elif col_str in lbl_to_id:
                        ts_data_new[lbl_to_id[col_str]] = df[col].tolist()
                    else:
                        st.warning(f"Column '{col_str}' doesn't match any concept ID or label — skipped.")

                if ts_data_new:
                    st.session_state["editor_ts_index"] = ts_index_new
                    st.session_state["editor_ts_data"] = ts_data_new
                    st.success(f"Imported {len(ts_index_new)} time points for "
                              f"{len(ts_data_new)} concepts.")
                    st.rerun()
                else:
                    st.error("No columns matched concept IDs or labels.")
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")

        # Manual time series entry
        ts_index = st.session_state["editor_ts_index"]
        ts_data = st.session_state["editor_ts_data"]

        if ts_index and ts_data:
            st.markdown("#### Current Data")
            id_to_lbl = {c["id"]: c["label"] for c in concepts}
            display_data = {"Time": ts_index}
            for cid, vals in ts_data.items():
                label = id_to_lbl.get(cid, cid)
                display_data[label] = vals
            display_df = pd.DataFrame(display_data)
            st.dataframe(display_df, use_container_width=True, height=300)

            if st.button("🗑️ Clear time series", type="secondary"):
                st.session_state["editor_ts_index"] = []
                st.session_state["editor_ts_data"] = {}
                st.rerun()
        else:
            st.info("No time-series data. Upload a CSV above to import data.")

        # Manual entry for small datasets
        with st.expander("✍️ Manual Entry (small datasets)"):
            st.markdown("Enter time points as comma-separated values.")
            idx_str = st.text_input(
                "Time index (comma-separated)",
                value=", ".join(str(x) for x in ts_index) if ts_index else "",
                placeholder="1990, 1991, 1992, ...",
            )

            manual_data = {}
            if idx_str.strip():
                try:
                    new_index = [x.strip() for x in idx_str.split(",")]
                    n_points = len(new_index)

                    for c in concepts:
                        existing = ts_data.get(c["id"], [])
                        existing_str = ", ".join(str(x) for x in existing) if existing else ""
                        vals_str = st.text_input(
                            f"{c['label']} (ID: {c['id']})",
                            value=existing_str,
                            placeholder=f"Enter {n_points} values...",
                            key=f"manual_ts_{c['id']}",
                        )
                        if vals_str.strip():
                            vals = [float(v.strip()) for v in vals_str.split(",")]
                            if len(vals) == n_points:
                                manual_data[c["id"]] = vals
                            else:
                                st.warning(f"{c['label']}: expected {n_points} values, got {len(vals)}")

                    if st.button("Save manual time series", use_container_width=True):
                        st.session_state["editor_ts_index"] = new_index
                        st.session_state["editor_ts_data"] = manual_data
                        st.success("Time series saved.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Parse error: {e}")


# ── Export tab ────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("### Export Project")

    concepts = st.session_state["editor_concepts"]
    edges = st.session_state["editor_edges"]
    meta = st.session_state["editor_meta"]
    ts_index = st.session_state["editor_ts_index"]
    ts_data = st.session_state["editor_ts_data"]

    # Summary
    st.markdown(f"""
    | Item | Count |
    |------|-------|
    | Concepts | {len(concepts)} |
    | Edges | {len(edges)} |
    | Time Points | {len(ts_index)} |
    | Series | {len(ts_data)} |
    """)

    if not concepts:
        st.warning("Add at least one concept before exporting.")
    else:
        # Build project dict
        export_project = {
            "meta": meta,
            "concepts": [
                {
                    "id": c["id"],
                    "label": c.get("label", ""),
                    "metadata": {
                        "color": c.get("color", ""),
                        "unit": c.get("unit", ""),
                    },
                }
                for c in concepts
            ],
            "edges": [
                {
                    "source": e["source"],
                    "target": e["target"],
                    "stakeholder_weight": e.get("stakeholder_weight"),
                    "metadata": {},
                }
                for e in edges
            ],
            "timeseries_index": ts_index,
            "timeseries_data": ts_data,
            "estimates": {},
            "settings": {},
            "results": {},
        }

        json_str = export_project_json(export_project)

        # Preview
        with st.expander("👁️ Preview JSON", expanded=False):
            st.code(json_str, language="json")

        # Download
        filename = f"{meta.get('project_id', 'project')}.fcm_project.json"

        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name=filename,
            mime="application/json",
            use_container_width=True,
            type="primary",
        )

        st.markdown("")

        # Load into visualizer
        if st.button("📥 Load into visualizer", use_container_width=True):
            set_project(export_project, source=f"Editor: {filename}")
            st.success("Project loaded! Navigate to other pages to visualize.")
