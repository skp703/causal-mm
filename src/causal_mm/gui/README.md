# FCM Project Visualizer — GUI

A **Streamlit-based** interactive GUI for visualizing, analyzing, and creating
Fuzzy Cognitive Map (FCM) project files (`fcm_project.json`).

---

## Features

| Page | Description |
|------|-------------|
| **🏠 Home** | Load projects via drag-and-drop or file browser; overview cards (concepts, edges, estimates, time points); project metadata |
| **🕸️ Network** | Interactive directed graph — nodes colored by concept, edges sized by weight, green/red for positive/negative; adjacency heatmap; multiple layouts (circular, spring, shell, Kamada-Kawai) |
| **⚖️ Weights** | Sortable edge-weight table; forest plot with 95% CI whiskers; sign-stability bars; stakeholder vs. estimated weight comparison |
| **📈 Time Series** | Multi-line interactive charts with concept selector; z-score toggle; summary statistics; pairwise correlation heatmap; per-concept sparklines |
| **📊 Metrics** | KPI cards (nodes, edges, density, transmitters, receivers); estimation quality indicators; centrality bar chart; node-role pie chart |
| **✏️ Editor** | Create new projects or edit existing ones; add/remove concepts and edges; import time-series from CSV; manual data entry; export/download JSON |

---

## Quick Start

### 1. Install dependencies

```bash
# From the repository root:
pip install -r gui/requirements.txt

# Or install individually:
pip install streamlit plotly networkx streamlit-agraph
```

> **Note:** The GUI also imports from the `causal_mm` package for full
> functionality. Install it first:
> ```bash
> pip install -e .
> ```
> The GUI works in standalone mode (raw JSON parsing) if `causal_mm` is not
> installed, but some metrics features will be limited.

### 2. Launch

```bash
cd gui
streamlit run app.py
```

Or from the repository root:

```bash
streamlit run gui/app.py
```

The app opens at **http://localhost:8501** in your default browser.

### 3. Load a project

- **Upload:** Drag and drop any `fcm_project.json` file onto the sidebar uploader.
- **Browse:** Select from available models in `data/models/` or `examples/` via the sidebar file browser.

---

## Directory Structure

```
gui/
├── app.py                  # Main entry point & landing page
├── pages/
│   ├── 1_Network.py        # Interactive network graph + adjacency heatmap
│   ├── 2_Weights.py        # Weight tables, forest plots, sign stability
│   ├── 3_TimeSeries.py     # Time-series charts, correlations, sparklines
│   ├── 4_Metrics.py        # Graph complexity & centrality dashboard
│   └── 5_Editor.py         # Create / edit / export FCM project files
├── utils.py                # Shared loading, graph-building, and export helpers
├── requirements.txt        # Python dependencies for the GUI
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration (dark mode)
└── README.md               # This file
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.30 | Web application framework |
| `plotly` | ≥ 5.18 | Interactive charts (time series, heatmaps, forest plots) |
| `networkx` | ≥ 3.0 | Graph data structures and layout algorithms |
| `streamlit-agraph` | ≥ 0.0.45 | Interactive graph component for Streamlit |
| `pandas` | ≥ 2.0 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical computations |

These are listed in `gui/requirements.txt`.

---

## Usage Notes

### Loading files

The GUI supports any valid `fcm_project.json` file. This includes:

- Files with only model structure (concepts + edges) — no estimates.
- Files with DML estimates (after running `causal-mm-run`) — full visualization.
- Files with bootstrap uncertainty (CI, sign stability) — forest plots and uncertainty panels.

### Creating new projects

Use the **✏️ Editor** page to:

1. Define project metadata (ID, creator, tags, notes).
2. Add concepts with labels, colors, and units.
3. Add directed edges with stakeholder weights.
4. Import time-series data from a CSV file (first column = time index, remaining columns = concept values).
5. Export the complete `fcm_project.json` file for use with `causal-mm-run`.

### Editing existing projects

Load a project, navigate to the Editor, modify concepts/edges/weights, and download the updated JSON.

### Scope

The GUI is a **visualization and editing tool only**. It does not run DML estimation.
To estimate causal weights, use the CLI:

```bash
causal-mm-run -i your_project.fcm_project.json -o output.json --bootstrap
```

Then reload the output file in the GUI to visualize the results.

---

## Theming

The app uses a dark theme configured in `.streamlit/config.toml`. To switch to
light mode, edit that file:

```toml
[theme]
primaryColor = "#4F8BF9"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'causal_mm'` | Install the package: `pip install -e .` from the repo root |
| `ModuleNotFoundError: No module named 'streamlit'` | Install GUI deps: `pip install -r gui/requirements.txt` |
| Port 8501 in use | Use `streamlit run app.py --server.port 8502` |
| No files in sidebar browser | Ensure `data/models/` or `examples/` contain `.json` files |
